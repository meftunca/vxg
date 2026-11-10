//! Automatic CSV/TSV/PSV header row detection, column-name-to-index resolution,
//! and zero-reencode block-0 header stripping.

use std::borrow::Cow;

use crate::checksum::crc32c;
use crate::frame::{v5, write_entry, ENTRY_SIZE_V4, HEADER_SIZE, MAGIC, VERSION_V4, VERSION_V5};
use crate::{block, query, DecodeOptions, Frame, Result, BLOCK_COLUMN};

use super::agg::{agg_line, decode_group_token_ref, AggStats, GroupKey, GroupRow};
use super::expr::{line_matches_expr, PredExpr};
use super::line::{decode_delim_str, find_line_col_span, scan_json_str_end, ColPred, ColTarget};

#[derive(Clone, Debug)]
pub struct DelimHeader {
    pub delim: u8,
    pub columns: Vec<String>,
    pub header_line: Vec<u8>,
    pub f0_idx: usize,
    pub b0_idx: usize,
    pub header_byte_len: usize,
}

impl DelimHeader {
    #[inline]
    pub fn resolve_index(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c == name)
            .or_else(|| self.columns.iter().position(|c| c.eq_ignore_ascii_case(name)))
    }

    #[inline]
    pub fn resolve_col<'a>(&self, col: ColTarget<'a>) -> Option<ColTarget<'a>> {
        match col {
            ColTarget::JsonKey(name) => self.resolve_index(name).map(|index| ColTarget::Delimited {
                delim: self.delim,
                index,
            }),
            d @ ColTarget::Delimited { .. } => Some(d),
        }
    }
}

fn parse_delim_header_line(line_with_nl: &[u8]) -> Option<(u8, Vec<String>)> {
    let line = line_with_nl
        .strip_suffix(b"\r\n")
        .or_else(|| line_with_nl.strip_suffix(b"\n"))?;
    if line.is_empty() || line.len() > 8192 {
        return None;
    }
    let mut first_non_ws = 0usize;
    while first_non_ws < line.len() && matches!(line[first_non_ws], b' ' | b'\t') {
        first_non_ws += 1;
    }
    if line.get(first_non_ws) == Some(&b'{') {
        return None;
    }
    for delim in [b',', b'\t', b'|'] {
        if !line.contains(&delim) {
            continue;
        }
        let mut cols = Vec::new();
        let mut pos = 0usize;
        let mut ok = true;
        loop {
            if pos > line.len() {
                ok = false;
                break;
            }
            let end = if line.get(pos) == Some(&b'"') {
                let Some(e) = scan_json_str_end(line, pos) else {
                    ok = false;
                    break;
                };
                if e < line.len() && line[e] != delim {
                    ok = false;
                    break;
                }
                e
            } else {
                line[pos..]
                    .iter()
                    .position(|&b| b == delim)
                    .map_or(line.len(), |off| pos + off)
            };
            let Some(dec) = decode_delim_str(&line[pos..end]) else {
                ok = false;
                break;
            };
            let Ok(s) = std::str::from_utf8(&dec) else {
                ok = false;
                break;
            };
            let trimmed = s.trim();
            let b0 = trimmed.as_bytes().first().copied().unwrap_or(0);
            if trimmed.is_empty()
                || !(b0.is_ascii_alphabetic() || b0 == b'_')
                || !trimmed
                    .bytes()
                    .all(|b| (0x20..=0x7e).contains(&b) && !matches!(b, b'{' | b'}' | b'[' | b']'))
            {
                ok = false;
                break;
            }
            cols.push(trimmed.to_owned());
            if end == line.len() {
                break;
            }
            pos = end + 1;
        }
        if ok && cols.len() >= 2 {
            return Some((delim, cols));
        }
    }
    None
}

fn try_peek_raw_or_grammar_header(
    f_idx: usize,
    b_idx: usize,
    kind: u32,
    payload: &[u8],
) -> Result<Option<Option<DelimHeader>>> {
    if kind == block::RAW {
        let Some(nl) = payload.iter().position(|&b| b == b'\n') else {
            return Ok(Some(None));
        };
        let hdr_len = nl + 1;
        let line_slice = &payload[..hdr_len];
        return Ok(Some(parse_delim_header_line(line_slice).map(
            |(delim, columns)| DelimHeader {
                delim,
                columns,
                header_line: line_slice.to_vec(),
                f0_idx: f_idx,
                b0_idx: b_idx,
                header_byte_len: hdr_len,
            },
        )));
    }
    if kind == BLOCK_COLUMN {
        if let Some(indep) = crate::columns::peek_grammar_boundary_indep(payload)? {
            if let Some(ref first_raw) = indep.first_raw {
                if first_raw.ends_with(b"\n") {
                    let hdr_len = first_raw.len();
                    return Ok(Some(parse_delim_header_line(first_raw).map(
                        |(delim, columns)| DelimHeader {
                            delim,
                            columns,
                            header_line: first_raw.to_vec(),
                            f0_idx: f_idx,
                            b0_idx: b_idx,
                            header_byte_len: hdr_len,
                        },
                    )));
                }
            } else if indep.first_seg_byte == Some(b'{') {
                return Ok(Some(None));
            }
            let mut row0 = Vec::new();
            if crate::columns::unpack_grammar_row_range(payload, 0, 1, &mut row0)?.is_some()
                && row0.ends_with(b"\n")
            {
                let hdr_len = row0.len();
                return Ok(Some(parse_delim_header_line(&row0).map(|(delim, columns)| {
                    DelimHeader {
                        delim,
                        columns,
                        header_line: row0,
                        f0_idx: f_idx,
                        b0_idx: b_idx,
                        header_byte_len: hdr_len,
                    }
                })));
            }
        }
    }
    Ok(None)
}

/// Zero-directory-allocation $O(1)$ header probe for callers that subsequently run `Frame::parse_all`.
pub(crate) fn peek_delim_header_fast(
    frame_bytes: &[u8],
    options: &DecodeOptions,
) -> Result<Option<DelimHeader>> {
    options.validate()?;
    if let Some((kind, _rn, payload)) = v5::peek_v4_v5_block0(frame_bytes, options.max_output) {
        if let Ok(Some(decided)) = try_peek_raw_or_grammar_header(0, 0, kind, payload) {
            return Ok(decided);
        }
    }
    peek_delim_header(frame_bytes, options)
}

/// Inspect block 0 of `frame_bytes` in microseconds (zero block decode if NDJSON `GRAMMAR_PACKET`
/// with raw first row or `RAW` block) and return its [`DelimHeader`] if line 0 is a delimited header.
pub fn peek_delim_header(frame_bytes: &[u8], options: &DecodeOptions) -> Result<Option<DelimHeader>> {
    options.validate()?;
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    let slices = query::frame_slices(frame_bytes, &frames);
    for (f_idx, (f_bytes, parsed)) in slices.iter().enumerate() {
        for (b_idx, info) in parsed.blocks().enumerate() {
            if info.raw_size == 0 {
                continue;
            }
            let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
            if let Some(decided) = try_peek_raw_or_grammar_header(f_idx, b_idx, info.kind, payload)? {
                return Ok(decided);
            }
            let mut blk = Vec::new();
            query::decode_block_into(f_bytes, b_idx, info.raw_size, options, &mut blk)?;
            let Some(nl) = blk.iter().position(|&b| b == b'\n') else {
                return Ok(None);
            };
            let hdr_len = nl + 1;
            let line_slice = &blk[..hdr_len];
            return Ok(
                parse_delim_header_line(line_slice).map(|(delim, columns)| DelimHeader {
                    delim,
                    columns,
                    header_line: line_slice.to_vec(),
                    f0_idx: f_idx,
                    b0_idx: b_idx,
                    header_byte_len: hdr_len,
                }),
            );
        }
    }
    Ok(None)
}

fn push_v4_header(out: &mut Vec<u8>, raw_size: usize, block_size: usize, count: usize, csum: bool) {
    let base = out.len();
    out.resize(base + HEADER_SIZE, 0);
    out[base..base + 4].copy_from_slice(MAGIC);
    out[base + 4..base + 6].copy_from_slice(&VERSION_V4.to_le_bytes());
    out[base + 6..base + 8].copy_from_slice(&u16::from(csum).to_le_bytes());
    out[base + 8..base + 16].copy_from_slice(&(raw_size as u64).to_le_bytes());
    out[base + 16..base + 20].copy_from_slice(&(block_size as u32).to_le_bytes());
    out[base + 20..base + 24].copy_from_slice(&(count as u32).to_le_bytes());
}

/// Strip global row 0 (`hdr.header_byte_len`) from `(hdr.f0_idx, hdr.b0_idx)` while copying all
/// remaining compressed blocks verbatim without re-compression.
pub fn strip_delim_header_bytes<'a>(
    frame_bytes: &'a [u8],
    hdr: &DelimHeader,
    options: &DecodeOptions,
) -> Result<Cow<'a, [u8]>> {
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    let slices = query::frame_slices(frame_bytes, &frames);
    let (f0_bytes, f0) = &slices[hdr.f0_idx];
    let b0_info = f0.blocks[hdr.b0_idx].info;
    let mut blk0 = Vec::new();
    query::decode_block_into(f0_bytes, hdr.b0_idx, b0_info.raw_size, options, &mut blk0)?;
    let rem0 = &blk0[hdr.header_byte_len..];

    let mut out = Vec::with_capacity(frame_bytes.len() + rem0.len() + 128);
    let csum = f0.checksummed;
    let bsize = f0.block_size;

    if !rem0.is_empty() {
        push_v4_header(&mut out, rem0.len(), bsize, 1, csum);
        let dir_pos = out.len();
        out.resize(dir_pos + ENTRY_SIZE_V4, 0);
        let crc = if csum { crc32c(rem0) } else { 0 };
        write_entry(
            &mut out,
            dir_pos,
            VERSION_V4,
            0,
            HEADER_SIZE + ENTRY_SIZE_V4,
            rem0.len(),
            rem0.len(),
            crc,
            block::RAW,
        );
        out.extend_from_slice(rem0);
    }

    let tail_blocks = &f0.blocks[hdr.b0_idx + 1..];
    if !tail_blocks.is_empty() {
        let f0_hdr = v5::parse_v1_to_v5_header(f0_bytes, options.max_output)?;
        let tail_raw: usize = tail_blocks.iter().map(|b| b.info.raw_size).sum();
        let tcount = tail_blocks.len();
        let use_v5 = f0_hdr.has_var_raw || f0_hdr.has_dict || f0_hdr.has_row_index;
        let ver = if use_v5 { VERSION_V5 } else { VERSION_V4 };
        let dict_wire = if f0_hdr.has_dict {
            Some((
                &f0_bytes[f0_hdr.dict_offset..f0_hdr.dict_offset + f0_hdr.dict_wire_len],
                f0_hdr.dict_raw_len,
            ))
        } else {
            None
        };
        let hdr_base = out.len();
        out.resize(hdr_base + HEADER_SIZE, 0);
        v5::write_header_bytes(
            &mut out[hdr_base..hdr_base + HEADER_SIZE],
            ver,
            csum,
            f0_hdr.has_var_raw,
            f0_hdr.has_row_index,
            dict_wire,
            tail_raw,
            bsize,
            tcount,
        );
        let dir_base = out.len();
        out.resize(dir_base + tcount * ENTRY_SIZE_V4, 0);
        for (i, b) in tail_blocks.iter().enumerate() {
            write_entry(
                &mut out,
                dir_base + i * ENTRY_SIZE_V4,
                ver,
                0,
                0,
                b.info.raw_size,
                b.info.payload_size,
                if csum { b.info.checksum } else { 0 },
                b.info.kind,
            );
        }
        if f0_hdr.has_var_raw {
            for b in tail_blocks {
                out.extend_from_slice(&(b.info.raw_size as u32).to_le_bytes());
            }
        }
        if let Some((dw, _)) = dict_wire {
            out.extend_from_slice(dw);
        }
        if f0_hdr.has_row_index {
            let ri_start = f0_hdr.row_index_offset + (hdr.b0_idx + 1) * 4;
            out.extend_from_slice(&f0_bytes[ri_start..ri_start + tcount * 4]);
        }
        let pay_start = tail_blocks[0].info.payload_offset;
        let last_info = tail_blocks[tcount - 1].info;
        let pay_end = last_info.payload_offset + last_info.payload_size;
        out.extend_from_slice(&f0_bytes[pay_start..pay_end]);
    }

    for &(later_bytes, _) in &slices[hdr.f0_idx + 1..] {
        out.extend_from_slice(later_bytes);
    }

    if out.is_empty() {
        push_v4_header(&mut out, 0, bsize, 0, csum);
    }
    Ok(Cow::Owned(out))
}

#[inline]
pub fn expr_all_json_keys_in_header(expr: &PredExpr<'_>, hdr: &DelimHeader) -> (bool, bool) {
    match *expr {
        PredExpr::True | PredExpr::RowContains(_) => (false, true),
        PredExpr::Col(ref p) => match p.col {
            ColTarget::JsonKey(name) => (true, hdr.resolve_index(name).is_some()),
            ColTarget::Delimited { .. } => (false, true),
        },
        PredExpr::Not(inner) => expr_all_json_keys_in_header(inner, hdr),
        PredExpr::All(ch) | PredExpr::Any(ch) => {
            let mut has_jk = false;
            for c in ch {
                let (hj, ok) = expr_all_json_keys_in_header(c, hdr);
                if !ok {
                    return (true, false);
                }
                has_jk |= hj;
            }
            (has_jk, true)
        }
    }
}

#[inline]
pub fn expr_contains_json_key(expr: &PredExpr<'_>) -> bool {
    match *expr {
        PredExpr::True | PredExpr::RowContains(_) => false,
        PredExpr::Col(ref p) => matches!(p.col, ColTarget::JsonKey(_)),
        PredExpr::Not(inner) => expr_contains_json_key(inner),
        PredExpr::All(ch) | PredExpr::Any(ch) => ch.iter().any(expr_contains_json_key),
    }
}

pub enum MappedExpr<'a> {
    True,
    Col(ColPred<'a>),
    RowContains(&'a [u8]),
    Not(Box<MappedExpr<'a>>),
    All(Vec<MappedExpr<'a>>),
    Any(Vec<MappedExpr<'a>>),
}

impl<'a> MappedExpr<'a> {
    pub fn from_expr(expr: &PredExpr<'a>, hdr: &DelimHeader) -> Self {
        match *expr {
            PredExpr::True => Self::True,
            PredExpr::RowContains(n) => Self::RowContains(n),
            PredExpr::Col(ref p) => Self::Col(ColPred {
                col: hdr.resolve_col(p.col).unwrap_or(p.col),
                pred: p.pred,
            }),
            PredExpr::Not(inner) => Self::Not(Box::new(Self::from_expr(inner, hdr))),
            PredExpr::All(ch) => Self::All(ch.iter().map(|c| Self::from_expr(c, hdr)).collect()),
            PredExpr::Any(ch) => Self::Any(ch.iter().map(|c| Self::from_expr(c, hdr)).collect()),
        }
    }

    pub fn with_pred_expr<R, F: FnOnce(&PredExpr<'_>) -> R>(&self, f: F) -> R {
        fn count_nodes(n: &MappedExpr<'_>) -> (usize, usize) {
            match n {
                MappedExpr::True | MappedExpr::Col(_) | MappedExpr::RowContains(_) => (0, 0),
                MappedExpr::Not(i) => {
                    let (p, s) = count_nodes(i);
                    (p + 1, s)
                }
                MappedExpr::All(v) | MappedExpr::Any(v) => {
                    let (mut p, mut s) = (v.len(), 1);
                    for c in v {
                        let (cp, cs) = count_nodes(c);
                        p += cp;
                        s += cs;
                    }
                    (p, s)
                }
            }
        }
        let (np, ns) = count_nodes(self);
        let mut p_arena: Vec<PredExpr<'_>> = Vec::with_capacity(np);
        let mut s_arena: Vec<&[PredExpr<'_>]> = Vec::with_capacity(ns);
        fn build<'b, 's>(
            node: &'b MappedExpr<'b>,
            pa: &mut Vec<PredExpr<'s>>,
            sa: &mut Vec<&'s [PredExpr<'s>]>,
        ) -> PredExpr<'s>
        where
            'b: 's,
        {
            match node {
                MappedExpr::True => PredExpr::True,
                MappedExpr::RowContains(b) => PredExpr::RowContains(b),
                MappedExpr::Col(cp) => PredExpr::Col(*cp),
                MappedExpr::Not(inner) => {
                    let child = build(inner, pa, sa);
                    let idx = pa.len();
                    pa.push(child);
                    let r: &'s PredExpr<'s> = unsafe { &*(pa.as_ptr().add(idx)) };
                    PredExpr::Not(r)
                }
                MappedExpr::All(ch) | MappedExpr::Any(ch) => {
                    let mut tmp = Vec::with_capacity(ch.len());
                    for c in ch {
                        tmp.push(build(c, pa, sa));
                    }
                    let base = pa.len();
                    pa.extend(tmp);
                    let sl: &'s [PredExpr<'s>] =
                        unsafe { std::slice::from_raw_parts(pa.as_ptr().add(base), ch.len()) };
                    sa.push(sl);
                    if matches!(node, MappedExpr::All(_)) {
                        PredExpr::All(sl)
                    } else {
                        PredExpr::Any(sl)
                    }
                }
            }
        }
        let root = build(self, &mut p_arena, &mut s_arena);
        f(&root)
    }
}

pub fn with_opt_mapped_expr<R, F: FnOnce(Option<&PredExpr<'_>>) -> R>(
    opt_mapped: Option<&MappedExpr<'_>>,
    f: F,
) -> R {
    match opt_mapped {
        None => f(None),
        Some(m) => m.with_pred_expr(|e| f(Some(e))),
    }
}

/// Returns `Some(DelimHeader)` iff `(where_expr, extra_cols)` contains at least one
/// `ColTarget::JsonKey` AND line 0 of `frame_bytes` is a valid delimited header resolving
/// **all** `ColTarget::JsonKey`s present in `where_expr` and `extra_cols`.
pub fn resolve_header_for_targets(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    extra_cols: &[ColTarget<'_>],
    options: &DecodeOptions,
) -> Result<Option<DelimHeader>> {
    let has_jk = where_expr.is_some_and(expr_contains_json_key)
        || extra_cols.iter().any(|c| matches!(c, ColTarget::JsonKey(_)));
    if !has_jk {
        return Ok(None);
    }
    let Some(hdr) = peek_delim_header_fast(frame_bytes, options)? else {
        return Ok(None);
    };
    if let Some(w) = where_expr {
        let (_, ok) = expr_all_json_keys_in_header(w, &hdr);
        if !ok {
            return Ok(None);
        }
    }
    for &c in extra_cols {
        if let ColTarget::JsonKey(name) = c {
            if hdr.resolve_index(name).is_none() {
                return Ok(None);
            }
        }
    }
    Ok(Some(hdr))
}

#[inline]
pub fn subtract_header_agg(
    hdr: &DelimHeader,
    mapped_where: Option<&PredExpr<'_>>,
    mapped_col: ColTarget<'_>,
    st: &mut AggStats,
) {
    let mut hdr_st = AggStats::default();
    agg_line(&hdr.header_line, mapped_where, mapped_col, &mut hdr_st);
    st.row_count = st.row_count.saturating_sub(hdr_st.row_count);
}

#[inline]
pub fn subtract_header_group_by(
    hdr: &DelimHeader,
    mapped_where: Option<&PredExpr<'_>>,
    mapped_gcol: ColTarget<'_>,
    rows: &mut Vec<GroupRow>,
) {
    if !mapped_where.is_none_or(|w| line_matches_expr(&hdr.header_line, w)) {
        return;
    }
    let Some((gs, ge)) = find_line_col_span(&hdr.header_line, mapped_gcol) else {
        return;
    };
    let Some((is_str, gkey)) = decode_group_token_ref(mapped_gcol, &hdr.header_line[gs..ge]) else {
        return;
    };
    let target = if is_str {
        GroupKey::Str(gkey.into_owned())
    } else {
        GroupKey::RawJson(gkey.into_owned())
    };
    if let Ok(pos) = rows.binary_search_by(|r| r.key.cmp(&target)) {
        rows[pos].stats.row_count = rows[pos].stats.row_count.saturating_sub(1);
        if rows[pos].stats.row_count == 0 {
            rows.remove(pos);
        }
    }
}
