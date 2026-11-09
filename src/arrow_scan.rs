//! Compressed-domain filtered & projected Apache Arrow scanner (`WHERE` + column projection
//! pushdown), multi-batch streaming (`ArrowWindowStream`), and dual SQL/VXG `WHERE` parser.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{
    arrow::{discover_ndjson_columns, ArrowBatch, ArrowType},
    arrow_builder::{push_utf8_lossy_bytes, ColBuilder},
    ops::{
        csv_header::{peek_delim_header_fast, resolve_header_for_targets, DelimHeader},
        line_matches_expr,
        par_meta::{build_window_meta_and_boundary, ParInBlock, SyncCell},
        ColTarget, ParsedWhere, PredExpr,
    },
    sql::{OwnedCol, SqlQuery, SqlWhere},
    DecodeOptions, Error, Result, BLOCK_COLUMN,
};

/// Default RecordBatch target row count (`65_536`) for multi-batch Arrow streaming.
pub const DEFAULT_ARROW_BATCH_ROWS: usize = 65_536;
pub(crate) const STREAM_WINDOW_BLOCKS: usize = 128;

#[derive(Clone, Debug)]
enum ArrowWhereInner {
    Sql(SqlWhere),
    Vxg(ParsedWhere),
}

/// Owned compressed-domain `WHERE` filter supporting both SQL `WHERE` syntax
/// (`payment_type = 1 AND fare >= 20.0`, `zone IN ('jfk', 'midtown')`, `LIKE`)
/// and VXG CLI shorthand (`payment_type==1 && zone^=mid`, `contains:error`).
#[derive(Clone, Debug)]
pub struct ArrowWhere(ArrowWhereInner);

fn looks_like_vxg_shorthand(s: &str) -> bool {
    const TAGS: &[&str] = &[
        "=eq:",
        "=ne:",
        "=prefix:",
        "=suffix:",
        "=contains:",
        "=num:",
        "=num-eq:",
        "=num-ne:",
        "=dec:",
        "=gt:",
        "=ge:",
        "=lt:",
        "=le:",
        "contains:",
    ];
    TAGS.iter().any(|t| s.contains(t))
}

impl ArrowWhere {
    pub fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidOption("empty WHERE predicate"));
        }
        if looks_like_vxg_shorthand(trimmed) {
            if let Ok(pw) = ParsedWhere::parse(trimmed) {
                return Ok(Self(ArrowWhereInner::Vxg(pw)));
            }
        }
        let sql_probe = format!("SELECT * FROM frame WHERE {trimmed}");
        if let Ok(q) = SqlQuery::parse(&sql_probe) {
            if q.group_by.is_empty() && q.order_by.is_empty() && q.limit.is_none() {
                return Ok(Self(ArrowWhereInner::Sql(
                    q.where_clause.unwrap_or(SqlWhere::True),
                )));
            }
        }
        ParsedWhere::parse(trimmed).map(|pw| Self(ArrowWhereInner::Vxg(pw)))
    }

    #[inline]
    pub(crate) fn from_sql_where(sw: SqlWhere) -> Self {
        Self(ArrowWhereInner::Sql(sw))
    }

    pub fn with_pred_expr<R, F: FnOnce(&PredExpr<'_>) -> R>(&self, f: F) -> R {
        match &self.0 {
            ArrowWhereInner::Sql(sw) => sw.with_pred_expr(f),
            ArrowWhereInner::Vxg(pw) => pw.with_expr(f),
        }
    }
}

/// Parse either a SQL `WHERE` clause or a VXG filter expression into an [`ArrowWhere`].
#[inline]
pub fn parse_arrow_where(raw: &str) -> Result<ArrowWhere> {
    ArrowWhere::parse(raw)
}

pub(crate) enum ResolvedPlan {
    RawRecord,
    Columns {
        names: Vec<String>,
        cols: Vec<OwnedCol>,
        hdr_opt: Option<DelimHeader>,
        skip_header_row: bool,
    },
}

pub(crate) fn target_to_owned_and_name(col: ColTarget<'_>) -> (String, OwnedCol) {
    let owned = match col {
        ColTarget::JsonKey(k) => OwnedCol::JsonKey(k.to_owned()),
        ColTarget::Delimited { delim, index } => OwnedCol::Delimited { delim, index },
    };
    (owned.display_name(), owned)
}

pub(crate) fn resolve_plan(
    frame_bytes: &[u8],
    explicit: Option<(Vec<String>, Vec<OwnedCol>)>,
    where_expr: Option<&PredExpr<'_>>,
    dec_options: &DecodeOptions,
) -> Result<ResolvedPlan> {
    if let Some((names, owned_cols)) = explicit {
        if owned_cols.is_empty() {
            return Err(Error::InvalidOption("at least one column is required"));
        }
        let targets: Vec<ColTarget<'_>> = owned_cols.iter().map(OwnedCol::as_target).collect();
        let hdr_opt = resolve_header_for_targets(frame_bytes, where_expr, &targets, dec_options)?;
        let skip_header_row = hdr_opt.is_some();
        let cols = owned_cols
            .into_iter()
            .map(|c| match (&hdr_opt, &c) {
                (Some(hdr), OwnedCol::JsonKey(k)) => hdr
                    .resolve_index(k)
                    .map_or(c, |index| OwnedCol::Delimited { delim: hdr.delim, index }),
                _ => c,
            })
            .collect();
        return Ok(ResolvedPlan::Columns {
            names,
            cols,
            hdr_opt,
            skip_header_row,
        });
    }

    if let Some(hdr) = peek_delim_header_fast(frame_bytes, dec_options)? {
        let names = hdr.columns.clone();
        let delim = hdr.delim;
        let cols = (0..names.len())
            .map(|index| OwnedCol::Delimited { delim, index })
            .collect();
        return Ok(ResolvedPlan::Columns {
            names,
            cols,
            hdr_opt: Some(hdr),
            skip_header_row: true,
        });
    }

    let keys = discover_ndjson_columns(frame_bytes, dec_options)?;
    if !keys.is_empty() {
        let names = keys.clone();
        let cols = keys.into_iter().map(OwnedCol::JsonKey).collect();
        return Ok(ResolvedPlan::Columns {
            names,
            cols,
            hdr_opt: None,
            skip_header_row: false,
        });
    }
    Ok(ResolvedPlan::RawRecord)
}

fn make_builders(names: &[String], cols: &[ColTarget<'_>]) -> Vec<ColBuilder> {
    names
        .iter()
        .zip(cols.iter())
        .map(|(n, c)| ColBuilder::new(n.clone(), matches!(c, ColTarget::JsonKey(_))))
        .collect()
}

pub(crate) fn infer_fallback_types(
    frame_bytes: &[u8],
    names: &[String],
    cols: &[OwnedCol],
    skip_header_row: bool,
    dec_options: &DecodeOptions,
) -> Vec<ArrowType> {
    let targets: Vec<ColTarget<'_>> = cols.iter().map(OwnedCol::as_target).collect();
    let probe_opts = DecodeOptions { threads: 1, ..*dec_options };
    let mut builders = make_builders(names, &targets);
    if let Ok(sample) =
        crate::extract_row_range(frame_bytes, usize::from(skip_header_row), 64, &probe_opts)
    {
        for line in sample.split_inclusive(|&b| b == b'\n') {
            if !line.is_empty() {
                for (sink, &col) in builders.iter_mut().zip(targets.iter()) {
                    sink.push_line_col(line, col);
                }
            }
        }
    }
    builders
        .iter()
        .map(|b| b.inferred_type_and_utf8_bytes().0.unwrap_or(ArrowType::Utf8))
        .collect()
}

pub(crate) fn raw_record_batches(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    batch_size: usize,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<(Vec<ArrowBatch>, Vec<String>, Vec<ArrowType>)> {
    let names = vec!["record".to_owned()];
    let types = vec![ArrowType::Utf8];
    if limit == 0 {
        return Ok((Vec::new(), names, types));
    }
    let expr = where_expr.unwrap_or(&PredExpr::True);
    let matches = crate::filter_where(frame_bytes, expr, limit, dec_options)?;
    if matches.is_empty() {
        return Ok((Vec::new(), names, types));
    }
    let step = batch_size.max(1);
    let mut batches = Vec::with_capacity(matches.len().div_ceil(step));
    for chunk in matches.chunks(step) {
        let mut b = ColBuilder::new("record".to_owned(), false);
        let mut vals = Vec::with_capacity(chunk.len() * 32);
        let mut offs = Vec::with_capacity(chunk.len() + 1);
        offs.push(0i64);
        for m in chunk {
            let mut s = m.bytes.as_slice();
            while s.last().is_some_and(|&c| c == b'\n' || c == b'\r') {
                s = &s[..s.len() - 1];
            }
            push_utf8_lossy_bytes(&mut vals, s);
            offs.push(vals.len() as i64);
        }
        b.len = chunk.len();
        b.state = crate::arrow_builder::BuilderState::Utf8 { offsets: offs, values: vals };
        let ty = if let crate::arrow_builder::BuilderState::Utf8 { ref values, .. } = b.state {
            if values.len() > i32::MAX as usize {
                ArrowType::LargeUtf8
            } else {
                ArrowType::Utf8
            }
        } else {
            ArrowType::Utf8
        };
        batches.push(ArrowBatch {
            num_rows: chunk.len(),
            columns: vec![b.finish_into_col(ty)],
        });
    }
    let final_ty = if batches
        .iter()
        .any(|b| b.columns[0].data.data_type() == ArrowType::LargeUtf8)
    {
        vec![ArrowType::LargeUtf8]
    } else {
        types
    };
    Ok((batches, names, final_ty))
}

fn scan_raw_block_interior(
    raw: &[u8],
    had_in_carry: bool,
    mut skip_first_row: bool,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    rem_cap: usize,
    sinks: &mut [ColBuilder],
) -> usize {
    let mut pos = 0usize;
    if had_in_carry {
        let Some(nl) = raw.iter().position(|&b| b == b'\n') else {
            return 0;
        };
        pos = nl + 1;
    }
    let mut emitted = 0usize;
    while pos < raw.len() && emitted < rem_cap {
        let Some(nl) = raw[pos..].iter().position(|&b| b == b'\n') else {
            break;
        };
        let line = &raw[pos..=pos + nl];
        pos += nl + 1;
        if skip_first_row {
            skip_first_row = false;
            continue;
        }
        if where_expr.is_none_or(|e| line_matches_expr(line, e)) {
            for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                sink.push_line_col(line, col);
            }
            emitted += 1;
        }
    }
    emitted
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_window_seq(
    window: &[ParInBlock<'_>],
    names: &[String],
    cols: &[ColTarget<'_>],
    where_expr: Option<&PredExpr<'_>>,
    skip_first_row: &mut bool,
    is_last_window: bool,
    carry: &mut Option<Vec<u8>>,
    rem_limit: &mut usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<Vec<ColBuilder>>> {
    if window.is_empty() || *rem_limit == 0 {
        return Ok(Vec::new());
    }
    let mut sinks = make_builders(names, cols);
    let mut raw_buf = Vec::new();
    for ib in window {
        if *rem_limit == 0 {
            break;
        }
        if ib.info.raw_size == 0 {
            continue;
        }
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(emitted) = crate::columns::scan_arrow_grammar_block_seq(
                ib.payload,
                carry,
                skip_first_row,
                where_expr,
                cols,
                *rem_limit,
                &mut sinks,
            )? {
                *rem_limit -= emitted;
                continue;
            }
        }
        ib.decode_into(dec_options, &mut raw_buf)?;
        let mut pos = 0usize;
        if let Some(mut c_bytes) = carry.take() {
            if let Some(nl) = raw_buf.iter().position(|&b| b == b'\n') {
                c_bytes.extend_from_slice(&raw_buf[..=nl]);
                pos = nl + 1;
                if *skip_first_row {
                    *skip_first_row = false;
                } else if *rem_limit > 0
                    && where_expr.is_none_or(|e| line_matches_expr(&c_bytes, e))
                {
                    for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                        sink.push_line_col(&c_bytes, col);
                    }
                    *rem_limit -= 1;
                }
            } else {
                c_bytes.extend_from_slice(&raw_buf);
                *carry = Some(c_bytes);
                continue;
            }
        }
        while pos < raw_buf.len() && *rem_limit > 0 {
            if let Some(nl) = raw_buf[pos..].iter().position(|&b| b == b'\n') {
                let line = &raw_buf[pos..=pos + nl];
                pos += nl + 1;
                if *skip_first_row {
                    *skip_first_row = false;
                } else if where_expr.is_none_or(|e| line_matches_expr(line, e)) {
                    for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                        sink.push_line_col(line, col);
                    }
                    *rem_limit -= 1;
                }
            } else {
                *carry = Some(raw_buf[pos..].to_vec());
                break;
            }
        }
    }
    if is_last_window && *rem_limit > 0 {
        if let Some(c_bytes) = carry.take() {
            if !c_bytes.is_empty() {
                if *skip_first_row {
                    *skip_first_row = false;
                } else if where_expr.is_none_or(|e| line_matches_expr(&c_bytes, e)) {
                    for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                        sink.push_line_col(&c_bytes, col);
                    }
                    *rem_limit -= 1;
                }
            }
        }
    }
    if sinks[0].len == 0 {
        Ok(Vec::new())
    } else {
        Ok(vec![sinks])
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_window_par<'a>(
    window: &[ParInBlock<'a>],
    names: &[String],
    cols: &[ColTarget<'_>],
    where_expr: Option<&PredExpr<'_>>,
    skip_global_row0: bool,
    is_last_window: bool,
    carry: &mut Option<Vec<u8>>,
    cur_global: &mut usize,
    rem_limit: &mut usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<Vec<ColBuilder>>> {
    if window.is_empty() || *rem_limit == 0 {
        return Ok(Vec::new());
    }
    let (meta, non_empty, carry_in_row, bset, next_carry, next_global) =
        build_window_meta_and_boundary(window, carry.take(), *cur_global, is_last_window, dec_options)?;
    *carry = next_carry;
    *cur_global = next_global;

    let header_interior_blk = if skip_global_row0 && carry_in_row.is_none() {
        non_empty.first().copied().filter(|&b0| {
            !(meta[b0].row_count == 1 && meta[b0].last_unterm_frag.is_some())
        })
    } else {
        None
    };

    let b_refs: Vec<_> = bset.iter().collect();
    let nb = window.len();
    let workers = dec_options.threads.min(nb).max(1);
    let num_chunks = (workers * 4).min(nb).max(1);
    let cap = *rem_limit;
    let slots: Vec<SyncCell<Option<Result<Vec<ColBuilder>>>>> =
        (0..num_chunks).map(|_| SyncCell::new(None)).collect();
    let next_chunk = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);

    crate::pool::parallel_try_run(workers, |_wid| {
        let mut raw_buf = Vec::new();
        loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let ci = next_chunk.fetch_add(1, Ordering::Relaxed);
            if ci >= num_chunks {
                break;
            }
            let b_start = ci * nb / num_chunks;
            let b_end = (ci + 1) * nb / num_chunks;
            let res: Result<Vec<ColBuilder>> = (|| {
                let mut sinks = make_builders(names, cols);
                let mut emitted = 0usize;
                if ci == 0 {
                    if let Some((g_idx, ref line)) = carry_in_row {
                        if !(skip_global_row0 && g_idx == 0)
                            && emitted < cap
                            && where_expr.is_none_or(|e| line_matches_expr(line, e))
                        {
                            for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                                sink.push_line_col(line, col);
                            }
                            emitted += 1;
                        }
                    }
                }
                let mut br_idx = b_refs.partition_point(|e| e.block_index < b_start);
                let br_end = b_refs.partition_point(|e| e.block_index < b_end);
                for b in b_start..b_end {
                    if emitted >= cap {
                        break;
                    }
                    let ib = &window[b];
                    if meta[b].row_count > 0 && ib.info.raw_size > 0 {
                        let skip_r0 = header_interior_blk == Some(b);
                        let mut done_grammar = false;
                        if ib.info.kind == BLOCK_COLUMN && meta[b].decoded_raw.is_none() {
                            if let Some(n) = crate::columns::scan_arrow_grammar_block_par(
                                ib.payload,
                                meta[b].had_in_carry,
                                skip_r0,
                                where_expr,
                                cols,
                                cap - emitted,
                                &mut sinks,
                            )? {
                                emitted += n;
                                done_grammar = true;
                            }
                        }
                        if !done_grammar {
                            let raw_slice: &[u8] = if let Some(ref dec) = meta[b].decoded_raw {
                                dec
                            } else {
                                ib.decode_into(dec_options, &mut raw_buf)?;
                                &raw_buf
                            };
                            emitted += scan_raw_block_interior(
                                raw_slice,
                                meta[b].had_in_carry,
                                skip_r0,
                                where_expr,
                                cols,
                                cap - emitted,
                                &mut sinks,
                            );
                        }
                    }
                    while br_idx < br_end && b_refs[br_idx].block_index == b {
                        let entry = b_refs[br_idx];
                        br_idx += 1;
                        if skip_global_row0 && entry.global_row_index == 0 {
                            continue;
                        }
                        if emitted < cap
                            && where_expr.is_none_or(|e| line_matches_expr(entry.bytes, e))
                        {
                            for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                                sink.push_line_col(entry.bytes, col);
                            }
                            emitted += 1;
                        }
                    }
                }
                Ok(sinks)
            })();
            let is_err = res.is_err();
            unsafe {
                *slots[ci].get() = Some(res);
            }
            if is_err {
                failed.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }
        Ok(())
    })?;

    let mut out = Vec::with_capacity(num_chunks);
    let mut first_err = None;
    for slot in slots {
        match slot.into_inner() {
            Some(Ok(mut c)) if first_err.is_none() => {
                let n = c[0].len;
                if n > 0 && *rem_limit > 0 {
                    let keep = n.min(*rem_limit);
                    if keep < n {
                        for col_b in &mut c {
                            col_b.truncate(keep);
                        }
                    }
                    *rem_limit -= keep;
                    out.push(c);
                }
            }
            Some(Ok(_)) | None => {}
            Some(Err(e)) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(out)
}
