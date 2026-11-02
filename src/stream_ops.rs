//! Bounded-memory multi-frame streaming compressed-domain execution (`FrameReader`,
//! `verify_frame`, `verify_stream`, and `*_stream` query/analytics/projection/mutation APIs)
//! with exact cross-frame split-line stitching.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::{
    block,
    checksum::crc32c,
    ops::{
        emit::{emit_ready_blocks, BytesRef, ReadyBlock},
        validate_projection_cols, AggStats, ColTarget, GroupKey, GroupRow, PredExpr,
    },
    query, DecodeOptions, Error, Frame, Options, Result, BLOCK_COLUMN,
};

mod top_k;
mod transform;
mod verify;

pub use top_k::{select_top_k_by_stream, top_k_by_stream};
pub use transform::{
    filter_frame_where_stream, map_column_stream, project_frame_stream, update_column_stream,
};
pub use verify::{verify_frame, verify_stream, FrameReader};

pub(crate) fn write_all_vxg<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|e| Error::Backend(e.to_string()))
}

pub(crate) struct FramePeeler {
    carry: Vec<u8>,
    work_buf: Vec<u8>,
    dec_blk: Vec<u8>,
}

impl FramePeeler {
    pub(crate) fn new() -> Self {
        Self {
            carry: Vec::new(),
            work_buf: Vec::new(),
            dec_blk: Vec::new(),
        }
    }

    fn push_raw_chunks<'a>(
        raw: &[u8],
        chunk_size: usize,
        checksummed: bool,
        ready: &mut Vec<ReadyBlock<'a>>,
    ) {
        for chunk in raw.chunks(chunk_size.max(1)) {
            ready.push(ReadyBlock {
                payload: BytesRef::Owned(chunk.to_vec()),
                kind: block::RAW,
                raw_size: chunk.len(),
                checksum: if checksummed { crc32c(chunk) } else { 0 },
            });
        }
    }

    fn block_ends_with_nl_fast(
        f_bytes: &[u8],
        info: &crate::frame::BlockInfo,
    ) -> Result<Option<bool>> {
        let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
        if info.kind == BLOCK_COLUMN {
            if let Some((rows, ends_nl)) = crate::columns::grammar_block_row_meta(payload)? {
                if rows > 0 {
                    return Ok(Some(ends_nl));
                }
            }
            return Ok(None);
        }
        if info.kind == block::RAW && !payload.is_empty() {
            return Ok(Some(payload.last() == Some(&b'\n')));
        }
        Ok(None)
    }

    /// Peels any unterminated trailing line off `f_bytes` and prepends any incoming `self.carry`
    /// from earlier frames, keeping all interior compressed `GRAMMAR_PACKET` blocks untouched.
    /// Returns `Ok(None)` if the frame contributed `0` complete `\n`-terminated lines.
    pub(crate) fn peel_frame<'a>(
        &'a mut self,
        f_bytes: &'a [u8],
        dec: &DecodeOptions,
    ) -> Result<Option<&'a [u8]>> {
        let parsed = Frame::parse(f_bytes, usize::MAX)?;
        let nb = parsed.block_count();
        if nb == 0 || parsed.raw_size() == 0 {
            return Ok(None);
        }
        let blocks: Vec<crate::frame::BlockInfo> = parsed.blocks().collect();
        let last_ends_nl_fast = Self::block_ends_with_nl_fast(f_bytes, &blocks[nb - 1])?;

        // Zero-copy fast path: no incoming carry and last block is known to end with `\n`.
        if self.carry.is_empty() && last_ends_nl_fast == Some(true) {
            return Ok(Some(f_bytes));
        }

        let bs = parsed.block_size();
        let csum = parsed.checksummed();
        let emit_opts = Options {
            block_size: bs,
            checksum: csum,
            ..Options::default()
        };

        // Legacy v1 frames use different LZ opcode-7 semantics (`short_distance == false`),
        // so materialize v1 frames once into `RAW` blocks.
        if parsed.version() < 2 {
            let full = crate::decompress(f_bytes, dec)?;
            let Some(last_nl) = full.iter().rposition(|&b| b == b'\n') else {
                self.carry.extend_from_slice(&full);
                return Ok(None);
            };
            let mut ready = Vec::new();
            if !self.carry.is_empty() {
                Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            }
            Self::push_raw_chunks(&full[..=last_nl], bs, csum, &mut ready);
            self.carry.clear();
            self.carry.extend_from_slice(&full[last_nl + 1..]);
            self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
            return Ok(Some(&self.work_buf));
        }

        // Fast incoming-carry-only path when last block ends with `\n` (`0` blocks decoded!).
        if last_ends_nl_fast == Some(true) {
            let mut ready = Vec::with_capacity(nb + 2);
            Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            self.carry.clear();
            for info in &blocks {
                ready.push(ReadyBlock {
                    payload: BytesRef::Borrowed(
                        &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
                    ),
                    kind: info.kind,
                    raw_size: info.raw_size,
                    checksum: info.checksum,
                });
            }
            self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
            return Ok(Some(&self.work_buf));
        }

        // General backward walk from `nb - 1` until we locate the last `\n` in `f_bytes`.
        let mut tail_rev_parts: Vec<Vec<u8>> = Vec::new();
        let mut split_blk: Option<(usize, Vec<u8>)> = None;

        for idx in (0..nb).rev() {
            let info = &blocks[idx];
            query::decode_block_into(f_bytes, idx, info.raw_size, dec, &mut self.dec_blk)?;
            if let Some(pos) = self.dec_blk.iter().rposition(|&b| b == b'\n') {
                if pos + 1 < self.dec_blk.len() {
                    tail_rev_parts.push(self.dec_blk[pos + 1..].to_vec());
                }
                split_blk = Some((idx, self.dec_blk[..=pos].to_vec()));
                break;
            } else if !self.dec_blk.is_empty() {
                tail_rev_parts.push(self.dec_blk.clone());
            }
        }

        let Some((split_idx, prefix_with_nl)) = split_blk else {
            // Entire frame had zero `\n` bytes — append all of it onto `self.carry`!
            for part in tail_rev_parts.iter().rev() {
                self.carry.extend_from_slice(part);
            }
            return Ok(None);
        };

        // Fast path if `self.carry` was empty AND the last block actually ended with `\n`
        // (e.g. a `FAST_LZ` block whose last byte happened to be `\n`).
        if self.carry.is_empty() && split_idx + 1 == nb && tail_rev_parts.is_empty() {
            return Ok(Some(f_bytes));
        }

        let mut ready = Vec::with_capacity(split_idx + 3);
        if !self.carry.is_empty() {
            Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            self.carry.clear();
        }
        for info in &blocks[..split_idx] {
            ready.push(ReadyBlock {
                payload: BytesRef::Borrowed(
                    &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
                ),
                kind: info.kind,
                raw_size: info.raw_size,
                checksum: info.checksum,
            });
        }
        Self::push_raw_chunks(&prefix_with_nl, bs, csum, &mut ready);

        for part in tail_rev_parts.iter().rev() {
            self.carry.extend_from_slice(part);
        }

        self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
        Ok(Some(&self.work_buf))
    }

    /// Emits any final unterminated trailing line left at `EOF` as a single `VXG1` frame.
    pub(crate) fn finish_tail(&mut self) -> Result<Option<&[u8]>> {
        if self.carry.is_empty() {
            return Ok(None);
        }
        let mut ready = Vec::new();
        Self::push_raw_chunks(&self.carry, 65536, false, &mut ready);
        self.carry.clear();
        let emit_opts = Options {
            checksum: false,
            ..Options::default()
        };
        self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
        Ok(Some(&self.work_buf))
    }
}

/// Counts total structured records across a `VXG1` stream (`R: Read`) with bounded memory
/// and exact cross-frame split-line stitching.
pub fn frame_row_count_stream<R: Read>(reader: R, dec_options: &DecodeOptions) -> Result<u64> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = 0u64;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            total = total
                .checked_add(crate::frame_row_count(eff)? as u64)
                .ok_or(Error::SizeOverflow)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        total = total
            .checked_add(crate::frame_row_count(tail_eff)? as u64)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(total)
}

/// Counts records matching compound `expr` across a `VXG1` stream (`R: Read`) with bounded memory.
pub fn count_where_stream<R: Read>(
    reader: R,
    expr: &PredExpr<'_>,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = 0u64;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            total = total
                .checked_add(crate::count_where(eff, expr, &sub_dec)?)
                .ok_or(Error::SizeOverflow)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        total = total
            .checked_add(crate::count_where(tail_eff, expr, &sub_dec)?)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(total)
}

/// Streams records matching compound `expr` from `reader` directly into `writer` with bounded
/// memory, stopping early as soon as `limit` matching records have been emitted.
/// Returns the number of matching records written.
pub fn filter_where_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    expr: &PredExpr<'_>,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    if limit == 0 {
        writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
        return Ok(0);
    }
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut rem = limit;
    let mut written_rows = 0u64;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let matches = crate::filter_where(eff, expr, rem, &sub_dec)?;
            for m in &matches {
                write_all_vxg(&mut writer, &m.bytes)?;
            }
            written_rows += matches.len() as u64;
            rem = rem.saturating_sub(matches.len());
            if rem == 0 {
                writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
                return Ok(written_rows);
            }
        }
    }
    if rem > 0 {
        if let Some(tail_eff) = peeler.finish_tail()? {
            let matches = crate::filter_where(tail_eff, expr, rem, &sub_dec)?;
            for m in &matches {
                write_all_vxg(&mut writer, &m.bytes)?;
            }
            written_rows += matches.len() as u64;
        }
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(written_rows)
}

/// Streams projected columns (`SELECT cols WHERE ...`) from `reader` directly into `writer`
/// with bounded memory, stopping early as soon as `limit` records have been emitted.
/// Returns the number of projected records written.
pub fn select_columns_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    validate_projection_cols(cols)?;
    if limit == 0 {
        writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
        return Ok(0);
    }
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut rem = limit;
    let mut written_rows = 0u64;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let proj = crate::select_columns(eff, where_expr, cols, rem, &sub_dec)?;
            if !proj.is_empty() {
                let rows = proj.split_inclusive(|&b| b == b'\n').count();
                write_all_vxg(&mut writer, &proj)?;
                written_rows += rows as u64;
                rem = rem.saturating_sub(rows);
                if rem == 0 {
                    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
                    return Ok(written_rows);
                }
            }
        }
    }
    if rem > 0 {
        if let Some(tail_eff) = peeler.finish_tail()? {
            let proj = crate::select_columns(tail_eff, where_expr, cols, rem, &sub_dec)?;
            if !proj.is_empty() {
                let rows = proj.split_inclusive(|&b| b == b'\n').count();
                write_all_vxg(&mut writer, &proj)?;
                written_rows += rows as u64;
            }
        }
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(written_rows)
}

/// Computes exact `AggStats` (`row_count`, `count`, `sum`, `min`, `max`, `avg`) across an
/// arbitrarily large `VXG1` stream (`R: Read`) in `O(1)` memory.
pub fn aggregate_column_stream<R: Read>(
    reader: R,
    where_expr: Option<&PredExpr<'_>>,
    col: ColTarget<'_>,
    dec_options: &DecodeOptions,
) -> Result<AggStats> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = AggStats::default();

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let part = crate::aggregate_column(eff, where_expr, col, &sub_dec)?;
            total.merge(&part);
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let part = crate::aggregate_column(tail_eff, where_expr, col, &sub_dec)?;
        total.merge(&part);
    }
    Ok(total)
}

/// Computes exact `GROUP BY` aggregations across an arbitrarily large `VXG1` stream (`R: Read`)
/// in `O(|distinct_groups|)` memory.
pub fn group_by_stream<R: Read>(
    reader: R,
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    dec_options: &DecodeOptions,
) -> Result<Vec<GroupRow>> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut merged: BTreeMap<GroupKey, AggStats> = BTreeMap::new();

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let rows = crate::group_by(eff, where_expr, group_col, agg_col, &sub_dec)?;
            for r in rows {
                merged.entry(r.key).or_default().merge(&r.stats);
            }
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let rows = crate::group_by(tail_eff, where_expr, group_col, agg_col, &sub_dec)?;
        for r in rows {
            merged.entry(r.key).or_default().merge(&r.stats);
        }
    }
    Ok(merged
        .into_iter()
        .map(|(key, stats)| GroupRow { key, stats })
        .collect())
}
