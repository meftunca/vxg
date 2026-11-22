//! Zero-dependency HTTP/1.1 `Range` random-access & remote SQL engine for `VXG` archives.
//!
//! Fetches ONLY the 32-byte header + compact block directory first, then issues coalesced
//! `Range: bytes=start-end` requests for the exact blocks required by `info`, `extract_block`,
//! `extract_row` ($O(\log B)$ via `FLAG_ROW_INDEX`), or `execute_sql` (with 0-block `COUNT(*)`
//! and lazy early-exit `LIMIT` streaming).

use std::io::{self, Read};

use crate::frame::v5::{
    block_raw_size, block_row_meta_raw, parse_v1_to_v5_header, unpack_block_row_meta,
    ParsedHeader, FLAG_CHECKSUM, FLAG_FRAME_DICT, FLAG_ROW_INDEX, FLAG_VAR_RAW,
};
use crate::frame::{entry_size, read_u32, HEADER_SIZE, MAGIC, V4_SIZE_MASK, VERSION_V4, VERSION_V5};
use crate::sql::{execute_sql_stream, SelectExpr, SelectProjection, SqlFormat, SqlQuery};
use crate::{DecodeOptions, Error, Result};

#[cfg(not(target_arch = "wasm32"))]
pub use crate::remote_http::{serve_http_range, serve_http_range_listener, HttpRangeReader};

/// Wire transfer telemetry recorded by [`RemoteFrame`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RangeStats {
    pub requests: usize,
    pub bytes_fetched: u64,
    pub total_bytes: u64,
}

impl RangeStats {
    /// Percentage of the remote archive fetched so far (`0.0..=100.0`).
    pub fn fetched_pct(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.bytes_fetched as f64) * 100.0 / (self.total_bytes as f64)
        }
    }
}

/// Byte-range reader abstraction (in-memory slice, local file, or `http://` Range origin).
pub trait RangeReader {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_range(&mut self, start: u64, end: u64) -> Result<Vec<u8>>;
}

/// In-memory byte-slice [`RangeReader`] (ideal for tests and `wasm32`).
pub struct SliceRangeReader<'a>(pub &'a [u8]);

impl RangeReader for SliceRangeReader<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_range(&mut self, start: u64, end: u64) -> Result<Vec<u8>> {
        if start > end || end > self.0.len() as u64 {
            return Err(Error::Truncated);
        }
        Ok(self.0[start as usize..end as usize].to_vec())
    }
}

#[derive(Clone, Copy, Debug)]
struct RemoteBlockMeta {
    offset: u64,
    payload_len: u32,
    raw_len: u32,
    word0: u32,
    crc32c: u32,
    row_word: Option<u32>,
}

/// Summary metadata parsed solely from a remote `VXG` header + directory (`0` blocks fetched).
#[derive(Clone, Debug)]
pub struct RemoteFrameInfo {
    pub version: u16,
    pub raw_size: usize,
    pub compressed_size: u64,
    pub block_size: usize,
    pub block_count: usize,
    pub checksummed: bool,
    pub has_dict: bool,
    pub has_row_index: bool,
    pub row_count: Option<u64>,
}

/// Random-access remote `VXG` reader operating over any [`RangeReader`].
pub struct RemoteFrame<R: RangeReader> {
    reader: R,
    hdr: ParsedHeader,
    prefix: Vec<u8>,
    blocks: Vec<RemoteBlockMeta>,
    row_prefix_sum: Vec<u64>,
    stats: RangeStats,
}

impl<R: RangeReader> RemoteFrame<R> {
    /// Open a remote `VXG` frame by fetching strictly its header + compact block directory.
    pub fn open(mut reader: R) -> Result<Self> {
        let total = reader.len();
        if total < HEADER_SIZE as u64 {
            return Err(Error::Truncated);
        }
        let mut stats = RangeStats {
            requests: 0,
            bytes_fetched: 0,
            total_bytes: total,
        };
        let init_len = total.min(1024);
        let mut prefix = reader.read_range(0, init_len)?;
        stats.requests += 1;
        stats.bytes_fetched += prefix.len() as u64;

        let hdr = parse_v1_to_v5_header(&prefix, usize::MAX)?;
        if hdr.version < VERSION_V4 || hdr.version > VERSION_V5 {
            return Err(Error::InvalidFrame(
                "RemoteFrame requires VXG format v4 or v5",
            ));
        }
        let need = hdr.payload_start as u64;
        if need > total {
            return Err(Error::Truncated);
        }
        if (prefix.len() as u64) < need {
            let extra = reader.read_range(prefix.len() as u64, need)?;
            stats.requests += 1;
            stats.bytes_fetched += extra.len() as u64;
            prefix.extend_from_slice(&extra);
        }

        let esize = entry_size(hdr.version);
        let mut blocks = Vec::with_capacity(hdr.count);
        let mut row_prefix_sum = Vec::with_capacity(hdr.count + 1);
        if hdr.has_row_index {
            row_prefix_sum.push(0);
        }
        let mut cur_off = hdr.payload_start as u64;
        let mut next_raw = 0usize;
        for i in 0..hdr.count {
            let e_off = HEADER_SIZE + i * esize;
            let word0 = read_u32(&prefix, e_off)?;
            let payload_len = word0 & V4_SIZE_MASK;
            let crc32c = read_u32(&prefix, e_off + 4)?;
            let rn = block_raw_size(&prefix, &hdr, i, next_raw)?;
            next_raw += rn;
            let row_word = block_row_meta_raw(&prefix, &hdr, i, rn)?;
            if let Some(rw) = row_word {
                let (rows, _) = unpack_block_row_meta(rw);
                let prev = *row_prefix_sum.last().unwrap();
                row_prefix_sum.push(prev + rows as u64);
            }
            blocks.push(RemoteBlockMeta {
                offset: cur_off,
                payload_len,
                raw_len: rn as u32,
                word0,
                crc32c,
                row_word,
            });
            cur_off += payload_len as u64;
        }
        Ok(Self {
            reader,
            hdr,
            prefix,
            blocks,
            row_prefix_sum,
            stats,
        })
    }

    pub fn stats(&self) -> RangeStats {
        self.stats
    }

    pub fn info(&self) -> RemoteFrameInfo {
        RemoteFrameInfo {
            version: self.hdr.version,
            raw_size: self.hdr.raw_size,
            compressed_size: self.stats.total_bytes,
            block_size: self.hdr.block_size,
            block_count: self.blocks.len(),
            checksummed: self.hdr.checksummed,
            has_dict: self.hdr.has_dict,
            has_row_index: self.hdr.has_row_index,
            row_count: self.row_prefix_sum.last().copied(),
        }
    }

    /// Fetch contiguous blocks `[b0..b1)` in a single Range request and wrap them into a valid `v5` sub-frame.
    pub fn fetch_subframe(&mut self, b0: usize, b1: usize) -> Result<Vec<u8>> {
        if b0 >= b1 || b1 > self.blocks.len() {
            return Err(Error::InvalidOption("remote block range out of bounds"));
        }
        let slice = &self.blocks[b0..b1];
        let start_off = slice[0].offset;
        let last = slice[slice.len() - 1];
        let end_off = last.offset + last.payload_len as u64;
        let payload = self.reader.read_range(start_off, end_off)?;
        self.stats.requests += 1;
        self.stats.bytes_fetched += payload.len() as u64;

        let count = slice.len();
        let sub_raw: usize = slice.iter().map(|b| b.raw_len as usize).sum();
        let max_blk_raw = slice.iter().map(|b| b.raw_len as usize).max().unwrap_or(1);
        let mut sub_bs = self.hdr.block_size.max(max_blk_raw);
        if count > 1 && sub_raw > 0 {
            let min_c = sub_raw.div_ceil(sub_bs);
            let max_c = sub_raw.min((min_c.saturating_mul(8) / 3).max(min_c + 4));
            if count > max_c {
                sub_bs = max_blk_raw.next_power_of_two().clamp(4096, 1_048_576);
            }
        }
        let dict_wire = if self.hdr.has_dict {
            &self.prefix[self.hdr.dict_offset..self.hdr.dict_offset + self.hdr.dict_wire_len]
        } else {
            &[]
        };
        let mut flags = FLAG_VAR_RAW;
        if self.hdr.checksummed {
            flags |= FLAG_CHECKSUM;
        }
        if self.hdr.has_dict {
            flags |= FLAG_FRAME_DICT;
        }
        if self.hdr.has_row_index {
            flags |= FLAG_ROW_INDEX;
        }
        let total = HEADER_SIZE
            + count * 8
            + count * 4
            + dict_wire.len()
            + if self.hdr.has_row_index { count * 4 } else { 0 }
            + payload.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION_V5.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&(sub_raw as u64).to_le_bytes());
        out.extend_from_slice(&(sub_bs as u32).to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.extend_from_slice(&(dict_wire.len() as u32).to_le_bytes());
        out.extend_from_slice(
            &if self.hdr.has_dict {
                self.hdr.dict_raw_len as u32
            } else {
                0u32
            }
            .to_le_bytes(),
        );
        for b in slice {
            out.extend_from_slice(&b.word0.to_le_bytes());
            out.extend_from_slice(&b.crc32c.to_le_bytes());
        }
        for b in slice {
            out.extend_from_slice(&b.raw_len.to_le_bytes());
        }
        out.extend_from_slice(dict_wire);
        if self.hdr.has_row_index {
            for b in slice {
                out.extend_from_slice(&b.row_word.unwrap_or(0).to_le_bytes());
            }
        }
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Fetch and decompress a single block `block_idx` (`1` Range request).
    pub fn extract_block(&mut self, block_idx: usize, dec: &DecodeOptions) -> Result<Vec<u8>> {
        let sub = self.fetch_subframe(block_idx, block_idx + 1)?;
        crate::decompress(&sub, dec)
    }

    /// Fetch strictly the single block containing `row_idx` ($O(\log B)$ via `FLAG_ROW_INDEX`)
    /// and return that exact row.
    pub fn extract_row(&mut self, row_idx: usize, dec: &DecodeOptions) -> Result<Option<Vec<u8>>> {
        if self.hdr.has_row_index {
            let total_rows = *self.row_prefix_sum.last().unwrap_or(&0) as usize;
            if row_idx >= total_rows {
                return Ok(None);
            }
            let pos = self.row_prefix_sum.partition_point(|&c| (c as usize) <= row_idx);
            let blk = pos.saturating_sub(1);
            let local_row = row_idx - (self.row_prefix_sum[blk] as usize);
            let sub = self.fetch_subframe(blk, blk + 1)?;
            return Ok(Some(crate::extract_row(&sub, local_row, dec)?));
        }
        let mut seen = 0usize;
        for blk in 0..self.blocks.len() {
            let raw = self.extract_block(blk, dec)?;
            for line in raw.split_inclusive(|&b| b == b'\n') {
                if seen == row_idx {
                    return Ok(Some(line.to_vec()));
                }
                seen += 1;
            }
        }
        Ok(None)
    }

    /// Execute any SQL query over the remote archive using Range requests:
    /// - `EXPLAIN`: **0 block payloads fetched**.
    /// - `SELECT COUNT(*)` (`FLAG_ROW_INDEX`): **at most 1 header-peek block** (`O(1)` wire transfer).
    /// - `LIMIT` scan without `ORDER BY`/`GROUP BY`/`DISTINCT`: lazy Range windows stopping early.
    /// - Full analytical SQL (`GROUP BY`, `HAVING`, `COUNT(DISTINCT)`, `ORDER BY`): coalesced streaming.
    pub fn execute_sql(
        &mut self,
        query: &SqlQuery,
        format: SqlFormat,
        no_header: bool,
        dec: &DecodeOptions,
    ) -> Result<Vec<u8>> {
        if query.explain {
            return crate::sql::having::build_explain_table(None, query, dec)?
                .to_bytes(format, no_header);
        }
        if !query.distinct
            && query.where_clause.is_none()
            && query.group_by.is_empty()
            && query.having.is_none()
            && query.offset == 0
            && self.hdr.has_row_index
        {
            if let SelectProjection::Items(items) = &query.projection {
                if items.len() == 1 && matches!(items[0].expr, SelectExpr::CountStar) {
                    let mut total = *self.row_prefix_sum.last().unwrap_or(&0);
                    if !self.blocks.is_empty() {
                        let sub0 = self.fetch_subframe(0, 1)?;
                        if crate::ops::csv_header::peek_delim_header_fast(&sub0, dec)?.is_some() {
                            total = total.saturating_sub(1);
                        }
                    }
                    let tbl = crate::SqlTable {
                        headers: vec![items[0].header_name()],
                        rows: if query.limit == Some(0) {
                            Vec::new()
                        } else {
                            vec![vec![crate::SqlVal::Int(total)]]
                        },
                    };
                    return tbl.to_bytes(format, no_header);
                }
            }
        }

        let mut cursor = RemoteWindowStream {
            remote: self,
            next_blk: 0,
            window: if query.limit.is_some()
                && query.order_by.is_empty()
                && query.group_by.is_empty()
                && !query.distinct
            {
                4
            } else {
                64
            },
            buf: Vec::new(),
            pos: 0,
        };
        let mut out = Vec::new();
        execute_sql_stream(&mut cursor, &mut out, query, format, no_header, dec)?;
        Ok(out)
    }
}

struct RemoteWindowStream<'a, R: RangeReader> {
    remote: &'a mut RemoteFrame<R>,
    next_blk: usize,
    window: usize,
    buf: Vec<u8>,
    pos: usize,
}

impl<R: RangeReader> Read for RemoteWindowStream<'_, R> {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.buf.len() {
            if self.next_blk >= self.remote.blocks.len() {
                return Ok(0);
            }
            let b0 = self.next_blk;
            let b1 = (b0 + self.window).min(self.remote.blocks.len());
            self.next_blk = b1;
            self.buf = self
                .remote
                .fetch_subframe(b0, b1)
                .map_err(|e| io::Error::other(e.to_string()))?;
            self.pos = 0;
        }
        let avail = &self.buf[self.pos..];
        let n = avail.len().min(dst.len());
        dst[..n].copy_from_slice(&avail[..n]);
        self.pos += n;
        Ok(n)
    }
}
