//! VXG v0.1: an experimental generic lossless block codec.
//!
//! ```
//! let input = b"abcabcabcabcabcabcabcabcabcabc";
//! let frame = vxg::compress(input, &vxg::Options::default()).unwrap();
//! let output = vxg::decompress(&frame, &vxg::DecodeOptions::default()).unwrap();
//! assert_eq!(output, input);
//! ```
//! See docs/PROOFS.md for the invariants and their exact scope. Performance is measured,
//! not proved. The format is experimental and is not LZ4/Zstandard-compatible.
#[cfg(feature = "bench")]
pub mod analysis;
#[cfg(feature = "bench")]
pub mod baselines;
mod block;
pub mod checksum;
pub mod columns;
pub(crate) mod dict_lz;
mod encode;
mod error;
pub mod ffi;
mod frame;
pub mod gpu;
#[cfg(feature = "bench")]
pub mod groups;
pub mod mmap;
pub mod ops;
#[cfg(feature = "bench")]
pub mod packing;
pub(crate) mod pool;
#[cfg(kani)]
mod proofs;
pub mod query;
pub mod simd;
pub mod sql;
pub mod stream;
pub mod stream_ops;
pub use block::{
    WorkStats, COLUMN as BLOCK_COLUMN, DICT_LZ as BLOCK_DICT_LZ, LZ as BLOCK_LZ, RAW as BLOCK_RAW,
};
pub use error::{Error, Result};
pub use frame::{
    max_compressed_size, recover_frame, BlockInfo, FailedBlock, Frame, RecoveredBlock,
    RecoveryReport, ValidatedFrame,
};
pub use mmap::{
    compress_file, decompress_file, mmap_supported, verify_file, InputBytes, MappedFile,
    MappedFileMut,
};
pub use ops::{
    aggregate_column, count_where, filter_column, filter_frame, filter_frame_where, filter_where,
    group_by, map_column, parse_col_list, parse_col_target, project_frame, select_columns,
    select_top_k_by, top_k_by, update_column, with_opt_where, AggStats, ColOp, ColPred, ColTarget,
    DecCmp, DecNum, GroupKey, GroupRow, ParsedColPred, ParsedWhere, PredExpr, SortDir, SortKey,
    StrCmp, ValuePred,
};
pub use query::{
    block_row_count, extract_block_row, extract_block_row_range, extract_row, extract_row_range,
    filter_json_u64, filter_rows, find_first, frame_row_count, NumCmp, RowMatch,
};
pub use simd::Simd;
pub use sql::{
    execute_sql, execute_sql_stream, execute_sql_table, SqlFormat, SqlQuery, SqlTable, SqlVal,
};
pub use stream::{compress_stream, decompress_stream, StreamDecoder, StreamEncoder};
pub use stream_ops::{
    aggregate_column_stream, count_where_stream, filter_frame_where_stream, filter_where_stream,
    frame_row_count_stream, group_by_stream, map_column_stream, project_frame_stream,
    select_columns_stream, select_top_k_by_stream, top_k_by_stream, update_column_stream,
    verify_frame, verify_stream, FrameReader,
};

#[derive(Clone, Debug)]
pub struct Options {
    pub block_size: usize,
    pub threads: usize,
    pub checksum: bool,
    /// Minimum failed-probe step. Valid range 1..64. No change to losslessness.
    pub acceleration: usize,
    pub simd: Simd,
    /// Emit legacy version-2 LZ-only frames instead of the unified adaptive
    /// release pipeline (v4: structured-column / stride / fast-LZ router with
    /// compact 8-byte directory). Off by default; v1..v3 frames remain readable.
    pub legacy_v2: bool,
    /// Snap block split points backward to newline (`\n`) boundaries when possible (`VERSION_V5`
    /// `FLAG_VAR_RAW`), eliminating cross-block partial rows on multi-GB tabular/NDJSON frames.
    pub record_aligned: bool,
    /// Compression effort level (`1..=9`; default `3`). Higher levels search deeper local/dictionary
    /// hash ways and lazy windows in `BLOCK_DICT_LZ` with zero change to wire format or decode speed.
    pub level: u8,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            block_size: 65536,
            threads: 1,
            checksum: true,
            acceleration: 1,
            simd: Simd::Auto,
            legacy_v2: false,
            record_aligned: false,
            level: 3,
        }
    }
}
impl Options {
    pub(crate) fn validate(&self) -> Result<()> {
        frame::check_block_size(self.block_size)?;
        if !(1..=256).contains(&self.threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        if !(1..=64).contains(&self.acceleration) {
            return Err(Error::InvalidOption("acceleration must be 1..64"));
        }
        if !(1..=9).contains(&self.level) {
            return Err(Error::InvalidOption("level must be 1..9"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    pub threads: usize,
    pub simd: Simd,
    pub max_output: usize,
    pub verify_checksum: bool,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            simd: Simd::Auto,
            max_output: 1 << 30,
            verify_checksum: true,
        }
    }
}
impl DecodeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=256).contains(&self.threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        Ok(())
    }
}

pub fn compress(input: &[u8], options: &Options) -> Result<Vec<u8>> {
    Ok(compress_inner::<false>(input, options)?.0)
}
pub fn compress_with_stats(input: &[u8], options: &Options) -> Result<(Vec<u8>, WorkStats)> {
    compress_inner::<true>(input, options)
}
/// Compresses `input` directly into caller-supplied `out[..written]`, avoiding per-call output
/// buffer allocation on both single-threaded and multi-threaded paths by reusing thread-local scratch arenas.
pub fn compress_into(input: &[u8], out: &mut [u8], options: &Options) -> Result<usize> {
    options.validate()?;
    max_compressed_size(input.len(), options.block_size)?;
    let engine = simd::Engine::new(options.simd)?;
    let ranges = frame::v5::compute_block_ranges(
        input,
        options.block_size,
        options.record_aligned && !options.legacy_v2,
    );
    let count = ranges
        .as_ref()
        .map_or_else(|| input.len().div_ceil(options.block_size), Vec::len);
    let workers = options.threads.min(count.max(1));
    let version = if options.legacy_v2 {
        frame::VERSION
    } else {
        frame::VERSION_RELEASE
    };
    let prep_dict = if !options.legacy_v2 {
        dict_lz::prepare_frame_dict(input, options.block_size, options.level)
    } else {
        None
    };

    let run = |out_buf: &mut [u8], dict_ref: Option<&dict_lz::PreparedDict>| -> Result<usize> {
        if workers == 1 {
            return encode::encode_single_into(
                input,
                out_buf,
                options,
                engine,
                count,
                version,
                ranges.as_deref(),
                dict_ref,
            );
        }
        let next_idx = std::sync::atomic::AtomicUsize::new(0);
        let (written, _) = frame::parallel_encode_assemble(
            workers,
            input.len(),
            count,
            options.block_size,
            options.checksum,
            version,
            ranges.is_some(),
            dict_ref.map(|d| (d.wire_payload.as_slice(), d.raw.len())),
            frame::AssembleDest::Slice(out_buf),
            |worker, finish| {
                encode::encode_worker_tls::<false, _>(
                    input,
                    options,
                    engine,
                    count,
                    ranges.as_deref(),
                    dict_ref,
                    worker,
                    workers,
                    &next_idx,
                    finish,
                )
            },
        )?;
        Ok(written)
    };

    match run(out, prep_dict.as_ref()) {
        Err(Error::InvalidFrame("dict_expansion")) => run(out, None),
        res => res,
    }
}

fn compress_inner<const STATS: bool>(
    input: &[u8],
    options: &Options,
) -> Result<(Vec<u8>, WorkStats)> {
    options.validate()?;
    max_compressed_size(input.len(), options.block_size)?;
    let engine = simd::Engine::new(options.simd)?;
    let ranges = frame::v5::compute_block_ranges(
        input,
        options.block_size,
        options.record_aligned && !options.legacy_v2,
    );
    let count = ranges
        .as_ref()
        .map_or_else(|| input.len().div_ceil(options.block_size), Vec::len);
    let workers = options.threads.min(count.max(1));
    let version = if options.legacy_v2 {
        frame::VERSION
    } else {
        frame::VERSION_RELEASE
    };
    let prep_dict = if !options.legacy_v2 {
        dict_lz::prepare_frame_dict(input, options.block_size, options.level)
    } else {
        None
    };

    let run = |dict_ref: Option<&dict_lz::PreparedDict>| -> Result<(Vec<u8>, WorkStats)> {
        if workers == 1 {
            return encode::encode_single_worker::<STATS>(
                input,
                options,
                engine,
                count,
                version,
                ranges.as_deref(),
                dict_ref,
            );
        }
        let next_idx = std::sync::atomic::AtomicUsize::new(0);
        let mut out = Vec::new();
        let (_, stats) = frame::parallel_encode_assemble(
            workers,
            input.len(),
            count,
            options.block_size,
            options.checksum,
            version,
            ranges.is_some(),
            dict_ref.map(|d| (d.wire_payload.as_slice(), d.raw.len())),
            frame::AssembleDest::Alloc(&mut out),
            |worker, finish| {
                encode::encode_worker_tls::<STATS, _>(
                    input,
                    options,
                    engine,
                    count,
                    ranges.as_deref(),
                    dict_ref,
                    worker,
                    workers,
                    &next_idx,
                    finish,
                )
            },
        )?;
        Ok((out, stats))
    };

    match run(prep_dict.as_ref()) {
        Err(Error::InvalidFrame("dict_expansion")) => run(None),
        res => res,
    }
}

pub fn decompress(src: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    options.validate()?;
    let mut v4_offsets = [core::mem::MaybeUninit::<u32>::uninit(); frame::MAX_STACK_DIR_BLOCKS + 1];
    let (raw_size, block_size, count, version, checksummed) =
        match frame::validate_header_and_directory_with_offsets(
            src,
            options.max_output,
            (options.threads > 1).then_some(&mut v4_offsets),
        ) {
            Ok(v) => v,
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                return stream::decompress_concatenated(src, options);
            }
            Err(e) => return Err(e),
        };
    if raw_size > isize::MAX as usize {
        return Err(Error::SizeOverflow);
    }
    let has_var_raw =
        version >= frame::VERSION_V5 && (src[6] & (frame::v5::FLAG_VAR_RAW as u8)) != 0;
    if options.threads == 1
        || (!has_var_raw
            && (version < frame::VERSION_V4
                || (count <= frame::MAX_STACK_DIR_BLOCKS && src.len() <= u32::MAX as usize)))
    {
        let mut out = vec![0u8; raw_size];
        if options.threads == 1 {
            frame::decode_validated_dir_single_thread(
                src,
                &mut out,
                raw_size,
                block_size,
                count,
                version,
                checksummed,
                options,
            )?;
        } else {
            frame::decode_validated_dir_multi_thread(
                src,
                &mut out,
                raw_size,
                block_size,
                count,
                version,
                checksummed,
                &v4_offsets,
                options,
            )?;
        }
        return Ok(out);
    }
    let frame = Frame::parse(src, options.max_output)?;
    let mut out = vec![0u8; frame.raw_size()];
    frame.decode_into(&mut out, options)?;
    Ok(out)
}

pub fn decompress_into(src: &[u8], out: &mut [u8], options: &DecodeOptions) -> Result<()> {
    options.validate()?;
    let mut v4_offsets = [core::mem::MaybeUninit::<u32>::uninit(); frame::MAX_STACK_DIR_BLOCKS + 1];
    let (raw_size, block_size, count, version, checksummed) =
        match frame::validate_header_and_directory_with_offsets(
            src,
            options.max_output,
            (options.threads > 1).then_some(&mut v4_offsets),
        ) {
            Ok(v) => v,
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                return stream::decompress_concatenated_into(src, out, options);
            }
            Err(e) => return Err(e),
        };
    if options.threads == 1 {
        return frame::decode_validated_dir_single_thread(
            src,
            out,
            raw_size,
            block_size,
            count,
            version,
            checksummed,
            options,
        );
    }
    let has_var_raw =
        version >= frame::VERSION_V5 && (src[6] & (frame::v5::FLAG_VAR_RAW as u8)) != 0;
    if !has_var_raw
        && (version < frame::VERSION_V4
            || (count <= frame::MAX_STACK_DIR_BLOCKS && src.len() <= u32::MAX as usize))
    {
        return frame::decode_validated_dir_multi_thread(
            src,
            out,
            raw_size,
            block_size,
            count,
            version,
            checksummed,
            &v4_offsets,
            options,
        );
    }
    Frame::parse(src, options.max_output)?.decode_into(out, options)
}

/// Zero-allocation random-access decompression of a single block `block_index`
/// directly into `out[..block_raw_size]`.
pub fn extract_block_into(
    src: &[u8],
    block_index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    frame::extract_block_validated_into(src, block_index, out, options)
}
