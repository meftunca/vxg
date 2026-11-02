//! Stage 2 compressed-domain structured operations (`filter_column`, `filter_frame`,
//! `update_column`, `map_column`) on VXG frames without full-frame decode/re-encode.

use crate::{query, DecodeOptions, Frame, Options, Result, RowMatch, BLOCK_COLUMN};

mod line;
mod transform;

pub use line::{
    apply_line_col_op, apply_token_op, decode_delim_str, encode_delim_str, encode_json_str,
    eval_token_pred, find_delimited_col_span, find_json_key_span, find_line_col_span,
    format_col_str, line_matches_col_pred, unescape_json_str, ColOp, ColPred, ColTarget, StrCmp,
    ValuePred,
};

/// Find up to `limit` records in `frame_bytes` satisfying column predicate `pred`.
///
/// On `GRAMMAR_PACKET` (`BLOCK_COLUMN`) blocks, only the addressed column is inspected
/// (with header-only 256-bit alphabet and numeric zone-map rejection), and non-target
/// columns are materialized only when at least one row in the block matches.
pub fn filter_column(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    let mut out = Vec::new();
    let mut cur_global = 0usize;
    let mut carry: Option<(usize, usize, usize, Vec<u8>)> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut global_blk = 0usize;

    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            let blk_idx = global_blk;
            global_blk += 1;
            if out.len() >= limit && carry.is_none() {
                return Ok(out);
            }
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some(new_rows) = crate::columns::search_grammar_block_col_pred(
                    payload,
                    blk_idx,
                    cur_global,
                    &mut carry,
                    pred,
                    limit,
                    &mut out,
                )? {
                    cur_global += new_rows;
                    if out.len() >= limit && carry.is_none() {
                        return Ok(out);
                    }
                    continue;
                }
            }
            query::decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            let raw: &[u8] = &buf;
            if raw.is_empty() {
                continue;
            }
            let had_carry = carry.is_some();
            let base_row = cur_global - usize::from(had_carry);
            let lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line_bytes) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                        c_bytes.extend_from_slice(line_bytes);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some((c_blk, c_row, c_glob, c_bytes));
                            break;
                        }
                        if line_matches_col_pred(&c_bytes, pred) && out.len() < limit {
                            out.push(RowMatch {
                                block_index: c_blk,
                                row_index: c_row,
                                global_row_index: c_glob,
                                bytes: c_bytes,
                            });
                        }
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some((blk_idx, row_idx, base_row + row_idx, line_bytes.to_vec()));
                    break;
                }
                if line_matches_col_pred(line_bytes, pred) && out.len() < limit {
                    out.push(RowMatch {
                        block_index: blk_idx,
                        row_index: row_idx,
                        global_row_index: base_row + row_idx,
                        bytes: line_bytes.to_vec(),
                    });
                }
            }
            cur_global = base_row + blk_rows;
        }
    }
    if let Some((c_blk, c_row, c_glob, c_bytes)) = carry.take() {
        if line_matches_col_pred(&c_bytes, pred) && out.len() < limit {
            out.push(RowMatch {
                block_index: c_blk,
                row_index: c_row,
                global_row_index: c_glob,
                bytes: c_bytes,
            });
        }
    }
    Ok(out)
}

/// Produce a new canonical VXG frame containing only records satisfying `pred`,
/// skipping non-matching blocks and columns in the compressed domain.
pub fn filter_frame(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    enc_options.validate()?;
    let matches = filter_column(frame_bytes, pred, usize::MAX, dec_options)?;
    let total_len: usize = matches.iter().map(|m| m.bytes.len()).sum();
    let mut raw = Vec::with_capacity(total_len);
    for m in matches {
        raw.extend_from_slice(&m.bytes);
    }
    crate::compress(&raw, enc_options)
}

/// Conditionally transform `target_col` with `op` on every record matching `where_pred`,
/// copying untouched blocks byte-for-byte and splicing only `target_col` on modified
/// `GRAMMAR_PACKET` blocks.
pub fn update_column(
    frame_bytes: &[u8],
    where_pred: &ColPred<'_>,
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    transform::transform_frame(
        frame_bytes,
        Some(where_pred),
        target_col,
        &op,
        enc_options,
        dec_options,
    )
}

/// Transform `target_col` with `op` across every record in `frame_bytes`, splicing only
/// `target_col` on `GRAMMAR_PACKET` blocks without decoding or re-encoding untouched columns.
pub fn map_column(
    frame_bytes: &[u8],
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    transform::transform_frame(frame_bytes, None, target_col, &op, enc_options, dec_options)
}
