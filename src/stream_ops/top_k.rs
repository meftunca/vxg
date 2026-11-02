//! Bounded-memory `ORDER BY` / `Top-K` streaming execution across concatenated `VXG1` frames.

use std::io::{Read, Write};

use crate::{
    ops::{
        top_k::{scan_frame_into_heap, TopKHeap},
        validate_projection_cols, ColTarget, PredExpr, SortDir,
    },
    DecodeOptions, Error, Result, RowMatch,
};

use super::{write_all_vxg, FramePeeler, FrameReader};

/// Computes the global `Top-K` (`ORDER BY order_col [ASC|DESC] LIMIT limit`) across an arbitrarily
/// large `VXG1` stream (`R: Read`) in `O(single_frame + limit)` memory, propagating the running
/// heap threshold across frames to skip dominated blocks in `O(1)` zone-map time.
pub fn top_k_by_stream<R: Read>(
    reader: R,
    where_expr: Option<&PredExpr<'_>>,
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    dec_options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut heap = TopKHeap::new(limit, dir);
    let mut global_blk = 0usize;
    let mut global_row = 0usize;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let (nb, nr) = scan_frame_into_heap(
                eff, global_blk, global_row, where_expr, None, order_col, &sub_dec, &mut heap,
            )?;
            global_blk += nb;
            global_row += nr;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let _ = scan_frame_into_heap(
            tail_eff, global_blk, global_row, where_expr, None, order_col, &sub_dec, &mut heap,
        )?;
    }
    Ok(heap.into_sorted_matches())
}

/// Computes projected `SELECT cols WHERE ... ORDER BY order_col [ASC|DESC] LIMIT limit` across an
/// arbitrarily large `VXG1` stream (`R: Read`) in `O(single_frame + limit)` memory and writes the
/// sorted projected records into `writer`, returning the number of records emitted.
#[allow(clippy::too_many_arguments)]
pub fn select_top_k_by_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    order_col: ColTarget<'_>,
    dir: SortDir,
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
    let mut heap = TopKHeap::new(limit, dir);
    let mut global_blk = 0usize;
    let mut global_row = 0usize;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let (nb, nr) = scan_frame_into_heap(
                eff,
                global_blk,
                global_row,
                where_expr,
                Some(cols),
                order_col,
                &sub_dec,
                &mut heap,
            )?;
            global_blk += nb;
            global_row += nr;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let _ = scan_frame_into_heap(
            tail_eff,
            global_blk,
            global_row,
            where_expr,
            Some(cols),
            order_col,
            &sub_dec,
            &mut heap,
        )?;
    }
    let matches = heap.into_sorted_matches();
    let count = matches.len() as u64;
    for m in &matches {
        write_all_vxg(&mut writer, &m.bytes)?;
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(count)
}
