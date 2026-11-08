//! Bounded-memory `ORDER BY` / `Top-K` streaming execution across concatenated `VXG1` frames.

use std::io::{Read, Write};

use crate::{
    ops::{
        csv_header::{resolve_header_for_targets, with_opt_mapped_expr, MappedExpr},
        top_k::{scan_frame_into_heap, TopKHeap},
        validate_projection_cols, ColTarget, PredExpr, SortDir,
    },
    DecodeOptions, Error, Result, RowMatch,
};

use super::{write_all_vxg, FramePeeler, FrameReader};

#[allow(clippy::too_many_arguments)]
fn top_k_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    order_col: ColTarget<'e>,
    sub_dec: &DecodeOptions,
    g_blk: &mut usize,
    g_row: &mut usize,
    first: &mut bool,
    st: &mut Option<(Option<MappedExpr<'e>>, ColTarget<'e>)>,
    hp: &mut TopKHeap,
) -> Result<()> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, &[order_col], sub_dec)? {
            hp.set_skip_row_0(true);
            let m_ocol = hdr.resolve_col(order_col).unwrap_or(order_col);
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            *st = Some((m_where, m_ocol));
        }
    }
    let (nb, nr) = if let Some((ref m_where, m_ocol)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            scan_frame_into_heap(eff, *g_blk, *g_row, mw, None, m_ocol, sub_dec, hp)
        })?
    } else {
        scan_frame_into_heap(eff, *g_blk, *g_row, where_expr, None, order_col, sub_dec, hp)?
    };
    *g_blk += nb;
    *g_row += nr;
    Ok(())
}

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
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            top_k_step(
                eff,
                where_expr,
                order_col,
                &sub_dec,
                &mut global_blk,
                &mut global_row,
                &mut first_seen,
                &mut mapped_state,
                &mut heap,
            )?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        top_k_step(
            tail_eff,
            where_expr,
            order_col,
            &sub_dec,
            &mut global_blk,
            &mut global_row,
            &mut first_seen,
            &mut mapped_state,
            &mut heap,
        )?;
    }
    Ok(heap.into_sorted_matches())
}

type SelectTopKState<'e> = (Option<MappedExpr<'e>>, Vec<ColTarget<'e>>, ColTarget<'e>);

#[allow(clippy::too_many_arguments)]
fn select_top_k_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    cols: &[ColTarget<'e>],
    order_col: ColTarget<'e>,
    sub_dec: &DecodeOptions,
    g_blk: &mut usize,
    g_row: &mut usize,
    first: &mut bool,
    st: &mut Option<SelectTopKState<'e>>,
    hp: &mut TopKHeap,
) -> Result<()> {
    if !*first {
        *first = true;
        let mut extras = Vec::with_capacity(cols.len() + 1);
        extras.extend_from_slice(cols);
        extras.push(order_col);
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, &extras, sub_dec)? {
            hp.set_skip_row_0(true);
            let m_cols: Vec<ColTarget<'e>> = cols
                .iter()
                .map(|&c| hdr.resolve_col(c).unwrap_or(c))
                .collect();
            let m_ocol = hdr.resolve_col(order_col).unwrap_or(order_col);
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            *st = Some((m_where, m_cols, m_ocol));
        }
    }
    let (nb, nr) = if let Some((ref m_where, ref m_cols, m_ocol)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            scan_frame_into_heap(eff, *g_blk, *g_row, mw, Some(m_cols), m_ocol, sub_dec, hp)
        })?
    } else {
        scan_frame_into_heap(
            eff,
            *g_blk,
            *g_row,
            where_expr,
            Some(cols),
            order_col,
            sub_dec,
            hp,
        )?
    };
    *g_blk += nb;
    *g_row += nr;
    Ok(())
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
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            select_top_k_step(
                eff,
                where_expr,
                cols,
                order_col,
                &sub_dec,
                &mut global_blk,
                &mut global_row,
                &mut first_seen,
                &mut mapped_state,
                &mut heap,
            )?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        select_top_k_step(
            tail_eff,
            where_expr,
            cols,
            order_col,
            &sub_dec,
            &mut global_blk,
            &mut global_row,
            &mut first_seen,
            &mut mapped_state,
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
