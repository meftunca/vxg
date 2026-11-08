//! Bounded-memory multi-frame streaming SQL execution (`execute_sql_stream`).

use std::io::{Read, Write};

use crate::{
    ops::{
        csv_header::{
            peek_delim_header, resolve_header_for_targets, with_opt_mapped_expr, DelimHeader,
            MappedExpr,
        },
        line_matches_expr,
        multi_agg::{
            aggregate_multi_columns, multi_group_by, subtract_header_multi_group, MultiAggAcc,
            MultiGroupMap, MultiGroupRow,
        },
        ColTarget, PredExpr,
    },
    stream_ops::{write_all_vxg, FramePeeler, FrameReader},
    DecodeOptions, Error, Result,
};

use super::exec::{
    execute_sql, finish_group_table, resolve_group_cols, resolve_order_expr, with_opt_sql_where,
};
use super::format::{self, SqlFormat, SqlTable, SqlVal};
use super::project::stream_projection;
use super::{OrderKey, OwnedCol, SelectExpr, SelectItem, SelectProjection, SqlQuery};

/// Execute `query` over a `VXG1` stream (`R: Read`) with bounded `O(single_frame)` /
/// `O(|distinct_groups|)` memory and write formatted output to `writer`.
pub fn execute_sql_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    query: &SqlQuery,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<()> {
    dec_options.validate()?;
    with_opt_sql_where(query.where_clause.as_ref(), |where_expr| match &query.projection {
        SelectProjection::Items(items) if !query.group_by.is_empty() => stream_group_by(
            reader,
            &mut writer,
            items,
            where_expr,
            query,
            format,
            no_header,
            dec_options,
        ),
        SelectProjection::Items(items) if items.iter().any(|it| it.expr.is_agg()) => {
            stream_scalar_agg(
                reader,
                &mut writer,
                items,
                where_expr,
                query.limit,
                format,
                no_header,
                dec_options,
            )
        }
        SelectProjection::Items(items) => stream_projection(
            reader,
            &mut writer,
            items,
            where_expr,
            query,
            format,
            no_header,
            dec_options,
        ),
        SelectProjection::Star => stream_star(
            reader,
            &mut writer,
            where_expr,
            query,
            format,
            no_header,
            dec_options,
        ),
    })
}

fn scalar_agg_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    targets: &[ColTarget<'e>],
    sub_dec: &DecodeOptions,
    first: &mut bool,
    st: &mut Option<(Option<MappedExpr<'e>>, Vec<ColTarget<'e>>)>,
) -> Result<MultiAggAcc> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, targets, sub_dec)? {
            let m_cols: Vec<ColTarget<'e>> = targets
                .iter()
                .map(|&c| hdr.resolve_col(c).unwrap_or(c))
                .collect();
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            let part = aggregate_multi_columns(eff, where_expr, targets, sub_dec)?;
            *st = Some((m_where, m_cols));
            return Ok(part);
        } else if targets.is_empty() {
            if let Some(hdr) = peek_delim_header(eff, sub_dec)? {
                let mut part = aggregate_multi_columns(eff, where_expr, &[], sub_dec)?;
                if where_expr.is_none_or(|w| line_matches_expr(&hdr.header_line, w)) {
                    part.row_count = part.row_count.saturating_sub(1);
                }
                return Ok(part);
            }
        }
    }
    if let Some((ref m_where, ref m_cols)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            aggregate_multi_columns(eff, mw, m_cols, sub_dec)
        })
    } else {
        aggregate_multi_columns(eff, where_expr, targets, sub_dec)
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_scalar_agg<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    limit: Option<usize>,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<()> {
    if items.iter().any(|it| matches!(it.expr, SelectExpr::Col(_))) {
        return Err(Error::InvalidOption(
            "non-aggregate column in SELECT without GROUP BY",
        ));
    }
    let headers: Vec<String> = items.iter().map(SelectItem::header_name).collect();
    if limit == Some(0) {
        return SqlTable {
            headers,
            rows: Vec::new(),
        }
        .write_formatted(writer, format, no_header);
    }
    let mut agg_cols: Vec<OwnedCol> = Vec::new();
    for it in items {
        if let SelectExpr::Agg { col, .. } = &it.expr {
            if !agg_cols.contains(col) {
                agg_cols.push(col.clone());
            }
        }
    }
    let targets: Vec<ColTarget<'_>> = agg_cols.iter().map(OwnedCol::as_target).collect();

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = MultiAggAcc::with_cols(targets.len());
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            total.merge(&scalar_agg_step(
                eff,
                where_expr,
                &targets,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )?);
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        total.merge(&scalar_agg_step(
            tail_eff,
            where_expr,
            &targets,
            &sub_dec,
            &mut first_seen,
            &mut mapped_state,
        )?);
    }

    let mut row = Vec::with_capacity(items.len());
    for it in items {
        match &it.expr {
            SelectExpr::CountStar => row.push(SqlVal::Int(total.row_count)),
            SelectExpr::Agg { func, col } => {
                let pos = agg_cols.iter().position(|c| c == col).unwrap();
                row.push(SqlVal::from_agg(*func, &total.cols[pos]));
            }
            SelectExpr::Col(_) => unreachable!(),
        }
    }
    SqlTable {
        headers,
        rows: vec![row],
    }
    .write_formatted(writer, format, no_header)
}

type GrpState<'e> = (
    Option<MappedExpr<'e>>,
    Vec<ColTarget<'e>>,
    Vec<ColTarget<'e>>,
    DelimHeader,
);

fn group_by_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    g_targets: &[ColTarget<'e>],
    a_targets: &[ColTarget<'e>],
    sub_dec: &DecodeOptions,
    first: &mut bool,
    st: &mut Option<GrpState<'e>>,
) -> Result<Vec<MultiGroupRow>> {
    if !*first {
        *first = true;
        let mut extras = Vec::with_capacity(g_targets.len() + a_targets.len());
        extras.extend_from_slice(g_targets);
        extras.extend_from_slice(a_targets);
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, &extras, sub_dec)? {
            let mg: Vec<ColTarget<'e>> = g_targets
                .iter()
                .map(|&c| hdr.resolve_col(c).unwrap_or(c))
                .collect();
            let ma: Vec<ColTarget<'e>> = a_targets
                .iter()
                .map(|&c| hdr.resolve_col(c).unwrap_or(c))
                .collect();
            let mw_owned = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            let rows = with_opt_mapped_expr(mw_owned.as_ref(), |mw| {
                let mut rs = multi_group_by(eff, mw, &mg, &ma, sub_dec)?;
                subtract_header_multi_group(&hdr, mw, &mg, &mut rs);
                Ok::<Vec<MultiGroupRow>, Error>(rs)
            })?;
            *st = Some((mw_owned, mg, ma, hdr));
            return Ok(rows);
        }
    }
    if let Some((ref mw_owned, ref mg, ref ma, _)) = *st {
        with_opt_mapped_expr(mw_owned.as_ref(), |mw| multi_group_by(eff, mw, mg, ma, sub_dec))
    } else {
        multi_group_by(eff, where_expr, g_targets, a_targets, sub_dec)
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_group_by<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<()> {
    let group_cols = resolve_group_cols(items, &query.group_by)?;
    for it in items {
        if let SelectExpr::Col(c) = &it.expr {
            if !group_cols.contains(c) {
                return Err(Error::InvalidOption(
                    "SELECT column must appear in GROUP BY or aggregate",
                ));
            }
        }
    }
    let mut agg_cols: Vec<OwnedCol> = Vec::new();
    for it in items {
        if let SelectExpr::Agg { col, .. } = &it.expr {
            if !agg_cols.contains(col) {
                agg_cols.push(col.clone());
            }
        }
    }
    for oi in &query.order_by {
        let (_, ex_opt) = resolve_order_expr(items, &oi.key)?;
        if let Some(SelectExpr::Agg { col, .. }) = ex_opt {
            if !agg_cols.contains(col) {
                agg_cols.push(col.clone());
            }
        }
    }
    let g_targets: Vec<ColTarget<'_>> = group_cols.iter().map(OwnedCol::as_target).collect();
    let a_targets: Vec<ColTarget<'_>> = agg_cols.iter().map(OwnedCol::as_target).collect();

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut merged = MultiGroupMap::default();
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            for r in group_by_step(
                eff,
                where_expr,
                &g_targets,
                &a_targets,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )? {
                merged.add_entry(r.keys, r.row_count, &r.aggs);
            }
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        for r in group_by_step(
            tail_eff,
            where_expr,
            &g_targets,
            &a_targets,
            &sub_dec,
            &mut first_seen,
            &mut mapped_state,
        )? {
            merged.add_entry(r.keys, r.row_count, &r.aggs);
        }
    }

    let tbl = finish_group_table(
        items,
        &query.order_by,
        query.limit,
        &group_cols,
        &agg_cols,
        merged.into_rows(),
    )?;
    tbl.write_formatted(writer, format, no_header)
}

#[allow(clippy::too_many_arguments)]
fn star_emit_step<'e, W: Write>(
    eff: &[u8],
    w_out: &mut W,
    where_expr: Option<&PredExpr<'e>>,
    format: SqlFormat,
    no_header: bool,
    sub_dec: &DecodeOptions,
    rem_n: &mut usize,
    first: &mut bool,
    hst: &mut Option<(DelimHeader, Option<MappedExpr<'e>>)>,
) -> Result<()> {
    let expr = where_expr.unwrap_or(&PredExpr::True);
    if !*first {
        *first = true;
        if let Some(hdr) = peek_delim_header(eff, sub_dec)? {
            if !no_header {
                SqlTable {
                    headers: hdr.columns.clone(),
                    rows: Vec::new(),
                }
                .write_formatted(&mut *w_out, format, false)?;
            }
            let mw = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            if *rem_n > 0 {
                let mut ms = with_opt_mapped_expr(mw.as_ref(), |me| {
                    crate::ops::expr::filter_where_impl(
                        eff,
                        me.unwrap_or(&PredExpr::True),
                        rem_n.saturating_add(1),
                        sub_dec,
                    )
                })?;
                ms.retain(|m| m.global_row_index != 0);
                ms.truncate(*rem_n);
                write_csv_star_matches(&mut *w_out, &hdr, &ms, format)?;
                *rem_n = rem_n.saturating_sub(ms.len());
            }
            *hst = Some((hdr, mw));
            return Ok(());
        }
    }
    if *rem_n == 0 {
        return Ok(());
    }
    if let Some((ref hdr, ref mw)) = *hst {
        let ms = with_opt_mapped_expr(mw.as_ref(), |me| {
            crate::ops::expr::filter_where_impl(
                eff,
                me.unwrap_or(&PredExpr::True),
                *rem_n,
                sub_dec,
            )
        })?;
        write_csv_star_matches(&mut *w_out, hdr, &ms, format)?;
        *rem_n = rem_n.saturating_sub(ms.len());
    } else {
        let ms = crate::ops::expr::filter_where_impl(eff, expr, *rem_n, sub_dec)?;
        for m in &ms {
            write_all_vxg(&mut *w_out, &m.bytes)?;
        }
        *rem_n = rem_n.saturating_sub(ms.len());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_star<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<()> {
    if !query.group_by.is_empty() {
        return Err(Error::InvalidOption(
            "SELECT * cannot be combined with GROUP BY",
        ));
    }
    let lim = query.limit.unwrap_or(usize::MAX);
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;

    if query.order_by.is_empty() {
        let mut rem = lim;
        let mut first_seen = false;
        let mut hdr_state = None;

        while let Some(f_bytes) = fr.next_frame()? {
            if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
                star_emit_step(
                    eff,
                    writer,
                    where_expr,
                    format,
                    no_header,
                    &sub_dec,
                    &mut rem,
                    &mut first_seen,
                    &mut hdr_state,
                )?;
                if rem == 0 && first_seen {
                    break;
                }
            }
        }
        if rem > 0 || !first_seen {
            if let Some(tail_eff) = peeler.finish_tail()? {
                star_emit_step(
                    tail_eff,
                    writer,
                    where_expr,
                    format,
                    no_header,
                    &sub_dec,
                    &mut rem,
                    &mut first_seen,
                    &mut hdr_state,
                )?;
            }
        }
        return writer.flush().map_err(|e| Error::Backend(e.to_string()));
    }

    if query.order_by.len() == 1 {
        let oi = &query.order_by[0];
        if let OrderKey::Expr(SelectExpr::Col(ref c)) = oi.key {
            let mut combined = Vec::new();
            while let Some(f_bytes) = fr.next_frame()? {
                combined.extend_from_slice(f_bytes);
            }
            if combined.is_empty() {
                return writer.flush().map_err(|e| Error::Backend(e.to_string()));
            }
            if format != SqlFormat::Csv && peek_delim_header(&combined, &sub_dec)?.is_none() {
                let matches = crate::top_k_by_stream(
                    std::io::Cursor::new(&combined),
                    where_expr,
                    c.as_target(),
                    oi.dir,
                    lim,
                    dec_options,
                )?;
                for m in &matches {
                    write_all_vxg(writer, &m.bytes)?;
                }
                return writer.flush().map_err(|e| Error::Backend(e.to_string()));
            }
            let out = execute_sql(&combined, query, format, no_header, dec_options)?;
            write_all_vxg(writer, &out)?;
            return writer.flush().map_err(|e| Error::Backend(e.to_string()));
        }
    }

    let mut combined = Vec::new();
    while let Some(f_bytes) = fr.next_frame()? {
        combined.extend_from_slice(f_bytes);
    }
    if combined.is_empty() {
        return writer.flush().map_err(|e| Error::Backend(e.to_string()));
    }
    let out = execute_sql(&combined, query, format, no_header, dec_options)?;
    write_all_vxg(writer, &out)?;
    writer.flush().map_err(|e| Error::Backend(e.to_string()))
}

fn write_csv_star_matches<W: Write>(
    writer: &mut W,
    hdr: &DelimHeader,
    matches: &[crate::RowMatch],
    format: SqlFormat,
) -> Result<()> {
    if matches.is_empty() {
        return Ok(());
    }
    let rows: Vec<Vec<SqlVal>> = matches
        .iter()
        .map(|m| {
            (0..hdr.columns.len())
                .map(|idx| {
                    format::extract_line_sql_val(
                        &m.bytes,
                        ColTarget::Delimited {
                            delim: hdr.delim,
                            index: idx,
                        },
                    )
                })
                .collect()
        })
        .collect();
    SqlTable {
        headers: hdr.columns.clone(),
        rows,
    }
    .write_formatted(writer, format, true)
}
