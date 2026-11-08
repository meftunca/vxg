//! Compressed-domain in-memory / `mmap` SQL execution engine.

use std::cmp::Ordering;

use crate::{
    ops::{
        csv_header::peek_delim_header,
        line_matches_expr,
        multi_agg::{aggregate_multi_columns, multi_group_by, MultiGroupRow},
        AggStats, ColTarget, GroupKey, PredExpr, SortDir,
    },
    DecodeOptions, Error, Result,
};

use super::format::{self, cmp_sql_val, SqlFormat, SqlTable, SqlVal};
use super::project::{exec_projection, exec_star};
use super::{
    GroupItem, OrderItem, OrderKey, OwnedCol, SelectExpr, SelectItem, SelectProjection, SqlQuery,
    SqlWhere,
};

pub(crate) fn with_opt_sql_where<R, F: FnOnce(Option<&PredExpr<'_>>) -> R>(
    opt_w: Option<&SqlWhere>,
    f: F,
) -> R {
    match opt_w {
        None => f(None),
        Some(w) => w.with_pred_expr(|e| f(Some(e))),
    }
}

pub(crate) fn resolve_group_cols(
    items: &[SelectItem],
    group_by: &[GroupItem],
) -> Result<Vec<OwnedCol>> {
    let mut out = Vec::with_capacity(group_by.len());
    for g in group_by {
        let col = match g {
            GroupItem::Ordinal(ord) => {
                let Some(it) = items.get(ord - 1) else {
                    return Err(Error::InvalidOption("GROUP BY ordinal out of range"));
                };
                match &it.expr {
                    SelectExpr::Col(c) => c.clone(),
                    _ => return Err(Error::InvalidOption("GROUP BY ordinal must refer to a column")),
                }
            }
            GroupItem::Col(c) => {
                if let OwnedCol::JsonKey(name) = c {
                    if let Some(it) = items
                        .iter()
                        .find(|it| it.alias.as_deref() == Some(name.as_str()))
                    {
                        if let SelectExpr::Col(orig) = &it.expr {
                            orig.clone()
                        } else {
                            c.clone()
                        }
                    } else {
                        c.clone()
                    }
                } else {
                    c.clone()
                }
            }
        };
        if !out.contains(&col) {
            out.push(col);
        }
    }
    Ok(out)
}

pub(crate) fn resolve_order_expr<'a>(
    items: &'a [SelectItem],
    key: &'a OrderKey,
) -> Result<(Option<usize>, Option<&'a SelectExpr>)> {
    match key {
        OrderKey::Ordinal(ord) => {
            if *ord == 0 || *ord > items.len() {
                return Err(Error::InvalidOption("ORDER BY ordinal out of range"));
            }
            Ok((Some(*ord - 1), Some(&items[*ord - 1].expr)))
        }
        OrderKey::Expr(e) => {
            if let SelectExpr::Col(OwnedCol::JsonKey(name)) = e {
                if let Some(pos) = items
                    .iter()
                    .position(|it| it.alias.as_deref() == Some(name.as_str()))
                {
                    return Ok((Some(pos), Some(&items[pos].expr)));
                }
            }
            if let Some(pos) = items.iter().position(|it| &it.expr == e) {
                return Ok((Some(pos), Some(&items[pos].expr)));
            }
            Ok((None, Some(e)))
        }
    }
}

pub(crate) fn eval_group_expr(
    expr: &SelectExpr,
    group_cols: &[OwnedCol],
    agg_cols: &[OwnedCol],
    keys: &[GroupKey],
    row_count: u64,
    aggs: &[AggStats],
) -> Result<SqlVal> {
    match expr {
        SelectExpr::CountStar => Ok(SqlVal::Int(row_count)),
        SelectExpr::Col(c) => {
            let Some(pos) = group_cols.iter().position(|gc| gc == c) else {
                return Err(Error::InvalidOption(
                    "SELECT/ORDER BY column must appear in GROUP BY or aggregate",
                ));
            };
            Ok(SqlVal::from_group_key(&keys[pos]))
        }
        SelectExpr::Agg { func, col } => {
            let pos = agg_cols
                .iter()
                .position(|ac| ac == col)
                .expect("agg_col registered");
            Ok(SqlVal::from_agg(*func, &aggs[pos]))
        }
    }
}

pub(crate) fn finish_group_table(
    items: &[SelectItem],
    order_by: &[OrderItem],
    limit: Option<usize>,
    group_cols: &[OwnedCol],
    agg_cols: &[OwnedCol],
    mrows: Vec<MultiGroupRow>,
) -> Result<SqlTable> {
    let headers: Vec<String> = items.iter().map(SelectItem::header_name).collect();
    if limit == Some(0) {
        return Ok(SqlTable {
            headers,
            rows: Vec::new(),
        });
    }
    let mut resolved_orders: Vec<(Option<usize>, Option<&SelectExpr>, SortDir)> =
        Vec::with_capacity(order_by.len());
    for oi in order_by {
        let (idx_opt, expr_opt) = resolve_order_expr(items, &oi.key)?;
        resolved_orders.push((idx_opt, expr_opt, oi.dir));
    }

    let mut decorated: Vec<(Vec<SqlVal>, Vec<SqlVal>)> = Vec::with_capacity(mrows.len());
    for mr in &mrows {
        let mut row = Vec::with_capacity(items.len());
        for it in items {
            row.push(eval_group_expr(
                &it.expr,
                group_cols,
                agg_cols,
                &mr.keys,
                mr.row_count,
                &mr.aggs,
            )?);
        }
        let mut extra = Vec::new();
        for &(idx_opt, expr_opt, _) in &resolved_orders {
            if idx_opt.is_none() {
                if let Some(ex) = expr_opt {
                    extra.push(eval_group_expr(
                        ex,
                        group_cols,
                        agg_cols,
                        &mr.keys,
                        mr.row_count,
                        &mr.aggs,
                    )?);
                }
            }
        }
        decorated.push((row, extra));
    }

    if !resolved_orders.is_empty() {
        decorated.sort_by(|(ra, ea), (rb, eb)| {
            let mut ei = 0usize;
            for &(idx_opt, _, dir) in &resolved_orders {
                let (va, vb) = if let Some(idx) = idx_opt {
                    (&ra[idx], &rb[idx])
                } else {
                    let pair = (&ea[ei], &eb[ei]);
                    ei += 1;
                    pair
                };
                let ord = cmp_sql_val(va, vb, dir);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        });
    }

    if let Some(lim) = limit {
        decorated.truncate(lim);
    }
    Ok(SqlTable {
        headers,
        rows: decorated.into_iter().map(|(r, _)| r).collect(),
    })
}

pub(crate) fn parse_projected_lines_to_rows(
    proj_bytes: &[u8],
    proj_cols: &[OwnedCol],
    delim_opt: Option<u8>,
) -> Vec<Vec<SqlVal>> {
    if proj_bytes.is_empty() {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for line in proj_bytes.split_inclusive(|&b| b == b'\n') {
        let mut cells = Vec::with_capacity(proj_cols.len());
        if let Some(delim) = delim_opt {
            for idx in 0..proj_cols.len() {
                cells.push(format::extract_line_sql_val(
                    line,
                    ColTarget::Delimited { delim, index: idx },
                ));
            }
        } else {
            for col in proj_cols {
                cells.push(format::extract_line_sql_val(line, col.as_target()));
            }
        }
        rows.push(cells);
    }
    rows
}

/// Execute `query` on `frame_bytes` directly in the compressed domain and materialize a [`SqlTable`].
pub fn execute_sql_table(
    frame_bytes: &[u8],
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
    with_opt_sql_where(query.where_clause.as_ref(), |where_expr| match &query.projection {
        SelectProjection::Star => exec_star(frame_bytes, where_expr, query, dec_options),
        SelectProjection::Items(items) => {
            let has_agg = items.iter().any(|it| it.expr.is_agg());
            if !query.group_by.is_empty() {
                return exec_group_by(frame_bytes, items, where_expr, query, dec_options);
            }
            if has_agg {
                return exec_scalar_agg(frame_bytes, items, where_expr, query.limit, dec_options);
            }
            exec_projection(frame_bytes, items, where_expr, query, dec_options)
        }
    })
}

fn exec_scalar_agg(
    frame_bytes: &[u8],
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    limit: Option<usize>,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
    if items.iter().any(|it| matches!(it.expr, SelectExpr::Col(_))) {
        return Err(Error::InvalidOption(
            "non-aggregate column in SELECT without GROUP BY",
        ));
    }
    let headers: Vec<String> = items.iter().map(SelectItem::header_name).collect();
    if limit == Some(0) {
        return Ok(SqlTable {
            headers,
            rows: Vec::new(),
        });
    }
    let mut agg_cols: Vec<OwnedCol> = Vec::new();
    for it in items {
        if let SelectExpr::Agg { col, .. } = &it.expr {
            if !agg_cols.contains(col) {
                agg_cols.push(col.clone());
            }
        }
    }

    let (row_count, col_stats) = if agg_cols.is_empty() {
        let cnt = match where_expr {
            None => {
                let mut n = crate::frame_row_count(frame_bytes)? as u64;
                if peek_delim_header(frame_bytes, dec_options)?.is_some() {
                    n = n.saturating_sub(1);
                }
                n
            }
            Some(w) => {
                let mut n = crate::count_where(frame_bytes, w, dec_options)?;
                if !crate::ops::csv_header::expr_contains_json_key(w) {
                    if let Some(hdr) = peek_delim_header(frame_bytes, dec_options)? {
                        if line_matches_expr(&hdr.header_line, w) {
                            n = n.saturating_sub(1);
                        }
                    }
                }
                n
            }
        };
        (cnt, Vec::new())
    } else if agg_cols.len() == 1 {
        let acol = agg_cols[0].as_target();
        let mut st = crate::aggregate_column(frame_bytes, where_expr, acol, dec_options)?;
        if matches!(acol, ColTarget::Delimited { .. })
            && where_expr.is_none_or(|w| !crate::ops::csv_header::expr_contains_json_key(w))
        {
            if let Some(hdr) = peek_delim_header(frame_bytes, dec_options)? {
                crate::ops::csv_header::subtract_header_agg(&hdr, where_expr, acol, &mut st);
            }
        }
        (st.row_count, vec![st])
    } else {
        let targets: Vec<ColTarget<'_>> = agg_cols.iter().map(OwnedCol::as_target).collect();
        let acc = aggregate_multi_columns(frame_bytes, where_expr, &targets, dec_options)?;
        (acc.row_count, acc.cols)
    };

    let mut row = Vec::with_capacity(items.len());
    for it in items {
        match &it.expr {
            SelectExpr::CountStar => row.push(SqlVal::Int(row_count)),
            SelectExpr::Agg { func, col } => {
                let pos = agg_cols.iter().position(|c| c == col).unwrap();
                row.push(SqlVal::from_agg(*func, &col_stats[pos]));
            }
            SelectExpr::Col(_) => unreachable!(),
        }
    }
    Ok(SqlTable {
        headers,
        rows: vec![row],
    })
}

fn exec_group_by(
    frame_bytes: &[u8],
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
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

    let mrows = if group_cols.len() == 1 && agg_cols.len() <= 1 {
        let gcol = group_cols[0].as_target();
        let acol = agg_cols.first().map(OwnedCol::as_target);
        let single_rows = crate::group_by(frame_bytes, where_expr, gcol, acol, dec_options)?;
        single_rows
            .into_iter()
            .map(|r| MultiGroupRow {
                keys: vec![r.key],
                row_count: r.stats.row_count,
                aggs: if acol.is_some() {
                    vec![r.stats]
                } else {
                    Vec::new()
                },
            })
            .collect()
    } else {
        let g_targets: Vec<ColTarget<'_>> = group_cols.iter().map(OwnedCol::as_target).collect();
        let a_targets: Vec<ColTarget<'_>> = agg_cols.iter().map(OwnedCol::as_target).collect();
        multi_group_by(frame_bytes, where_expr, &g_targets, &a_targets, dec_options)?
    };

    finish_group_table(
        items,
        &query.order_by,
        query.limit,
        &group_cols,
        &agg_cols,
        mrows,
    )
}

/// Execute `query` on `frame_bytes` directly in the compressed domain and serialize to `format`.
pub fn execute_sql(
    frame_bytes: &[u8],
    query: &SqlQuery,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if matches!(query.projection, SelectProjection::Star)
        && query.group_by.is_empty()
        && format != SqlFormat::Csv
        && peek_delim_header(frame_bytes, dec_options)?.is_none()
    {
        let lim = query.limit.unwrap_or(usize::MAX);
        if lim == 0 {
            return Ok(Vec::new());
        }
        if query.order_by.is_empty() {
            if query.where_clause.is_none() && query.limit.is_none() {
                return crate::decompress(frame_bytes, dec_options);
            }
            return with_opt_sql_where(query.where_clause.as_ref(), |w| {
                let rows = crate::filter_where(
                    frame_bytes,
                    w.unwrap_or(&PredExpr::True),
                    lim,
                    dec_options,
                )?;
                let cap: usize = rows.iter().map(|r| r.bytes.len()).sum();
                let mut out = Vec::with_capacity(cap);
                for r in rows {
                    out.extend_from_slice(&r.bytes);
                }
                Ok(out)
            });
        }
        if query.order_by.len() == 1 {
            let oi = &query.order_by[0];
            if let OrderKey::Expr(SelectExpr::Col(ref c)) = oi.key {
                return with_opt_sql_where(query.where_clause.as_ref(), |w| {
                    let rows =
                        crate::top_k_by(frame_bytes, w, c.as_target(), oi.dir, lim, dec_options)?;
                    let cap: usize = rows.iter().map(|r| r.bytes.len()).sum();
                    let mut out = Vec::with_capacity(cap);
                    for r in rows {
                        out.extend_from_slice(&r.bytes);
                    }
                    Ok(out)
                });
            }
        }
    }
    let table = execute_sql_table(frame_bytes, query, dec_options)?;
    table.to_bytes(format, no_header)
}
