//! Compressed-domain `SELECT *` and non-aggregate column projection (`SELECT cols ...`).

use std::cmp::Ordering;
use std::io::{Read, Write};

use crate::{
    ops::{
        csv_header::{peek_delim_header, resolve_header_for_targets},
        ColTarget, PredExpr, SortDir,
    },
    DecodeOptions, Error, Result,
};

use super::exec::{parse_projected_lines_to_rows, resolve_order_expr};
use super::format::{self, cmp_sql_val, SqlFormat, SqlTable, SqlVal};
use super::{OrderItem, OrderKey, OwnedCol, SelectExpr, SelectItem, SqlQuery};

pub(super) fn exec_star(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
    if !query.group_by.is_empty() {
        return Err(Error::InvalidOption(
            "SELECT * cannot be combined with GROUP BY",
        ));
    }
    let lim = query.limit.unwrap_or(usize::MAX);
    if let Some(hdr) = peek_delim_header(frame_bytes, dec_options)? {
        let headers = hdr.columns.clone();
        if lim == 0 {
            return Ok(SqlTable {
                headers,
                rows: Vec::new(),
            });
        }
        let fetch_lim = if query.order_by.len() >= 2 {
            usize::MAX
        } else {
            lim.saturating_add(1)
        };
        let mut matches = if query.order_by.len() == 1 {
            let oi = &query.order_by[0];
            let ocol = match &oi.key {
                OrderKey::Ordinal(ord) => {
                    if *ord == 0 || *ord > headers.len() {
                        return Err(Error::InvalidOption("ORDER BY ordinal out of range"));
                    }
                    ColTarget::Delimited {
                        delim: hdr.delim,
                        index: *ord - 1,
                    }
                }
                OrderKey::Expr(SelectExpr::Col(c)) => c.as_target(),
                _ => {
                    return Err(Error::InvalidOption(
                        "cannot ORDER BY aggregate without GROUP BY",
                    ))
                }
            };
            crate::top_k_by(frame_bytes, where_expr, ocol, oi.dir, fetch_lim, dec_options)?
        } else {
            let expr = where_expr.unwrap_or(&PredExpr::True);
            crate::filter_where(frame_bytes, expr, fetch_lim, dec_options)?
        };
        matches.retain(|m| m.global_row_index != 0);
        let mut rows: Vec<Vec<SqlVal>> = matches
            .iter()
            .map(|m| {
                (0..headers.len())
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
        if query.order_by.len() >= 2 {
            sort_star_rows(&headers, &query.order_by, &mut rows)?;
        }
        rows.truncate(lim);
        return Ok(SqlTable { headers, rows });
    }

    let headers = vec!["record".to_owned()];
    if lim == 0 {
        return Ok(SqlTable {
            headers,
            rows: Vec::new(),
        });
    }
    let matches = if query.order_by.len() == 1 {
        let oi = &query.order_by[0];
        let OrderKey::Expr(SelectExpr::Col(ref c)) = oi.key else {
            return Err(Error::InvalidOption("invalid ORDER BY on SELECT *"));
        };
        crate::top_k_by(
            frame_bytes,
            where_expr,
            c.as_target(),
            oi.dir,
            lim,
            dec_options,
        )?
    } else {
        let expr = where_expr.unwrap_or(&PredExpr::True);
        let fetch_lim = if query.order_by.is_empty() {
            lim
        } else {
            usize::MAX
        };
        let mut ms = crate::filter_where(frame_bytes, expr, fetch_lim, dec_options)?;
        if !query.order_by.is_empty() {
            sort_raw_matches_multi(&query.order_by, &mut ms)?;
            ms.truncate(lim);
        }
        ms
    };
    let rows = matches
        .into_iter()
        .map(|m| {
            let s = String::from_utf8_lossy(&m.bytes)
                .trim_end_matches(['\r', '\n'])
                .to_owned();
            vec![SqlVal::Str(s)]
        })
        .collect();
    Ok(SqlTable { headers, rows })
}

fn sort_star_rows(
    headers: &[String],
    order_by: &[OrderItem],
    rows: &mut [Vec<SqlVal>],
) -> Result<()> {
    let mut cols = Vec::with_capacity(order_by.len());
    for oi in order_by {
        let idx = match &oi.key {
            OrderKey::Ordinal(ord) if *ord >= 1 && *ord <= headers.len() => *ord - 1,
            OrderKey::Expr(SelectExpr::Col(OwnedCol::JsonKey(name))) => headers
                .iter()
                .position(|h| h == name || h.eq_ignore_ascii_case(name))
                .ok_or(Error::InvalidOption("unknown column in ORDER BY"))?,
            _ => return Err(Error::InvalidOption("invalid column in ORDER BY")),
        };
        cols.push((idx, oi.dir));
    }
    rows.sort_by(|a, b| {
        for &(idx, dir) in &cols {
            let ord = cmp_sql_val(&a[idx], &b[idx], dir);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
    Ok(())
}

fn sort_raw_matches_multi(order_by: &[OrderItem], ms: &mut [crate::RowMatch]) -> Result<()> {
    let mut cols = Vec::with_capacity(order_by.len());
    for oi in order_by {
        let OrderKey::Expr(SelectExpr::Col(ref c)) = oi.key else {
            return Err(Error::InvalidOption("invalid ORDER BY on SELECT *"));
        };
        cols.push((c.clone(), oi.dir));
    }
    ms.sort_by(|a, b| {
        for (c, dir) in &cols {
            let va = format::extract_line_sql_val(&a.bytes, c.as_target());
            let vb = format::extract_line_sql_val(&b.bytes, c.as_target());
            let ord = cmp_sql_val(&va, &vb, *dir);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.global_row_index.cmp(&b.global_row_index)
    });
    Ok(())
}

pub(super) fn exec_projection(
    frame_bytes: &[u8],
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
    let headers: Vec<String> = items.iter().map(SelectItem::header_name).collect();
    let lim = query.limit.unwrap_or(usize::MAX);
    if lim == 0 {
        return Ok(SqlTable {
            headers,
            rows: Vec::new(),
        });
    }
    let mut proj_cols: Vec<OwnedCol> = Vec::with_capacity(items.len());
    for it in items {
        let SelectExpr::Col(ref c) = it.expr else {
            return Err(Error::InvalidOption("unexpected aggregate in projection"));
        };
        proj_cols.push(c.clone());
    }

    let mut fetch_cols = proj_cols.clone();
    let mut order_col_indices: Vec<(usize, SortDir)> = Vec::with_capacity(query.order_by.len());
    for oi in &query.order_by {
        let (idx_opt, ex_opt) = resolve_order_expr(items, &oi.key)?;
        if let Some(idx) = idx_opt {
            order_col_indices.push((idx, oi.dir));
        } else if let Some(SelectExpr::Col(c)) = ex_opt {
            let pos = if let Some(p) = fetch_cols.iter().position(|fc| fc == c) {
                p
            } else {
                fetch_cols.push(c.clone());
                fetch_cols.len() - 1
            };
            order_col_indices.push((pos, oi.dir));
        } else {
            return Err(Error::InvalidOption(
                "cannot ORDER BY aggregate without GROUP BY",
            ));
        }
    }

    let fetch_targets: Vec<ColTarget<'_>> = fetch_cols.iter().map(OwnedCol::as_target).collect();
    let hdr_opt = resolve_header_for_targets(frame_bytes, where_expr, &fetch_targets, dec_options)?;
    let delim_opt = if let Some(ref hdr) = hdr_opt {
        Some(hdr.delim)
    } else if let ColTarget::Delimited { delim, .. } = fetch_targets[0] {
        Some(delim)
    } else {
        None
    };

    if query.order_by.is_empty() {
        let proj_targets: Vec<ColTarget<'_>> = proj_cols.iter().map(OwnedCol::as_target).collect();
        let bytes = crate::select_columns(frame_bytes, where_expr, &proj_targets, lim, dec_options)?;
        let rows = parse_projected_lines_to_rows(&bytes, &proj_cols, delim_opt);
        return Ok(SqlTable { headers, rows });
    }
    if query.order_by.len() == 1 {
        let proj_targets: Vec<ColTarget<'_>> = proj_cols.iter().map(OwnedCol::as_target).collect();
        let (o_idx, dir) = order_col_indices[0];
        let ocol = fetch_cols[o_idx].as_target();
        let bytes = crate::select_top_k_by(
            frame_bytes,
            where_expr,
            &proj_targets,
            ocol,
            dir,
            lim,
            dec_options,
        )?;
        let rows = parse_projected_lines_to_rows(&bytes, &proj_cols, delim_opt);
        return Ok(SqlTable { headers, rows });
    }

    let bytes = crate::select_columns(
        frame_bytes,
        where_expr,
        &fetch_targets,
        usize::MAX,
        dec_options,
    )?;
    let mut full_rows = parse_projected_lines_to_rows(&bytes, &fetch_cols, delim_opt);
    full_rows.sort_by(|a, b| {
        for &(idx, dir) in &order_col_indices {
            let ord = cmp_sql_val(&a[idx], &b[idx], dir);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
    full_rows.truncate(lim);
    for r in &mut full_rows {
        r.truncate(proj_cols.len());
    }
    Ok(SqlTable {
        headers,
        rows: full_rows,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stream_projection<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    items: &[SelectItem],
    where_expr: Option<&PredExpr<'_>>,
    query: &SqlQuery,
    format: SqlFormat,
    no_header: bool,
    dec_options: &DecodeOptions,
) -> Result<()> {
    let headers: Vec<String> = items.iter().map(SelectItem::header_name).collect();
    let lim = query.limit.unwrap_or(usize::MAX);
    if lim == 0 {
        return SqlTable {
            headers,
            rows: Vec::new(),
        }
        .write_formatted(writer, format, no_header);
    }
    let mut proj_cols = Vec::with_capacity(items.len());
    for it in items {
        let SelectExpr::Col(ref c) = it.expr else {
            return Err(Error::InvalidOption("unexpected aggregate in projection"));
        };
        proj_cols.push(c.clone());
    }
    let mut fetch_cols = proj_cols.clone();
    let mut order_indices: Vec<(usize, SortDir)> = Vec::with_capacity(query.order_by.len());
    for oi in &query.order_by {
        let (idx_opt, ex_opt) = resolve_order_expr(items, &oi.key)?;
        if let Some(idx) = idx_opt {
            order_indices.push((idx, oi.dir));
        } else if let Some(SelectExpr::Col(c)) = ex_opt {
            let pos = fetch_cols.iter().position(|fc| fc == c).unwrap_or_else(|| {
                fetch_cols.push(c.clone());
                fetch_cols.len() - 1
            });
            order_indices.push((pos, oi.dir));
        } else {
            return Err(Error::InvalidOption(
                "cannot ORDER BY aggregate without GROUP BY",
            ));
        }
    }

    let mut raw_proj = Vec::new();
    if query.order_by.is_empty() {
        let targets: Vec<ColTarget<'_>> = proj_cols.iter().map(OwnedCol::as_target).collect();
        crate::select_columns_stream(reader, &mut raw_proj, where_expr, &targets, lim, dec_options)?;
        let delim_opt = detect_proj_delim(&raw_proj, &targets);
        let rows = parse_projected_lines_to_rows(&raw_proj, &proj_cols, delim_opt);
        return SqlTable { headers, rows }.write_formatted(writer, format, no_header);
    }
    if query.order_by.len() == 1 {
        let targets: Vec<ColTarget<'_>> = proj_cols.iter().map(OwnedCol::as_target).collect();
        let (o_idx, dir) = order_indices[0];
        let ocol = fetch_cols[o_idx].as_target();
        crate::select_top_k_by_stream(
            reader,
            &mut raw_proj,
            where_expr,
            &targets,
            ocol,
            dir,
            lim,
            dec_options,
        )?;
        let delim_opt = detect_proj_delim(&raw_proj, &targets);
        let rows = parse_projected_lines_to_rows(&raw_proj, &proj_cols, delim_opt);
        return SqlTable { headers, rows }.write_formatted(writer, format, no_header);
    }

    let fetch_targets: Vec<ColTarget<'_>> = fetch_cols.iter().map(OwnedCol::as_target).collect();
    crate::select_columns_stream(
        reader,
        &mut raw_proj,
        where_expr,
        &fetch_targets,
        usize::MAX,
        dec_options,
    )?;
    let delim_opt = detect_proj_delim(&raw_proj, &fetch_targets);
    let mut full_rows = parse_projected_lines_to_rows(&raw_proj, &fetch_cols, delim_opt);
    full_rows.sort_by(|a, b| {
        for &(idx, dir) in &order_indices {
            let ord = cmp_sql_val(&a[idx], &b[idx], dir);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
    full_rows.truncate(lim);
    for r in &mut full_rows {
        r.truncate(proj_cols.len());
    }
    SqlTable {
        headers,
        rows: full_rows,
    }
    .write_formatted(writer, format, no_header)
}

fn detect_proj_delim(proj_bytes: &[u8], targets: &[ColTarget<'_>]) -> Option<u8> {
    if let Some(&ColTarget::Delimited { delim, .. }) = targets.first() {
        return Some(delim);
    }
    let first_non_ws = proj_bytes
        .iter()
        .copied()
        .find(|&b| !matches!(b, b' ' | b'\t'));
    if first_non_ws == Some(b'{') || proj_bytes.is_empty() {
        None
    } else if proj_bytes.contains(&b'\t') && !proj_bytes.contains(&b',') {
        Some(b'\t')
    } else {
        Some(b',')
    }
}
