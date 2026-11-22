//! Compressed-domain `HAVING`, `COUNT(DISTINCT col)`, `SELECT DISTINCT`, and `OFFSET` helpers.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::{
    ops::{
        multi_agg::{multi_group_by, MultiGroupMap, MultiGroupRow},
        ColTarget, DecNum, GroupKey, PredExpr, SortDir,
    },
    DecodeOptions, Error, Result,
};

use super::exec::{resolve_group_cols, resolve_order_expr};
use super::format::{cmp_sql_val, SqlTable, SqlVal};
use super::parser::SqlParser;
use super::{OwnedCol, SelectExpr, SelectItem, SelectProjection, SqlLeafPred, SqlQuery, SqlWhere};

/// Comparison operator inside a SQL `HAVING` predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HavingCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Boolean AST for SQL `HAVING` clauses evaluated over grouped or scalar aggregate outputs.
#[derive(Clone, Debug, PartialEq)]
pub enum SqlHaving {
    Cmp {
        expr: SelectExpr,
        op: HavingCmp,
        rhs: SqlVal,
    },
    Not(Box<SqlHaving>),
    All(Vec<SqlHaving>),
    Any(Vec<SqlHaving>),
}

/// Apply SQL `OFFSET` and `LIMIT` deterministically to any materialized row slice.
pub(crate) fn apply_offset_limit<T>(rows: &mut Vec<T>, offset: usize, limit: Option<usize>) {
    if offset > 0 {
        if offset >= rows.len() {
            rows.clear();
        } else {
            rows.drain(..offset);
        }
    }
    if let Some(lim) = limit {
        rows.truncate(lim);
    }
}

/// Effective fetch count (`offset + limit`, saturated at `usize::MAX`) for early-exit scans.
#[inline]
pub(crate) fn fetch_window(offset: usize, limit: Option<usize>) -> usize {
    match limit {
        None => usize::MAX,
        Some(0) => 0,
        Some(lim) => offset.saturating_add(lim),
    }
}

/// Resolve an expression that may reference a `SELECT` alias inside `HAVING`.
pub(crate) fn resolve_having_expr<'a>(
    items: &'a [SelectItem],
    expr: &'a SelectExpr,
) -> &'a SelectExpr {
    if let SelectExpr::Col(OwnedCol::JsonKey(name)) = expr {
        if let Some(it) = items
            .iter()
            .find(|it| it.alias.as_deref() == Some(name.as_str()))
        {
            return &it.expr;
        }
    }
    expr
}

/// Register any aggregate or `COUNT(DISTINCT col)` columns referenced by `having`.
pub(crate) fn collect_having_cols(
    having: &SqlHaving,
    items: &[SelectItem],
    agg_cols: &mut Vec<OwnedCol>,
    distinct_cols: &mut Vec<OwnedCol>,
) {
    match having {
        SqlHaving::Cmp { expr, .. } => match resolve_having_expr(items, expr) {
            SelectExpr::Agg { col, .. } if !agg_cols.contains(col) => {
                agg_cols.push(col.clone());
            }
            SelectExpr::CountDistinct(col) if !distinct_cols.contains(col) => {
                distinct_cols.push(col.clone());
            }
            _ => {}
        },
        SqlHaving::Not(inner) => collect_having_cols(inner, items, agg_cols, distinct_cols),
        SqlHaving::All(list) | SqlHaving::Any(list) => {
            for h in list {
                collect_having_cols(h, items, agg_cols, distinct_cols);
            }
        }
    }
}

fn cmp_having_vals(lhs: &SqlVal, rhs: &SqlVal) -> Option<Ordering> {
    if matches!(lhs, SqlVal::Null) || matches!(rhs, SqlVal::Null) {
        return None;
    }
    Some(cmp_sql_val(lhs, rhs, SortDir::Asc))
}

/// Evaluate a parsed `HAVING` AST against an aggregate row evaluator.
pub(crate) fn eval_having<F>(having: &SqlHaving, items: &[SelectItem], eval: &F) -> Result<bool>
where
    F: Fn(&SelectExpr) -> Result<SqlVal>,
{
    match having {
        SqlHaving::Cmp { expr, op, rhs } => {
            let resolved = resolve_having_expr(items, expr);
            let lhs = eval(resolved)?;
            let Some(ord) = cmp_having_vals(&lhs, rhs) else {
                return Ok(false);
            };
            Ok(match op {
                HavingCmp::Eq => ord == Ordering::Equal,
                HavingCmp::Ne => ord != Ordering::Equal,
                HavingCmp::Lt => ord == Ordering::Less,
                HavingCmp::Le => ord != Ordering::Greater,
                HavingCmp::Gt => ord == Ordering::Greater,
                HavingCmp::Ge => ord != Ordering::Less,
            })
        }
        SqlHaving::Not(inner) => Ok(!eval_having(inner, items, eval)?),
        SqlHaving::All(list) => {
            for h in list {
                if !eval_having(h, items, eval)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        SqlHaving::Any(list) => {
            for h in list {
                if eval_having(h, items, eval)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

/// Parse a literal token inside `HAVING` into a typed [`SqlVal`].
pub(crate) fn literal_to_sql_val(quoted: bool, raw: &str) -> SqlVal {
    if !quoted {
        if raw.eq_ignore_ascii_case("null") {
            return SqlVal::Null;
        }
        if raw.eq_ignore_ascii_case("true") {
            return SqlVal::Bool(true);
        }
        if raw.eq_ignore_ascii_case("false") {
            return SqlVal::Bool(false);
        }
        if let Ok(u) = raw.parse::<u64>() {
            return SqlVal::Int(u);
        }
        if let Some(d) = DecNum::parse(raw.as_bytes()) {
            return SqlVal::Dec(d);
        }
        if let Ok(f) = raw.parse::<f64>() {
            return SqlVal::Float(f);
        }
    }
    SqlVal::Str(raw.to_owned())
}

/// Compressed-domain scalar `COUNT(DISTINCT col)` via single-column `group_by`.
pub(crate) fn count_distinct_scalar(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    col: &OwnedCol,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    let groups = crate::group_by(frame_bytes, where_expr, col.as_target(), None, dec_options)?;
    Ok(groups.len() as u64)
}

/// Compressed-domain grouped `COUNT(DISTINCT dcol)` per `group_cols` tuple.
pub(crate) fn count_distinct_grouped(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_cols: &[OwnedCol],
    distinct_cols: &[OwnedCol],
    dec_options: &DecodeOptions,
) -> Result<Vec<HashMap<Vec<GroupKey>, u64>>> {
    let g_len = group_cols.len();
    let mut out = Vec::with_capacity(distinct_cols.len());
    for dcol in distinct_cols {
        let mut ext_cols: Vec<ColTarget<'_>> =
            group_cols.iter().map(OwnedCol::as_target).collect();
        ext_cols.push(dcol.as_target());
        let rows: Vec<MultiGroupRow> =
            multi_group_by(frame_bytes, where_expr, &ext_cols, &[], dec_options)?;
        let mut counts: HashMap<Vec<GroupKey>, u64> = HashMap::new();
        for r in rows {
            if r.keys.len() > g_len {
                let gkey = r.keys[..g_len].to_vec();
                *counts.entry(gkey).or_insert(0) += 1;
            }
        }
        out.push(counts);
    }
    Ok(out)
}

/// Resolve `(group_cols, agg_cols, distinct_cols)` for a `GROUP BY` or `SELECT DISTINCT` query.
pub(crate) fn collect_group_query_cols(
    items: &[SelectItem],
    query: &SqlQuery,
) -> Result<(Vec<OwnedCol>, Vec<OwnedCol>, Vec<OwnedCol>)> {
    let group_cols = if query.group_by.is_empty() && query.distinct {
        let mut g = Vec::with_capacity(items.len());
        for it in items {
            if let SelectExpr::Col(c) = &it.expr {
                if !g.contains(c) {
                    g.push(c.clone());
                }
            }
        }
        g
    } else {
        resolve_group_cols(items, &query.group_by)?
    };
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
    let mut distinct_cols: Vec<OwnedCol> = Vec::new();
    for it in items {
        match &it.expr {
            SelectExpr::Agg { col, .. } if !agg_cols.contains(col) => agg_cols.push(col.clone()),
            SelectExpr::CountDistinct(col) if !distinct_cols.contains(col) => {
                distinct_cols.push(col.clone())
            }
            _ => {}
        }
    }
    if let Some(ref h) = query.having {
        collect_having_cols(h, items, &mut agg_cols, &mut distinct_cols);
    }
    for oi in &query.order_by {
        let (_, ex_opt) = resolve_order_expr(items, &oi.key)?;
        match ex_opt {
            Some(SelectExpr::Agg { col, .. }) if !agg_cols.contains(col) => {
                agg_cols.push(col.clone())
            }
            Some(SelectExpr::CountDistinct(col)) if !distinct_cols.contains(col) => {
                distinct_cols.push(col.clone())
            }
            _ => {}
        }
    }
    Ok((group_cols, agg_cols, distinct_cols))
}

/// Convert streamed `[group_cols..., distinct_col]` maps into per-group distinct counts.
pub(crate) fn maps_to_distinct_counts(
    d_maps: Vec<MultiGroupMap>,
    g_len: usize,
) -> Vec<HashMap<Vec<GroupKey>, u64>> {
    d_maps
        .into_iter()
        .map(|m| {
            let mut counts: HashMap<Vec<GroupKey>, u64> = HashMap::new();
            for r in m.into_rows() {
                if r.keys.len() > g_len {
                    *counts.entry(r.keys[..g_len].to_vec()).or_insert(0) += 1;
                }
            }
            counts
        })
        .collect()
}

/// Parse a SQL `HAVING` boolean predicate expression using `SqlParser` primitives.
pub(super) fn parse_having_or(p: &mut SqlParser<'_>) -> Result<SqlHaving> {
    let first = parse_having_and(p)?;
    let mut items = vec![first];
    while p.eat_kw("OR") || p.eat_sym("||") {
        items.push(parse_having_and(p)?);
    }
    Ok(if items.len() == 1 {
        items.pop().unwrap()
    } else {
        SqlHaving::Any(items)
    })
}

fn parse_having_and(p: &mut SqlParser<'_>) -> Result<SqlHaving> {
    let first = parse_having_unary(p)?;
    let mut items = vec![first];
    while p.eat_kw("AND") || p.eat_sym("&&") {
        items.push(parse_having_unary(p)?);
    }
    Ok(if items.len() == 1 {
        items.pop().unwrap()
    } else {
        SqlHaving::All(items)
    })
}

fn parse_having_unary(p: &mut SqlParser<'_>) -> Result<SqlHaving> {
    if p.eat_kw("NOT") {
        return Ok(SqlHaving::Not(Box::new(parse_having_unary(p)?)));
    }
    if p.eat_char(b'(') {
        let inner = parse_having_or(p)?;
        if !p.eat_char(b')') {
            return Err(crate::Error::InvalidOption("unclosed '(' in SQL HAVING"));
        }
        return Ok(inner);
    }
    let expr = p.parse_select_expr()?;
    let op = if p.eat_sym(">=") {
        HavingCmp::Ge
    } else if p.eat_sym("<=") {
        HavingCmp::Le
    } else if p.eat_sym("!=") || p.eat_sym("<>") {
        HavingCmp::Ne
    } else if p.eat_sym("==") || p.eat_sym("=") {
        HavingCmp::Eq
    } else if p.eat_sym(">") {
        HavingCmp::Gt
    } else if p.eat_sym("<") {
        HavingCmp::Lt
    } else {
        return Err(crate::Error::InvalidOption(
            "expected comparison operator in SQL HAVING",
        ));
    };
    let (quoted, raw) = p.parse_literal_val()?;
    Ok(SqlHaving::Cmp {
        expr,
        op,
        rhs: literal_to_sql_val(quoted, &raw),
    })
}

/// Build `col IS [NOT] NULL` matching both empty CSV cells and JSON `null` literals.
pub(super) fn parse_is_null_after_is(p: &mut SqlParser<'_>, col: OwnedCol) -> Result<SqlWhere> {
    let neg = p.eat_kw("NOT");
    if !p.eat_kw("NULL") {
        return Err(Error::InvalidOption("expected NULL after IS [NOT] in SQL WHERE"));
    }
    let is_null = SqlWhere::Any(vec![
        SqlWhere::Col {
            col: col.clone(),
            pred: SqlLeafPred::StrEq(Vec::new()),
        },
        SqlWhere::Col {
            col,
            pred: SqlLeafPred::StrEq(b"null".to_vec()),
        },
    ]);
    Ok(if neg {
        SqlWhere::Not(Box::new(is_null))
    } else {
        is_null
    })
}

/// Produce a structured `EXPLAIN` execution plan table for `query` against `frame_bytes`.
pub(crate) fn build_explain_table(
    frame_bytes: Option<&[u8]>,
    query: &SqlQuery,
    dec_options: &DecodeOptions,
) -> Result<SqlTable> {
    let mut rows: Vec<Vec<SqlVal>> = Vec::new();
    let mut push = |stage: &str, detail: String| {
        rows.push(vec![SqlVal::Str(stage.to_owned()), SqlVal::Str(detail)]);
    };
    let (kind, proj_str) = match &query.projection {
        SelectProjection::Star => ("ScanStar", "*".to_owned()),
        SelectProjection::Items(items) => {
            let has_agg = items.iter().any(|it| it.expr.is_agg());
            let k = if !query.group_by.is_empty() {
                "CompressedGroupBy"
            } else if query.distinct && !has_agg {
                "CompressedDistinct"
            } else if has_agg {
                "CompressedScalarAgg"
            } else {
                "CompressedProject"
            };
            let p = items
                .iter()
                .map(SelectItem::header_name)
                .collect::<Vec<_>>()
                .join(", ");
            (k, p)
        }
    };
    push("operator", kind.to_owned());
    push("projection", proj_str);
    push("distinct", query.distinct.to_string());
    push(
        "where",
        if query.where_clause.is_some() {
            "compressed_domain_pushdown".to_owned()
        } else {
            "none".to_owned()
        },
    );
    if !query.group_by.is_empty() {
        push("group_by_cols", query.group_by.len().to_string());
    }
    if query.having.is_some() {
        push("having", "post_agg_filter".to_owned());
    }
    if !query.order_by.is_empty() {
        push("order_by_keys", query.order_by.len().to_string());
    }
    if query.limit.is_some() || query.offset > 0 {
        push(
            "slice",
            format!("limit={:?} offset={}", query.limit, query.offset),
        );
    }
    if let Some(bytes) = frame_bytes {
        if let Ok(h) = crate::frame::v5::parse_v1_to_v5_header(bytes, dec_options.max_output) {
            let ratio = if h.raw_size > 0 {
                bytes.len() as f64 / h.raw_size as f64
            } else {
                0.0
            };
            push(
                "archive",
                format!(
                    "v{} blocks={} raw={}B compressed={}B ratio={:.4}",
                    h.version,
                    h.count,
                    h.raw_size,
                    bytes.len(),
                    ratio
                ),
            );
            let zero_block = matches!(&query.projection, SelectProjection::Items(items) if items.len() == 1 && matches!(items[0].expr, SelectExpr::CountStar))
                && query.where_clause.is_none()
                && query.group_by.is_empty()
                && h.has_row_index;
            if zero_block {
                push("fast_path", "O(1) FLAG_ROW_INDEX header count (0 blocks decoded)".to_owned());
            }
        }
    }
    Ok(SqlTable {
        headers: vec!["stage".to_owned(), "detail".to_owned()],
        rows,
    })
}
