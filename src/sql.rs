//! Compressed-domain SQL front-end (`SELECT ... [FROM ...] [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT ...]`)
//! compiling SQL queries directly onto VXG's SIMD/NEON compressed-domain operators.

use crate::{
    ops::{parse_col_target, ColPred, ColTarget, DecCmp, PredExpr, SortDir, StrCmp, ValuePred},
    Result,
};

pub(crate) mod exec;
pub(crate) mod format;
mod parser;
pub(crate) mod project;
pub(crate) mod stream;

pub use exec::{execute_sql, execute_sql_table};
pub use format::{SqlFormat, SqlTable, SqlVal};
pub use stream::execute_sql_stream;

/// SQL aggregate function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl AggFunc {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Avg => "avg",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

/// Owned column target resolved from SQL identifiers (`payment_type`, `actor.login`, `csv:0`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwnedCol {
    JsonKey(String),
    Delimited { delim: u8, index: usize },
}

impl OwnedCol {
    pub fn parse(raw: &str) -> Result<Self> {
        match parse_col_target(raw.trim())? {
            ColTarget::JsonKey(k) => Ok(Self::JsonKey(k.to_owned())),
            ColTarget::Delimited { delim, index } => Ok(Self::Delimited { delim, index }),
        }
    }

    #[inline]
    pub fn as_target(&self) -> ColTarget<'_> {
        match self {
            Self::JsonKey(s) => ColTarget::JsonKey(s.as_str()),
            Self::Delimited { delim, index } => ColTarget::Delimited {
                delim: *delim,
                index: *index,
            },
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::JsonKey(s) => s.clone(),
            Self::Delimited { delim: b',', index } => format!("csv:{index}"),
            Self::Delimited {
                delim: b'\t',
                index,
            } => format!("tsv:{index}"),
            Self::Delimited { delim: b'|', index } => format!("psv:{index}"),
            Self::Delimited { index, .. } => format!("col:{index}"),
        }
    }
}

/// Expression inside a SQL `SELECT` item or `ORDER BY` clause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectExpr {
    Col(OwnedCol),
    CountStar,
    Agg { func: AggFunc, col: OwnedCol },
}

impl SelectExpr {
    pub fn default_header(&self) -> String {
        match self {
            Self::Col(c) => c.display_name(),
            Self::CountStar => "count(*)".to_owned(),
            Self::Agg { func, col } => format!("{}({})", func.name(), col.display_name()),
        }
    }

    #[inline]
    pub fn is_agg(&self) -> bool {
        matches!(self, Self::CountStar | Self::Agg { .. })
    }
}

/// Projected `SELECT` item with optional `AS` alias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectItem {
    pub expr: SelectExpr,
    pub alias: Option<String>,
}

impl SelectItem {
    pub fn header_name(&self) -> String {
        self.alias
            .clone()
            .unwrap_or_else(|| self.expr.default_header())
    }
}

/// Top-level `SELECT` list (`*` or explicit items).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectProjection {
    Star,
    Items(Vec<SelectItem>),
}

/// `GROUP BY` item (column target or 1-based `SELECT` ordinal).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupItem {
    Ordinal(usize),
    Col(OwnedCol),
}

/// `ORDER BY` key (1-based `SELECT` ordinal, aggregate expression, or column/alias name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderKey {
    Ordinal(usize),
    Expr(SelectExpr),
}

/// `ORDER BY` specification item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderItem {
    pub key: OrderKey,
    pub dir: SortDir,
}

#[derive(Clone, Debug)]
pub enum SqlLeafPred {
    StrEq(Vec<u8>),
    StrNe(Vec<u8>),
    StrPrefix(Vec<u8>),
    StrSuffix(Vec<u8>),
    StrContains(Vec<u8>),
    Dec(DecCmp),
}

impl SqlLeafPred {
    #[inline]
    fn as_value_pred(&self) -> ValuePred<'_> {
        match self {
            Self::StrEq(b) => ValuePred::Str(StrCmp::Eq(b)),
            Self::StrNe(b) => ValuePred::Str(StrCmp::Ne(b)),
            Self::StrPrefix(b) => ValuePred::Str(StrCmp::Prefix(b)),
            Self::StrSuffix(b) => ValuePred::Str(StrCmp::Suffix(b)),
            Self::StrContains(b) => ValuePred::Str(StrCmp::Contains(b)),
            Self::Dec(d) => ValuePred::Dec(*d),
        }
    }
}

/// Owned boolean `WHERE` AST lowered zero-copy to [`PredExpr`].
#[derive(Clone, Debug)]
pub enum SqlWhere {
    True,
    Col { col: OwnedCol, pred: SqlLeafPred },
    RowContains(Vec<u8>),
    Not(Box<SqlWhere>),
    All(Vec<SqlWhere>),
    Any(Vec<SqlWhere>),
}

impl SqlWhere {
    pub fn with_pred_expr<R, F: FnOnce(&PredExpr<'_>) -> R>(&self, f: F) -> R {
        fn count_nodes(n: &SqlWhere) -> (usize, usize) {
            match n {
                SqlWhere::True | SqlWhere::Col { .. } | SqlWhere::RowContains(_) => (0, 0),
                SqlWhere::Not(i) => {
                    let (p, s) = count_nodes(i);
                    (p + 1, s)
                }
                SqlWhere::All(v) | SqlWhere::Any(v) => {
                    let (mut p, mut s) = (v.len(), 1);
                    for c in v {
                        let (cp, cs) = count_nodes(c);
                        p += cp;
                        s += cs;
                    }
                    (p, s)
                }
            }
        }
        let (np, ns) = count_nodes(self);
        let mut pa: Vec<PredExpr<'_>> = Vec::with_capacity(np);
        let mut sa: Vec<&[PredExpr<'_>]> = Vec::with_capacity(ns);
        fn build<'b, 's>(
            node: &'b SqlWhere,
            pa: &mut Vec<PredExpr<'s>>,
            sa: &mut Vec<&'s [PredExpr<'s>]>,
        ) -> PredExpr<'s>
        where
            'b: 's,
        {
            match node {
                SqlWhere::True => PredExpr::True,
                SqlWhere::RowContains(b) => PredExpr::RowContains(b),
                SqlWhere::Col { col, pred } => PredExpr::Col(ColPred {
                    col: col.as_target(),
                    pred: pred.as_value_pred(),
                }),
                SqlWhere::Not(inner) => {
                    let ch = build(inner, pa, sa);
                    let idx = pa.len();
                    pa.push(ch);
                    let r: &'s PredExpr<'s> = unsafe { &*(pa.as_ptr().add(idx)) };
                    PredExpr::Not(r)
                }
                SqlWhere::All(ch) | SqlWhere::Any(ch) => {
                    let mut tmp = Vec::with_capacity(ch.len());
                    for c in ch {
                        tmp.push(build(c, pa, sa));
                    }
                    let base = pa.len();
                    pa.extend(tmp);
                    let sl: &'s [PredExpr<'s>] =
                        unsafe { std::slice::from_raw_parts(pa.as_ptr().add(base), ch.len()) };
                    sa.push(sl);
                    if matches!(node, SqlWhere::All(_)) {
                        PredExpr::All(sl)
                    } else {
                        PredExpr::Any(sl)
                    }
                }
            }
        }
        let root = build(self, &mut pa, &mut sa);
        f(&root)
    }
}

/// Parsed compressed-domain SQL query.
#[derive(Clone, Debug)]
pub struct SqlQuery {
    pub projection: SelectProjection,
    pub from_table: Option<String>,
    pub where_clause: Option<SqlWhere>,
    pub group_by: Vec<GroupItem>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<usize>,
}

impl SqlQuery {
    /// Parse a SQL `SELECT` statement.
    pub fn parse(sql: &str) -> Result<Self> {
        parser::SqlParser::new(sql).parse_query()
    }
}
