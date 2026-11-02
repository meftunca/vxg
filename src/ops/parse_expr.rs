//! Human-readable column target (`json:key`, `csv:0`, `tsv:1`, `psv:2`) and compound boolean
//! `--where` expression parser shared across Rust callers, CLI subcommands, and C/FFI bindings.

use std::cell::RefCell;

use crate::{query::NumCmp, Error, Result};

use super::expr::PredExpr;
use super::line::{ColPred, ColTarget, StrCmp, ValuePred};

/// Parse a single column target descriptor (`"service"`, `"json:service"`, `"csv:0"`, `"tsv:2"`, `"psv:1"`).
pub fn parse_col_target(s: &str) -> Result<ColTarget<'_>> {
    if let Some(rest) = s.strip_prefix("json:") {
        if rest.is_empty() {
            return Err(Error::InvalidOption("empty json column key"));
        }
        Ok(ColTarget::JsonKey(rest))
    } else if let Some(rest) = s.strip_prefix("csv:") {
        let idx = rest
            .parse::<usize>()
            .map_err(|_| Error::InvalidOption("invalid csv column index"))?;
        Ok(ColTarget::Delimited {
            delim: b',',
            index: idx,
        })
    } else if let Some(rest) = s.strip_prefix("tsv:") {
        let idx = rest
            .parse::<usize>()
            .map_err(|_| Error::InvalidOption("invalid tsv column index"))?;
        Ok(ColTarget::Delimited {
            delim: b'\t',
            index: idx,
        })
    } else if let Some(rest) = s.strip_prefix("psv:") {
        let idx = rest
            .parse::<usize>()
            .map_err(|_| Error::InvalidOption("invalid psv column index"))?;
        Ok(ColTarget::Delimited {
            delim: b'|',
            index: idx,
        })
    } else if !s.is_empty() {
        Ok(ColTarget::JsonKey(s))
    } else {
        Err(Error::InvalidOption("empty column target"))
    }
}

/// Parse a comma-separated list of column targets (`"ts,service,latency_ms"` or `"csv:0,csv:2"`).
pub fn parse_col_list(s: &str) -> Result<Vec<ColTarget<'_>>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            return Err(Error::InvalidOption("empty column in column list"));
        }
        out.push(parse_col_target(p)?);
    }
    if out.is_empty() {
        return Err(Error::InvalidOption("column list must not be empty"));
    }
    Ok(out)
}

#[derive(Clone, Debug)]
enum OwnedTarget {
    JsonKey(String),
    Delimited { delim: u8, index: usize },
}

impl OwnedTarget {
    fn from_col_target(col: ColTarget<'_>) -> Self {
        match col {
            ColTarget::JsonKey(k) => Self::JsonKey(k.to_owned()),
            ColTarget::Delimited { delim, index } => Self::Delimited { delim, index },
        }
    }

    fn as_col_target(&self) -> ColTarget<'_> {
        match self {
            Self::JsonKey(k) => ColTarget::JsonKey(k.as_str()),
            Self::Delimited { delim, index } => ColTarget::Delimited {
                delim: *delim,
                index: *index,
            },
        }
    }
}

#[derive(Clone, Debug)]
enum OwnedPred {
    StrEq(Vec<u8>),
    StrNe(Vec<u8>),
    StrPrefix(Vec<u8>),
    StrSuffix(Vec<u8>),
    StrContains(Vec<u8>),
    Num(NumCmp),
}

impl OwnedPred {
    fn as_value_pred(&self) -> ValuePred<'_> {
        match self {
            Self::StrEq(b) => ValuePred::Str(StrCmp::Eq(b)),
            Self::StrNe(b) => ValuePred::Str(StrCmp::Ne(b)),
            Self::StrPrefix(b) => ValuePred::Str(StrCmp::Prefix(b)),
            Self::StrSuffix(b) => ValuePred::Str(StrCmp::Suffix(b)),
            Self::StrContains(b) => ValuePred::Str(StrCmp::Contains(b)),
            Self::Num(c) => ValuePred::Num(*c),
        }
    }
}

/// Owned single-column predicate parsed from string syntax (e.g. `"status>=500"`, `"level=ERROR"`).
#[derive(Clone, Debug)]
pub struct ParsedColPred {
    target: OwnedTarget,
    pred: OwnedPred,
}

impl ParsedColPred {
    /// Parse a single leaf column predicate specification.
    pub fn parse(spec: &str) -> Result<Self> {
        parse_where_spec_owned(spec)
    }

    /// Borrow as a zero-copy [`ColPred`].
    #[inline]
    pub fn as_col_pred(&self) -> ColPred<'_> {
        ColPred {
            col: self.target.as_col_target(),
            pred: self.pred.as_value_pred(),
        }
    }
}

#[derive(Clone, Debug)]
enum OwnedExpr {
    True,
    Col(ParsedColPred),
    RowContains(Vec<u8>),
    All(Vec<OwnedExpr>),
    Any(Vec<OwnedExpr>),
    Not(Box<OwnedExpr>),
}

struct PredArena<'s> {
    vecs: RefCell<Vec<Box<[PredExpr<'s>]>>>,
}

impl<'s> PredArena<'s> {
    fn new() -> Self {
        Self {
            vecs: RefCell::new(Vec::new()),
        }
    }

    fn alloc(&'s self, items: Vec<PredExpr<'s>>) -> &'s [PredExpr<'s>] {
        let b: Box<[PredExpr<'s>]> = items.into_boxed_slice();
        let ptr: *const [PredExpr<'s>] = &*b;
        self.vecs.borrow_mut().push(b);
        // SAFETY: `b` is a heap-allocated `Box<[T]>` whose backing slice address does not move
        // when `vecs` reallocates, and lives until `PredArena` is dropped at the end of `'s`.
        unsafe { &*ptr }
    }

    fn build(&'s self, expr: &'s OwnedExpr) -> PredExpr<'s> {
        match expr {
            OwnedExpr::True => PredExpr::True,
            OwnedExpr::Col(cp) => PredExpr::Col(cp.as_col_pred()),
            OwnedExpr::RowContains(b) => PredExpr::RowContains(b.as_slice()),
            OwnedExpr::All(children) => {
                let built = children.iter().map(|c| self.build(c)).collect();
                PredExpr::All(self.alloc(built))
            }
            OwnedExpr::Any(children) => {
                let built = children.iter().map(|c| self.build(c)).collect();
                PredExpr::Any(self.alloc(built))
            }
            OwnedExpr::Not(inner) => {
                let s = self.alloc(vec![self.build(inner)]);
                PredExpr::Not(&s[0])
            }
        }
    }
}

/// Owned compound boolean predicate AST parsed from human-readable syntax
/// (`status>=500 && (service=="auth" || level=ERROR) && !contains:healthz`).
#[derive(Clone, Debug)]
pub struct ParsedWhere {
    root: OwnedExpr,
}

impl ParsedWhere {
    /// Parse a compound boolean predicate string into an owned AST.
    pub fn parse(s: &str) -> Result<Self> {
        let mut p = ExprParser { src: s, pos: 0 };
        let root = p.parse_or()?;
        p.skip_ws();
        if p.pos != s.len() {
            return Err(Error::InvalidOption(
                "unexpected trailing token in where expression",
            ));
        }
        Ok(Self { root })
    }

    /// Evaluate a closure with a borrowed [`PredExpr`] materialized from this AST.
    pub fn with_expr<R>(&self, f: impl FnOnce(&PredExpr<'_>) -> R) -> R {
        let arena = PredArena::new();
        let expr = arena.build(&self.root);
        f(&expr)
    }
}

/// Evaluate `f` with `Some(&PredExpr)` when `opt` is `Some`, or `None` otherwise.
pub fn with_opt_where<R>(
    opt: Option<&ParsedWhere>,
    f: impl FnOnce(Option<&PredExpr<'_>>) -> R,
) -> R {
    match opt {
        Some(pw) => pw.with_expr(|e| f(Some(e))),
        None => f(None),
    }
}

fn unquote_opt(s: &str) -> &[u8] {
    let t = s.trim();
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        &t.as_bytes()[1..t.len() - 1]
    } else {
        s.as_bytes()
    }
}

fn parse_u64_rhs(s: &str) -> Result<u64> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| Error::InvalidOption("expected unsigned 64-bit integer in predicate"))
}

fn parse_canonical_cli_u64(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 20 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    s.parse().ok()
}

fn parse_auto_eq_pred(rhs: &str, is_eq: bool) -> OwnedPred {
    let t = rhs.trim();
    let is_quoted = t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')));
    if !is_quoted {
        if let Some(n) = parse_canonical_cli_u64(t) {
            return OwnedPred::Num(if is_eq { NumCmp::Eq(n) } else { NumCmp::Ne(n) });
        }
    }
    let bytes = unquote_opt(rhs).to_vec();
    if is_eq {
        OwnedPred::StrEq(bytes)
    } else {
        OwnedPred::StrNe(bytes)
    }
}

fn parse_where_spec_owned(spec: &str) -> Result<ParsedColPred> {
    for (op_tok, ctor) in [
        (">=", (|n| NumCmp::Ge(n)) as fn(u64) -> NumCmp),
        ("<=", |n| NumCmp::Le(n)),
    ] {
        if let Some((lhs, rhs)) = spec.split_once(op_tok) {
            return Ok(ParsedColPred {
                target: OwnedTarget::from_col_target(parse_col_target(lhs.trim())?),
                pred: OwnedPred::Num(ctor(parse_u64_rhs(rhs)?)),
            });
        }
    }
    if let Some((lhs, rhs)) = spec.split_once("!=") {
        let rhs_eff = if lhs.ends_with(' ') || lhs.ends_with('\t') {
            rhs.trim_start()
        } else {
            rhs
        };
        return Ok(ParsedColPred {
            target: OwnedTarget::from_col_target(parse_col_target(lhs.trim())?),
            pred: parse_auto_eq_pred(rhs_eff, false),
        });
    }
    if let Some((lhs, rhs)) = spec.split_once("==") {
        let rhs_eff = if lhs.ends_with(' ') || lhs.ends_with('\t') {
            rhs.trim_start()
        } else {
            rhs
        };
        return Ok(ParsedColPred {
            target: OwnedTarget::from_col_target(parse_col_target(lhs.trim())?),
            pred: parse_auto_eq_pred(rhs_eff, true),
        });
    }
    if let Some((lhs, rhs)) = spec.split_once('=') {
        let target = OwnedTarget::from_col_target(parse_col_target(lhs.trim())?);
        let pred = if let Some(v) = rhs.strip_prefix("eq:") {
            OwnedPred::StrEq(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("ne:") {
            OwnedPred::StrNe(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("prefix:") {
            OwnedPred::StrPrefix(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("suffix:") {
            OwnedPred::StrSuffix(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("contains:") {
            OwnedPred::StrContains(v.as_bytes().to_vec())
        } else if let Some(v) = rhs
            .strip_prefix("num:")
            .or_else(|| rhs.strip_prefix("num-eq:"))
        {
            OwnedPred::Num(NumCmp::Eq(parse_u64_rhs(v)?))
        } else if let Some(v) = rhs.strip_prefix("num-ne:") {
            OwnedPred::Num(NumCmp::Ne(parse_u64_rhs(v)?))
        } else if let Some(v) = rhs.strip_prefix("gt:") {
            OwnedPred::Num(NumCmp::Gt(parse_u64_rhs(v)?))
        } else if let Some(v) = rhs.strip_prefix("ge:") {
            OwnedPred::Num(NumCmp::Ge(parse_u64_rhs(v)?))
        } else if let Some(v) = rhs.strip_prefix("lt:") {
            OwnedPred::Num(NumCmp::Lt(parse_u64_rhs(v)?))
        } else if let Some(v) = rhs.strip_prefix("le:") {
            OwnedPred::Num(NumCmp::Le(parse_u64_rhs(v)?))
        } else {
            parse_auto_eq_pred(rhs, true)
        };
        return Ok(ParsedColPred { target, pred });
    }
    for (op_ch, ctor) in [
        ('>', (|n| NumCmp::Gt(n)) as fn(u64) -> NumCmp),
        ('<', |n| NumCmp::Lt(n)),
    ] {
        if let Some((lhs, rhs)) = spec.split_once(op_ch) {
            return Ok(ParsedColPred {
                target: OwnedTarget::from_col_target(parse_col_target(lhs.trim())?),
                pred: OwnedPred::Num(ctor(parse_u64_rhs(rhs)?)),
            });
        }
    }
    Err(Error::InvalidOption(
        "invalid where predicate (examples: level=ERROR, status==500 && bytes>=10000)",
    ))
}

fn parse_leaf_expr(spec: &str) -> Result<OwnedExpr> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidOption("empty predicate in where expression"));
    }
    if trimmed.eq_ignore_ascii_case("true") || trimmed == "*" {
        return Ok(OwnedExpr::True);
    }
    if let Some(rest) = trimmed
        .strip_prefix("contains:")
        .or_else(|| trimmed.strip_prefix("row:"))
    {
        return Ok(OwnedExpr::RowContains(unquote_opt(rest).to_vec()));
    }
    Ok(OwnedExpr::Col(parse_where_spec_owned(trimmed)?))
}

struct ExprParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn skip_ws(&mut self) {
        let b = self.src.as_bytes();
        while self.pos < b.len() && b[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn eat_kw_op(&mut self, sym: &str, kw: &str) -> bool {
        self.skip_ws();
        let rem = &self.src[self.pos..];
        if rem.starts_with(sym) {
            self.pos += sym.len();
            return true;
        }
        if rem.len() > kw.len() && rem[..kw.len()].eq_ignore_ascii_case(kw) {
            let next = rem.as_bytes()[kw.len()];
            if next.is_ascii_whitespace() || next == b'(' || next == b'!' {
                self.pos += kw.len();
                return true;
            }
        }
        false
    }

    fn parse_or(&mut self) -> Result<OwnedExpr> {
        let first = self.parse_and()?;
        let mut items = vec![first];
        while self.eat_kw_op("||", "or") {
            items.push(self.parse_and()?);
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap()
        } else {
            OwnedExpr::Any(items)
        })
    }

    fn parse_and(&mut self) -> Result<OwnedExpr> {
        let first = self.parse_unary()?;
        let mut items = vec![first];
        while self.eat_kw_op("&&", "and") {
            items.push(self.parse_unary()?);
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap()
        } else {
            OwnedExpr::All(items)
        })
    }

    fn parse_unary(&mut self) -> Result<OwnedExpr> {
        self.skip_ws();
        let rem = &self.src[self.pos..];
        if rem.starts_with('!') && !rem.starts_with("!=") {
            self.pos += 1;
            return Ok(OwnedExpr::Not(Box::new(self.parse_unary()?)));
        }
        if rem.len() > 3 && rem[..3].eq_ignore_ascii_case("not") {
            let next = rem.as_bytes()[3];
            if next.is_ascii_whitespace() || next == b'(' {
                self.pos += 3;
                return Ok(OwnedExpr::Not(Box::new(self.parse_unary()?)));
            }
        }
        if rem.starts_with('(') {
            self.pos += 1;
            let inner = self.parse_or()?;
            self.skip_ws();
            if !self.src[self.pos..].starts_with(')') {
                return Err(Error::InvalidOption("unclosed '(' in where expression"));
            }
            self.pos += 1;
            return Ok(inner);
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<OwnedExpr> {
        self.skip_ws();
        let start = self.pos;
        let bytes = self.src.as_bytes();
        let mut quote: Option<u8> = None;
        while self.pos < bytes.len() {
            let c = bytes[self.pos];
            if let Some(q) = quote {
                if c == b'\\' && self.pos + 1 < bytes.len() {
                    self.pos += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
                self.pos += 1;
                continue;
            }
            if c == b'"' || c == b'\'' {
                quote = Some(c);
                self.pos += 1;
                continue;
            }
            if c == b')' {
                break;
            }
            let rem = &self.src[self.pos..];
            if rem.starts_with("&&") || rem.starts_with("||") {
                break;
            }
            if c.is_ascii_whitespace() {
                let mut k = self.pos;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                let after = &self.src[k..];
                if after.starts_with("&&")
                    || after.starts_with("||")
                    || after.starts_with(')')
                    || (after.len() > 3
                        && after[..3].eq_ignore_ascii_case("and")
                        && (after.as_bytes()[3].is_ascii_whitespace()
                            || after.as_bytes()[3] == b'('
                            || after.as_bytes()[3] == b'!'))
                    || (after.len() > 2
                        && after[..2].eq_ignore_ascii_case("or")
                        && (after.as_bytes()[2].is_ascii_whitespace()
                            || after.as_bytes()[2] == b'('
                            || after.as_bytes()[2] == b'!'))
                {
                    break;
                }
            }
            self.pos += 1;
        }
        parse_leaf_expr(&self.src[start..self.pos])
    }
}
