//! Hand-rolled zero-dependency recursive-descent SQL `SELECT` parser.

use crate::{
    ops::{DecCmp, DecNum, SortDir},
    Error, Result,
};

use super::{
    AggFunc, GroupItem, OrderItem, OrderKey, OwnedCol, SelectExpr, SelectItem, SelectProjection,
    SqlLeafPred, SqlQuery, SqlWhere,
};

pub(super) struct SqlParser<'a> {
    src: &'a str,
    pos: usize,
}

const CLAUSE_KWS: &[&str] = &[
    "FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "OFFSET", "ASC", "DESC", "AND", "OR",
];

impl<'a> SqlParser<'a> {
    pub(super) fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn skip_ws(&mut self) {
        let b = self.src.as_bytes();
        while self.pos < b.len() {
            if b[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            } else if b[self.pos..].starts_with(b"--") {
                while self.pos < b.len() && b[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn peek_byte(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.as_bytes().get(self.pos).copied()
    }

    fn eat_char(&mut self, ch: u8) -> bool {
        self.skip_ws();
        if self.src.as_bytes().get(self.pos) == Some(&ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_kw(&mut self, kw: &str) -> bool {
        self.skip_ws();
        let rem = &self.src[self.pos..];
        if rem.len() < kw.len() || !rem[..kw.len()].eq_ignore_ascii_case(kw) {
            return false;
        }
        if rem.len() == kw.len() {
            return true;
        }
        let next = rem.as_bytes()[kw.len()];
        !(next.is_ascii_alphanumeric() || matches!(next, b'_' | b'.' | b':'))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.peek_kw(kw) {
            self.pos += kw.len();
            true
        } else {
            false
        }
    }

    fn read_quoted(&mut self, quote: u8) -> Result<String> {
        self.pos += 1;
        let b = self.src.as_bytes();
        let mut out = Vec::new();
        while self.pos < b.len() {
            let c = b[self.pos];
            self.pos += 1;
            if c == quote {
                if self.pos < b.len() && b[self.pos] == quote {
                    out.push(quote);
                    self.pos += 1;
                    continue;
                }
                return String::from_utf8(out)
                    .map_err(|_| Error::InvalidOption("invalid UTF-8 in SQL quoted literal"));
            }
            if c == b'\\' && self.pos < b.len() {
                let esc = b[self.pos];
                self.pos += 1;
                out.push(match esc {
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    other => other,
                });
                continue;
            }
            out.push(c);
        }
        Err(Error::InvalidOption("unterminated quoted string in SQL"))
    }

    fn read_ident(&mut self) -> Result<String> {
        self.skip_ws();
        let Some(b0) = self.src.as_bytes().get(self.pos).copied() else {
            return Err(Error::InvalidOption("unexpected end of SQL query"));
        };
        if b0 == b'"' || b0 == b'`' {
            return self.read_quoted(b0);
        }
        let start = self.pos;
        let b = self.src.as_bytes();
        while self.pos < b.len() {
            let c = b[self.pos];
            if c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b':' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(Error::InvalidOption("expected identifier in SQL query"));
        }
        Ok(self.src[start..self.pos].to_owned())
    }

    fn parse_select_expr(&mut self) -> Result<SelectExpr> {
        for (kw, func) in [
            ("COUNT", AggFunc::Count),
            ("SUM", AggFunc::Sum),
            ("AVG", AggFunc::Avg),
            ("MIN", AggFunc::Min),
            ("MAX", AggFunc::Max),
        ] {
            let save = self.pos;
            if self.eat_kw(kw) && self.eat_char(b'(') {
                if func == AggFunc::Count && (self.eat_char(b'*') || self.eat_kw("1")) {
                    if !self.eat_char(b')') {
                        return Err(Error::InvalidOption("expected ')' after COUNT(*)"));
                    }
                    return Ok(SelectExpr::CountStar);
                }
                let col_name = self.read_ident()?;
                if !self.eat_char(b')') {
                    return Err(Error::InvalidOption("expected ')' after aggregate column"));
                }
                return Ok(SelectExpr::Agg {
                    func,
                    col: OwnedCol::parse(&col_name)?,
                });
            }
            self.pos = save;
        }
        let col_name = self.read_ident()?;
        Ok(SelectExpr::Col(OwnedCol::parse(&col_name)?))
    }

    fn parse_select_list(&mut self) -> Result<SelectProjection> {
        if self.eat_char(b'*') {
            return Ok(SelectProjection::Star);
        }
        let mut items = Vec::new();
        loop {
            let expr = self.parse_select_expr()?;
            let mut alias = None;
            if self.eat_kw("AS") {
                alias = Some(if matches!(self.peek_byte(), Some(b'\'')) {
                    self.read_quoted(b'\'')?
                } else {
                    self.read_ident()?
                });
            } else {
                let save = self.pos;
                if let Some(b0) = self.peek_byte() {
                    if b0 == b'"' || b0 == b'`' || b0.is_ascii_alphabetic() || b0 == b'_' {
                        let cand = self.read_ident()?;
                        if CLAUSE_KWS.iter().any(|k| cand.eq_ignore_ascii_case(k)) {
                            self.pos = save;
                        } else {
                            alias = Some(cand);
                        }
                    }
                }
            }
            items.push(SelectItem { expr, alias });
            if !self.eat_char(b',') {
                break;
            }
        }
        Ok(SelectProjection::Items(items))
    }

    fn parse_literal_val(&mut self) -> Result<(bool, String)> {
        self.skip_ws();
        let Some(b0) = self.peek_byte() else {
            return Err(Error::InvalidOption("expected literal value in SQL WHERE"));
        };
        if b0 == b'\'' || b0 == b'"' {
            return Ok((true, self.read_quoted(b0)?));
        }
        let start = self.pos;
        let b = self.src.as_bytes();
        if matches!(b0, b'+' | b'-') {
            self.pos += 1;
        }
        while self.pos < b.len() {
            let c = b[self.pos];
            if c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-' | b':') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(Error::InvalidOption("expected value in SQL predicate"));
        }
        Ok((false, self.src[start..self.pos].to_owned()))
    }

    fn parse_dec_val(&mut self) -> Result<DecNum> {
        let (_, raw) = self.parse_literal_val()?;
        DecNum::parse(raw.trim().as_bytes())
            .ok_or(Error::InvalidOption("expected numeric value in SQL comparison"))
    }

    fn make_eq_pred(quoted: bool, val: &str, is_eq: bool) -> SqlLeafPred {
        if !quoted {
            if let Some(d) = DecNum::parse(val.as_bytes()) {
                return SqlLeafPred::Dec(if is_eq { DecCmp::Eq(d) } else { DecCmp::Ne(d) });
            }
        }
        if is_eq {
            SqlLeafPred::StrEq(val.as_bytes().to_vec())
        } else {
            SqlLeafPred::StrNe(val.as_bytes().to_vec())
        }
    }

    fn parse_where_or(&mut self) -> Result<SqlWhere> {
        let first = self.parse_where_and()?;
        let mut items = vec![first];
        while self.eat_kw("OR") || self.eat_sym("||") {
            items.push(self.parse_where_and()?);
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap()
        } else {
            SqlWhere::Any(items)
        })
    }

    fn parse_where_and(&mut self) -> Result<SqlWhere> {
        let first = self.parse_where_unary()?;
        let mut items = vec![first];
        while self.eat_kw("AND") || self.eat_sym("&&") {
            items.push(self.parse_where_unary()?);
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap()
        } else {
            SqlWhere::All(items)
        })
    }

    fn eat_sym(&mut self, sym: &str) -> bool {
        self.skip_ws();
        if self.src[self.pos..].starts_with(sym) {
            self.pos += sym.len();
            true
        } else {
            false
        }
    }

    fn parse_where_unary(&mut self) -> Result<SqlWhere> {
        if self.eat_kw("NOT")
            || (self.peek_byte() == Some(b'!') && !self.src[self.pos..].starts_with("!="))
        {
            if self.src.as_bytes().get(self.pos) == Some(&b'!') {
                self.pos += 1;
            }
            return Ok(SqlWhere::Not(Box::new(self.parse_where_unary()?)));
        }
        if self.eat_char(b'(') {
            let inner = self.parse_where_or()?;
            if !self.eat_char(b')') {
                return Err(Error::InvalidOption("unclosed '(' in SQL WHERE"));
            }
            return Ok(inner);
        }
        self.parse_where_leaf()
    }

    fn parse_where_leaf(&mut self) -> Result<SqlWhere> {
        if self.eat_kw("TRUE") {
            return Ok(SqlWhere::True);
        }
        if self.eat_kw("FALSE") {
            return Ok(SqlWhere::Not(Box::new(SqlWhere::True)));
        }
        if self.eat_kw("CONTAINS") && self.eat_char(b'(') {
            let (_, needle) = self.parse_literal_val()?;
            if !self.eat_char(b')') {
                return Err(Error::InvalidOption("expected ')' after CONTAINS(...)"));
            }
            return Ok(SqlWhere::RowContains(needle.into_bytes()));
        }
        let save = self.pos;
        if let Some(b0) = self.peek_byte() {
            if b0.is_ascii_digit() {
                let (_, lhs_lit) = self.parse_literal_val()?;
                if self.eat_sym("==") || self.eat_sym("=") {
                    let (_, rhs_lit) = self.parse_literal_val()?;
                    return Ok(if lhs_lit == rhs_lit {
                        SqlWhere::True
                    } else {
                        SqlWhere::Not(Box::new(SqlWhere::True))
                    });
                }
                self.pos = save;
            }
        }

        let col = OwnedCol::parse(&self.read_ident()?)?;
        let neg_op = self.eat_kw("NOT");

        if self.eat_kw("BETWEEN") {
            let lo = self.parse_dec_val()?;
            if !self.eat_kw("AND") {
                return Err(Error::InvalidOption("expected AND in BETWEEN predicate"));
            }
            let hi = self.parse_dec_val()?;
            let expr = SqlWhere::All(vec![
                SqlWhere::Col {
                    col: col.clone(),
                    pred: SqlLeafPred::Dec(DecCmp::Ge(lo)),
                },
                SqlWhere::Col {
                    col,
                    pred: SqlLeafPred::Dec(DecCmp::Le(hi)),
                },
            ]);
            return Ok(if neg_op {
                SqlWhere::Not(Box::new(expr))
            } else {
                expr
            });
        }

        if self.eat_kw("IN") {
            if !self.eat_char(b'(') {
                return Err(Error::InvalidOption("expected '(' after IN"));
            }
            let mut alts = Vec::new();
            loop {
                let (q, v) = self.parse_literal_val()?;
                alts.push(SqlWhere::Col {
                    col: col.clone(),
                    pred: Self::make_eq_pred(q, &v, true),
                });
                if !self.eat_char(b',') {
                    break;
                }
            }
            if !self.eat_char(b')') {
                return Err(Error::InvalidOption("expected ')' after IN (...)"));
            }
            let expr = if alts.len() == 1 {
                alts.pop().unwrap()
            } else {
                SqlWhere::Any(alts)
            };
            return Ok(if neg_op {
                SqlWhere::Not(Box::new(expr))
            } else {
                expr
            });
        }

        if self.eat_kw("LIKE") {
            let (_, pat) = self.parse_literal_val()?;
            let pb = pat.as_bytes();
            let pred = match (pb.starts_with(b"%"), pb.ends_with(b"%")) {
                (true, true) if pb.len() >= 2 => {
                    SqlLeafPred::StrContains(pb[1..pb.len() - 1].to_vec())
                }
                (true, false) => SqlLeafPred::StrSuffix(pb[1..].to_vec()),
                (false, true) => SqlLeafPred::StrPrefix(pb[..pb.len() - 1].to_vec()),
                _ => SqlLeafPred::StrEq(pb.to_vec()),
            };
            let expr = SqlWhere::Col { col, pred };
            return Ok(if neg_op {
                SqlWhere::Not(Box::new(expr))
            } else {
                expr
            });
        }

        if neg_op {
            return Err(Error::InvalidOption("expected BETWEEN, IN, or LIKE after NOT"));
        }

        if self.eat_sym(">=") {
            return Ok(SqlWhere::Col {
                col,
                pred: SqlLeafPred::Dec(DecCmp::Ge(self.parse_dec_val()?)),
            });
        }
        if self.eat_sym("<=") {
            return Ok(SqlWhere::Col {
                col,
                pred: SqlLeafPred::Dec(DecCmp::Le(self.parse_dec_val()?)),
            });
        }
        if self.eat_sym("!=") || self.eat_sym("<>") {
            let (q, v) = self.parse_literal_val()?;
            return Ok(SqlWhere::Col {
                col,
                pred: Self::make_eq_pred(q, &v, false),
            });
        }
        if self.eat_sym("==") || self.eat_sym("=") {
            let (q, v) = self.parse_literal_val()?;
            return Ok(SqlWhere::Col {
                col,
                pred: Self::make_eq_pred(q, &v, true),
            });
        }
        if self.eat_sym(">") {
            return Ok(SqlWhere::Col {
                col,
                pred: SqlLeafPred::Dec(DecCmp::Gt(self.parse_dec_val()?)),
            });
        }
        if self.eat_sym("<") {
            return Ok(SqlWhere::Col {
                col,
                pred: SqlLeafPred::Dec(DecCmp::Lt(self.parse_dec_val()?)),
            });
        }
        Err(Error::InvalidOption("expected comparison operator in SQL WHERE"))
    }

    pub(super) fn parse_query(&mut self) -> Result<SqlQuery> {
        if !self.eat_kw("SELECT") {
            return Err(Error::InvalidOption("SQL query must begin with SELECT"));
        }
        let projection = self.parse_select_list()?;

        let from_table = if self.eat_kw("FROM") {
            Some(if matches!(self.peek_byte(), Some(b'\'')) {
                self.read_quoted(b'\'')?
            } else {
                self.read_ident()?
            })
        } else {
            None
        };

        let where_clause = if self.eat_kw("WHERE") {
            Some(self.parse_where_or()?)
        } else {
            None
        };

        let mut group_by = Vec::new();
        if self.eat_kw("GROUP") {
            if !self.eat_kw("BY") {
                return Err(Error::InvalidOption("expected BY after GROUP"));
            }
            loop {
                let save = self.pos;
                if matches!(self.peek_byte(), Some(b) if b.is_ascii_digit()) {
                    let (_, tok) = self.parse_literal_val()?;
                    if let Ok(ord) = tok.parse::<usize>() {
                        if ord == 0 {
                            return Err(Error::InvalidOption("GROUP BY ordinal must be >= 1"));
                        }
                        group_by.push(GroupItem::Ordinal(ord));
                    } else {
                        self.pos = save;
                        group_by.push(GroupItem::Col(OwnedCol::parse(&self.read_ident()?)?));
                    }
                } else {
                    group_by.push(GroupItem::Col(OwnedCol::parse(&self.read_ident()?)?));
                }
                if !self.eat_char(b',') {
                    break;
                }
            }
        }

        let mut order_by = Vec::new();
        if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err(Error::InvalidOption("expected BY after ORDER"));
            }
            loop {
                let save = self.pos;
                let key = if matches!(self.peek_byte(), Some(b) if b.is_ascii_digit()) {
                    let (_, tok) = self.parse_literal_val()?;
                    if let Ok(ord) = tok.parse::<usize>() {
                        if ord == 0 {
                            return Err(Error::InvalidOption("ORDER BY ordinal must be >= 1"));
                        }
                        OrderKey::Ordinal(ord)
                    } else {
                        self.pos = save;
                        OrderKey::Expr(self.parse_select_expr()?)
                    }
                } else {
                    OrderKey::Expr(self.parse_select_expr()?)
                };
                let dir = if self.eat_kw("DESC") {
                    SortDir::Desc
                } else {
                    let _ = self.eat_kw("ASC");
                    SortDir::Asc
                };
                order_by.push(OrderItem { key, dir });
                if !self.eat_char(b',') {
                    break;
                }
            }
        }

        let limit = if self.eat_kw("LIMIT") {
            let (_, tok) = self.parse_literal_val()?;
            Some(
                tok.parse::<usize>()
                    .map_err(|_| Error::InvalidOption("invalid integer in SQL LIMIT"))?,
            )
        } else {
            None
        };

        let _ = self.eat_char(b';');
        self.skip_ws();
        if self.pos != self.src.len() {
            return Err(Error::InvalidOption("unexpected trailing tokens in SQL query"));
        }

        Ok(SqlQuery {
            projection,
            from_table,
            where_clause,
            group_by,
            order_by,
            limit,
        })
    }
}
