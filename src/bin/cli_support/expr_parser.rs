enum OwnedTarget {
    JsonKey(String),
    Delimited { delim: u8, index: usize },
}

impl OwnedTarget {
    fn as_col_target(&self) -> vxg::ColTarget<'_> {
        match self {
            Self::JsonKey(k) => vxg::ColTarget::JsonKey(k.as_str()),
            Self::Delimited { delim, index } => vxg::ColTarget::Delimited {
                delim: *delim,
                index: *index,
            },
        }
    }
}

enum OwnedExpr {
    True,
    Col { target: OwnedTarget, pred: OwnedPred },
    RowContains(Vec<u8>),
    All(Vec<OwnedExpr>),
    Any(Vec<OwnedExpr>),
    Not(Box<OwnedExpr>),
}

struct PredArena<'s> {
    vecs: std::cell::RefCell<Vec<Box<[vxg::PredExpr<'s>]>>>,
}

impl<'s> PredArena<'s> {
    fn new() -> Self {
        Self {
            vecs: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn alloc(&'s self, items: Vec<vxg::PredExpr<'s>>) -> &'s [vxg::PredExpr<'s>] {
        let b: Box<[vxg::PredExpr<'s>]> = items.into_boxed_slice();
        let ptr: *const [vxg::PredExpr<'s>] = &*b;
        self.vecs.borrow_mut().push(b);
        // SAFETY: `b` is a heap-allocated `Box<[T]>` whose backing slice address does not move
        // when `vecs` reallocates, and lives until `PredArena` is dropped at the end of `'s`.
        unsafe { &*ptr }
    }

    fn build(&'s self, expr: &'s OwnedExpr) -> vxg::PredExpr<'s> {
        match expr {
            OwnedExpr::True => vxg::PredExpr::True,
            OwnedExpr::Col { target, pred } => vxg::PredExpr::Col(vxg::ColPred {
                col: target.as_col_target(),
                pred: pred.as_value_pred(),
            }),
            OwnedExpr::RowContains(b) => vxg::PredExpr::RowContains(b.as_slice()),
            OwnedExpr::All(children) => {
                let built = children.iter().map(|c| self.build(c)).collect();
                vxg::PredExpr::All(self.alloc(built))
            }
            OwnedExpr::Any(children) => {
                let built = children.iter().map(|c| self.build(c)).collect();
                vxg::PredExpr::Any(self.alloc(built))
            }
            OwnedExpr::Not(inner) => {
                let s = self.alloc(vec![self.build(inner)]);
                vxg::PredExpr::Not(&s[0])
            }
        }
    }
}

fn parse_col_list(s: &str) -> R<Vec<vxg::ColTarget<'_>>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            return Err("empty column in column list".into());
        }
        out.push(parse_col_target(p)?);
    }
    if out.is_empty() {
        return Err("column list must not be empty".into());
    }
    Ok(out)
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

fn parse_leaf_expr(spec: &str) -> R<OwnedExpr> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err("empty predicate in --where expression".into());
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
    let (col, pred) = parse_where_spec(trimmed)?;
    let target = match col {
        vxg::ColTarget::JsonKey(k) => OwnedTarget::JsonKey(k.to_owned()),
        vxg::ColTarget::Delimited { delim, index } => OwnedTarget::Delimited { delim, index },
    };
    Ok(OwnedExpr::Col { target, pred })
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

    fn parse_or(&mut self) -> R<OwnedExpr> {
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

    fn parse_and(&mut self) -> R<OwnedExpr> {
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

    fn parse_unary(&mut self) -> R<OwnedExpr> {
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
                return Err("unclosed '(' in --where expression".into());
            }
            self.pos += 1;
            return Ok(inner);
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> R<OwnedExpr> {
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

fn parse_where_expr(s: &str) -> R<OwnedExpr> {
    let mut p = ExprParser { src: s, pos: 0 };
    let expr = p.parse_or()?;
    p.skip_ws();
    if p.pos != s.len() {
        return Err(format!("unexpected trailing '{}' in --where", &s[p.pos..]).into());
    }
    Ok(expr)
}
