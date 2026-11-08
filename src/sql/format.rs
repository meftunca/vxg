//! Value representation, SQL `ORDER BY` comparator, and TSV / CSV / NDJSON formatters.

use std::cmp::Ordering;
use std::io::Write;

use crate::{
    ops::{
        decode_delim_str, encode_delim_str, encode_json_str, find_delimited_col_span,
        find_json_key_span, parse_canonical_u64, unescape_json_str, AggStats, ColTarget, DecNum,
        GroupKey, SortDir,
    },
    Error, Result,
};

use super::AggFunc;

/// Output serialization format for [`execute_sql`](super::execute_sql) and `vxg sql`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqlFormat {
    /// Tab-separated values with header row (default).
    Tsv,
    /// RFC-4180 comma-separated values with header row.
    Csv,
    /// Newline-delimited JSON objects (`NDJSON`).
    Json,
}

/// Typed cell value produced by compressed-domain SQL evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SqlVal {
    Null,
    Bool(bool),
    Int(u64),
    Dec(DecNum),
    Float(f64),
    Str(String),
}

impl SqlVal {
    #[inline]
    pub fn from_agg(func: AggFunc, st: &AggStats) -> Self {
        match func {
            AggFunc::Count => Self::Int(st.num_count),
            AggFunc::Sum => {
                if st.num_count == 0 {
                    Self::Null
                } else if st.has_signed_or_dec {
                    Self::Dec(st.sum_dec())
                } else if let Ok(u) = u64::try_from(st.sum) {
                    Self::Int(u)
                } else {
                    Self::Dec(st.sum_dec())
                }
            }
            AggFunc::Avg => st.avg().map_or(Self::Null, Self::Float),
            AggFunc::Min => {
                if st.num_count == 0 {
                    Self::Null
                } else if st.has_signed_or_dec {
                    st.min_dec().map_or(Self::Null, Self::Dec)
                } else {
                    st.min.map_or(Self::Null, Self::Int)
                }
            }
            AggFunc::Max => {
                if st.num_count == 0 {
                    Self::Null
                } else if st.has_signed_or_dec {
                    st.max_dec().map_or(Self::Null, Self::Dec)
                } else {
                    st.max.map_or(Self::Null, Self::Int)
                }
            }
        }
    }

    pub fn from_group_key(key: &GroupKey) -> Self {
        match key {
            GroupKey::Str(b) => {
                if let Some(u) = parse_canonical_u64(b) {
                    return Self::Str(u.to_string());
                }
                Self::Str(String::from_utf8_lossy(b).into_owned())
            }
            GroupKey::RawJson(b) => Self::from_raw_json_token(b),
        }
    }

    pub fn from_raw_json_token(tok: &[u8]) -> Self {
        if tok == b"null" {
            return Self::Null;
        }
        if tok == b"true" {
            return Self::Bool(true);
        }
        if tok == b"false" {
            return Self::Bool(false);
        }
        if tok.first() == Some(&b'"') {
            if let Some(unesc) = unescape_json_str(tok) {
                return Self::Str(String::from_utf8_lossy(&unesc).into_owned());
            }
        }
        if let Some(u) = parse_canonical_u64(tok) {
            return Self::Int(u);
        }
        if let Some(d) = DecNum::parse(tok) {
            return Self::Dec(d);
        }
        Self::Str(String::from_utf8_lossy(tok).into_owned())
    }

    pub fn from_delim_token(tok: &[u8]) -> Self {
        if tok.is_empty() {
            return Self::Str(String::new());
        }
        let is_quoted = tok.first() == Some(&b'"');
        let Some(dec) = decode_delim_str(tok) else {
            return Self::Str(String::from_utf8_lossy(tok).into_owned());
        };
        if !is_quoted {
            if let Some(u) = parse_canonical_u64(&dec) {
                return Self::Int(u);
            }
            if let Some(d) = DecNum::parse(&dec) {
                return Self::Dec(d);
            }
        }
        Self::Str(String::from_utf8_lossy(&dec).into_owned())
    }

    #[inline]
    fn as_f64_num(&self) -> Option<f64> {
        match *self {
            Self::Int(u) => Some(u as f64),
            Self::Dec(d) => Some(d.to_f64()),
            Self::Float(f) => Some(f),
            Self::Str(ref s) => DecNum::parse(s.as_bytes()).map(|d| d.to_f64()),
            Self::Null | Self::Bool(_) => None,
        }
    }

    #[inline]
    fn as_dec_exact(&self) -> Option<DecNum> {
        match *self {
            Self::Int(u) => Some(DecNum::from_u64(u)),
            Self::Dec(d) => Some(d),
            Self::Str(ref s) => DecNum::parse(s.as_bytes()),
            _ => None,
        }
    }
}

/// Compare two [`SqlVal`]s with SQL `NULLS LAST` semantics.
pub fn cmp_sql_val(a: &SqlVal, b: &SqlVal, dir: SortDir) -> Ordering {
    match (a, b) {
        (SqlVal::Null, SqlVal::Null) => Ordering::Equal,
        (SqlVal::Null, _) => Ordering::Greater,
        (_, SqlVal::Null) => Ordering::Less,
        _ => {
            let asc = if let (Some(da), Some(db)) = (a.as_dec_exact(), b.as_dec_exact()) {
                da.cmp(&db)
            } else if let (Some(fa), Some(fb)) = (a.as_f64_num(), b.as_f64_num()) {
                fa.total_cmp(&fb)
            } else {
                format_plain_val(a).cmp(&format_plain_val(b))
            };
            match dir {
                SortDir::Asc => asc,
                SortDir::Desc => asc.reverse(),
            }
        }
    }
}

pub fn format_float_clean(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_owned();
    }
    let s = format!("{v:.6}");
    let trimmed = s.trim_end_matches('0');
    if let Some(int_part) = trimmed.strip_suffix('.') {
        format!("{int_part}.0")
    } else {
        trimmed.to_owned()
    }
}

pub fn format_plain_val(v: &SqlVal) -> String {
    match v {
        SqlVal::Null => "NULL".to_owned(),
        SqlVal::Bool(b) => b.to_string(),
        SqlVal::Int(u) => u.to_string(),
        SqlVal::Dec(d) => d.to_string(),
        SqlVal::Float(f) => format_float_clean(*f),
        SqlVal::Str(s) => s.clone(),
    }
}

/// Materialized tabular result of a SQL query.
#[derive(Clone, Debug, PartialEq)]
pub struct SqlTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<SqlVal>>,
}

impl SqlTable {
    pub fn write_formatted<W: Write>(
        &self,
        mut w: W,
        format: SqlFormat,
        no_header: bool,
    ) -> Result<()> {
        let mut buf = Vec::with_capacity((self.rows.len() + 1) * 48);
        match format {
            SqlFormat::Tsv => {
                if !no_header && !self.headers.is_empty() {
                    for (i, h) in self.headers.iter().enumerate() {
                        if i > 0 {
                            buf.push(b'\t');
                        }
                        buf.extend_from_slice(h.as_bytes());
                    }
                    buf.push(b'\n');
                }
                for row in &self.rows {
                    for (i, cell) in row.iter().enumerate() {
                        if i > 0 {
                            buf.push(b'\t');
                        }
                        let s = format_plain_val(cell);
                        for &b in s.as_bytes() {
                            buf.push(match b {
                                b'\t' | b'\n' | b'\r' => b' ',
                                other => other,
                            });
                        }
                    }
                    buf.push(b'\n');
                }
            }
            SqlFormat::Csv => {
                if !no_header && !self.headers.is_empty() {
                    for (i, h) in self.headers.iter().enumerate() {
                        if i > 0 {
                            buf.push(b',');
                        }
                        encode_delim_str(h.as_bytes(), b',', &mut buf);
                    }
                    buf.push(b'\n');
                }
                for row in &self.rows {
                    for (i, cell) in row.iter().enumerate() {
                        if i > 0 {
                            buf.push(b',');
                        }
                        match cell {
                            SqlVal::Null => {}
                            SqlVal::Str(s) => encode_delim_str(s.as_bytes(), b',', &mut buf),
                            other => buf.extend_from_slice(format_plain_val(other).as_bytes()),
                        }
                    }
                    buf.push(b'\n');
                }
            }
            SqlFormat::Json => {
                for row in &self.rows {
                    buf.push(b'{');
                    for (i, (h, cell)) in self.headers.iter().zip(row.iter()).enumerate() {
                        if i > 0 {
                            buf.push(b',');
                        }
                        encode_json_str(h.as_bytes(), &mut buf);
                        buf.push(b':');
                        match cell {
                            SqlVal::Null => buf.extend_from_slice(b"null"),
                            SqlVal::Bool(b) => {
                                buf.extend_from_slice(if *b { b"true" } else { b"false" })
                            }
                            SqlVal::Int(u) => buf.extend_from_slice(u.to_string().as_bytes()),
                            SqlVal::Dec(d) => buf.extend_from_slice(d.to_string().as_bytes()),
                            SqlVal::Float(f) => {
                                buf.extend_from_slice(format_float_clean(*f).as_bytes())
                            }
                            SqlVal::Str(s) => encode_json_str(s.as_bytes(), &mut buf),
                        }
                    }
                    buf.extend_from_slice(b"}\n");
                }
            }
        }
        w.write_all(&buf)
            .map_err(|e| Error::Backend(e.to_string()))?;
        w.flush().map_err(|e| Error::Backend(e.to_string()))
    }

    pub fn to_bytes(&self, format: SqlFormat, no_header: bool) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_formatted(&mut out, format, no_header)?;
        Ok(out)
    }
}

pub(crate) fn extract_line_sql_val(line: &[u8], col: ColTarget<'_>) -> SqlVal {
    match col {
        ColTarget::JsonKey(key) => {
            if let Some((s, e)) = find_json_key_span(line, key.as_bytes()) {
                SqlVal::from_raw_json_token(&line[s..e])
            } else {
                SqlVal::Null
            }
        }
        ColTarget::Delimited { delim, index } => {
            if let Some((s, e)) = find_delimited_col_span(line, delim, index) {
                SqlVal::from_delim_token(&line[s..e])
            } else {
                SqlVal::Null
            }
        }
    }
}
