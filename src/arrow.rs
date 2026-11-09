//! Zero-dependency Apache Arrow columnar batch export (`ArrowBatch`, `ArrowCol`, `ArrowType`)
//! directly from compressed VXG frames and compressed-domain SQL queries.

use crate::{
    ops::{csv_header::peek_delim_header, unescape_json_str},
    sql::{
        execute_sql_table, OwnedCol, SelectExpr, SelectItem, SelectProjection, SqlQuery, SqlTable,
        SqlVal,
    },
    DecodeOptions, Result,
};

/// Logical Apache Arrow column data type exported by VXG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrowType {
    Boolean,
    Int64,
    Float64,
    Utf8,
    LargeUtf8,
}

impl ArrowType {
    /// Arrow C Data Interface format string (`"b"`, `"l"`, `"g"`, `"u"`, `"U"`).
    #[inline]
    pub const fn c_format_cstr(self) -> &'static [u8] {
        match self {
            Self::Boolean => b"b\0",
            Self::Int64 => b"l\0",
            Self::Float64 => b"g\0",
            Self::Utf8 => b"u\0",
            Self::LargeUtf8 => b"U\0",
        }
    }
}

/// Typed columnar payload for an [`ArrowCol`].
#[derive(Clone, Debug, PartialEq)]
pub enum ArrowData {
    Boolean(Vec<u8>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8 { offsets: Vec<i32>, values: Vec<u8> },
    LargeUtf8 { offsets: Vec<i64>, values: Vec<u8> },
}

impl ArrowData {
    #[inline]
    pub const fn data_type(&self) -> ArrowType {
        match self {
            Self::Boolean(_) => ArrowType::Boolean,
            Self::Int64(_) => ArrowType::Int64,
            Self::Float64(_) => ArrowType::Float64,
            Self::Utf8 { .. } => ArrowType::Utf8,
            Self::LargeUtf8 { .. } => ArrowType::LargeUtf8,
        }
    }
}

/// A single typed column in an [`ArrowBatch`].
#[derive(Clone, Debug, PartialEq)]
pub struct ArrowCol {
    pub name: String,
    /// LSB-first Arrow validity bitmap (`1` = valid, `0` = null), or `None` when `null_count == 0`.
    pub validity: Option<Vec<u8>>,
    pub null_count: usize,
    pub data: ArrowData,
}

/// Materialized columnar Apache Arrow RecordBatch.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrowBatch {
    pub num_rows: usize,
    pub columns: Vec<ArrowCol>,
}

#[inline]
fn cell_as_i64(v: &SqlVal) -> Option<i64> {
    match v {
        SqlVal::Int(u) => i64::try_from(*u).ok(),
        SqlVal::Dec(d) if d.scale == 0 => i64::try_from(d.mantissa).ok(),
        _ => None,
    }
}

#[inline]
fn cell_as_f64(v: &SqlVal) -> Option<f64> {
    match v {
        SqlVal::Int(u) => Some(*u as f64),
        SqlVal::Dec(d) => Some(d.to_f64()),
        SqlVal::Float(f) => Some(*f),
        _ => None,
    }
}

fn cell_to_utf8(v: &SqlVal, dst: &mut Vec<u8>) {
    match v {
        SqlVal::Null => {}
        SqlVal::Bool(b) => dst.extend_from_slice(if *b { b"true" } else { b"false" }),
        SqlVal::Int(u) => dst.extend_from_slice(u.to_string().as_bytes()),
        SqlVal::Dec(d) => dst.extend_from_slice(d.to_string().as_bytes()),
        SqlVal::Float(f) => {
            dst.extend_from_slice(crate::sql::format::format_float_clean(*f).as_bytes())
        }
        SqlVal::Str(s) => dst.extend_from_slice(s.as_bytes()),
    }
}

impl ArrowBatch {
    /// Convert a compressed-domain [`SqlTable`] into a typed columnar [`ArrowBatch`].
    pub fn from_sql_table(table: &SqlTable) -> Self {
        let num_rows = table.rows.len();
        let mut columns = Vec::with_capacity(table.headers.len());

        for (col_idx, hdr) in table.headers.iter().enumerate() {
            let mut null_count = 0usize;
            let mut all_bool = true;
            let mut all_i64 = true;
            let mut all_num = true;

            for row in &table.rows {
                match row.get(col_idx).unwrap_or(&SqlVal::Null) {
                    SqlVal::Null => null_count += 1,
                    SqlVal::Bool(_) => {
                        all_i64 = false;
                        all_num = false;
                    }
                    v => {
                        all_bool = false;
                        if cell_as_i64(v).is_none() {
                            all_i64 = false;
                        }
                        if cell_as_f64(v).is_none() {
                            all_num = false;
                        }
                    }
                }
            }

            let non_null = num_rows - null_count;
            let validity = if null_count == 0 {
                None
            } else {
                let mut bm = vec![0u8; num_rows.div_ceil(8)];
                for (r_idx, row) in table.rows.iter().enumerate() {
                    if !matches!(row.get(col_idx).unwrap_or(&SqlVal::Null), SqlVal::Null) {
                        bm[r_idx >> 3] |= 1u8 << (r_idx & 7);
                    }
                }
                Some(bm)
            };

            let lower_hdr = hdr.to_ascii_lowercase();
            let chosen = if non_null == 0 {
                if lower_hdr.starts_with("count(") || lower_hdr.starts_with("sum(") {
                    ArrowType::Int64
                } else if lower_hdr.starts_with("avg(") {
                    ArrowType::Float64
                } else {
                    ArrowType::Utf8
                }
            } else if all_bool {
                ArrowType::Boolean
            } else if all_i64 {
                ArrowType::Int64
            } else if all_num {
                ArrowType::Float64
            } else {
                ArrowType::Utf8
            };

            let data = match chosen {
                ArrowType::Boolean => {
                    let mut packed = vec![0u8; num_rows.div_ceil(8)];
                    for (r_idx, row) in table.rows.iter().enumerate() {
                        if let Some(SqlVal::Bool(true)) = row.get(col_idx) {
                            packed[r_idx >> 3] |= 1u8 << (r_idx & 7);
                        }
                    }
                    ArrowData::Boolean(packed)
                }
                ArrowType::Int64 => {
                    let mut vals = Vec::with_capacity(num_rows);
                    for row in &table.rows {
                        let v = row
                            .get(col_idx)
                            .and_then(cell_as_i64)
                            .unwrap_or(0);
                        vals.push(v);
                    }
                    ArrowData::Int64(vals)
                }
                ArrowType::Float64 => {
                    let mut vals = Vec::with_capacity(num_rows);
                    for row in &table.rows {
                        let v = row
                            .get(col_idx)
                            .and_then(cell_as_f64)
                            .unwrap_or(0.0);
                        vals.push(v);
                    }
                    ArrowData::Float64(vals)
                }
                ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                    let mut values = Vec::with_capacity(num_rows * 12);
                    let mut off64 = Vec::with_capacity(num_rows + 1);
                    off64.push(0i64);
                    for row in &table.rows {
                        if let Some(cell) = row.get(col_idx) {
                            cell_to_utf8(cell, &mut values);
                        }
                        off64.push(values.len() as i64);
                    }
                    if values.len() <= i32::MAX as usize {
                        let offsets: Vec<i32> = off64.into_iter().map(|x| x as i32).collect();
                        ArrowData::Utf8 { offsets, values }
                    } else {
                        ArrowData::LargeUtf8 {
                            offsets: off64,
                            values,
                        }
                    }
                }
            };

            columns.push(ArrowCol {
                name: hdr.clone(),
                validity,
                null_count,
                data,
            });
        }

        Self { num_rows, columns }
    }
}

/// Scan a single JSON object line for top-level keys in appearance order.
fn scan_top_level_json_keys(line: &[u8], out_keys: &mut Vec<String>) {
    let mut i = 0usize;
    while i < line.len() && line[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= line.len() || line[i] != b'{' {
        return;
    }
    i += 1;
    let mut depth = 1usize;
    while i < line.len() && depth > 0 {
        while i < line.len() && line[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= line.len() {
            break;
        }
        match line[i] {
            b'}' | b']' => {
                depth -= 1;
                i += 1;
            }
            b'{' | b'[' => {
                depth += 1;
                i += 1;
            }
            b'"' => {
                let q_start = i;
                i += 1;
                while i < line.len() {
                    if line[i] == b'\\' {
                        i = (i + 2).min(line.len());
                        continue;
                    }
                    if line[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                let q_end = i;
                while i < line.len() && line[i].is_ascii_whitespace() {
                    i += 1;
                }
                if depth == 1 && i < line.len() && line[i] == b':' {
                    i += 1;
                    if let Some(unesc) = unescape_json_str(&line[q_start..q_end]) {
                        if let Ok(k) = String::from_utf8(unesc.into_owned()) {
                            if !k.is_empty() && !out_keys.contains(&k) {
                                out_keys.push(k);
                            }
                        }
                    }
                }
            }
            _ => {
                i += 1;
            }
        }
    }
}

pub use crate::arrow_scan::{parse_arrow_where, ArrowWhere, DEFAULT_ARROW_BATCH_ROWS};
pub use crate::arrow_stream_scan::{
    export_arrow_batch_where, export_arrow_batches_where, export_arrow_ipc_where,
};

pub(crate) fn discover_ndjson_columns(
    frame_bytes: &[u8],
    dec_options: &DecodeOptions,
) -> Result<Vec<String>> {
    let probe_opts = DecodeOptions {
        threads: 1,
        ..*dec_options
    };
    let sample = crate::extract_row_range(frame_bytes, 0, 64, &probe_opts)?;
    let mut keys = Vec::new();
    for raw_ln in sample.split(|&b| b == b'\n') {
        let ln = raw_ln.strip_suffix(b"\r").unwrap_or(raw_ln);
        if !ln.is_empty() {
            scan_top_level_json_keys(ln, &mut keys);
        }
        if keys.len() >= 128 {
            break;
        }
    }
    Ok(keys)
}

/// Export a compressed VXG frame (optionally filtered/aggregated/projected by `query`)
/// into a typed columnar [`ArrowBatch`].
///
/// When `query` is `None` or `SELECT *` without `GROUP BY`:
/// - CSV/TSV frames with a header row automatically expose every named column.
/// - NDJSON frames automatically discover top-level JSON object keys across the first 64 records
///   and extract each key as a typed Arrow column in the compressed domain.
pub fn export_arrow_batch(
    frame_bytes: &[u8],
    query: Option<&SqlQuery>,
    dec_options: &DecodeOptions,
) -> Result<ArrowBatch> {
    let default_star = SqlQuery {
        projection: SelectProjection::Star,
        from_table: None,
        where_clause: None,
        group_by: Vec::new(),
        order_by: Vec::new(),
        limit: None,
    };
    let q = query.unwrap_or(&default_star);
    if let Some(res) = crate::arrow_stream_scan::try_scan_sql_query_arrow(frame_bytes, q, dec_options)
    {
        return res;
    }

    if matches!(q.projection, SelectProjection::Star)
        && q.group_by.is_empty()
        && peek_delim_header(frame_bytes, dec_options)?.is_none()
    {
        let keys = discover_ndjson_columns(frame_bytes, dec_options)?;
        if !keys.is_empty() {
            let items: Vec<SelectItem> = keys
                .into_iter()
                .map(|k| SelectItem {
                    expr: SelectExpr::Col(OwnedCol::JsonKey(k)),
                    alias: None,
                })
                .collect();
            let expanded = SqlQuery {
                projection: SelectProjection::Items(items),
                from_table: q.from_table.clone(),
                where_clause: q.where_clause.clone(),
                group_by: Vec::new(),
                order_by: q.order_by.clone(),
                limit: q.limit,
            };
            let table = execute_sql_table(frame_bytes, &expanded, dec_options)?;
            return Ok(ArrowBatch::from_sql_table(&table));
        }
    }

    let table = execute_sql_table(frame_bytes, q, dec_options)?;
    Ok(ArrowBatch::from_sql_table(&table))
}

/// Export a compressed VXG frame (optionally queried via `query`) directly into a standard
/// Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`, MetadataVersion `V5`).
pub fn export_arrow_ipc(
    frame_bytes: &[u8],
    query: Option<&SqlQuery>,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let batch = export_arrow_batch(frame_bytes, query, dec_options)?;
    Ok(batch.to_ipc_stream())
}
