//! Bit-packed validity/boolean buffer splicing, cross-chunk `ArrowType` unification,
//! `ColBuilder` finalization, and zero-overhead `ArrowBatch` coalescing/concatenation.

use crate::{
    arrow::{ArrowBatch, ArrowCol, ArrowData, ArrowType},
    arrow_builder::{BuilderState, ColBuilder},
};

#[inline]
pub(crate) fn set_bit_range_ones(bm: &mut [u8], start: usize, count: usize) {
    if count == 0 {
        return;
    }
    let end = start + count;
    let first_byte = start >> 3;
    let last_byte = (end - 1) >> 3;
    if first_byte == last_byte {
        let mask = ((1u16 << (end - start)) - 1) as u8;
        bm[first_byte] |= mask << (start & 7);
        return;
    }
    bm[first_byte] |= 0xFFu8 << (start & 7);
    if first_byte + 1 < last_byte {
        bm[first_byte + 1..last_byte].fill(0xFF);
    }
    let end_rem = end & 7;
    bm[last_byte] |= if end_rem == 0 { 0xFF } else { (1u8 << end_rem) - 1 };
}

pub(crate) fn append_bits(dst: &mut Vec<u8>, dst_bits: usize, src: &[u8], src_bits: usize) {
    if src_bits == 0 {
        return;
    }
    if (dst_bits & 7) != 0 && !dst.is_empty() {
        dst[dst_bits >> 3] &= (1u8 << (dst_bits & 7)) - 1;
    }
    let needed = (dst_bits + src_bits).div_ceil(8);
    if dst.len() < needed {
        dst.resize(needed, 0);
    }
    let shift = dst_bits & 7;
    let base = dst_bits >> 3;
    let full = src_bits >> 3;
    let rem = src_bits & 7;
    if shift == 0 {
        dst[base..base + full].copy_from_slice(&src[..full]);
        if rem != 0 {
            dst[base + full] = src[full] & ((1u8 << rem) - 1);
        }
        return;
    }
    for (bi, &b) in src[..full].iter().enumerate() {
        let di = base + bi;
        dst[di] |= b << shift;
        dst[di + 1] |= b >> (8 - shift);
    }
    if rem != 0 {
        let b = src[full] & ((1u8 << rem) - 1);
        let di = base + full;
        dst[di] |= b << shift;
        if shift + rem > 8 {
            dst[di + 1] |= b >> (8 - shift);
        }
    }
}

impl ColBuilder {
    pub(crate) fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        if self.null_count > 0 {
            let mut removed_nulls = 0usize;
            for i in new_len..self.len {
                if !self.is_valid_at(i) {
                    removed_nulls += 1;
                }
            }
            self.null_count -= removed_nulls;
            if self.null_count == 0 {
                self.validity.clear();
            } else {
                let b_len = new_len.div_ceil(8);
                self.validity.truncate(b_len);
                if (new_len & 7) != 0 {
                    self.validity[b_len - 1] &= (1u8 << (new_len & 7)) - 1;
                }
            }
        }
        match self.state {
            BuilderState::Empty => {}
            BuilderState::Boolean(ref mut packed) => {
                let b_len = new_len.div_ceil(8);
                packed.truncate(b_len);
                if (new_len & 7) != 0 && !packed.is_empty() {
                    packed[b_len - 1] &= (1u8 << (new_len & 7)) - 1;
                }
            }
            BuilderState::Int64(ref mut vals) => vals.truncate(new_len),
            BuilderState::Float64 { ref mut vals, ref mut scales, ref mut overflow, .. } => {
                vals.truncate(new_len);
                if !scales.is_empty() {
                    scales.truncate(new_len);
                }
                while overflow.last().is_some_and(|&(idx, _)| idx >= new_len) {
                    overflow.pop();
                }
            }
            BuilderState::Utf8 { ref mut offsets, ref mut values } => {
                offsets.truncate(new_len + 1);
                values.truncate(offsets[new_len] as usize);
            }
        }
        self.len = new_len;
    }

    pub(crate) fn finish_into_col(mut self, target_type: ArrowType) -> ArrowCol {
        match target_type {
            ArrowType::Boolean => {
                if !matches!(self.state, BuilderState::Boolean(_)) {
                    self.state = BuilderState::Boolean(vec![0u8; self.len.div_ceil(8)]);
                }
            }
            ArrowType::Int64 => match self.state {
                BuilderState::Int64(_) => {}
                BuilderState::Float64 { ref vals, .. } => {
                    let iv: Vec<i64> = vals.iter().map(|&f| f as i64).collect();
                    self.state = BuilderState::Int64(iv);
                }
                _ => self.state = BuilderState::Int64(vec![0i64; self.len]),
            },
            ArrowType::Float64 => {
                if matches!(self.state, BuilderState::Empty | BuilderState::Int64(_)) {
                    self.widen_to_float64();
                } else if !matches!(self.state, BuilderState::Float64 { .. }) {
                    self.state = BuilderState::Float64 {
                        vals: vec![0.0; self.len],
                        uniform_scale: None,
                        scales: Vec::new(),
                        overflow: Vec::new(),
                    };
                }
            }
            ArrowType::Utf8 | ArrowType::LargeUtf8 => self.widen_to_utf8(),
        }
        let data = match (target_type, self.state) {
            (ArrowType::Boolean, BuilderState::Boolean(packed)) => ArrowData::Boolean(packed),
            (ArrowType::Int64, BuilderState::Int64(vals)) => ArrowData::Int64(vals),
            (ArrowType::Float64, BuilderState::Float64 { vals, .. }) => ArrowData::Float64(vals),
            (ArrowType::Utf8, BuilderState::Utf8 { offsets, values })
                if values.len() <= i32::MAX as usize =>
            {
                let off32: Vec<i32> = offsets.into_iter().map(|x| x as i32).collect();
                ArrowData::Utf8 { offsets: off32, values }
            }
            (_, BuilderState::Utf8 { offsets, values }) => ArrowData::LargeUtf8 { offsets, values },
            _ => unreachable!(),
        };
        ArrowCol {
            name: self.name,
            validity: (self.null_count > 0).then_some(self.validity),
            null_count: self.null_count,
            data,
        }
    }
}

#[inline]
pub(crate) fn merge_arrow_type(a: Option<ArrowType>, b: Option<ArrowType>) -> Option<ArrowType> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(ArrowType::Int64), Some(ArrowType::Float64))
        | (Some(ArrowType::Float64), Some(ArrowType::Int64)) => Some(ArrowType::Float64),
        (Some(ArrowType::LargeUtf8), _) | (_, Some(ArrowType::LargeUtf8)) => {
            Some(ArrowType::LargeUtf8)
        }
        _ => Some(ArrowType::Utf8),
    }
}

pub(crate) fn empty_arrow_batch(names: &[String], col_types: &[ArrowType]) -> ArrowBatch {
    let columns = names
        .iter()
        .zip(col_types.iter())
        .map(|(n, &ty)| ArrowCol {
            name: n.clone(),
            validity: None,
            null_count: 0,
            data: match ty {
                ArrowType::Boolean => ArrowData::Boolean(Vec::new()),
                ArrowType::Int64 => ArrowData::Int64(Vec::new()),
                ArrowType::Float64 => ArrowData::Float64(Vec::new()),
                ArrowType::Utf8 => ArrowData::Utf8 { offsets: vec![0], values: Vec::new() },
                ArrowType::LargeUtf8 => {
                    ArrowData::LargeUtf8 { offsets: vec![0], values: Vec::new() }
                }
            },
        })
        .collect();
    ArrowBatch { num_rows: 0, columns }
}

pub(crate) fn infer_chunk_col_types(
    chunks: &[Vec<ColBuilder>],
    num_cols: usize,
) -> (Vec<Option<ArrowType>>, Vec<usize>) {
    let mut types = vec![None; num_cols];
    let mut str_bytes = vec![0usize; num_cols];
    for chunk in chunks {
        for j in 0..num_cols {
            let (ty, sb) = chunk[j].inferred_type_and_utf8_bytes();
            types[j] = merge_arrow_type(types[j], ty);
            let est = if sb > 0 { sb } else { chunk[j].len.saturating_mul(20) };
            str_bytes[j] = str_bytes[j].saturating_add(est);
        }
    }
    for j in 0..num_cols {
        if types[j] == Some(ArrowType::Utf8) && str_bytes[j] > i32::MAX as usize {
            types[j] = Some(ArrowType::LargeUtf8);
        }
    }
    (types, str_bytes)
}

pub(crate) fn finalize_chunk_to_batch(
    chunk: Vec<ColBuilder>,
    col_types: &[ArrowType],
) -> ArrowBatch {
    let num_rows = chunk.first().map_or(0, |c| c.len);
    let columns = chunk
        .into_iter()
        .zip(col_types.iter())
        .map(|(b, &ty)| b.finish_into_col(ty))
        .collect();
    ArrowBatch { num_rows, columns }
}

pub(crate) fn concat_arrow_batches(
    mut batches: Vec<ArrowBatch>,
    names: &[String],
    col_types: &[ArrowType],
) -> ArrowBatch {
    batches.retain(|b| b.num_rows > 0);
    if batches.is_empty() {
        return empty_arrow_batch(names, col_types);
    }
    if batches.len() == 1 {
        return batches.pop().unwrap();
    }
    let total_rows: usize = batches.iter().map(|b| b.num_rows).sum();
    let num_cols = names.len();
    let mut cols_by_idx: Vec<Vec<ArrowCol>> = (0..num_cols)
        .map(|_| Vec::with_capacity(batches.len()))
        .collect();
    let row_counts: Vec<usize> = batches.iter().map(|b| b.num_rows).collect();
    for b in batches {
        for (j, col) in b.columns.into_iter().enumerate() {
            cols_by_idx[j].push(col);
        }
    }

    let mut out_cols = Vec::with_capacity(num_cols);
    for (j, parts) in cols_by_idx.into_iter().enumerate() {
        let total_nulls: usize = parts.iter().map(|c| c.null_count).sum();
        let validity = if total_nulls == 0 {
            None
        } else {
            let mut bm = vec![0u8; total_rows.div_ceil(8)];
            let mut cur = 0usize;
            for (part, &rc) in parts.iter().zip(row_counts.iter()) {
                match part.validity {
                    None => set_bit_range_ones(&mut bm, cur, rc),
                    Some(ref v) => append_bits(&mut bm, cur, v, rc),
                }
                cur += rc;
            }
            Some(bm)
        };

        let target_ty = if col_types[j] == ArrowType::Utf8 {
            let total_bytes: usize = parts
                .iter()
                .map(|p| match &p.data {
                    ArrowData::Utf8 { values, .. } | ArrowData::LargeUtf8 { values, .. } => {
                        values.len()
                    }
                    _ => 0,
                })
                .sum();
            if total_bytes > i32::MAX as usize {
                ArrowType::LargeUtf8
            } else {
                ArrowType::Utf8
            }
        } else {
            col_types[j]
        };

        let data = match target_ty {
            ArrowType::Boolean => {
                let mut packed = Vec::with_capacity(total_rows.div_ceil(8));
                let mut cur = 0usize;
                for (part, &rc) in parts.into_iter().zip(row_counts.iter()) {
                    if let ArrowData::Boolean(src) = part.data {
                        append_bits(&mut packed, cur, &src, rc);
                    }
                    cur += rc;
                }
                ArrowData::Boolean(packed)
            }
            ArrowType::Int64 => {
                let mut vals = Vec::with_capacity(total_rows);
                for part in parts {
                    if let ArrowData::Int64(v) = part.data {
                        vals.extend_from_slice(&v);
                    }
                }
                ArrowData::Int64(vals)
            }
            ArrowType::Float64 => {
                let mut vals = Vec::with_capacity(total_rows);
                for part in parts {
                    if let ArrowData::Float64(v) = part.data {
                        vals.extend_from_slice(&v);
                    }
                }
                ArrowData::Float64(vals)
            }
            ArrowType::Utf8 => {
                let total_bytes: usize = parts
                    .iter()
                    .map(|p| match &p.data {
                        ArrowData::Utf8 { values, .. } => values.len(),
                        _ => 0,
                    })
                    .sum();
                let mut offsets = Vec::with_capacity(total_rows + 1);
                offsets.push(0i32);
                let mut values = Vec::with_capacity(total_bytes);
                for part in parts {
                    if let ArrowData::Utf8 { offsets: po, values: pv } = part.data {
                        let base = values.len() as i32;
                        values.extend_from_slice(&pv);
                        for &o in &po[1..] {
                            offsets.push(base + o);
                        }
                    }
                }
                ArrowData::Utf8 { offsets, values }
            }
            ArrowType::LargeUtf8 => {
                let total_bytes: usize = parts
                    .iter()
                    .map(|p| match &p.data {
                        ArrowData::Utf8 { values, .. } | ArrowData::LargeUtf8 { values, .. } => {
                            values.len()
                        }
                        _ => 0,
                    })
                    .sum();
                let mut offsets = Vec::with_capacity(total_rows + 1);
                offsets.push(0i64);
                let mut values = Vec::with_capacity(total_bytes);
                for part in parts {
                    match part.data {
                        ArrowData::Utf8 { offsets: po, values: pv } => {
                            let base = values.len() as i64;
                            values.extend_from_slice(&pv);
                            for &o in &po[1..] {
                                offsets.push(base + o as i64);
                            }
                        }
                        ArrowData::LargeUtf8 { offsets: po, values: pv } => {
                            let base = values.len() as i64;
                            values.extend_from_slice(&pv);
                            for &o in &po[1..] {
                                offsets.push(base + o);
                            }
                        }
                        _ => {}
                    }
                }
                ArrowData::LargeUtf8 { offsets, values }
            }
        };

        out_cols.push(ArrowCol {
            name: names[j].clone(),
            validity,
            null_count: total_nulls,
            data,
        });
    }

    ArrowBatch { num_rows: total_rows, columns: out_cols }
}

fn extract_bit_slice(src: &[u8], bit_start: usize, len: usize) -> (Vec<u8>, usize) {
    let nbytes = len.div_ceil(8);
    let mut out = Vec::with_capacity(nbytes);
    let byte_idx = bit_start >> 3;
    let shift = bit_start & 7;
    if shift == 0 {
        out.extend_from_slice(&src[byte_idx..byte_idx + nbytes]);
    } else {
        for bi in 0..nbytes {
            let s = byte_idx + bi;
            let lo = src[s] >> shift;
            let hi = src.get(s + 1).copied().unwrap_or(0) << (8 - shift);
            out.push(lo | hi);
        }
    }
    if (len & 7) != 0 {
        if let Some(last) = out.last_mut() {
            *last &= (1u8 << (len & 7)) - 1;
        }
    }
    let ones: usize = out.iter().map(|b| b.count_ones() as usize).sum();
    (out, ones)
}

pub(crate) fn slice_arrow_batch(batch: &ArrowBatch, start: usize, len: usize) -> ArrowBatch {
    let end = start + len;
    let mut columns = Vec::with_capacity(batch.columns.len());
    for col in &batch.columns {
        let (validity, null_count) = match col.validity {
            None => (None, 0),
            Some(ref bm) => {
                let (sub, ones) = extract_bit_slice(bm, start, len);
                let nc = len - ones;
                if nc == 0 { (None, 0) } else { (Some(sub), nc) }
            }
        };
        let data = match &col.data {
            ArrowData::Boolean(bm) => ArrowData::Boolean(extract_bit_slice(bm, start, len).0),
            ArrowData::Int64(v) => ArrowData::Int64(v[start..end].to_vec()),
            ArrowData::Float64(v) => ArrowData::Float64(v[start..end].to_vec()),
            ArrowData::Utf8 { offsets, values } => {
                let base = offsets[start];
                let b0 = base as usize;
                let b1 = offsets[end] as usize;
                let sub_offs = offsets[start..=end].iter().map(|&o| o - base).collect();
                ArrowData::Utf8 {
                    offsets: sub_offs,
                    values: values[b0..b1].to_vec(),
                }
            }
            ArrowData::LargeUtf8 { offsets, values } => {
                let base = offsets[start];
                let b0 = base as usize;
                let b1 = offsets[end] as usize;
                let sub_offs = offsets[start..=end].iter().map(|&o| o - base).collect();
                ArrowData::LargeUtf8 {
                    offsets: sub_offs,
                    values: values[b0..b1].to_vec(),
                }
            }
        };
        columns.push(ArrowCol {
            name: col.name.clone(),
            validity,
            null_count,
            data,
        });
    }
    ArrowBatch { num_rows: len, columns }
}

pub(crate) fn coalesce_arrow_batches(
    batches: Vec<ArrowBatch>,
    batch_size: usize,
    names: &[String],
    col_types: &[ArrowType],
) -> Vec<ArrowBatch> {
    let target = batch_size.max(1);
    let mut out = Vec::new();
    let mut group = Vec::new();
    let mut group_rows = 0usize;
    for b in batches {
        if b.num_rows == 0 {
            continue;
        }
        if group_rows == 0 && b.num_rows == target {
            out.push(b);
            continue;
        }
        if group_rows + b.num_rows <= target {
            group_rows += b.num_rows;
            group.push(b);
            if group_rows == target {
                out.push(concat_arrow_batches(core::mem::take(&mut group), names, col_types));
                group_rows = 0;
            }
            continue;
        }
        let mut off = 0usize;
        while off < b.num_rows {
            let rem = b.num_rows - off;
            let need = target - group_rows;
            if group_rows == 0 && rem >= target {
                out.push(slice_arrow_batch(&b, off, target));
                off += target;
            } else if rem >= need {
                group.push(slice_arrow_batch(&b, off, need));
                out.push(concat_arrow_batches(core::mem::take(&mut group), names, col_types));
                group_rows = 0;
                off += need;
            } else {
                group.push(slice_arrow_batch(&b, off, rem));
                group_rows += rem;
                break;
            }
        }
    }
    if !group.is_empty() {
        out.push(concat_arrow_batches(group, names, col_types));
    }
    out
}
