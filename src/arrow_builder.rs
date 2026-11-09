//! Adaptive zero-ASCII columnar Apache Arrow array builders (`ColBuilder`) with lazy
//! LSB-first validity bitmaps, exact type widening (`Boolean` / `Int64` / `Float64` / `Utf8`),
//! and fast chunk concatenation.

use std::fmt::Write as _;

use crate::{
    arrow::ArrowType,
    arrow_merge::set_bit_range_ones,
    ops::{
        decode_delim_str, dec_num::POW10_I128, find_delimited_col_span, find_json_key_span,
        parse_canonical_u64, unescape_json_str, ColTarget, DecNum,
    },
};

const MAX_EXACT_F64_MANTISSA: u128 = 1u128 << 52;

#[inline]
pub(crate) fn push_utf8_lossy_bytes(values: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.is_ascii() || std::str::from_utf8(bytes).is_ok() {
        values.extend_from_slice(bytes);
    } else {
        values.extend_from_slice(String::from_utf8_lossy(bytes).as_bytes());
    }
}

#[inline]
fn write_i64_ascii(dst: &mut Vec<u8>, v: i64) {
    let mut s = String::with_capacity(20);
    let _ = write!(&mut s, "{v}");
    dst.extend_from_slice(s.as_bytes());
}

#[inline]
fn write_dec_ascii(dst: &mut Vec<u8>, d: DecNum) {
    let mut s = String::with_capacity(24);
    let _ = write!(&mut s, "{d}");
    dst.extend_from_slice(s.as_bytes());
}

pub(crate) enum BuilderState {
    Empty,
    Boolean(Vec<u8>),
    Int64(Vec<i64>),
    Float64 {
        vals: Vec<f64>,
        uniform_scale: Option<u8>,
        scales: Vec<u8>,
        overflow: Vec<(usize, DecNum)>,
    },
    Utf8 {
        offsets: Vec<i64>,
        values: Vec<u8>,
    },
}

pub(crate) struct ColBuilder {
    pub(crate) name: String,
    pub(crate) is_json: bool,
    pub(crate) len: usize,
    pub(crate) null_count: usize,
    pub(crate) validity: Vec<u8>,
    pub(crate) state: BuilderState,
}

impl ColBuilder {
    pub(crate) fn new(name: String, is_json: bool) -> Self {
        Self {
            name,
            is_json,
            len: 0,
            null_count: 0,
            validity: Vec::new(),
            state: BuilderState::Empty,
        }
    }

    #[inline]
    pub(crate) fn reserve_rows(&mut self, additional: usize) {
        match self.state {
            BuilderState::Empty => {}
            BuilderState::Boolean(ref mut b) => {
                b.reserve((self.len + additional).div_ceil(8).saturating_sub(b.len()));
            }
            BuilderState::Int64(ref mut v) => v.reserve(additional),
            BuilderState::Float64 { ref mut vals, .. } => vals.reserve(additional),
            BuilderState::Utf8 { ref mut offsets, ref mut values } => {
                offsets.reserve(additional);
                values.reserve(additional * 8);
            }
        }
    }

    #[inline]
    pub(crate) fn inferred_type_and_utf8_bytes(&self) -> (Option<ArrowType>, usize) {
        match self.state {
            BuilderState::Empty => (None, 0),
            BuilderState::Boolean(_) => (Some(ArrowType::Boolean), 0),
            BuilderState::Int64(_) => (Some(ArrowType::Int64), 0),
            BuilderState::Float64 { .. } => (Some(ArrowType::Float64), 0),
            BuilderState::Utf8 { ref values, .. } => {
                let ty = if values.len() <= i32::MAX as usize {
                    ArrowType::Utf8
                } else {
                    ArrowType::LargeUtf8
                };
                (Some(ty), values.len())
            }
        }
    }

    #[inline(always)]
    pub(crate) fn is_valid_at(&self, idx: usize) -> bool {
        self.null_count == 0 || (self.validity[idx >> 3] & (1u8 << (idx & 7))) != 0
    }

    fn mark_nulls(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let new_bytes = (self.len + n).div_ceil(8);
        if self.null_count == 0 {
            self.validity.resize(new_bytes, 0);
            set_bit_range_ones(&mut self.validity, 0, self.len);
        } else {
            self.validity.resize(new_bytes, 0);
        }
        self.null_count += n;
    }

    #[inline]
    fn mark_valid_n(&mut self, n: usize) {
        if n == 0 || self.null_count == 0 {
            return;
        }
        let new_bytes = (self.len + n).div_ceil(8);
        if self.validity.len() < new_bytes {
            self.validity.resize(new_bytes, 0);
        }
        set_bit_range_ones(&mut self.validity, self.len, n);
    }

    pub(crate) fn push_nulls(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        self.mark_nulls(n);
        let new_len = self.len + n;
        match self.state {
            BuilderState::Empty => {}
            BuilderState::Boolean(ref mut b) => b.resize(new_len.div_ceil(8), 0),
            BuilderState::Int64(ref mut v) => v.resize(new_len, 0),
            BuilderState::Float64 { ref mut vals, uniform_scale, ref mut scales, .. } => {
                vals.resize(new_len, 0.0);
                if !scales.is_empty() {
                    scales.resize(new_len, uniform_scale.unwrap_or(0));
                }
            }
            BuilderState::Utf8 { ref mut offsets, ref values } => {
                offsets.resize(new_len + 1, values.len() as i64);
            }
        }
        self.len = new_len;
    }

    #[inline]
    pub(crate) fn push_null(&mut self) {
        self.push_nulls(1);
    }

    #[cold]
    pub(crate) fn widen_to_utf8(&mut self) {
        if matches!(self.state, BuilderState::Utf8 { .. }) {
            return;
        }
        let n = self.len;
        let old = core::mem::replace(&mut self.state, BuilderState::Empty);
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0i64);
        let mut values = Vec::with_capacity(n * 8);
        match old {
            BuilderState::Empty => offsets.resize(n + 1, 0),
            BuilderState::Boolean(packed) => {
                for i in 0..n {
                    if self.is_valid_at(i) {
                        let bit = (packed[i >> 3] & (1u8 << (i & 7))) != 0;
                        values.extend_from_slice(if bit { b"true" } else { b"false" });
                    }
                    offsets.push(values.len() as i64);
                }
            }
            BuilderState::Int64(vals) => {
                for (i, &v) in vals.iter().enumerate() {
                    if self.is_valid_at(i) {
                        write_i64_ascii(&mut values, v);
                    }
                    offsets.push(values.len() as i64);
                }
            }
            BuilderState::Float64 { vals, uniform_scale, scales, overflow } => {
                let mut ov_idx = 0usize;
                for (i, &fv) in vals.iter().enumerate() {
                    if self.is_valid_at(i) {
                        while ov_idx < overflow.len() && overflow[ov_idx].0 < i {
                            ov_idx += 1;
                        }
                        if ov_idx < overflow.len() && overflow[ov_idx].0 == i {
                            write_dec_ascii(&mut values, overflow[ov_idx].1);
                        } else {
                            let sc = if scales.is_empty() {
                                uniform_scale.unwrap_or(0)
                            } else {
                                scales[i]
                            };
                            let m = (fv * (POW10_I128[sc as usize] as f64)).round() as i128;
                            write_dec_ascii(&mut values, DecNum::new(m, sc));
                        }
                    }
                    offsets.push(values.len() as i64);
                }
            }
            BuilderState::Utf8 { .. } => unreachable!(),
        }
        self.state = BuilderState::Utf8 { offsets, values };
    }

    #[cold]
    pub(crate) fn widen_to_float64(&mut self) {
        if matches!(self.state, BuilderState::Float64 { .. } | BuilderState::Utf8 { .. }) {
            return;
        }
        let n = self.len;
        match core::mem::replace(&mut self.state, BuilderState::Empty) {
            BuilderState::Empty => {
                self.state = BuilderState::Float64 {
                    vals: vec![0.0; n],
                    uniform_scale: None,
                    scales: Vec::new(),
                    overflow: Vec::new(),
                };
            }
            BuilderState::Int64(iv) => {
                let mut vals = Vec::with_capacity(n);
                let mut overflow = Vec::new();
                for (i, &v) in iv.iter().enumerate() {
                    vals.push(v as f64);
                    if self.is_valid_at(i) && v.unsigned_abs() as u128 > MAX_EXACT_F64_MANTISSA {
                        overflow.push((i, DecNum::from_i64(v)));
                    }
                }
                self.state = BuilderState::Float64 {
                    vals,
                    uniform_scale: (n > self.null_count).then_some(0),
                    scales: Vec::new(),
                    overflow,
                };
            }
            BuilderState::Boolean(b) => {
                self.state = BuilderState::Boolean(b);
                self.widen_to_utf8();
            }
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn push_bool(&mut self, b: bool) {
        if let BuilderState::Boolean(ref mut packed) = self.state {
            if self.null_count == 0 {
                if (self.len & 7) == 0 {
                    packed.push(u8::from(b));
                } else if b {
                    let idx = self.len >> 3;
                    packed[idx] |= 1u8 << (self.len & 7);
                }
                self.len += 1;
                return;
            }
        }
        self.push_bool_repeat(b, 1);
    }

    pub(crate) fn push_bool_repeat(&mut self, b: bool, n: usize) {
        if n == 0 {
            return;
        }
        if matches!(self.state, BuilderState::Empty) {
            self.state = BuilderState::Boolean(vec![0u8; self.len.div_ceil(8)]);
        } else if !matches!(self.state, BuilderState::Boolean(_)) {
            self.widen_to_utf8();
        }
        if matches!(self.state, BuilderState::Utf8 { .. }) {
            return self.push_utf8_repeat(if b { b"true" } else { b"false" }, n);
        }
        self.mark_valid_n(n);
        if let BuilderState::Boolean(ref mut packed) = self.state {
            let new_len = self.len + n;
            packed.resize(new_len.div_ceil(8), 0);
            if b {
                set_bit_range_ones(packed, self.len, n);
            }
            self.len = new_len;
        }
    }

    #[inline(always)]
    pub(crate) fn push_i64(&mut self, v: i64) {
        if let BuilderState::Int64(ref mut vals) = self.state {
            if self.null_count == 0 {
                vals.push(v);
                self.len += 1;
                return;
            }
        }
        self.push_i64_repeat(v, 1);
    }

    pub(crate) fn push_i64_repeat(&mut self, v: i64, n: usize) {
        if n == 0 {
            return;
        }
        if matches!(self.state, BuilderState::Empty) {
            let mut vals = vec![0i64; self.len];
            vals.reserve(n);
            self.state = BuilderState::Int64(vals);
        } else if matches!(self.state, BuilderState::Boolean(_)) {
            self.widen_to_utf8();
        }
        match self.state {
            BuilderState::Float64 { .. } => return self.push_dec_repeat(DecNum::from_i64(v), n),
            BuilderState::Utf8 { .. } => {
                let mut buf = Vec::with_capacity(20);
                write_i64_ascii(&mut buf, v);
                return self.push_utf8_repeat(&buf, n);
            }
            _ => {}
        }
        self.mark_valid_n(n);
        if let BuilderState::Int64(ref mut vals) = self.state {
            vals.resize(self.len + n, v);
            self.len += n;
        }
    }

    #[inline(always)]
    pub(crate) fn push_dec(&mut self, d: DecNum) {
        if d.scale == 0 && !matches!(self.state, BuilderState::Float64 { .. }) {
            if let Ok(iv) = i64::try_from(d.mantissa) {
                self.push_i64(iv);
                return;
            }
        }
        if let BuilderState::Float64 {
            ref mut vals,
            uniform_scale: Some(us),
            ref scales,
            ..
        } = self.state
        {
            if self.null_count == 0
                && us == d.scale
                && d.scale <= 15
                && scales.is_empty()
                && d.mantissa.unsigned_abs() <= MAX_EXACT_F64_MANTISSA
            {
                vals.push(d.to_f64());
                self.len += 1;
                return;
            }
        }
        self.push_dec_repeat(d, 1);
    }

    pub(crate) fn push_dec_repeat(&mut self, d: DecNum, n: usize) {
        if n == 0 {
            return;
        }
        if d.scale == 0 && !matches!(self.state, BuilderState::Float64 { .. }) {
            if let Ok(iv) = i64::try_from(d.mantissa) {
                self.push_i64_repeat(iv, n);
                return;
            }
        }
        if matches!(self.state, BuilderState::Empty | BuilderState::Int64(_)) {
            self.widen_to_float64();
        } else if matches!(self.state, BuilderState::Boolean(_)) {
            self.widen_to_utf8();
        }
        if matches!(self.state, BuilderState::Utf8 { .. }) {
            let mut buf = Vec::with_capacity(24);
            write_dec_ascii(&mut buf, d);
            return self.push_utf8_repeat(&buf, n);
        }
        self.mark_valid_n(n);
        if let BuilderState::Float64 {
            ref mut vals,
            ref mut uniform_scale,
            ref mut scales,
            ref mut overflow,
        } = self.state
        {
            let fv = d.to_f64();
            let need_ov = d.scale > 15 || d.mantissa.unsigned_abs() > MAX_EXACT_F64_MANTISSA;
            if scales.is_empty() {
                match *uniform_scale {
                    None => *uniform_scale = Some(d.scale),
                    Some(us) if us == d.scale => {}
                    Some(us) => scales.resize(self.len, us),
                }
            }
            if !scales.is_empty() {
                scales.resize(self.len + n, d.scale);
            }
            if need_ov {
                for k in 0..n {
                    overflow.push((self.len + k, d));
                }
            }
            vals.resize(self.len + n, fv);
            self.len += n;
        }
    }

    #[inline(always)]
    pub(crate) fn push_utf8_bytes(&mut self, bytes: &[u8]) {
        if let BuilderState::Utf8 { ref mut offsets, ref mut values } = self.state {
            if self.null_count == 0 && bytes.is_ascii() {
                values.extend_from_slice(bytes);
                offsets.push(values.len() as i64);
                self.len += 1;
                return;
            }
        }
        self.push_utf8_repeat(bytes, 1);
    }

    pub(crate) fn push_utf8_repeat(&mut self, bytes: &[u8], n: usize) {
        if n == 0 {
            return;
        }
        if !matches!(self.state, BuilderState::Utf8 { .. }) {
            self.widen_to_utf8();
        }
        self.mark_valid_n(n);
        let BuilderState::Utf8 { ref mut offsets, ref mut values } = self.state else {
            unreachable!();
        };
        let start_val = values.len();
        push_utf8_lossy_bytes(values, bytes);
        let unit_len = values.len() - start_val;
        if n > 1 && unit_len > 0 {
            values.reserve(unit_len * (n - 1));
            for _ in 1..n {
                values.extend_from_within(start_val..start_val + unit_len);
            }
        }
        let base_off = start_val as i64;
        offsets.reserve(n);
        for k in 1..=n {
            offsets.push(base_off + (k * unit_len) as i64);
        }
        self.len += n;
    }

    pub(crate) fn push_raw_token_repeat(&mut self, tok: &[u8], n: usize) {
        if n == 0 {
            return;
        }
        if self.is_json {
            if tok == b"null" {
                return self.push_nulls(n);
            }
            if tok == b"true" || tok == b"false" {
                return self.push_bool_repeat(tok == b"true", n);
            }
            if tok.first() == Some(&b'"') {
                if let Some(unesc) = unescape_json_str(tok) {
                    return self.push_utf8_repeat(&unesc, n);
                }
            }
            if let Some(u) = parse_canonical_u64(tok) {
                return match i64::try_from(u) {
                    Ok(iv) => self.push_i64_repeat(iv, n),
                    Err(_) => self.push_dec_repeat(DecNum::from_u64(u), n),
                };
            }
            if let Some(d) = DecNum::parse(tok) {
                return self.push_dec_repeat(d, n);
            }
            self.push_utf8_repeat(tok, n);
        } else {
            if tok.is_empty() {
                return self.push_utf8_repeat(b"", n);
            }
            let is_quoted = tok.first() == Some(&b'"');
            let Some(dec) = decode_delim_str(tok) else {
                return self.push_utf8_repeat(tok, n);
            };
            if !is_quoted {
                if let Some(u) = parse_canonical_u64(&dec) {
                    return match i64::try_from(u) {
                        Ok(iv) => self.push_i64_repeat(iv, n),
                        Err(_) => self.push_dec_repeat(DecNum::from_u64(u), n),
                    };
                }
                if let Some(d) = DecNum::parse(&dec) {
                    return self.push_dec_repeat(d, n);
                }
            }
            self.push_utf8_repeat(&dec, n);
        }
    }

    #[inline(always)]
    pub(crate) fn push_raw_token(&mut self, tok: &[u8]) {
        self.push_raw_token_repeat(tok, 1);
    }

    #[inline]
    pub(crate) fn push_line_col(&mut self, line: &[u8], col: ColTarget<'_>) {
        let span = match col {
            ColTarget::JsonKey(key) => find_json_key_span(line, key.as_bytes()),
            ColTarget::Delimited { delim, index } => find_delimited_col_span(line, delim, index),
        };
        match span {
            Some((s, e)) => self.push_raw_token(&line[s..e]),
            None => self.push_null(),
        }
    }
}
