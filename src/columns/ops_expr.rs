struct RowBitset {
    inline: [u64; 32],
    overflow: Vec<u64>,
    wlen: usize,
    len: usize,
}

impl RowBitset {
    #[inline]
    fn all_zeros(len: usize) -> Self {
        let wlen = len.div_ceil(64);
        Self {
            inline: [0u64; 32],
            overflow: if wlen > 32 { vec![0u64; wlen] } else { Vec::new() },
            wlen,
            len,
        }
    }

    #[inline(always)]
    fn words(&self) -> &[u64] {
        if self.wlen <= 32 { &self.inline[..self.wlen] } else { &self.overflow[..self.wlen] }
    }

    #[inline(always)]
    fn words_mut(&mut self) -> &mut [u64] {
        if self.wlen <= 32 { &mut self.inline[..self.wlen] } else { &mut self.overflow[..self.wlen] }
    }

    #[inline]
    fn last_mask(&self) -> u64 {
        let rem = self.len & 63;
        if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 }
    }

    #[inline]
    fn set(&mut self, idx: usize) {
        self.words_mut()[idx >> 6] |= 1u64 << (idx & 63);
    }

    #[inline]
    fn count_ones(&self) -> usize {
        self.words().iter().map(|w| w.count_ones() as usize).sum()
    }

    #[inline]
    fn invert(&mut self) {
        if self.wlen == 0 {
            return;
        }
        let lm = self.last_mask();
        let ws = self.words_mut();
        for w in ws.iter_mut() {
            *w = !*w;
        }
        if let Some(last) = ws.last_mut() {
            *last &= lm;
        }
    }

    #[inline]
    fn and_with(&mut self, other: &Self) -> usize {
        let mut pop = 0usize;
        for (a, &b) in self.words_mut().iter_mut().zip(other.words().iter()) {
            *a &= b;
            pop += a.count_ones() as usize;
        }
        pop
    }

    #[inline]
    fn or_with(&mut self, other: &Self) -> usize {
        let mut pop = 0usize;
        for (a, &b) in self.words_mut().iter_mut().zip(other.words().iter()) {
            *a |= b;
            pop += a.count_ones() as usize;
        }
        pop
    }

    fn collect_hits(&self, limit: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count_ones().min(limit));
        for (wi, &word0) in self.words().iter().enumerate() {
            let mut w = word0;
            let base = wi << 6;
            while w != 0 {
                let tz = w.trailing_zeros() as usize;
                out.push(base + tz);
                if out.len() >= limit {
                    return out;
                }
                w &= w - 1;
            }
        }
        out
    }
}

enum TplMask {
    Empty,
    All,
    Bits(RowBitset, usize),
}

impl TplMask {
    #[inline]
    fn from_pop(bs: RowBitset, pop: usize, ones: usize) -> Self {
        if pop == 0 {
            Self::Empty
        } else if pop == ones {
            Self::All
        } else {
            Self::Bits(bs, pop)
        }
    }

    #[inline]
    fn normalize(bs: RowBitset, ones: usize) -> Self {
        let pop = bs.count_ones();
        Self::from_pop(bs, pop, ones)
    }

    #[inline]
    fn from_hits(hits: &[usize], ones: usize) -> Self {
        if hits.is_empty() {
            return Self::Empty;
        }
        if hits.len() == ones {
            return Self::All;
        }
        let mut bs = RowBitset::all_zeros(ones);
        for &h in hits {
            bs.set(h);
        }
        Self::Bits(bs, hits.len())
    }

    #[inline]
    fn popcount(&self, ones: usize) -> usize {
        match *self {
            Self::Empty => 0,
            Self::All => ones,
            Self::Bits(_, pop) => pop,
        }
    }

    #[inline]
    fn invert(self, ones: usize) -> Self {
        if ones == 0 {
            return Self::Empty;
        }
        match self {
            Self::Empty => Self::All,
            Self::All => Self::Empty,
            Self::Bits(mut bs, pop) => {
                bs.invert();
                Self::Bits(bs, ones - pop)
            }
        }
    }

    #[inline]
    fn bitand(self, rhs: Self, ones: usize) -> Self {
        match (self, rhs) {
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::All, r) => r,
            (l, Self::All) => l,
            (Self::Bits(mut a, _), Self::Bits(b, _)) => {
                let pop = a.and_with(&b);
                Self::from_pop(a, pop, ones)
            }
        }
    }

    #[inline]
    fn bitor(self, rhs: Self, ones: usize) -> Self {
        match (self, rhs) {
            (Self::All, _) | (_, Self::All) => if ones == 0 { Self::Empty } else { Self::All },
            (Self::Empty, r) => r,
            (l, Self::Empty) => l,
            (Self::Bits(mut a, _), Self::Bits(b, _)) => {
                let pop = a.or_with(&b);
                Self::from_pop(a, pop, ones)
            }
        }
    }

    fn into_hits(self, ones: usize, limit: usize) -> Vec<usize> {
        match self {
            Self::Empty => Vec::new(),
            Self::All => (0..ones.min(limit)).collect(),
            Self::Bits(bs, _) => bs.collect_hits(limit),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TemplateTruth {
    AlwaysFalse,
    AlwaysTrue,
    Maybe,
}

/// Header-only Kleene three-valued precheck on a template before unpacking any column stream.
/// Returns `Ok(None)` only when a fallback-requiring leaf cannot be short-circuited away.
fn precheck_expr_on_template(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    expr: &crate::ops::PredExpr<'_>,
) -> Result<Option<TemplateTruth>> {
    if ones == 0 {
        return Ok(Some(TemplateTruth::AlwaysFalse));
    }
    if !segs.last().is_some_and(|s| s.ends_with(b"\n")) {
        return Ok(None);
    }
    match *expr {
        crate::ops::PredExpr::True => Ok(Some(TemplateTruth::AlwaysTrue)),
        crate::ops::PredExpr::Col(ref p) => match resolve_template_slot(segs, col_slices, p.col)? {
            SlotLookup::Absent => Ok(Some(TemplateTruth::AlwaysFalse)),
            SlotLookup::Fallback => Ok(None),
            SlotLookup::Slot(s) => {
                if !can_col_match_pred_fast(col_slices[s], ones, p)? {
                    Ok(Some(TemplateTruth::AlwaysFalse))
                } else {
                    Ok(Some(TemplateTruth::Maybe))
                }
            }
        },
        crate::ops::PredExpr::RowContains(needle) => {
            Ok(Some(match precheck_template_substr(segs, col_slices, ones, needle)? {
                TemplatePreCheck::Never => TemplateTruth::AlwaysFalse,
                TemplatePreCheck::Always => TemplateTruth::AlwaysTrue,
                TemplatePreCheck::SingleSlot(_) | TemplatePreCheck::CheckTemplate => {
                    TemplateTruth::Maybe
                }
            }))
        }
        crate::ops::PredExpr::Not(inner) => Ok(precheck_expr_on_template(
            segs, col_slices, ones, inner,
        )?
        .map(|t| match t {
            TemplateTruth::AlwaysFalse => TemplateTruth::AlwaysTrue,
            TemplateTruth::AlwaysTrue => TemplateTruth::AlwaysFalse,
            TemplateTruth::Maybe => TemplateTruth::Maybe,
        })),
        crate::ops::PredExpr::All(children) => {
            let mut any_maybe = false;
            let mut any_fallback = false;
            for c in children {
                match precheck_expr_on_template(segs, col_slices, ones, c)? {
                    Some(TemplateTruth::AlwaysFalse) => return Ok(Some(TemplateTruth::AlwaysFalse)),
                    Some(TemplateTruth::AlwaysTrue) => {}
                    Some(TemplateTruth::Maybe) => any_maybe = true,
                    None => any_fallback = true,
                }
            }
            if any_fallback {
                Ok(None)
            } else if any_maybe {
                Ok(Some(TemplateTruth::Maybe))
            } else {
                Ok(Some(TemplateTruth::AlwaysTrue))
            }
        }
        crate::ops::PredExpr::Any(children) => {
            let mut any_maybe = false;
            let mut any_fallback = false;
            for c in children {
                match precheck_expr_on_template(segs, col_slices, ones, c)? {
                    Some(TemplateTruth::AlwaysTrue) => return Ok(Some(TemplateTruth::AlwaysTrue)),
                    Some(TemplateTruth::AlwaysFalse) => {}
                    Some(TemplateTruth::Maybe) => any_maybe = true,
                    None => any_fallback = true,
                }
            }
            if any_fallback {
                Ok(None)
            } else if any_maybe {
                Ok(Some(TemplateTruth::Maybe))
            } else {
                Ok(Some(TemplateTruth::AlwaysFalse))
            }
        }
    }
}

fn eval_col_pred_mask(
    col_bytes: &[u8],
    ones: usize,
    pred: &crate::ops::ColPred<'_>,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<TplMask> {
    if !can_col_match_pred_fast(col_bytes, ones, pred)? {
        return Ok(TplMask::Empty);
    }
    ensure_col_raw_arena(col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let col = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;
    match col.const_full {
        ConstValue::Slice(s) => {
            let ok = crate::ops::eval_token_pred(pred.col, s, pred.pred);
            reset_scratch_to(scratch, raw_mark, slice_mark);
            return Ok(if ok { TplMask::All } else { TplMask::Empty });
        }
        ConstValue::Inline { ref buf, len } => {
            let ok = crate::ops::eval_token_pred(pred.col, &buf[..len as usize], pred.pred);
            reset_scratch_to(scratch, raw_mark, slice_mark);
            return Ok(if ok { TplMask::All } else { TplMask::Empty });
        }
        ConstValue::QuotedDecimal {
            prefix,
            ref buf,
            len,
        } => {
            let mut tmp = Vec::with_capacity(prefix.len() + len as usize + 2);
            tmp.push(b'"');
            tmp.extend_from_slice(prefix);
            tmp.extend_from_slice(&buf[..len as usize]);
            tmp.push(b'"');
            let ok = crate::ops::eval_token_pred(pred.col, &tmp, pred.pred);
            reset_scratch_to(scratch, raw_mark, slice_mark);
            return Ok(if ok { TplMask::All } else { TplMask::Empty });
        }
        ConstValue::None => {}
    }

    if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
        if let Some((slice_off, entries)) = col.dict_slices {
            if entries <= 256 {
                let mut valid = [false; 256];
                let mut match_count = 0usize;
                for (e, slot_ok) in valid.iter_mut().enumerate().take(entries) {
                    let (ptr, len) = scratch.slice_arena[slice_off + e];
                    let entry_bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
                    if crate::ops::eval_token_pred(pred.col, entry_bytes, pred.pred) {
                        *slot_ok = true;
                        match_count += 1;
                    }
                }
                if match_count == 0 {
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(TplMask::Empty);
                }
                if match_count == entries {
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(TplMask::All);
                }
                let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
                let mut bs = RowBitset::all_zeros(ones);
                let ws = bs.words_mut();
                let mut pop = 0usize;
                for (wi, chunk) in ids.chunks(64).enumerate() {
                    let mut w = 0u64;
                    for (bit, &id) in chunk.iter().enumerate() {
                        w |= u64::from(valid[id as usize]) << bit;
                    }
                    ws[wi] = w;
                    pop += w.count_ones() as usize;
                }
                reset_scratch_to(scratch, raw_mark, slice_mark);
                return Ok(TplMask::from_pop(bs, pop, ones));
            }
        }
    }

    if let crate::ops::ValuePred::Num(cmp) = pred.pred {
        if matches!(col.const_lead, LeadValue::Empty) && col.const_trail.is_empty() {
            match col.stream {
                ColumnStream::DecimalSmall { u16_offset } => {
                    let vals = &scratch.u16_arena[u16_offset..u16_offset + ones];
                    let mut bs = RowBitset::all_zeros(ones);
                    let ws = bs.words_mut();
                    let mut pop = 0usize;
                    for (wi, chunk) in vals.chunks(64).enumerate() {
                        let mut w = 0u64;
                        for (bit, &v) in chunk.iter().enumerate() {
                            w |= u64::from(cmp.test(u64::from(v))) << bit;
                        }
                        ws[wi] = w;
                        pop += w.count_ones() as usize;
                    }
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(if pop == 0 {
                        TplMask::Empty
                    } else if pop == ones {
                        TplMask::All
                    } else {
                        TplMask::Bits(bs, pop)
                    });
                }
                ColumnStream::DecimalValues { u64_offset } => {
                    let vals = &scratch.u64_arena[u64_offset..u64_offset + ones];
                    let mut bs = RowBitset::all_zeros(ones);
                    let ws = bs.words_mut();
                    let mut pop = 0usize;
                    for (wi, chunk) in vals.chunks(64).enumerate() {
                        let mut w = 0u64;
                        for (bit, &v) in chunk.iter().enumerate() {
                            w |= u64::from(cmp.test(v)) << bit;
                        }
                        ws[wi] = w;
                        pop += w.count_ones() as usize;
                    }
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(if pop == 0 {
                        TplMask::Empty
                    } else if pop == ones {
                        TplMask::All
                    } else {
                        TplMask::Bits(bs, pop)
                    });
                }
                ColumnStream::DecimalAffineSmall { current, step }
                | ColumnStream::DecimalAffine { current, step } => {
                    let mut bs = RowBitset::all_zeros(ones);
                    for t in 0..ones {
                        if cmp.test(current.wrapping_add(step.wrapping_mul(t as u64))) {
                            bs.set(t);
                        }
                    }
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(TplMask::normalize(bs, ones));
                }
                ColumnStream::DecimalPacked { min, mut reader } => {
                    let mut bs = RowBitset::all_zeros(ones);
                    for t in 0..ones {
                        if cmp.test(min.wrapping_add(reader.next_u64())) {
                            bs.set(t);
                        }
                    }
                    reset_scratch_to(scratch, raw_mark, slice_mark);
                    return Ok(TplMask::normalize(bs, ones));
                }
                _ => {}
            }
        }
    }

    ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let mut val_buf = vec![0u8; tpl.row_budget.max(512) + 64];
    let mut bs = RowBitset::all_zeros(ones);
    for t in 0..ones {
        let written = unsafe {
            let sp = val_buf.as_mut_ptr();
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        if crate::ops::eval_token_pred(pred.col, &val_buf[..written], pred.pred) {
            bs.set(t);
        }
    }
    reset_scratch_to(scratch, raw_mark, slice_mark);
    Ok(TplMask::normalize(bs, ones))
}

#[allow(clippy::too_many_arguments)]
fn eval_row_contains_mask(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    needle: &[u8],
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<TplMask> {
    match precheck_template_substr(segs, col_slices, ones, needle)? {
        TemplatePreCheck::Never => Ok(TplMask::Empty),
        TemplatePreCheck::Always => Ok(TplMask::All),
        TemplatePreCheck::SingleSlot(s) => {
            ensure_col_raw_arena(col_slices[s], hdr_raw_len, &mut scratch.raw_arena);
            let (_, hits) = eval_substr_on_single_slot(col_slices[s], ones, ones, needle, scratch)?;
            reset_scratch_to(scratch, raw_mark, slice_mark);
            Ok(TplMask::from_hits(&hits, ones))
        }
        TemplatePreCheck::CheckTemplate => {
            ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
            let slots = col_slices.len();
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(col_slices[s], ones, scratch)?;
            }
            let unbound = UnboundTemplate::build(segs, &cols[..slots], scratch);
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut tpl = unbound.bind(scratch);
            let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
            let mut bs = RowBitset::all_zeros(ones);
            for t in 0..ones {
                let written = unsafe {
                    let sp = row_buf.as_mut_ptr();
                    let ep = tpl.emit_row(sp);
                    ep.offset_from(sp) as usize
                };
                if contains_subslice(&row_buf[..written], needle) {
                    bs.set(t);
                }
            }
            reset_scratch_to(scratch, raw_mark, slice_mark);
            Ok(TplMask::normalize(bs, ones))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_expr_on_template(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    expr: &crate::ops::PredExpr<'_>,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<Option<TplMask>> {
    match precheck_expr_on_template(segs, col_slices, ones, expr)? {
        None => return Ok(None),
        Some(TemplateTruth::AlwaysFalse) => return Ok(Some(TplMask::Empty)),
        Some(TemplateTruth::AlwaysTrue) => return Ok(Some(TplMask::All)),
        Some(TemplateTruth::Maybe) => {}
    }
    eval_expr_inner(segs, col_slices, ones, expr, hdr_raw_len, raw_mark, slice_mark, scratch)
}

#[allow(clippy::too_many_arguments)]
fn eval_expr_inner(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    expr: &crate::ops::PredExpr<'_>,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<Option<TplMask>> {
    match *expr {
        crate::ops::PredExpr::True => Ok(Some(TplMask::All)),
        crate::ops::PredExpr::Col(ref p) => match resolve_template_slot(segs, col_slices, p.col)? {
            SlotLookup::Absent => Ok(Some(TplMask::Empty)),
            SlotLookup::Fallback => Ok(None),
            SlotLookup::Slot(s) => Ok(Some(eval_col_pred_mask(
                col_slices[s], ones, p, hdr_raw_len, raw_mark, slice_mark, scratch,
            )?)),
        },
        crate::ops::PredExpr::RowContains(needle) => Ok(Some(eval_row_contains_mask(
            segs, col_slices, ones, needle, hdr_raw_len, raw_mark, slice_mark, scratch,
        )?)),
        crate::ops::PredExpr::Not(inner) => Ok(eval_expr_inner(
            segs, col_slices, ones, inner, hdr_raw_len, raw_mark, slice_mark, scratch,
        )?
        .map(|m| m.invert(ones))),
        crate::ops::PredExpr::All(children) => {
            let mut acc = TplMask::All;
            for c in children {
                if matches!(acc, TplMask::Empty) {
                    break;
                }
                let Some(cm) = eval_expr_inner(
                    segs, col_slices, ones, c, hdr_raw_len, raw_mark, slice_mark, scratch,
                )?
                else {
                    return Ok(None);
                };
                acc = acc.bitand(cm, ones);
            }
            Ok(Some(acc))
        }
        crate::ops::PredExpr::Any(children) => {
            let mut acc = TplMask::Empty;
            for c in children {
                if matches!(acc, TplMask::All) {
                    break;
                }
                let Some(cm) = eval_expr_inner(
                    segs, col_slices, ones, c, hdr_raw_len, raw_mark, slice_mark, scratch,
                )?
                else {
                    return Ok(None);
                };
                acc = acc.bitor(cm, ones);
            }
            Ok(Some(acc))
        }
    }
}
