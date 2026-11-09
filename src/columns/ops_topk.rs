struct CandItem {
    dir: crate::ops::SortDir,
    key: crate::ops::SortKey,
    tpl_idx: usize,
    block_row: usize,
    global_row: usize,
}

impl PartialEq for CandItem {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for CandItem {}

impl PartialOrd for CandItem {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CandItem {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        crate::ops::top_k::cmp_key_ref(
            self.key.as_ref(),
            self.global_row,
            other.key.as_ref(),
            other.global_row,
            self.dir,
        )
    }
}

struct CandTracker {
    k: usize,
    dir: crate::ops::SortDir,
    heap: std::collections::BinaryHeap<CandItem>,
}

impl CandTracker {
    #[inline]
    fn new(k: usize, dir: crate::ops::SortDir) -> Self {
        Self {
            k,
            dir,
            heap: std::collections::BinaryHeap::new(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[inline]
    fn can_accept(&self, key: crate::ops::top_k::SortKeyRef<'_>, global_row: usize) -> bool {
        if self.k == 0 {
            return false;
        }
        if self.heap.len() < self.k {
            return true;
        }
        let worst = self.heap.peek().unwrap();
        crate::ops::top_k::cmp_key_ref(
            key,
            global_row,
            worst.key.as_ref(),
            worst.global_row,
            self.dir,
        ) == std::cmp::Ordering::Less
    }

    #[inline]
    fn try_push(
        &mut self,
        key: crate::ops::top_k::SortKeyRef<'_>,
        tpl_idx: usize,
        block_row: usize,
        global_row: usize,
        outer: &crate::ops::top_k::TopKHeap,
    ) -> bool {
        if !outer.can_accept_key(key, global_row) || !self.can_accept(key, global_row) {
            return false;
        }
        let item = CandItem {
            dir: self.dir,
            key: key.to_owned_key(),
            tpl_idx,
            block_row,
            global_row,
        };
        if self.heap.len() < self.k {
            if self.heap.capacity() == 0 {
                self.heap.reserve(self.k.min(256));
            }
            self.heap.push(item);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            *worst = item;
        }
        true
    }

    #[inline]
    fn tight_num_bound(&self) -> Option<u64> {
        if self.k > 0 && self.heap.len() == self.k {
            if let crate::ops::SortKey::Num(b) = self.heap.peek().unwrap().key {
                return Some(b);
            }
        }
        None
    }

    fn into_tpl_sorted(self) -> Vec<CandItem> {
        let mut v: Vec<CandItem> = self.heap.into_vec();
        v.sort_unstable_by_key(|c| c.tpl_idx);
        v
    }
}

enum TplRowMap<'a> {
    Offset(usize),
    Cursor(SelectorBitCursor<'a>),
    Explicit(&'a [usize]),
}

impl TplRowMap<'_> {
    #[inline]
    fn row_of(&mut self, tpl_idx: usize) -> usize {
        match self {
            Self::Offset(off) => tpl_idx + *off,
            Self::Cursor(cur) => cur.advance_to(tpl_idx),
            Self::Explicit(map) => map[tpl_idx],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_numeric_slice_cands<T: Copy + Into<u64>>(
    vals: &[T],
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    init_cmp: crate::query::NumCmp,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
) {
    use crate::ops::top_k::SortKeyRef;
    use crate::ops::SortDir;

    match (outer.dir(), init_cmp, mask) {
        (SortDir::Desc, crate::query::NumCmp::Ge(mut lo), TplMask::All) => {
            for (ti, &vt) in vals.iter().enumerate() {
                let v: u64 = vt.into();
                if v < lo {
                    continue;
                }
                let b_row = row_map.row_of(ti);
                if cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer) {
                    if let Some(nb) = cands.tight_num_bound() {
                        lo = lo.max(nb);
                    }
                }
            }
        }
        (SortDir::Asc, crate::query::NumCmp::Le(mut hi), TplMask::All) => {
            for (ti, &vt) in vals.iter().enumerate() {
                let v: u64 = vt.into();
                if v > hi {
                    continue;
                }
                let b_row = row_map.row_of(ti);
                if cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer) {
                    if let Some(nb) = cands.tight_num_bound() {
                        hi = hi.min(nb);
                    }
                }
            }
        }
        (SortDir::Desc, crate::query::NumCmp::Ge(mut lo), TplMask::Bits(bs, _)) => {
            for (wi, &word0) in bs.words().iter().enumerate() {
                let mut w = word0;
                let base = wi << 6;
                while w != 0 {
                    let ti = base + w.trailing_zeros() as usize;
                    let v: u64 = vals[ti].into();
                    if v >= lo {
                        let b_row = row_map.row_of(ti);
                        if cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer) {
                            if let Some(nb) = cands.tight_num_bound() {
                                lo = lo.max(nb);
                            }
                        }
                    }
                    w &= w - 1;
                }
            }
        }
        (SortDir::Asc, crate::query::NumCmp::Le(mut hi), TplMask::Bits(bs, _)) => {
            for (wi, &word0) in bs.words().iter().enumerate() {
                let mut w = word0;
                let base = wi << 6;
                while w != 0 {
                    let ti = base + w.trailing_zeros() as usize;
                    let v: u64 = vals[ti].into();
                    if v <= hi {
                        let b_row = row_map.row_of(ti);
                        if cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer) {
                            if let Some(nb) = cands.tight_num_bound() {
                                hi = hi.min(nb);
                            }
                        }
                    }
                    w &= w - 1;
                }
            }
        }
        _ => {}
    }
}

#[inline]
fn push_const_key_cands(
    key_ref: crate::ops::top_k::SortKeyRef<'_>,
    ones: usize,
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
) {
    match mask {
        TplMask::All => {
            for ti in 0..ones {
                let b_row = row_map.row_of(ti);
                if !cands.try_push(key_ref, ti, b_row, base_row + b_row, outer) {
                    break;
                }
            }
        }
        TplMask::Bits(bs, _) => {
            'words: for (wi, &word0) in bs.words().iter().enumerate() {
                let mut w = word0;
                let base = wi << 6;
                while w != 0 {
                    let ti = base + w.trailing_zeros() as usize;
                    let b_row = row_map.row_of(ti);
                    if !cands.try_push(key_ref, ti, b_row, base_row + b_row, outer) {
                        break 'words;
                    }
                    w &= w - 1;
                }
            }
        }
        TplMask::Empty => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_order_slot_cands(
    order_col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    order_col: crate::ops::ColTarget<'_>,
    hdr_raw_len: usize,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
    scratch: &mut DecodeScratch,
) -> Result<()> {
    let digit_delim = matches!(
        order_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    );
    if try_scan_scaled_dec_topk(
        order_col_bytes,
        ones,
        mask,
        row_map,
        base_row,
        digit_delim,
        outer,
        cands,
    )? || try_scan_decimal_topk(
        order_col_bytes,
        ones,
        mask,
        row_map,
        base_row,
        digit_delim,
        shared_bound,
        outer,
        cands,
    )? {
        return Ok(());
    }
    scan_order_slot_cands_scratch(
        order_col_bytes,
        ones,
        mask,
        row_map,
        base_row,
        order_col,
        digit_delim,
        hdr_raw_len,
        shared_bound,
        outer,
        cands,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_order_slot_cands_scratch(
    order_col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    order_col: crate::ops::ColTarget<'_>,
    digit_delim: bool,
    hdr_raw_len: usize,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
    scratch: &mut DecodeScratch,
) -> Result<()> {
    use crate::ops::top_k::{extract_token_sort_key, extract_token_sort_key_ref, SortKeyRef};

    ensure_col_raw_arena(order_col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let col = PreparedColumn::parse_and_validate(order_col_bytes, ones, scratch)?;

    match col.const_full {
        ConstValue::Slice(s) => {
            let ck = extract_token_sort_key(order_col, s);
            push_const_key_cands(ck.as_ref(), ones, mask, row_map, base_row, outer, cands);
            return Ok(());
        }
        ConstValue::Inline { ref buf, len } => {
            let ck = extract_token_sort_key(order_col, &buf[..len as usize]);
            push_const_key_cands(ck.as_ref(), ones, mask, row_map, base_row, outer, cands);
            return Ok(());
        }
        ConstValue::QuotedDecimal {
            prefix,
            ref buf,
            len,
        } => {
            let mut raw = Vec::with_capacity(prefix.len() + len as usize + 2);
            raw.push(b'"');
            raw.extend_from_slice(prefix);
            raw.extend_from_slice(&buf[..len as usize]);
            raw.push(b'"');
            let ck = extract_token_sort_key(order_col, &raw);
            push_const_key_cands(ck.as_ref(), ones, mask, row_map, base_row, outer, cands);
            return Ok(());
        }
        ConstValue::None => {}
    }

    if !digit_delim && matches!(col.const_lead, LeadValue::Empty) && col.const_trail.is_empty() {
        match col.stream {
            ColumnStream::DecimalSmall { u16_offset } => {
                let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(());
                };
                if let crate::query::NumCmp::Ge(lo) = cmp {
                    if lo > u64::from(u16::MAX) {
                        return Ok(());
                    }
                }
                let vals = &scratch.u16_arena[u16_offset..u16_offset + ones];
                scan_numeric_slice_cands(vals, mask, row_map, base_row, cmp, outer, cands);
                return Ok(());
            }
            ColumnStream::DecimalValues { u64_offset } => {
                let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(());
                };
                if !can_numeric_col_match_fast(order_col_bytes, ones, cmp) {
                    return Ok(());
                }
                let vals = &scratch.u64_arena[u64_offset..u64_offset + ones];
                scan_numeric_slice_cands(vals, mask, row_map, base_row, cmp, outer, cands);
                return Ok(());
            }
            ColumnStream::DecimalPacked { min, mut reader } => {
                let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(());
                };
                let u64_offset = scratch.u64_arena.len();
                scratch.u64_arena.reserve(ones);
                for _ in 0..ones {
                    scratch.u64_arena.push(min.wrapping_add(reader.next_u64()));
                }
                let vals = &scratch.u64_arena[u64_offset..u64_offset + ones];
                scan_numeric_slice_cands(vals, mask, row_map, base_row, cmp, outer, cands);
                return Ok(());
            }
            ColumnStream::DecimalAffineSmall { current, step }
            | ColumnStream::DecimalAffine { current, step } => {
                let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(());
                };
                match mask {
                    TplMask::All => {
                        for ti in 0..ones {
                            let v = current.wrapping_add(step.wrapping_mul(ti as u64));
                            if !cmp.test(v) {
                                continue;
                            }
                            let b_row = row_map.row_of(ti);
                            cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer);
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        for (wi, &word0) in bs.words().iter().enumerate() {
                            let mut w = word0;
                            let base = wi << 6;
                            while w != 0 {
                                let ti = base + w.trailing_zeros() as usize;
                                let v = current.wrapping_add(step.wrapping_mul(ti as u64));
                                if cmp.test(v) {
                                    let b_row = row_map.row_of(ti);
                                    cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    );
                                }
                                w &= w - 1;
                            }
                        }
                    }
                    TplMask::Empty => {}
                }
                return Ok(());
            }
            _ => {}
        }
    }

    if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
        if let Some((slice_off, entries)) = col.dict_slices {
            if entries <= 256 {
                let mut dict_keys = Vec::with_capacity(entries);
                let mut alive = [false; 256];
                let mut any_alive = false;
                for (e, slot_alive) in alive.iter_mut().enumerate().take(entries) {
                    let (ptr, len) = scratch.slice_arena[slice_off + e];
                    let raw_tok = unsafe { std::slice::from_raw_parts(ptr, len) };
                    let dk = extract_token_sort_key(order_col, raw_tok);
                    if outer.can_accept_key_from(dk.as_ref(), base_row) {
                        *slot_alive = true;
                        any_alive = true;
                    }
                    dict_keys.push(dk);
                }
                if !any_alive {
                    return Ok(());
                }
                let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
                match mask {
                    TplMask::All => {
                        for (ti, &id) in ids.iter().enumerate() {
                            let idx = id as usize;
                            if !alive[idx] {
                                continue;
                            }
                            let b_row = row_map.row_of(ti);
                            cands.try_push(
                                dict_keys[idx].as_ref(),
                                ti,
                                b_row,
                                base_row + b_row,
                                outer,
                            );
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        for (wi, &word0) in bs.words().iter().enumerate() {
                            let mut w = word0;
                            let base = wi << 6;
                            while w != 0 {
                                let ti = base + w.trailing_zeros() as usize;
                                let idx = ids[ti] as usize;
                                if alive[idx] {
                                    let b_row = row_map.row_of(ti);
                                    cands.try_push(
                                        dict_keys[idx].as_ref(),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    );
                                }
                                w &= w - 1;
                            }
                        }
                    }
                    TplMask::Empty => {}
                }
                return Ok(());
            }
        }
    }

    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let mut val_buf = vec![0u8; tpl.row_budget.max(512) + 64];
    let mut cur_t = 0usize;

    let mut visit_row = |ti: usize| {
        let written = unsafe {
            tpl.skip_rows(ti - cur_t);
            let sp = val_buf.as_mut_ptr();
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        cur_t = ti + 1;
        let tok = &val_buf[..written];
        let b_row = row_map.row_of(ti);
        if let Some(kref) = extract_token_sort_key_ref(order_col, tok) {
            cands.try_push(kref, ti, b_row, base_row + b_row, outer);
        } else {
            let owned = extract_token_sort_key(order_col, tok);
            cands.try_push(owned.as_ref(), ti, b_row, base_row + b_row, outer);
        }
    };

    match mask {
        TplMask::All => {
            for ti in 0..ones {
                visit_row(ti);
            }
        }
        TplMask::Bits(bs, _) => {
            for (wi, &word0) in bs.words().iter().enumerate() {
                let mut w = word0;
                let base = wi << 6;
                while w != 0 {
                    visit_row(base + w.trailing_zeros() as usize);
                    w &= w - 1;
                }
            }
        }
        TplMask::Empty => {}
    }
    Ok(())
}
