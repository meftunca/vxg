#[inline(always)]
const fn bg_hash(a: u8, b: u8) -> usize {
    (((a as usize).wrapping_mul(251)) ^ (b as usize) ^ ((a as usize) >> 3)) & 4095
}

const fn build_charset_mask(bytes: &[u8]) -> [u64; 4] {
    let mut m = [0u64; 4];
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        m[(b >> 6) as usize] |= 1u64 << (b & 63);
        i += 1;
    }
    m
}

const fn build_all_pairs_bigrams(bytes: &[u8]) -> [u64; 64] {
    let mut bg = [0u64; 64];
    let mut i = 0;
    while i < bytes.len() {
        let mut j = 0;
        while j < bytes.len() {
            let h = bg_hash(bytes[i], bytes[j]);
            bg[h >> 6] |= 1u64 << (h & 63);
            j += 1;
        }
        i += 1;
    }
    bg
}

const fn build_ts_bigrams() -> [u64; 64] {
    let mut bg = [0u64; 64];
    let digits = b"0123456789";
    let mut i = 0;
    while i < 10 {
        let d = digits[i];
        let h0 = bg_hash(b'"', d);
        bg[h0 >> 6] |= 1u64 << (h0 & 63);
        let mut j = 0;
        while j < 10 {
            let h = bg_hash(d, digits[j]);
            bg[h >> 6] |= 1u64 << (h & 63);
            j += 1;
        }
        let after = b"-T:Z";
        let mut k = 0;
        while k < after.len() {
            let s = after[k];
            let h1 = bg_hash(d, s);
            bg[h1 >> 6] |= 1u64 << (h1 & 63);
            if s != b'Z' {
                let h2 = bg_hash(s, d);
                bg[h2 >> 6] |= 1u64 << (h2 & 63);
            }
            k += 1;
        }
        i += 1;
    }
    let hz = bg_hash(b'Z', b'"');
    bg[hz >> 6] |= 1u64 << (hz & 63);
    bg
}

const DIGIT_CHAR_MASK: [u64; 4] = build_charset_mask(b"0123456789");
const HEX_CHAR_MASK: [u64; 4] = build_charset_mask(b"0123456789abcdefABCDEF");
const TS_CHAR_MASK: [u64; 4] = build_charset_mask(b"\"0123456789-:TZ");
const QUOTE_SINGLE_MASK: [u64; 4] = build_charset_mask(b"\"");
const DIGIT_BIGRAMS: [u64; 64] = build_all_pairs_bigrams(b"0123456789");
const HEX_BIGRAMS: [u64; 64] = build_all_pairs_bigrams(b"0123456789abcdefABCDEF");
const TS_BIGRAMS: [u64; 64] = build_ts_bigrams();

struct BigramState {
    byte_mask: [u64; 4],
    bg: [u64; 64],
    prev_end: [u64; 4],
    exact: bool,
}

impl BigramState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            byte_mask: [0u64; 4],
            bg: [0u64; 64],
            prev_end: [0u64; 4],
            exact: true,
        }
    }

    #[inline(always)]
    fn add_bg(&mut self, a: u8, b: u8) {
        let h = bg_hash(a, b);
        self.bg[h >> 6] |= 1u64 << (h & 63);
    }

    #[inline(always)]
    fn cross_prev_with_byte(&mut self, next_b: u8) {
        let prev = self.prev_end;
        for (wi, &w0) in prev.iter().enumerate() {
            let mut w = w0;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                self.add_bg(((wi << 6) | bit) as u8, next_b);
            }
        }
    }

    #[inline(always)]
    fn cross_prev_with_mask(&mut self, first_mask: &[u64; 4]) {
        let prev = self.prev_end;
        for (wi, &wa0) in prev.iter().enumerate() {
            let mut wa = wa0;
            while wa != 0 {
                let ba = ((wi << 6) | (wa.trailing_zeros() as usize)) as u8;
                wa &= wa - 1;
                for (wj, &wb0) in first_mask.iter().enumerate() {
                    let mut wb = wb0;
                    while wb != 0 {
                        let bb = ((wj << 6) | (wb.trailing_zeros() as usize)) as u8;
                        wb &= wb - 1;
                        self.add_bg(ba, bb);
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn feed_static(&mut self, s: &[u8]) {
        if s.is_empty() {
            return;
        }
        add_bytes_to_mask(&mut self.byte_mask, s);
        if self.exact {
            self.cross_prev_with_byte(s[0]);
            for w in s.windows(2) {
                self.add_bg(w[0], w[1]);
            }
            self.prev_end = [0u64; 4];
            let last = s[s.len() - 1];
            self.prev_end[(last >> 6) as usize] = 1u64 << (last & 63);
        }
    }

    #[inline(always)]
    fn feed_class(
        &mut self,
        chars: &[u64; 4],
        first: &[u64; 4],
        last: &[u64; 4],
        interior_bg: &[u64; 64],
    ) {
        for i in 0..4 {
            self.byte_mask[i] |= chars[i];
        }
        if self.exact {
            self.cross_prev_with_mask(first);
            for i in 0..64 {
                self.bg[i] |= interior_bg[i];
            }
            self.prev_end = *last;
        }
    }

    #[inline]
    fn feed_arbitrary_mask(&mut self, m: &[u64; 4], can_be_empty: bool) {
        for i in 0..4 {
            self.byte_mask[i] |= m[i];
        }
        if !self.exact {
            return;
        }
        let pop: u32 = m.iter().map(|w| w.count_ones()).sum();
        if pop <= 24 {
            self.cross_prev_with_mask(m);
            let mut bytes = [0u8; 24];
            let mut n = 0usize;
            for (wi, &w0) in m.iter().enumerate() {
                let mut w = w0;
                while w != 0 {
                    bytes[n] = ((wi << 6) | (w.trailing_zeros() as usize)) as u8;
                    n += 1;
                    w &= w - 1;
                }
            }
            for &a in &bytes[..n] {
                for &b in &bytes[..n] {
                    self.add_bg(a, b);
                }
            }
            if can_be_empty {
                for i in 0..4 {
                    self.prev_end[i] |= m[i];
                }
            } else {
                self.prev_end = *m;
            }
        } else {
            self.exact = false;
        }
    }

    #[inline(always)]
    fn matches_needle(&self, needle: &[u8]) -> bool {
        if !needle.iter().all(|&b| mask_has_byte(&self.byte_mask, b)) {
            return false;
        }
        if self.exact && needle.len() >= 2 {
            for w in needle.windows(2) {
                let h = bg_hash(w[0], w[1]);
                if (self.bg[h >> 6] & (1u64 << (h & 63))) == 0 {
                    return false;
                }
            }
        }
        true
    }
}

/// Exact / conservative header check whether a single column `col_bytes` (`ones` rows) can contain
/// `needle` completely inside one of its emitted values (`!overlaps_any_seg`).
fn col_can_contain_isolated(col_bytes: &[u8], ones: usize, needle: &[u8]) -> Result<bool> {
    let mut c = Cursor::new(col_bytes);
    let tag = c.byte()?;
    match tag {
        MODEL_RAW => {
            while !c.done() {
                let len = c.uleb()?;
                if contains_subslice(c.take(len)?, needle) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        MODEL_DICTIONARY => {
            let entries_len = c.uleb()?;
            for _ in 0..entries_len {
                let len = c.uleb()?;
                if contains_subslice(c.take(len)?, needle) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        MODEL_FRONT_DICT => {
            let entries_len = c.uleb()?;
            let dict_mode = c.byte()? & 0x03;
            if let Some(found) =
                front_dict_visit_entries(&mut c, entries_len, dict_mode, |e| {
                    contains_subslice(e, needle)
                })?
            {
                return Ok(found);
            }
            if dict_mode == 2 {
                let lz_len = c.uleb()?;
                let lz_bytes = c.take(lz_len)?;
                if lz_bytes.first() != Some(&FAST_LZ_PACKET) {
                    return Ok(true);
                }
                let mut m = [0u64; 4];
                if !fast_lz_alphabet_mask(Cursor::new(&lz_bytes[1..]), &mut m) {
                    return Ok(true);
                }
                return Ok(needle.iter().all(|&b| mask_has_byte(&m, b)));
            }
            Ok(true)
        }
        MODEL_LZ_RAW => {
            let mut m = [0u64; 4];
            if !fast_lz_alphabet_mask(c, &mut m) {
                return Ok(true);
            }
            Ok(needle.iter().all(|&b| mask_has_byte(&m, b)))
        }
        MODEL_DECIMAL_PACKED => {
            let min = c.u64()?;
            let width = c.byte()?;
            if width > 64 {
                return Ok(true);
            }
            let span = if width == 0 {
                0
            } else if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            Ok(decimal_range_can_contain(min, min.saturating_add(span), None, needle))
        }
        MODEL_DECIMAL_AFFINE => {
            let first = c.u64()?;
            let step = c.i64()?;
            if ones > 0 && ones < usize::MAX {
                let last_i = first as i128 + (step as i128) * ((ones - 1) as i128);
                if let Ok(last) = u64::try_from(last_i) {
                    return Ok(decimal_range_can_contain(
                        first.min(last),
                        first.max(last),
                        None,
                        needle,
                    ));
                }
            }
            Ok(decimal_range_can_contain(0, u64::MAX, None, needle))
        }
        MODEL_DECIMAL_DELTA => {
            let first = c.u64()?;
            let width = c.byte()?;
            if width > 64 {
                return Ok(true);
            }
            if width == 0 {
                return Ok(decimal_range_can_contain(first, first, None, needle));
            }
            if ones > 0 && ones < usize::MAX && width < 63 {
                let max_step = 1u128 << (width - 1);
                let max_delta = max_step.saturating_mul((ones - 1) as u128);
                let lo = (first as u128).saturating_sub(max_delta) as u64;
                let hi = (first as u128).saturating_add(max_delta).min(u64::MAX as u128) as u64;
                return Ok(decimal_range_can_contain(lo, hi, None, needle));
            }
            Ok(decimal_range_can_contain(0, u64::MAX, None, needle))
        }
        MODEL_DECIMAL_SUFFIX => {
            let plen = c.uleb()?;
            let prefix = c.take(plen)?;
            let min = c.u64()?;
            let width = c.byte()?;
            if width > 64 {
                return Ok(true);
            }
            let span = if width == 0 {
                0
            } else if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let hi = min.saturating_add(span);
            let mut m = [0u64; 4];
            add_bytes_to_mask(&mut m, prefix);
            add_bytes_to_mask(&mut m, b"\"0123456789");
            if !needle.iter().all(|&b| mask_has_byte(&m, b)) {
                return Ok(false);
            }
            if plen + 1 <= 128 {
                let mut lead = [0u8; 130];
                lead[0] = b'"';
                lead[1..1 + plen].copy_from_slice(prefix);
                let ls = &lead[..1 + plen];
                if contains_subslice(ls, needle) {
                    return Ok(true);
                }
                let max_p = (needle.len() - 1).min(ls.len());
                let overlaps_lead = (1..=max_p).any(|p| ls.ends_with(&needle[..p]));
                if !overlaps_lead && needle.last() != Some(&b'"') {
                    return Ok(decimal_range_can_contain(min, hi, None, needle));
                }
            }
            Ok(true)
        }
        MODEL_HEX_SUFFIX => {
            if let Some(hits) = hex_suffix_scan_nibble_hits(col_bytes, ones, needle, None)? {
                return Ok(!hits.is_empty());
            }
            let plen = c.uleb()?;
            let prefix = c.take(plen)?;
            let is_quoted = c.uleb()? <= 64;
            let mut m = [0u64; 4];
            add_bytes_to_mask(&mut m, prefix);
            add_bytes_to_mask(&mut m, b"0123456789abcdef");
            if is_quoted {
                add_bytes_to_mask(&mut m, b"\"");
            }
            Ok(needle.iter().all(|&b| mask_has_byte(&m, b)))
        }
        MODEL_EPOCH_COUNTER | MODEL_GF2_AFFINE => {
            let plen = c.uleb()?;
            let prefix = c.take(plen)?;
            let mut m = [0u64; 4];
            add_bytes_to_mask(&mut m, prefix);
            add_bytes_to_mask(&mut m, b"\"0123456789abcdefABCDEF");
            Ok(needle.iter().all(|&b| mask_has_byte(&m, b)))
        }
        MODEL_TIMESTAMP => timestamp_can_contain(c, needle),
        MODEL_PATTERN => pattern_can_contain_isolated(c, needle),
        _ => Ok(true),
    }
}

fn feed_col_into_bigram_state(col_bytes: &[u8], st: &mut BigramState) -> Result<()> {
    let mut c = Cursor::new(col_bytes);
    let tag = c.byte()?;
    match tag {
        MODEL_RAW => {
            let mut first_m = [0u64; 4];
            let mut last_m = [0u64; 4];
            let mut can_empty = false;
            while !c.done() {
                let len = c.uleb()?;
                let e = c.take(len)?;
                add_bytes_to_mask(&mut st.byte_mask, e);
                if st.exact {
                    if e.is_empty() {
                        can_empty = true;
                    } else {
                        first_m[(e[0] >> 6) as usize] |= 1u64 << (e[0] & 63);
                        let lb = e[e.len() - 1];
                        last_m[(lb >> 6) as usize] |= 1u64 << (lb & 63);
                        for w in e.windows(2) {
                            st.add_bg(w[0], w[1]);
                        }
                    }
                }
            }
            if st.exact {
                st.cross_prev_with_mask(&first_m);
                if can_empty {
                    for i in 0..4 {
                        st.prev_end[i] |= last_m[i];
                    }
                } else {
                    st.prev_end = last_m;
                }
            }
        }
        MODEL_DICTIONARY => {
            let entries_len = c.uleb()?;
            let mut first_m = [0u64; 4];
            let mut last_m = [0u64; 4];
            let mut can_empty = false;
            for _ in 0..entries_len {
                let len = c.uleb()?;
                let e = c.take(len)?;
                add_bytes_to_mask(&mut st.byte_mask, e);
                if st.exact {
                    if e.is_empty() {
                        can_empty = true;
                    } else {
                        first_m[(e[0] >> 6) as usize] |= 1u64 << (e[0] & 63);
                        let lb = e[e.len() - 1];
                        last_m[(lb >> 6) as usize] |= 1u64 << (lb & 63);
                        for w in e.windows(2) {
                            st.add_bg(w[0], w[1]);
                        }
                    }
                }
            }
            if st.exact {
                st.cross_prev_with_mask(&first_m);
                if can_empty {
                    for i in 0..4 {
                        st.prev_end[i] |= last_m[i];
                    }
                } else {
                    st.prev_end = last_m;
                }
            }
        }
        MODEL_FRONT_DICT => {
            let entries_len = c.uleb()?;
            let dict_mode = c.byte()? & 0x03;
            let mut first_m = [0u64; 4];
            let mut last_m = [0u64; 4];
            let mut can_empty = false;
            if front_dict_visit_entries(&mut c, entries_len, dict_mode, |e| {
                add_bytes_to_mask(&mut st.byte_mask, e);
                if st.exact {
                    if e.is_empty() {
                        can_empty = true;
                    } else {
                        first_m[(e[0] >> 6) as usize] |= 1u64 << (e[0] & 63);
                        let lb = e[e.len() - 1];
                        last_m[(lb >> 6) as usize] |= 1u64 << (lb & 63);
                        for w in e.windows(2) {
                            st.add_bg(w[0], w[1]);
                        }
                    }
                }
                false
            })?
            .is_some()
            {
                if st.exact {
                    st.cross_prev_with_mask(&first_m);
                    if can_empty {
                        for i in 0..4 {
                            st.prev_end[i] |= last_m[i];
                        }
                    } else {
                        st.prev_end = last_m;
                    }
                }
            } else if dict_mode == 2 {
                let lz_len = c.uleb()?;
                let lz_bytes = c.take(lz_len)?;
                let mut m = [0u64; 4];
                if lz_bytes.first() == Some(&FAST_LZ_PACKET)
                    && fast_lz_alphabet_mask(Cursor::new(&lz_bytes[1..]), &mut m)
                {
                    st.feed_arbitrary_mask(&m, true);
                } else {
                    st.byte_mask = [u64::MAX; 4];
                    st.exact = false;
                }
            } else {
                st.byte_mask = [u64::MAX; 4];
                st.exact = false;
            }
        }
        MODEL_LZ_RAW => {
            let mut m = [0u64; 4];
            if fast_lz_alphabet_mask(c, &mut m) {
                st.feed_arbitrary_mask(&m, true);
            } else {
                st.byte_mask = [u64::MAX; 4];
                st.exact = false;
            }
        }
        MODEL_DECIMAL_AFFINE | MODEL_DECIMAL_PACKED | MODEL_DECIMAL_DELTA => {
            st.feed_class(&DIGIT_CHAR_MASK, &DIGIT_CHAR_MASK, &DIGIT_CHAR_MASK, &DIGIT_BIGRAMS);
        }
        MODEL_DECIMAL_SUFFIX => {
            let plen = c.uleb()?;
            let prefix = c.take(plen)?;
            st.feed_static(b"\"");
            st.feed_static(prefix);
            st.feed_class(&DIGIT_CHAR_MASK, &DIGIT_CHAR_MASK, &DIGIT_CHAR_MASK, &DIGIT_BIGRAMS);
            st.feed_static(b"\"");
        }
        MODEL_HEX_SUFFIX | MODEL_EPOCH_COUNTER | MODEL_GF2_AFFINE => {
            let plen = c.uleb()?;
            let prefix = c.take(plen)?;
            let is_quoted = if tag == MODEL_HEX_SUFFIX { c.uleb()? <= 64 } else { true };
            if is_quoted {
                st.feed_static(b"\"");
            }
            st.feed_static(prefix);
            st.feed_class(&HEX_CHAR_MASK, &HEX_CHAR_MASK, &HEX_CHAR_MASK, &HEX_BIGRAMS);
            if is_quoted {
                st.feed_static(b"\"");
            }
        }
        MODEL_TIMESTAMP => {
            st.feed_class(&TS_CHAR_MASK, &QUOTE_SINGLE_MASK, &QUOTE_SINGLE_MASK, &TS_BIGRAMS);
        }
        MODEL_PATTERN => {
            let mut m = [0u64; 4];
            pattern_col_alphabet(&mut c, &mut m)?;
            st.feed_arbitrary_mask(&m, false);
        }
        _ => {
            st.byte_mask = [u64::MAX; 4];
            st.exact = false;
        }
    }
    Ok(())
}
