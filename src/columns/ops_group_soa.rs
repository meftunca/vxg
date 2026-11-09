// Zero-allocation SoA bucket accumulators and group-key decoders (`PACKED` & `DICTIONARY`).

struct SoaU32 {
    c_s: [u32; 16],
    s_s: [u64; 16],
    lo_s: [u32; 16],
    hi_s: [u32; 16],
    c_v: Vec<u32>,
    s_v: Vec<u64>,
    lo_v: Vec<u32>,
    hi_v: Vec<u32>,
    n: usize,
}

impl SoaU32 {
    #[inline(always)]
    fn new(n: usize) -> Self {
        let ov = n > 16;
        Self {
            c_s: [0; 16],
            s_s: [0; 16],
            lo_s: [u32::MAX; 16],
            hi_s: [0; 16],
            c_v: if ov { vec![0; n] } else { Vec::new() },
            s_v: if ov { vec![0; n] } else { Vec::new() },
            lo_v: if ov { vec![u32::MAX; n] } else { Vec::new() },
            hi_v: if ov { vec![0; n] } else { Vec::new() },
            n,
        }
    }

    #[inline(always)]
    fn slices_mut(&mut self) -> (&mut [u32], &mut [u64], &mut [u32], &mut [u32]) {
        if self.n <= 16 {
            (
                &mut self.c_s[..self.n],
                &mut self.s_s[..self.n],
                &mut self.lo_s[..self.n],
                &mut self.hi_s[..self.n],
            )
        } else {
            (&mut self.c_v, &mut self.s_v, &mut self.lo_v, &mut self.hi_v)
        }
    }
}

struct SoaU64 {
    c_s: [u32; 16],
    s_s: [u128; 16],
    lo_s: [u64; 16],
    hi_s: [u64; 16],
    c_v: Vec<u32>,
    s_v: Vec<u128>,
    lo_v: Vec<u64>,
    hi_v: Vec<u64>,
    n: usize,
}

impl SoaU64 {
    #[inline(always)]
    fn new(n: usize) -> Self {
        let ov = n > 16;
        Self {
            c_s: [0; 16],
            s_s: [0; 16],
            lo_s: [u64::MAX; 16],
            hi_s: [0; 16],
            c_v: if ov { vec![0; n] } else { Vec::new() },
            s_v: if ov { vec![0; n] } else { Vec::new() },
            lo_v: if ov { vec![u64::MAX; n] } else { Vec::new() },
            hi_v: if ov { vec![0; n] } else { Vec::new() },
            n,
        }
    }

    #[inline(always)]
    fn slices_mut(&mut self) -> (&mut [u32], &mut [u128], &mut [u64], &mut [u64]) {
        if self.n <= 16 {
            (
                &mut self.c_s[..self.n],
                &mut self.s_s[..self.n],
                &mut self.lo_s[..self.n],
                &mut self.hi_s[..self.n],
            )
        } else {
            (&mut self.c_v, &mut self.s_v, &mut self.lo_v, &mut self.hi_v)
        }
    }
}

enum GroupKeyBacking<'a> {
    Packed {
        min: u64,
        entries: usize,
    },
    Dict {
        raw_inline: [&'a [u8]; 16],
        raw_ov: Vec<&'a [u8]>,
        entries: usize,
    },
}

impl<'a> GroupKeyBacking<'a> {
    #[inline(always)]
    fn entries(&self) -> usize {
        match *self {
            Self::Packed { entries, .. } | Self::Dict { entries, .. } => entries,
        }
    }

    #[inline(always)]
    fn dict_entry(&self, idx: usize) -> &'a [u8] {
        match *self {
            Self::Dict {
                ref raw_inline,
                ref raw_ov,
                entries,
            } => if entries <= 16 { raw_inline[idx] } else { raw_ov[idx] },
            Self::Packed { .. } => unreachable!(),
        }
    }

    fn flush_rows_only(
        &self,
        cnt: &[u32],
        group_col: crate::ops::ColTarget<'_>,
        out_map: &mut crate::ops::agg::GroupMap,
    ) {
        for (i, &c32) in cnt[..self.entries()].iter().enumerate() {
            if c32 == 0 {
                continue;
            }
            let c = u64::from(c32);
            match *self {
                Self::Packed { min, .. } => out_map.merge_u64_rows_only(group_col, min + i as u64, c),
                Self::Dict { .. } => {
                    let raw_e = self.dict_entry(i);
                    if let Some(u) = crate::ops::parse_canonical_u64(raw_e) {
                        out_map.merge_u64_rows_only(group_col, u, c);
                    } else if let Some((is_str, k)) =
                        crate::ops::decode_group_token_ref(group_col, raw_e)
                    {
                        let mut st = crate::ops::AggStats::default();
                        st.add_rows_only(c);
                        out_map.merge_decoded(is_str, &k, &st);
                    }
                }
            }
        }
    }

    fn flush_u32_residuals(
        &self,
        soa: &SoaU32,
        a_min: u64,
        group_col: crate::ops::ColTarget<'_>,
        out_map: &mut crate::ops::agg::GroupMap,
    ) {
        let (cnt, r_sum, r_min, r_max) = if soa.n <= 16 {
            (&soa.c_s[..soa.n], &soa.s_s[..soa.n], &soa.lo_s[..soa.n], &soa.hi_s[..soa.n])
        } else {
            (&soa.c_v[..], &soa.s_v[..], &soa.lo_v[..], &soa.hi_v[..])
        };
        for (i, &c32) in cnt.iter().enumerate() {
            if c32 == 0 {
                continue;
            }
            let c = u64::from(c32);
            let sum = u128::from(a_min) * u128::from(c) + u128::from(r_sum[i]);
            let (lo, hi) = (a_min + u64::from(r_min[i]), a_min + u64::from(r_max[i]));
            match *self {
                Self::Packed { min, .. } => {
                    out_map.merge_u64_unsigned_batch(group_col, min + i as u64, c, sum, lo, hi);
                }
                Self::Dict { .. } => {
                    let raw_e = self.dict_entry(i);
                    if let Some(u) = crate::ops::parse_canonical_u64(raw_e) {
                        out_map.merge_u64_unsigned_batch(group_col, u, c, sum, lo, hi);
                    } else if let Some((is_str, k)) =
                        crate::ops::decode_group_token_ref(group_col, raw_e)
                    {
                        out_map.merge_decoded_unsigned_batch(is_str, &k, c, sum, lo, hi);
                    }
                }
            }
        }
    }

    fn flush_scaled_residuals(
        &self,
        soa: &SoaU32,
        min_v: i64,
        scale: u8,
        group_col: crate::ops::ColTarget<'_>,
        out_map: &mut crate::ops::agg::GroupMap,
    ) {
        let (cnt, r_sum, r_min, r_max) = if soa.n <= 16 {
            (&soa.c_s[..soa.n], &soa.s_s[..soa.n], &soa.lo_s[..soa.n], &soa.hi_s[..soa.n])
        } else {
            (&soa.c_v[..], &soa.s_v[..], &soa.lo_v[..], &soa.hi_v[..])
        };
        let min_i128 = i128::from(min_v);
        for (i, &c32) in cnt.iter().enumerate() {
            if c32 == 0 {
                continue;
            }
            let c = u64::from(c32);
            let mut st = crate::ops::AggStats::default();
            st.add_scaled_batch(
                min_i128 * i128::from(c) + i128::from(r_sum[i]),
                min_i128 + i128::from(r_min[i]),
                min_i128 + i128::from(r_max[i]),
                scale,
                c,
            );
            match *self {
                Self::Packed { min, .. } => out_map.merge_u64_key(group_col, min + i as u64, &st),
                Self::Dict { .. } => {
                    let raw_e = self.dict_entry(i);
                    if let Some(u) = crate::ops::parse_canonical_u64(raw_e) {
                        out_map.merge_u64_key(group_col, u, &st);
                    } else if let Some((is_str, k)) =
                        crate::ops::decode_group_token_ref(group_col, raw_e)
                    {
                        out_map.merge_decoded(is_str, &k, &st);
                    }
                }
            }
        }
    }

    fn flush_u64_vals(
        &self,
        soa: &SoaU64,
        group_col: crate::ops::ColTarget<'_>,
        out_map: &mut crate::ops::agg::GroupMap,
    ) {
        let (cnt, v_sum, v_min, v_max) = if soa.n <= 16 {
            (&soa.c_s[..soa.n], &soa.s_s[..soa.n], &soa.lo_s[..soa.n], &soa.hi_s[..soa.n])
        } else {
            (&soa.c_v[..], &soa.s_v[..], &soa.lo_v[..], &soa.hi_v[..])
        };
        for (i, &c32) in cnt.iter().enumerate() {
            if c32 == 0 {
                continue;
            }
            let c = u64::from(c32);
            match *self {
                Self::Packed { min, .. } => out_map.merge_u64_unsigned_batch(
                    group_col, min + i as u64, c, v_sum[i], v_min[i], v_max[i],
                ),
                Self::Dict { .. } => {
                    let raw_e = self.dict_entry(i);
                    if let Some(u) = crate::ops::parse_canonical_u64(raw_e) {
                        out_map.merge_u64_unsigned_batch(
                            group_col, u, c, v_sum[i], v_min[i], v_max[i],
                        );
                    } else if let Some((is_str, k)) =
                        crate::ops::decode_group_token_ref(group_col, raw_e)
                    {
                        out_map.merge_decoded_unsigned_batch(
                            is_str, &k, c, v_sum[i], v_min[i], v_max[i],
                        );
                    }
                }
            }
        }
    }
}

fn parse_group_backing<'a>(
    g_col_bytes: &'a [u8],
    ones: usize,
) -> Result<Option<(GroupKeyBacking<'a>, u8, &'a [u8], bool)>> {
    match g_col_bytes.first().copied() {
        Some(MODEL_DECIMAL_PACKED) if ones >= 2 => {
            let mut gc = Cursor::new(&g_col_bytes[1..]);
            let g_min = gc.u64()?;
            let g_width = gc.byte()?;
            if g_width > 64 {
                return Err(invalid("column packed decimal width"));
            }
            if g_width > 8 {
                return Ok(None);
            }
            let g_packed = gc.take(bit_bytes(ones, g_width)?)?;
            if !gc.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(g_packed, ones, g_width)?;
            let g_max_r = if g_width == 0 { 0 } else { (1u64 << g_width) - 1 };
            if g_min.checked_add(g_max_r).is_none() {
                return Ok(None);
            }
            let entries = if g_width == 0 { 1 } else { 1usize << g_width };
            Ok(Some((
                GroupKeyBacking::Packed { min: g_min, entries },
                g_width,
                g_packed,
                false,
            )))
        }
        Some(MODEL_DICTIONARY) if ones >= 2 => {
            let mut gc = Cursor::new(&g_col_bytes[1..]);
            let entries = gc.uleb()?;
            if entries == 0 || entries >= ones {
                return Err(invalid("column dictionary cardinality"));
            }
            if entries > 256 {
                return Ok(None);
            }
            let mut prev_entry: Option<&[u8]> = None;
            let mut raw_inline = [&[][..]; 16];
            let mut raw_ov = if entries > 16 { Vec::with_capacity(entries) } else { Vec::new() };
            for i in 0..entries {
                let len = gc.uleb()?;
                let entry = gc.take(len)?;
                if prev_entry.is_some_and(|last| last >= entry) {
                    return Err(invalid("column dictionary ordering"));
                }
                prev_entry = Some(entry);
                if entries <= 16 { raw_inline[i] = entry; } else { raw_ov.push(entry); }
            }
            let g_width = gc.byte()?;
            if g_width != bit_width((entries - 1) as u64) {
                return Err(invalid("column dictionary width"));
            }
            let g_packed = gc.take(bit_bytes(ones, g_width)?)?;
            if !gc.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(g_packed, ones, g_width)?;
            let needs_bounds = g_width > 0 && entries < (1usize << g_width);
            Ok(Some((
                GroupKeyBacking::Dict { raw_inline, raw_ov, entries },
                g_width,
                g_packed,
                needs_bounds,
            )))
        }
        _ => Ok(None),
    }
}

fn accum_group_counts(
    g_packed: &[u8],
    g_width: u8,
    entries: usize,
    needs_bounds: bool,
    ones: usize,
    mask: &TplMask,
    cnt_out: &mut [u32],
) -> Result<()> {
    match mask {
        TplMask::Empty => Ok(()),
        TplMask::All if g_width == 0 => {
            cnt_out[0] = ones as u32;
            Ok(())
        }
        TplMask::All if g_width == 1 && !needs_bounds => {
            let c1 = popcount_bytes(g_packed) as u32;
            cnt_out[1] = c1;
            cnt_out[0] = (ones as u32) - c1;
            Ok(())
        }
        TplMask::All if g_width == 2 => {
            let (mut c0, mut c1, mut c2, mut c3) = (0u32, 0u32, 0u32, 0u32);
            let mut row = 0usize;
            while row < ones {
                let n = (ones - row).min(32);
                let w = read_packed_u64_word(g_packed, (row >> 5) * 8);
                let ev = w & 0x5555_5555_5555_5555;
                let od = (w >> 1) & 0x5555_5555_5555_5555;
                let p3 = (ev & od).count_ones();
                if needs_bounds && entries == 3 && p3 != 0 {
                    return Err(invalid("column bit index out of bounds"));
                }
                let (p1, p2) = ((ev & !od).count_ones(), (!ev & od).count_ones());
                c1 += p1;
                c2 += p2;
                c3 += p3;
                c0 += (n as u32) - p1 - p2 - p3;
                row += n;
            }
            cnt_out[0] += c0;
            cnt_out[1] += c1;
            cnt_out[2] += c2;
            if entries > 3 {
                cnt_out[3] += c3;
            }
            Ok(())
        }
        TplMask::All => {
            let mut g_buf = [0u16; 64];
            let mut start_row = 0usize;
            while start_row < ones {
                let n = (ones - start_row).min(64);
                unpack_u16_chunk64(g_packed, start_row, n, g_width, &mut g_buf);
                if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                    return Err(invalid("column bit index out of bounds"));
                }
                for &id in &g_buf[..n] {
                    cnt_out[id as usize] += 1;
                }
                start_row += n;
            }
            Ok(())
        }
        TplMask::Bits(bs, _) => {
            let mut g_buf = [0u16; 64];
            for (wi, &w0) in bs.words().iter().enumerate() {
                if w0 == 0 && !needs_bounds {
                    continue;
                }
                let start_row = wi << 6;
                let n = (ones - start_row).min(64);
                if !needs_bounds && w0.count_ones() <= 16 {
                    let mut w = w0;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let id = if g_width == 0 {
                            0
                        } else {
                            read_packed_bits_at(g_packed, (start_row + bit) * g_width as usize, g_width)
                                as usize
                        };
                        cnt_out[id] += 1;
                        w &= w - 1;
                    }
                } else {
                    unpack_u16_chunk64(g_packed, start_row, n, g_width, &mut g_buf);
                    if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    let mut w = w0;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        cnt_out[g_buf[bit] as usize] += 1;
                        w &= w - 1;
                    }
                }
            }
            Ok(())
        }
    }
}
