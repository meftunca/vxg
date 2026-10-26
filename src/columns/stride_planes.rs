/// Append `stride` delta planes (first byte, kind, payload) from contiguous `comb[0..stride*count]` to `m`:
/// - `kind = 4`: constant plane (`all_zero[p] == true`, 0 payload bytes!)
/// - `kind = 3`: combined `FAST_LZ_PACKET` across `run_len >= 2` consecutive dense planes (zero-copy slice!)
/// - `kind = 2`: 2-level 64-bit hierarchical zero-mask (ultra-fast for sparse planes with `>= 88%` zero words)
/// - `kind = 1`: nested `FAST_LZ_PACKET`
/// - `kind = 0`: raw plane
/// Returns `false` once `m` reaches `limit`.
#[allow(clippy::too_many_arguments)]
fn encode_stride_planes(
    m: &mut Vec<u8>,
    comb: &[u8],
    firsts: &[u8; 8],
    all_zero: &[bool; 8],
    any_nz_words: Option<&[u64; 8]>,
    stride: usize,
    count: usize,
    limit: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn encode_stride_planes_x86_v3(
            m: &mut Vec<u8>,
            comb: &[u8],
            firsts: &[u8; 8],
            all_zero: &[bool; 8],
            any_nz_words: Option<&[u64; 8]>,
            stride: usize,
            count: usize,
            limit: usize,
        ) -> bool {
            encode_stride_planes_impl(m, comb, firsts, all_zero, any_nz_words, stride, count, limit)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
        {
            return unsafe {
                encode_stride_planes_x86_v3(
                    m,
                    comb,
                    firsts,
                    all_zero,
                    any_nz_words,
                    stride,
                    count,
                    limit,
                )
            };
        }
    }
    encode_stride_planes_impl(m, comb, firsts, all_zero, any_nz_words, stride, count, limit)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn encode_stride_planes_impl(
    m: &mut Vec<u8>,
    comb: &[u8],
    firsts: &[u8; 8],
    all_zero: &[bool; 8],
    any_nz_words: Option<&[u64; 8]>,
    stride: usize,
    count: usize,
    limit: usize,
) -> bool {
    let words = count.div_ceil(64);
    let full_words = count / 64;
    let super_words = words.div_ceil(64);
    let step = (full_words / 16).max(1);
    let mut sparse_cand = [false; 8];
    if let Some(nz_w) = any_nz_words {
        let total_nz_words: usize = nz_w[..super_words]
            .iter()
            .map(|&x| x.count_ones() as usize)
            .sum();
        if total_nz_words * 8 <= words {
            for p in 0..stride {
                if !all_zero[p] {
                    sparse_cand[p] = true;
                }
            }
        }
    } else {
        for p in 0..stride {
            if all_zero[p] {
                continue;
            }
            let p64_base = unsafe { comb.as_ptr().add(p * count).cast::<u64>() };
            let mut sparse_est = 0usize;
            let mut sw_i = 0usize;
            let mut sampled = 0usize;
            while sw_i < full_words {
                let u = unsafe {
                    p64_base.add(sw_i * 8).read_unaligned()
                        | p64_base.add(sw_i * 8 + 1).read_unaligned()
                        | p64_base.add(sw_i * 8 + 2).read_unaligned()
                        | p64_base.add(sw_i * 8 + 3).read_unaligned()
                        | p64_base.add(sw_i * 8 + 4).read_unaligned()
                        | p64_base.add(sw_i * 8 + 5).read_unaligned()
                        | p64_base.add(sw_i * 8 + 6).read_unaligned()
                        | p64_base.add(sw_i * 8 + 7).read_unaligned()
                };
                sparse_est += (u != 0) as usize;
                sampled += 1;
                sw_i += step;
            }
            sparse_cand[p] = sampled > 0 && sparse_est * 8 <= sampled;
        }
    }

    let mut lz = Vec::new();
    let mut p = 0usize;
    while p < stride {
        m.push(firsts[p]);
        if all_zero[p] {
            m.push(4);
            if m.len() >= limit {
                return false;
            }
            p += 1;
            continue;
        }

        let plane = &comb[p * count..(p + 1) * count];

        if sparse_cand[p] {
            let p_base = plane.as_ptr();
            let mark = m.len();
            let used_kind2 = STRIDE_SCRATCH.with(|cell| {
                let mut sc = cell.borrow_mut();
                if sc.super_mask_buf.len() < super_words {
                    sc.super_mask_buf.resize(super_words, 0);
                }
                if sc.mask_buf.len() < words {
                    sc.mask_buf.resize(words, 0);
                }
                if sc.nz_buf.len() < count {
                    sc.nz_buf.resize(count, 0);
                }
                sc.super_mask_buf[..super_words].fill(0);
                let super_ptr = sc.super_mask_buf.as_mut_ptr();
                let mask_ptr = sc.mask_buf.as_mut_ptr();
                let nz_ptr = sc.nz_buf.as_mut_ptr();
                let mut nz_masks_len = 0usize;
                let mut nz_len = 0usize;
                if let Some(nz_w) = any_nz_words {
                    for sw in 0..super_words {
                        let mut cand = nz_w[sw];
                        while cand != 0 {
                            let sb = cand.trailing_zeros() as usize;
                            cand &= cand - 1;
                            let w = sw * 64 + sb;
                            if w >= words {
                                break;
                            }
                            let (w0, w1, w2, w3, w4, w5, w6, w7) = if w < full_words {
                                let p64 = unsafe { p_base.add(w * 64).cast::<u64>() };
                                unsafe {
                                    (
                                        p64.add(0).read_unaligned().to_le(),
                                        p64.add(1).read_unaligned().to_le(),
                                        p64.add(2).read_unaligned().to_le(),
                                        p64.add(3).read_unaligned().to_le(),
                                        p64.add(4).read_unaligned().to_le(),
                                        p64.add(5).read_unaligned().to_le(),
                                        p64.add(6).read_unaligned().to_le(),
                                        p64.add(7).read_unaligned().to_le(),
                                    )
                                }
                            } else {
                                let mut tail_buf = [0u8; 64];
                                let rem = count - w * 64;
                                tail_buf[..rem].copy_from_slice(&plane[w * 64..count]);
                                let p64 = tail_buf.as_ptr().cast::<u64>();
                                unsafe {
                                    (
                                        p64.add(0).read_unaligned().to_le(),
                                        p64.add(1).read_unaligned().to_le(),
                                        p64.add(2).read_unaligned().to_le(),
                                        p64.add(3).read_unaligned().to_le(),
                                        p64.add(4).read_unaligned().to_le(),
                                        p64.add(5).read_unaligned().to_le(),
                                        p64.add(6).read_unaligned().to_le(),
                                        p64.add(7).read_unaligned().to_le(),
                                    )
                                }
                            };
                            if (w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0 {
                                continue;
                            }
                            let mut mask = 0u64;
                            let chunks = [w0, w1, w2, w3, w4, w5, w6, w7];
                            for (q, &chunk) in chunks.iter().enumerate() {
                                let mut c = chunk;
                                while c != 0 {
                                    let b = (c.trailing_zeros() >> 3) as usize;
                                    let v = (c >> (b * 8)) as u8;
                                    mask |= 1u64 << (q * 8 + b);
                                    unsafe {
                                        *nz_ptr.add(nz_len) = v;
                                    }
                                    nz_len += 1;
                                    c &= !(0xffu64 << (b * 8));
                                }
                            }
                            unsafe {
                                let cur_sm = u64::from_le(*super_ptr.add(sw)) | (1u64 << sb);
                                *super_ptr.add(sw) = cur_sm.to_le();
                                *mask_ptr.add(nz_masks_len) = mask.to_le();
                            }
                            nz_masks_len += 1;
                        }
                    }
                } else {
                    for w in 0..words {
                        let (w0, w1, w2, w3, w4, w5, w6, w7) = if w < full_words {
                            let p64 = unsafe { p_base.add(w * 64).cast::<u64>() };
                            unsafe {
                                (
                                    p64.add(0).read_unaligned().to_le(),
                                    p64.add(1).read_unaligned().to_le(),
                                    p64.add(2).read_unaligned().to_le(),
                                    p64.add(3).read_unaligned().to_le(),
                                    p64.add(4).read_unaligned().to_le(),
                                    p64.add(5).read_unaligned().to_le(),
                                    p64.add(6).read_unaligned().to_le(),
                                    p64.add(7).read_unaligned().to_le(),
                                )
                            }
                        } else {
                            let mut tail_buf = [0u8; 64];
                            let rem = count - w * 64;
                            tail_buf[..rem].copy_from_slice(&plane[w * 64..count]);
                            let p64 = tail_buf.as_ptr().cast::<u64>();
                            unsafe {
                                (
                                    p64.add(0).read_unaligned().to_le(),
                                    p64.add(1).read_unaligned().to_le(),
                                    p64.add(2).read_unaligned().to_le(),
                                    p64.add(3).read_unaligned().to_le(),
                                    p64.add(4).read_unaligned().to_le(),
                                    p64.add(5).read_unaligned().to_le(),
                                    p64.add(6).read_unaligned().to_le(),
                                    p64.add(7).read_unaligned().to_le(),
                                )
                            }
                        };
                        if (w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0 {
                            continue;
                        }
                        let mut mask = 0u64;
                        let chunks = [w0, w1, w2, w3, w4, w5, w6, w7];
                        for (q, &chunk) in chunks.iter().enumerate() {
                            let mut c = chunk;
                            while c != 0 {
                                let b = (c.trailing_zeros() >> 3) as usize;
                                let v = (c >> (b * 8)) as u8;
                                mask |= 1u64 << (q * 8 + b);
                                unsafe {
                                    *nz_ptr.add(nz_len) = v;
                                }
                                nz_len += 1;
                                c &= !(0xffu64 << (b * 8));
                            }
                        }
                        unsafe {
                            let sw = w >> 6;
                            let sb = w & 63;
                            let cur_sm = u64::from_le(*super_ptr.add(sw)) | (1u64 << sb);
                            *super_ptr.add(sw) = cur_sm.to_le();
                            *mask_ptr.add(nz_masks_len) = mask.to_le();
                        }
                        nz_masks_len += 1;
                    }
                }
                let k2_size = uleb_len(nz_masks_len)
                    + uleb_len(nz_len)
                    + super_words * 8
                    + nz_masks_len * 8
                    + nz_len;
                if k2_size * 12 < count {
                    m.push(2);
                    put_uleb(m, nz_masks_len);
                    put_uleb(m, nz_len);
                    let super_bytes = unsafe {
                        std::slice::from_raw_parts(super_ptr.cast::<u8>(), super_words * 8)
                    };
                    let mask_bytes = unsafe {
                        std::slice::from_raw_parts(mask_ptr.cast::<u8>(), nz_masks_len * 8)
                    };
                    let nz_bytes = unsafe { std::slice::from_raw_parts(nz_ptr, nz_len) };
                    m.extend_from_slice(super_bytes);
                    m.extend_from_slice(mask_bytes);
                    m.extend_from_slice(nz_bytes);
                    true
                } else {
                    false
                }
            });
            if used_kind2 {
                if m.len() >= limit {
                    return false;
                }
                p += 1;
                continue;
            }
            m.truncate(mark);
        }

        // Count consecutive dense planes starting at `p` (`p .. p + run_len`), already contiguous in `comb`!
        let mut run_len = 1usize;
        while p + run_len < stride && !all_zero[p + run_len] && !sparse_cand[p + run_len] {
            run_len += 1;
        }
        if run_len >= 2 && run_len * count <= MAX_BLOCK_SIZE {
            let comb_slice = &comb[p * count..(p + run_len) * count];
            lz.clear();
            if pack_fast_lz_into(comb_slice, &mut lz).is_some() && lz.len() + 2 < comb_slice.len() {
                m.push(3);
                m.push(run_len as u8);
                for k in 1..run_len {
                    m.push(firsts[p + k]);
                }
                put_uleb(m, lz.len());
                m.extend_from_slice(&lz);
                if m.len() >= limit {
                    return false;
                }
                p += run_len;
                continue;
            }
        }

        lz.clear();
        if pack_fast_lz_into(plane, &mut lz).is_some() && lz.len() < count {
            m.push(1);
            put_uleb(m, lz.len());
            m.extend_from_slice(&lz);
        } else {
            m.push(0);
            m.extend_from_slice(plane);
        }
        if m.len() >= limit {
            return false;
        }
        p += 1;
    }
    true
}

#[inline(always)]
unsafe fn fill_px4_run(dst_u32: *mut u32, start: usize, end: usize, px32: u32) {
    let len = end - start;
    if len == 0 {
        return;
    }
    let p = unsafe { dst_u32.add(start) };
    if len == 1 {
        unsafe { p.write_unaligned(px32) };
        return;
    }
    if len <= 4 {
        for i in 0..len {
            unsafe { p.add(i).write_unaligned(px32) };
        }
        return;
    }
    let mut i = 0usize;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_set1_epi32(px32 as i32);
        while i + 16 <= len {
            _mm_storeu_si128(p.add(i).cast::<__m128i>(), v);
            _mm_storeu_si128(p.add(i + 4).cast::<__m128i>(), v);
            _mm_storeu_si128(p.add(i + 8).cast::<__m128i>(), v);
            _mm_storeu_si128(p.add(i + 12).cast::<__m128i>(), v);
            i += 16;
        }
        while i + 4 <= len {
            _mm_storeu_si128(p.add(i).cast::<__m128i>(), v);
            i += 4;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let px128 = (px32 as u128) * 0x00000001_00000001_00000001_00000001u128;
        while i + 16 <= len {
            unsafe {
                p.add(i).cast::<u128>().write_unaligned(px128);
                p.add(i + 4).cast::<u128>().write_unaligned(px128);
                p.add(i + 8).cast::<u128>().write_unaligned(px128);
                p.add(i + 12).cast::<u128>().write_unaligned(px128);
            }
            i += 16;
        }
        while i + 4 <= len {
            unsafe {
                p.add(i).cast::<u128>().write_unaligned(px128);
            }
            i += 4;
        }
    }
    while i < len {
        unsafe {
            p.add(i).write_unaligned(px32);
        }
        i += 1;
    }
}

#[inline(always)]
unsafe fn fill_px3_run(dst_ptr: *mut u8, start: usize, end: usize, c0: u8, c1: u8, c2: u8) {
    let len = end - start;
    if len == 0 {
        return;
    }
    let p = unsafe { dst_ptr.add(start * 3) };
    if len == 1 {
        unsafe {
            *p = c0;
            *p.add(1) = c1;
            *p.add(2) = c2;
        }
        return;
    }
    if c0 == c1 && c1 == c2 {
        unsafe {
            std::ptr::write_bytes(p, c0, len * 3);
        }
        return;
    }
    let mut i = 0usize;
    if len >= 8 {
        const REP24: u64 = 0x0001_0000_0100_0001;
        let q0 = (((c0 as u64) | ((c1 as u64) << 8) | ((c2 as u64) << 16))
            .wrapping_mul(REP24))
            .to_le();
        let q1 = (((c2 as u64) | ((c0 as u64) << 8) | ((c1 as u64) << 16))
            .wrapping_mul(REP24))
            .to_le();
        let q2 = (((c1 as u64) | ((c2 as u64) << 8) | ((c0 as u64) << 16))
            .wrapping_mul(REP24))
            .to_le();
        #[cfg(target_arch = "x86_64")]
        if len >= 16 {
            unsafe {
                use core::arch::x86_64::*;
                let v0 = _mm_set_epi64x(q1 as i64, q0 as i64);
                let v1 = _mm_set_epi64x(q0 as i64, q2 as i64);
                let v2 = _mm_set_epi64x(q2 as i64, q1 as i64);
                while i + 16 <= len {
                    let d = p.add(i * 3);
                    _mm_storeu_si128(d.cast::<__m128i>(), v0);
                    _mm_storeu_si128(d.add(16).cast::<__m128i>(), v1);
                    _mm_storeu_si128(d.add(32).cast::<__m128i>(), v2);
                    i += 16;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        if len >= 16 {
            let w0 = (u64::from_le(q0) as u128) | ((u64::from_le(q1) as u128) << 64);
            let w1 = (u64::from_le(q2) as u128) | ((u64::from_le(q0) as u128) << 64);
            let w2 = (u64::from_le(q1) as u128) | ((u64::from_le(q2) as u128) << 64);
            while i + 16 <= len {
                unsafe {
                    let d = p.add(i * 3);
                    d.cast::<u128>().write_unaligned(w0.to_le());
                    d.add(16).cast::<u128>().write_unaligned(w1.to_le());
                    d.add(32).cast::<u128>().write_unaligned(w2.to_le());
                }
                i += 16;
            }
        }
        while i + 8 <= len {
            unsafe {
                let d = p.add(i * 3);
                d.cast::<u64>().write_unaligned(q0);
                d.add(8).cast::<u64>().write_unaligned(q1);
                d.add(16).cast::<u64>().write_unaligned(q2);
            }
            i += 8;
        }
    }
    while i < len {
        unsafe {
            let d = p.add(i * 3);
            *d = c0;
            *d.add(1) = c1;
            *d.add(2) = c2;
        }
        i += 1;
    }
}

