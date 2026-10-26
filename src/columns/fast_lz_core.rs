/// Core 4-stream LZ77 + repeat-offset match finder.
/// Returns `(seq_count, off_count, lit_total_len)`.
#[allow(clippy::too_many_arguments)]
fn pack_fast_lz_core(
    data: &[u8],
    use_dict7: bool,
    dict4_vec: &mut Vec<u32>,
    dict7_vec: &mut Vec<u16>,
    tokens_vec: &mut Vec<u8>,
    offsets_vec: &mut Vec<u8>,
    ext_vec: &mut Vec<u8>,
    lits_vec: &mut Vec<u8>,
    hash_len: u32,
) -> (usize, usize, usize) {
    match (hash_len, use_dict7) {
        (4, false) => pack_fast_lz_core_const::<4, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (4, true) => pack_fast_lz_core_const::<4, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (5, false) => pack_fast_lz_core_const::<5, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (5, true) => pack_fast_lz_core_const::<5, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (_, false) => pack_fast_lz_core_const::<6, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (_, true) => pack_fast_lz_core_const::<6, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn pack_fast_lz_core_const<const HASH_LEN: u32, const USE_DICT7: bool>(
    data: &[u8],
    dict4_vec: &mut Vec<u32>,
    dict7_vec: &mut Vec<u16>,
    tokens_vec: &mut Vec<u8>,
    offsets_vec: &mut Vec<u8>,
    ext_vec: &mut Vec<u8>,
    lits_vec: &mut Vec<u8>,
) -> (usize, usize, usize) {
    let n = data.len();
    let ptr = data.as_ptr();
    let (d4_len, shift4) = if n <= 512 {
        (256usize, 24u32)
    } else if n <= 2048 {
        (1024usize, 22u32)
    } else if n <= 8192 {
        (2048usize, 21u32)
    } else if n <= 24576 {
        (4096usize, 20u32)
    } else {
        (8192usize, 19u32)
    };
    let (d7_len, shift7) = (4096usize, 52u32);
    if dict4_vec.len() < d4_len {
        dict4_vec.resize(d4_len, 0);
    } else {
        dict4_vec[..d4_len].fill(0);
    }
    if USE_DICT7 {
        if dict7_vec.len() < d7_len {
            dict7_vec.resize(d7_len, 0);
        } else {
            dict7_vec[..d7_len].fill(0);
        }
    }

    let max_seqs = n / 4 + 16;
    if tokens_vec.len() < max_seqs {
        tokens_vec.resize(max_seqs, 0);
    }
    if offsets_vec.len() < max_seqs * 2 {
        offsets_vec.resize(max_seqs * 2, 0);
    }
    if lits_vec.len() < n + 64 {
        lits_vec.resize(n + 64, 0);
    }
    ext_vec.clear();
    if ext_vec.capacity() < max_seqs * 6 {
        ext_vec.reserve(max_seqs * 6);
    }

    let dict4 = dict4_vec.as_mut_ptr();
    let dict7 = dict7_vec.as_mut_ptr();
    let tok_ptr = tokens_vec.as_mut_ptr();
    let off_ptr = offsets_vec.as_mut_ptr().cast::<u16>();
    let lit_ptr = lits_vec.as_mut_ptr();
    let ext_ptr = ext_vec.as_mut_ptr();
    let mut seq_count = 0usize;
    let mut lit_pos = 0usize;
    let mut ext_pos = 0usize;

    const { assert!(HASH_LEN >= 4 && HASH_LEN <= 6) };
    let m4: u64 = u64::MAX >> (64 - 8 * HASH_LEN);
    let hsh: u32 = 64 - 8 * HASH_LEN;
    let rsh4: u32 = 32 + shift4;
    #[inline(always)]
    fn hash4_fn<const HL: u32>(w: u64, hsh: u32, shift4: u32, rsh4: u32) -> usize {
        if HL > 4 {
            ((w << hsh).wrapping_mul(0xCF1B_BCDC_BB55_E8A5) >> rsh4) as usize
        } else {
            ((w as u32).wrapping_mul(0x9E37_79B1) >> shift4) as usize
        }
    }
    let hash4 = |w: u64| -> usize { hash4_fn::<HASH_LEN>(w, hsh, shift4, rsh4) };
    let hash7 = |w: u64| -> usize {
        (((w << 8).wrapping_mul(0xCF1B_BCDC_BB55_E8A5)) >> shift7) as usize
    };
    #[inline(always)]
    unsafe fn match_len_from8(ptr: *const u8, ip: usize, cand: usize, n: usize) -> usize {
        let a1 = unsafe { ptr.add(ip + 8).cast::<u64>().read_unaligned().to_le() };
        let b1 = unsafe { ptr.add(cand + 8).cast::<u64>().read_unaligned().to_le() };
        let diff1 = a1 ^ b1;
        if diff1 != 0 {
            return 8 + (diff1.trailing_zeros() >> 3) as usize;
        }
        let mut ml = 16usize;
        while ip + ml + 8 <= n {
            let a = unsafe { ptr.add(ip + ml).cast::<u64>().read_unaligned().to_le() };
            let b = unsafe { ptr.add(cand + ml).cast::<u64>().read_unaligned().to_le() };
            let d = a ^ b;
            if d == 0 {
                ml += 8;
            } else {
                return ml + (d.trailing_zeros() >> 3) as usize;
            }
        }
        while ip + ml < n && unsafe { *ptr.add(ip + ml) == *ptr.add(cand + ml) } {
            ml += 1;
        }
        ml
    }

    let w0 = unsafe { ptr.cast::<u64>().read_unaligned().to_le() };
    unsafe {
        *dict4.add(hash4(w0)) = 0;
        if USE_DICT7 {
            *dict7.add(hash7(w0)) = 0;
        }
    }

    let mut anchor = 0usize;
    let mut ip = 1usize;
    let limit = n.saturating_sub(16);

    while ip < limit {
        let mut match_pos = 0usize;
        let mut best_ml = 0usize;
        let mut miss = 0usize;
        let matched_so_far = anchor - lit_pos;

        if !USE_DICT7 {
            let dict16 = dict4.cast::<u16>();
            let rsh16 = rsh4 - 1;
            let sh16 = shift4 - 1;
            let hash16 = |w: u64| -> usize { hash4_fn::<HASH_LEN>(w, hsh, sh16, rsh16) };
            while ip < limit {
                let w = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
                let h0 = hash16(w);
                let h1 = hash16(w >> 8);
                let cand0 = unsafe { *dict16.add(h0) as usize };
                let cand1 = unsafe { *dict16.add(h1) as usize };
                unsafe {
                    *dict16.add(h0) = ip as u16;
                    *dict16.add(h1) = (ip + 1) as u16;
                }
                let cw0 = unsafe { ptr.add(cand0).cast::<u64>().read_unaligned().to_le() };
                let d0 = w ^ cw0;
                if (d0 & m4) == 0 {
                    match_pos = cand0;
                    if d0 == 0 {
                        best_ml = unsafe { match_len_from8(ptr, ip, cand0, n) };
                    } else {
                        best_ml = (d0.trailing_zeros() >> 3) as usize;
                        if ip + 1 < limit {
                            let w_next =
                                unsafe { ptr.add(ip + 1).cast::<u64>().read_unaligned().to_le() };
                            let cw1 =
                                unsafe { ptr.add(cand1).cast::<u64>().read_unaligned().to_le() };
                            let d1 = w_next ^ cw1;
                            if d1 == 0 {
                                ip += 1;
                                match_pos = cand1;
                                best_ml = unsafe { match_len_from8(ptr, ip, cand1, n) };
                            } else {
                                let ml1 = (d1.trailing_zeros() >> 3) as usize;
                                if ml1 > best_ml + 1 {
                                    ip += 1;
                                    match_pos = cand1;
                                    best_ml = ml1;
                                }
                            }
                        }
                    }
                    break;
                }
                let cw1 = unsafe { ptr.add(cand1).cast::<u64>().read_unaligned().to_le() };
                let d1 = (w >> 8) ^ cw1;
                if (d1 & m4) == 0 && ip + 1 < limit {
                    ip += 1;
                    match_pos = cand1;
                    let w1 = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
                    let d1_full = w1 ^ cw1;
                    best_ml = if d1_full == 0 {
                        unsafe { match_len_from8(ptr, ip, cand1, n) }
                    } else {
                        (d1_full.trailing_zeros() >> 3) as usize
                    };
                    break;
                }
                miss += 1;
                ip += 2 + (miss >> 3);
                if (miss & 127) == 0 && ip >= 8192 && matched_so_far < 128 && ip - anchor >= 2048 {
                    return (0, 0, n);
                }
            }
        } else {
            let dict16 = dict4.cast::<u16>();
            let rsh16 = rsh4 - 1;
            let sh16 = shift4 - 1;
            let hash16 = |w: u64| -> usize { hash4_fn::<HASH_LEN>(w, hsh, sh16, rsh16) };
            while ip < limit {
                let w = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
                let h0 = hash16(w);
                let cand0 = unsafe { *dict16.add(h0) as usize };
                let h7_0 = hash7(w);
                let cw0 = unsafe { ptr.add(cand0).cast::<u64>().read_unaligned().to_le() };
                let d0 = w ^ cw0;
                if (d0 & m4) == 0 {
                    let h1 = hash16(w >> 8);
                    let cand1 = unsafe { *dict16.add(h1) as usize };
                    unsafe {
                        *dict16.add(h0) = ip as u16;
                        *dict16.add(h1) = (ip + 1) as u16;
                    }
                    match_pos = cand0;
                    if d0 == 0 {
                        best_ml = unsafe { match_len_from8(ptr, ip, cand0, n) };
                    } else {
                        best_ml = (d0.trailing_zeros() >> 3) as usize;
                        let cand7 = unsafe { *dict7.add(h7_0) as usize };
                        let cw7 = unsafe { ptr.add(cand7).cast::<u64>().read_unaligned().to_le() };
                        let d7 = w ^ cw7;
                        if (d7 & 0x00ff_ffff_ffff_ffff) == 0 {
                            let ml7 = if d7 == 0 {
                                unsafe { match_len_from8(ptr, ip, cand7, n) }
                            } else {
                                7
                            };
                            if ml7 > best_ml {
                                best_ml = ml7;
                                match_pos = cand7;
                            }
                        }
                        if best_ml < 8 && ip + 1 < limit {
                            let w_next =
                                unsafe { ptr.add(ip + 1).cast::<u64>().read_unaligned().to_le() };
                            let cw1 =
                                unsafe { ptr.add(cand1).cast::<u64>().read_unaligned().to_le() };
                            if w_next == cw1 {
                                ip += 1;
                                match_pos = cand1;
                                best_ml = unsafe { match_len_from8(ptr, ip, cand1, n) };
                            }
                        }
                    }
                    unsafe {
                        *dict7.add(h7_0) = ip as u16;
                    }
                    break;
                } else if ((w ^ (w >> 8)) as u32) == 0 && ip + 1 < limit {
                    unsafe {
                        *dict16.add(h0) = ip as u16;
                        *dict7.add(h7_0) = ip as u16;
                    }
                    match_pos = ip;
                    ip += 1;
                    let w1 = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
                    let d_rle = w ^ w1;
                    best_ml = if d_rle == 0 {
                        unsafe { match_len_from8(ptr, ip, match_pos, n) }
                    } else {
                        (d_rle.trailing_zeros() >> 3) as usize
                    };
                    break;
                }
                unsafe {
                    *dict16.add(h0) = ip as u16;
                    if miss < 4 {
                        let h1 = hash16(w >> 8);
                        *dict16.add(h1) = (ip + 1) as u16;
                    }
                    *dict7.add(h7_0) = ip as u16;
                }
                miss += 1;
                ip += 2 + (miss >> 3);
                if (miss & 127) == 0 && ip >= 8192 && matched_so_far < 128 && ip - anchor >= 2048 {
                    return (0, 0, n);
                }
            }
        }
        if ip >= limit {
            break;
        }

        let next_ip = ip + best_ml;

        if match_pos >= 4 && ip >= 4 {
            let a = unsafe { ptr.add(ip - 4).cast::<u32>().read_unaligned().to_le() };
            let b = unsafe { ptr.add(match_pos - 4).cast::<u32>().read_unaligned().to_le() };
            let back = (((a ^ b).leading_zeros() >> 3) as usize).min(ip - anchor);
            ip -= back;
            match_pos -= back;
        } else {
            while ip > anchor
                && match_pos > 0
                && unsafe { *ptr.add(ip - 1) == *ptr.add(match_pos - 1) }
            {
                ip -= 1;
                match_pos -= 1;
            }
        }

        let lit_len = ip - anchor;
        let offset = ip - match_pos;
        let ml = next_ip - ip;

        if next_ip < limit {
            unsafe {
                let d16 = dict4.cast::<u16>();
                if !USE_DICT7 && ml >= 7 {
                    let p2 = ip + 2;
                    let pw = ptr.add(p2).cast::<u64>().read_unaligned().to_le();
                    *d16.add(hash4_fn::<HASH_LEN>(pw, hsh, shift4 - 1, rsh4 - 1)) = p2 as u16;
                }
                let tail_p = next_ip - 2;
                let tw = ptr.add(tail_p).cast::<u64>().read_unaligned().to_le();
                let th0 = hash4_fn::<HASH_LEN>(tw, hsh, shift4 - 1, rsh4 - 1);
                let th1 = hash4_fn::<HASH_LEN>(tw >> 8, hsh, shift4 - 1, rsh4 - 1);
                *d16.add(th0) = tail_p as u16;
                *d16.add(th1) = (tail_p + 1) as u16;
                if USE_DICT7 {
                    *dict7.add(hash7(tw)) = tail_p as u16;
                    *dict7.add(hash7(tw >> 8)) = (tail_p + 1) as u16;
                }
            }
        }

        unsafe {
            if lit_len <= 32 && anchor + 32 <= n {
                let lsrc = ptr.add(anchor);
                let ldst = lit_ptr.add(lit_pos);
                ldst.cast::<u128>()
                    .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                ldst.add(16)
                    .cast::<u128>()
                    .write_unaligned(lsrc.add(16).cast::<u128>().read_unaligned());
                lit_pos += lit_len;
            } else if lit_len > 0 {
                std::ptr::copy_nonoverlapping(ptr.add(anchor), lit_ptr.add(lit_pos), lit_len);
                lit_pos += lit_len;
            }
        }

        let lit_code = lit_len.min(15) as u8;
        let ml_minus_4 = ml - 4;
        let ml_code = ml_minus_4.min(15) as u8;
        let tok = lit_code | (ml_code << 4);
        unsafe {
            *tok_ptr.add(seq_count) = tok;
            off_ptr
                .add(seq_count)
                .write_unaligned((offset as u16).to_le());
        }
        seq_count += 1;

        if (lit_len < 270) & (ml_minus_4 < 270) {
            unsafe {
                *ext_ptr.add(ext_pos) = lit_len.wrapping_sub(15) as u8;
                ext_pos += (lit_len >= 15) as usize;
                *ext_ptr.add(ext_pos) = ml_minus_4.wrapping_sub(15) as u8;
                ext_pos += (ml_minus_4 >= 15) as usize;
            }
        } else {
            if lit_code == 15 {
                let rem = lit_len - 15;
                unsafe {
                    if rem < 255 {
                        *ext_ptr.add(ext_pos) = rem as u8;
                        ext_pos += 1;
                    } else {
                        *ext_ptr.add(ext_pos) = 255;
                        ext_ptr
                            .add(ext_pos + 1)
                            .cast::<u16>()
                            .write_unaligned(((rem - 255) as u16).to_le());
                        ext_pos += 3;
                    }
                }
            }
            if ml_code == 15 {
                let rem = ml_minus_4 - 15;
                unsafe {
                    if rem < 255 {
                        *ext_ptr.add(ext_pos) = rem as u8;
                        ext_pos += 1;
                    } else {
                        *ext_ptr.add(ext_pos) = 255;
                        ext_ptr
                            .add(ext_pos + 1)
                            .cast::<u16>()
                            .write_unaligned(((rem - 255) as u16).to_le());
                        ext_pos += 3;
                    }
                }
            }
        }

        ip = next_ip;
        anchor = ip;
    }

    unsafe {
        ext_vec.set_len(ext_pos);
    }
    if seq_count > 0 && anchor < n {
        let rem_lits = n - anchor;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(anchor), lit_ptr.add(lit_pos), rem_lits);
        }
        lit_pos += rem_lits;
    }
    (seq_count, seq_count, lit_pos)
}

