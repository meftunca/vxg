/// Detect and pack fixed-stride pixel / numeric / binary records (`stride in {2, 3, 4, 8}`)
/// using byte-plane transpose + optional Subtract-Green color decorrelation + horizontal delta.
///
/// Format of `STRIDE_PACKET` (`selector == 2`):
/// - `raw_len: ULEB`
/// - `prefix_len: ULEB`, followed by `prefix_bytes[..prefix_len]`
/// - `stride: u8` (`2`, `3`, `4`, or `8`), `mode: u8` (`1` = 2-level 64-bit word zero-mask,
///   `2` = per-plane `FAST_LZ`, `3` = Subtract-Green + per-plane `FAST_LZ`)
fn pack_stride_into(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
    let n = data.len();
    if n < 256 || n > MAX_BLOCK_SIZE {
        return None;
    }
    // Quick check: detect BMP header (`BM` + data_offset at 10..14 and bpp at 28..30)
    // or test strides 2, 3, 4, and 8 on prefix_len = 0.
    let (prefix_len, stride) = if n >= 54 && data[0] == b'B' && data[1] == b'M' {
        let off = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
        let bpp = u16::from_le_bytes(data[28..30].try_into().unwrap());
        let s = match bpp {
            16 => 2usize,
            24 => 3usize,
            32 => 4usize,
            _ => 4usize,
        };
        if off <= 1024 && off + s * 64 <= n {
            (off, s)
        } else {
            (0usize, 4usize)
        }
    } else {
        // Quick 32-byte midpoint reject before scanning four 128-byte windows:
        let mid = (n / 2).saturating_sub(16);
        let quick = &data[mid..(mid + 32).min(n)];
        let mut q2 = 0usize;
        let mut q3 = 0usize;
        let mut q4 = 0usize;
        let mut q8 = 0usize;
        let mut q1 = 0usize;
        for i in 8..quick.len() {
            q2 += (quick[i] == quick[i - 2]) as usize;
            q3 += (quick[i] == quick[i - 3]) as usize;
            q4 += (quick[i] == quick[i - 4]) as usize;
            q8 += (quick[i] == quick[i - 8]) as usize;
            q1 += (quick[i] == quick[i - 1]) as usize;
        }
        let max_q = q2.max(q3).max(q4).max(q8);
        if max_q < 7 || max_q <= q1 + 2 {
            return None;
        }
        // Sample four 128-byte windows across the block at strides 2, 3, 4, and 8:
        let mut z2 = 0usize;
        let mut z3 = 0usize;
        let mut z4 = 0usize;
        let mut z8 = 0usize;
        let mut z1 = 0usize;
        let mut total = 0usize;
        let win = 128.min(n);
        for q in 0..4 {
            let start = ((n - win) * q) / 3;
            let sample = &data[start..start + win];
            for i in 8..sample.len() {
                z2 += (sample[i] == sample[i - 2]) as usize;
                z3 += (sample[i] == sample[i - 3]) as usize;
                z4 += (sample[i] == sample[i - 4]) as usize;
                z8 += (sample[i] == sample[i - 8]) as usize;
                z1 += (sample[i] == sample[i - 1]) as usize;
            }
            total += sample.len() - 8;
        }
        let max_z = z2.max(z3).max(z4).max(z8);
        // Require at least 45% byte-wise horizontal repetition and clear dominance over 1-byte RLE:
        if max_z * 100 < total * 45 || max_z <= z1 + total / 8 {
            return None;
        }
        let (even_s, even_z) = if z8 > z4 + total / 10 && z8 > z2 + total / 10 {
            (8usize, z8)
        } else if z2 > z4 + 6 {
            (2usize, z2)
        } else {
            (4usize, z4)
        };
        if z3 > even_z + 8 {
            (0usize, 3usize)
        } else {
            (0usize, even_s)
        }
    };

    let body = &data[prefix_len..];
    let count = body.len() / stride;
    if count < 64 {
        return None;
    }
    let tail = &body[count * stride..];

    // Sample 256 pixels across 4 windows to decide whether Subtract-Green reduces L1 residual energy
    // (only applicable to 3-channel RGB and 4-channel RGBA):
    let use_sub_green = if stride == 3 || stride == 4 {
        let mut raw_cost = 0usize;
        let mut sg_cost = 0usize;
        let swin = 64.min(count);
        for q in 0..4 {
            let base_px = ((count - swin) * q) / 3;
            for i in (base_px + 1)..(base_px + swin) {
                let o = i * stride;
                let po = (i - 1) * stride;
                let dr = body[o].wrapping_sub(body[po]);
                let dg = body[o + 1].wrapping_sub(body[po + 1]);
                let db = body[o + 2].wrapping_sub(body[po + 2]);
                let dr_sg = dr.wrapping_sub(dg);
                let db_sg = db.wrapping_sub(dg);
                raw_cost += (dr != 0) as usize
                    + (db != 0) as usize
                    + (dr.wrapping_add(1) > 2) as usize
                    + (db.wrapping_add(1) > 2) as usize;
                sg_cost += (dr_sg != 0) as usize
                    + (db_sg != 0) as usize
                    + (dr_sg.wrapping_add(1) > 2) as usize
                    + (db_sg.wrapping_add(1) > 2) as usize;
            }
        }
        sg_cost + 8 < raw_cost
    } else {
        false
    };

    let start_len = out.len();
    let mode_byte = if use_sub_green { 3u8 } else { 2u8 };
    out.push(STRIDE_PACKET);
    put_uleb(out, n);
    put_uleb(out, prefix_len);
    out.extend_from_slice(&data[..prefix_len]);
    out.push(stride as u8);
    out.push(mode_byte);

    let ok = STRIDE_COMBINED.with(|cell| {
        let mut comb = cell.borrow_mut();
        let total_cap = stride * count + 64;
        if comb.len() < total_cap {
            comb.resize(total_cap, 0);
        }

        let mut first_bytes = [0u8; 8];
        let mut all_zero = [false; 8];
        let mut any_nz_words_arr = [0u64; 8];
        let mut has_any_nz_words = false;
        let words = count.div_ceil(64);
        #[allow(unused_variables)]
        let body_ptr = body.as_ptr();
        #[allow(unused_variables)]
        let comb_ptr = comb.as_mut_ptr();

        #[cfg(target_arch = "aarch64")]
        extract_stride_planes_neon(
            body,
            &mut comb,
            stride,
            count,
            words,
            use_sub_green,
            &mut first_bytes,
            &mut all_zero,
            &mut any_nz_words_arr,
            &mut has_any_nz_words,
        );
        #[cfg(target_arch = "x86_64")]
        extract_stride_planes_sse2(
            body,
            &mut comb,
            stride,
            count,
            words,
            use_sub_green,
            &mut first_bytes,
            &mut all_zero,
            &mut any_nz_words_arr,
            &mut has_any_nz_words,
        );
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for p in 0..stride {
                let f = body[p];
                first_bytes[p] = if use_sub_green && (p == 0 || p == 2) {
                    f.wrapping_sub(body[1])
                } else {
                    f
                };
                let mut prev = first_bytes[p];
                let mut s_or = 0u8;
                let p_slice = &mut comb[p * count..(p + 1) * count];
                p_slice[0] = 0;
                for idx in 1..count {
                    let cur = if use_sub_green && (p == 0 || p == 2) {
                        body[idx * stride + p].wrapping_sub(body[idx * stride + 1])
                    } else {
                        body[idx * stride + p]
                    };
                    let d = cur.wrapping_sub(prev);
                    p_slice[idx] = d;
                    s_or |= d;
                    prev = cur;
                }
                all_zero[p] = s_or == 0;
            }
        }

        encode_stride_planes(
            out,
            &comb[..stride * count],
            &first_bytes,
            &all_zero,
            if has_any_nz_words {
                Some(&any_nz_words_arr)
            } else {
                None
            },
            stride,
            count,
            start_len + n - tail.len(),
        )
    });

    if ok {
        out.extend_from_slice(tail);
        if out.len() - start_len < n {
            return Some(());
        }
    }
    out.truncate(start_len);
    None
}

