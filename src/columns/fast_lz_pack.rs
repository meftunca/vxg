/// Emit an entropy-only `FAST_LZ_PACKET` (`seq_count == 0`): the smaller of the NEON palette
/// stream (`comp_len > 0`, palette bytes) and a Huffman blob (`comp_len == 0`, ULEB length, blob).
fn emit_entropy_only(
    data: &[u8],
    tmp: &mut Vec<u8>,
    huf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> Option<()> {
    let n = data.len();
    let mut best_kind = 0u8;
    let mut best_len = n;
    huf.clear();
    if huf_encode(data, huf, best_len) && 1 + uleb_len(huf.len()) + huf.len() < best_len {
        best_kind = 2;
        best_len = 1 + uleb_len(huf.len()) + huf.len();
    } else if n >= 256 && pack_palette_stream(data, tmp) {
        best_kind = 1;
        best_len = uleb_len(tmp.len()) + tmp.len();
    }
    if best_kind == 0 || 1 + uleb_len(n) + 1 + best_len + 16 + (n >> 6) >= n {
        return None;
    }
    out.push(FAST_LZ_PACKET);
    put_uleb(out, n);
    put_uleb(out, 0);
    if best_kind == 1 {
        put_uleb(out, tmp.len());
        out.extend_from_slice(tmp);
    } else {
        put_uleb(out, 0);
        put_uleb(out, huf.len());
        out.extend_from_slice(huf);
    }
    Some(())
}

/// Fast 4-stream separated LZ77 + repeat-offset + NEON palette block compressor (`FAST_LZ_PACKET = 3`)
/// for non-schematic text, source code, mixed blocks, Base64 blobs, and general binary data.
fn pack_fast_lz_into(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn pack_fast_lz_into_x86_v3(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
            pack_fast_lz_into_impl(data, out)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
        {
            return unsafe { pack_fast_lz_into_x86_v3(data, out) };
        }
    }
    pack_fast_lz_into_impl(data, out)
}

#[inline(always)]
fn pack_fast_lz_into_impl(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
    let n = data.len();
    if !(64..=65536).contains(&n) {
        return None;
    }
    let mut is_pure_base64 = false;
    if n >= 256 {
        const B64_M0: u64 = (1 << b'\t')
            | (1 << b'\n')
            | (1 << b'\r')
            | (1 << b'+')
            | (1 << b'-')
            | (0x7ff << b'/')
            | (1 << b'=');
        const B64_M1: u64 = (0x03ff_ffff << (b'A' - 64))
            | (1 << (b'_' - 64))
            | (0x03ff_ffff << (b'a' - 64));
        let mut bmap = [0u64; 4];
        for &b in &data[..256] {
            bmap[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let u0 = bmap[0].count_ones()
            + bmap[1].count_ones()
            + bmap[2].count_ones()
            + bmap[3].count_ones();
        if u0 >= 196 {
            return None;
        }
        if n >= 4096
            && (52..=66).contains(&u0)
            && (bmap[0] & !B64_M0) == 0
            && (bmap[1] & !B64_M1) == 0
            && (bmap[2] | bmap[3]) == 0
        {
            let mid = n / 2;
            let mut bmap_m = [0u64; 4];
            for &b in &data[mid..mid + 256] {
                bmap_m[(b >> 6) as usize] |= 1u64 << (b & 63);
            }
            for &b in &data[n - 256..] {
                bmap_m[(b >> 6) as usize] |= 1u64 << (b & 63);
            }
            let um = bmap_m[0].count_ones()
                + bmap_m[1].count_ones()
                + bmap_m[2].count_ones()
                + bmap_m[3].count_ones();
            if (52..=66).contains(&um)
                && (bmap_m[0] & !B64_M0) == 0
                && (bmap_m[1] & !B64_M1) == 0
                && (bmap_m[2] | bmap_m[3]) == 0
            {
                is_pure_base64 = true;
            }
        }
    }

    STRIDE_SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        if is_pure_base64 {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        // Text-like blocks (almost no control/high bytes in a 256-byte probe) use a
        // 5-byte primary hash; binary / pixel data prefers 6 bytes.
        let probe = &data[..n.min(256)];
        let binary_bytes = probe
            .iter()
            .filter(|&&b| b >= 0x80 || b < 0x09 || (b > 0x0d && b < 0x20))
            .count();
        let hash_len = if binary_bytes * 32 <= probe.len() { 5 } else { 6 };
        let use_dict7 = n >= 2048 && hash_len != 5;
        let (seq_count, off_count, lit_total_len) = pack_fast_lz_core(
            data,
            use_dict7,
            &mut sc.lz_dict4,
            &mut sc.lz_dict7,
            &mut sc.lz_tokens,
            &mut sc.lz_offsets,
            &mut sc.lz_ext,
            &mut sc.lz_lits,
            hash_len,
        );

        // Entropy-only block (palette or Huffman) when LZ found no matches:
        if seq_count == 0 {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        let offs_byte_len = off_count * 2;
        let ext_len = sc.lz_ext.len();
        let meta_raw_len = seq_count + offs_byte_len + ext_len;

        let mut flags = 0u8;
        let mut meta_size = meta_raw_len;

        // Split-stream Huffman meta (flag 8): tokens | offset lo | offset hi | ext.
        if hash_len == 5 && seq_count >= 160 {
            sc.lz_meta_huf.clear();
            put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_tokens[..seq_count]);
            if sc.lz_sub_lits.len() < 2 * off_count {
                sc.lz_sub_lits.resize(2 * off_count, 0);
            }
            let (lo_dst, hi_dst) = sc.lz_sub_lits[..2 * off_count].split_at_mut(off_count);
            let mut oi = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let src_p = sc.lz_offsets.as_ptr();
                let lo_p = lo_dst.as_mut_ptr();
                let hi_p = hi_dst.as_mut_ptr();
                while oi + 16 <= off_count {
                    let pair = vld2q_u8(src_p.add(oi * 2));
                    vst1q_u8(lo_p.add(oi), pair.0);
                    vst1q_u8(hi_p.add(oi), pair.1);
                    oi += 16;
                }
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use core::arch::x86_64::*;
                let src_p = sc.lz_offsets.as_ptr();
                let lo_p = lo_dst.as_mut_ptr();
                let hi_p = hi_dst.as_mut_ptr();
                let mask = _mm_set1_epi16(0x00ff);
                while oi + 16 <= off_count {
                    let v0 = _mm_loadu_si128(src_p.add(oi * 2).cast::<__m128i>());
                    let v1 = _mm_loadu_si128(src_p.add(oi * 2 + 16).cast::<__m128i>());
                    let lo = _mm_packus_epi16(_mm_and_si128(v0, mask), _mm_and_si128(v1, mask));
                    let hi = _mm_packus_epi16(_mm_srli_epi16(v0, 8), _mm_srli_epi16(v1, 8));
                    _mm_storeu_si128(lo_p.add(oi).cast::<__m128i>(), lo);
                    _mm_storeu_si128(hi_p.add(oi).cast::<__m128i>(), hi);
                    oi += 16;
                }
            }
            while oi < off_count {
                lo_dst[oi] = sc.lz_offsets[2 * oi];
                hi_dst[oi] = sc.lz_offsets[2 * oi + 1];
                oi += 1;
            }
            if hash_len == 5 {
                put_uleb(&mut sc.lz_meta_huf, off_count);
                sc.lz_meta_huf.extend_from_slice(&sc.lz_sub_lits[..off_count]);
                put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_sub_lits[off_count..2 * off_count]);
                put_uleb(&mut sc.lz_meta_huf, ext_len);
                sc.lz_meta_huf.extend_from_slice(&sc.lz_ext);
            } else {
                put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_sub_lits[..off_count]);
                put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_sub_lits[off_count..2 * off_count]);
                put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_ext);
            }
            if sc.lz_meta_huf.len() + 8 < meta_size {
                meta_size = sc.lz_meta_huf.len();
                flags |= 8;
            }
        }

        // Only try secondary LZ on meta stream (flag 1) when split-stream Huffman wasn't used:
        if (flags & 8) == 0
            && n >= 1024
            && meta_raw_len >= 256
            && seq_count >= 48
            && meta_raw_len + lit_total_len + (n >> 5) < n
        {
            sc.lz_meta_raw.clear();
            sc.lz_meta_raw.reserve(meta_raw_len);
            sc.lz_meta_raw.extend_from_slice(&sc.lz_tokens[..seq_count]);
            sc.lz_meta_raw
                .extend_from_slice(&sc.lz_offsets[..offs_byte_len]);
            sc.lz_meta_raw.extend_from_slice(&sc.lz_ext);

            let (m_seq, m_off, m_lit) = pack_fast_lz_core(
                &sc.lz_meta_raw,
                false,
                &mut sc.lz_dict4,
                &mut sc.lz_dict7,
                &mut sc.lz_sub_tokens,
                &mut sc.lz_sub_offsets,
                &mut sc.lz_sub_ext,
                &mut sc.lz_sub_lits,
                4,
            );
            if m_seq > 0 {
                let m_offs_bytes = m_off * 2;
                let m_ext_len = sc.lz_sub_ext.len();
                let m_total = uleb_len(m_seq)
                    + uleb_len(m_lit)
                    + uleb_len(m_off)
                    + uleb_len(m_ext_len)
                    + m_seq
                    + m_offs_bytes
                    + m_ext_len
                    + m_lit;
                let m_framed = uleb_len(m_total) + m_total;
                if m_total + 16 + (meta_raw_len >> 5) < meta_raw_len && m_framed + 8 < meta_size {
                    sc.lz_meta_comp.clear();
                    sc.lz_meta_comp.reserve(m_total);
                    put_uleb(&mut sc.lz_meta_comp, m_seq);
                    put_uleb(&mut sc.lz_meta_comp, m_lit);
                    put_uleb(&mut sc.lz_meta_comp, m_off);
                    put_uleb(&mut sc.lz_meta_comp, m_ext_len);
                    sc.lz_meta_comp
                        .extend_from_slice(&sc.lz_sub_tokens[..m_seq]);
                    sc.lz_meta_comp
                        .extend_from_slice(&sc.lz_sub_offsets[..m_offs_bytes]);
                    sc.lz_meta_comp.extend_from_slice(&sc.lz_sub_ext);
                    sc.lz_meta_comp.extend_from_slice(&sc.lz_sub_lits[..m_lit]);
                    meta_size = m_framed;
                    flags = (flags & !8) | 1;
                }
            }
        }

        // Literals: try Huffman (flag 4) first, falling back to NEON palette (flag 2) only if Huffman didn't compress.
        let mut lits_size = lit_total_len;
        if lit_total_len >= HUF_MIN {
            sc.lz_lits_huf.clear();
            if huf_encode(&sc.lz_lits[..lit_total_len], &mut sc.lz_lits_huf, lits_size)
                && uleb_len(sc.lz_lits_huf.len()) + sc.lz_lits_huf.len() < lits_size
            {
                flags |= 4;
                lits_size = uleb_len(sc.lz_lits_huf.len()) + sc.lz_lits_huf.len();
            }
        }
        if (flags & 4) == 0
            && lit_total_len >= 192
            && meta_size + lit_total_len + (n >> 6) < n
            && pack_palette_stream(&sc.lz_lits[..lit_total_len], &mut sc.lz_lits_comp)
            && uleb_len(sc.lz_lits_comp.len()) + sc.lz_lits_comp.len() < lits_size
        {
            flags |= 2;
            lits_size = uleb_len(sc.lz_lits_comp.len()) + sc.lz_lits_comp.len();
        }
        let lit_hdr = if flags != 0 {
            n + 1 + ((flags as usize) << 16) + lit_total_len
        } else {
            lit_total_len
        };

        let total_size = 1
            + uleb_len(n)
            + uleb_len(seq_count)
            + uleb_len(lit_hdr)
            + uleb_len(off_count)
            + uleb_len(ext_len)
            + meta_size
            + lits_size;
        if total_size + 16 + (n >> 6) >= n {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        out.reserve(total_size);
        out.push(FAST_LZ_PACKET);
        put_uleb(out, n);
        put_uleb(out, seq_count);
        put_uleb(out, lit_hdr);
        put_uleb(out, off_count);
        put_uleb(out, ext_len);
        if (flags & 1) != 0 {
            put_uleb(out, sc.lz_meta_comp.len());
            out.extend_from_slice(&sc.lz_meta_comp);
        } else if (flags & 8) != 0 {
            out.extend_from_slice(&sc.lz_meta_huf);
        } else {
            out.extend_from_slice(&sc.lz_tokens[..seq_count]);
            out.extend_from_slice(&sc.lz_offsets[..offs_byte_len]);
            out.extend_from_slice(&sc.lz_ext);
        }
        if (flags & 2) != 0 {
            put_uleb(out, sc.lz_lits_comp.len());
            out.extend_from_slice(&sc.lz_lits_comp);
        } else if (flags & 4) != 0 {
            put_uleb(out, sc.lz_lits_huf.len());
            out.extend_from_slice(&sc.lz_lits_huf);
        } else {
            out.extend_from_slice(&sc.lz_lits[..lit_total_len]);
        }
        Some(())
    })
}

