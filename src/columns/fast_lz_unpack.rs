fn unpack_fast_lz_core(
    seq_count: usize,
    off_count: usize,
    toks: &[u8],
    offs_bytes: &[u8],
    ext_bytes: &[u8],
    lits: &[u8],
    out: &mut [u8],
) -> Result<()> {
    if off_count != seq_count {
        return Err(invalid("fast lz offset underflow"));
    }
    let raw_len = out.len();
    let lit_total_len = lits.len();
    let ext_len = ext_bytes.len();
    let out_ptr = out.as_mut_ptr();
    let lit_ptr = lits.as_ptr();
    let tok_ptr = toks.as_ptr();
    let off_ptr = offs_bytes.as_ptr().cast::<u16>();
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut ext_pos = 0usize;

    for seq_idx in 0..seq_count {
        let tok = unsafe { *tok_ptr.add(seq_idx) };
        let mut lit_len = (tok & 0x0f) as usize;
        let ml_code = (tok >> 4) as usize;
        let offset = u16::from_le(unsafe { off_ptr.add(seq_idx).read_unaligned() }) as usize;
        if offset == 0 {
            return Err(invalid("fast lz zero offset"));
        }

        if (lit_len < 15)
            & (dst_pos + 48 <= raw_len)
            & (lit_pos + 16 <= lit_total_len)
            & (offset <= dst_pos + lit_len)
        {
            if offset >= 16 && ext_pos < ext_len {
                let is_ml15 = (ml_code == 15) as usize;
                let raw_e = unsafe { *ext_bytes.get_unchecked(ext_pos) } as usize;
                let e = if is_ml15 != 0 { raw_e } else { 0 };
                if e <= 13 {
                    ext_pos += is_ml15;
                    let match_len = ml_code + 4 + e;
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let msrc = out_ptr.add(dst_pos - offset);
                        let mdst = out_ptr.add(dst_pos);
                        mdst.cast::<u128>()
                            .write_unaligned(msrc.cast::<u128>().read_unaligned());
                        mdst.add(16)
                            .cast::<u128>()
                            .write_unaligned(msrc.add(16).cast::<u128>().read_unaligned());
                        dst_pos += match_len;
                    }
                    continue;
                }
            }
            if ml_code < 15 {
                let match_len = ml_code + 4;
                if offset >= 16 {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let msrc = out_ptr.add(dst_pos - offset);
                        let mdst = out_ptr.add(dst_pos);
                        mdst.cast::<u128>()
                            .write_unaligned(msrc.cast::<u128>().read_unaligned());
                        mdst.add(16)
                            .cast::<u128>()
                            .write_unaligned(msrc.add(16).cast::<u128>().read_unaligned());
                        dst_pos += match_len;
                    }
                    continue;
                } else if offset >= match_len {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let msrc = out_ptr.add(dst_pos - offset);
                        let mdst = out_ptr.add(dst_pos);
                        mdst.cast::<u128>()
                            .write_unaligned(msrc.cast::<u128>().read_unaligned());
                        dst_pos += match_len;
                    }
                    continue;
                }
            }
        }

        if lit_len == 15 {
            if ext_pos >= ext_len {
                return Err(invalid("fast lz ext underflow"));
            }
            let e = ext_bytes[ext_pos] as usize;
            ext_pos += 1;
            if e < 255 {
                lit_len += e;
            } else {
                if ext_pos + 2 > ext_len {
                    return Err(invalid("fast lz ext underflow"));
                }
                let extra =
                    u16::from_le_bytes([ext_bytes[ext_pos], ext_bytes[ext_pos + 1]]) as usize;
                ext_pos += 2;
                lit_len += 255 + extra;
            }
        }
        let mut match_len = ml_code + 4;
        if ml_code == 15 {
            if ext_pos >= ext_len {
                return Err(invalid("fast lz ext underflow"));
            }
            let e = ext_bytes[ext_pos] as usize;
            ext_pos += 1;
            if e < 255 {
                match_len += e;
            } else {
                if ext_pos + 2 > ext_len {
                    return Err(invalid("fast lz ext underflow"));
                }
                let extra =
                    u16::from_le_bytes([ext_bytes[ext_pos], ext_bytes[ext_pos + 1]]) as usize;
                ext_pos += 2;
                match_len += 255 + extra;
            }
        }
        if dst_pos + lit_len + match_len + 32 <= raw_len
            && lit_pos + lit_len + 16 <= lit_total_len
            && offset <= dst_pos + lit_len
        {
            unsafe {
                let lsrc = lit_ptr.add(lit_pos);
                let ldst = out_ptr.add(dst_pos);
                ldst.cast::<u128>()
                    .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                if lit_len > 16 {
                    // Slack verified above: dst has >= 32 spare bytes, lits >= 16.
                    let mut i = 16usize;
                    while i < lit_len {
                        ldst.add(i)
                            .cast::<u128>()
                            .write_unaligned(lsrc.add(i).cast::<u128>().read_unaligned());
                        i += 16;
                    }
                }
                lit_pos += lit_len;
                dst_pos += lit_len;

                let msrc = out_ptr.add(dst_pos - offset);
                let mdst = out_ptr.add(dst_pos);
                if offset >= 16 || match_len <= offset {
                    mdst.cast::<u128>()
                        .write_unaligned(msrc.cast::<u128>().read_unaligned());
                    if match_len > 16 {
                        mdst.add(16)
                            .cast::<u128>()
                            .write_unaligned(msrc.add(16).cast::<u128>().read_unaligned());
                        if match_len > 32 {
                            let mut i = 32usize;
                            while i < match_len {
                                mdst.add(i)
                                    .cast::<u128>()
                                    .write_unaligned(msrc.add(i).cast::<u128>().read_unaligned());
                                i += 16;
                            }
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 1 {
                    let b = *msrc;
                    let v = u128::from_le_bytes([b; 16]);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 4 {
                    let p = msrc.cast::<u32>().read_unaligned() as u128;
                    let v = p | (p << 32) | (p << 64) | (p << 96);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 8 {
                    let p = msrc.cast::<u64>().read_unaligned() as u128;
                    let v = p | (p << 64);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 2 {
                    let p = msrc.cast::<u16>().read_unaligned() as u128;
                    let p32 = p | (p << 16);
                    let v = p32 | (p32 << 32) | (p32 << 64) | (p32 << 96);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                }
            }
            // Remaining overlap case (offset < 16, offset < match_len): periodic 16-byte copies.
            // Each store's first `s` bytes are valid; trailing bytes are rewritten by the next
            // store or fall inside the verified 32-byte slack.
            unsafe {
                let mdst = out_ptr.add(dst_pos);
                let step = (16 / offset) * offset;
                let mut c = 0usize;
                while c < match_len {
                    let s = if c + offset >= step { step } else { offset };
                    let v = mdst.add(c).sub(s).cast::<u128>().read_unaligned();
                    mdst.add(c).cast::<u128>().write_unaligned(v);
                    c += s;
                }
            }
            dst_pos += match_len;
            continue;
        }

        if lit_pos + lit_len > lit_total_len || dst_pos + lit_len + match_len > raw_len {
            return Err(invalid("fast lz bounds"));
        }
        if lit_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), lit_len);
            }
            lit_pos += lit_len;
            dst_pos += lit_len;
        }
        if offset > dst_pos {
            return Err(invalid("fast lz match before start"));
        }
        if offset >= match_len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_ptr.add(dst_pos - offset),
                    out_ptr.add(dst_pos),
                    match_len,
                );
            }
        } else if offset == 1 {
            let b = unsafe { *out_ptr.add(dst_pos - 1) };
            unsafe {
                std::ptr::write_bytes(out_ptr.add(dst_pos), b, match_len);
            }
        } else {
            let mut copied = 0usize;
            while copied < match_len {
                let avail = offset + copied;
                let take = avail.min(match_len - copied);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_ptr.add(dst_pos - offset),
                        out_ptr.add(dst_pos + copied),
                        take,
                    );
                }
                copied += take;
            }
        }
        dst_pos += match_len;
    }

    let rem_lits = lit_total_len - lit_pos;
    if ext_pos != ext_len || dst_pos + rem_lits != raw_len {
        return Err(invalid("fast lz final consumption"));
    }
    if rem_lits > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), rem_lits);
        }
    }
    Ok(())
}

fn unpack_fast_lz_into(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
    let raw_len = cursor.uleb()?;
    if raw_len == 0 || raw_len > 65536 || raw_len > MAX_BLOCK_SIZE || raw_len != out.len() {
        return Err(invalid("fast lz raw length"));
    }
    let seq_count = cursor.uleb()?;
    if seq_count == 0 {
        let comp_len = cursor.uleb()?;
        if comp_len == 0 {
            let huf_len = cursor.uleb()?;
            let huf = cursor.take(huf_len)?;
            if !cursor.done() {
                return Err(invalid("fast lz trailing bytes"));
            }
            return huf_decode(huf, out);
        }
        let comp = cursor.take(comp_len)?;
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }
        return unpack_palette_stream(comp, out);
    }

    let lit_hdr = cursor.uleb()?;
    let (flags, lit_total_len) = if lit_hdr <= raw_len {
        (0usize, lit_hdr)
    } else {
        let rem = lit_hdr - (raw_len + 1);
        (rem >> 16, rem & 0xffff)
    };
    let off_count = cursor.uleb()?;
    let ext_len = cursor.uleb()?;
    if flags > 15
        || (flags & 9) == 9
        || (flags & 6) == 6
        || seq_count > raw_len / 4 + 16
        || off_count > seq_count
        || lit_total_len > raw_len
        || ext_len > seq_count * 6
    {
        return Err(invalid("fast lz header counts"));
    }

    if flags == 0 {
        let toks = cursor.take(seq_count)?;
        let offs_bytes = cursor.take(off_count * 2)?;
        let ext_bytes = cursor.take(ext_len)?;
        let lits = cursor.take(lit_total_len)?;
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }
        return unpack_fast_lz_core(
            seq_count,
            off_count,
            toks,
            offs_bytes,
            ext_bytes,
            lits,
            out,
        );
    }

    STRIDE_SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        let meta_raw_len = seq_count + off_count * 2 + ext_len;
        let (toks_slice, offs_slice, ext_slice) = if (flags & 1) != 0 {
            let meta_comp_len = cursor.uleb()?;
            let meta_comp = cursor.take(meta_comp_len)?;
            let mut mc = Cursor::new(meta_comp);
            let m_seq = mc.uleb()?;
            let m_lit = mc.uleb()?;
            let m_off = mc.uleb()?;
            let m_ext = mc.uleb()?;
            if m_seq == 0
                || m_seq > meta_raw_len / 4 + 16
                || m_off > m_seq
                || m_lit > meta_raw_len
                || m_ext > m_seq * 6
            {
                return Err(invalid("fast lz meta header"));
            }
            let m_toks = mc.take(m_seq)?;
            let m_offs = mc.take(m_off * 2)?;
            let m_exts = mc.take(m_ext)?;
            let m_lits = mc.take(m_lit)?;
            if !mc.done() {
                return Err(invalid("fast lz meta trailing bytes"));
            }
            if sc.lz_meta_raw.len() < meta_raw_len + 64 {
                sc.lz_meta_raw.resize(meta_raw_len + 64, 0);
            }
            unpack_fast_lz_core(
                m_seq,
                m_off,
                m_toks,
                m_offs,
                m_exts,
                m_lits,
                &mut sc.lz_meta_raw[..meta_raw_len],
            )?;
            let m_raw = &sc.lz_meta_raw[..meta_raw_len];
            (
                &m_raw[..seq_count],
                &m_raw[seq_count..seq_count + off_count * 2],
                &m_raw[seq_count + off_count * 2..],
            )
        } else if (flags & 8) != 0 {
            if sc.lz_tokens.len() < seq_count {
                sc.lz_tokens.resize(seq_count, 0);
            }
            if sc.lz_offsets.len() < 2 * off_count {
                sc.lz_offsets.resize(2 * off_count, 0);
            }
            if sc.lz_meta_raw.len() < ext_len {
                sc.lz_meta_raw.resize(ext_len, 0);
            }
            if sc.lz_sub_lits.len() < 2 * off_count {
                sc.lz_sub_lits.resize(2 * off_count, 0);
            }
            let toks_s = take_huf_or_raw(cursor, &mut sc.lz_tokens[..seq_count])?;
            let (lo_buf, hi_buf) = sc.lz_sub_lits[..2 * off_count].split_at_mut(off_count);
            let lo = take_huf_or_raw(cursor, lo_buf)?;
            let hi = take_huf_or_raw(cursor, hi_buf)?;
            let mut oi = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let lo_p = lo.as_ptr();
                let hi_p = hi.as_ptr();
                let dst_p = sc.lz_offsets.as_mut_ptr();
                while oi + 16 <= off_count {
                    let v_lo = vld1q_u8(lo_p.add(oi));
                    let v_hi = vld1q_u8(hi_p.add(oi));
                    vst2q_u8(dst_p.add(oi * 2), uint8x16x2_t(v_lo, v_hi));
                    oi += 16;
                }
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use core::arch::x86_64::*;
                let lo_p = lo.as_ptr();
                let hi_p = hi.as_ptr();
                let dst_p = sc.lz_offsets.as_mut_ptr();
                while oi + 16 <= off_count {
                    let v_lo = _mm_loadu_si128(lo_p.add(oi).cast::<__m128i>());
                    let v_hi = _mm_loadu_si128(hi_p.add(oi).cast::<__m128i>());
                    let il_lo = _mm_unpacklo_epi8(v_lo, v_hi);
                    let il_hi = _mm_unpackhi_epi8(v_lo, v_hi);
                    _mm_storeu_si128(dst_p.add(oi * 2).cast::<__m128i>(), il_lo);
                    _mm_storeu_si128(dst_p.add(oi * 2 + 16).cast::<__m128i>(), il_hi);
                    oi += 16;
                }
            }
            while oi < off_count {
                sc.lz_offsets[2 * oi] = lo[oi];
                sc.lz_offsets[2 * oi + 1] = hi[oi];
                oi += 1;
            }
            let ext_s = take_huf_or_raw(cursor, &mut sc.lz_meta_raw[..ext_len])?;
            (toks_s, &sc.lz_offsets[..2 * off_count], ext_s)
        } else {
            let t = cursor.take(seq_count)?;
            let o = cursor.take(off_count * 2)?;
            let e = cursor.take(ext_len)?;
            (t, o, e)
        };

        let lits_slice = if (flags & 2) != 0 {
            let lits_comp_len = cursor.uleb()?;
            let lits_comp = cursor.take(lits_comp_len)?;
            if sc.lz_lits.len() < lit_total_len + 64 {
                sc.lz_lits.resize(lit_total_len + 64, 0);
            }
            unpack_palette_stream(lits_comp, &mut sc.lz_lits[..lit_total_len])?;
            &sc.lz_lits[..lit_total_len]
        } else if (flags & 4) != 0 {
            let huf_len = cursor.uleb()?;
            let huf = cursor.take(huf_len)?;
            if sc.lz_lits.len() < lit_total_len + 64 {
                sc.lz_lits.resize(lit_total_len + 64, 0);
            }
            huf_decode(huf, &mut sc.lz_lits[..lit_total_len])?;
            &sc.lz_lits[..lit_total_len]
        } else {
            cursor.take(lit_total_len)?
        };
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }

        unpack_fast_lz_core(
            seq_count,
            off_count,
            toks_slice,
            offs_slice,
            ext_slice,
            lits_slice,
            out,
        )
    })
}

