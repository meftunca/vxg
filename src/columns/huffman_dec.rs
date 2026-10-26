#[inline(always)]
fn huf_decode_tail(
    src: &[u8],
    mut bp: usize,
    dst: *mut u8,
    mut idx: usize,
    end: usize,
    table: &[u16],
    mask: u64,
) -> bool {
    let slen = src.len();
    let sptr = src.as_ptr();
    unsafe {
        while idx < end {
            let byte_pos = bp >> 3;
            let b = if byte_pos + 8 <= slen {
                sptr.add(byte_pos).cast::<u64>().read_unaligned().to_le() >> (bp & 7)
            } else {
                let mut v = 0u64;
                let mut k = 0usize;
                while byte_pos + k < slen && k < 8 {
                    v |= (*sptr.add(byte_pos + k) as u64) << (k * 8);
                    k += 1;
                }
                v >> (bp & 7)
            };
            let e = *table.get_unchecked((b & mask) as usize);
            *dst.add(idx) = (e >> 8) as u8;
            bp += (e as u8) as usize;
            idx += 1;
        }
    }
    bp <= slen * 8 && bp + 8 > slen * 8
}

fn huf_decode(blob: &[u8], dst: &mut [u8]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn huf_decode_x86_v3(blob: &[u8], dst: &mut [u8]) -> Result<()> {
            huf_decode_impl(blob, dst)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            return unsafe { huf_decode_x86_v3(blob, dst) };
        }
    }
    huf_decode_impl(blob, dst)
}

#[inline(always)]
fn huf_decode_impl(blob: &[u8], dst: &mut [u8]) -> Result<()> {
    let n = dst.len();
    let mut cur = Cursor::new(blob);
    match cur.byte()? {
        HUF_KIND_RLE => {
            let sym = cur.byte()?;
            if !cur.done() {
                return Err(invalid("huffman trailing bytes"));
            }
            dst.fill(sym);
            return Ok(());
        }
        HUF_KIND_HUF => {}
        _ => return Err(invalid("huffman kind")),
    }
    let max_sym = cur.byte()? as usize;
    const NIB_STEP: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 6, 12, 24];
    const NIB_LEN: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0];
    let mut lens = [0u8; 280];
    let mut bl_count = [0u16; 16];
    let mut s = 0usize;
    while s <= max_sym {
        let b = cur.byte()? as usize;
        let n0 = b & 15;
        let l0 = NIB_LEN[n0];
        lens[s] = l0;
        bl_count[l0 as usize] += 1;
        s += NIB_STEP[n0] as usize;
        if s <= max_sym {
            let n1 = b >> 4;
            let l1 = NIB_LEN[n1];
            lens[s] = l1;
            bl_count[l1 as usize] += 1;
            s += NIB_STEP[n1] as usize;
        } else if (b >> 4) != 0 {
            return Err(invalid("huffman length run"));
        }
    }
    if s != max_sym + 1 {
        return Err(invalid("huffman length run"));
    }
    if lens[max_sym] == 0 {
        return Err(invalid("huffman max symbol"));
    }
    bl_count[0] = 0;
    // Require a complete prefix code (Kraft sum == 2^max_len) from the per-length counts.
    let max_len = (1..16).rev().find(|&l| bl_count[l] != 0).unwrap_or(0) as u32;
    let mut kraft = 0u32;
    let mut used = 0usize;
    for l in 1..=max_len as usize {
        kraft += (bl_count[l] as u32) << (max_len as usize - l);
        used += bl_count[l] as usize;
    }
    if used < 2 || kraft != 1 << max_len {
        return Err(invalid("huffman incomplete code"));
    }
    let split4 = n >= HUF_SPLIT4;
    let (s0, s1, s2) = if split4 {
        (cur.uleb()?, cur.uleb()?, cur.uleb()?)
    } else {
        (blob.len() - cur.pos, 0, 0)
    };
    let rest = &blob[cur.pos..];
    let (a, rest) = rest.split_at(s0.min(rest.len()));
    let (b, rest) = rest.split_at(s1.min(rest.len()));
    let (c, d) = rest.split_at(s2.min(rest.len()));
    if a.len() != s0 || b.len() != s1 || c.len() != s2 {
        return Err(invalid("huffman stream sizes"));
    }
    let size = 1usize << max_len;
    thread_local! {
        static HUF_DEC_TABLE: std::cell::UnsafeCell<[u16; 1 << HUF_BITS]> =
            const { std::cell::UnsafeCell::new([0u16; 1 << HUF_BITS]) };
    }
    let tbl_ptr = HUF_DEC_TABLE.with(|cell| cell.get().cast::<u16>());
    // Doubling build: at level `l` the table (size 2^l) is its lower half copied
    // once (contiguous memcpy), then every code of length `l` writes its single
    // bit-reversed slot. Low byte holds symbol (`e as u8`), high byte holds code length (`e >> 8`).
    {
        let min_len = (1..=max_len as usize)
            .find(|&l| bl_count[l] != 0)
            .unwrap_or(1);
        let mut start = [0u16; 17];
        for l in min_len..16 {
            start[l + 1] = start[l] + bl_count[l];
        }
        let mut by_len = [core::mem::MaybeUninit::<u8>::uninit(); 512];
        let by_len_ptr = by_len.as_mut_ptr().cast::<u8>();
        let mut fill = start;
        fill[0] = 256;
        for (sym, &l) in lens[..=max_sym].iter().enumerate() {
            let li = l as usize;
            unsafe {
                *by_len_ptr.add(fill[li] as usize) = sym as u8;
            }
            fill[li] += 1;
        }
        unsafe {
            let mut cur = 1usize << min_len;
            let mut code = 0u32;
            let l_lo0 = min_len as u16;
            let rsh0 = 32 - min_len;
            for idx_sym in start[min_len] as usize..start[min_len + 1] as usize {
                let sym = *by_len_ptr.add(idx_sym);
                let idx = (code.reverse_bits() >> rsh0) as usize;
                *tbl_ptr.add(idx) = ((sym as u16) << 8) | l_lo0;
                code += 1;
            }
            for l in (min_len + 1)..=max_len as usize {
                if cur >= 8 {
                    let mut c = 0usize;
                    while c < cur {
                        tbl_ptr
                            .add(cur + c)
                            .cast::<u128>()
                            .write_unaligned(tbl_ptr.add(c).cast::<u128>().read_unaligned());
                        c += 8;
                    }
                } else {
                    for c in 0..cur {
                        *tbl_ptr.add(cur + c) = *tbl_ptr.add(c);
                    }
                }
                cur <<= 1;
                code <<= 1;
                let l_lo = l as u16;
                let rsh = 32 - l;
                for idx_sym in start[l] as usize..start[l + 1] as usize {
                    let sym = *by_len_ptr.add(idx_sym);
                    let idx = (code.reverse_bits() >> rsh) as usize;
                    *tbl_ptr.add(idx) = ((sym as u16) << 8) | l_lo;
                    code += 1;
                }
            }
        }
    }
    // SAFETY: `kraft == 1 << max_len == size` guarantees every slot in `0..size` was written above.
    let table = unsafe { std::slice::from_raw_parts(tbl_ptr, size) };
    let mask = (size - 1) as u64;
    if !split4 {
        let fast_lim = n.saturating_sub(5);
        let a_bit_lim = a.len().saturating_sub(8) * 8;
        let a_ptr = a.as_ptr();
        let p0 = dst.as_mut_ptr();
        let mut wp0 = p0;
        let mut bp0 = 0usize;
        let mut i = 0usize;
        unsafe {
            while i < fast_lim && bp0 <= a_bit_lim {
                let mut b0 = a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
                macro_rules! step1 {
                    ($off:expr) => {{
                        let e0 = *tbl_ptr.add((b0 & mask) as usize);
                        let l0 = (e0 as u8) as usize;
                        b0 >>= l0;
                        bp0 += l0;
                        *wp0.add($off) = (e0 >> 8) as u8;
                    }};
                    ($off:expr, last) => {{
                        let e0 = *tbl_ptr.add((b0 & mask) as usize);
                        bp0 += (e0 as u8) as usize;
                        *wp0.add($off) = (e0 >> 8) as u8;
                    }};
                }
                step1!(0);
                step1!(1);
                step1!(2);
                step1!(3);
                step1!(4, last);
                wp0 = wp0.add(5);
                i += 5;
            }
        }
        return if huf_decode_tail(a, bp0, p0, i, n, table, mask) {
            Ok(())
        } else {
            Err(invalid("huffman stream consumption"))
        };
    }
    let q = n.div_ceil(4);
    let lens4 = [
        q.min(n),
        q.min(n.saturating_sub(q)),
        q.min(n.saturating_sub(2 * q)),
        n.saturating_sub(3 * q),
    ];
    let base = dst.as_mut_ptr();
    let fast_lim = lens4[3].saturating_sub(4);
    let (a_bit_lim, b_bit_lim, c_bit_lim, d_bit_lim) = (
        a.len().saturating_sub(13) * 8,
        b.len().saturating_sub(13) * 8,
        c.len().saturating_sub(13) * 8,
        d.len().saturating_sub(13) * 8,
    );
    let (a_ptr, b_ptr, c_ptr, d_ptr) = (a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr());
    let (mut bp0, mut bp1, mut bp2, mut bp3) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0usize;
    // SAFETY: stream k writes `dst[k*q .. k*q + lens4[k]]`, all inside `dst`.
    let (p0, p1, p2, p3) = unsafe {
        (
            base,
            base.add(q.min(n)),
            base.add((2 * q).min(n)),
            base.add((3 * q).min(n)),
        )
    };
    let (mut wp0, mut wp1, mut wp2, mut wp3) = (p0, p1, p2, p3);
    if a.len() >= 13 && b.len() >= 13 && c.len() >= 13 && d.len() >= 13 {
        unsafe {
            let mut b0 = a_ptr.cast::<u64>().read_unaligned().to_le();
            let mut b1 = b_ptr.cast::<u64>().read_unaligned().to_le();
            let mut b2 = c_ptr.cast::<u64>().read_unaligned().to_le();
            let mut b3 = d_ptr.cast::<u64>().read_unaligned().to_le();
            while (i < fast_lim)
                & (bp0 <= a_bit_lim)
                & (bp1 <= b_bit_lim)
                & (bp2 <= c_bit_lim)
                & (bp3 <= d_bit_lim)
            {
                macro_rules! step4 {
                    ($off:expr) => {{
                        let e0 = *tbl_ptr.add((b0 & mask) as usize);
                        let e1 = *tbl_ptr.add((b1 & mask) as usize);
                        let e2 = *tbl_ptr.add((b2 & mask) as usize);
                        let e3 = *tbl_ptr.add((b3 & mask) as usize);
                        let l0 = (e0 as u8) as usize;
                        let l1 = (e1 as u8) as usize;
                        let l2 = (e2 as u8) as usize;
                        let l3 = (e3 as u8) as usize;
                        b0 >>= l0;
                        b1 >>= l1;
                        b2 >>= l2;
                        b3 >>= l3;
                        bp0 += l0;
                        bp1 += l1;
                        bp2 += l2;
                        bp3 += l3;
                        *wp0.add($off) = (e0 >> 8) as u8;
                        *wp1.add($off) = (e1 >> 8) as u8;
                        *wp2.add($off) = (e2 >> 8) as u8;
                        *wp3.add($off) = (e3 >> 8) as u8;
                    }};
                    ($off:expr, reload) => {{
                        let e0 = *tbl_ptr.add((b0 & mask) as usize);
                        let e1 = *tbl_ptr.add((b1 & mask) as usize);
                        let e2 = *tbl_ptr.add((b2 & mask) as usize);
                        let e3 = *tbl_ptr.add((b3 & mask) as usize);
                        let nb0 =
                            a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
                        let nb1 =
                            b_ptr.add(bp1 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp1 & 7);
                        let nb2 =
                            c_ptr.add(bp2 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp2 & 7);
                        let nb3 =
                            d_ptr.add(bp3 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp3 & 7);
                        let l0 = (e0 as u8) as usize;
                        let l1 = (e1 as u8) as usize;
                        let l2 = (e2 as u8) as usize;
                        let l3 = (e3 as u8) as usize;
                        b0 = nb0 >> l0;
                        b1 = nb1 >> l1;
                        b2 = nb2 >> l2;
                        b3 = nb3 >> l3;
                        bp0 += l0;
                        bp1 += l1;
                        bp2 += l2;
                        bp3 += l3;
                        *wp0.add($off) = (e0 >> 8) as u8;
                        *wp1.add($off) = (e1 >> 8) as u8;
                        *wp2.add($off) = (e2 >> 8) as u8;
                        *wp3.add($off) = (e3 >> 8) as u8;
                    }};
                }
                step4!(0);
                step4!(1);
                step4!(2);
                step4!(3, reload);
                wp0 = wp0.add(4);
                wp1 = wp1.add(4);
                wp2 = wp2.add(4);
                wp3 = wp3.add(4);
                i += 4;
            }
        }
    }
    let ok0 = huf_decode_tail(a, bp0, p0, i, lens4[0], table, mask);
    let ok1 = huf_decode_tail(b, bp1, p1, i, lens4[1], table, mask);
    let ok2 = huf_decode_tail(c, bp2, p2, i, lens4[2], table, mask);
    let ok3 = huf_decode_tail(d, bp3, p3, i, lens4[3], table, mask);
    if ok0 && ok1 && ok2 && ok3 {
        Ok(())
    } else {
        Err(invalid("huffman stream consumption"))
    }
}

/// `ULEB len` + Huffman blob when strictly smaller than `src`, else `ULEB src.len()` + raw bytes.
fn put_huf_or_raw(out: &mut Vec<u8>, src: &[u8]) {
    let mark = out.len();
    let reserve = uleb_len(src.len());
    out.resize(mark + reserve, 0);
    if huf_encode(src, out, src.len()) {
        let blob_len = out.len() - (mark + reserve);
        let mut hdr = [0u8; 5];
        let mut hlen = 0usize;
        let mut v = blob_len;
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            hdr[hlen] = b;
            hlen += 1;
            if v == 0 {
                break;
            }
        }
        out[mark..mark + hlen].copy_from_slice(&hdr[..hlen]);
        if hlen < reserve {
            out.copy_within(mark + reserve.., mark + hlen);
            out.truncate(out.len() - (reserve - hlen));
        }
    } else {
        out.truncate(mark);
        put_uleb(out, src.len());
        out.extend_from_slice(src);
    }
}

/// Inverse of [`put_huf_or_raw`]: `len == dst.len()` is raw (zero-copy slice), `len < dst.len()` is Huffman.
fn take_huf_or_raw<'a, 'b: 'a>(cursor: &mut Cursor<'b>, dst: &'a mut [u8]) -> Result<&'a [u8]> {
    let len = cursor.uleb()?;
    let bytes = cursor.take(len)?;
    if len == dst.len() {
        Ok(bytes)
    } else if len < dst.len() {
        huf_decode(bytes, dst)?;
        Ok(dst)
    } else {
        Err(invalid("huffman substream length"))
    }
}

