#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn huf_decode_split4(
    a: &[u8],
    b: &[u8],
    c: &[u8],
    d: &[u8],
    dst: &mut [u8],
    n: usize,
    max_len: u8,
    tbl_ptr: *const u16,
    table: &[u16],
    mask: u64,
) -> Result<()> {
    let q = n.div_ceil(4);
    let lens4 = [
        q.min(n),
        q.min(n.saturating_sub(q)),
        q.min(n.saturating_sub(2 * q)),
        n.saturating_sub(3 * q),
    ];
    let base = dst.as_mut_ptr();
    let (a_bit_lim, b_bit_lim, c_bit_lim, d_bit_lim) = (
        a.len().saturating_sub(8) * 8,
        b.len().saturating_sub(8) * 8,
        c.len().saturating_sub(8) * 8,
        d.len().saturating_sub(8) * 8,
    );
    let (a_ptr, b_ptr, c_ptr, d_ptr) = (a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr());
    let (mut bp0, mut bp1, mut bp2, mut bp3) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0usize;
    let (p0, p1, p2, p3) = unsafe {
        (
            base,
            base.add(q.min(n)),
            base.add((2 * q).min(n)),
            base.add((3 * q).min(n)),
        )
    };
    let (mut wp0, mut wp1, mut wp2, mut wp3) = (p0, p1, p2, p3);
    if a.len() >= 8 && b.len() >= 8 && c.len() >= 8 && d.len() >= 8 {
        unsafe {
            macro_rules! step4 {
                ($b0:ident, $b1:ident, $b2:ident, $b3:ident, $off:expr) => {{
                    let e0 = *tbl_ptr.add(($b0 & mask) as usize);
                    let e1 = *tbl_ptr.add(($b1 & mask) as usize);
                    let e2 = *tbl_ptr.add(($b2 & mask) as usize);
                    let e3 = *tbl_ptr.add(($b3 & mask) as usize);
                    let l0 = (e0 >> 8) as usize;
                    let l1 = (e1 >> 8) as usize;
                    let l2 = (e2 >> 8) as usize;
                    let l3 = (e3 >> 8) as usize;
                    $b0 >>= l0;
                    $b1 >>= l1;
                    $b2 >>= l2;
                    $b3 >>= l3;
                    bp0 += l0;
                    bp1 += l1;
                    bp2 += l2;
                    bp3 += l3;
                    *wp0.add($off) = e0 as u8;
                    *wp1.add($off) = e1 as u8;
                    *wp2.add($off) = e2 as u8;
                    *wp3.add($off) = e3 as u8;
                }};
                ($b0:ident, $b1:ident, $b2:ident, $b3:ident, $off:expr, last) => {{
                    let e0 = *tbl_ptr.add(($b0 & mask) as usize);
                    let e1 = *tbl_ptr.add(($b1 & mask) as usize);
                    let e2 = *tbl_ptr.add(($b2 & mask) as usize);
                    let e3 = *tbl_ptr.add(($b3 & mask) as usize);
                    bp0 += (e0 >> 8) as usize;
                    bp1 += (e1 >> 8) as usize;
                    bp2 += (e2 >> 8) as usize;
                    bp3 += (e3 >> 8) as usize;
                    *wp0.add($off) = e0 as u8;
                    *wp1.add($off) = e1 as u8;
                    *wp2.add($off) = e2 as u8;
                    *wp3.add($off) = e3 as u8;
                }};
            }
            if max_len <= 9 {
                let fast_lim6 = lens4[3].saturating_sub(6);
                while (i < fast_lim6)
                    & (bp0 <= a_bit_lim)
                    & (bp1 <= b_bit_lim)
                    & (bp2 <= c_bit_lim)
                    & (bp3 <= d_bit_lim)
                {
                    let mut b0 = a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
                    let mut b1 = b_ptr.add(bp1 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp1 & 7);
                    let mut b2 = c_ptr.add(bp2 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp2 & 7);
                    let mut b3 = d_ptr.add(bp3 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp3 & 7);
                    step4!(b0, b1, b2, b3, 0);
                    step4!(b0, b1, b2, b3, 1);
                    step4!(b0, b1, b2, b3, 2);
                    step4!(b0, b1, b2, b3, 3);
                    step4!(b0, b1, b2, b3, 4);
                    step4!(b0, b1, b2, b3, 5, last);
                    wp0 = wp0.add(6);
                    wp1 = wp1.add(6);
                    wp2 = wp2.add(6);
                    wp3 = wp3.add(6);
                    i += 6;
                }
            } else {
                let fast_lim5 = lens4[3].saturating_sub(5);
                while (i < fast_lim5)
                    & (bp0 <= a_bit_lim)
                    & (bp1 <= b_bit_lim)
                    & (bp2 <= c_bit_lim)
                    & (bp3 <= d_bit_lim)
                {
                    let mut b0 = a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
                    let mut b1 = b_ptr.add(bp1 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp1 & 7);
                    let mut b2 = c_ptr.add(bp2 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp2 & 7);
                    let mut b3 = d_ptr.add(bp3 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp3 & 7);
                    step4!(b0, b1, b2, b3, 0);
                    step4!(b0, b1, b2, b3, 1);
                    step4!(b0, b1, b2, b3, 2);
                    step4!(b0, b1, b2, b3, 3);
                    step4!(b0, b1, b2, b3, 4, last);
                    wp0 = wp0.add(5);
                    wp1 = wp1.add(5);
                    wp2 = wp2.add(5);
                    wp3 = wp3.add(5);
                    i += 5;
                }
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
