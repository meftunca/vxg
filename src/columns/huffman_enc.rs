#[derive(Default)]
struct StrideScratch {
    planes: [Vec<u8>; 8],
    super_mask_buf: Vec<u64>,
    mask_buf: Vec<u64>,
    nz_buf: Vec<u8>,
    lz_dict4: Vec<u32>,
    lz_dict7: Vec<u16>,
    lz_tokens: Vec<u8>,
    lz_offsets: Vec<u8>,
    lz_ext: Vec<u8>,
    lz_lits: Vec<u8>,
    lz_meta_raw: Vec<u8>,
    lz_meta_comp: Vec<u8>,
    lz_lits_comp: Vec<u8>,
    lz_lits_huf: Vec<u8>,
    lz_meta_huf: Vec<u8>,
    lz_sub_lits: Vec<u8>,
    lz_sub_tokens: Vec<u8>,
    lz_sub_offsets: Vec<u8>,
    lz_sub_ext: Vec<u8>,
}

thread_local! {
    static STRIDE_SCRATCH: std::cell::RefCell<StrideScratch> =
        std::cell::RefCell::new(StrideScratch::default());
    /// Plane buffers for `STRIDE_PACKET` mode 2 (delta planes + nested `FAST_LZ`); kept separate
    /// from `STRIDE_SCRATCH` because the nested `FAST_LZ` coder borrows that scratch itself.
    static STRIDE_PLANES2: std::cell::RefCell<[Vec<u8>; 8]> =
        std::cell::RefCell::new(Default::default());
    static STRIDE_COMBINED: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------------------------
// Canonical Huffman (order-0) entropy stage for FAST_LZ literal / meta streams.
//
// Blob format:
//   kind 1: `sym`                                   (every byte equals `sym`)
//   kind 2: `max_sym`, code-length nibbles for symbols `0..=max_sym` (low nibble first;
//           0..=11 = length, 12/13/14/15 = run of 3/6/12/24 zero lengths), ULEB sizes of
//           streams 0..=2, then 4 LSB-first bitstreams; stream `k` holds output quarter `k`
//           (`q = ceil(n/4)`). Codes are canonical, length-limited to `HUF_BITS`, and must form
//           a complete prefix code; every stream must be consumed to its last byte.
// ---------------------------------------------------------------------------------------------
const HUF_BITS: u32 = 11;
const HUF_KIND_RLE: u8 = 1;
const HUF_KIND_HUF: u8 = 2;
/// Streams shorter than this stay raw: per-blob table setup would cost more than it saves.
const HUF_MIN: usize = 160;
/// Streams of at least this many symbols are split into 4 interleaved bitstreams.
const HUF_SPLIT4: usize = 1024;

#[inline(never)]
fn huf_lengths(freq: &[u32; 256]) -> Option<[u8; 256]> {
    let mut leaves_uninit = [core::mem::MaybeUninit::<u32>::uninit(); 256];
    let leaves_ptr = leaves_uninit.as_mut_ptr().cast::<u32>();
    let mut k = 0usize;
    for (s, &c) in freq.iter().enumerate() {
        if c > 0 {
            unsafe {
                *leaves_ptr.add(k) = (c << 8) | (s as u32);
            }
            k += 1;
        }
    }
    if k < 2 {
        return None;
    }
    let leaves = unsafe { std::slice::from_raw_parts_mut(leaves_ptr, k) };
    leaves.sort_unstable();

    let mut q1_w = [core::mem::MaybeUninit::<u32>::uninit(); 256];
    let mut leaf_parent = [core::mem::MaybeUninit::<u8>::uninit(); 256];
    let mut int_parent = [core::mem::MaybeUninit::<u8>::uninit(); 256];
    let mut int_depth = [core::mem::MaybeUninit::<u8>::uninit(); 256];
    let q1_ptr = q1_w.as_mut_ptr().cast::<u32>();
    let lp_ptr = leaf_parent.as_mut_ptr().cast::<u8>();
    let ip_ptr = int_parent.as_mut_ptr().cast::<u8>();
    let id_ptr = int_depth.as_mut_ptr().cast::<u8>();

    let mut i0 = 0usize;
    let mut i1 = 0usize;
    let mut n1 = 0usize;
    let mut w0 = (leaves[0] >> 8).max(1);
    for _ in 0..(k - 1) {
        let w1 = if i1 < n1 {
            unsafe { *q1_ptr.add(i1) }
        } else {
            u32::MAX
        };
        let wa = if w0 <= w1 {
            unsafe {
                *lp_ptr.add(i0) = n1 as u8;
            }
            let w = w0;
            i0 += 1;
            w0 = if i0 < k {
                (unsafe { *leaves.get_unchecked(i0) } >> 8).max(1)
            } else {
                u32::MAX
            };
            w
        } else {
            unsafe {
                *ip_ptr.add(i1) = n1 as u8;
            }
            i1 += 1;
            w1
        };

        let w1b = if i1 < n1 {
            unsafe { *q1_ptr.add(i1) }
        } else {
            u32::MAX
        };
        let wb = if w0 <= w1b {
            unsafe {
                *lp_ptr.add(i0) = n1 as u8;
            }
            let w = w0;
            i0 += 1;
            w0 = if i0 < k {
                (unsafe { *leaves.get_unchecked(i0) } >> 8).max(1)
            } else {
                u32::MAX
            };
            w
        } else {
            unsafe {
                *ip_ptr.add(i1) = n1 as u8;
            }
            i1 += 1;
            w1b
        };

        unsafe {
            *q1_ptr.add(n1) = wa + wb;
        }
        n1 += 1;
    }
    let root = k - 2;
    unsafe {
        *id_ptr.add(root) = 0;
        for node in (0..root).rev() {
            let p = *ip_ptr.add(node) as usize;
            *id_ptr.add(node) = *id_ptr.add(p) + 1;
        }
        let max_len = *id_ptr + 1;
        let mut lens = [0u8; 256];
        if u32::from(max_len) <= HUF_BITS {
            for idx in 0..k {
                let sym = (*leaves.get_unchecked(idx) & 0xff) as usize;
                let p = *lp_ptr.add(idx) as usize;
                lens[sym] = *id_ptr.add(p) + 1;
            }
            return Some(lens);
        }
        let max_b = HUF_BITS as usize;
        let mut bl_count = [0u16; 16];
        let mut kraft = 0u32;
        for idx in 0..k {
            let p = *lp_ptr.add(idx) as usize;
            let d = ((*id_ptr.add(p) + 1) as usize).min(max_b);
            bl_count[d] += 1;
            kraft += 1u32 << (max_b - d);
        }
        let mut over = kraft - (1u32 << max_b);
        while over > 0 {
            let mut l = max_b - 1;
            while bl_count[l] == 0 {
                l -= 1;
            }
            bl_count[l] -= 1;
            bl_count[l + 1] += 2;
            bl_count[max_b] -= 1;
            over -= 1;
        }
        let mut idx = 0usize;
        for l in (1..=max_b).rev() {
            let cnt = bl_count[l] as usize;
            for _ in 0..cnt {
                let sym = (*leaves.get_unchecked(idx) & 0xff) as usize;
                lens[sym] = l as u8;
                idx += 1;
            }
        }
        Some(lens)
    }
}

/// Canonical codes (bit-reversed for LSB-first emission), ordered by `(len, sym)`.
fn huf_codes(lens: &[u8; 256], max_sym: usize) -> [u16; 256] {
    let mut bl_count = [0u16; 16];
    for &l in &lens[..=max_sym] {
        bl_count[l as usize] += 1;
    }
    bl_count[0] = 0;
    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = [0u16; 256];
    for s in 0..=max_sym {
        let l = lens[s] as usize;
        if l > 0 {
            let c = next_code[l];
            next_code[l] += 1;
            codes[s] = c.reverse_bits() >> (16 - l);
        }
    }
    codes
}

/// Append a Huffman blob for `src` to `out` if it is strictly smaller than `limit` bytes.
fn huf_encode(src: &[u8], out: &mut Vec<u8>, limit: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn huf_encode_x86_v3(src: &[u8], out: &mut Vec<u8>, limit: usize) -> bool {
            huf_encode_impl(src, out, limit)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi2")
        {
            return unsafe { huf_encode_x86_v3(src, out, limit) };
        }
    }
    huf_encode_impl(src, out, limit)
}

#[inline(always)]
fn huf_encode_impl(src: &[u8], out: &mut Vec<u8>, limit: usize) -> bool {
    let n = src.len();
    if n < HUF_MIN {
        return false;
    }
    let mut freq = [0u32; 256];
    let ptr = src.as_ptr();
    if n >= 16384 {
        #[inline(never)]
        fn huf_hist_large(src: &[u8], freq: &mut [u32; 256]) {
            let (mut f1, mut f2, mut f3) = ([0u32; 256], [0u32; 256], [0u32; 256]);
            let (ptr, chunks8) = (src.as_ptr(), src.len() / 8);
            for i in 0..chunks8 {
                let w = unsafe { ptr.add(i * 8).cast::<u64>().read_unaligned().to_le() };
                freq[(w & 0xff) as usize] += 1;
                f1[((w >> 8) & 0xff) as usize] += 1;
                f2[((w >> 16) & 0xff) as usize] += 1;
                f3[((w >> 24) & 0xff) as usize] += 1;
                freq[((w >> 32) & 0xff) as usize] += 1;
                f1[((w >> 40) & 0xff) as usize] += 1;
                f2[((w >> 48) & 0xff) as usize] += 1;
                f3[(w >> 56) as usize] += 1;
            }
            for &b in &src[chunks8 * 8..] {
                freq[b as usize] += 1;
            }
            for s in 0..256 {
                freq[s] += f1[s] + f2[s] + f3[s];
            }
        }
        huf_hist_large(src, &mut freq);
    } else {
        let chunks8 = n / 8;
        for i in 0..chunks8 {
            unsafe {
                let w = ptr.add(i * 8).cast::<u64>().read_unaligned().to_le();
                freq[(w & 0xff) as usize] += 1;
                freq[((w >> 8) & 0xff) as usize] += 1;
                freq[((w >> 16) & 0xff) as usize] += 1;
                freq[((w >> 24) & 0xff) as usize] += 1;
                freq[((w >> 32) & 0xff) as usize] += 1;
                freq[((w >> 40) & 0xff) as usize] += 1;
                freq[((w >> 48) & 0xff) as usize] += 1;
                freq[(w >> 56) as usize] += 1;
            }
        }
        for &b in &src[chunks8 * 8..] {
            freq[b as usize] += 1;
        }
    }
    let (mut distinct, mut max_sym, mut max_freq, mut heavy_sum) = (0usize, 0usize, 0u32, 0u32);
    let heavy_thresh = (n as u32) >> 5;
    for (s, &c) in freq.iter().enumerate() {
        if c > 0 {
            distinct += 1;
            max_sym = s;
            max_freq = max_freq.max(c);
            if c > heavy_thresh {
                heavy_sum += c;
            }
        }
    }
    if distinct == 1 {
        if 2 >= limit {
            return false;
        }
        out.extend_from_slice(&[HUF_KIND_RLE, src[0]]);
        return true;
    }
    // Adaptive Entropy Gate: if the alphabet is wide and flat, skip Huffman tree & bitstream build.
    if (distinct >= 200 && (max_freq as usize) * 18 < n)
        || (distinct >= 160 && (heavy_sum as usize) * 6 < n && (max_freq as usize) * 12 < n)
    {
        return false;
    }
    let Some(lens) = huf_lengths(&freq) else {
        return false;
    };
    let mut bits = 0u64;
    for s in 0..=max_sym {
        bits += freq[s] as u64 * lens[s] as u64;
    }
    // Table nibbles on stack.
    let mut nibbles = [0u8; 256];
    let mut nib_len = 0usize;
    let mut s = 0usize;
    while s <= max_sym {
        if lens[s] != 0 {
            nibbles[nib_len] = lens[s];
            nib_len += 1;
            s += 1;
            continue;
        }
        let mut run = 0usize;
        while s + run <= max_sym && lens[s + run] == 0 {
            run += 1;
        }
        s += run;
        for (code, size) in [(15u8, 24usize), (14, 12), (13, 6), (12, 3)] {
            while run >= size {
                nibbles[nib_len] = code;
                nib_len += 1;
                run -= size;
            }
        }
        while run > 0 {
            nibbles[nib_len] = 0;
            nib_len += 1;
            run -= 1;
        }
    }
    let est = 2 + nib_len.div_ceil(2) + if n >= HUF_SPLIT4 { 9 } else { 0 } + (bits as usize).div_ceil(8) + 4;
    // Decode-cost gate: a Huffman stream costs a table build plus ~1.5 ns/byte on
    // decode versus a near-free raw copy, so demand at least 3% and 24 bytes savings.
    if est + 24 >= limit || est * 100 > limit * 97 {
        return false;
    }
    let codes = huf_codes(&lens, max_sym);
    let mut packed_code = [0u32; 256];
    for s in 0..=max_sym {
        packed_code[s] = (codes[s] as u32) | ((lens[s] as u32) << 16);
    }
    let start = out.len();
    out.reserve(est + 32);
    out.push(HUF_KIND_HUF);
    out.push(max_sym as u8);
    for pair in nibbles[..nib_len].chunks(2) {
        out.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
    }

    #[inline(always)]
    unsafe fn encode_stream_fast(
        part: &[u8],
        packed_code: &[u32; 256],
        out_ptr: *mut u8,
        mut out_pos: usize,
    ) -> usize {
        let mut acc = 0u64;
        let mut nb = 0u32;
        let p_in = part.as_ptr();
        let len5 = part.len() / 5 * 5;
        let mut i = 0usize;
        while i < len5 {
            unsafe {
                let pc0 = *packed_code.get_unchecked(*p_in.add(i) as usize);
                let pc1 = *packed_code.get_unchecked(*p_in.add(i + 1) as usize);
                let pc2 = *packed_code.get_unchecked(*p_in.add(i + 2) as usize);
                let pc3 = *packed_code.get_unchecked(*p_in.add(i + 3) as usize);
                let pc4 = *packed_code.get_unchecked(*p_in.add(i + 4) as usize);
                let l0 = pc0 >> 16;
                let l1 = pc1 >> 16;
                let l2 = pc2 >> 16;
                let l3 = pc3 >> 16;
                let l4 = pc4 >> 16;
                let l01 = l0 + l1;
                let l23 = l2 + l3;
                let l0123 = l01 + l23;
                let chunk_len = l0123 + l4;
                let c01 = u64::from(pc0 & 0xffff) | (u64::from(pc1 & 0xffff) << l0);
                let c23 = u64::from(pc2 & 0xffff) | (u64::from(pc3 & 0xffff) << l2);
                let c0123 = c01 | (c23 << l01);
                let chunk_bits = c0123 | (u64::from(pc4 & 0xffff) << l0123);
                acc |= chunk_bits << nb;
                nb += chunk_len;
                out_ptr.add(out_pos).cast::<u64>().write_unaligned(acc.to_le());
                out_pos += (nb >> 3) as usize;
                acc >>= nb & !7;
                nb &= 7;
            }
            i += 5;
        }
        while i < part.len() {
            let pc = unsafe { *packed_code.get_unchecked(*p_in.add(i) as usize) };
            acc |= u64::from(pc & 0xffff) << nb;
            nb += pc >> 16;
            i += 1;
        }
        while nb >= 8 {
            unsafe {
                *out_ptr.add(out_pos) = acc as u8;
            }
            out_pos += 1;
            acc >>= 8;
            nb -= 8;
        }
        if nb > 0 {
            unsafe {
                *out_ptr.add(out_pos) = acc as u8;
            }
            out_pos += 1;
        }
        out_pos
    }

    #[inline(always)]
    unsafe fn encode_pair_fast(
        part0: &[u8],
        part1: &[u8],
        packed_code: &[u32; 256],
        out_ptr: *mut u8,
        mut pos0: usize,
        mut pos1: usize,
    ) -> (usize, usize) {
        let (mut acc0, mut acc1) = (0u64, 0u64);
        let (mut nb0, mut nb1) = (0u32, 0u32);
        let (p0, p1) = (part0.as_ptr(), part1.as_ptr());
        let len5 = part0.len().min(part1.len()) / 5 * 5;
        let mut i = 0usize;
        while i < len5 {
            unsafe {
                let a0 = *packed_code.get_unchecked(*p0.add(i) as usize);
                let b0 = *packed_code.get_unchecked(*p1.add(i) as usize);
                let a1 = *packed_code.get_unchecked(*p0.add(i + 1) as usize);
                let b1 = *packed_code.get_unchecked(*p1.add(i + 1) as usize);
                let a2 = *packed_code.get_unchecked(*p0.add(i + 2) as usize);
                let b2 = *packed_code.get_unchecked(*p1.add(i + 2) as usize);
                let a3 = *packed_code.get_unchecked(*p0.add(i + 3) as usize);
                let b3 = *packed_code.get_unchecked(*p1.add(i + 3) as usize);
                let a4 = *packed_code.get_unchecked(*p0.add(i + 4) as usize);
                let b4 = *packed_code.get_unchecked(*p1.add(i + 4) as usize);

                let la01 = (a0 >> 16) + (a1 >> 16);
                let lb01 = (b0 >> 16) + (b1 >> 16);
                let la23 = (a2 >> 16) + (a3 >> 16);
                let lb23 = (b2 >> 16) + (b3 >> 16);
                let la0123 = la01 + la23;
                let lb0123 = lb01 + lb23;
                let ca01 = u64::from(a0 & 0xffff) | (u64::from(a1 & 0xffff) << (a0 >> 16));
                let cb01 = u64::from(b0 & 0xffff) | (u64::from(b1 & 0xffff) << (b0 >> 16));
                let ca23 = u64::from(a2 & 0xffff) | (u64::from(a3 & 0xffff) << (a2 >> 16));
                let cb23 = u64::from(b2 & 0xffff) | (u64::from(b3 & 0xffff) << (b2 >> 16));
                let ca = (ca01 | (ca23 << la01)) | (u64::from(a4 & 0xffff) << la0123);
                let cb = (cb01 | (cb23 << lb01)) | (u64::from(b4 & 0xffff) << lb0123);

                acc0 |= ca << nb0;
                acc1 |= cb << nb1;
                nb0 += la0123 + (a4 >> 16);
                nb1 += lb0123 + (b4 >> 16);
                out_ptr.add(pos0).cast::<u64>().write_unaligned(acc0.to_le());
                out_ptr.add(pos1).cast::<u64>().write_unaligned(acc1.to_le());
                pos0 += (nb0 >> 3) as usize;
                pos1 += (nb1 >> 3) as usize;
                acc0 >>= nb0 & !7;
                acc1 >>= nb1 & !7;
                nb0 &= 7;
                nb1 &= 7;
            }
            i += 5;
        }
        let mut i0 = i;
        while i0 < part0.len() {
            let pc = unsafe { *packed_code.get_unchecked(*p0.add(i0) as usize) };
            acc0 |= u64::from(pc & 0xffff) << nb0;
            nb0 += pc >> 16;
            if nb0 >= 32 {
                unsafe {
                    out_ptr.add(pos0).cast::<u32>().write_unaligned((acc0 as u32).to_le());
                }
                pos0 += 4;
                acc0 >>= 32;
                nb0 -= 32;
            }
            i0 += 1;
        }
        while nb0 >= 8 {
            unsafe { *out_ptr.add(pos0) = acc0 as u8 };
            pos0 += 1;
            acc0 >>= 8;
            nb0 -= 8;
        }
        if nb0 > 0 {
            unsafe { *out_ptr.add(pos0) = acc0 as u8 };
            pos0 += 1;
        }
        let mut i1 = i;
        while i1 < part1.len() {
            let pc = unsafe { *packed_code.get_unchecked(*p1.add(i1) as usize) };
            acc1 |= u64::from(pc & 0xffff) << nb1;
            nb1 += pc >> 16;
            if nb1 >= 32 {
                unsafe {
                    out_ptr.add(pos1).cast::<u32>().write_unaligned((acc1 as u32).to_le());
                }
                pos1 += 4;
                acc1 >>= 32;
                nb1 -= 32;
            }
            i1 += 1;
        }
        while nb1 >= 8 {
            unsafe { *out_ptr.add(pos1) = acc1 as u8 };
            pos1 += 1;
            acc1 >>= 8;
            nb1 -= 8;
        }
        if nb1 > 0 {
            unsafe { *out_ptr.add(pos1) = acc1 as u8 };
            pos1 += 1;
        }
        (pos0, pos1)
    }

    if n < HUF_SPLIT4 {
        let s_start = out.len();
        out.reserve((bits as usize).div_ceil(8) + 16);
        let new_len = unsafe { encode_stream_fast(src, &packed_code, out.as_mut_ptr(), s_start) };
        unsafe { out.set_len(new_len) };
    } else {
        let q = n.div_ceil(4);
        let max_s = (q * 11).div_ceil(8) + 16;
        let hdr_pos = out.len();
        out.reserve(9 + 4 * max_s + 32);
        let b0 = hdr_pos + 9;
        let b1 = b0 + max_s;
        let b2 = b0 + 2 * max_s;
        let b3 = b0 + 3 * max_s;
        let out_ptr = out.as_mut_ptr();
        let (e0, e1) = unsafe {
            encode_pair_fast(&src[..q], &src[q..2 * q], &packed_code, out_ptr, b0, b1)
        };
        let (e2, e3) = unsafe {
            encode_pair_fast(&src[2 * q..3 * q], &src[3 * q..n], &packed_code, out_ptr, b2, b3)
        };
        let stream_lens = [e0 - b0, e1 - b1, e2 - b2, e3 - b3];
        let mut uleb_hdr = [0u8; 9];
        let mut u_len = 0usize;
        for &sl in &stream_lens[..3] {
            let mut v = sl;
            loop {
                let mut b = (v & 0x7f) as u8;
                v >>= 7;
                if v != 0 {
                    b |= 0x80;
                }
                uleb_hdr[u_len] = b;
                u_len += 1;
                if v == 0 {
                    break;
                }
            }
        }
        unsafe {
            out_ptr.add(hdr_pos).copy_from_nonoverlapping(uleb_hdr.as_ptr(), u_len);
            let mut dst_pos = hdr_pos + u_len;
            std::ptr::copy(out_ptr.add(b0), out_ptr.add(dst_pos), stream_lens[0]);
            dst_pos += stream_lens[0];
            std::ptr::copy(out_ptr.add(b1), out_ptr.add(dst_pos), stream_lens[1]);
            dst_pos += stream_lens[1];
            std::ptr::copy(out_ptr.add(b2), out_ptr.add(dst_pos), stream_lens[2]);
            dst_pos += stream_lens[2];
            std::ptr::copy(out_ptr.add(b3), out_ptr.add(dst_pos), stream_lens[3]);
            dst_pos += stream_lens[3];
            out.set_len(dst_pos);
        }
    }
    if out.len() - start >= limit {
        out.truncate(start);
        return false;
    }
    true
}

