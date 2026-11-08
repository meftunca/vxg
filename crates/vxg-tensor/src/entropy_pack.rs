//! Fast zero-dependency Canonical Huffman (`12`-bit L1 LUT) + Zero-Sign-Elided `EC-Turbo`
//! tensor container (`KIND_TURBO_WHT_EC`) and LZ77+Huffman metadata compressor (`VXLZ`).

use crate::format_io::{push_u8_slice, Cur};
use crate::huff::{huff_decode, huff_encode};
use crate::wht_turbo::TurboWhtMatrix;
use crate::wht_turbo_enc::{rademacher_sgn2, rademacher_sgn4};
use crate::wht_turbo_tables::{ec_orth2d_nz_mask, orth4d64_nz_mask, unpack_t25_lane};
use crate::{Error, Result};

const CHUNK_TILES: usize = 4096;

#[inline(always)]
fn push_bits(sbuf: &mut u64, sbits: &mut u32, val: u64, n: u32, out: &mut Vec<u8>) {
    *sbuf |= val << *sbits;
    *sbits += n;
    if *sbits >= 32 {
        out.extend_from_slice(&(*sbuf as u32).to_le_bytes());
        *sbuf >>= 32;
        *sbits -= 32;
    }
}

#[inline(always)]
fn flush_bits(mut sbuf: u64, mut sbits: u32, out: &mut Vec<u8>) {
    while sbits > 0 {
        out.push((sbuf & 0xff) as u8);
        sbuf >>= 8;
        sbits = sbits.saturating_sub(8);
    }
}

#[inline(always)]
fn pop_bits(sbuf: &mut u64, sbits: &mut u32, spos: &mut usize, src: &[u8], n: u32) -> u32 {
    if *sbits < n {
        while *sbits <= 56 && *spos < src.len() {
            *sbuf |= (src[*spos] as u64) << *sbits;
            *sbits += 8;
            *spos += 1;
        }
    }
    let v = (*sbuf & ((1u64 << n) - 1)) as u32;
    *sbuf >>= n;
    *sbits = sbits.saturating_sub(n);
    v
}

fn pack_ec_chunk(bits: u8, _t0: usize, _steps: usize, sc: &[u16], pw: &[u32]) -> Vec<u8> {
    let wps = TurboWhtMatrix::words_per_tile_step(bits);
    let total_ts = sc.len() / 8;
    let mut out = Vec::new();
    let sc_mode = if sc.iter().all(|&v| (v & 0x003f) == 0) {
        2u8
    } else if sc.iter().all(|&v| (v & 0x000f) == 0) {
        1u8
    } else {
        0u8
    };
    out.push(sc_mode);

    let mut prev_hi = [0u8; 4];
    let sc_hi: Vec<u8> = sc
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let hi = (v >> 8) as u8;
            let d = hi.wrapping_sub(prev_hi[i & 3]);
            prev_hi[i & 3] = hi;
            d
        })
        .collect();
    huff_encode(&sc_hi, &mut out);
    if sc_mode == 2 {
        out.extend(sc.chunks_exact(4).map(|p| {
            (((p[0] >> 6) & 3) as u8)
                | ((((p[1] >> 6) & 3) as u8) << 2)
                | ((((p[2] >> 6) & 3) as u8) << 4)
                | ((((p[3] >> 6) & 3) as u8) << 6)
        }));
    } else if sc_mode == 1 {
        out.extend(
            sc.chunks_exact(2)
                .map(|p| (((p[0] >> 4) & 0x0f) as u8) | ((((p[1] >> 4) & 0x0f) as u8) << 4)),
        );
    } else {
        out.extend(sc.iter().map(|&v| (v & 0xff) as u8));
    }

    if bits == 2 {
        let mut mag_pairs = Vec::with_capacity(total_ts * 128);
        let mut sgn_bytes = Vec::with_capacity(total_ts * 128);
        for ts in 0..total_ts {
            let p64 = &pw[ts * 64..(ts + 1) * 64];
            for lane in 0..32 {
                let (w0, w1) = (p64[lane * 2], p64[lane * 2 + 1]);
                for c16 in [w0 & 0xffff, w0 >> 16, w1 & 0xffff, w1 >> 16] {
                    let (b0, b1) = ((c16 & 0xff) as u8, (c16 >> 8) as u8);
                    mag_pairs.push((b0 & 0x0f) | ((b1 & 0x0f) << 4));
                    sgn_bytes.push((b0 >> 4) | (b1 & 0xf0));
                }
            }
        }
        huff_encode(&mag_pairs, &mut out);
        push_u8_slice(&mut out, &sgn_bytes);
    } else if bits == 5 {
        let mut mag_syms = Vec::with_capacity(total_ts * 256);
        let mut sgn_bytes = Vec::with_capacity(total_ts * 128);
        let (mut sbuf, mut sbits) = (0u64, 0u32);
        for ts in 0..total_ts {
            let p80 = &pw[ts * wps..(ts + 1) * wps];
            for lane in 0..32 {
                let cc = unpack_t25_lane(p80, lane);
                for sr in 0..4 {
                    for hk in 0..2 {
                        let mag = (cc[sr][hk] & 63) as u8;
                        let s4 = (cc[sr][hk] >> 6) as u8;
                        let nz4 = orth4d64_nz_mask(mag as usize);
                        mag_syms.push(mag);
                        for d in 0..4 {
                            if (nz4 >> d) & 1 != 0 {
                                push_bits(&mut sbuf, &mut sbits, ((s4 >> d) & 1) as u64, 1, &mut sgn_bytes);
                            }
                        }
                    }
                }
            }
        }
        flush_bits(sbuf, sbits, &mut sgn_bytes);
        huff_encode(&mag_syms, &mut out);
        push_u8_slice(&mut out, &sgn_bytes);
    } else {
        let zero_elide = bits == 4;
        let mut idx_bytes = Vec::with_capacity(total_ts * 256);
        let mut sgn_bytes = Vec::with_capacity(total_ts * 128);
        let (mut sbuf, mut sbits) = (0u64, 0u32);
        for ts in 0..total_ts {
            let p96 = &pw[ts * 96..(ts + 1) * 96];
            for lane in 0..32 {
                let (w0, w1, sw) = (p96[lane * 2], p96[lane * 2 + 1], p96[64 + lane]);
                let i16s = [w0 & 0xffff, w0 >> 16, w1 & 0xffff, w1 >> 16];
                for sr in 0..4 {
                    let idx16 = i16s[sr];
                    idx_bytes.push((idx16 & 0xff) as u8);
                    idx_bytes.push((idx16 >> 8) as u8);
                    let sgn8 = (sw >> (sr * 8)) & 0xff;
                    for pr in 0..4 {
                        let idx4 = ((idx16 >> (pr * 4)) & 0x0f) as usize;
                        let s2 = (sgn8 >> (pr * 2)) & 3;
                        if zero_elide {
                            let nz2 = ec_orth2d_nz_mask(idx4);
                            for d in 0..2 {
                                if (nz2 >> d) & 1 != 0 {
                                    push_bits(&mut sbuf, &mut sbits, ((s2 >> d) & 1) as u64, 1, &mut sgn_bytes);
                                }
                            }
                        } else {
                            push_bits(&mut sbuf, &mut sbits, s2 as u64, 2, &mut sgn_bytes);
                        }
                    }
                }
            }
        }
        flush_bits(sbuf, sbits, &mut sgn_bytes);
        huff_encode(&idx_bytes, &mut out);
        push_u8_slice(&mut out, &sgn_bytes);
    }
    out
}

fn unpack_ec_chunk(
    bits: u8,
    t0: usize,
    steps: usize,
    bytes: &[u8],
    sc_out: &mut [u16],
    pw_out: &mut [u32],
) -> Result<()> {
    let mut r = Cur { b: bytes, pos: 0 };
    let sc_mode = r.u8()?;
    let mut sc_hi = huff_decode(&mut r, sc_out.len())?;
    let mut prev_hi = [0u8; 4];
    for (i, b) in sc_hi.iter_mut().enumerate() {
        let hi = b.wrapping_add(prev_hi[i & 3]);
        prev_hi[i & 3] = hi;
        *b = hi;
    }
    if sc_mode == 2 {
        for (i, &qb) in r.take(sc_out.len() / 4)?.iter().enumerate() {
            for k in 0..4 {
                sc_out[i * 4 + k] =
                    ((sc_hi[i * 4 + k] as u16) << 8) | ((((qb >> (k * 2)) & 3) as u16) << 6);
            }
        }
    } else if sc_mode == 1 {
        for (i, &mb) in r.take(sc_out.len() / 2)?.iter().enumerate() {
            sc_out[i * 2] = ((sc_hi[i * 2] as u16) << 8) | (((mb & 0x0f) as u16) << 4);
            sc_out[i * 2 + 1] = ((sc_hi[i * 2 + 1] as u16) << 8) | (((mb >> 4) as u16) << 4);
        }
    } else {
        for (i, &lb) in r.take(sc_out.len())?.iter().enumerate() {
            sc_out[i] = ((sc_hi[i] as u16) << 8) | (lb as u16);
        }
    }

    let total_ts = sc_out.len() / 8;
    if bits == 2 {
        let mag_pairs = huff_decode(&mut r, total_ts * 128)?;
        let sgn_bytes = r.u8_vec()?;
        if sgn_bytes.len() != total_ts * 128 {
            return Err(Error::Format("corrupt EC-Turbo2 sign plane".into()));
        }
        for ts in 0..total_ts {
            let p64 = &mut pw_out[ts * 64..(ts + 1) * 64];
            for lane in 0..32 {
                let mut c16 = [0u32; 4];
                for sr in 0..4 {
                    let k = (ts * 32 + lane) * 4 + sr;
                    let (mp, sb) = (mag_pairs[k] as u32, sgn_bytes[k] as u32);
                    c16[sr] = ((mp & 0x0f) | ((sb & 0x0f) << 4)) | (((mp >> 4) | (sb & 0xf0)) << 8);
                }
                p64[lane * 2] = c16[0] | (c16[1] << 16);
                p64[lane * 2 + 1] = c16[2] | (c16[3] << 16);
            }
        }
    } else if bits == 5 {
        let mag_syms = huff_decode(&mut r, total_ts * 256)?;
        let sgn_bytes = r.u8_vec()?;
        let (mut sbuf, mut sbits, mut spos) = (0u64, 0u32, 0usize);
        for ts_local in 0..total_ts {
            let global_ts = t0 * steps + ts_local;
            let p80 = &mut pw_out[ts_local * 80..(ts_local + 1) * 80];
            p80.fill(0);
            for lane in 0..32 {
                let mut c = [[0u32; 2]; 4];
                for sr in 0..4 {
                    let k = (ts_local * 32 + lane) * 4 + sr;
                    for hk in 0..2 {
                        let mag = mag_syms[k * 2 + hk] as u32;
                        let nz4 = orth4d64_nz_mask(mag as usize);
                        let rs4 = rademacher_sgn4(global_ts, lane, sr, hk);
                        let mut s4 = 0u32;
                        for d in 0..4 {
                            let bit = if (nz4 >> d) & 1 != 0 {
                                pop_bits(&mut sbuf, &mut sbits, &mut spos, &sgn_bytes, 1)
                            } else {
                                ((rs4 >> d) & 1) as u32
                            };
                            s4 |= bit << d;
                        }
                        c[sr][hk] = mag | (s4 << 6);
                    }
                }
                p80[lane * 2] = c[0][0] | (c[0][1] << 10) | (c[1][0] << 20) | ((c[3][0] >> 8) << 30);
                p80[lane * 2 + 1] =
                    c[1][1] | (c[2][0] << 10) | (c[2][1] << 20) | ((c[3][1] >> 8) << 30);
                let h16 = (c[3][0] & 0xff) | ((c[3][1] & 0xff) << 8);
                p80[64 + (lane >> 1)] |= h16 << ((lane & 1) * 16);
            }
        }
    } else {
        let zero_elide = bits == 4;
        let idx_bytes = huff_decode(&mut r, total_ts * 256)?;
        let sgn_bytes = r.u8_vec()?;
        let (mut sbuf, mut sbits, mut spos) = (0u64, 0u32, 0usize);
        for ts_local in 0..total_ts {
            let global_ts = t0 * steps + ts_local;
            let p96 = &mut pw_out[ts_local * 96..(ts_local + 1) * 96];
            for lane in 0..32 {
                let mut i16s = [0u32; 4];
                let mut sw = 0u32;
                for sr in 0..4 {
                    let ib_off = ((ts_local * 32 + lane) * 4 + sr) * 2;
                    let idx16 = (idx_bytes[ib_off] as u32) | ((idx_bytes[ib_off + 1] as u32) << 8);
                    i16s[sr] = idx16;
                    let mut sgn8 = 0u32;
                    for pr in 0..4 {
                        let idx4 = ((idx16 >> (pr * 4)) & 0x0f) as usize;
                        let s2 = if zero_elide {
                            let nz2 = ec_orth2d_nz_mask(idx4);
                            let rs2 = rademacher_sgn2(global_ts, lane, sr, pr);
                            let mut v = 0u32;
                            for d in 0..2 {
                                let bit = if (nz2 >> d) & 1 != 0 {
                                    pop_bits(&mut sbuf, &mut sbits, &mut spos, &sgn_bytes, 1)
                                } else {
                                    ((rs2 >> d) & 1) as u32
                                };
                                v |= bit << d;
                            }
                            v
                        } else {
                            pop_bits(&mut sbuf, &mut sbits, &mut spos, &sgn_bytes, 2)
                        };
                        sgn8 |= s2 << (pr * 2);
                    }
                    sw |= sgn8 << (sr * 8);
                }
                p96[lane * 2] = i16s[0] | (i16s[1] << 16);
                p96[lane * 2 + 1] = i16s[2] | (i16s[3] << 16);
                p96[64 + lane] = sw;
            }
        }
    }
    Ok(())
}

pub(crate) fn pack_turbo_ec(m: &TurboWhtMatrix, out: &mut Vec<u8>) {
    let row_tiles = m.padded_rows / 4;
    let steps = m.padded_cols / 256;
    let wps = TurboWhtMatrix::words_per_tile_step(m.bits);
    let n_chunks = row_tiles.div_ceil(CHUNK_TILES);
    let mut chunks: Vec<Vec<u8>> = vec![Vec::new(); n_chunks];

    std::thread::scope(|s| {
        for (ci, slot) in chunks.iter_mut().enumerate() {
            let t0 = ci * CHUNK_TILES;
            let ct = (row_tiles - t0).min(CHUNK_TILES);
            let sc = &m.scales_f16[t0 * steps * 8..(t0 + ct) * steps * 8];
            let pw = &m.packed_words[t0 * steps * wps..(t0 + ct) * steps * wps];
            let bits = m.bits;
            s.spawn(move || {
                *slot = pack_ec_chunk(bits, t0, steps, sc, pw);
            });
        }
    });

    out.extend_from_slice(&(n_chunks as u32).to_le_bytes());
    for c in &chunks {
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
        out.extend_from_slice(c);
    }
}

pub(crate) fn unpack_turbo_ec(
    bits: u8,
    rows: usize,
    cols: usize,
    padded_cols: usize,
    r: &mut Cur<'_>,
) -> Result<TurboWhtMatrix> {
    if !matches!(bits, 2 | 3 | 4 | 5) || padded_cols != cols.div_ceil(256) * 256 {
        return Err(Error::Format("invalid VXT EC-Turbo header".into()));
    }
    let payload_start = r.pos;
    let padded_rows = rows.div_ceil(4) * 4;
    let row_tiles = padded_rows / 4;
    let steps = padded_cols / 256;
    let wps = TurboWhtMatrix::words_per_tile_step(bits);
    let expected_chunks = row_tiles.div_ceil(CHUNK_TILES);
    let n_chunks = r.u32()? as usize;
    if n_chunks != expected_chunks {
        return Err(Error::Format("corrupt VXT EC-Turbo chunk count".into()));
    }
    let mut chunk_slices = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        let clen = r.u32()? as usize;
        chunk_slices.push(r.take(clen)?);
    }
    let disk_bytes = r.pos - payload_start;

    let mut scales_f16 = vec![0u16; row_tiles * steps * 8];
    let mut packed_words = vec![0u32; row_tiles * steps * wps];
    let mut errs: Vec<Option<Error>> = (0..n_chunks).map(|_| None).collect();

    std::thread::scope(|s| {
        for (((ci, &bytes), (sc_ch, pw_ch)), err_slot) in chunk_slices
            .iter()
            .enumerate()
            .zip(
                scales_f16
                    .chunks_mut(CHUNK_TILES * steps * 8)
                    .zip(packed_words.chunks_mut(CHUNK_TILES * steps * wps)),
            )
            .zip(errs.iter_mut())
        {
            s.spawn(move || {
                if let Err(e) = unpack_ec_chunk(bits, ci * CHUNK_TILES, steps, bytes, sc_ch, pw_ch)
                {
                    *err_slot = Some(e);
                }
            });
        }
    });
    if let Some(e) = errs.into_iter().flatten().next() {
        return Err(e);
    }
    Ok(TurboWhtMatrix {
        bits,
        rows,
        cols,
        padded_rows,
        padded_cols,
        disk_bytes,
        scales_f16,
        packed_words,
    })
}
