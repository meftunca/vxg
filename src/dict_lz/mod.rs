//! Frame-shared prefix dictionary trainer and high-density `BLOCK_DICT_LZ` (`kind = 3`) codec.
//!
//! Combines a FastCover-inspired sampled cross-block prefix dictionary (adaptively sized from
//! 4 KiB up to 256 KiB with macroscopic verbatim span coalescing, pre-indexed once per frame and
//! shared immutably across all worker threads) with a 5-stream canonical Huffman + 3-repcode +
//! log2-offset + L1-resident `u16`-hashed LZ77 block codec.
//! 100% pure Rust, zero C dependencies (`wasm32-unknown-unknown` compatible).

use std::cell::RefCell;

use crate::{columns, Error, Result};

mod decode;
mod encode;
mod pack;
mod probe;
mod train;

pub(crate) use decode::{decode_dict_block, validate_dict_block};
pub(crate) use encode::DictLzScratch;

pub(crate) const MAX_DICT_RAW: usize = 262_144;
pub(crate) const MIN_DICT_RAW: usize = 64;
pub(crate) const DICT_HT4_BITS: usize = 13;
pub(crate) const DICT_HT4_SIZE: usize = 1 << DICT_HT4_BITS; // 8192 x [u32; 2] = 64 KiB
pub(crate) const LOCAL_HT4_BITS: usize = 13;
pub(crate) const LOCAL_HT4_SIZE: usize = 1 << LOCAL_HT4_BITS; // 8192 x [u16; 2] = 32 KiB
pub(crate) const LOCAL_HT6_BITS: usize = 13;
pub(crate) const LOCAL_HT6_SIZE: usize = 1 << LOCAL_HT6_BITS; // 8192 x [u16; 2] = 32 KiB
pub(super) const DICT_POS_MASK: u32 = 0x0007_ffff; // 19 bits (1..=262_144), top 13 bits = tag

/// Pre-indexed frame-level prefix dictionary shared immutably across worker threads.
pub(crate) struct PreparedDict {
    pub(crate) raw: Vec<u8>,
    pub(crate) wire_payload: Vec<u8>,
    pub(crate) prefer_over_grammar: bool,
    pub(crate) try_structured: bool,
    pub(crate) ht4: Box<[[u32; 4]; DICT_HT4_SIZE]>,
    pub(crate) ht6: Box<[[u32; 4]]>,
    pub(crate) ht6_shift: u32,
}

thread_local! {
    static DICT_LZ_SCRATCH: RefCell<DictLzScratch> = RefCell::new(DictLzScratch::new());
}

#[inline(always)]
pub(super) fn read_u32(buf: &[u8], pos: usize) -> u32 {
    unsafe { buf.as_ptr().add(pos).cast::<u32>().read_unaligned().to_le() }
}

#[inline(always)]
pub(super) fn read_u64(buf: &[u8], pos: usize) -> u64 {
    unsafe { buf.as_ptr().add(pos).cast::<u64>().read_unaligned().to_le() }
}

#[inline(always)]
pub(super) fn mul4(w4: u32) -> u32 {
    w4.wrapping_mul(0x9e37_79b1)
}

#[inline(always)]
pub(super) fn dict_tag4(w4: u32) -> u32 {
    w4.wrapping_mul(0x85eb_ca6b) & !DICT_POS_MASK
}

#[inline(always)]
pub(super) fn mul6(w8: u64) -> u64 {
    (w8 & 0x0000_ffff_ffff_ffff).wrapping_mul(0x9e37_79b1_85eb_ca87)
}

#[inline(always)]
pub(super) fn dict_tag6(m6: u64) -> u32 {
    ((m6 >> 16) as u32) & !DICT_POS_MASK
}

pub(super) fn build_dict_tables(
    raw: &[u8],
) -> (Box<[[u32; 4]; DICT_HT4_SIZE]>, Box<[[u32; 4]]>, u32) {
    let ht6_bits: u32 = if raw.len() <= 16_384 {
        13 // 8192 x 16 B = 128 KiB
    } else if raw.len() <= 65_536 {
        14 // 16384 x 16 B = 256 KiB
    } else if raw.len() <= 131_072 {
        15 // 32768 x 16 B = 512 KiB
    } else {
        16 // 65536 x 16 B = 1 MiB
    };
    let ht6_shift = 64 - ht6_bits;
    let mut ht4 = Box::new([[0u32; 4]; DICT_HT4_SIZE]);
    let mut ht6 = vec![[0u32; 4]; 1usize << ht6_bits].into_boxed_slice();
    if raw.len() >= 8 {
        let mut p = 0usize;
        while p + 8 <= raw.len() {
            let w8 = read_u64(raw, p);
            let w4 = w8 as u32;
            let w5 = w8 & 0x0000_00ff_ffff_ffff;
            let m6 = mul6(w8);
            let p1 = (p + 1) as u32;
            let e4 = dict_tag4(w4) | p1;
            let e6 = dict_tag6(m6) | p1;
            unsafe {
                let b4 = ht4.get_unchecked_mut((mul4(w4) >> (32 - DICT_HT4_BITS)) as usize);
                if b4[0] != 0 && (b4[0] ^ e4) <= DICT_POS_MASK {
                    let prev = ((b4[0] & DICT_POS_MASK) - 1) as usize;
                    if (read_u64(raw, prev) & 0x0000_00ff_ffff_ffff) == w5 {
                        b4[0] = e4;
                    } else {
                        *b4 = [e4, b4[0], b4[1], b4[2]];
                    }
                } else {
                    *b4 = [e4, b4[0], b4[1], b4[2]];
                }
                let b6 = ht6.get_unchecked_mut((m6 >> ht6_shift) as usize);
                *b6 = [e6, b6[0], b6[1], b6[2]];
            }
            p += 1;
        }
    }
    (ht4, ht6, ht6_shift)
}

/// Compress raw dictionary bytes for storage in a `VERSION_V5` frame header.
pub(crate) fn encode_wire_dict(raw: &[u8], level: u8) -> Vec<u8> {
    if raw.len() >= 128 {
        let mut pkt = Vec::with_capacity(raw.len());
        if encode_dict_block_into(raw, None, &mut pkt, raw.len() - 8, level).is_some() {
            return pkt;
        }
    }
    raw.to_vec()
}

/// Decode a `VERSION_V5` frame header dictionary payload back into its exact `raw_len` bytes.
pub(crate) fn decode_wire_dict(wire: &[u8], raw_len: usize) -> Result<Vec<u8>> {
    if raw_len == 0 {
        if !wire.is_empty() {
            return Err(Error::InvalidFrame("non-empty wire dict with zero raw size"));
        }
        return Ok(Vec::new());
    }
    if !(MIN_DICT_RAW..=MAX_DICT_RAW).contains(&raw_len) || wire.is_empty() || wire.len() > raw_len {
        return Err(Error::InvalidFrame("invalid frame dictionary size"));
    }
    if wire.len() == raw_len {
        return Ok(wire.to_vec());
    }
    let mut out = vec![0u8; raw_len];
    if (wire[0] & 0x80) != 0 {
        decode_dict_block(wire, &[], &mut out)?;
    } else {
        columns::decode_block_into(wire, &mut out)?;
    }
    Ok(out)
}

pub(crate) fn prepare_frame_dict(
    input: &[u8],
    block_size: usize,
    level: u8,
) -> Option<PreparedDict> {
    train::build_prepared_dict(input, block_size, level)
}

pub(crate) fn encode_dict_block_into(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    out: &mut Vec<u8>,
    max_len: usize,
    level: u8,
) -> Option<(usize, usize)> {
    if chunk.len() < 64 || max_len <= 16 {
        return None;
    }
    DICT_LZ_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        encode::encode_into(chunk, dict, &mut sc, out, max_len, level)
    })
}
