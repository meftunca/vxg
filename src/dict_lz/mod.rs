//! Frame-shared prefix dictionary trainer and high-density `BLOCK_DICT_LZ` (`kind = 3`) codec.
//!
//! Combines a FastCover-inspired sampled cross-block prefix dictionary (up to 32 KiB, pre-indexed
//! once per frame and shared immutably across all worker threads) with a 5-stream canonical
//! Huffman + 3-repcode + log2-offset + lazy/backward-extended LZ77 block codec.
//! 100% pure Rust, zero C dependencies (`wasm32-unknown-unknown` compatible).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{columns, Error, Result};

mod decode;
mod encode;
mod pack;
mod train;

pub(crate) use decode::{decode_dict_block, validate_dict_block};
pub(crate) use encode::DictLzScratch;

pub(crate) const MAX_DICT_RAW: usize = 32_768;
pub(crate) const MIN_DICT_RAW: usize = 64;
pub(crate) const HT4_BITS: usize = 12;
pub(crate) const HT4_SIZE: usize = 1 << HT4_BITS; // 4096 x 4 x u32 = 64 KiB
pub(crate) const HT6_BITS: usize = 12;
pub(crate) const HT6_SIZE: usize = 1 << HT6_BITS; // 4096 x 2 x u32 = 32 KiB

static NEXT_DICT_ID: AtomicU64 = AtomicU64::new(1);

/// Pre-indexed frame-level prefix dictionary shared immutably across worker threads.
pub(crate) struct PreparedDict {
    pub(crate) id: u64,
    pub(crate) raw: Vec<u8>,
    pub(crate) wire_payload: Vec<u8>,
    pub(crate) prefer_over_grammar: bool,
    pub(crate) try_structured: bool,
    pub(crate) ht4: Box<[[u32; 4]; HT4_SIZE]>,
    pub(crate) ht6: Box<[[u32; 2]; HT6_SIZE]>,
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
pub(super) fn hash4(w4: u32) -> usize {
    (w4.wrapping_mul(0x9e37_79b1) >> (32 - HT4_BITS)) as usize
}

#[inline(always)]
pub(super) fn hash6(w8: u64) -> usize {
    ((w8 & 0x0000_ffff_ffff_ffff).wrapping_mul(0x9e37_79b1_85eb_ca87) >> (64 - HT6_BITS)) as usize
}

#[inline(always)]
pub(super) fn insert_pos(
    win: &[u8],
    vpos: usize,
    ht4: &mut [[u32; 4]; HT4_SIZE],
    ht6: &mut [[u32; 2]; HT6_SIZE],
) {
    let w8 = read_u64(win, vpos);
    let p1 = (vpos + 1) as u32;
    let b4 = unsafe { ht4.get_unchecked_mut(hash4(w8 as u32)) };
    *b4 = [p1, b4[0], b4[1], b4[2]];
    let b6 = unsafe { ht6.get_unchecked_mut(hash6(w8)) };
    *b6 = [p1, b6[0]];
}

pub(super) fn index_dict_tables(
    raw: &[u8],
    ht4: &mut [[u32; 4]; HT4_SIZE],
    ht6: &mut [[u32; 2]; HT6_SIZE],
) {
    if raw.len() < 8 {
        return;
    }
    let mut p = 0usize;
    while p + 8 <= raw.len() {
        insert_pos(raw, p, ht4, ht6);
        p += 1;
    }
}

/// Compress raw dictionary bytes for storage in a `VERSION_V5` frame header.
pub(crate) fn encode_wire_dict(raw: &[u8]) -> Vec<u8> {
    if raw.len() >= 128 {
        let mut pkt = Vec::with_capacity(raw.len());
        if encode_dict_block_into(raw, None, &mut pkt, raw.len() - 8).is_some() {
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

pub(crate) fn prepare_frame_dict(input: &[u8], block_size: usize) -> Option<PreparedDict> {
    train::build_prepared_dict(input, block_size, &NEXT_DICT_ID)
}

pub(crate) fn encode_dict_block_into(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    out: &mut Vec<u8>,
    max_len: usize,
) -> Option<(usize, usize)> {
    if chunk.len() < 64 || max_len <= 16 {
        return None;
    }
    DICT_LZ_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        encode::encode_into(chunk, dict, &mut sc, out, max_len)
    })
}

#[inline(always)]
pub(crate) fn next_id() -> u64 {
    NEXT_DICT_ID.fetch_add(1, Ordering::Relaxed)
}
