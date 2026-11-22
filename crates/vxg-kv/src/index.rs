//! Snapshot Fence-Pointer Index (`VXI1`), per-block 10-bit/key Bloom Filters, and
//! byte-budgeted LRU decoded block cache.

use std::collections::{BTreeMap, HashMap};

use crate::types::Record;

const MAGIC_VXI1: &[u8; 4] = b"VXI1";

#[derive(Clone, Debug)]
pub struct BlockIndexEntry {
    pub block_idx: u32,
    pub row_count: u32,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub k_hashes: u8,
    pub bloom: Vec<u8>,
}

impl BlockIndexEntry {
    pub fn build(block_idx: u32, keys: &[&[u8]]) -> Self {
        let min_key = keys.first().map(|k| k.to_vec()).unwrap_or_default();
        let max_key = keys.last().map(|k| k.to_vec()).unwrap_or_default();
        let bits = (keys.len().max(1) * 10).next_multiple_of(8).max(64);
        let mut bloom = vec![0u8; bits / 8];
        let k_hashes = 7u8;
        for &k in keys {
            bloom_insert(&mut bloom, k_hashes, k);
        }
        Self {
            block_idx,
            row_count: keys.len() as u32,
            min_key,
            max_key,
            k_hashes,
            bloom,
        }
    }

    #[inline]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        bloom_contains(&self.bloom, self.k_hashes, key)
    }
}

#[inline]
fn hash_pair(key: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x9e3779b97f4a7c15;
    for &b in key {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 = h2.wrapping_add(b as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        h2 ^= h2 >> 27;
    }
    h1 ^= h1 >> 33;
    h2 |= 1; // ensure odd stride
    (h1, h2)
}

fn bloom_insert(bits: &mut [u8], k: u8, key: &[u8]) {
    let m = (bits.len() * 8) as u64;
    if m == 0 {
        return;
    }
    let (h1, h2) = hash_pair(key);
    for i in 0..k as u64 {
        let idx = (h1.wrapping_add(i.wrapping_mul(h2)) % m) as usize;
        bits[idx >> 3] |= 1u8 << (idx & 7);
    }
}

fn bloom_contains(bits: &[u8], k: u8, key: &[u8]) -> bool {
    let m = (bits.len() * 8) as u64;
    if m == 0 {
        return true;
    }
    let (h1, h2) = hash_pair(key);
    for i in 0..k as u64 {
        let idx = (h1.wrapping_add(i.wrapping_mul(h2)) % m) as usize;
        if (bits[idx >> 3] & (1u8 << (idx & 7))) == 0 {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupPlan {
    /// Key falls outside all block ranges (0 Range I/O).
    OutOfRange,
    /// Key fell inside a block range, but Bloom filter definitively ruled it out (0 Range I/O).
    BloomFiltered,
    /// Fetch this exact VXG block index (1 Range request).
    CandidateBlock(usize),
}

/// Complete fence-pointer + Bloom filter index for one immutable sorted `.vxg` snapshot.
#[derive(Clone, Debug, Default)]
pub struct SnapshotIndex {
    pub total_keys: u64,
    pub blocks: Vec<BlockIndexEntry>,
}

impl SnapshotIndex {
    /// $O(\log B)$ fence-pointer binary search + per-block Bloom filter precheck.
    pub fn plan_lookup(&self, key: &[u8]) -> LookupPlan {
        if self.blocks.is_empty() {
            return LookupPlan::OutOfRange;
        }
        // Find rightmost block with `min_key <= key`
        let pos = self.blocks.partition_point(|b| b.min_key.as_slice() <= key);
        if pos == 0 {
            return LookupPlan::OutOfRange;
        }
        let blk = &self.blocks[pos - 1];
        if key > blk.max_key.as_slice() {
            return LookupPlan::OutOfRange;
        }
        if !blk.may_contain(key) {
            return LookupPlan::BloomFiltered;
        }
        LookupPlan::CandidateBlock(blk.block_idx as usize)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.blocks.len() * 128);
        out.extend_from_slice(MAGIC_VXI1);
        out.extend_from_slice(&self.total_keys.to_le_bytes());
        out.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for b in &self.blocks {
            out.extend_from_slice(&b.block_idx.to_le_bytes());
            out.extend_from_slice(&b.row_count.to_le_bytes());
            out.extend_from_slice(&(b.min_key.len() as u32).to_le_bytes());
            out.extend_from_slice(&b.min_key);
            out.extend_from_slice(&(b.max_key.len() as u32).to_le_bytes());
            out.extend_from_slice(&b.max_key);
            out.push(b.k_hashes);
            out.extend_from_slice(&(b.bloom.len() as u32).to_le_bytes());
            out.extend_from_slice(&b.bloom);
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..4] != MAGIC_VXI1 {
            return Err("invalid VXI1 snapshot index".to_owned());
        }
        let mut pos = 4usize;
        let read_u32 = |p: &mut usize| -> Result<u32, String> {
            let s = bytes.get(*p..*p + 4).ok_or("truncated VXI1 u32")?;
            *p += 4;
            Ok(u32::from_le_bytes(s.try_into().unwrap()))
        };
        let total_keys = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let count = read_u32(&mut pos)? as usize;
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block_idx = read_u32(&mut pos)?;
            let row_count = read_u32(&mut pos)?;
            let min_len = read_u32(&mut pos)? as usize;
            let min_key = bytes
                .get(pos..pos + min_len)
                .ok_or("truncated VXI1 min_key")?
                .to_vec();
            pos += min_len;
            let max_len = read_u32(&mut pos)? as usize;
            let max_key = bytes
                .get(pos..pos + max_len)
                .ok_or("truncated VXI1 max_key")?
                .to_vec();
            pos += max_len;
            let k_hashes = *bytes.get(pos).ok_or("truncated VXI1 k_hashes")?;
            pos += 1;
            let bloom_len = read_u32(&mut pos)? as usize;
            let bloom = bytes
                .get(pos..pos + bloom_len)
                .ok_or("truncated VXI1 bloom")?
                .to_vec();
            pos += bloom_len;
            blocks.push(BlockIndexEntry {
                block_idx,
                row_count,
                min_key,
                max_key,
                k_hashes,
                bloom,
            });
        }
        Ok(Self { total_keys, blocks })
    }
}

/// Decoded contents of one VXG snapshot block cached in pod RAM.
pub type DecodedBlock = BTreeMap<Vec<u8>, Record>;

struct CacheEntry {
    records: DecodedBlock,
    approx_bytes: usize,
    last_tick: u64,
}

/// Byte-budgeted LRU cache keyed by `(snapshot_epoch, block_idx)`.
pub struct LruBlockCache {
    max_bytes: usize,
    cur_bytes: usize,
    tick: u64,
    map: HashMap<(u64, usize), CacheEntry>,
}

impl LruBlockCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes: max_bytes.max(1024 * 1024),
            cur_bytes: 0,
            tick: 0,
            map: HashMap::new(),
        }
    }

    /// Lock-free read-only cache lookup (no LRU tick update; ideal under `RwLock::read`).
    #[inline]
    pub fn peek(&self, epoch: u64, block_idx: usize) -> Option<&DecodedBlock> {
        self.map.get(&(epoch, block_idx)).map(|e| &e.records)
    }

    pub fn get(&mut self, epoch: u64, block_idx: usize) -> Option<&DecodedBlock> {
        self.tick += 1;
        let t = self.tick;
        let e = self.map.get_mut(&(epoch, block_idx))?;
        e.last_tick = t;
        Some(&e.records)
    }

    pub fn insert(
        &mut self,
        epoch: u64,
        block_idx: usize,
        records: DecodedBlock,
        approx_bytes: usize,
    ) {
        self.tick += 1;
        if let Some(old) = self.map.remove(&(epoch, block_idx)) {
            self.cur_bytes = self.cur_bytes.saturating_sub(old.approx_bytes);
        }
        while self.cur_bytes + approx_bytes > self.max_bytes && !self.map.is_empty() {
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, v)| v.last_tick)
                .map(|(k, _)| *k);
            if let Some(vk) = victim {
                if let Some(removed) = self.map.remove(&vk) {
                    self.cur_bytes = self.cur_bytes.saturating_sub(removed.approx_bytes);
                }
            } else {
                break;
            }
        }
        self.cur_bytes += approx_bytes;
        self.map.insert(
            (epoch, block_idx),
            CacheEntry {
                records,
                approx_bytes,
                last_tick: self.tick,
            },
        );
    }

    pub fn bytes_used(&self) -> usize {
        self.cur_bytes
    }

    pub fn cached_blocks(&self) -> usize {
        self.map.len()
    }
}
