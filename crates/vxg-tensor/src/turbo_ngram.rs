//! L1-resident (`32 KiB`) lossless Prompt-Lookup + Output N-gram drafter (`M = 4` bursts)
//! for `MetalTurboRunner`. Proposes 3-token continuations `[d1, d2, d3]` verified in a single
//! `batch_m = 4` pass on the exact target model (`100%` greedy token-for-token equivalence).

const TABLE_BITS: usize = 11;
const TABLE_LEN: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_LEN - 1;
const EMPTY_POS: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub(crate) enum DraftKey {
    Tri(usize, u64),
    Bi(usize, u64),
}

pub(crate) struct TurboNgramTable {
    prompt_len: usize,
    hist: Vec<u32>,
    indexed_end: usize,
    tri: Vec<(u64, u32)>,
    bi: Vec<(u64, u32)>,
}

#[inline]
fn key3(a: u32, b: u32, c: u32) -> u64 {
    ((a as u64) << 42) ^ ((b as u64) << 21) ^ (c as u64) ^ 0x9e37_79b9_7f4a_7c15
}

#[inline]
fn hash3(k: u64) -> usize {
    (k.wrapping_mul(0xbf58_476d_1ce4_e5b9) >> (64 - TABLE_BITS)) as usize & TABLE_MASK
}

#[inline]
fn key2(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline]
fn hash2(k: u64) -> usize {
    (k.wrapping_mul(0x94d0_49bb_1331_11eb) >> (64 - TABLE_BITS)) as usize & TABLE_MASK
}

impl TurboNgramTable {
    pub(crate) fn new(prompt_ids: &[u32], first_tok: u32) -> Self {
        let mut hist = Vec::with_capacity(prompt_ids.len() + 512);
        hist.extend_from_slice(prompt_ids);
        hist.push(first_tok);
        let mut s = Self {
            prompt_len: prompt_ids.len(),
            hist,
            indexed_end: 0,
            tri: vec![(0u64, EMPTY_POS); TABLE_LEN],
            bi: vec![(0u64, EMPTY_POS); TABLE_LEN],
        };
        s.advance_index();
        s
    }

    #[inline]
    pub(crate) fn sync_generated(&mut self, gen_ids: &[u32]) {
        let have_gen = self.hist.len().saturating_sub(self.prompt_len);
        if gen_ids.len() > have_gen {
            self.hist.extend_from_slice(&gen_ids[have_gen..]);
            self.advance_index();
        }
    }

    fn advance_index(&mut self) {
        let ready = self.hist.len().saturating_sub(3);
        while self.indexed_end < ready {
            let i = self.indexed_end;
            let t2 = self.hist[i];
            if i >= 1 {
                let t1 = self.hist[i - 1];
                let k2 = key2(t1, t2);
                self.bi[hash2(k2)] = (k2, i as u32);
                if i >= 2 {
                    let t0 = self.hist[i - 2];
                    let k3 = key3(t0, t1, t2);
                    self.tri[hash3(k3)] = (k3, i as u32);
                }
            }
            self.indexed_end += 1;
        }
    }

    #[inline]
    pub(crate) fn draft3(&self, allow_bigram: bool, stop_ids: &[u32]) -> Option<([u32; 3], DraftKey)> {
        let n = self.hist.len();
        if n >= 3 {
            let k3 = key3(self.hist[n - 3], self.hist[n - 2], self.hist[n - 1]);
            let slot = hash3(k3);
            let (sk, pos) = self.tri[slot];
            let idx = pos as usize;
            if pos != EMPTY_POS && sk == k3 && idx + 3 < n {
                let d = [self.hist[idx + 1], self.hist[idx + 2], self.hist[idx + 3]];
                if !d.iter().any(|t| stop_ids.contains(t)) {
                    return Some((d, DraftKey::Tri(slot, k3)));
                }
            }
        }
        if allow_bigram && n >= 2 {
            let k2 = key2(self.hist[n - 2], self.hist[n - 1]);
            let slot = hash2(k2);
            let (sk, pos) = self.bi[slot];
            let idx = pos as usize;
            if pos != EMPTY_POS && sk == k2 && idx + 3 < n {
                let d = [self.hist[idx + 1], self.hist[idx + 2], self.hist[idx + 3]];
                if !d.iter().any(|t| stop_ids.contains(t)) {
                    return Some((d, DraftKey::Bi(slot, k2)));
                }
            }
        }
        None
    }

    #[inline]
    pub(crate) fn invalidate(&mut self, dk: DraftKey) {
        match dk {
            DraftKey::Tri(slot, k) => {
                if self.tri[slot].0 == k {
                    self.tri[slot].1 = EMPTY_POS;
                }
            }
            DraftKey::Bi(slot, k) => {
                if self.bi[slot].0 == k {
                    self.bi[slot].1 = EMPTY_POS;
                }
            }
        }
    }
}
