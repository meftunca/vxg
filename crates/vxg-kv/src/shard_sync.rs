//! Index-native $O(1)$ `DBSIZE`, fence-pointer pruned streaming `SCAN`/`KEYS`,
//! and horizontally sharded multi-writer WAL partitioning (`--wal-shards N`).

use std::collections::{BTreeSet, HashSet};

use crate::dispatch_xgroup::INTERNAL_KEY_PREFIX;
use crate::engine::KvEngine;
use crate::index::LookupPlan;
use crate::snapshot::{decode_wal_segment, encode_wal_segment, Manifest, WalRef};
use crate::store::PutOutcome;
use crate::types::{glob_match, now_ms, FastMap, Record};

#[inline]
pub(crate) fn is_internal_key(k: &[u8]) -> bool {
    k.starts_with(INTERNAL_KEY_PREFIX)
}

#[inline]
fn fx_hash_key(k: &[u8]) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in k {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    (h ^ (h >> 33)) as usize
}

fn shard_manifest_key(shard: usize) -> String {
    format!("shards/s{shard:02}.json")
}

/// Extract literal byte prefix before the first glob wildcard (`*`, `?`, `[`, `\`).
fn glob_literal_prefix(pat: &[u8]) -> &[u8] {
    for (i, &b) in pat.iter().enumerate() {
        if matches!(b, b'*' | b'?' | b'[' | b'\\') {
            return &pat[..i];
        }
    }
    pat
}

/// Compute the exclusive upper bound for keys starting with `prefix` (`None` if unbounded).
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None;
    }
    let mut up = prefix.to_vec();
    while let Some(last) = up.pop() {
        if last < u8::MAX {
            up.push(last + 1);
            return Some(up);
        }
    }
    None
}

impl KvEngine {
    /// Ensure a snapshot block is loaded in the LRU cache (fetches 1 Range block on miss).
    pub(crate) fn load_base_block(&mut self, blk: usize) -> Result<u64, String> {
        let Some((snap, _)) = self.base.as_ref() else {
            return Err("ERR no base snapshot".to_owned());
        };
        let epoch = snap.epoch;
        if self.cache.peek(epoch, blk).is_none() {
            self.stats.cache_misses += 1;
            let snap_c = snap.clone();
            let (decoded, approx) = self.fetch_block_cached_handle(&snap_c, blk)?;
            self.cache.insert(epoch, blk, decoded, approx);
        } else {
            self.stats.cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(epoch)
    }

    /// Check whether `key` is currently live inside `self.base` (using Bloom/Fence filter first).
    fn base_has_live_key(&mut self, key: &[u8], now: u64) -> Result<bool, String> {
        let plan = match self.base.as_ref() {
            Some((_, idx)) => idx.plan_lookup(key),
            None => return Ok(false),
        };
        let blk = match plan {
            LookupPlan::OutOfRange => {
                self.stats
                    .out_of_range_filtered
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(false);
            }
            LookupPlan::BloomFiltered => {
                self.stats
                    .bloom_filtered
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(false);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        let epoch = self.load_base_block(blk)?;
        Ok(self
            .cache
            .get(epoch, blk)
            .and_then(|m| m.get(key))
            .is_some_and(|r| r.is_live(now)))
    }

    /// Index-native $O(1)$ `DBSIZE`: returns exact live user-key count without full snapshot download!
    /// On a cold pod with clean MemTable/WAL, performs **0 HTTP Range requests (0 bytes fetched)**!
    pub fn dbsize_fast(&mut self) -> Result<usize, String> {
        let now = now_ms();
        let mut count: i64 = match self.base.as_ref() {
            Some((snap, _)) => snap.key_count as i64,
            None => 0,
        };
        // Subtract any internal metadata keys (always sorted into Block 0 since prefix starts with 0x00)
        let maybe_internal = self.base.as_ref().is_some_and(|(_, idx)| {
            idx.blocks
                .first()
                .is_some_and(|b| b.min_key.starts_with(INTERNAL_KEY_PREFIX))
        });
        if maybe_internal {
            let epoch = self.load_base_block(0)?;
            if let Some(m) = self.cache.get(epoch, 0) {
                let n_int = m
                    .iter()
                    .filter(|(k, r)| is_internal_key(k) && r.is_live(now))
                    .count() as i64;
                count -= n_int;
            }
        }
        if self.mem.is_empty() && self.wal_overlay.is_empty() {
            return Ok(count.max(0) as usize);
        }
        // Collect unique user keys touched by MemTable or WAL overlay
        let mut touched: Vec<(Vec<u8>, bool)> =
            Vec::with_capacity(self.mem.len() + self.wal_overlay.len());
        for (k, r) in &self.mem {
            if !is_internal_key(k) {
                touched.push((k.clone(), r.is_live(now)));
            }
        }
        for (k, r) in &self.wal_overlay {
            if !is_internal_key(k) && !self.mem.contains_key(k) {
                touched.push((k.clone(), r.is_live(now)));
            }
        }
        for (k, live_now) in touched {
            let in_base = self.base_has_live_key(&k, now)?;
            if live_now && !in_base {
                count += 1;
            } else if !live_now && in_base {
                count -= 1;
            }
        }
        Ok(count.max(0) as usize)
    }

    /// Collect candidate block indices from `SnapshotIndex` whose `[min_key..=max_key]`
    /// can overlap keys matching `pattern`.
    fn candidate_blocks_for_pattern(&self, pattern: &[u8]) -> Vec<(usize, Vec<u8>)> {
        let Some((_, idx)) = self.base.as_ref() else {
            return Vec::new();
        };
        let prefix = glob_literal_prefix(pattern);
        let upper = prefix_upper_bound(prefix);
        let mut out = Vec::new();
        for b in &idx.blocks {
            if !prefix.is_empty() && b.max_key.as_slice() < prefix {
                continue;
            }
            if let Some(ref up) = upper {
                if b.min_key.as_slice() >= up.as_slice() {
                    break;
                }
            }
            out.push((b.block_idx as usize, b.min_key.clone()));
        }
        out
    }

    /// Fence-pruned streaming `SCAN cursor [MATCH pat] [COUNT n]`.
    /// Skips non-overlapping snapshot blocks via `SnapshotIndex` and stops fetching blocks as soon
    /// as the requested `[cursor..cursor+count]` slice is proven complete!
    pub fn scan_pruned(
        &mut self,
        cursor: usize,
        pattern: &[u8],
        count: usize,
    ) -> Result<(usize, Vec<Vec<u8>>), String> {
        let now = now_ms();
        let match_all = pattern == b"*";
        let mut live: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut dead: HashSet<Vec<u8>> = HashSet::new();

        for (k, r) in &self.wal_overlay {
            if is_internal_key(k) || self.mem.contains_key(k) {
                continue;
            }
            if r.is_live(now) {
                if match_all || glob_match(pattern, k) {
                    live.insert(k.clone());
                }
            } else {
                dead.insert(k.clone());
            }
        }
        for (k, r) in &self.mem {
            if is_internal_key(k) {
                continue;
            }
            if r.is_live(now) {
                if match_all || glob_match(pattern, k) {
                    live.insert(k.clone());
                }
            } else {
                dead.insert(k.clone());
            }
        }

        let need = cursor.saturating_add(count).saturating_add(1);
        let cands = self.candidate_blocks_for_pattern(pattern);
        let mut stopped_early = false;

        for (blk_idx, min_key) in cands {
            // Early-stop invariant: if we already hold `need` sorted keys strictly `< min_key`,
            // no subsequent snapshot block can ever enter `0..need`!
            if live.len() >= need {
                if let Some(nth) = live.iter().nth(need - 1) {
                    if nth.as_slice() < min_key.as_slice() {
                        stopped_early = true;
                        break;
                    }
                }
            }
            let epoch = self.load_base_block(blk_idx)?;
            if let Some(m) = self.cache.get(epoch, blk_idx) {
                for (k, r) in m {
                    if is_internal_key(k) || dead.contains(k) || !r.is_live(now) {
                        continue;
                    }
                    if match_all || glob_match(pattern, k) {
                        live.insert(k.clone());
                    }
                }
            }
        }

        let total = live.len();
        let batch: Vec<Vec<u8>> = live.into_iter().skip(cursor).take(count).collect();
        let next_cursor = if stopped_early || cursor + batch.len() < total {
            cursor + batch.len()
        } else {
            0
        };
        Ok((next_cursor, batch))
    }

    /// Fence-pruned `KEYS pattern`: downloads ONLY snapshot blocks overlapping the pattern's prefix!
    pub fn matching_keys_pruned(&mut self, pattern: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let (_, all) = self.scan_pruned(0, pattern, usize::MAX / 2)?;
        Ok(all)
    }

    // =========================================================================
    // Sharded Multi-Writer WAL Partitioning (`--wal-shards N`, N > 1)
    // =========================================================================

    pub(crate) fn pull_one_shard(&mut self, shard: usize) -> Result<usize, String> {
        let skey = shard_manifest_key(shard);
        let Some(meta) = self.store.head(&skey)? else {
            return Ok(0);
        };
        if self.shard_etags[shard].as_deref() == Some(meta.etag.as_str()) {
            return Ok(0);
        }
        let Some((bytes, etag)) = self.store.get(&skey)? else {
            return Ok(0);
        };
        let remote = Manifest::from_bytes(&bytes)?;
        let mut applied = 0usize;
        for w in &remote.wals {
            if self.applied_shard_wals[shard].contains(&w.epoch) {
                continue;
            }
            let Some((wb, _)) = self.store.get(&w.key)? else {
                continue;
            };
            let (recs, events) = decode_wal_segment(&wb)?;
            for (k, r) in recs {
                let ver = r.version.max(self.next_ver + 1);
                self.next_ver = self.next_ver.max(ver);
                if !self.key_versions.is_empty() {
                    if let Some(slot) = self.key_versions.get_mut(&k) {
                        *slot = ver;
                    }
                }
                self.wal_overlay.insert(k, r);
            }
            for ev in events {
                if ev.origin_pod != self.cfg.pod_id {
                    self.broker.publish_local(&ev.channel, &ev.payload);
                    self.stats.pubsub_remote_delivered += 1;
                }
            }
            self.applied_shard_wals[shard].insert(w.epoch);
            applied += 1;
        }
        self.shard_manifests[shard] = remote;
        self.shard_etags[shard] = Some(etag);
        Ok(applied)
    }

    pub(crate) fn pull_all_shards(&mut self) -> Result<usize, String> {
        let n = self.shard_manifests.len();
        let mut total = 0usize;
        for s in 0..n {
            total += self.pull_one_shard(s)?;
        }
        Ok(total)
    }

    /// Horizontally partitioned WAL commit across `N` independent shard manifests (`shards/sXX.json`).
    pub(crate) fn commit_sharded(&mut self) -> Result<u64, String> {
        let n = self.shard_manifests.len();
        let mut buckets: Vec<FastMap<Vec<u8>, Record>> = (0..n).map(|_| FastMap::default()).collect();
        for (k, r) in std::mem::take(&mut self.mem) {
            let s = fx_hash_key(&k) % n;
            buckets[s].insert(k, r);
        }
        let events = std::mem::take(&mut self.pending_events);
        let mut max_epoch = self.manifest.epoch;

        for (s, bucket) in buckets.into_iter().enumerate() {
            let shard_events = if s == 0 { &events[..] } else { &[] };
            if bucket.is_empty() && shard_events.is_empty() {
                continue;
            }
            let skey = shard_manifest_key(s);
            let mut committed = false;
            for _ in 0..16 {
                let next_ep = self.shard_manifests[s].epoch + 1;
                let wal_bytes = encode_wal_segment(&bucket, shard_events)?;
                let wal_key = format!("wal/s{s:02}_wal_{next_ep:08}_{}.vxg", self.cfg.pod_id);
                if matches!(
                    self.store.put_if_none_match(&wal_key, &wal_bytes)?,
                    PutOutcome::PreconditionFailed
                ) {
                    self.stats.cas_retries += 1;
                    self.pull_one_shard(s)?;
                    continue;
                }
                let mut next_m = self.shard_manifests[s].clone();
                next_m.epoch = next_ep;
                next_m.wals.push(WalRef {
                    epoch: next_ep,
                    key: wal_key,
                    pod_id: self.cfg.pod_id.clone(),
                    compressed_bytes: wal_bytes.len() as u64,
                    created_ms: now_ms(),
                });
                let raw_m = next_m.to_bytes();
                let outcome = match &self.shard_etags[s] {
                    None => self.store.put_if_none_match(&skey, &raw_m)?,
                    Some(et) => self.store.put_if_match(&skey, et, &raw_m)?,
                };
                match outcome {
                    PutOutcome::Written(new_etag) => {
                        for (k, r) in bucket {
                            self.wal_overlay.insert(k, r);
                        }
                        self.applied_shard_wals[s].insert(next_ep);
                        self.shard_manifests[s] = next_m;
                        self.shard_etags[s] = Some(new_etag);
                        self.stats.cas_commits += 1;
                        max_epoch = max_epoch.max(next_ep);
                        committed = true;
                        break;
                    }
                    PutOutcome::PreconditionFailed => {
                        self.stats.cas_retries += 1;
                        self.pull_one_shard(s)?;
                    }
                }
            }
            if !committed {
                return Err(format!("ERR sharded commit failed on shard {s}"));
            }
        }
        Ok(max_epoch)
    }

    /// Reset shard WAL lists cleanly after full compaction or `FLUSHALL`.
    pub(crate) fn clear_all_shards(&mut self) -> Result<(), String> {
        let n = self.shard_manifests.len();
        for s in 0..n {
            if self.shard_manifests[s].wals.is_empty() && self.shard_etags[s].is_none() {
                continue;
            }
            let skey = shard_manifest_key(s);
            let mut next_m = self.shard_manifests[s].clone();
            next_m.epoch += 1;
            next_m.wals.clear();
            let raw_m = next_m.to_bytes();
            let etag = self.store.put(&skey, &raw_m)?;
            self.shard_manifests[s] = next_m;
            self.shard_etags[s] = Some(etag);
            self.applied_shard_wals[s].clear();
        }
        Ok(())
    }

    /// Total active WAL segments across main manifest and any sharded manifests.
    pub fn active_wal_count(&self) -> usize {
        let shard_wals: usize = self.shard_manifests.iter().map(|m| m.wals.len()).sum();
        self.manifest.wals.len() + shard_wals
    }
}
