//! Core `KvEngine`: LSM MemTable + VXG WAL Overlay + Remote Range Snapshot + Bloom/LRU Cache
//! + Atomic ETag CAS Multi-Pod Commit/Pull/Compaction + Time-Travel + `VXG.SQL`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use vxg::{DecodeOptions, Options, RemoteFrame, SqlFormat};

use crate::index::{DecodedBlock, LookupPlan, LruBlockCache, SnapshotIndex};
use crate::pubsub::{PubSubBroker, PubSubEvent};
use crate::snapshot::{
    decode_wal_segment, encode_wal_segment, execute_vxg_sql_on_frame, read_full_snapshot,
    write_snapshot, Manifest, SnapshotRef, WalRef, MANIFEST_KEY,
};
use crate::store::{ObjectStore, PutOutcome, StoreRangeReader};
use crate::types::{decode_record_line, encode_record_line, glob_match, now_ms, Record, Value};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub pod_id: String,
    pub cache_bytes: usize,
    pub block_size: usize,
    pub auto_compact_wals: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            pod_id: format!("pod-{}", std::process::id()),
            cache_bytes: 64 * 1024 * 1024,
            block_size: 16_384,
            auto_compact_wals: 16,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub range_requests: u64,
    pub range_bytes: u64,
    pub bloom_filtered: u64,
    pub out_of_range_filtered: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cas_commits: u64,
    pub cas_retries: u64,
    pub compactions: u64,
    pub pulls: u64,
    pub pubsub_remote_delivered: u64,
}

pub struct KvEngine {
    pub cfg: EngineConfig,
    pub store: Arc<dyn ObjectStore>,
    pub broker: Arc<PubSubBroker>,
    pub manifest: Manifest,
    pub manifest_etag: Option<String>,
    pub mem: BTreeMap<Vec<u8>, Record>,
    pub wal_overlay: BTreeMap<Vec<u8>, Record>,
    pub applied_wals: BTreeSet<u64>,
    pub pending_events: Vec<PubSubEvent>,
    pub base: Option<(SnapshotRef, SnapshotIndex)>,
    pub hist_indices: HashMap<u64, SnapshotIndex>,
    remote_handle: Option<(u64, RemoteFrame<StoreRangeReader>)>,
    pub cache: LruBlockCache,
    pub next_ver: u64,
    pub key_versions: HashMap<Vec<u8>, u64>,
    pub stats: EngineStats,
}

impl KvEngine {
    pub fn open(
        store: Arc<dyn ObjectStore>,
        broker: Arc<PubSubBroker>,
        cfg: EngineConfig,
    ) -> Result<Self, String> {
        let cache = LruBlockCache::new(cfg.cache_bytes);
        let mut eng = Self {
            cfg,
            store,
            broker,
            manifest: Manifest::default(),
            manifest_etag: None,
            mem: BTreeMap::new(),
            wal_overlay: BTreeMap::new(),
            applied_wals: BTreeSet::new(),
            pending_events: Vec::new(),
            base: None,
            hist_indices: HashMap::new(),
            remote_handle: None,
            cache,
            next_ver: 1,
            key_versions: HashMap::new(),
            stats: EngineStats::default(),
        };
        eng.pull()?;
        Ok(eng)
    }

    pub fn has_dirty(&self) -> bool {
        !self.mem.is_empty() || !self.pending_events.is_empty()
    }

    pub fn key_version(&self, key: &[u8]) -> u64 {
        self.key_versions.get(key).copied().unwrap_or(0)
    }

    fn bump_ver(&mut self, key: &[u8]) -> u64 {
        self.next_ver += 1;
        let v = self.next_ver;
        self.key_versions.insert(key.to_vec(), v);
        v
    }

    /// Look up the current live record for `key` (MemTable -> WAL overlay -> Base Snapshot Range).
    pub fn get_record(&mut self, key: &[u8]) -> Result<Option<Record>, String> {
        let now = now_ms();
        if let Some(rec) = self.mem.get(key) {
            return Ok(rec.is_live(now).then(|| rec.clone()));
        }
        if let Some(rec) = self.wal_overlay.get(key) {
            return Ok(rec.is_live(now).then(|| rec.clone()));
        }
        let (epoch, plan) = match self.base.as_ref() {
            Some((snap, idx)) => (snap.epoch, idx.plan_lookup(key)),
            None => return Ok(None),
        };
        let blk = match plan {
            LookupPlan::OutOfRange => {
                self.stats.out_of_range_filtered += 1;
                return Ok(None);
            }
            LookupPlan::BloomFiltered => {
                self.stats.bloom_filtered += 1;
                return Ok(None);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        if let Some(decoded) = self.cache.get(epoch, blk) {
            self.stats.cache_hits += 1;
            return Ok(decoded.get(key).filter(|r| r.is_live(now)).cloned());
        }
        self.stats.cache_misses += 1;
        let snap = self.base.as_ref().unwrap().0.clone();
        let (decoded, approx) = self.fetch_block_cached_handle(&snap, blk)?;
        let res = decoded.get(key).filter(|r| r.is_live(now)).cloned();
        self.cache.insert(epoch, blk, decoded, approx);
        Ok(res)
    }

    /// Time-Travel lookup: read `key` strictly as it existed in historical snapshot `epoch`.
    pub fn get_record_at_epoch(&mut self, epoch: u64, key: &[u8]) -> Result<Option<Record>, String> {
        let Some(snap) = self.manifest.find_snapshot(epoch).cloned() else {
            return Err(format!("ERR snapshot epoch {epoch} not found"));
        };
        if !self.hist_indices.contains_key(&epoch) {
            let Some((raw_idx, _)) = self.store.get(&snap.idx_key)? else {
                return Err(format!("ERR missing index {}", snap.idx_key));
            };
            let decoded = SnapshotIndex::decode(&raw_idx)?;
            self.hist_indices.insert(epoch, decoded);
        }
        let plan = self.hist_indices.get(&epoch).unwrap().plan_lookup(key);
        let blk = match plan {
            LookupPlan::OutOfRange => {
                self.stats.out_of_range_filtered += 1;
                return Ok(None);
            }
            LookupPlan::BloomFiltered => {
                self.stats.bloom_filtered += 1;
                return Ok(None);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        if let Some(decoded) = self.cache.get(epoch, blk) {
            self.stats.cache_hits += 1;
            return Ok(decoded.get(key).cloned());
        }
        self.stats.cache_misses += 1;
        let (decoded, approx) = self.fetch_block_cached_handle(&snap, blk)?;
        let res = decoded.get(key).cloned();
        self.cache.insert(epoch, blk, decoded, approx);
        Ok(res)
    }

    fn fetch_block_cached_handle(
        &mut self,
        snap: &SnapshotRef,
        blk: usize,
    ) -> Result<(DecodedBlock, usize), String> {
        let need_open = self
            .remote_handle
            .as_ref()
            .is_none_or(|(ep, _)| *ep != snap.epoch);
        if need_open {
            let reader = StoreRangeReader::open(
                Arc::clone(&self.store),
                &snap.vxg_key,
                snap.compressed_bytes,
            );
            let rf = RemoteFrame::open(reader).map_err(|e| format!("ERR remote open: {e}"))?;
            let st = rf.stats();
            self.stats.range_requests += st.requests as u64;
            self.stats.range_bytes += st.bytes_fetched;
            self.remote_handle = Some((snap.epoch, rf));
        }
        let (_, rf) = self.remote_handle.as_mut().unwrap();
        let before = rf.stats();
        let raw = rf
            .extract_block(blk, &DecodeOptions::default())
            .map_err(|e| format!("ERR remote extract_block({blk}): {e}"))?;
        let after = rf.stats();
        self.stats.range_requests += (after.requests - before.requests) as u64;
        self.stats.range_bytes += after.bytes_fetched - before.bytes_fetched;
        let approx = raw.len().max(128);
        let mut map = BTreeMap::new();
        for line in raw.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Some((k, rec)) = decode_record_line(line) {
                map.insert(k, rec);
            }
        }
        Ok((map, approx))
    }

    /// Insert or update a live record in MemTable.
    pub fn set_record(&mut self, key: Vec<u8>, value: Value, expires_at_ms: Option<u64>) {
        let v = self.bump_ver(&key);
        self.mem.insert(key, Record::live(value, expires_at_ms, v));
    }

    /// Delete a key (writing a tombstone if it existed anywhere). Returns true if key existed.
    pub fn del_key(&mut self, key: &[u8]) -> Result<bool, String> {
        let existed = self.get_record(key)?.is_some();
        let v = self.bump_ver(key);
        if existed || self.wal_overlay.contains_key(key) || self.base.is_some() {
            self.mem.insert(key.to_vec(), Record::tombstone(v));
        } else {
            self.mem.remove(key);
        }
        Ok(existed)
    }

    /// Update expiry of an existing live key. Returns true if key exists and was updated.
    pub fn set_expiry(&mut self, key: &[u8], expires_at_ms: Option<u64>) -> Result<bool, String> {
        let Some(mut rec) = self.get_record(key)? else {
            return Ok(false);
        };
        rec.expires_at_ms = expires_at_ms;
        rec.version = self.bump_ver(key);
        rec.updated_ms = now_ms();
        self.mem.insert(key.to_vec(), rec);
        Ok(true)
    }

    /// Queue a cross-pod Pub/Sub event and deliver immediately to local subscribers.
    pub fn publish(&mut self, channel: Vec<u8>, payload: Vec<u8>) -> usize {
        let local = self.broker.publish_local(&channel, &payload);
        self.pending_events.push(PubSubEvent {
            origin_pod: self.cfg.pod_id.clone(),
            channel,
            payload,
        });
        local
    }

    /// Linearizable cross-pod Compare-And-Swap (`VXG.CAS key expected new_val [exp]`).
    /// Pulls latest state, checks expected (`None` = must not exist), writes, and CAS-commits.
    pub fn cas_string(
        &mut self,
        key: Vec<u8>,
        expected: Option<&[u8]>,
        new_val: Vec<u8>,
        expires_at_ms: Option<u64>,
    ) -> Result<bool, String> {
        for _ in 0..8 {
            self.pull()?;
            let cur = self.get_record(&key)?;
            let matches = match (expected, cur.as_ref().and_then(|r| r.value.as_ref())) {
                (None, None) => true,
                (Some(exp), Some(Value::Str(b))) => b.as_slice() == exp,
                _ => false,
            };
            if !matches {
                return Ok(false);
            }
            let prev_mem = self.mem.get(&key).cloned();
            self.set_record(key.clone(), Value::Str(new_val.clone()), expires_at_ms);
            match self.try_commit_once()? {
                Some(_) => return Ok(true),
                None => {
                    self.stats.cas_retries += 1;
                    if let Some(old) = prev_mem {
                        self.mem.insert(key.clone(), old);
                    } else {
                        self.mem.remove(&key);
                    }
                }
            }
        }
        Err("ERR VXG.CAS contention retry limit exceeded".to_owned())
    }

    /// Clear all local and remote keys (commits an empty compacted snapshot).
    pub fn flush_all(&mut self) -> Result<u64, String> {
        self.mem.clear();
        self.wal_overlay.clear();
        self.pending_events.clear();
        self.applied_wals.clear();
        self.base = None;
        self.remote_handle = None;
        let next_epoch = self.manifest.epoch + 1;
        let (snap_ref, idx) = write_snapshot(
            &self.store,
            next_epoch,
            &BTreeMap::new(),
            self.cfg.block_size,
        )?;
        let mut next_m = self.manifest.clone();
        if let Some(old) = next_m.base_snapshot.take() {
            next_m.history.push(old);
        }
        next_m.epoch = next_epoch;
        next_m.base_snapshot = Some(snap_ref.clone());
        next_m.wals.clear();
        let etag = self.store.put(MANIFEST_KEY, &next_m.to_bytes())?;
        self.manifest = next_m;
        self.manifest_etag = Some(etag);
        self.base = Some((snap_ref, idx));
        Ok(next_epoch)
    }

    /// Pull latest manifest + any unseen WAL segments from the shared `ObjectStore`.
    /// Fast path: `HEAD manifest.json` -> unchanged ETag = **0 payload bytes transferred!**
    pub fn pull(&mut self) -> Result<usize, String> {
        self.stats.pulls += 1;
        let Some(meta) = self.store.head(MANIFEST_KEY)? else {
            return Ok(0);
        };
        if self.manifest_etag.as_deref() == Some(meta.etag.as_str()) {
            return Ok(0);
        }
        let Some((bytes, etag)) = self.store.get(MANIFEST_KEY)? else {
            return Ok(0);
        };
        let remote = Manifest::from_bytes(&bytes)?;
        let mut applied_count = 0usize;

        // Reload base snapshot index if compaction advanced the base snapshot
        let base_changed = match (&self.base, &remote.base_snapshot) {
            (Some((cur, _)), Some(rem)) => cur.epoch != rem.epoch,
            (None, Some(_)) | (Some(_), None) => true,
            (None, None) => false,
        };
        if base_changed {
            self.wal_overlay.clear();
            self.applied_wals.clear();
            self.remote_handle = None;
            if let Some(ref b) = remote.base_snapshot {
                let Some((idx_bytes, _)) = self.store.get(&b.idx_key)? else {
                    return Err(format!("ERR missing base index {}", b.idx_key));
                };
                let idx = SnapshotIndex::decode(&idx_bytes)?;
                self.hist_indices.insert(b.epoch, idx.clone());
                self.base = Some((b.clone(), idx));
            } else {
                self.base = None;
            }
        }

        // Apply any unseen WAL segments in epoch order
        for w in &remote.wals {
            if self.applied_wals.contains(&w.epoch) {
                continue;
            }
            let Some((wb, _)) = self.store.get(&w.key)? else {
                continue;
            };
            let (recs, events) = decode_wal_segment(&wb)?;
            for (k, r) in recs {
                let ver = r.version.max(self.next_ver + 1);
                self.next_ver = self.next_ver.max(ver);
                self.key_versions.insert(k.clone(), ver);
                self.wal_overlay.insert(k, r);
            }
            for ev in events {
                if ev.origin_pod != self.cfg.pod_id {
                    self.broker.publish_local(&ev.channel, &ev.payload);
                    self.stats.pubsub_remote_delivered += 1;
                }
            }
            self.applied_wals.insert(w.epoch);
            applied_count += 1;
        }
        self.manifest = remote;
        self.manifest_etag = Some(etag);
        Ok(applied_count)
    }

    fn try_commit_once(&mut self) -> Result<Option<u64>, String> {
        if !self.has_dirty() {
            return Ok(Some(self.manifest.epoch));
        }
        let next_epoch = self.manifest.epoch + 1;
        let wal_bytes = encode_wal_segment(&self.mem, &self.pending_events)?;
        let wal_key = format!("wal/wal_{next_epoch:08}_{}.vxg", self.cfg.pod_id);
        match self.store.put_if_none_match(&wal_key, &wal_bytes)? {
            PutOutcome::PreconditionFailed => return Ok(None),
            PutOutcome::Written(_) => {}
        }
        let mut next_m = self.manifest.clone();
        next_m.epoch = next_epoch;
        next_m.wals.push(WalRef {
            epoch: next_epoch,
            key: wal_key,
            pod_id: self.cfg.pod_id.clone(),
            compressed_bytes: wal_bytes.len() as u64,
            created_ms: now_ms(),
        });
        let raw_m = next_m.to_bytes();
        let outcome = match &self.manifest_etag {
            None => self.store.put_if_none_match(MANIFEST_KEY, &raw_m)?,
            Some(et) => self.store.put_if_match(MANIFEST_KEY, et, &raw_m)?,
        };
        match outcome {
            PutOutcome::Written(new_etag) => {
                // Move committed mem into wal_overlay
                for (k, r) in std::mem::take(&mut self.mem) {
                    self.wal_overlay.insert(k, r);
                }
                self.pending_events.clear();
                self.applied_wals.insert(next_epoch);
                self.manifest = next_m;
                self.manifest_etag = Some(new_etag);
                self.stats.cas_commits += 1;
                Ok(Some(next_epoch))
            }
            PutOutcome::PreconditionFailed => Ok(None),
        }
    }

    /// Atomic multi-pod CAS commit with automatic `412` pull + rebase + retry!
    pub fn commit(&mut self) -> Result<u64, String> {
        if !self.has_dirty() {
            return Ok(self.manifest.epoch);
        }
        for _ in 0..16 {
            if let Some(ep) = self.try_commit_once()? {
                return Ok(ep);
            }
            self.stats.cas_retries += 1;
            self.pull()?;
        }
        Err("ERR VXG.COMMIT exceeded CAS retry budget".to_owned())
    }

    /// Materialize the complete current live keyspace (base + WAL overlay + MemTable).
    pub fn materialize_live(&mut self) -> Result<BTreeMap<Vec<u8>, Record>, String> {
        let now = now_ms();
        let mut merged = if let Some((ref snap, _)) = self.base {
            read_full_snapshot(&self.store, snap)?
        } else {
            BTreeMap::new()
        };
        for (k, r) in &self.wal_overlay {
            if r.is_live(now) {
                merged.insert(k.clone(), r.clone());
            } else {
                merged.remove(k);
            }
        }
        for (k, r) in &self.mem {
            if r.is_live(now) {
                merged.insert(k.clone(), r.clone());
            } else {
                merged.remove(k);
            }
        }
        merged.retain(|_, r| r.is_live(now));
        Ok(merged)
    }

    /// Merge base snapshot + all active WALs + local MemTable into a clean sorted VXG v5
    /// snapshot (`snap_<epoch>.vxg` + `VXI1` Bloom/Fence index), archiving old snapshot in history!
    pub fn compact(&mut self) -> Result<SnapshotRef, String> {
        for _ in 0..8 {
            self.pull()?;
            let merged = self.materialize_live()?;
            let next_epoch = self.manifest.epoch + 1;
            let (snap_ref, idx) =
                write_snapshot(&self.store, next_epoch, &merged, self.cfg.block_size)?;
            let mut next_m = self.manifest.clone();
            if let Some(old_base) = next_m.base_snapshot.take() {
                next_m.history.push(old_base);
            }
            next_m.epoch = next_epoch;
            next_m.base_snapshot = Some(snap_ref.clone());
            next_m.wals.clear();
            let raw_m = next_m.to_bytes();
            let outcome = match &self.manifest_etag {
                None => self.store.put_if_none_match(MANIFEST_KEY, &raw_m)?,
                Some(et) => self.store.put_if_match(MANIFEST_KEY, et, &raw_m)?,
            };
            match outcome {
                PutOutcome::Written(new_etag) => {
                    self.mem.clear();
                    self.wal_overlay.clear();
                    self.applied_wals.clear();
                    self.pending_events.clear();
                    self.remote_handle = None;
                    self.hist_indices.insert(snap_ref.epoch, idx.clone());
                    self.base = Some((snap_ref.clone(), idx));
                    self.manifest = next_m;
                    self.manifest_etag = Some(new_etag);
                    self.stats.compactions += 1;
                    return Ok(snap_ref);
                }
                PutOutcome::PreconditionFailed => {
                    self.stats.cas_retries += 1;
                }
            }
        }
        Err("ERR VXG.COMPACT exceeded CAS retry budget".to_owned())
    }

    /// Return all live keys matching `pattern` (`*` matches everything).
    pub fn matching_keys(&mut self, pattern: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let merged = self.materialize_live()?;
        Ok(merged
            .into_keys()
            .filter(|k| pattern == b"*" || glob_match(pattern, k))
            .collect())
    }

    /// Execute compressed-domain `VXG.SQL` over either a pinned historical snapshot or live state.
    pub fn execute_sql(
        &mut self,
        sql: &str,
        format: SqlFormat,
        pinned_epoch: Option<u64>,
    ) -> Result<Vec<u8>, String> {
        if let Some(ep) = pinned_epoch {
            let Some(snap) = self.manifest.find_snapshot(ep).cloned() else {
                return Err(format!("ERR snapshot epoch {ep} not found"));
            };
            let Some((frame, _)) = self.store.get(&snap.vxg_key)? else {
                return Err(format!("ERR snapshot missing: {}", snap.vxg_key));
            };
            return execute_vxg_sql_on_frame(&frame, sql, format);
        }
        if self.mem.is_empty() && self.wal_overlay.is_empty() {
            if let Some((ref snap, _)) = self.base {
                if let Some((frame, _)) = self.store.get(&snap.vxg_key)? {
                    return execute_vxg_sql_on_frame(&frame, sql, format);
                }
            }
        }
        let merged = self.materialize_live()?;
        let mut raw = Vec::with_capacity(merged.len() * 64);
        for (k, r) in &merged {
            encode_record_line(&mut raw, k, r);
        }
        let frame = vxg::compress(
            &raw,
            &Options {
                block_size: self.cfg.block_size.clamp(4096, 262_144),
                record_aligned: true,
                row_index: true,
                checksum: true,
                ..Options::default()
            },
        )
        .map_err(|e| format!("ERR sql frame: {e}"))?;
        execute_vxg_sql_on_frame(&frame, sql, format)
    }
}
