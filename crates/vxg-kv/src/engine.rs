//! Core `KvEngine`: LSM MemTable + VXG WAL Overlay + Remote Range Snapshot + Bloom/LRU Cache
//! + Atomic ETag CAS Multi-Pod Commit/Pull/Compaction + Time-Travel + `VXG.SQL`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use vxg::{DecodeOptions, RemoteFrame};

use crate::index::{DecodedBlock, LookupPlan, LruBlockCache, SnapshotIndex};
use crate::pubsub::{PubSubBroker, PubSubEvent};
use crate::snapshot::{Manifest, SnapshotRef};
use crate::store::{ObjectStore, StoreRangeReader};
use crate::types::{decode_record_line, FastMap, Record, Value};

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

#[derive(Debug, Default)]
pub struct EngineStats {
    pub range_requests: u64,
    pub range_bytes: u64,
    pub bloom_filtered: AtomicU64,
    pub out_of_range_filtered: AtomicU64,
    pub cache_hits: AtomicU64,
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
    pub mem: FastMap<Vec<u8>, Record>,
    pub wal_overlay: FastMap<Vec<u8>, Record>,
    pub applied_wals: BTreeSet<u64>,
    pub pending_events: Vec<PubSubEvent>,
    pub base: Option<(SnapshotRef, SnapshotIndex)>,
    pub hist_indices: HashMap<u64, SnapshotIndex>,
    pub(crate) remote_handle: Option<(u64, RemoteFrame<StoreRangeReader>)>,
    pub cache: LruBlockCache,
    pub next_ver: u64,
    pub key_versions: FastMap<Vec<u8>, u64>,
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
            mem: FastMap::default(),
            wal_overlay: FastMap::default(),
            applied_wals: BTreeSet::new(),
            pending_events: Vec::new(),
            base: None,
            hist_indices: HashMap::new(),
            remote_handle: None,
            cache,
            next_ver: 1,
            key_versions: FastMap::default(),
            stats: EngineStats::default(),
        };
        eng.pull()?;
        Ok(eng)
    }

    pub fn has_dirty(&self) -> bool {
        !self.mem.is_empty() || !self.pending_events.is_empty()
    }

    /// Register a `WATCH` on `key` and return its current version.
    pub fn watch_key(&mut self, key: &[u8]) -> u64 {
        let cur = self.next_ver;
        *self.key_versions.entry(key.to_vec()).or_insert(cur)
    }

    #[inline]
    pub fn key_version(&self, key: &[u8]) -> u64 {
        self.key_versions.get(key).copied().unwrap_or(0)
    }

    #[inline]
    fn bump_ver(&mut self, key: &[u8]) -> u64 {
        self.next_ver += 1;
        let v = self.next_ver;
        if !self.key_versions.is_empty() {
            if let Some(slot) = self.key_versions.get_mut(key) {
                *slot = v;
            }
        }
        v
    }

    /// Zero-clone, lock-free (`&self`) read-only lookup for hot `RwLock::read` parallel queries.
    /// Returns `Some(res)` if resolved from MemTable, WAL, Bloom/Range filter, or LRU cache;
    /// returns `None` ONLY when a cold snapshot block must be fetched via HTTP Range (`&mut self`).
    #[inline]
    pub fn get_record_ref(&self, key: &[u8]) -> Option<Option<&Record>> {
        if let Some(rec) = self.mem.get(key) {
            return Some(rec.is_live_lazy().then_some(rec));
        }
        if let Some(rec) = self.wal_overlay.get(key) {
            return Some(rec.is_live_lazy().then_some(rec));
        }
        let Some((snap, idx)) = self.base.as_ref() else {
            return Some(None);
        };
        let blk = match idx.plan_lookup(key) {
            LookupPlan::OutOfRange => {
                self.stats.out_of_range_filtered.fetch_add(1, Ordering::Relaxed);
                return Some(None);
            }
            LookupPlan::BloomFiltered => {
                self.stats.bloom_filtered.fetch_add(1, Ordering::Relaxed);
                return Some(None);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        let decoded = self.cache.peek(snap.epoch, blk)?;
        self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(decoded.get(key).filter(|r| r.is_live_lazy()))
    }

    /// Zero-clone lookup returning `&Record` (loads cold snapshot block on cache miss).
    pub fn get_record_or_load(&mut self, key: &[u8]) -> Result<Option<&Record>, String> {
        if self.mem.contains_key(key) {
            return Ok(self.mem.get(key).filter(|r| r.is_live_lazy()));
        }
        if self.wal_overlay.contains_key(key) {
            return Ok(self.wal_overlay.get(key).filter(|r| r.is_live_lazy()));
        }
        let (epoch, plan) = match self.base.as_ref() {
            Some((snap, idx)) => (snap.epoch, idx.plan_lookup(key)),
            None => return Ok(None),
        };
        let blk = match plan {
            LookupPlan::OutOfRange => {
                self.stats.out_of_range_filtered.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            LookupPlan::BloomFiltered => {
                self.stats.bloom_filtered.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        if self.cache.peek(epoch, blk).is_none() {
            self.stats.cache_misses += 1;
            let snap = self.base.as_ref().unwrap().0.clone();
            let (decoded, approx) = self.fetch_block_cached_handle(&snap, blk)?;
            self.cache.insert(epoch, blk, decoded, approx);
        } else {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        Ok(self
            .cache
            .get(epoch, blk)
            .and_then(|d| d.get(key))
            .filter(|r| r.is_live_lazy()))
    }

    /// Look up the current live record for `key` (MemTable -> WAL overlay -> Base Snapshot Range).
    pub fn get_record(&mut self, key: &[u8]) -> Result<Option<Record>, String> {
        Ok(self.get_record_or_load(key)?.cloned())
    }

    /// Zero-clone in-place `&mut Record` if live in MemTable, or promotes from WAL/Snapshot once.
    pub fn get_or_promote_mut(&mut self, key: &[u8]) -> Result<Option<&mut Record>, String> {
        if self.mem.contains_key(key) {
            let live = self.mem.get(key).is_some_and(Record::is_live_lazy);
            if !live {
                return Ok(None);
            }
            let v = self.bump_ver(key);
            let rec = self.mem.get_mut(key).unwrap();
            rec.version = v;
            return Ok(Some(rec));
        }
        let Some(mut rec) = self.get_record(key)? else {
            return Ok(None);
        };
        rec.version = self.bump_ver(key);
        self.mem.insert(key.to_vec(), rec);
        Ok(self.mem.get_mut(key))
    }

    /// Zero-clone in-place `&mut Record`, creating with `init()` if missing/expired/tombstone.
    pub fn ensure_entry_mut<F>(&mut self, key: &[u8], init: F) -> Result<&mut Record, String>
    where
        F: FnOnce() -> Value,
    {
        if self.mem.contains_key(key) {
            let v = self.bump_ver(key);
            let rec = self.mem.get_mut(key).unwrap();
            if !rec.is_live_lazy() {
                *rec = Record::live(init(), None, v);
            } else {
                rec.version = v;
            }
            return Ok(rec);
        }
        let v = self.bump_ver(key);
        let rec = match self.get_record(key)? {
            Some(mut existing) => {
                existing.version = v;
                existing
            }
            None => Record::live(init(), None, v),
        };
        self.mem.insert(key.to_vec(), rec);
        Ok(self.mem.get_mut(key).unwrap())
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
                self.stats.out_of_range_filtered.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            LookupPlan::BloomFiltered => {
                self.stats.bloom_filtered.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            LookupPlan::CandidateBlock(b) => b,
        };
        if let Some(decoded) = self.cache.get(epoch, blk) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
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
    #[inline]
    pub fn set_record(&mut self, key: Vec<u8>, value: Value, expires_at_ms: Option<u64>) {
        let v = self.bump_ver(&key);
        self.mem.insert(key, Record::live(value, expires_at_ms, v));
    }

    /// Zero-allocation in-place string `SET`: reuses existing key & value Vec capacity on overwrite!
    #[inline]
    pub fn set_str_in_place(&mut self, key: &[u8], val: &[u8]) {
        let v = self.bump_ver(key);
        if let Some(rec) = self.mem.get_mut(key) {
            rec.expires_at_ms = 0;
            rec.version = v;
            if let Some(Value::Str(buf)) = rec.value.as_mut() {
                buf.clear();
                buf.extend_from_slice(val);
            } else {
                rec.value = Some(Value::Str(val.to_vec()));
            }
            return;
        }
        self.mem
            .insert(key.to_vec(), Record::live(Value::Str(val.to_vec()), None, v));
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
        let Some(rec) = self.get_or_promote_mut(key)? else {
            return Ok(false);
        };
        rec.expires_at_ms = expires_at_ms.unwrap_or(0);
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
}
