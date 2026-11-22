//! Multi-pod synchronization (`pull`, `commit`, `compact`), `flush_all`, and `VXG.SQL` execution
//! methods on [`KvEngine`].

use std::collections::BTreeMap;

use vxg::{Options, SqlFormat};

use crate::engine::KvEngine;
use crate::index::SnapshotIndex;
use crate::shard_sync::is_internal_key;
use crate::snapshot::{
    decode_wal_segment, encode_wal_segment, execute_vxg_sql_on_frame, read_full_snapshot,
    write_snapshot, write_snapshot_fastmap, Manifest, SnapshotRef, WalRef, MANIFEST_KEY,
};
use crate::store::PutOutcome;
use crate::types::{encode_record_line, now_ms, FastMap, Record};

impl KvEngine {
    /// Clear all local and remote keys (commits an empty compacted snapshot).
    pub fn flush_all(&mut self) -> Result<u64, String> {
        self.mem = FastMap::default();
        self.wal_overlay = FastMap::default();
        self.pending_events.clear();
        self.applied_wals.clear();
        self.key_versions.clear();
        self.key_versions.shrink_to_fit();
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
        if self.cfg.wal_shards > 1 {
            self.clear_all_shards()?;
        }
        Ok(next_epoch)
    }

    /// Pull latest manifest + any unseen WAL segments from the shared `ObjectStore`.
    /// Fast path: `HEAD manifest.json` -> unchanged ETag = **0 payload bytes transferred!**
    pub fn pull(&mut self) -> Result<usize, String> {
        self.stats.pulls += 1;
        let mut applied_count = 0usize;
        if let Some(meta) = self.store.head(MANIFEST_KEY)? {
            if self.manifest_etag.as_deref() != Some(meta.etag.as_str()) {
                if let Some((bytes, etag)) = self.store.get(MANIFEST_KEY)? {
                    let remote = Manifest::from_bytes(&bytes)?;
                    let base_changed = match (&self.base, &remote.base_snapshot) {
                        (Some((cur, _)), Some(rem)) => cur.epoch != rem.epoch,
                        (None, Some(_)) | (Some(_), None) => true,
                        (None, None) => false,
                    };
                    if base_changed {
                        self.wal_overlay = FastMap::default();
                        self.applied_wals.clear();
                        self.remote_handle = None;
                        for s in &mut self.applied_shard_wals {
                            s.clear();
                        }
                        self.shard_etags.fill(None);
                        if let Some(ref b) = remote.base_snapshot {
                            let Some((idx_bytes, _)) = self.store.get(&b.idx_key)? else {
                                return Err(format!("ERR missing base index {}", b.idx_key));
                            };
                            let idx = SnapshotIndex::decode(&idx_bytes)?;
                            self.base = Some((b.clone(), idx));
                        } else {
                            self.base = None;
                        }
                    }

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
                        self.applied_wals.insert(w.epoch);
                        applied_count += 1;
                    }
                    self.manifest = remote;
                    self.manifest_etag = Some(etag);
                }
            }
        }
        if self.cfg.wal_shards > 1 {
            applied_count += self.pull_all_shards()?;
        }
        Ok(applied_count)
    }

    pub(crate) fn try_commit_once(&mut self) -> Result<Option<u64>, String> {
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
        if self.cfg.wal_shards > 1 {
            return self.commit_sharded();
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
            if snap.key_count > 0 {
                read_full_snapshot(&self.store, snap)?
            } else {
                BTreeMap::new()
            }
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
            let next_epoch = self.manifest.epoch + 1;
            let base_empty = self.base.as_ref().is_none_or(|(s, _)| s.key_count == 0);
            let (snap_ref, idx) = if base_empty && self.wal_overlay.is_empty() {
                // Zero-clone fast path: sort references & serialize directly from MemTable!
                write_snapshot_fastmap(&self.store, next_epoch, &self.mem, self.cfg.block_size)?
            } else {
                let merged = self.materialize_live()?;
                write_snapshot(&self.store, next_epoch, &merged, self.cfg.block_size)?
            };
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
                    self.mem = FastMap::default();
                    self.wal_overlay = FastMap::default();
                    self.applied_wals.clear();
                    self.pending_events.clear();
                    self.remote_handle = None;
                    self.base = Some((snap_ref.clone(), idx));
                    self.manifest = next_m;
                    self.manifest_etag = Some(new_etag);
                    self.stats.compactions += 1;
                    if self.cfg.wal_shards > 1 {
                        self.clear_all_shards()?;
                    }
                    return Ok(snap_ref);
                }
                PutOutcome::PreconditionFailed => {
                    self.stats.cas_retries += 1;
                }
            }
        }
        Err("ERR VXG.COMPACT exceeded CAS retry budget".to_owned())
    }

    /// Return all live keys matching `pattern` (`*` matches everything) using fence-pointer pruning.
    pub fn matching_keys(&mut self, pattern: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        self.matching_keys_pruned(pattern)
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
            if !is_internal_key(k) {
                encode_record_line(&mut raw, k, r);
            }
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
