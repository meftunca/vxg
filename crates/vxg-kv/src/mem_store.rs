//! In-memory `ObjectStore` with real ETag CAS and byte-range reads.
//!
//! Powers the browser `wasm32` playground (synced to IndexedDB / `localStorage`) and fast
//! zero-filesystem tests.

use std::collections::BTreeMap;
use std::sync::RwLock;

use crate::store::{compute_etag, ObjectMeta, ObjectStore, PutOutcome, StoreResult};

#[derive(Default)]
pub struct MemoryObjectStore {
    inner: RwLock<BTreeMap<String, (Vec<u8>, String)>>,
}

impl MemoryObjectStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all stored objects.
    pub fn clear(&self) {
        self.inner.write().unwrap().clear();
    }

    /// Insert raw bytes directly (used when hydrating from IndexedDB / `localStorage`).
    pub fn hydrate(&self, key: String, data: Vec<u8>) -> String {
        let etag = compute_etag(&data);
        self.inner
            .write()
            .unwrap()
            .insert(key, (data, etag.clone()));
        etag
    }

    /// Snapshot metadata list `(key, byte_len, etag)` sorted by key.
    pub fn list_meta(&self) -> Vec<(String, usize, String)> {
        self.inner
            .read()
            .unwrap()
            .iter()
            .map(|(k, (v, et))| (k.clone(), v.len(), et.clone()))
            .collect()
    }
}

impl ObjectStore for MemoryObjectStore {
    fn head(&self, key: &str) -> StoreResult<Option<ObjectMeta>> {
        let map = self.inner.read().unwrap();
        Ok(map.get(key).map(|(v, et)| ObjectMeta {
            len: v.len() as u64,
            etag: et.clone(),
        }))
    }

    fn get(&self, key: &str) -> StoreResult<Option<(Vec<u8>, String)>> {
        let map = self.inner.read().unwrap();
        Ok(map.get(key).cloned())
    }

    fn get_range(&self, key: &str, start: u64, end: u64) -> StoreResult<Vec<u8>> {
        let map = self.inner.read().unwrap();
        let Some((v, _)) = map.get(key) else {
            return Err(format!("GET Range {key}: not found"));
        };
        let s = (start as usize).min(v.len());
        let e = (end as usize).min(v.len());
        if s >= e {
            return Ok(Vec::new());
        }
        Ok(v[s..e].to_vec())
    }

    fn put(&self, key: &str, data: &[u8]) -> StoreResult<String> {
        let etag = compute_etag(data);
        self.inner
            .write()
            .unwrap()
            .insert(key.to_owned(), (data.to_vec(), etag.clone()));
        Ok(etag)
    }

    fn put_if_none_match(&self, key: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let mut map = self.inner.write().unwrap();
        if map.contains_key(key) {
            return Ok(PutOutcome::PreconditionFailed);
        }
        let etag = compute_etag(data);
        map.insert(key.to_owned(), (data.to_vec(), etag.clone()));
        Ok(PutOutcome::Written(etag))
    }

    fn put_if_match(&self, key: &str, expected_etag: &str, data: &[u8]) -> StoreResult<PutOutcome> {
        let mut map = self.inner.write().unwrap();
        match map.get(key) {
            Some((_, cur)) if cur == expected_etag => {
                let etag = compute_etag(data);
                map.insert(key.to_owned(), (data.to_vec(), etag.clone()));
                Ok(PutOutcome::Written(etag))
            }
            _ => Ok(PutOutcome::PreconditionFailed),
        }
    }
}
