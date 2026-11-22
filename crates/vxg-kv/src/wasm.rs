//! Zero-WASI `wasm32-unknown-unknown` C ABI for the interactive browser `VXG-KV` Playground
//! (dual-pod `Pod A` + `Pod B` backed by a shared [`MemoryObjectStore`] synced to IndexedDB /
//! `localStorage`).

#![cfg(target_arch = "wasm32")]
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::dispatch::{dispatch_command_unlocked, ConnState};
use crate::engine::{EngineConfig, KvEngine};
use crate::mem_store::MemoryObjectStore;
use crate::pubsub::PubSubBroker;
use crate::resp::{RespBuffer, RespVal};
use crate::store::ObjectStore;
use crate::types::{push_json_str, set_wasm_now_ms};

struct PodSlot {
    engine: KvEngine,
    conn: ConnState,
}

struct WasmRuntime {
    store: Arc<MemoryObjectStore>,
    pods: [PodSlot; 2],
}

impl WasmRuntime {
    fn new(store: Arc<MemoryObjectStore>) -> Self {
        let pods = [
            Self::make_pod(&store, "pod-a", 1),
            Self::make_pod(&store, "pod-b", 2),
        ];
        Self { store, pods }
    }

    fn make_pod(store: &Arc<MemoryObjectStore>, id: &str, sub_id: u64) -> PodSlot {
        let obj: Arc<dyn ObjectStore> = Arc::clone(store) as Arc<dyn ObjectStore>;
        let broker = Arc::new(PubSubBroker::new());
        let cfg = EngineConfig {
            pod_id: id.to_owned(),
            block_size: 8_192,
            auto_compact_wals: 16,
            ..EngineConfig::default()
        };
        let engine = KvEngine::open(obj, broker, cfg).expect("open wasm pod");
        PodSlot {
            engine,
            conn: ConnState::new(sub_id),
        }
    }

    fn reopen_pods(&mut self) {
        self.pods = [
            Self::make_pod(&self.store, "pod-a", 1),
            Self::make_pod(&self.store, "pod-b", 2),
        ];
    }
}

thread_local! {
    static RT: RefCell<Option<WasmRuntime>> = const { RefCell::new(None) };
}

fn with_rt<F, R>(f: F) -> R
where
    F: FnOnce(&mut WasmRuntime) -> R,
{
    RT.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(WasmRuntime::new(Arc::new(MemoryObjectStore::new())));
        }
        f(slot.as_mut().unwrap())
    })
}

fn return_bytes(mut bytes: Vec<u8>, out_len: *mut usize) -> *mut u8 {
    bytes.shrink_to_fit();
    let ptr = bytes.as_mut_ptr();
    let len = bytes.len();
    std::mem::forget(bytes);
    if !out_len.is_null() {
        unsafe {
            *out_len = len;
        }
    }
    ptr
}

fn format_resp(v: &RespVal, indent: usize) -> String {
    match v {
        RespVal::Simple(s) => s.clone(),
        RespVal::Error(e) => format!("(error) {e}"),
        RespVal::Integer(n) => format!("(integer) {n}"),
        RespVal::Bulk(None) => "(nil)".to_owned(),
        RespVal::Bulk(Some(b)) => {
            let s = String::from_utf8_lossy(b);
            if s.contains('\n') {
                s.into_owned()
            } else {
                format!("\"{s}\"")
            }
        }
        RespVal::Array(None) => "(nil)".to_owned(),
        RespVal::Array(Some(items)) if items.is_empty() => "(empty array)".to_owned(),
        RespVal::Array(Some(items)) => {
            let pad = " ".repeat(indent);
            items
                .iter()
                .enumerate()
                .map(|(i, it)| format!("{pad}{}) {}", i + 1, format_resp(it, indent + 3)))
                .collect::<Vec<_>>()
                .join("\n")
        }
        RespVal::Map(pairs) => {
            let pad = " ".repeat(indent);
            pairs
                .iter()
                .enumerate()
                .map(|(i, (k, val))| {
                    format!(
                        "{pad}{}) {} => {}",
                        i + 1,
                        format_resp(k, 0),
                        format_resp(val, indent + 3)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

#[no_mangle]
pub extern "C" fn vxg_kv_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn vxg_kv_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[no_mangle]
pub extern "C" fn vxg_kv_set_now_ms(ms: f64) {
    if ms.is_finite() && ms > 0.0 {
        set_wasm_now_ms(ms as u64);
    }
}

#[no_mangle]
pub extern "C" fn vxg_kv_init() {
    with_rt(|_| {});
}

#[no_mangle]
pub extern "C" fn vxg_kv_reset() {
    with_rt(|rt| {
        rt.store.clear();
        rt.reopen_pods();
    });
}

#[no_mangle]
pub extern "C" fn vxg_kv_reopen_pods() {
    with_rt(|rt| {
        rt.reopen_pods();
    });
}

/// Hydrate one object `(key, data)` from browser IndexedDB / `localStorage` into the shared store.
#[no_mangle]
pub unsafe extern "C" fn vxg_kv_store_put(
    key_ptr: *const u8,
    key_len: usize,
    val_ptr: *const u8,
    val_len: usize,
) -> u32 {
    let key = std::slice::from_raw_parts(key_ptr, key_len);
    let Ok(key_str) = std::str::from_utf8(key) else {
        return 0;
    };
    let val = std::slice::from_raw_parts(val_ptr, val_len).to_vec();
    with_rt(|rt| {
        rt.store.hydrate(key_str.to_owned(), val);
    });
    1
}

/// Return raw bytes of one object in the shared store (for saving to IndexedDB / `localStorage` or download).
#[no_mangle]
pub unsafe extern "C" fn vxg_kv_store_get(
    key_ptr: *const u8,
    key_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let key = std::slice::from_raw_parts(key_ptr, key_len);
    let Ok(key_str) = std::str::from_utf8(key) else {
        return return_bytes(Vec::new(), out_len);
    };
    let bytes = with_rt(|rt| {
        rt.store
            .get(key_str)
            .ok()
            .flatten()
            .map(|(b, _)| b)
            .unwrap_or_default()
    });
    return_bytes(bytes, out_len)
}

/// Return JSON array `[{"key":"...","bytes":123,"etag":"..."}, ...]` of all objects in the store.
#[no_mangle]
pub extern "C" fn vxg_kv_store_list_json(out_len: *mut usize) -> *mut u8 {
    let json = with_rt(|rt| {
        let mut out = Vec::with_capacity(256);
        out.push(b'[');
        for (i, (k, sz, et)) in rt.store.list_meta().into_iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(b"{\"key\":");
            push_json_str(&mut out, &k);
            out.extend_from_slice(format!(",\"bytes\":{sz},\"etag\":").as_bytes());
            push_json_str(&mut out, &et);
            out.push(b'}');
        }
        out.push(b']');
        out
    });
    return_bytes(json, out_len)
}

/// Execute one inline Redis / `VXG.*` command on `pod_idx` (`0` = Pod A, `1` = Pod B).
/// Returns JSON: `{"ok":true,"reply":"...","mutated_store":false}`.
#[no_mangle]
pub unsafe extern "C" fn vxg_kv_exec(
    pod_idx: u32,
    cmd_ptr: *const u8,
    cmd_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let raw = std::slice::from_raw_parts(cmd_ptr, cmd_len);
    let idx = usize::from(pod_idx != 0);
    let json = with_rt(|rt| {
        let mut line = Vec::with_capacity(raw.len() + 2);
        line.extend_from_slice(raw);
        line.extend_from_slice(b"\r\n");
        let mut buf = RespBuffer::new();
        let _ = buf.read_more(&mut line.as_slice());
        let args = match buf.next_command() {
            Ok(Some(a)) => a,
            Ok(None) => return b"{\"ok\":true,\"reply\":\"\",\"mutated_store\":false}".to_vec(),
            Err(e) => {
                let mut o = b"{\"ok\":false,\"reply\":".to_vec();
                push_json_str(&mut o, &format!("(error) {e}"));
                o.extend_from_slice(b",\"mutated_store\":false}");
                return o;
            }
        };
        let pod = &mut rt.pods[idx];
        let commits_before = pod.engine.stats.cas_commits + pod.engine.stats.compactions;
        let broker = Arc::clone(&pod.engine.broker);
        let replies = dispatch_command_unlocked(&mut pod.engine, &broker, &mut pod.conn, args);
        let commits_after = pod.engine.stats.cas_commits + pod.engine.stats.compactions;
        let mutated_store = commits_after != commits_before
            || raw
                .windows(5)
                .any(|w| w.eq_ignore_ascii_case(b"FLUSH"));
        let is_err = replies.iter().any(|r| matches!(r, RespVal::Error(_)));
        let text = replies
            .iter()
            .map(|r| format_resp(r, 0))
            .collect::<Vec<_>>()
            .join("\n");
        let mut o = Vec::with_capacity(text.len() + 64);
        o.extend_from_slice(if is_err {
            b"{\"ok\":false,\"reply\":"
        } else {
            b"{\"ok\":true,\"reply\":"
        });
        push_json_str(&mut o, &text);
        o.extend_from_slice(if mutated_store {
            b",\"mutated_store\":true}"
        } else {
            b",\"mutated_store\":false}"
        });
        o
    });
    return_bytes(json, out_len)
}

/// Return JSON stats for `pod_idx` (`0` = Pod A, `1` = Pod B).
#[no_mangle]
pub extern "C" fn vxg_kv_pod_stats_json(pod_idx: u32, out_len: *mut usize) -> *mut u8 {
    let idx = usize::from(pod_idx != 0);
    let json = with_rt(|rt| {
        let eng = &mut rt.pods[idx].engine;
        let dbsize = eng.dbsize_fast().unwrap_or(0);
        let (base_epoch, base_keys, base_raw, base_comp, base_blocks) = match &eng.base {
            Some((s, i)) => (
                s.epoch,
                s.key_count,
                s.raw_bytes,
                s.compressed_bytes,
                i.blocks.len(),
            ),
            None => (0, 0, 0, 0, 0),
        };
        let s = format!(
            "{{\"pod_id\":\"{}\",\"epoch\":{},\"dbsize\":{},\"mem_dirty\":{},\"wal_overlay\":{},\"active_wals\":{},\"history\":{},\"base_epoch\":{},\"base_keys\":{},\"base_raw_bytes\":{},\"base_compressed_bytes\":{},\"base_blocks\":{},\"range_requests\":{},\"range_bytes\":{},\"bloom_filtered\":{},\"cache_blocks\":{},\"cache_hits\":{}}}",
            eng.cfg.pod_id,
            eng.manifest.epoch,
            dbsize,
            eng.mem.len(),
            eng.wal_overlay.len(),
            eng.active_wal_count(),
            eng.manifest.history.len(),
            base_epoch,
            base_keys,
            base_raw,
            base_comp,
            base_blocks,
            eng.stats.range_requests,
            eng.stats.range_bytes,
            eng.stats.bloom_filtered.load(Ordering::Relaxed),
            eng.cache.cached_blocks(),
            eng.stats.cache_hits.load(Ordering::Relaxed),
        );
        s.into_bytes()
    });
    return_bytes(json, out_len)
}
