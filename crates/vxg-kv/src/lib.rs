//! `vxg-kv`: Zero-dependency Redis-compatible (RESP2) stateless snapshot KV, Pub/Sub &
//! compressed-domain SQL engine backed by `VXG v5` + HTTP/1.1 Range & ETag CAS Object Storage.

pub mod blocking;
pub mod dispatch;
pub mod dispatch_ds;
pub mod dispatch_geo;
pub mod dispatch_json;
pub mod dispatch_stream;
pub mod dispatch_vxg;
pub mod dispatch_xgroup;
pub mod dispatch_zset;
pub mod engine;
mod engine_sync;
pub mod fast_path;
pub mod index;
pub mod mem_store;
#[cfg(not(target_arch = "wasm32"))]
pub mod origin;
pub mod pubsub;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod reactor;
pub mod resp;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
mod shard_sync;
pub mod snapshot;
pub mod store;
pub mod types;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use engine::{EngineConfig, EngineStats, KvEngine};
pub use mem_store::MemoryObjectStore;
#[cfg(not(target_arch = "wasm32"))]
pub use origin::{serve_origin, serve_origin_listener};
pub use pubsub::PubSubBroker;
pub use resp::{encode_command, RespBuffer, RespVal};
#[cfg(not(target_arch = "wasm32"))]
pub use server::{PodServer, ServerConfig};
#[cfg(not(target_arch = "wasm32"))]
pub use store::{HttpObjectStore, LocalFsStore};
pub use store::{ObjectMeta, ObjectStore, PutOutcome};
pub use types::{Record, Value};
