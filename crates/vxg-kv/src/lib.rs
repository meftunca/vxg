//! `vxg-kv`: Zero-dependency Redis-compatible (RESP2) stateless snapshot KV, Pub/Sub &
//! compressed-domain SQL engine backed by `VXG v5` + HTTP/1.1 Range & ETag CAS Object Storage.

pub mod dispatch;
pub mod dispatch_ds;
pub mod dispatch_stream;
pub mod dispatch_vxg;
pub mod dispatch_zset;
pub mod engine;
mod engine_sync;
pub mod fast_path;
pub mod index;
pub mod origin;
pub mod pubsub;
pub mod resp;
pub mod server;
pub mod snapshot;
pub mod store;
pub mod types;

pub use engine::{EngineConfig, EngineStats, KvEngine};
pub use origin::{serve_origin, serve_origin_listener};
pub use pubsub::PubSubBroker;
pub use resp::{encode_command, RespBuffer, RespVal};
pub use server::{PodServer, ServerConfig};
pub use store::{HttpObjectStore, LocalFsStore, ObjectMeta, ObjectStore, PutOutcome};
pub use types::{Record, Value};
