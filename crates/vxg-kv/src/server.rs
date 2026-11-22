//! Multi-client zero-dependency RESP2 TCP server:
//! - macOS & Linux: single-threaded zero-lock native `kqueue` / `epoll` reactor ([`crate::reactor`])
//! - Windows & other targets: portable `RwLock` thread-per-client server.

use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::engine::KvEngine;
use crate::pubsub::{PubSubBroker, PushMessage};
use crate::resp::RespVal;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub flush_ms: u64,
    pub sync_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            flush_ms: 100,
            sync_ms: 250,
        }
    }
}

/// Running `vxg-kv` pod server handle (exposes bound address and graceful shutdown).
pub struct PodServer {
    pub addr: SocketAddr,
    pub broker: Arc<PubSubBroker>,
    shutdown: Arc<AtomicBool>,
}

impl PodServer {
    /// Start a pod server on an already-bound `TcpListener` (supports `127.0.0.1:0` in tests).
    pub fn spawn(
        listener: TcpListener,
        engine: KvEngine,
        cfg: ServerConfig,
    ) -> io::Result<Self> {
        let addr = listener.local_addr()?;
        let broker = Arc::clone(&engine.broker);
        let shutdown = Arc::new(AtomicBool::new(false));

        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let br_r = Arc::clone(&broker);
            let sd_r = Arc::clone(&shutdown);
            std::thread::spawn(move || {
                crate::reactor::run_reactor(listener, engine, br_r, sd_r, cfg);
            });
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            portable::spawn_threaded(listener, engine, Arc::clone(&broker), Arc::clone(&shutdown), cfg);
        }

        Ok(Self {
            addr,
            broker,
            shutdown,
        })
    }

    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

impl Drop for PodServer {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) fn encode_push(msg: PushMessage) -> RespVal {
    match msg {
        PushMessage::Message { channel, payload } => RespVal::array(vec![
            RespVal::bulk(b"message".to_vec()),
            RespVal::bulk(channel),
            RespVal::bulk(payload),
        ]),
        PushMessage::PMessage {
            pattern,
            channel,
            payload,
        } => RespVal::array(vec![
            RespVal::bulk(b"pmessage".to_vec()),
            RespVal::bulk(pattern),
            RespVal::bulk(channel),
            RespVal::bulk(payload),
        ]),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod portable {
    use std::io::{self, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, Instant};

    use super::{encode_push, ServerConfig};
    use crate::dispatch::{dispatch_command, ConnState};
    use crate::engine::KvEngine;
    use crate::fast_path::dispatch_span_batch;
    use crate::pubsub::{PubSubBroker, PushMessage};
    use crate::resp::{RespBuffer, RespVal};

    pub fn spawn_threaded(
        listener: TcpListener,
        engine: KvEngine,
        broker: Arc<PubSubBroker>,
        shutdown: Arc<AtomicBool>,
        cfg: ServerConfig,
    ) {
        let engine = Arc::new(RwLock::new(engine));
        if cfg.flush_ms > 0 || cfg.sync_ms > 0 {
            let eng_bg = Arc::clone(&engine);
            let sd_bg = Arc::clone(&shutdown);
            std::thread::spawn(move || {
                let mut last_flush = Instant::now();
                let mut last_sync = Instant::now();
                let tick = Duration::from_millis(cfg.flush_ms.min(cfg.sync_ms.max(1)).clamp(10, 50));
                while !sd_bg.load(Ordering::Relaxed) {
                    std::thread::sleep(tick);
                    let mut eng = eng_bg.write().unwrap();
                    if cfg.flush_ms > 0
                        && eng.has_dirty()
                        && last_flush.elapsed() >= Duration::from_millis(cfg.flush_ms)
                    {
                        let _ = eng.commit();
                        if eng.cfg.auto_compact_wals > 0
                            && eng.active_wal_count() >= eng.cfg.auto_compact_wals
                        {
                            let _ = eng.compact();
                        }
                        last_flush = Instant::now();
                    }
                    if cfg.sync_ms > 0 && last_sync.elapsed() >= Duration::from_millis(cfg.sync_ms) {
                        let _ = eng.pull();
                        last_sync = Instant::now();
                    }
                }
            });
        }
        let _ = listener.set_nonblocking(true);
        std::thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let e = Arc::clone(&engine);
                        let b = Arc::clone(&broker);
                        let sd = Arc::clone(&shutdown);
                        std::thread::spawn(move || handle_client(stream, e, b, sd));
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
    }

    fn handle_client(
        mut stream: TcpStream,
        engine: Arc<RwLock<KvEngine>>,
        broker: Arc<PubSubBroker>,
        shutdown: Arc<AtomicBool>,
    ) {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
        let (tx, rx) = mpsc::channel::<PushMessage>();
        let sub_id = broker.register(tx);
        let init_authed = engine.read().unwrap().cfg.password.is_none();
        let mut conn = ConnState::new_with_auth(sub_id, init_authed);
        let mut buf = RespBuffer::new();
        let mut out_wire = Vec::with_capacity(16_384);
        let mut arg_spans: Vec<(usize, usize)> = Vec::with_capacity(128);
        let mut cmd_ranges: Vec<(usize, usize)> = Vec::with_capacity(64);
        let mut sub_mode = false;

        while !shutdown.load(Ordering::Relaxed) && !conn.quit {
            if sub_mode {
                out_wire.clear();
                while let Ok(pm) = rx.try_recv() {
                    encode_push(pm).encode(&mut out_wire);
                }
                if !out_wire.is_empty() && stream.write_all(&out_wire).is_err() {
                    break;
                }
            }
            match buf.read_more(&mut stream) {
                Ok(0) => break,
                Ok(_) => {}
                Err(ref e)
                    if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) =>
                {
                    continue;
                }
                Err(_) => break,
            }
            out_wire.clear();
            let mut had_sub_cmd = false;
            match buf.parse_batch_spans(&mut arg_spans, &mut cmd_ranges) {
                Ok(true) if !cmd_ranges.is_empty() => {
                    let raw = buf.raw_buf();
                    had_sub_cmd = cmd_ranges.iter().any(|&(a, b)| {
                        a < b && {
                            let (s, e) = arg_spans[a];
                            let c = &raw[s..e];
                            c.eq_ignore_ascii_case(b"SUBSCRIBE")
                                || c.eq_ignore_ascii_case(b"UNSUBSCRIBE")
                                || c.eq_ignore_ascii_case(b"PSUBSCRIBE")
                                || c.eq_ignore_ascii_case(b"PUNSUBSCRIBE")
                        }
                    });
                    dispatch_span_batch(
                        &engine,
                        &broker,
                        &mut conn,
                        raw,
                        &arg_spans,
                        &cmd_ranges,
                        &mut out_wire,
                    );
                }
                Ok(true) => {}
                Ok(false) => {
                    if !cmd_ranges.is_empty() {
                        dispatch_span_batch(
                            &engine,
                            &broker,
                            &mut conn,
                            buf.raw_buf(),
                            &arg_spans,
                            &cmd_ranges,
                            &mut out_wire,
                        );
                    }
                    while let Ok(Some(args)) = buf.next_command() {
                        for reply in dispatch_command(&engine, &broker, &mut conn, args) {
                            reply.encode(&mut out_wire);
                        }
                        if conn.quit {
                            break;
                        }
                    }
                }
                Err(err) => RespVal::err(err).encode(&mut out_wire),
            }
            if !out_wire.is_empty() && stream.write_all(&out_wire).is_err() {
                break;
            }
            if had_sub_cmd {
                let now_sub = broker.subscription_count(sub_id) > 0;
                if now_sub != sub_mode {
                    sub_mode = now_sub;
                    let timeout = if sub_mode {
                        Duration::from_millis(15)
                    } else {
                        Duration::from_millis(250)
                    };
                    let _ = stream.set_read_timeout(Some(timeout));
                }
            }
        }
        broker.unregister(sub_id);
    }
}
