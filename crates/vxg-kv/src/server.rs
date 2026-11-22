//! Multi-client zero-dependency RESP2 TCP server + background multi-pod CAS sync/flush/compaction.

use std::io::{self, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::dispatch::{dispatch_command, ConnState};
use crate::engine::KvEngine;
use crate::pubsub::{PubSubBroker, PushMessage};
use crate::resp::{RespBuffer, RespVal};

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
    pub engine: Arc<Mutex<KvEngine>>,
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
        let engine = Arc::new(Mutex::new(engine));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Background sync/flush/auto-compact loop
        if cfg.flush_ms > 0 || cfg.sync_ms > 0 {
            let eng_bg = Arc::clone(&engine);
            let sd_bg = Arc::clone(&shutdown);
            std::thread::spawn(move || background_sync_loop(eng_bg, sd_bg, cfg));
        }

        // TCP accept loop
        let eng_ac = Arc::clone(&engine);
        let br_ac = Arc::clone(&broker);
        let sd_ac = Arc::clone(&shutdown);
        let _ = listener.set_nonblocking(true);
        std::thread::spawn(move || {
            while !sd_ac.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let e = Arc::clone(&eng_ac);
                        let b = Arc::clone(&br_ac);
                        let sd = Arc::clone(&sd_ac);
                        std::thread::spawn(move || handle_client(stream, e, b, sd));
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            addr,
            engine,
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

fn background_sync_loop(
    engine: Arc<Mutex<KvEngine>>,
    shutdown: Arc<AtomicBool>,
    cfg: ServerConfig,
) {
    let mut last_flush = Instant::now();
    let mut last_sync = Instant::now();
    let tick = Duration::from_millis(
        cfg.flush_ms
            .min(cfg.sync_ms.max(1))
            .clamp(10, 50),
    );
    while !shutdown.load(Ordering::Relaxed) {
        std::thread::sleep(tick);
        let mut eng = engine.lock().unwrap();
        if cfg.flush_ms > 0
            && eng.has_dirty()
            && last_flush.elapsed() >= Duration::from_millis(cfg.flush_ms)
        {
            let _ = eng.commit();
            if eng.cfg.auto_compact_wals > 0
                && eng.manifest.wals.len() >= eng.cfg.auto_compact_wals
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
}

fn encode_push(msg: PushMessage) -> RespVal {
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

fn handle_client(
    mut stream: TcpStream,
    engine: Arc<Mutex<KvEngine>>,
    broker: Arc<PubSubBroker>,
    shutdown: Arc<AtomicBool>,
) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_nonblocking(false);
    let (tx, rx) = mpsc::channel::<PushMessage>();
    let sub_id = broker.register(tx);
    let mut conn = ConnState::new(sub_id);
    let mut buf = RespBuffer::new();
    let mut out_wire = Vec::with_capacity(4096);

    while !shutdown.load(Ordering::Relaxed) && !conn.quit {
        let subscribed = broker.subscription_count(sub_id) > 0;
        let timeout = if subscribed {
            Duration::from_millis(15)
        } else {
            Duration::from_millis(250)
        };
        let _ = stream.set_read_timeout(Some(timeout));

        // Drain any pending Pub/Sub messages first
        out_wire.clear();
        while let Ok(pm) = rx.try_recv() {
            encode_push(pm).encode(&mut out_wire);
        }
        if !out_wire.is_empty() && stream.write_all(&out_wire).is_err() {
            break;
        }

        // Read incoming RESP frames
        match buf.read_more(&mut stream) {
            Ok(0) => break,
            Ok(_) => {}
            Err(ref e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(_) => break,
        }

        out_wire.clear();
        loop {
            match buf.next_command() {
                Ok(Some(args)) => {
                    for reply in dispatch_command(&engine, &broker, &mut conn, args) {
                        reply.encode(&mut out_wire);
                    }
                    if conn.quit {
                        break;
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    RespVal::err(err).encode(&mut out_wire);
                    break;
                }
            }
        }
        if !out_wire.is_empty() && stream.write_all(&out_wire).is_err() {
            break;
        }
    }
    broker.unregister(sub_id);
}
