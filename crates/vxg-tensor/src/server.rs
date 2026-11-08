//! Pure-Rust OpenAI-compatible HTTP/1.1 + SSE streaming server (`EmbeddedServer` & `vxg-tensor serve`).
//!
//! Owns a dedicated single-thread Metal inference worker so Objective-C/Metal handles never cross
//! threads, while HTTP handlers in [`crate::server_http`] query non-blocking [`ServerSnapshot`]s.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

use serde_json::{json, Value};

use crate::chat::{ChatMessage, ChatParams, ChatSession, ModelSummary, TurnMetrics};
use crate::server_http::handle_http_conn;
use crate::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct ServerSnapshot {
    pub port: u16,
    pub busy: bool,
    pub cached_tokens: usize,
    pub model: Option<ModelSummary>,
    pub last_metrics: Option<TurnMetrics>,
    pub last_error: Option<String>,
}

pub(crate) enum StreamEvent {
    Chunk(String),
    Done { full_text: String, metrics: TurnMetrics },
    Err(String),
}

pub(crate) enum WorkerJob {
    Load {
        path: String,
        use_gpu: bool,
        threads: usize,
        reply: mpsc::Sender<Result<ModelSummary>>,
    },
    Unload {
        reply: mpsc::Sender<()>,
    },
    Reset {
        reply: mpsc::Sender<()>,
    },
    Chat {
        messages: Vec<ChatMessage>,
        params: ChatParams,
        events: mpsc::Sender<StreamEvent>,
    },
    Shutdown,
}

#[derive(Clone)]
pub struct EmbeddedServerHandle {
    port: u16,
    tx: Arc<Mutex<mpsc::Sender<WorkerJob>>>,
    snap: Arc<Mutex<ServerSnapshot>>,
    cancel: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

pub struct EmbeddedServer {
    handle: EmbeddedServerHandle,
    accept_join: Option<JoinHandle<()>>,
    worker_join: Option<JoinHandle<()>>,
}

impl EmbeddedServerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub fn snapshot(&self) -> ServerSnapshot {
        self.snap.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn abort(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn load_model(&self, path: String, use_gpu: bool, threads: usize) -> Result<ModelSummary> {
        self.abort();
        let (rtx, rrx) = mpsc::channel();
        self.send_job(WorkerJob::Load { path, use_gpu, threads, reply: rtx })?;
        rrx.recv().map_err(|_| Error::Io("inference worker disconnected".into()))?
    }

    pub fn unload_model(&self) -> Result<()> {
        self.abort();
        let (rtx, rrx) = mpsc::channel();
        self.send_job(WorkerJob::Unload { reply: rtx })?;
        let _ = rrx.recv();
        Ok(())
    }

    pub fn reset_cache(&self) -> Result<()> {
        self.abort();
        let (rtx, rrx) = mpsc::channel();
        self.send_job(WorkerJob::Reset { reply: rtx })?;
        let _ = rrx.recv();
        Ok(())
    }

    pub(crate) fn send_job(&self, job: WorkerJob) -> Result<()> {
        let guard = self.tx.lock().map_err(|_| Error::Io("worker mutex poisoned".into()))?;
        guard.send(job).map_err(|_| Error::Io("inference worker closed".into()))
    }
}

impl EmbeddedServer {
    pub fn start(host: &str, port: u16) -> Result<Self> {
        let listener = TcpListener::bind((host, port))
            .map_err(|e| Error::Io(format!("bind {host}:{port} failed: {e}")))?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| Error::Io(e.to_string()))?
            .port();

        let (tx, rx) = mpsc::channel::<WorkerJob>();
        let snap = Arc::new(Mutex::new(ServerSnapshot {
            port: bound_port,
            ..ServerSnapshot::default()
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let snap_w = Arc::clone(&snap);
        let cancel_w = Arc::clone(&cancel);
        let worker_join = thread::Builder::new()
            .name("vxt-infer-worker".into())
            .spawn(move || run_worker_loop(rx, snap_w, cancel_w))
            .map_err(|e| Error::Io(e.to_string()))?;

        let handle = EmbeddedServerHandle {
            port: bound_port,
            tx: Arc::new(Mutex::new(tx)),
            snap,
            cancel,
            stop: Arc::clone(&stop),
        };
        let accept_handle = handle.clone();
        let accept_join = thread::Builder::new()
            .name("vxt-http-accept".into())
            .spawn(move || {
                for incoming in listener.incoming() {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(stream) = incoming {
                        let h = accept_handle.clone();
                        let _ = thread::spawn(move || {
                            let _ = handle_http_conn(stream, h);
                        });
                    }
                }
            })
            .map_err(|e| Error::Io(e.to_string()))?;

        Ok(Self {
            handle,
            accept_join: Some(accept_join),
            worker_join: Some(worker_join),
        })
    }

    pub fn handle(&self) -> EmbeddedServerHandle {
        self.handle.clone()
    }

    pub fn wait_forever(mut self) {
        if let Some(j) = self.accept_join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for EmbeddedServer {
    fn drop(&mut self) {
        self.handle.stop.store(true, Ordering::Relaxed);
        self.handle.abort();
        let _ = self.handle.send_job(WorkerJob::Shutdown);
        let _ = TcpStream::connect(("127.0.0.1", self.handle.port));
        if let Some(j) = self.worker_join.take() {
            let _ = j.join();
        }
        if let Some(j) = self.accept_join.take() {
            let _ = j.join();
        }
    }
}

fn run_worker_loop(
    rx: mpsc::Receiver<WorkerJob>,
    snap: Arc<Mutex<ServerSnapshot>>,
    cancel: Arc<AtomicBool>,
) {
    let mut session: Option<ChatSession> = None;
    while let Ok(job) = rx.recv() {
        match job {
            WorkerJob::Shutdown => break,
            WorkerJob::Load { path, use_gpu, threads, reply } => {
                if let Ok(mut s) = snap.lock() {
                    s.busy = true;
                    s.last_error = None;
                }
                session = None;
                match ChatSession::load(&path, use_gpu, threads) {
                    Ok(sess) => {
                        let sum = sess.summary.clone();
                        if let Ok(mut s) = snap.lock() {
                            s.busy = false;
                            s.cached_tokens = 0;
                            s.model = Some(sum.clone());
                            s.last_metrics = None;
                        }
                        session = Some(sess);
                        let _ = reply.send(Ok(sum));
                    }
                    Err(e) => {
                        if let Ok(mut s) = snap.lock() {
                            s.busy = false;
                            s.model = None;
                            s.last_error = Some(e.to_string());
                        }
                        let _ = reply.send(Err(e));
                    }
                }
            }
            WorkerJob::Unload { reply } => {
                session = None;
                if let Ok(mut s) = snap.lock() {
                    s.busy = false;
                    s.cached_tokens = 0;
                    s.model = None;
                }
                let _ = reply.send(());
            }
            WorkerJob::Reset { reply } => {
                if let Some(sess) = session.as_mut() {
                    sess.reset();
                }
                if let Ok(mut s) = snap.lock() {
                    s.cached_tokens = 0;
                }
                let _ = reply.send(());
            }
            WorkerJob::Chat { messages, params, events } => {
                let Some(sess) = session.as_mut() else {
                    let _ = events.send(StreamEvent::Err(
                        "No model loaded in VXT server. Load a .vxt or .gguf model first.".into(),
                    ));
                    continue;
                };
                if let Ok(mut s) = snap.lock() {
                    s.busy = true;
                    s.last_error = None;
                }
                let res = sess.generate_chat(&messages, &params, &cancel, |chunk| {
                    events.send(StreamEvent::Chunk(chunk.to_owned())).is_ok()
                });
                match res {
                    Ok((full_text, metrics)) => {
                        if let Ok(mut s) = snap.lock() {
                            s.busy = false;
                            s.cached_tokens = sess.cached_token_count();
                            s.last_metrics = Some(metrics.clone());
                        }
                        let _ = events.send(StreamEvent::Done { full_text, metrics });
                    }
                    Err(e) => {
                        if let Ok(mut s) = snap.lock() {
                            s.busy = false;
                            s.cached_tokens = sess.cached_token_count();
                            s.last_error = Some(e.to_string());
                        }
                        let _ = events.send(StreamEvent::Err(e.to_string()));
                    }
                }
            }
        }
    }
}

pub fn snapshot_to_json(s: &ServerSnapshot) -> Value {
    let model = s.model.as_ref().map(|m| {
        json!({
            "path": m.model_path,
            "name": m.model_name,
            "arch": m.arch,
            "n_layer": m.n_layer,
            "n_embd": m.n_embd,
            "n_vocab": m.n_vocab,
            "compressed_bytes": m.compressed_bytes,
            "compressed_mib": ((m.compressed_bytes as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0,
            "backend": m.backend,
            "cb4_fast": m.cb4_fast,
            "turbo_fast": m.turbo_fast,
            "load_ms": (m.load_ms * 10.0).round() / 10.0,
        })
    });
    let metrics = s.last_metrics.as_ref().map(|t| {
        json!({
            "prefill_ms": (t.prefill_ms * 100.0).round() / 100.0,
            "decode_ms": (t.decode_ms * 100.0).round() / 100.0,
            "prefill_tps": (t.prefill_tps * 10.0).round() / 10.0,
            "decode_tps": (t.decode_tps * 100.0).round() / 100.0,
            "stream_gbs": (t.stream_gbs * 100.0).round() / 100.0,
            "prompt_tokens": t.prompt_tokens,
            "cached_tokens": t.cached_tokens,
            "new_prefill_tokens": t.new_prefill_tokens,
            "generated_tokens": t.generated_tokens,
            "finish_reason": t.finish_reason,
            "decode_path": t.decode_path,
        })
    });
    json!({
        "ok": true,
        "port": s.port,
        "base_url": format!("http://127.0.0.1:{}/v1", s.port),
        "busy": s.busy,
        "cached_tokens": s.cached_tokens,
        "model": model,
        "last_metrics": metrics,
        "last_error": s.last_error,
    })
}

pub fn cmd_serve_llm(args: &[String]) -> Result<()> {
    let mut model_path: Option<String> = None;
    let mut host = String::from("127.0.0.1");
    let mut port = 11435u16;
    let mut use_gpu = true;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                port = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --port".into()))?;
            }
            "--host" => {
                i += 1;
                host = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| Error::Format("missing --host".into()))?;
            }
            "--cpu" => use_gpu = false,
            s if !s.starts_with('-') && model_path.is_none() => model_path = Some(s.to_owned()),
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }

    let srv = EmbeddedServer::start(&host, port)?;
    if let Some(mp) = model_path {
        let sum = srv.handle().load_model(mp, use_gpu, threads)?;
        println!(
            "Loaded '{}' ({}, {:.2} MiB) on {} in {:.0} ms",
            sum.model_name,
            sum.arch,
            sum.compressed_bytes as f64 / (1024.0 * 1024.0),
            sum.backend,
            sum.load_ms
        );
    }
    println!(
        "VXT OpenAI-compatible API listening on http://{host}:{}/v1\n  POST /v1/chat/completions  (SSE stream & JSON)\n  GET  /v1/models",
        srv.handle().port()
    );
    srv.wait_forever();
    Ok(())
}
