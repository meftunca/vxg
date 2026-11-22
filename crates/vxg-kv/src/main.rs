use std::env;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vxg::SqlFormat;
use vxg_kv::{
    encode_command, serve_origin, serve_origin_listener, EngineConfig, HttpObjectStore, KvEngine,
    LocalFsStore, ObjectStore, PodServer, PubSubBroker, RespBuffer, ServerConfig, Value,
};

const HELP: &str = r#"vxg-kv — Stateless Snapshot Redis (RESP2) + Pub/Sub + Compressed-Domain SQL over VXG

Usage:
  vxg-kv origin DIR     [--port 9000]
  vxg-kv server         [--port 6379] [--origin http://127.0.0.1:9000 | --dir ./vxg-kv-data]
                        [--pod-id NAME] [--cache-mb 64] [--block-size 16384]
                        [--flush-ms 100] [--sync-ms 250] [--auto-compact 16]
  vxg-kv bench          [--keys 25000]
  vxg-kv --help|-h
"#;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn next_val<'a>(args: &'a [String], i: &mut usize) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| "missing option value".to_owned())
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(cmd) = args.first().map(String::as_str) else {
        print!("{HELP}");
        return Ok(());
    };
    match cmd {
        "-h" | "--help" | "help" => {
            print!("{HELP}");
            Ok(())
        }
        "origin" => {
            if args.len() < 2 {
                return Err("usage: vxg-kv origin DIR [--port 9000]".to_owned());
            }
            let dir = PathBuf::from(&args[1]);
            let mut port = 9000u16;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--port" | "-p" => {
                        port = next_val(args, &mut i)?
                            .parse()
                            .map_err(|_| "invalid port")?
                    }
                    x => return Err(format!("unknown option: {x}")),
                }
                i += 1;
            }
            serve_origin(&dir, port)
        }
        "server" => run_server(&args[1..]),
        "bench" => run_bench(&args[1..]),
        other => Err(format!("unknown command '{other}'\n{HELP}")),
    }
}

fn run_server(args: &[String]) -> Result<(), String> {
    let mut port = 6379u16;
    let mut origin_url: Option<String> = None;
    let mut data_dir = PathBuf::from("./vxg-kv-data");
    let mut ecfg = EngineConfig::default();
    let mut scfg = ServerConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => port = next_val(args, &mut i)?.parse().map_err(|_| "bad port")?,
            "--origin" => origin_url = Some(next_val(args, &mut i)?.to_owned()),
            "--dir" => data_dir = PathBuf::from(next_val(args, &mut i)?),
            "--pod-id" => ecfg.pod_id = next_val(args, &mut i)?.to_owned(),
            "--cache-mb" => {
                let mb: usize = next_val(args, &mut i)?.parse().map_err(|_| "bad cache-mb")?;
                ecfg.cache_bytes = mb.max(1) * 1024 * 1024;
            }
            "--block-size" => {
                ecfg.block_size = next_val(args, &mut i)?
                    .parse()
                    .map_err(|_| "bad block-size")?;
            }
            "--flush-ms" => {
                scfg.flush_ms = next_val(args, &mut i)?.parse().map_err(|_| "bad flush-ms")?;
            }
            "--sync-ms" => {
                scfg.sync_ms = next_val(args, &mut i)?.parse().map_err(|_| "bad sync-ms")?;
            }
            "--auto-compact" => {
                ecfg.auto_compact_wals = next_val(args, &mut i)?
                    .parse()
                    .map_err(|_| "bad auto-compact")?;
            }
            x => return Err(format!("unknown option: {x}")),
        }
        i += 1;
    }

    let store: Arc<dyn ObjectStore> = if let Some(ref url) = origin_url {
        Arc::new(HttpObjectStore::connect(url)?)
    } else {
        Arc::new(LocalFsStore::new(&data_dir)?)
    };
    let broker = Arc::new(PubSubBroker::new());
    let engine = KvEngine::open(store, broker, ecfg.clone())?;
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    let _srv = PodServer::spawn(listener, engine, scfg.clone())
        .map_err(|e| format!("spawn server: {e}"))?;

    println!(
        "vxg-kv RESP2 server listening on 127.0.0.1:{port} (pod={}, backend={}, flush={}ms, sync={}ms)",
        ecfg.pod_id,
        origin_url.as_deref().unwrap_or(data_dir.to_str().unwrap_or("local")),
        scfg.flush_ms,
        scfg.sync_ms
    );
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

fn run_bench(args: &[String]) -> Result<(), String> {
    let mut n_keys = 25_000usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--keys" | "-n" => n_keys = next_val(args, &mut i)?.parse().map_err(|_| "bad keys")?,
            x => return Err(format!("unknown option: {x}")),
        }
        i += 1;
    }

    let tmp = std::env::temp_dir().join(format!("vxg_kv_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;

    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let dir_c = tmp.clone();
    std::thread::spawn(move || {
        let _ = serve_origin_listener(listener, &dir_c);
    });
    let origin_url = format!("http://{addr}");

    let plans = ["free", "pro", "team", "enterprise"];
    let regions = ["eu", "us", "ap", "sa"];

    // Pod A writes + compacts snapshot to HTTP Origin
    let store_a: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&origin_url)?);
    let mut pod_a = KvEngine::open(
        store_a,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "bench-writer".into(),
            block_size: 16_384,
            ..EngineConfig::default()
        },
    )?;
    let t0 = Instant::now();
    for k in 0..n_keys {
        let key = format!("user:{k:06}").into_bytes();
        let val = format!(
            "{{\"plan\":\"{}\",\"region\":\"{}\",\"spend\":{}}}",
            plans[k % plans.len()],
            regions[(k * 3) % regions.len()],
            (k % 250) + 10
        )
        .into_bytes();
        pod_a.set_record(key, Value::Str(val), None);
    }
    let write_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tc = Instant::now();
    let snap = pod_a.compact()?;
    let compact_ms = tc.elapsed().as_secs_f64() * 1000.0;

    // Cold Pod B opens ONLY manifest + index (0 blocks in RAM!)
    let store_b: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&origin_url)?);
    let mut pod_b = KvEngine::open(
        store_b,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "bench-cold-reader".into(),
            ..EngineConfig::default()
        },
    )?;

    // 1) Cold single-key GET over HTTP/1.1 Range
    let t_cold = Instant::now();
    let target = format!("user:{:06}", n_keys / 2);
    let got = pod_b.get_record(target.as_bytes())?;
    assert!(got.is_some());
    let cold_us = t_cold.elapsed().as_secs_f64() * 1e6;
    let cold_bytes = pod_b.stats.range_bytes;
    let cold_reqs = pod_b.stats.range_requests;

    // 2) 1,000 missing-key lookups filtered by Bloom Filter (0 HTTP Range!)
    let t_bloom = Instant::now();
    let reqs_before = pod_b.stats.range_requests;
    for k in 0..1000 {
        let miss = format!("user:{:06}_nope", k * 17);
        assert!(pod_b.get_record(miss.as_bytes())?.is_none());
    }
    let bloom_us = t_bloom.elapsed().as_secs_f64() * 1e6;
    let bloom_extra_reqs = pod_b.stats.range_requests - reqs_before;

    // 3) 10,000 hot LRU cached GETs
    let t_hot = Instant::now();
    for _ in 0..10_000 {
        assert!(pod_b.get_record(target.as_bytes())?.is_some());
    }
    let hot_ms = t_hot.elapsed().as_secs_f64() * 1000.0;

    // 4) Compressed-domain VXG.SQL analytical query
    let t_sql = Instant::now();
    let sql_out = pod_b.execute_sql(
        "SELECT data.plan, COUNT(*) AS users, SUM(data.spend) AS revenue GROUP BY data.plan ORDER BY data.plan ASC",
        SqlFormat::Tsv,
        None,
    )?;
    let sql_ms = t_sql.elapsed().as_secs_f64() * 1000.0;

    println!("=== vxg-kv Benchmark ({n_keys} JSON keys over live HTTP/1.1 Range Origin) ===");
    println!(
        "Ingest (MemTable):   {write_ms:.2} ms ({:.0} keys/s)",
        n_keys as f64 / (write_ms / 1000.0)
    );
    println!(
        "VXG Snapshot:        {}B raw -> {}B VXG (ratio={:.4}, {:.1}x smaller) in {compact_ms:.2} ms",
        snap.raw_bytes,
        snap.compressed_bytes,
        snap.compressed_bytes as f64 / snap.raw_bytes.max(1) as f64,
        snap.raw_bytes as f64 / snap.compressed_bytes.max(1) as f64
    );
    println!(
        "Cold Single GET:     {cold_us:.0} us | fetched {cold_bytes}B / {}B ({:.2}%, {cold_reqs} HTTP Range reqs)",
        snap.compressed_bytes,
        cold_bytes as f64 * 100.0 / snap.compressed_bytes.max(1) as f64
    );
    println!(
        "1,000 Miss GETs:     {bloom_us:.0} us total ({:.2} us/op) | Bloom filtered={}, extra HTTP reqs={bloom_extra_reqs}",
        bloom_us / 1000.0,
        pod_b.stats.bloom_filtered
    );
    println!(
        "10,000 Hot GETs:     {hot_ms:.2} ms ({:.0} ops/s)",
        10_000.0 / (hot_ms / 1000.0)
    );
    println!("VXG.SQL GROUP BY:    {sql_ms:.2} ms\n{}", String::from_utf8_lossy(&sql_out));

    // 5) Live TCP RESP2 Pipelined Benchmark (pipeline=32 over real TCP loopback!)
    let tcp_listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
    let tcp_srv = PodServer::spawn(
        tcp_listener,
        pod_b,
        ServerConfig {
            flush_ms: 0,
            sync_ms: 0,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut sock = TcpStream::connect(tcp_srv.addr).map_err(|e| e.to_string())?;
    let _ = sock.set_nodelay(true);
    let mut rbuf = RespBuffer::new();
    let run_pipe = |sock: &mut TcpStream, rbuf: &mut RespBuffer, cmds: &[Vec<u8>]| -> Result<f64, String> {
        let t = Instant::now();
        for chunk in cmds.chunks(32) {
            let mut batch = Vec::with_capacity(chunk.len() * 40);
            for c in chunk {
                batch.extend_from_slice(c);
            }
            sock.write_all(&batch).map_err(|e| e.to_string())?;
            for _ in 0..chunk.len() {
                loop {
                    if rbuf.next_value()?.is_some() {
                        break;
                    }
                    if rbuf.read_more(sock).map_err(|e| e.to_string())? == 0 {
                        return Err("EOF in TCP bench".to_owned());
                    }
                }
            }
        }
        let secs = t.elapsed().as_secs_f64();
        Ok(cmds.len() as f64 / secs)
    };
    let n_tcp = 10_000usize;
    let set_cmds: Vec<Vec<u8>> = (0..n_tcp)
        .map(|j| {
            let k = format!("tcp:{j:06}");
            encode_command(&[b"SET", k.as_bytes(), b"payload_value_123"])
        })
        .collect();
    let get_cmds: Vec<Vec<u8>> = (0..n_tcp)
        .map(|j| {
            let k = format!("tcp:{j:06}");
            encode_command(&[b"GET", k.as_bytes()])
        })
        .collect();
    let incr_cmds: Vec<Vec<u8>> = (0..n_tcp)
        .map(|_| encode_command(&[b"INCR", b"tcp:counter"]))
        .collect();
    let xadd_cmds: Vec<Vec<u8>> = (0..2_000)
        .map(|j| {
            let v = j.to_string();
            encode_command(&[b"XADD", b"tcp:stream", b"MAXLEN", b"~", b"2000", b"*", b"id", v.as_bytes()])
        })
        .collect();

    let set_ops = run_pipe(&mut sock, &mut rbuf, &set_cmds)?;
    let get_ops = run_pipe(&mut sock, &mut rbuf, &get_cmds)?;
    let incr_ops = run_pipe(&mut sock, &mut rbuf, &incr_cmds)?;
    let xadd_ops = run_pipe(&mut sock, &mut rbuf, &xadd_cmds)?;
    println!("--- Live TCP RESP2 Pipelined (P=32) ---");
    println!("TCP SET:             {set_ops:.0} ops/s");
    println!("TCP GET:             {get_ops:.0} ops/s");
    println!("TCP INCR:            {incr_ops:.0} ops/s");
    println!("TCP XADD (Stream):   {xadd_ops:.0} ops/s");

    tcp_srv.stop();
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
