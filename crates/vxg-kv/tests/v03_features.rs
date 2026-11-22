use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use vxg_kv::dispatch::{dispatch_command_unlocked, ConnState};
use vxg_kv::{
    encode_command, serve_origin_listener, EngineConfig, HttpObjectStore, KvEngine, LocalFsStore,
    ObjectStore, PodServer, PubSubBroker, RespBuffer, RespVal, ServerConfig, Value,
};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "vxg_kv_v03_{tag}_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn spawn_http_origin(dir: &std::path::Path) -> String {
    let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = l.local_addr().unwrap();
    let d = dir.to_path_buf();
    thread::spawn(move || {
        let _ = serve_origin_listener(l, &d);
    });
    format!("http://{addr}")
}

fn cmd_send(sock: &mut TcpStream, rbuf: &mut RespBuffer, args: &[&[u8]]) -> RespVal {
    sock.write_all(&encode_command(args)).unwrap();
    loop {
        if let Some(v) = rbuf.next_value().unwrap() {
            return v;
        }
        let n = rbuf.read_more(sock).unwrap();
        assert!(n > 0, "unexpected EOF from server");
    }
}

fn exec_local(eng: &mut KvEngine, conn: &mut ConnState, args: &[&[u8]]) -> RespVal {
    let owned: Vec<Vec<u8>> = args.iter().map(|a| a.to_vec()).collect();
    let br = Arc::clone(&eng.broker);
    dispatch_command_unlocked(eng, &br, conn, owned)
        .into_iter()
        .next()
        .unwrap()
}

#[test]
fn test_auth_acl_and_resp3_hello() {
    let dir = temp_dir("auth");
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFsStore::new(&dir).unwrap());
    let eng = KvEngine::open(
        store,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "auth-pod".into(),
            password: Some("s3cr3t!".into()),
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let srv = PodServer::spawn(
        listener,
        eng,
        ServerConfig {
            flush_ms: 0,
            sync_ms: 0,
        },
    )
    .unwrap();

    // Client 1: rejected until HELLO 3 AUTH default s3cr3t!
    let mut c1 = TcpStream::connect(srv.addr).unwrap();
    let mut b1 = RespBuffer::new();
    match cmd_send(&mut c1, &mut b1, &[b"PING"]) {
        RespVal::Error(e) => assert!(e.starts_with("NOAUTH"), "got {e}"),
        other => panic!("expected NOAUTH, got {other:?}"),
    }
    match cmd_send(&mut c1, &mut b1, &[b"AUTH", b"wrong"]) {
        RespVal::Error(e) => assert!(e.starts_with("WRONGPASS"), "got {e}"),
        other => panic!("expected WRONGPASS, got {other:?}"),
    }
    match cmd_send(&mut c1, &mut b1, &[b"HELLO", b"3"]) {
        RespVal::Error(e) => assert!(e.starts_with("NOAUTH"), "got {e}"),
        other => panic!("expected NOAUTH on HELLO 3, got {other:?}"),
    }
    match cmd_send(&mut c1, &mut b1, &[b"HELLO", b"3", b"AUTH", b"default", b"s3cr3t!"]) {
        RespVal::Map(pairs) => {
            assert!(pairs.iter().any(|(k, v)| {
                *k == RespVal::bulk(b"proto".to_vec()) && *v == RespVal::int(3)
            }));
        }
        other => panic!("expected RESP3 Map from HELLO 3, got {other:?}"),
    }
    assert_eq!(
        cmd_send(&mut c1, &mut b1, &[b"ACL", b"WHOAMI"]),
        RespVal::bulk(b"default".to_vec())
    );
    assert_eq!(
        cmd_send(&mut c1, &mut b1, &[b"SET", b"k1", b"v1"]),
        RespVal::ok()
    );
    assert_eq!(
        cmd_send(&mut c1, &mut b1, &[b"GET", b"k1"]),
        RespVal::bulk(b"v1".to_vec())
    );

    // Client 2: classic AUTH + HELLO 2
    let mut c2 = TcpStream::connect(srv.addr).unwrap();
    let mut b2 = RespBuffer::new();
    assert_eq!(
        cmd_send(&mut c2, &mut b2, &[b"AUTH", b"s3cr3t!"]),
        RespVal::ok()
    );
    match cmd_send(&mut c2, &mut b2, &[b"HELLO", b"2"]) {
        RespVal::Array(Some(items)) => assert!(items.contains(&RespVal::int(2))),
        other => panic!("expected RESP2 Array from HELLO 2, got {other:?}"),
    }
    assert_eq!(
        cmd_send(&mut c2, &mut b2, &[b"GET", b"k1"]),
        RespVal::bulk(b"v1".to_vec())
    );

    srv.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_json_and_vxg_sql_interop() {
    let dir = temp_dir("json");
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFsStore::new(&dir).unwrap());
    let mut eng = KvEngine::open(store, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let mut conn = ConnState::new(1);

    assert_eq!(
        exec_local(
            &mut eng,
            &mut conn,
            &[b"JSON.SET", b"u:1", b"$", br#"{"name":"Ada","score":10,"meta":{"tier":"pro"}}"#]
        ),
        RespVal::ok()
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.GET", b"u:1", b"$.name"]),
        RespVal::bulk(br#""Ada""#.to_vec())
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.SET", b"u:1", b"$.meta.city", br#""Istanbul""#]),
        RespVal::ok()
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.NUMINCRBY", b"u:1", b"$.score", b"5.5"]),
        RespVal::bulk(b"15.5".to_vec())
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.SET", b"u:1", b"$.name", br#""No""#, b"NX"]),
        RespVal::nil()
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.SET", b"u:1", b"$.name", br#""Grace""#, b"XX"]),
        RespVal::ok()
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.DEL", b"u:1", b"$.meta.tier"]),
        RespVal::int(1)
    );
    assert_eq!(
        exec_local(&mut eng, &mut conn, &[b"JSON.GET", b"u:1", b"$.meta.tier"]),
        RespVal::nil()
    );

    // Query JSON.SET document directly with VXG.SQL!
    match exec_local(
        &mut eng,
        &mut conn,
        &[b"VXG.SQL", b"SELECT data.name, data.score WHERE data.score > 12", b"TSV"],
    ) {
        RespVal::Bulk(Some(b)) => {
            let txt = String::from_utf8_lossy(&b);
            assert!(txt.contains("Grace") && txt.contains("15.5"), "unexpected sql: {txt}");
        }
        other => panic!("unexpected VXG.SQL reply: {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_geospatial_real_cities() {
    let dir = temp_dir("geo");
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFsStore::new(&dir).unwrap());
    let mut eng = KvEngine::open(store, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let mut conn = ConnState::new(1);

    assert_eq!(
        exec_local(
            &mut eng,
            &mut conn,
            &[
                b"GEOADD", b"cities",
                b"28.9784", b"41.0082", b"istanbul",
                b"32.8597", b"39.9334", b"ankara",
                b"27.1428", b"38.4237", b"izmir",
                b"29.0609", b"40.1885", b"bursa",
            ]
        ),
        RespVal::int(4)
    );
    match exec_local(&mut eng, &mut conn, &[b"GEODIST", b"cities", b"istanbul", b"ankara", b"km"]) {
        RespVal::Bulk(Some(b)) => {
            let km: f64 = String::from_utf8(b).unwrap().parse().unwrap();
            assert!((340.0..=365.0).contains(&km), "istanbul-ankara km={km}");
        }
        other => panic!("unexpected GEODIST: {other:?}"),
    }
    match exec_local(
        &mut eng,
        &mut conn,
        &[b"GEOSEARCH", b"cities", b"FROMMEMBER", b"istanbul", b"BYRADIUS", b"150", b"km", b"ASC", b"WITHDIST"],
    ) {
        RespVal::Array(Some(items)) => {
            assert_eq!(items.len(), 2, "expected istanbul + bursa, got {items:?}");
            assert_eq!(
                items[0],
                RespVal::array(vec![RespVal::bulk(b"istanbul".to_vec()), RespVal::bulk(b"0.0000".to_vec())])
            );
            if let RespVal::Array(Some(pair)) = &items[1] {
                assert_eq!(pair[0], RespVal::bulk(b"bursa".to_vec()));
            } else {
                panic!("expected array pair for bursa");
            }
        }
        other => panic!("unexpected GEOSEARCH: {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_stream_consumer_groups_cross_pod() {
    let dir = temp_dir("xgroup");
    let url = spawn_http_origin(&dir);

    let store_a: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_a = KvEngine::open(
        store_a,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "xg-a".into(),
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let mut ca = ConnState::new(1);

    assert_eq!(
        exec_local(&mut pod_a, &mut ca, &[b"XGROUP", b"CREATE", b"jobs", b"workers", b"0", b"MKSTREAM"]),
        RespVal::ok()
    );
    let id1 = match exec_local(&mut pod_a, &mut ca, &[b"XADD", b"jobs", b"*", b"t", b"a"]) {
        RespVal::Bulk(Some(b)) => b,
        other => panic!("bad XADD: {other:?}"),
    };
    let _id2 = exec_local(&mut pod_a, &mut ca, &[b"XADD", b"jobs", b"*", b"t", b"b"]);
    let _id3 = exec_local(&mut pod_a, &mut ca, &[b"XADD", b"jobs", b"*", b"t", b"c"]);

    // Consumer c1 reads first 2
    match exec_local(
        &mut pod_a,
        &mut ca,
        &[b"XREADGROUP", b"GROUP", b"workers", b"c1", b"COUNT", b"2", b"STREAMS", b"jobs", b">"],
    ) {
        RespVal::Array(Some(streams)) => {
            assert_eq!(streams.len(), 1);
            if let RespVal::Array(Some(pair)) = &streams[0] {
                if let RespVal::Array(Some(entries)) = &pair[1] {
                    assert_eq!(entries.len(), 2);
                } else {
                    panic!("expected entries array");
                }
            }
        }
        other => panic!("bad XREADGROUP: {other:?}"),
    }
    // Ack first entry -> 1 remaining pending
    assert_eq!(
        exec_local(&mut pod_a, &mut ca, &[b"XACK", b"jobs", b"workers", &id1]),
        RespVal::int(1)
    );
    pod_a.commit().unwrap();

    // Pod B pulls and sees persisted group offset + PEL!
    let store_b: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_b = KvEngine::open(
        store_b,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "xg-b".into(),
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let mut cb = ConnState::new(2);

    // Internal metadata key MUST be hidden from DBSIZE & KEYS!
    assert_eq!(exec_local(&mut pod_b, &mut cb, &[b"DBSIZE"]), RespVal::int(1));
    assert_eq!(
        exec_local(&mut pod_b, &mut cb, &[b"KEYS", b"*"]),
        RespVal::array(vec![RespVal::bulk(b"jobs".to_vec())])
    );

    match exec_local(&mut pod_b, &mut cb, &[b"XPENDING", b"jobs", b"workers"]) {
        RespVal::Array(Some(v)) => assert_eq!(v[0], RespVal::int(1)),
        other => panic!("unexpected XPENDING: {other:?}"),
    }
    // Consumer c2 on Pod B reads only the remaining 3rd entry!
    match exec_local(
        &mut pod_b,
        &mut cb,
        &[b"XREADGROUP", b"GROUP", b"workers", b"c2", b"COUNT", b"5", b"STREAMS", b"jobs", b">"],
    ) {
        RespVal::Array(Some(streams)) => {
            if let RespVal::Array(Some(pair)) = &streams[0] {
                if let RespVal::Array(Some(entries)) = &pair[1] {
                    assert_eq!(entries.len(), 1);
                } else {
                    panic!("expected 1 remaining entry");
                }
            }
        }
        other => panic!("bad Pod B XREADGROUP: {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_blpop_brpop_live_wakeup_and_timeout() {
    let dir = temp_dir("blpop");
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFsStore::new(&dir).unwrap());
    let eng = KvEngine::open(store, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let srv = PodServer::spawn(
        listener,
        eng,
        ServerConfig {
            flush_ms: 0,
            sync_ms: 0,
        },
    )
    .unwrap();
    let addr = srv.addr;

    // Thread 1 blocks on empty queues `q1`, `q2`
    let t1 = thread::spawn(move || {
        let mut s = TcpStream::connect(addr).unwrap();
        let mut b = RespBuffer::new();
        let t0 = Instant::now();
        let reply = cmd_send(&mut s, &mut b, &[b"BLPOP", b"q1", b"q2", b"2"]);
        (reply, t0.elapsed())
    });

    thread::sleep(Duration::from_millis(60));
    let mut prod = TcpStream::connect(addr).unwrap();
    let mut pb = RespBuffer::new();
    assert_eq!(
        cmd_send(&mut prod, &mut pb, &[b"RPUSH", b"q2", b"job-42"]),
        RespVal::int(1)
    );

    let (woke_reply, elapsed) = t1.join().unwrap();
    assert_eq!(
        woke_reply,
        RespVal::array(vec![RespVal::bulk(b"q2".to_vec()), RespVal::bulk(b"job-42".to_vec())])
    );
    assert!(elapsed < Duration::from_millis(500), "woke too slowly: {elapsed:?}");

    // Timeout case: BRPOP on empty list expires with nil array `*-1`
    let t_exp = Instant::now();
    let timeout_reply = cmd_send(&mut prod, &mut pb, &[b"BRPOP", b"never_pushed", b"0.12"]);
    assert_eq!(timeout_reply, RespVal::nil_array());
    assert!(t_exp.elapsed() >= Duration::from_millis(100));

    srv.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_o1_dbsize_and_fence_pruned_scan_zero_io() {
    let dir = temp_dir("pruned");
    let url = spawn_http_origin(&dir);

    let store_a: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_a = KvEngine::open(
        store_a,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "writer".into(),
            block_size: 8_192,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    for i in 0..5_000usize {
        let k = format!("item:{i:05}").into_bytes();
        let v = format!("payload_value_for_record_{i:05}_padding_bytes").into_bytes();
        pod_a.set_record(k, Value::Str(v), None);
    }
    let snap = pod_a.compact().unwrap();
    assert_eq!(snap.key_count, 5_000);

    // Cold Pod B: O(1) DBSIZE must do ZERO Range requests!
    let store_b: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_b = KvEngine::open(store_b, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let total_blocks = pod_b.base.as_ref().unwrap().1.blocks.len();
    assert!(total_blocks >= 8, "expected multi-block snapshot, got {total_blocks}");

    assert_eq!(pod_b.dbsize_fast().unwrap(), 5_000);
    assert_eq!(pod_b.stats.range_requests, 0, "cold DBSIZE must do 0 Range requests!");
    assert_eq!(pod_b.stats.range_bytes, 0, "cold DBSIZE must fetch 0 Range bytes!");

    // Mutate overlay: 1 delete + 2 brand-new keys -> exact delta!
    pod_b.del_key(b"item:00500").unwrap();
    pod_b.set_str_in_place(b"zzz:1", b"a");
    pod_b.set_str_in_place(b"zzz:2", b"b");
    assert_eq!(pod_b.dbsize_fast().unwrap(), 5_001);

    // Cold Pod C: fence-pruned prefix SCAN fetches <= 2 blocks out of total_blocks!
    let store_c: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_c = KvEngine::open(store_c, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let (next_c, keys) = pod_c.scan_pruned(0, b"item:0250*", 50).unwrap();
    assert_eq!(next_c, 0);
    assert_eq!(keys.len(), 10);
    assert!(
        pod_c.cache.cached_blocks() <= 2,
        "prefix SCAN should prune non-matching blocks (cached={} / total={total_blocks})",
        pod_c.cache.cached_blocks()
    );

    // Cold Pod D: streaming SCAN 0 COUNT 10 stops after fetching <= 2 blocks!
    let store_d: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_d = KvEngine::open(store_d, Arc::new(PubSubBroker::new()), EngineConfig::default()).unwrap();
    let (next_d, batch) = pod_d.scan_pruned(0, b"*", 10).unwrap();
    assert_eq!(next_d, 10);
    assert_eq!(batch.len(), 10);
    assert!(
        pod_d.cache.cached_blocks() <= 2,
        "streaming SCAN 0 COUNT 10 should stop early (cached={} / total={total_blocks})",
        pod_d.cache.cached_blocks()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_sharded_multi_writer_wal_concurrent_pods() {
    let dir = temp_dir("sharded");
    let url = spawn_http_origin(&dir);

    let mut handles = Vec::new();
    for p in 0..4usize {
        let u = url.clone();
        handles.push(thread::spawn(move || {
            let st: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&u).unwrap());
            let mut eng = KvEngine::open(
                st,
                Arc::new(PubSubBroker::new()),
                EngineConfig {
                    pod_id: format!("shard-pod-{p}"),
                    wal_shards: 4,
                    ..EngineConfig::default()
                },
            )
            .unwrap();
            for batch in 0..5usize {
                for j in 0..10usize {
                    let k = format!("p{p}:b{batch}:k{j}").into_bytes();
                    let v = format!("val_{p}_{batch}_{j}").into_bytes();
                    eng.set_record(k, Value::Str(v), None);
                }
                eng.commit().unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Cold Reader Pod E with wal_shards=4 pulls all 4 shards & verifies 200/200 keys!
    let st_e: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_e = KvEngine::open(
        st_e,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "reader-e".into(),
            wal_shards: 4,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    assert_eq!(pod_e.dbsize_fast().unwrap(), 200);
    for p in 0..4usize {
        for batch in 0..5usize {
            for j in 0..10usize {
                let k = format!("p{p}:b{batch}:k{j}");
                let expected = format!("val_{p}_{batch}_{j}").into_bytes();
                let got = pod_e.get_record(k.as_bytes()).unwrap().expect(&k);
                assert_eq!(got.value, Some(Value::Str(expected)));
            }
        }
    }

    // Compact across all 4 shards and verify Cold Pod F reads all 200 keys cleanly!
    let snap = pod_e.compact().unwrap();
    assert_eq!(snap.key_count, 200);
    assert_eq!(pod_e.active_wal_count(), 0);

    let st_f: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&url).unwrap());
    let mut pod_f = KvEngine::open(
        st_f,
        Arc::new(PubSubBroker::new()),
        EngineConfig {
            pod_id: "reader-f".into(),
            wal_shards: 4,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    assert_eq!(pod_f.dbsize_fast().unwrap(), 200);
    let _ = std::fs::remove_dir_all(&dir);
}
