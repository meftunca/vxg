use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use vxg_kv::{
    encode_command, serve_origin_listener, EngineConfig, HttpObjectStore, KvEngine, LocalFsStore,
    ObjectStore, PodServer, PubSubBroker, RespBuffer, RespVal, ServerConfig,
};

struct RespClient {
    stream: TcpStream,
    buf: RespBuffer,
}

impl RespClient {
    fn connect(addr: SocketAddr) -> Self {
        let stream = TcpStream::connect(addr).unwrap();
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        Self {
            stream,
            buf: RespBuffer::new(),
        }
    }

    fn send_raw(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).unwrap();
    }

    fn read_reply(&mut self) -> RespVal {
        loop {
            if let Some(v) = self.buf.next_value().unwrap() {
                return v;
            }
            let n = self.buf.read_more(&mut self.stream).unwrap();
            assert!(n > 0, "unexpected EOF from server");
        }
    }

    fn cmd(&mut self, args: &[&str]) -> RespVal {
        let bargs: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
        self.send_raw(&encode_command(&bargs));
        self.read_reply()
    }

    fn cmd_bytes(&mut self, args: &[&[u8]]) -> RespVal {
        self.send_raw(&encode_command(args));
        self.read_reply()
    }
}

fn bulk_str(v: &RespVal) -> String {
    match v {
        RespVal::Bulk(Some(b)) => String::from_utf8_lossy(b).into_owned(),
        other => panic!("expected Bulk, got {other:?}"),
    }
}

#[test]
fn resp_all_data_types_transactions_ttl_and_pipelining() {
    let tmp = std::env::temp_dir().join(format!("vxg_kv_types_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFsStore::new(&tmp).unwrap());
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

    let mut c = RespClient::connect(srv.addr);
    assert_eq!(c.cmd(&["PING"]), RespVal::pong());
    assert_eq!(c.cmd(&["ECHO", "hello"]), RespVal::bulk(b"hello".to_vec()));

    // String + NX/XX/GET + binary safety + counters
    assert_eq!(c.cmd(&["SET", "k1", "v1"]), RespVal::ok());
    assert_eq!(c.cmd(&["SET", "k1", "v2", "NX"]), RespVal::nil());
    assert_eq!(
        c.cmd(&["SET", "k1", "v2", "XX", "GET"]),
        RespVal::bulk(b"v1".to_vec())
    );
    assert_eq!(c.cmd(&["APPEND", "k1", "_end"]), RespVal::int(6));
    assert_eq!(c.cmd(&["STRLEN", "k1"]), RespVal::int(6));
    assert_eq!(
        c.cmd(&["GETRANGE", "k1", "0", "2"]),
        RespVal::bulk(b"v2_".to_vec())
    );
    let bin_k = b"\xff\x00bin_key";
    let bin_v = b"\x00\x01\xfe\xffraw";
    assert_eq!(c.cmd_bytes(&[b"SET", bin_k, bin_v]), RespVal::ok());
    assert_eq!(
        c.cmd_bytes(&[b"GET", bin_k]),
        RespVal::bulk(bin_v.to_vec())
    );

    assert_eq!(c.cmd(&["INCR", "ctr"]), RespVal::int(1));
    assert_eq!(c.cmd(&["INCRBY", "ctr", "9"]), RespVal::int(10));
    assert_eq!(c.cmd(&["DECRBY", "ctr", "3"]), RespVal::int(7));
    assert_eq!(c.cmd(&["INCRBYFLOAT", "ctr", "0.5"]), RespVal::bulk(b"7.5".to_vec()));

    // TTL expiration
    assert_eq!(c.cmd(&["SET", "temp", "gone", "PX", "40"]), RespVal::ok());
    assert_eq!(c.cmd(&["GET", "temp"]), RespVal::bulk(b"gone".to_vec()));
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(c.cmd(&["GET", "temp"]), RespVal::nil());
    assert_eq!(c.cmd(&["TTL", "temp"]), RespVal::int(-2));

    // Hash
    assert_eq!(c.cmd(&["HSET", "h1", "a", "1", "b", "2"]), RespVal::int(2));
    assert_eq!(c.cmd(&["HSETNX", "h1", "a", "9"]), RespVal::int(0));
    assert_eq!(c.cmd(&["HINCRBY", "h1", "a", "5"]), RespVal::int(6));
    assert_eq!(c.cmd(&["HGET", "h1", "a"]), RespVal::bulk(b"6".to_vec()));
    assert_eq!(c.cmd(&["HLEN", "h1"]), RespVal::int(2));
    assert_eq!(c.cmd(&["TYPE", "h1"]), RespVal::Simple("hash".to_owned()));

    // List
    assert_eq!(c.cmd(&["RPUSH", "q", "a", "b", "c"]), RespVal::int(3));
    assert_eq!(c.cmd(&["LPUSH", "q", "z"]), RespVal::int(4));
    assert_eq!(
        c.cmd(&["LRANGE", "q", "0", "-1"]),
        RespVal::array(vec![
            RespVal::bulk(b"z".to_vec()),
            RespVal::bulk(b"a".to_vec()),
            RespVal::bulk(b"b".to_vec()),
            RespVal::bulk(b"c".to_vec()),
        ])
    );
    assert_eq!(
        c.cmd(&["LMOVE", "q", "q2", "LEFT", "RIGHT"]),
        RespVal::bulk(b"z".to_vec())
    );
    assert_eq!(c.cmd(&["LLEN", "q2"]), RespVal::int(1));

    // Set
    assert_eq!(c.cmd(&["SADD", "s1", "a", "b", "c"]), RespVal::int(3));
    assert_eq!(c.cmd(&["SADD", "s2", "b", "c", "d"]), RespVal::int(3));
    assert_eq!(
        c.cmd(&["SINTER", "s1", "s2"]),
        RespVal::array(vec![
            RespVal::bulk(b"b".to_vec()),
            RespVal::bulk(b"c".to_vec()),
        ])
    );
    assert_eq!(
        c.cmd(&["SDIFF", "s1", "s2"]),
        RespVal::array(vec![RespVal::bulk(b"a".to_vec())])
    );

    // ZSet
    assert_eq!(
        c.cmd(&["ZADD", "lb", "100", "alice", "250", "bob", "180", "carol"]),
        RespVal::int(3)
    );
    assert_eq!(c.cmd(&["ZREVRANK", "lb", "bob"]), RespVal::int(0));
    assert_eq!(
        c.cmd(&["ZRANGE", "lb", "0", "1", "REV", "WITHSCORES"]),
        RespVal::array(vec![
            RespVal::bulk(b"bob".to_vec()),
            RespVal::bulk(b"250".to_vec()),
            RespVal::bulk(b"carol".to_vec()),
            RespVal::bulk(b"180".to_vec()),
        ])
    );
    assert_eq!(c.cmd(&["ZCOUNT", "lb", "100", "200"]), RespVal::int(2));

    // Transactions + WATCH optimistic abort
    let mut c2 = RespClient::connect(srv.addr);
    assert_eq!(c.cmd(&["WATCH", "k1"]), RespVal::ok());
    assert_eq!(c2.cmd(&["SET", "k1", "mutated_by_c2"]), RespVal::ok());
    assert_eq!(c.cmd(&["MULTI"]), RespVal::ok());
    assert_eq!(
        c.cmd(&["SET", "k1", "should_abort"]),
        RespVal::Simple("QUEUED".to_owned())
    );
    assert_eq!(c.cmd(&["EXEC"]), RespVal::nil_array());
    assert_eq!(
        c.cmd(&["GET", "k1"]),
        RespVal::bulk(b"mutated_by_c2".to_vec())
    );

    // Pipelining (10 commands in 1 TCP write!)
    let mut pipe = Vec::new();
    for i in 0..10 {
        let k = format!("pipe:{i}");
        let v = format!("val:{i}");
        pipe.extend_from_slice(&encode_command(&[b"SET", k.as_bytes(), v.as_bytes()]));
    }
    c.send_raw(&pipe);
    for _ in 0..10 {
        assert_eq!(c.read_reply(), RespVal::ok());
    }
    assert_eq!(c.cmd(&["GET", "pipe:9"]), RespVal::bulk(b"val:9".to_vec()));

    // Streams (XADD / XLEN / XRANGE / XREVRANGE / XREAD / XDEL / XTRIM)
    let id1 = bulk_str(&c.cmd(&["XADD", "ev", "1000-0", "kind", "login", "u", "alice"]));
    assert_eq!(id1, "1000-0");
    let id2 = bulk_str(&c.cmd(&["XADD", "ev", "1001-0", "kind", "buy", "u", "bob"]));
    assert_eq!(id2, "1001-0");
    let id3 = bulk_str(&c.cmd(&["XADD", "ev", "*", "kind", "logout", "u", "alice"]));
    assert!(id3.contains('-'));
    assert_eq!(c.cmd(&["XLEN", "ev"]), RespVal::int(3));
    assert_eq!(c.cmd(&["TYPE", "ev"]), RespVal::Simple("stream".to_owned()));
    let xr = c.cmd(&["XRANGE", "ev", "-", "+", "COUNT", "2"]);
    assert!(matches!(xr, RespVal::Array(Some(ref items)) if items.len() == 2));
    let xread = c.cmd(&["XREAD", "STREAMS", "ev", "1000-0"]);
    assert!(matches!(xread, RespVal::Array(Some(ref items)) if items.len() == 1));
    assert_eq!(c.cmd(&["XDEL", "ev", "1000-0"]), RespVal::int(1));
    assert_eq!(c.cmd(&["XTRIM", "ev", "MAXLEN", "1"]), RespVal::int(1));
    assert_eq!(c.cmd(&["XLEN", "ev"]), RespVal::int(1));

    // Bitmaps (SETBIT / GETBIT / BITCOUNT / BITOP / BITPOS)
    assert_eq!(c.cmd(&["SETBIT", "bm1", "7", "1"]), RespVal::int(0));
    assert_eq!(c.cmd(&["SETBIT", "bm1", "15", "1"]), RespVal::int(0));
    assert_eq!(c.cmd(&["GETBIT", "bm1", "7"]), RespVal::int(1));
    assert_eq!(c.cmd(&["BITCOUNT", "bm1"]), RespVal::int(2));
    assert_eq!(c.cmd(&["BITPOS", "bm1", "1"]), RespVal::int(7));
    assert_eq!(c.cmd(&["SETBIT", "bm2", "7", "1"]), RespVal::int(0));
    assert_eq!(c.cmd(&["BITOP", "AND", "bm_and", "bm1", "bm2"]), RespVal::int(2));
    assert_eq!(c.cmd(&["BITCOUNT", "bm_and"]), RespVal::int(1));

    // HyperLogLog (PFADD / PFCOUNT / PFMERGE)
    for i in 0..500 {
        let u = format!("visitor:{i}");
        c.cmd(&["PFADD", "dau:1", &u]);
    }
    for i in 250..750 {
        let u = format!("visitor:{i}");
        c.cmd(&["PFADD", "dau:2", &u]);
    }
    let RespVal::Integer(c1) = c.cmd(&["PFCOUNT", "dau:1"]) else {
        panic!("expected int");
    };
    assert!((470..=530).contains(&c1), "HLL dau:1 estimate={c1}");
    assert_eq!(c.cmd(&["PFMERGE", "dau:all", "dau:1", "dau:2"]), RespVal::ok());
    let RespVal::Integer(c_all) = c.cmd(&["PFCOUNT", "dau:all"]) else {
        panic!("expected int");
    };
    assert!((700..=800).contains(&c_all), "HLL dau:all estimate={c_all}");

    srv.stop();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn multipod_http_origin_cas_range_bloom_timetravel_sql_and_pubsub() {
    let tmp = std::env::temp_dir().join(format!("vxg_kv_multipod_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Start single shared HTTP/1.1 CAS + Range Origin
    let orig_listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let orig_addr = orig_listener.local_addr().unwrap();
    let orig_dir = tmp.clone();
    std::thread::spawn(move || {
        let _ = serve_origin_listener(orig_listener, &orig_dir);
    });
    let origin_url = format!("http://{orig_addr}");

    let spawn_pod = |pod_id: &str, flush_ms: u64, sync_ms: u64| -> PodServer {
        let store: Arc<dyn ObjectStore> = Arc::new(HttpObjectStore::connect(&origin_url).unwrap());
        let eng = KvEngine::open(
            store,
            Arc::new(PubSubBroker::new()),
            EngineConfig {
                pod_id: pod_id.to_owned(),
                block_size: 4096,
                auto_compact_wals: 0,
                ..EngineConfig::default()
            },
        )
        .unwrap();
        let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        PodServer::spawn(l, eng, ServerConfig { flush_ms, sync_ms }).unwrap()
    };

    let pod_a = spawn_pod("pod-a", 0, 0);
    let pod_b = spawn_pod("pod-b", 0, 30); // auto-pull every 30ms for live cross-pod Pub/Sub!
    let pod_c = spawn_pod("pod-c", 0, 0);

    let mut ca = RespClient::connect(pod_a.addr);
    let mut cb = RespClient::connect(pod_b.addr);
    let mut cc = RespClient::connect(pod_c.addr);

    // 1) Concurrent multi-pod writes -> CAS 412 automatic rebase & retry (0 data loss!)
    assert_eq!(ca.cmd(&["SET", "from_a", "hello_a"]), RespVal::ok());
    assert_eq!(cb.cmd(&["SET", "from_b", "hello_b"]), RespVal::ok());
    assert_eq!(ca.cmd(&["HSET", "shared_hash", "f1", "10"]), RespVal::int(1));
    assert_eq!(cb.cmd(&["ZADD", "shared_z", "42", "hero"]), RespVal::int(1));
    assert_eq!(
        ca.cmd(&["XADD", "shared_stream", "100-0", "ev", "boot"]),
        RespVal::bulk(b"100-0".to_vec())
    );

    let ep1 = ca.cmd(&["VXG.COMMIT"]);
    assert!(matches!(ep1, RespVal::Integer(n) if n >= 1));
    let ep2 = cb.cmd(&["VXG.COMMIT"]); // triggers 412 -> pull -> rebase -> retry!
    assert!(matches!(ep2, RespVal::Integer(n) if n >= 2));

    assert!(matches!(cc.cmd(&["VXG.PULL"]), RespVal::Integer(_)));
    assert_eq!(cc.cmd(&["GET", "from_a"]), RespVal::bulk(b"hello_a".to_vec()));
    assert_eq!(cc.cmd(&["GET", "from_b"]), RespVal::bulk(b"hello_b".to_vec()));
    assert_eq!(cc.cmd(&["HGET", "shared_hash", "f1"]), RespVal::bulk(b"10".to_vec()));
    assert_eq!(cc.cmd(&["ZSCORE", "shared_z", "hero"]), RespVal::bulk(b"42".to_vec()));
    assert_eq!(cc.cmd(&["XLEN", "shared_stream"]), RespVal::int(1));

    // 2) Linearizable cross-pod Compare-And-Swap (VXG.CAS)
    assert_eq!(
        ca.cmd(&["VXG.CAS", "lock:leader", "__NIL__", "pod-a"]),
        RespVal::int(1)
    );
    assert_eq!(
        cb.cmd(&["VXG.CAS", "lock:leader", "__NIL__", "pod-b"]),
        RespVal::int(0)
    );
    assert_eq!(
        cb.cmd(&["VXG.CAS", "lock:leader", "pod-a", "pod-b"]),
        RespVal::int(1)
    );
    cc.cmd(&["VXG.PULL"]);
    assert_eq!(
        cc.cmd(&["GET", "lock:leader"]),
        RespVal::bulk(b"pod-b".to_vec())
    );

    // 3) Load 3,000 sorted JSON records on Pod A & COMPACT -> Snapshot E1
    {
        let plans = ["free", "pro", "enterprise"];
        let mut pipe = Vec::new();
        for i in 0..3000usize {
            let k = format!("user:{i:06}");
            let v = format!(
                "{{\"plan\":\"{}\",\"spend\":{}}}",
                plans[i % 3],
                (i % 100) + 1
            );
            pipe.extend_from_slice(&encode_command(&[b"SET", k.as_bytes(), v.as_bytes()]));
        }
        ca.send_raw(&pipe);
        for _ in 0..3000 {
            assert_eq!(ca.read_reply(), RespVal::ok());
        }
    }
    let compact_e1 = ca.cmd(&["VXG.COMPACT"]);
    let e1_str = match &compact_e1 {
        RespVal::Simple(s) => s.clone(),
        other => panic!("expected OK compact, got {other:?}"),
    };
    let snap_e1: u64 = e1_str
        .split_whitespace()
        .find_map(|w| w.strip_prefix("epoch="))
        .unwrap()
        .parse()
        .unwrap();

    // 4) Cold Pod C pulls ONLY manifest+index and fetches 1 key via HTTP Range (< 20% of archive!)
    cc.cmd(&["VXG.PULL"]);
    let got_u = bulk_str(&cc.cmd(&["GET", "user:001500"]));
    assert_eq!(got_u, "{\"plan\":\"free\",\"spend\":1}");

    let stats_after_cold = bulk_str(&cc.cmd(&["VXG.STATS"]));
    let range_bytes: u64 = stats_after_cold
        .lines()
        .find_map(|l| l.strip_prefix("range_bytes:"))
        .unwrap()
        .parse()
        .unwrap();
    let comp_bytes: u64 = stats_after_cold
        .lines()
        .find_map(|l| l.strip_prefix("base_compressed_bytes:"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (range_bytes as f64) * 100.0 / (comp_bytes as f64) < 25.0,
        "expected <25% fetched on cold single GET, got {range_bytes}B / {comp_bytes}B"
    );

    // 5) Bloom filter blocks 200 missing keys inside key range with ~0 Range requests!
    let reqs_before: u64 = stats_after_cold
        .lines()
        .find_map(|l| l.strip_prefix("range_requests:"))
        .unwrap()
        .parse()
        .unwrap();
    for i in 0..200 {
        let miss = format!("user:{:06}_missing", i * 11 + 5);
        assert_eq!(cc.cmd(&["GET", &miss]), RespVal::nil());
    }
    let stats_after_bloom = bulk_str(&cc.cmd(&["VXG.STATS"]));
    let bloom_filtered: u64 = stats_after_bloom
        .lines()
        .find_map(|l| l.strip_prefix("bloom_filtered:"))
        .unwrap()
        .parse()
        .unwrap();
    let reqs_after: u64 = stats_after_bloom
        .lines()
        .find_map(|l| l.strip_prefix("range_requests:"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        bloom_filtered >= 185,
        "expected >=185/200 Bloom short-circuits, got {bloom_filtered}"
    );
    assert!(
        reqs_after - reqs_before <= 15,
        "expected <=15 false-positive Range reqs out of 200, got {}",
        reqs_after - reqs_before
    );

    // 6) Compressed-domain VXG.SQL directly on Pod C!
    let sql_res = bulk_str(&cc.cmd(&[
        "VXG.SQL",
        "SELECT data.plan, COUNT(*) AS n, SUM(data.spend) AS total WHERE _t = 'json' GROUP BY data.plan ORDER BY data.plan ASC",
    ]));
    assert!(sql_res.contains("enterprise\t1000\t"), "sql_res:\n{sql_res}");
    assert!(sql_res.contains("free\t1000\t"), "sql_res:\n{sql_res}");
    assert!(sql_res.contains("pro\t1000\t"), "sql_res:\n{sql_res}");

    // 7) Time-Travel (VXG.AT & VXG.PIN / VXG.UNPIN) across compacted snapshots!
    assert_eq!(
        ca.cmd(&["SET", "user:001500", "{\"plan\":\"vip\",\"spend\":9999}"]),
        RespVal::ok()
    );
    ca.cmd(&["VXG.COMPACT"]);
    cc.cmd(&["VXG.PULL"]);
    assert_eq!(
        bulk_str(&cc.cmd(&["GET", "user:001500"])),
        "{\"plan\":\"vip\",\"spend\":9999}"
    );
    assert_eq!(
        bulk_str(&cc.cmd(&["VXG.AT", &snap_e1.to_string(), "GET", "user:001500"])),
        "{\"plan\":\"free\",\"spend\":1}"
    );
    assert_eq!(cc.cmd(&["VXG.PIN", &snap_e1.to_string()]), RespVal::ok());
    assert_eq!(
        bulk_str(&cc.cmd(&["GET", "user:001500"])),
        "{\"plan\":\"free\",\"spend\":1}"
    );
    assert_eq!(cc.cmd(&["VXG.UNPIN"]), RespVal::ok());
    assert_eq!(
        bulk_str(&cc.cmd(&["GET", "user:001500"])),
        "{\"plan\":\"vip\",\"spend\":9999}"
    );

    // 8) Real-time Cross-Pod Pub/Sub (Publish on Pod A -> Subscribe & PSubscribe on Pod B!)
    let mut sub_b = RespClient::connect(pod_b.addr);
    assert_eq!(
        sub_b.cmd(&["SUBSCRIBE", "orders"]),
        RespVal::array(vec![
            RespVal::bulk(b"subscribe".to_vec()),
            RespVal::bulk(b"orders".to_vec()),
            RespVal::int(1),
        ])
    );
    assert_eq!(
        sub_b.cmd(&["PSUBSCRIBE", "alerts:*"]),
        RespVal::array(vec![
            RespVal::bulk(b"psubscribe".to_vec()),
            RespVal::bulk(b"alerts:*".to_vec()),
            RespVal::int(2),
        ])
    );
    ca.cmd(&["PUBLISH", "orders", "order-777"]);
    ca.cmd(&["PUBLISH", "alerts:cpu", "99pct"]);
    ca.cmd(&["VXG.COMMIT"]);

    // Pod B's background sync loop (30ms) pulls the WAL and fans out BOTH messages to sub_b!
    let msg1 = sub_b.read_reply();
    let msg2 = sub_b.read_reply();
    assert_eq!(
        msg1,
        RespVal::array(vec![
            RespVal::bulk(b"message".to_vec()),
            RespVal::bulk(b"orders".to_vec()),
            RespVal::bulk(b"order-777".to_vec()),
        ])
    );
    assert_eq!(
        msg2,
        RespVal::array(vec![
            RespVal::bulk(b"pmessage".to_vec()),
            RespVal::bulk(b"alerts:*".to_vec()),
            RespVal::bulk(b"alerts:cpu".to_vec()),
            RespVal::bulk(b"99pct".to_vec()),
        ])
    );

    pod_a.stop();
    pod_b.stop();
    pod_c.stop();
    let _ = std::fs::remove_dir_all(&tmp);
}
