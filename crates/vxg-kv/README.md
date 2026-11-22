<p align="center">
  <img src="../../assets/logos/vxg-kv.svg" width="88" height="88" alt="VXG-KV Logo" />
</p>

<h1 align="center">VXG-KV <code>v0.3.0</code></h1>
<p align="center">
  <strong>Stateless Snapshot Redis (RESP2/RESP3) · JSON &amp; Geo · Streams &amp; Consumer Groups · Cross-Pod Pub/Sub · Compressed-Domain SQL</strong><br/>
  Zero external dependencies · Native <code>kqueue</code>/<code>epoll</code> reactor · <code>11/11</code> faster than Redis 8.10.3 · <code>20×</code> smaller snapshots
</p>

<p align="center">
  <a href="https://github.com/meftunca/vxg/releases/tag/vxg-kv-v0.3.0"><img src="https://img.shields.io/badge/release-vxg--kv--v0.3.0-F97316?style=flat-square" alt="Release" /></a>
  <a href="https://meftunca.github.io/vxg/KV"><img src="https://img.shields.io/badge/docs-VXG--KV-10B981?style=flat-square" alt="Docs" /></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT License" /></a>
</p>

**Zero external dependencies. Zero dedicated Redis cluster. Object-storage source of truth for thousands of stateless pods.**

`vxg-kv` speaks standard **Redis RESP2 & RESP3 (`HELLO 3`)** (`redis-cli`, `ioredis`, `redis-py`, `go-redis`) while storing state as immutable, key-sorted **`VXG v5`** snapshots + atomic CAS delta WAL segments over any `ObjectStore` (`HttpObjectStore`, `LocalFsStore`, or your own 6-method S3/R2/GCS adapter).

## Why `vxg-kv`?

- **No Redis host / cluster to run:** Pods keep only their hot working set (`--cache-mb 64`) in RAM; cold keys are fetched via **1 HTTP/1.1 `Range` request** for strictly the 1 matching VXG block (`~4–5%` of archive), non-existent keys are rejected locally by per-block **10-bit/key Bloom filters (`0` Range I/O)**, `DBSIZE` runs in **$O(1)$ (`0` Range bytes)**, and `SCAN`/`KEYS` use **`SnapshotIndex` fence-pointer pruning + early-stop streaming**.
- **20×–40× smaller state:** Sorted keys + JSON payloads compress via `VXG v5`'s `FrontCodedDict` + `BLOCK_COLUMN` (`12 MB -> 271 KiB` on 100k JSON records vs Redis `dump.rdb` `5,458 KiB`).
- **Linearizable + horizontally sharded multi-pod writes:** Atomic `ETag` Conditional PUT (`If-Match` / `If-None-Match: *`) with automatic `412` pull + rebase + retry, optional **`--wal-shards N`** partitioned WAL manifests (`shards/s00.json..sNN.json`) for high-concurrency multi-pod writes, plus **`VXG.CAS`** for distributed locks & leader election.
- **Full Redis & Superpower feature set (`v0.3.0`):**
  - **Security & RESP3:** `--requirepass` / `VXG_KV_PASSWORD`, `AUTH`, `ACL WHOAMI|LIST`, `HELLO 2|3 [AUTH user pass]`
  - **Native RedisJSON & Geospatial:** `JSON.SET/GET/DEL/NUMINCRBY` (directly queryable via `VXG.SQL`!) + `GEOADD/GEOPOS/GEODIST/GEOSEARCH`
  - **Non-blocking Queues & Stream Consumer Groups:** reactor-native zero-thread `BLPOP`/`BRPOP` + persisted cross-pod `XGROUP`/`XREADGROUP`/`XACK`/`XPENDING`
  - **Time-Travel & Compressed-Domain SQL:** `VXG.SNAPSHOTS`, `VXG.AT <epoch> GET <key>`, `VXG.PIN <epoch>`, `VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) GROUP BY data.plan"` (`26.8×` faster than Redis `MGET`+JSON)
  - **Cross-Pod Pub/Sub:** `SUBSCRIBE` / `PSUBSCRIBE` / `PUBLISH` fans out both locally (`<50 µs`) and across pods through the shared origin WAL.

## Quickstart (Multi-Pod Demo)

```bash
# 1. Start the built-in zero-dep HTTP/1.1 CAS + Range Object Store origin
cargo run --release -p vxg-kv -- origin ./kv-origin --port 9000

# 2. Start Pod A & Pod B pointing to the same origin (optional: --wal-shards 4 --requirepass secret)
cargo run --release -p vxg-kv -- server --port 6379 --origin http://127.0.0.1:9000 --pod-id pod-a --wal-shards 4
cargo run --release -p vxg-kv -- server --port 6380 --origin http://127.0.0.1:9000 --pod-id pod-b --wal-shards 4

# 3. Talk to either pod with standard redis-cli!
redis-cli -p 6379 JSON.SET user:1 '$' '{"plan":"pro","spend":120}'
redis-cli -p 6379 VXG.COMMIT
redis-cli -p 6380 JSON.GET user:1 '$.plan'
redis-cli -p 6380 VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) WHERE _t = 'json' GROUP BY data.plan"

# 4. Run the built-in 100k-key live HTTP Range + Bloom + SQL + TCP benchmark
cargo run --release -p vxg-kv -- bench --keys 100000
```

## Head-to-Head vs `Redis 8.10.3` (`redis-benchmark`, Same Host)

Zero-dependency native **`kqueue` (macOS) / `epoll` (Linux)** single-threaded lock-free reactor + 64-bit `FxHashMap` + zero-allocation slice RESP2 parser + 1-cycle jump-table fast path:

| Benchmark (`redis-benchmark`) | `Redis 8.10.3` | `vxg-kv v0.3.0` | Speedup / Savings |
| :--- | ---: | ---: | :---: |
| **`c=1, P=16` (11 / 11 commands)** | 220k – 592k ops/s (`1.29s` CPU) | **629k – 725k ops/s (`0.58s` CPU)** | **`1.15× – 2.86×`** (`2.22×` less CPU!) |
| **`c=16, P=16` — `PING` / `GET` / `SET` / `INCR`** | 1.27M – 1.56M ops/s | **1.67M – 1.85M ops/s** | **`1.10× – 1.34×`** |
| **`c=16, P=16` — `LPUSH` / `RPUSH` / `LPOP`** | 1.19M – 1.35M ops/s | **1.75M – 1.85M ops/s** | **`1.35× – 1.54×`** |
| **`c=16, P=16` — `SADD` / `HSET` / `ZADD`** | 1.19M – 1.52M ops/s | **1.69M – 1.85M ops/s** | **`1.14× – 1.56×`** |
| **`c=16, P=16` — `MSET (10 keys)`** | 310,559 ops/s | **1,666,667 ops/s** | **`5.37×`** |
| **`c=16, P=16` — Total CPU Time (11 commands)** | `1.06s` | **`0.55s`** | **`1.93×` less CPU!** |
| **100k JSON Load (`12 MB` raw, `P=500`)** | `190.7 ms` | **`152.6 ms`** *(+ `60.3 ms` compact)* | **`1.25×` faster** |
| **100k JSON Disk / Snapshot** | `5,458 KiB` (`dump.rdb`) | **`271.3 KiB`** (`snap.vxg`) | **`20.1×` smaller** |
| **100k JSON Cold Reader Pod RAM (RSS)** | `20.12 MiB` | **`2.86 MiB`** | **`7.03×` less RAM** |
| **100k JSON `GROUP BY` Analytics** | `210.99 ms` (`MGET` + JSON) | **`7.88 ms`** (`VXG.SQL`) | **`26.8×` faster** |

## Supported Commands

- **Connection & Security:** `PING`, `ECHO`, `HELLO [2|3] [AUTH user pass]`, `AUTH [user] pass`, `ACL WHOAMI|LIST`, `CLIENT`, `COMMAND`, `TIME`, `QUIT`
- **Strings & Counters:** `GET`, `SET [EX|PX|EXAT|PXAT|NX|XX|GET|KEEPTTL|SYNC]`, `MGET`, `MSET`, `MSETNX`, `DEL`, `UNLINK`, `EXISTS`, `GETDEL`, `APPEND`, `STRLEN`, `GETRANGE`, `INCR`, `DECR`, `INCRBY`, `DECRBY`, `INCRBYFLOAT`
- **TTL & Keys (Index-Native):** `EXPIRE`, `PEXPIRE`, `EXPIREAT`, `PEXPIREAT`, `TTL`, `PTTL`, `PERSIST`, `TYPE`, `RENAME`, `RENAMENX`, `KEYS` *(fence-pruned)*, `SCAN` *(fence-pruned streaming)*, `DBSIZE` *($O(1)$ index-native)*, `FLUSHDB`, `FLUSHALL`, `INFO`
- **JSON & Geospatial:** `JSON.SET [NX|XX]`, `JSON.GET`, `JSON.DEL`, `JSON.FORGET`, `JSON.NUMINCRBY`, `GEOADD`, `GEOPOS`, `GEODIST`, `GEOSEARCH [FROMMEMBER|FROMLONLAT] BYRADIUS ... [ASC|DESC] [WITHDIST] [COUNT n]`
- **Hash:** `HSET`, `HSETNX`, `HGET`, `HMGET`, `HGETALL`, `HDEL`, `HEXISTS`, `HLEN`, `HKEYS`, `HVALS`, `HINCRBY`, `HINCRBYFLOAT`
- **List & Blocking Queues:** `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `BLPOP`, `BRPOP` *(zero-thread reactor waiters)*, `LLEN`, `LRANGE`, `LINDEX`, `LSET`, `LTRIM`, `LREM`, `LMOVE`
- **Set:** `SADD`, `SREM`, `SMEMBERS`, `SISMEMBER`, `SMISMEMBER`, `SCARD`, `SPOP`, `SINTER`, `SUNION`, `SDIFF`
- **Sorted Set:** `ZADD [NX|XX|GT|LT|CH]`, `ZREM`, `ZSCORE`, `ZRANK`, `ZREVRANK`, `ZCARD`, `ZRANGE [BYSCORE|REV|WITHSCORES|LIMIT]`, `ZRANGEBYSCORE`, `ZCOUNT`, `ZINCRBY`, `ZPOPMIN`, `ZPOPMAX`
- **Streams & Consumer Groups:** `XADD [MAXLEN [~] n]`, `XLEN`, `XRANGE`, `XREVRANGE`, `XREAD [COUNT n] STREAMS ...`, `XDEL`, `XTRIM`, `XGROUP CREATE|DESTROY|SETID`, `XREADGROUP GROUP g c [COUNT n] [NOACK] STREAMS ...`, `XACK`, `XPENDING`
- **Bitmaps & HyperLogLog:** `SETBIT`, `GETBIT`, `BITCOUNT`, `BITOP AND|OR|XOR|NOT`, `BITPOS`, `PFADD`, `PFCOUNT`, `PFMERGE`
- **Transactions & Pub/Sub:** `MULTI`, `EXEC`, `DISCARD`, `WATCH`, `UNWATCH`, `PUBLISH`, `SUBSCRIBE`, `UNSUBSCRIBE`, `PSUBSCRIBE`, `PUNSUBSCRIBE`, `PUBSUB CHANNELS|NUMSUB`
- **VXG Extensions:** `VXG.COMMIT`, `VXG.PULL`, `VXG.COMPACT`, `VXG.SNAPSHOTS`, `VXG.AT <epoch> GET <key>`, `VXG.PIN <epoch>`, `VXG.UNPIN`, `VXG.CAS <key> <expected|__NIL__> <new> [EX|PX n]`, `VXG.SQL "<sql>" [TSV|CSV|JSON]`, `VXG.STATS`
