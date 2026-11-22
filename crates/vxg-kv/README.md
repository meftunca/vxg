# `vxg-kv` — Stateless Snapshot Redis (RESP2) + Cross-Pod Pub/Sub + Compressed-Domain SQL over VXG

**Zero external dependencies. Zero dedicated Redis cluster. Object-storage source of truth for thousands of stateless pods.**

`vxg-kv` speaks standard **Redis RESP2** (`redis-cli`, `ioredis`, `redis-py`, `go-redis`) while storing state as immutable, key-sorted **`VXG v5`** snapshots + atomic CAS delta WAL segments over any `ObjectStore` (`HttpObjectStore`, `LocalFsStore`, or your own 6-method S3/R2/GCS adapter).

## Why `vxg-kv`?

- **No Redis host / cluster to run:** Pods keep only their hot working set (`--cache-mb 64`) in RAM; cold keys are fetched via **1 HTTP/1.1 `Range` request** for strictly the 1 matching VXG block (`~4–5%` of archive), and non-existent keys are rejected locally by per-block **10-bit/key Bloom filters (`0` Range I/O)**.
- **40× smaller state:** Sorted keys + JSON payloads compress via `VXG v5`'s `FrontCodedDict` + `BLOCK_COLUMN` (`2.99 MB -> 72 KB` on 25,000 JSON user records).
- **Linearizable multi-pod source of truth:** Atomic `ETag` Conditional PUT (`If-Match` / `If-None-Match: *`) with automatic `412` pull + rebase + retry, plus **`VXG.CAS`** for distributed locks & leader election.
- **Superpowers Redis doesn't have:**
  - **Time-Travel:** `VXG.SNAPSHOTS`, `VXG.AT <epoch> GET <key>`, `VXG.PIN <epoch>` / `VXG.UNPIN`
  - **Compressed-Domain SQL:** `VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) WHERE _t = 'json' GROUP BY data.plan"`
  - **Cross-Pod Pub/Sub:** `SUBSCRIBE` / `PSUBSCRIBE` / `PUBLISH` fans out both locally (`<50 µs`) and across pods through the shared origin WAL.

## Quickstart (Multi-Pod Demo)

```bash
# 1. Start the built-in zero-dep HTTP/1.1 CAS + Range Object Store origin
cargo run --release -p vxg-kv -- origin ./kv-origin --port 9000

# 2. Start Pod A & Pod B pointing to the same origin
cargo run --release -p vxg-kv -- server --port 6379 --origin http://127.0.0.1:9000 --pod-id pod-a
cargo run --release -p vxg-kv -- server --port 6380 --origin http://127.0.0.1:9000 --pod-id pod-b

# 3. Talk to either pod with standard redis-cli!
redis-cli -p 6379 SET user:1 '{"plan":"pro","spend":120}' SYNC
redis-cli -p 6380 GET user:1
redis-cli -p 6380 VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) WHERE _t = 'json' GROUP BY data.plan"

# 4. Run the built-in 100k-key live HTTP Range + Bloom + SQL + TCP benchmark
cargo run --release -p vxg-kv -- bench --keys 100000
```

## Head-to-Head vs `Redis 8.10.3` (`redis-benchmark`, Same Host)

Zero-dependency native **`kqueue` (macOS) / `epoll` (Linux)** single-threaded lock-free reactor + 64-bit `FxHashMap` + zero-allocation slice RESP2 parser + 1-cycle jump-table fast path:

| Benchmark (`redis-benchmark`) | `Redis 8.10.3` | `vxg-kv v0.2.0` | Speedup / Savings |
| :--- | ---: | ---: | :---: |
| **`c=1, P=16` (11 / 11 commands)** | 234k – 588k ops/s (`1.23s` CPU) | **649k – 725k ops/s (`0.59s` CPU)** | **`1.17× – 2.77×`** (`2.1×` less CPU!) |
| **`c=16, P=16` — `PING` / `GET` / `SET` / `INCR`** | 1.33M – 1.61M ops/s | **1.69M – 1.79M ops/s** | **`1.09× – 1.32×`** |
| **`c=16, P=16` — `LPUSH` / `RPUSH` / `LPOP`** | 1.32M – 1.52M ops/s | **1.56M – 1.79M ops/s** | **`1.14× – 1.36×`** |
| **`c=16, P=16` — `SADD` / `HSET` / `ZADD`** | 1.37M – 1.64M ops/s | **1.85M – 2.17M ops/s** | **`1.13× – 1.59×`** |
| **`c=16, P=16` — `MSET (10 keys)`** | 333,333 ops/s | **1,515,152 ops/s** | **`4.55×`** |
| **`c=16, P=16` — Total CPU Time (11 commands)** | `0.99s` | **`0.55s`** | **`1.80×` less CPU!** |
| **100k JSON Load (`12 MB` raw, `P=500`)** | `199.4 ms` | **`156.2 ms`** *(+ `58.8 ms` compact)* | **`1.28×` faster** |
| **100k JSON Disk / Snapshot** | `5,458 KiB` (`dump.rdb`) | **`271.3 KiB`** (`snap.vxg`) | **`20.1×` smaller** |
| **100k JSON Cold Reader Pod RAM (RSS)** | `20.58 MiB` | **`2.80 MiB`** | **`7.35×` less RAM** |
| **100k JSON `GROUP BY` Analytics** | `202.29 ms` (`MGET` + JSON) | **`7.96 ms`** (`VXG.SQL`) | **`25.4×` faster** |

## Supported Commands

- **Strings & Counters:** `GET`, `SET [EX|PX|EXAT|PXAT|NX|XX|GET|KEEPTTL|SYNC]`, `MGET`, `MSET`, `MSETNX`, `DEL`, `UNLINK`, `EXISTS`, `GETDEL`, `APPEND`, `STRLEN`, `GETRANGE`, `INCR`, `DECR`, `INCRBY`, `DECRBY`, `INCRBYFLOAT`
- **TTL & Keys:** `EXPIRE`, `PEXPIRE`, `EXPIREAT`, `PEXPIREAT`, `TTL`, `PTTL`, `PERSIST`, `TYPE`, `RENAME`, `RENAMENX`, `KEYS`, `SCAN`, `DBSIZE`, `FLUSHDB`, `INFO`
- **Hash:** `HSET`, `HSETNX`, `HGET`, `HMGET`, `HGETALL`, `HDEL`, `HEXISTS`, `HLEN`, `HKEYS`, `HVALS`, `HINCRBY`, `HINCRBYFLOAT`
- **List:** `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `LLEN`, `LRANGE`, `LINDEX`, `LSET`, `LTRIM`, `LREM`, `LMOVE`
- **Set:** `SADD`, `SREM`, `SMEMBERS`, `SISMEMBER`, `SMISMEMBER`, `SCARD`, `SPOP`, `SINTER`, `SUNION`, `SDIFF`
- **Sorted Set:** `ZADD [NX|XX|GT|LT|CH]`, `ZREM`, `ZSCORE`, `ZRANK`, `ZREVRANK`, `ZCARD`, `ZRANGE [BYSCORE|REV|WITHSCORES|LIMIT]`, `ZRANGEBYSCORE`, `ZCOUNT`, `ZINCRBY`, `ZPOPMIN`, `ZPOPMAX`
- **Streams:** `XADD [MAXLEN [~] n]`, `XLEN`, `XRANGE`, `XREVRANGE`, `XREAD [COUNT n] STREAMS ...`, `XDEL`, `XTRIM`
- **Bitmaps & HyperLogLog:** `SETBIT`, `GETBIT`, `BITCOUNT`, `BITOP AND|OR|XOR|NOT`, `BITPOS`, `PFADD`, `PFCOUNT`, `PFMERGE`
- **Transactions & Pub/Sub:** `MULTI`, `EXEC`, `DISCARD`, `WATCH`, `UNWATCH`, `PUBLISH`, `SUBSCRIBE`, `UNSUBSCRIBE`, `PSUBSCRIBE`, `PUNSUBSCRIBE`, `PUBSUB CHANNELS|NUMSUB`
- **VXG Extensions:** `VXG.COMMIT`, `VXG.PULL`, `VXG.COMPACT`, `VXG.SNAPSHOTS`, `VXG.AT <epoch> GET <key>`, `VXG.PIN <epoch>`, `VXG.UNPIN`, `VXG.CAS <key> <expected|__NIL__> <new> [EX|PX n]`, `VXG.SQL "<sql>" [TSV|CSV|JSON]`, `VXG.STATS`
