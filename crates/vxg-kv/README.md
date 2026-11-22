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

| Benchmark (`redis-benchmark`) | `Redis 8.10.3` | `vxg-kv` | Speedup |
| :--- | ---: | ---: | :---: |
| **`c=1, P=1` (11 commands: `PING`/`SET`/`GET`/`INCR`/`LPUSH`/`HSET`/`ZADD`/`MSET`…)** | 42.6k – 48.9k ops/s (`1.74s` CPU, `7.8 MiB`) | **46.7k – 51.4k ops/s (`1.44s` CPU, `5.0 MiB`)** | **`1.03× – 1.13×`** (11/11) |
| **`c=1, P=16` (`PING`/`SET`/`GET`/`INCR`/`LPUSH`/`LPOP`/`HSET`/`ZADD`/`MSET`)** | 235k – 595k ops/s (`1.23s` CPU) | **353k – 637k ops/s (`0.89s` CPU)** | **`1.00× – 1.51×`** |
| **`c=16, P=16` (`SET`/`INCR`/`LPUSH`/`RPUSH`/`LPOP`/`SADD`/`HSET`/`ZADD`/`MSET`)** | 325k – 1.41M ops/s | **962k – 1.56M ops/s** | **`1.09× – 2.96×`** |
| **100k JSON Load (`12 MB` raw, `P=500`)** | `201.3 ms` | **`181.2 ms`** *(+ `86.6 ms` compact)* | **`1.11×`** |
| **100k JSON Disk / Snapshot** | `5,458 KiB` (`dump.rdb`) | **`271.3 KiB`** (`snap.vxg`) | **`20.1×` smaller** |
| **100k JSON Cold Reader Pod RAM (RSS)** | `20.39 MiB` | **`2.91 MiB`** | **`7.0×` less RAM** |
| **100k JSON `GROUP BY` Analytics** | `212.83 ms` (`MGET` + JSON) | **`8.73 ms`** (`VXG.SQL`) | **`24.4×` faster** |
| **Live TCP Streams `XADD` (`P=32`)** | — | **`692,471 ops/s`** | Zero-clone in-place |

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
