---
title: VXG-KV — Stateless Snapshot Redis
description: Zero-dependency Redis-compatible (RESP2/RESP3) stateless snapshot KV, JSON/Geo, Streams & Consumer Groups, Pub/Sub & compressed-domain SQL over VXG + Object Storage
---

<p align="center">
  <img src="/logos/vxg-kv.svg" width="88" height="88" alt="VXG-KV Logo" />
</p>

# `vxg-kv` `v0.3.0` — Stateless Snapshot Redis over VXG

**Zero external dependencies. Zero dedicated Redis cluster. Object-storage source of truth for thousands of stateless pods.**

> **Try it right now in your browser (no install!):** Open the **[Live VXG-KV WASM Playground](/kv-playground)** — two pods backed by real **IndexedDB / localStorage** compressed VXG snapshots!

`vxg-kv` speaks standard **Redis RESP2 & RESP3 (`HELLO 3`)** (`redis-cli`, `ioredis`, `redis-py`, `go-redis`) while persisting state as immutable, key-sorted **`VXG v5`** snapshots + atomic ETag CAS delta WAL segments over any `ObjectStore` (`HttpObjectStore`, `LocalFsStore`, or your own 6-method S3/R2/GCS adapter).

## Key Advantages

- **Zero Dedicated Redis Cluster:** Pods keep only their hot working set (`--cache-mb 64`) in RAM. Cold keys fetch strictly the 1 matching VXG block over **HTTP/1.1 `Range`**, missing keys are rejected locally by per-block **10-bit/key Bloom filters (`0` Range I/O)**, `DBSIZE` runs in **$O(1)$ (`0` Range bytes)**, and `SCAN`/`KEYS` use **`SnapshotIndex` fence-pointer pruning + early-stop streaming**.
- **20×–40× Smaller State:** Sorted keys + JSON payloads compress via `VXG v5`'s `FrontCodedDict` + `BLOCK_COLUMN` (`12 MB -> 271 KiB` on 100k JSON records vs Redis `dump.rdb` `5,458 KiB`).
- **Horizontally Sharded Multi-Pod Writes (`--wal-shards N`):** Atomic `ETag` Conditional PUT (`If-Match` / `If-None-Match: *`) with automatic `412` pull + rebase + retry, optional `--wal-shards N` partitioned WAL manifests (`shards/s00.json..sNN.json`), plus `VXG.CAS` for linearizable distributed locks.
- **Built-in Redis Extensions & Superpowers:**
  - **Security & RESP3:** `--requirepass` / `VXG_KV_PASSWORD`, `AUTH`, `ACL WHOAMI|LIST`, `HELLO 2|3 [AUTH user pass]`
  - **Native RedisJSON & Geospatial:** `JSON.SET/GET/DEL/NUMINCRBY` (directly queryable by `VXG.SQL`!) + `GEOADD/GEOPOS/GEODIST/GEOSEARCH`
  - **Non-Blocking Queues & Consumer Groups:** Zero-thread reactor `BLPOP`/`BRPOP` + cross-pod persisted `XGROUP`/`XREADGROUP`/`XACK`/`XPENDING`
  - **Time-Travel & Compressed-Domain SQL:** `VXG.SNAPSHOTS`, `VXG.AT <epoch> GET <key>`, `VXG.PIN <epoch>`, `VXG.SQL "SELECT ... GROUP BY ..."` (`26.8×` faster than Redis `MGET`+JSON)

## Head-to-Head vs `Redis 8.10.3` (`redis-benchmark`, Same Host)

| Benchmark (`redis-benchmark`) | `Redis 8.10.3` | `vxg-kv v0.3.0` | Speedup / Savings |
| :--- | ---: | ---: | :---: |
| **`c=1, P=16` (11 / 11 commands)** | 220k – 592k ops/s (`1.29s` CPU) | **629k – 725k ops/s (`0.58s` CPU)** | **`1.15× – 2.86×`** (`2.22×` less CPU!) |
| **`c=16, P=16` — `PING` / `GET` / `SET` / `INCR`** | 1.27M – 1.56M ops/s | **1.67M – 1.85M ops/s** | **`1.10× – 1.34×`** |
| **`c=16, P=16` — `LPUSH` / `RPUSH` / `LPOP`** | 1.19M – 1.35M ops/s | **1.75M – 1.85M ops/s** | **`1.35× – 1.54×`** |
| **`c=16, P=16` — `SADD` / `HSET` / `ZADD`** | 1.19M – 1.52M ops/s | **1.69M – 1.85M ops/s** | **`1.14× – 1.56×`** |
| **`c=16, P=16` — `MSET (10 keys)`** | 310,559 ops/s | **1,666,667 ops/s** | **`5.37×`** |
| **`c=16, P=16` — Total CPU Time (11 commands)** | `1.06s` | **`0.55s`** | **`1.93×` less CPU!** |
| **100k JSON Disk / Snapshot** | `5,458 KiB` (`dump.rdb`) | **`271.3 KiB`** (`snap.vxg`) | **`20.1×` smaller** |
| **100k JSON Cold Reader Pod RAM (RSS)** | `20.12 MiB` | **`2.86 MiB`** | **`7.03×` less RAM** |
| **100k JSON `GROUP BY` Analytics** | `210.99 ms` (`MGET` + JSON) | **`7.88 ms`** (`VXG.SQL`) | **`26.8×` faster** |

## Quickstart

```bash
# 1. Start the built-in zero-dep HTTP/1.1 CAS + Range Object Store origin
cargo run --release -p vxg-kv -- origin ./kv-origin --port 9000

# 2. Start Pod A & Pod B pointing to the same origin (with 4 WAL shards + optional password)
cargo run --release -p vxg-kv -- server --port 6379 --origin http://127.0.0.1:9000 --pod-id pod-a --wal-shards 4
cargo run --release -p vxg-kv -- server --port 6380 --origin http://127.0.0.1:9000 --pod-id pod-b --wal-shards 4

# 3. Talk to either pod with standard redis-cli
redis-cli -p 6379 JSON.SET user:1 '$' '{"plan":"pro","spend":120}'
redis-cli -p 6379 VXG.COMMIT
redis-cli -p 6380 JSON.GET user:1 '$.plan'
redis-cli -p 6380 VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) WHERE _t = 'json' GROUP BY data.plan"
```

Prebuilt binaries for Linux (`x86_64` / `arm64`), macOS (`arm64`), and Windows (`x86_64`) are published on [GitHub Releases (`vxg-kv-v0.3.0`)](https://github.com/meftunca/vxg/releases/tag/vxg-kv-v0.3.0).
