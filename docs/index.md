---
layout: home

hero:
  name: "VXG"
  text: "Compute Directly on Compressed Data"
  tagline: "One core idea — query, serve, crop, and run inference without unpacking first. Pure Rust, zero core dependencies. Install only the standalone module you need."
  image:
    src: /logos/vxg.svg
    alt: VXG Logo
  actions:
    - theme: brand
      text: Try SQL Playground →
      link: /playground
    - theme: brand
      text: Try Image Studio →
      link: /vimg-playground
    - theme: alt
      text: Quick Start
      link: /guide/getting-started

features:
  - icon:
      src: /logos/vxg.svg
    title: "VXG Core · v0.7.0"
    details: "Adaptive lossless data codec + compressed-domain SQL, Arrow/DuckDB bridge, and zero-dep HTTP/1.1 Range queries over 40M+ rows."
    link: /guide/getting-started
    linkText: Explore VXG Core
  - icon:
      src: /logos/vxg-kv.svg
    title: "VXG-KV · v0.3.0"
    details: "Stateless snapshot Redis (RESP2/RESP3) over Object Storage. Beats Redis 8.10.3 11/11 (1.85M ops/s) with 20× smaller snapshots & VXG.SQL."
    link: /KV
    linkText: Explore VXG-KV
  - icon:
      src: /logos/vxt-turbo.svg
    title: "VXT-Turbo · v0.6.0"
    details: "E8 Gosset lattice LLM/VLM/MoE codec beating GGUF Q6_K & Q8_0 at smaller sizes. 963 tok/s CUDA Resident & 298 tok/s Apple Metal."
    link: /TENSOR
    linkText: Explore VXT-Turbo
  - icon:
      src: /logos/vimg.svg
    title: "VIMG · v0.4.0"
    details: "HTTP-Range hierarchical wavelet image & animation codec (8–16-bit HDR). 1.00 ms UI decode, zero-decode crop, and 277 KiB WASM."
    link: /VIMG
    linkText: Explore VIMG
---

## Which Module Do You Need?

Every module ships as an **independent crate and binary** — you never have to pull in the rest of the workspace.

| If you want to… | Use | Install / Try |
| :--- | :--- | :--- |
| Compress JSON/CSV/logs losslessly and run SQL or DuckDB queries on local or `http://` files | **[VXG Core](/guide/getting-started)** | [Live SQL Playground](/playground) · [v0.7.0 Release](https://github.com/meftunca/vxg/releases/tag/v0.7.0) |
| Run a Redis-compatible server with zero Redis cluster, 20× smaller snapshots & SQL analytics | **[VXG-KV](/KV)** | [VXG-KV Docs](/KV) · [v0.3.0 Release](https://github.com/meftunca/vxg/releases/tag/vxg-kv-v0.3.0) |
| Pack GGUF/SafeTensors into smaller E8 models and run fast local Metal/CUDA inference | **[VXT-Turbo](/TENSOR)** | [VXT-Turbo Spec](/TENSOR) |
| Serve HDR images or animations with ~1 ms remote HTTP Range crops, resizes & WASM decoding | **[VIMG](/VIMG)** | [Live Image Studio](/vimg-playground) · [v0.4.0 Release](https://github.com/meftunca/vxg/releases/tag/vimg-v0.4.0) |

---

## Live Benchmarks (Switch by Module & Platform)

<BenchmarkExplorer />

---

## Quick Start in 30 Seconds

::: code-group

```bash [VXG Core (Data & SQL)]
# Compress NDJSON/CSV verbatim (100% byte-exact reversible) & query directly
cargo run --release --bin vxg -- compress trips.ndjson trips.vxg --record-aligned -t auto -f
cargo run --release --bin vxg -- sql trips.vxg \
  "SELECT payment_type, COUNT(*), SUM(total_cents) GROUP BY payment_type ORDER BY SUM(total_cents) DESC"

# Serve & query remotely over HTTP/1.1 Range (fetches only matching blocks!)
cargo run --release --bin vxg -- serve ./data --port 8080
cargo run --release --bin vxg -- sql http://127.0.0.1:8080/trips.vxg "SELECT COUNT(*) WHERE tip_cents >= 1000"
```

```bash [VXG-KV (Stateless Redis)]
# Start stateless Redis server backed by local dir or HTTP CAS origin
cargo run --release -p vxg-kv -- server --port 6379 --dir ./kv-data --wal-shards 4

# Connect with standard redis-cli / ioredis / redis-py / go-redis
redis-cli -p 6379 JSON.SET user:1 '$' '{"plan":"pro","spend":120}'
redis-cli -p 6379 GEOADD cities 28.9784 41.0082 istanbul 32.8597 39.9334 ankara
redis-cli -p 6379 VXG.SQL "SELECT data.plan, COUNT(*), SUM(data.spend) GROUP BY data.plan"
```

```bash [VXT-Turbo (AI LLM)]
# Pack GGUF or SafeTensors into a compressed E8 .vxt container (zero-OOM streaming)
cargo run --release -p vxg-tensor -- pack model.gguf model.vxt --preset q6

# Run local Metal / CUDA generation or OpenAI-compatible SSE server
cargo run --release -p vxg-tensor -- run model.vxt --prompt "Explain compressed-domain computing."
cargo run --release -p vxg-tensor -- serve model.vxt --port 8080
```

```bash [VIMG (Wavelet Imaging)]
# Encode photo or 16-bit HDR with Tile-Adaptive Quantization (--aq)
cargo run --release -p vimg -- encode photo.png photo.vimg -q 80 --aq

# Zero-decode spatial crop (no IDWT!) or start HTTP 206 Range server
cargo run --release -p vimg -- crop photo.vimg tile.vimg --rect 256,256,512,512
cargo run --release -p vimg -- serve ./images --port 8080
```

```rust [Rust Library (vxg)]
use vxg::{DecodeOptions, Options};

fn main() -> vxg::Result<()> {
    let raw = b"{\"service\":\"auth\",\"status\":200,\"ms\":12}\n".repeat(1000);
    let frame = vxg::compress(&raw, &Options::default())?;

    // O(1) random-access single-block decode without unpacking the whole file
    let block0_len = vxg::frame_block_raw_size(&frame, 0)?;
    let mut out = vec![0u8; block0_len];
    vxg::extract_block_into(&frame, 0, &mut out, &DecodeOptions::default())?;
    assert_eq!(&out[..], &raw[..block0_len]);
    Ok(())
}
```

:::
