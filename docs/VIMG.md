---
title: VIMG — HTTP-Range Wavelet Image Codec
description: Pure-Rust, zero-dependency HTTP/1.1 Range-native hierarchical wavelet image codec with 2D adaptive binary range coding, direct compressed-domain resize/crop/effects, and 144 KiB zero-WASI WebAssembly support.
---

# VIMG (`VIMG1`) — HTTP-Range Wavelet Image Codec

**VIMG** (`crates/vimg`) is a standalone, pure-Rust, **zero-dependency** (`[lib]`) hierarchical wavelet image codec and compressed-domain transform engine designed to VXG engineering standards.

Unlike static monolithic codecs (`JPEG`, `WebP`, `AVIF`) that force CDNs, edge proxies, or browsers to download and decode 100% of an image before cropping, resizing, or filtering, **VIMG organizes tiles by `(pyramid_level, color_plane, tile_id)` on disk** so any `HTTP/1.1 Range` client fetches **strictly the levels, planes, and spatial tiles needed** for the requested transform.

[Launch Live Browser `vimg.wasm` Playground →](/vimg-playground)

---

## Core Architecture

1. **Dual-Mode Wavelet Pyramid (`YCoCg-R` + `CDF 5/3` / `CDF 9/7`)**:
   - **Lossless (`q = 100`)**: 100% bit-exact reversible integer `YCoCg-R 4:4:4` + Le Gall `CDF 5/3` lifting (RGB & RGBA). Beats PNG by **-38.6%** across real 3K+ photography.
   - **Lossy (`q = 1..99`)**: `YCoCg-R 4:2:0` + DC-unity `CDF 9/7` biorthogonal lifting + JPEG 2000-style Laplacian MMSE dead-zone quantization (`q = floor(|c|/Δ)`, `c_hat = sign(q) * (|q| + 0.375) * Δ`) + true 2D synthesis-norm level weighting.
2. **Zero-Header 2D Adaptive Binary Range Coder (`entropy.rs`)**:
   - **0 bytes** of static frequency table headers per `(level, plane, tile)` chunk.
   - **128 online causal 2D contexts** per chunk: `4×4` block-significance skipping (`75–90%` of zero blocks skipped in a single bit), 48 neighborhood significance contexts (`W, N, NW, NE`), 27 sign-correlation contexts, and 8 logarithmic energy buckets.
3. **HTTP/1.1 Range-Native Wire Format (`VIMG1`)**:
   ```text
   [48-byte VIMG1 Header]
   [8-byte/entry Directory: (len: u32, crc32c: u32) × (levels × planes × tiles)]
   [Level 0 (1/8 LL):   Plane Y tiles.. | Plane Co tiles.. | Plane Cg tiles..]
   [Level 1 (1/4):      Plane Y tiles.. | Plane Co tiles.. | Plane Cg tiles..]
   [Level 2 (1/2):      Plane Y tiles.. | Plane Co tiles.. | Plane Cg tiles..]
   [Level 3 (1/1):      Plane Y tiles.. | Plane Co tiles.. | Plane Cg tiles..]
   ```
   - Initial `4 KiB` prefix fetch is cached and reused — early `L0` chunks require **zero** extra requests.
   - Contiguous chunk spans are coalesced via `coalesce_ranges` to minimize HTTP round trips.
4. **Direct Compressed-Domain Operations**:
   - **Pyramid Resize**: Stops fetching/decoding at the coarsest wavelet level `>= target`.
   - **Spatial Crop & Zero-Decode Crop**: Fetches only intersecting tiles; `vimg crop` re-packages tile payloads verbatim (`vimg → vimg`) with **zero IDWT or color conversion**.
   - **Grayscale / Sepia**: Skips `Co`/`Cg` chroma planes completely (`0` chroma bytes transferred).
   - **Subband Blur / Sharpen / Brightness / Contrast**: Modulates wavelet detail gains and `LL` coefficients directly before IDWT.
   - **Per-Chunk Hardware `CRC32C`** (`ARMv8`, `SSE4.2`, software fallback) + `recover_corrupt` tile salvaging.
5. **Portability & Determinism**:
   - Core `[lib]` has **0 external crates**, deterministic `1T == NT` multi-threaded encode, parallel multi-core tile decode.
   - Stripped `wasm32-unknown-unknown` binary: **144 KiB**, zero WASI (`WebAssembly.instantiateStreaming(res, {})`).
   - Panic-safe C99 header at `crates/vimg/include/vimg.h`.

---

## Real 5-Photo 3K+ Unsplash Benchmark (Apple M2 Max, Release)

All codecs encoded from the exact same lossless PNG references across 5 diverse Unsplash photographs (`01_portrait` 3000×3751, `02_architecture` 3200×2133, `03_landscape` 3200×2136, `04_forest` 3200×2133, `05_tokyo_night` 3200×2076):

### Lossless Totals (5 Photos, 100% Bit-Exact)

| Codec | Total Size | vs PNG | Total Enc Time |
|---|---:|---:|---:|
| PNG (lossless) | 53,588 KiB | — | 213 ms |
| **VIMG Lossless (`q=100`)** | **32,919 KiB** | **-38.6%** | **305 ms** |
| WebP Lossless (`-z 6`) | 31,299 KiB | -41.6% | 18,306 ms *(60× slower)* |

### Lossy Full-File Comparison (5 Photos)

| Photo | JPEG `q=75` | **VIMG `q=80`** | **VIMG `q=85`** | WebP `q=75` | AVIF `q=62 s=6 MT` |
|---|---:|---:|---:|---:|---:|
| `01_portrait` (11.3 MP) | 897 KiB (`41.5 dB`) | **271.6 KiB** (`39.22 dB`, 25.7 ms) | 469.4 KiB (`41.03 dB`) | 301.1 KiB (`39.11 dB`, 649 ms) | 454.6 KiB (`41.81 dB`, 431 ms) |
| `02_architecture` | 935 KiB (`39.4 dB`) | **432.7 KiB** (`37.88 dB`, 16.1 ms) | 612.7 KiB (`39.90 dB`) | 399.6 KiB (`37.45 dB`, 461 ms) | 464.7 KiB (`39.11 dB`, 364 ms) |
| `03_landscape` | 850 KiB (`38.1 dB`) | **432.8 KiB** (`37.11 dB`, 19.1 ms) | 673.2 KiB (`39.45 dB`) | 447.4 KiB (`36.64 dB`, 456 ms) | 507.1 KiB (`38.28 dB`, 463 ms) |
| `04_forest` | 1,005 KiB (`36.9 dB`) | **596.1 KiB** (`36.74 dB`, 21.2 ms) | 825.6 KiB (`38.98 dB`) | 522.4 KiB (`36.16 dB`, 470 ms) | 613.8 KiB (`37.42 dB`, 364 ms) |
| `05_tokyo_night` | 1,811 KiB (`34.1 dB`) | **1,189.9 KiB** (`32.76 dB`, 23.7 ms) | 1,621.5 KiB (`34.48 dB`) | 1,077.2 KiB (`32.25 dB`, 589 ms) | 996.4 KiB (`33.76 dB`, 550 ms) |
| **TOTAL (5 photos)** | **5,497 KiB** (`38.01 dB`, 351 ms) | **2,923 KiB (-46.8% vs JPEG)** (`36.74 dB`, **106 ms**) | **4,202 KiB (-23.6% vs JPEG)** (`38.77 dB`, **125 ms**) | **2,748 KiB** (`36.32 dB`, 2,624 ms) | **3,037 KiB** (`38.08 dB`, 2,172 ms) |

### Compressed-Domain Remote Queries (`VIMG q=80` vs Classic Full Decode)

| Operation | VIMG Range Latency | VIMG Wire Transfer | Classic JPEG Full + Transform | Speedup / Traffic Saved |
|---|---:|---:|---:|---|
| **1/4 Resize (`~800px`)** | **4.57 – 8.80 ms** | **140.3 – 319.3 KiB** (`26.8–51.6%`) | 43.8 – 68.9 ms (`850–1,811 KiB`) | **5×–12× faster**, up to **6× less traffic** |
| **256×256 Center Crop** | **0.96 – 1.86 ms** | **28.1 – 95.9 KiB** (`4` tiles, `6.5–15.6%`) | 20.1 – 33.6 ms (`850–1,811 KiB`) | **12×–27× faster**, **10×–30× less traffic** |

---

## Quick Start (CLI, Edge Proxy & API)

```bash
# Run core zero-dependency tests
cargo test --manifest-path crates/vimg/Cargo.toml --release

# Build zero-WASI WebAssembly module (144 KiB stripped)
RUSTFLAGS="-C strip=symbols" cargo build --manifest-path crates/vimg/Cargo.toml --release --lib --target wasm32-unknown-unknown

# Build CLI with PNG/JPEG/WebP/BMP I/O
cargo build --manifest-path crates/vimg/Cargo.toml --release --features io --bin vimg

# Encode, inspect, benchmark
vimg encode photo.png photo.vimg -q 80
vimg info photo.vimg
vimg bench photo.png -q 80

# Start HTTP/1.1 Range Origin + Live Compressed-Domain Edge Proxy + Playground UI
vimg serve ./images --port 8090

# Fetch a remote 512x512 crop + subband blur + sepia over HTTP/1.1 Range
vimg decode http://127.0.0.1:8090/photo.vimg out.png --crop 1200,800,512,512 --blur 0.4 --sepia

# Zero-decode tile crop (vimg -> vimg without IDWT)
vimg crop http://127.0.0.1:8090/photo.vimg cropped.vimg --tiles 4,3,2,2
```

### Rust API

```rust
use vimg::{decode_from_source, encode, EncodeOptions, HttpRangeSource, TransformQuery};

// 1. Encode RGB/RGBA pixels
let vimg_bytes = encode(&rgb_pixels, width, height, 3, &EncodeOptions::default())?;

// 2. Remote HTTP/1.1 Range partial decode (fetches only needed levels, planes & tiles)
let remote = HttpRangeSource::connect("http://127.0.0.1:8090/photo.vimg")?;
let img = decode_from_source(&remote, &TransformQuery {
    crop: Some((1200, 800, 512, 512)),
    blur: 0.35,
    sepia: true,
    ..TransformQuery::default()
})?;
println!("Fetched {:.1}% ({} tiles)", img.stats.fetched_pct(), img.tiles_fetched);
```
