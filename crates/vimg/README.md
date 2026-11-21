# VIMG (`VIMG1`) — HTTP-Range-Native Hierarchical Wavelet Image Codec

`vimg` is a standalone, pure-Rust, **zero-dependency** (`[lib]`) image codec and compressed-domain query engine engineered to VXG standards:

- **HTTP/1.1 Range-Native Wire Layout (`(level, plane, tile_id)`)**: Fetch only the prefix directory plus the exact pyramid levels, color planes, and spatial tiles needed for your query.
- **Dual-Mode Wavelet Pyramid (`YCoCg-R` + `CDF 5/3` / `CDF 9/7`)**:
  - `q = 100`: **100% bit-exact lossless** (`4:4:4`, integer `CDF 5/3`, RGB & RGBA) — **33.4% smaller than PNG**, faster encode & decode than PNG.
  - `q = 1..99`: **DC-unity `CDF 9/7` biorthogonal lifting** (`4:2:0`) + Laplacian MMSE dead-zone quantization + 4×4 oriented micro-block scanning + 32-symbol Hybrid-Golomb FSE.
- **Direct Compressed-Domain Operations**:
  - **Resize** — stop decoding at the coarsest wavelet level `>= target` (skip L2/L3 over the wire).
  - **Spatial Crop** — fetch only intersecting tiles; `vimg crop` performs **zero-decode `vimg -> vimg`** tile cropping without running IDWT or color conversion.
  - **Grayscale / Sepia** — skip `Co`/`Cg` chroma planes completely (`0` chroma bytes fetched).
  - **Subband Blur / Sharpen / Brightness / Contrast** — scale wavelet detail gains and `LL` coefficients directly before IDWT.
  - **Rotate / Flip** + **Per-Chunk Hardware `CRC32C`** (`ARMv8`, `SSE4.2`, software fallback) with `recover_corrupt` tile salvaging.
- **Zero-Dependency Core + C ABI + Zero-WASI `wasm32`**:
  - Core `[lib]` has **0 external crates**.
  - Deterministic `1T == NT` bit-exact encoding and parallel multi-core tile decoding.
  - `wasm32-unknown-unknown` release build: **163 KiB** `.wasm`, zero WASI (includes HTTP-Range planner & sparse streaming decoder).
  - Panic-safe C99 header at [`include/vimg.h`](include/vimg.h).

---

## Real 5-Photo 3K+ Unsplash Benchmark (Apple M2 Max, Release)

5 diverse Unsplash photographs (`01_portrait` 3000×3751, `02_architecture` 3200×2133, `03_landscape` 3200×2136, `04_forest` 3200×2133, `05_tokyo_night` 3200×2076), all encoded from the exact same lossless PNG reference:

### Lossless Totals (5 photos, 100% bit-exact)
| Codec | Total Size | vs PNG | Total Enc Time |
|---|---:|---:|---:|
| PNG | 53,588 KiB | — | 212.5 ms |
| **VIMG Lossless (`q=100`)** | **35,668 KiB** | **-33.4%** | **214.5 ms** |
| WebP Lossless (`-z 6`) | 31,299 KiB | -41.6% | 18,306 ms *(85× slower)* |

### Lossy Totals (5 photos, full file)
| Codec | Total Size | Avg PSNR | Total Enc Time | vs JPEG Size |
|---|---:|---:|---:|---:|
| JPEG (`q=75`) | 5,497 KiB | 38.01 dB | 366 ms | — |
| **VIMG (`q=80`, `CDF 9/7`)** | **3,345 KiB** | **36.85 dB** | **100 ms** *(26× faster than WebP)* | **-39.1%** |
| **VIMG (`q=85`, `CDF 9/7`)** | **4,811 KiB** | **38.93 dB** *(+0.92 dB vs JPEG)* | **105 ms** | **-12.5%** |
| WebP (`q=75`) | 2,748 KiB | 36.32 dB | 2,624 ms | -50.0% |
| AVIF (`q=62, s=6, MT`) | 3,037 KiB | 38.08 dB | 2,172 ms | -44.8% |

### Compressed-Domain Remote Queries (`VIMG q=80`, 5-photo range)
- **1/4 Resize (`~800px`)**: **3.09 – 7.28 ms**, **166 – 337 KiB (`24.8% – 52.5%`)** vs JPEG full+resize **44.2 – 69.1 ms (`850 – 1,811 KiB`)** (**7× – 22× faster!**)
- **256×256 Center Crop**: **0.94 – 1.33 ms**, **30.6 – 105.4 KiB (`6.2% – 15.7%`, `4` tiles)** vs JPEG full+crop **19.6 – 33.4 ms** (**16× – 28× faster, 9× – 28× less traffic!**)

---

## Real Benchmarks (Apple M2 Max, Release Build)

### 4K UHD Photograph (`3840×2160` RGB, 24.3 MiB raw — macOS Sonoma aerial crop)

| Codec / Operation | Size / Transferred | bpp | PSNR (dB) | Enc (ms) | Dec / Query (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| PNG (lossless) | 9,431.4 KiB | 9.31 | ∞ | 44.78 | 38.68 | 100% full stream |
| **VIMG Lossless (`q=100`)** | **6,835.0 KiB** | **6.75** | **∞** | **48.39** | **57.67** | **27.5% smaller than PNG**, bit-exact |
| JPEG (`q=75`) | 270.7 KiB | 0.27 | 44.06 | 57.63 | 13.86 | 100% full decode required |
| **VIMG Lossy (`q=85`)** | **226.3 KiB** | **0.22** | **43.19** | **14.59** | **39.67** | **16.4% smaller & 3.95× faster encode than JPEG** |
| **1/4 Resize (`960×540`)** | **140.9 KiB (62.3%)** | — | — | — | **2.59 ms** | **15.5× faster** than JPEG full+resize (`40.19 ms`) |
| **256×256 Center Crop** | **26.1 KiB (11.5%)** | — | — | — | **0.64 ms** | **17.5× faster**, **10.4× less traffic** (`2/135` tiles) |
| **1/2 Gray + Blur (`1920×1080`)** | **152.8 KiB (67.5%)** | — | — | — | **2.36 ms** | `0` chroma bytes fetched |

### 6K Photograph (`6016×6016` RGB, 36.2 MP / 106 MiB raw — macOS Sonoma full)

| Codec / Operation | Size / Transferred | bpp | PSNR (dB) | Enc (ms) | Dec / Query (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| PNG (lossless) | 40,927.4 KiB | 9.26 | ∞ | 193.41 | 165.49 | 100% full stream |
| **VIMG Lossless (`q=100`)** | **29,334.5 KiB** | **6.64** | **∞** | **182.65** | **237.87** | **28.3% smaller than PNG** |
| JPEG (`q=75`) | 1,144.8 KiB | 0.26 | 44.38 | 245.30 | 55.34 | 100% full decode required |
| **VIMG Lossy (`q=85`)** | **880.5 KiB** | **0.20** | **43.30** | **55.79** | **172.36** | **23.1% smaller & 4.4× faster encode than JPEG** |
| **1/4 Resize (`1504×1504`)** | **574.5 KiB (65.2%)** | — | — | — | **12.09 ms** | **14.0× faster** than JPEG full+resize (`168.96 ms`) |
| **256×256 Center Crop** | **65.8 KiB (7.5%)** | — | — | — | **0.80 ms** | **57.4× faster**, **17.4× less traffic** (`4/576` tiles) |

### Live HTTP/1.1 Range Origin + Edge Proxy (`6016×6016`, 880.5 KiB `.vimg`)
- Remote `512×512` crop + subband blur + sepia: fetched **68.5 KiB (`7.8%`, `9/576` tiles)** in **8.62 ms**.
- Remote zero-decode `vimg -> vimg` crop (`2×2` tiles): fetched **10.4%** in **4.89 ms**.
- Edge proxy `GET /sonoma_6k.vimg?w=752&gray=1&blur=0.5`: `X-VIMG-Fetched-Pct: 26.8`, `X-VIMG-Level: 0`.

---

## Build & Usage

```bash
# Core library tests (zero external dependencies)
cargo test --manifest-path crates/vimg/Cargo.toml --release

# Zero-WASI WebAssembly build (143 KiB .wasm)
cargo build --manifest-path crates/vimg/Cargo.toml --release --lib --target wasm32-unknown-unknown

# Build CLI with PNG/JPEG/WebP/BMP I/O
cargo build --manifest-path crates/vimg/Cargo.toml --release --features io --bin vimg

# Encode, inspect, benchmark
vimg encode photo.png photo.vimg -q 85
vimg info photo.vimg
vimg bench photo.png

# Start HTTP/1.1 Range Origin + Live Compressed-Domain Edge Proxy + Interactive Web Playground
vimg serve ./images --port 8090
# Open http://127.0.0.1:8090/ in your browser for interactive drag-to-crop,
# tile-grid heatmap, live X-VIMG-* wire telemetry, and Browser vimg.wasm HTTP 206 Range mode!

# Fetch remote crop + effects directly via HTTP Range
vimg decode http://127.0.0.1:8090/photo.vimg out.png --crop 2500,2500,512,512 --blur 0.4 --sepia

# Or call the Edge Proxy directly from a browser / curl
curl "http://127.0.0.1:8090/photo.vimg?w=752&gray=1&blur=0.5" -o thumb.png
```

---

## Honest Engineering Trade-Offs

- **Where `vimg` shines**: Remote HTTP Range CDN/edge pipelines, high-res deep-zoom/crop/thumbnail workloads, fast parallel encoding, lossless archiving (beats PNG), and sub-millisecond compressed-domain crops/effects.
- **Where `AVIF` / `JPEG XL` / `WebP` still win**: Full-image static downloads at extreme low bitrates (`< 0.15 bpp`) where intra-prediction + heavy RDO beat a fast non-RDO `CDF 5/3` lifting wavelet, and native browser `<img>` tag support without a JS/WASM decoder or Edge Proxy.
