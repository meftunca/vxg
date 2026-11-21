# VIMG (`VIMG1`) — HTTP-Range-Native Hierarchical Wavelet Image Codec

`vimg` is a standalone, pure-Rust, **zero-dependency** (`[lib]`) image codec and compressed-domain query engine engineered to VXG standards:

- **HTTP/1.1 Range-Native Wire Layout (`(level, plane, tile_id)`)**: Fetch only the prefix directory plus the exact pyramid levels, color planes, and spatial tiles needed for your query.
- **Dual-Mode Wavelet Pyramid (`YCoCg-R` + `CDF 5/3` / `CDF 9/7` + Adaptive Binary Range Coder)**:
  - `q = 100`: **100% bit-exact lossless** (`4:4:4`, integer `CDF 5/3`, RGB & RGBA) — **38.6% smaller than PNG**, within **5.1%** of WebP Lossless while encoding **60× faster**!
  - `q = 1..99`: **DC-unity `CDF 9/7` biorthogonal lifting** (`4:2:0`) + Laplacian MMSE dead-zone quantization + `4×4` adaptive block-significance skipping + zero-header-overhead 128-context 2D adaptive binary range coding.
- **Direct Compressed-Domain Operations**:
  - **Resize** — stop decoding at the coarsest wavelet level `>= target` (skip L2/L3 over the wire).
  - **Spatial Crop** — fetch only intersecting tiles; `vimg crop` performs **zero-decode `vimg -> vimg`** tile cropping without running IDWT or color conversion.
  - **Grayscale / Sepia** — skip `Co`/`Cg` chroma planes completely (`0` chroma bytes fetched).
  - **Subband Blur / Sharpen / Brightness / Contrast** — scale wavelet detail gains and `LL` coefficients directly before IDWT.
  - **Rotate / Flip** + **Per-Chunk Hardware `CRC32C`** (`ARMv8`, `SSE4.2`, software fallback) with `recover_corrupt` tile salvaging.
- **Hardware SIMD Acceleration (`ARMv8 NEON` + `x86_64 AVX2` + `wasm32 simd128`)**:
  - Vectorized `RGB <-> YCoCg-R`, `4:2:0` box downsample & `2×2` bilinear upsample, horizontal & transpose-free row-major vertical `CDF 9/7` + `CDF 5/3` lifting, `FCVTAS` rounding, branchless `CMOV` range coding, L1-hot tile-local color split & merge, per-worker reusable `WaveletScratch` (zero per-tile heap allocs), zero-alloc `CODEC_ALL_ZERO` detail skip, and dynamic `AtomicUsize` work-stealing (**~1.17 GP/s encode** & **~980 MP/s decode** on Apple M2 Max!).
- **Production Security, Network, HDR, Color Management & UI Mode (`v0.3`)**:
  - **Tile-Adaptive Psychovisual Quantization (`--aq` / `CODEC_BLK4_AQ`)**: Smooth skin/sky tiles receive finer quantization (`0.75×–0.875×`) to prevent banding while chaotic high-frequency textures receive contrast-masked coarser steps (`1.19×–1.50×`), with 1-byte per-chunk overhead and full HTTP-Range independence.
  - **Lossless UI / Screenshot Spatial Palette+RLE (`CODEC_SPATIAL_PALETTE` + 2D Scanline Repeat)**: Automatic `<= 32`-symbol spatial palette + horizontal RLE + 2D previous-row copy on lossless UI/diagram/solid-alpha tiles — decodes 1080p UI screenshots in **`1.00 ms` (zero IDWT, faster than PNG `1.38 ms`)** at **`0.20 bpp` (`118×` compression)** while preserving `L0` thumbnails and tile random access!
  - **CICP HDR Signaling + EXIF / XMP Trailing TLV Metadata (`FLAG_HAS_META`)**: ITU-T H.273 CICP (`sRGB`, `Display-P3`, `BT.2020/PQ`, `BT.2020/HLG`) + raw EXIF & XMP packets stored in a `CRC32C`-protected trailing TLV block (skipped on remote partial queries unless requested, preserved verbatim by zero-decode `vimg crop`, exposed via `vimg_extract_cicp`, `vimg_extract_exif_alloc`, `vimg_extract_xmp_alloc`).
  - **Direct `https://` TLS 1.3 Range Streaming (`--features tls` / `io`)**: Pure-Rust `rustls` + `webpki-roots` (zero OpenSSL) with `http:// -> https://` redirects, `Connection: keep-alive` pooling, parallel 4-stream multi-range fetch, IPv4-first multi-address DNS fallback, and 1-RTT adaptive `16 KiB` prefix (default `[lib]` stays **0 external dependencies**).
  - **10-bit & 12-bit HDR + Embedded ICC Color Profiles**: Bit-exact lossless `CDF 5/3` and lossy `CDF 9/7` HDR (`encode_u16` / `--bit-depth 10|12`) + `CRC32C`-protected Display P3 / AdobeRGB / sRGB ICC profiles (`vimg_extract_icc_alloc`).
  - **Hardened Bounds, Server & Fuzzing**: `MAX_IMAGE_DIM`, `MAX_IMAGE_PIXELS`, `MAX_DIR_BYTES`, `MAX_ICC_BYTES`, `MAX_META_BYTES`, `MAX_CHUNK_BYTES`, 12,000-case fuzz suite (`tests/vimg_fuzz.rs`), `cargo-fuzz` / `libFuzzer` harness (`crates/vimg/fuzz/`), cross-arch golden vectors (`tests/golden_vimg.rs`), and hardened `vimg serve` (`pread` `206`, `ETag/304`, 64-worker `503` protection, EXIF orientation).
- **Zero-Dependency Core + C99 ABI + Zero-WASI `wasm32`**:
  - Core `[lib]` has **0 external crates**.
  - Deterministic `1T == NT` bit-exact encoding (IEEE-754 identical across `aarch64`, `x86_64 AVX2`, and `wasm32`) and parallel multi-core tile decoding.
  - `wasm32-unknown-unknown` (`+simd128`) release build: **263 KiB** stripped `.wasm`, zero WASI (includes HTTP-Range planner & sparse streaming decoder).
  - Panic-safe C99 header at [`include/vimg.h`](include/vimg.h) (`VimgQueryC`, `vimg_decode_query_v2_alloc`, `vimg_extract_icc_alloc`, `vimg_extract_cicp`, `vimg_extract_exif_alloc`, `vimg_extract_xmp_alloc`, and `vimg_wasm_*` exports, CI-verified with `cc -std=c99 -Wall -Wextra -Werror`).

---

## Real 5-Photo 3K+ Unsplash Benchmark (Apple M2 Max, Release)

5 diverse Unsplash photographs (`01_portrait` 3000×3751, `02_architecture` 3200×2133, `03_landscape` 3200×2136, `04_forest` 3200×2133, `05_tokyo_night` 3200×2076), all encoded from the exact same lossless PNG reference:

### Lossless Totals (5 photos, 100% bit-exact)
| Codec | Total Size | vs PNG | Total Enc Time |
|---|---:|---:|---:|
| PNG | 53,588 KiB | — | 213 ms |
| **VIMG Lossless (`q=100`, 2D Adaptive AC)** | **32,919 KiB** | **-38.6%** | **305 ms** |
| WebP Lossless (`-z 6`) | 31,299 KiB | -41.6% | 18,306 ms *(60× slower)* |

### Lossy Totals (5 photos, full file)
| Codec | Total Size | Avg PSNR | Total Enc Time | vs JPEG Size |
|---|---:|---:|---:|---:|
| JPEG (`q=75`) | 5,497 KiB | 38.01 dB | 351 ms | — |
| **VIMG (`q=80`, `CDF 9/7` + 2D Adaptive AC)** | **2,923 KiB** | **36.74 dB** *(+0.42 dB vs WebP)* | **106 ms** *(25× faster than WebP)* | **-46.8%** |
| **VIMG (`q=85`, `CDF 9/7` + 2D Adaptive AC)** | **4,202 KiB** | **38.77 dB** *(+0.76 dB vs JPEG)* | **125 ms** | **-23.6%** |
| WebP (`q=75`) | 2,748 KiB | 36.32 dB | 2,624 ms | -50.0% |
| AVIF (`q=62, s=6, MT`) | 3,037 KiB | 38.08 dB | 2,172 ms | -44.8% |

- Highlights:
  - `01_portrait` (11.3 MP): **VIMG `q=80` (`271.6 KiB @ 39.22 dB`, `25.7 ms`) beats WebP `q=75` (`301.1 KiB @ 39.11 dB`, `649 ms`) by -9.8% size, +0.11 dB PSNR, and 25× speed!**
  - `03_landscape` (6.8 MP): **VIMG `q=80` (`432.8 KiB @ 37.11 dB`, `19.1 ms`) beats WebP `q=75` (`447.4 KiB @ 36.64 dB`, `456 ms`) by -3.3% size, +0.47 dB PSNR, and 24× speed!**

### SIMD-Accelerated Decode & Compressed-Domain Queries (`VIMG q=80` & `q=100`, Apple M2 Max Release)

| Photo | Codec / Query | Size / Wire | Enc ms | **Dec / Query ms** | vs Pre-SIMD / vs JPEG & PNG |
|---|---|---:|---:|---:|---|
| **`01_portrait`** (`3000×3751`, 11.3 MP) | **VIMG Lossy (`q=80`, `39.22 dB`)** | **271.6 KiB** | **16.86 ms** | **13.07 ms** | **4.71× faster** (`61.57 → 13.07 ms`), **2.22× faster dec / 5.4× faster enc than JPEG** (`29.01 / 91.32 ms`) |
| | **VIMG Lossless (`q=100`, bit-exact)** | **6,578.8 KiB** | **55.62 ms** | **51.46 ms** | **-48.0% smaller & 1.57× faster dec than PNG** (`80.79 ms`) |
| | **1/4 Resize (`750×938`) / Center Crop (`256²`)** | **140.3 / 32.1 KiB** | — | **2.20 / 0.63 ms** | **32× / 42× faster** than JPEG full+transform (`70.80 / 26.40 ms`) |
| **`03_landscape`** (`3200×2136`, 6.8 MP) | **VIMG Lossy (`q=80`, `37.11 dB`)** | **432.8 KiB** | **14.05 ms** | **9.30 ms** | **4.81× faster** (`44.77 → 9.30 ms`), **2.30× faster dec / 4.5× faster enc than JPEG** (`21.38 / 62.99 ms`) |
| | **VIMG Lossless (`q=100`, bit-exact)** | **5,043.1 KiB** | **35.46 ms** | **33.08 ms** | **-47.3% smaller & 1.40× faster dec than PNG** (`46.40 ms`) |
| | **1/4 Resize (`800×534`) / Center Crop (`256²`)** | **152.5 / 28.1 KiB** | — | **1.63 / 0.77 ms** | **28× / 25× faster** than JPEG full+transform (`45.05 / 19.60 ms`) |
| **`05_tokyo_night`** (`3200×2076`, 6.6 MP) | **VIMG Lossy (`q=80`, `32.76 dB`)** | **1,189.9 KiB** | **16.57 ms** | **16.13 ms** | **3.13× faster** (`50.46 → 16.13 ms`), **2.18× faster dec / 4.9× faster enc than JPEG** (`35.09 / 80.47 ms`) |
| | **VIMG Lossless (`q=100`, bit-exact)** | **9,871.7 KiB** | **53.24 ms** | **50.61 ms** | **-18.7% smaller & 1.11× faster dec than PNG** (`56.33 ms`) |
| | **1/4 Resize (`800×519`) / Center Crop (`256²`)** | **319.3 / 95.9 KiB** | — | **2.38 / 1.80 ms** | **24× / 19× faster** than JPEG full+transform (`57.71 / 33.70 ms`) |

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
