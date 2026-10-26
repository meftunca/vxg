# Cross-Language FFI & Zero-Overhead Integration Guide (`libvxg`)

VXG builds simultaneously as:
- **Rust library (`rlib`):** `vxg` crate
- **Static C ABI archive (`staticlib`):** `libvxg.a` (Linux / macOS) or `vxg.lib` (Windows MSVC)
- **Dynamic shared library (`cdylib`):** `libvxg.so` (Linux), `libvxg.dylib` (macOS), or `vxg.dll` (Windows)

All foreign-language bindings share the panic-safe C ABI declared in [`include/vxg.h`](../include/vxg.h) and implemented in [`src/ffi.rs`](../src/ffi.rs).

---

## 1. Building Static and Dynamic Libraries

```sh
cargo build --release --lib
```

Artifacts produced in `target/release/`:

| Platform | Static Archive (`staticlib`) | Dynamic Shared Library (`cdylib`) | Header |
| :--- | :--- | :--- | :--- |
| **Linux** | `libvxg.a` | `libvxg.so` | `include/vxg.h` |
| **macOS** | `libvxg.a` | `libvxg.dylib` | `include/vxg.h` |
| **Windows** | `vxg.lib` | `vxg.dll` + `vxg.dll.lib` | `include/vxg.h` |

---

## 2. Zero-Allocation Caller-Buffer Design (Rust & C ABI)

To achieve multi-GB/s throughput across language boundaries without heap-allocation or serialization overhead, VXG provides matching caller-buffer (`_into`) APIs in both Rust ([`src/lib.rs`](../src/lib.rs)) and C ([`include/vxg.h`](../include/vxg.h)):

1. **Compression (`vxg::compress_into` / `vxg_compress_into`):**
   - Query the upper bound once with `vxg::compress_bound(raw_len, block_size)` / `vxg_compress_bound(raw_len, block_size)`.
   - Pass a caller-owned buffer (or reusable arena/pool slice) directly to `compress_into` / `vxg_compress_into(...)`.
2. **Decompression (`vxg::decompress_into` / `vxg_decompress_into` — Zero Per-Block Heap Allocation):**
   - Inspect the 32-byte frame header in $O(1)$ time with `vxg::frame_raw_size(src)` / `vxg_frame_raw_size(src, src_len)`.
   - Pass a caller-allocated buffer (e.g. `&mut [u8]`, `uint8_t*`, Python `bytearray`, Go `[]byte`, JS `Uint8Array`, .NET `Span<byte>`) directly to `decompress_into` / `vxg_decompress_into(...)`. Both single-thread and multi-thread (`threads > 1`) paths decode directly into disjoint slices of the destination buffer.
3. **$O(1)$ Random-Access Single-Block Extraction (`vxg::extract_block_into` / `vxg_extract_block_into`):**
   - Query the uncompressed size of block `i` in $O(1)$ time with `vxg::frame_block_raw_size(src, i)` / `vxg_frame_block_raw_size(src, src_len, i)`.
   - Decode only block `i` directly into your destination buffer with `extract_block_into` / `vxg_extract_block_into(...)`.

---

## 3. C ABI Reference (`include/vxg.h`)

| Function | Signature | Description |
| :--- | :--- | :--- |
| `vxg_version_string` | `const char* vxg_version_string(void)` | Returns null-terminated version string (`"0.4.0"`). |
| `vxg_error_message` | `const char* vxg_error_message(vxg_ssize_t code)` | Returns static human-readable message for `< 0` error codes. |
| `vxg_compress_bound` | `vxg_ssize_t vxg_compress_bound(size_t raw_len, size_t block_size)` | Worst-case compressed byte size (`block_size = 0` uses default `65536`). |
| `vxg_frame_raw_size` | `vxg_ssize_t vxg_frame_raw_size(const uint8_t* src, size_t src_len)` | $O(1)$ header inspection returning exact uncompressed size in bytes. |
| `vxg_frame_block_count` | `vxg_ssize_t vxg_frame_block_count(const uint8_t* src, size_t src_len)` | $O(1)$ header inspection returning number of independent blocks. |
| `vxg_frame_block_raw_size` | `vxg_ssize_t vxg_frame_block_raw_size(const uint8_t* src, size_t src_len, size_t idx)` | $O(1)$ header inspection returning uncompressed byte size of block `idx`. |
| `vxg_compress_into` | `vxg_ssize_t vxg_compress_into(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap, size_t threads, size_t block_size, int32_t checksum)` | Compresses `src` into `dst`. `threads = 0` uses all CPU cores. Returns bytes written (`>= 0`) or `VXG_ERR_*` (`< 0`). |
| `vxg_decompress_into` | `vxg_ssize_t vxg_decompress_into(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap, size_t threads, int32_t verify_checksum)` | Zero-allocation decode of `src` directly into `dst`. Returns bytes written (`>= 0`) or `VXG_ERR_*` (`< 0`). |
| `vxg_extract_block_into` | `vxg_ssize_t vxg_extract_block_into(const uint8_t* src, size_t src_len, size_t idx, uint8_t* dst, size_t dst_cap, int32_t verify_checksum)` | Zero-allocation $O(1)$ random-access decode of block `idx` into `dst`. |
| `vxg_compress_alloc` / `vxg_decompress_alloc` / `vxg_free` | `uint8_t* ...` | Convenience heap-allocated wrappers when caller-side buffer management is not desired. |

---

## 4. Ready-to-Run Language Examples

All runnable examples are located in [`examples/ffi/`](../examples/ffi/):

### C / C++ ([`examples/ffi/c_example.c`](../examples/ffi/c_example.c))
```sh
cargo build --release --lib
cc -O3 -Iinclude examples/ffi/c_example.c target/release/libvxg.a -lpthread -ldl -lm -o c_example
./c_example
```

### Python ([`examples/ffi/python_example.py`](../examples/ffi/python_example.py))
Uses standard-library `ctypes` + `memoryview` / `bytearray` with zero external packages:
```sh
cargo build --release --lib
python3 examples/ffi/python_example.py
```

### Go ([`examples/ffi/go_example.go`](../examples/ffi/go_example.go))
Passes Go slice pointers directly via `unsafe.SliceData` with static linking (`libvxg.a`):
```sh
cargo build --release --lib
CGO_CFLAGS="-I$(pwd)/include" CGO_LDFLAGS="$(pwd)/target/release/libvxg.a -lpthread -ldl -lm" \
  go run examples/ffi/go_example.go
```

### JavaScript / TypeScript / Bun ([`examples/ffi/node_bun_example.mjs`](../examples/ffi/node_bun_example.mjs))
Passes `Uint8Array` memory pointers directly via `bun:ffi` `ptr()`:
```sh
cargo build --release --lib
bun run examples/ffi/node_bun_example.mjs
```

### C# / .NET 8+ ([`examples/ffi/csharp_example.cs`](../examples/ffi/csharp_example.cs))
Pins `ReadOnlySpan<byte>` and `Span<byte>` in place via source-generated `[LibraryImport("vxg")]` P/Invoke with zero managed heap copies.

### WebAssembly (`wasm32-unknown-unknown` — Browser / Edge / Node.js / Bun) ([`examples/ffi/wasm_example.mjs`](../examples/ffi/wasm_example.mjs))
Builds a self-contained `vxg.wasm` binary requiring **zero WASI or host imports** (`WebAssembly.instantiate(wasmBytes, {})`). Uses `vxg_alloc` and `vxg_free` to slice directly into `exports.memory`:
```sh
rustup target add wasm32-unknown-unknown
cargo build --release --lib --target wasm32-unknown-unknown
node examples/ffi/wasm_example.mjs
```

