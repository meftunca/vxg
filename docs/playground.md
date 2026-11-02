---
title: Live WebAssembly Playground
description: Interactive in-browser VXG v4 encode, decode, O(1) random block access, and compressed-domain SQL & Top-K query runner powered by zero-dependency vxg.wasm.
aside: false
---

# Live WebAssembly (`vxg.wasm`) Studio

Run the exact Rust `VXG1` v4 codec and compressed-domain analytics engine directly inside your browser via `wasm32-unknown-unknown` — **100% client-side, zero WASI/host imports, zero server uploads**.

<WasmPlayground />

---

## How Browser Execution Works (`0` Dependencies)

Because VXG has **zero external runtime dependencies**, compiling the library to `wasm32-unknown-unknown` produces a self-contained `vxg.wasm` binary requiring **`0` host or WASI imports** (`WebAssembly.instantiate(bytes, {})`):

```js
const { instance } = await WebAssembly.instantiateStreaming(fetch('/vxg/vxg.wasm'), {})
const vxg = instance.exports

// 1. Allocate input & output slices directly in WASM linear memory
const bound = vxg.vxg_compress_bound(raw.byteLength, 65536)
const srcPtr = vxg.vxg_alloc(raw.byteLength) >>> 0
const dstPtr = vxg.vxg_alloc(bound) >>> 0

new Uint8Array(vxg.memory.buffer, srcPtr, raw.byteLength).set(raw)
const written = vxg.vxg_compress_into(srcPtr, raw.byteLength, dstPtr, bound, 1, 65536, 1)
const compressed = new Uint8Array(vxg.memory.buffer, dstPtr, written).slice()

vxg.vxg_free(srcPtr, raw.byteLength)
vxg.vxg_free(dstPtr, bound)
```

### Supported In-Browser C ABI Calls
- **Encode / Decode / Verify**: `vxg_compress_into`, `vxg_decompress_into`, `vxg_verify_frame`
- **$O(1)$ Random Block Extraction**: `vxg_frame_block_count`, `vxg_frame_block_raw_size`, `vxg_extract_block_into`
- **Compressed-Domain SQL, `Top-K` & Splicing**: `vxg_count_where_str`, `vxg_select_alloc`, `vxg_aggregate_col`, `vxg_project_frame_alloc`
