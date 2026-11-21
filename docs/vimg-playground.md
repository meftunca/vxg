---
title: VIMG Live HTTP-Range WASM Playground
description: Interactive in-browser VIMG1 wavelet image codec playground — real HTTP/1.1 206 Partial Content tile range streaming, client-side PNG/JPEG/WebP encoding, and compressed-domain resize, crop, blur, sharpen & sepia powered by 144 KiB zero-WASI vimg.wasm.
aside: false
---

# Live `vimg.wasm` HTTP-Range Playground

Explore real **3K Unsplash `.vimg` images** streamed directly from GitHub Pages using **HTTP/1.1 `206 Partial Content` byte ranges**, or **drop your own PNG / JPEG / WebP / BMP** to encode and explore it **100% client-side** inside our **144 KiB zero-WASI `vimg.wasm`** module — zero backend server, zero uploads.

::: tip Try Opening DevTools → Network
Drag a crop rectangle on the **Tile Navigator** below and watch the browser send tiny `Range: bytes=...` (`206 Partial Content`) requests for **only the intersecting tiles and pyramid levels** instead of downloading the full image.
:::

<VimgPlayground />

---

## Zero-Dependency Browser Integration (`vimg.wasm`)

`vimg.wasm` (`144 KiB` stripped) requires **`0` WASI or host imports** and exposes both client-side encoding (`vimg_wasm_encode_rgba`) and sparse HTTP-Range decoding (`vimg_wasm_parse_header` → `vimg_wasm_plan` → `vimg_wasm_decode_sparse`):

```js
const { instance } = await WebAssembly.instantiateStreaming(fetch('/vxg/vimg.wasm'), {})
const vimg = instance.exports

// 1. Fetch only the initial 4 KiB prefix via HTTP Range
const prefix = new Uint8Array(
  await (await fetch('/vxg/samples/landscape_3k_q80.vimg', {
    headers: { Range: 'bytes=0-4095' },
  })).arrayBuffer()
)

// 2. Ask vimg.wasm for the exact coalesced byte ranges needed for a 512×512 crop
const pPtr = vimg.vimg_wasm_alloc(prefix.length)
new Uint8Array(vimg.memory.buffer, pPtr, prefix.length).set(prefix)
vimg.vimg_wasm_plan(pPtr, prefix.length, 1200, 800, 512, 512, 0, 0, 0, 0)

// 3. Fetch only those HTTP 206 spans and decode straight into RGBA8 ImageData!
```

See the full architecture, benchmarks, CLI, and C99/Rust API in the [VIMG Documentation →](/VIMG).
