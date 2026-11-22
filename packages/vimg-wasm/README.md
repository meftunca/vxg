<p align="center">
  <img src="../../assets/logos/vimg.svg" width="72" height="72" alt="VIMG WASM Logo" />
</p>

# `@vimg/wasm`

Zero-WASI WebAssembly (`+simd128`, ~266 KiB) wrapper for **VIMG** (`VIMG1`) — the HTTP/HTTPS Range-native hierarchical wavelet image codec & compressed-domain transform engine.

## Quick Start

```js
import { VimgCodec } from '@vimg/wasm';

const codec = await VimgCodec.init();

// 1. Remote HTTP/1.1 206 Range partial crop + color grading (fetches only needed tiles!)
const frame = await codec.decodeUrl('https://example.com/photo.vimg', {
  crop: [1200, 800, 512, 512],
  saturation: 1.2,
  temperature: 0.25,
  sharpen: 0.4,
});
ctx.putImageData(new ImageData(frame.rgba, frame.width, frame.height), 0, 0);

// 2. Encode browser ImageData straight to .vimg with Adaptive Quantization (--aq)
const vimgBytes = codec.encodeRgba(imageData.data, imageData.width, imageData.height, {
  quality: 80,
  tileSize: 256,
  adaptiveQuant: true,
});

// 3. Progressive L0 -> L1 -> L2 -> L3 decode
for (const step of codec.progressiveBuffer(vimgBytes)) {
  console.log(`L${step.decodedLevel}: ${step.width}x${step.height} (${step.bytesFetched} B)`);
}
```
