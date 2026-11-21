import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { VimgCodec } from './index.mjs';

const wasmPath = new URL('../../docs/public/vimg.wasm', import.meta.url);
const wasmBytes = await fs.readFile(wasmPath);
const codec = await VimgCodec.init(wasmBytes);

// Synthetic 512x384 RGBA textured photo test image (>4 KiB compressed so L0 < full file)
const W = 512;
const H = 384;
const rgba = new Uint8ClampedArray(W * H * 4);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    const h = (((x * 73) ^ (y * 137)) * 2654435761) >>> 24;
    rgba[o] = ((x * 200) / W + (h & 47)) & 0xff;
    rgba[o + 1] = ((y * 200) / H + ((h >> 2) & 47)) & 0xff;
    rgba[o + 2] = (((x ^ y) & 0x7f) + ((h >> 1) & 63)) & 0xff;
    rgba[o + 3] = 255;
  }
}

// 1. Lossless encode + bit-exact roundtrip
const llBytes = codec.encodeRgba(rgba, W, H, { quality: 100, tileSize: 128 });
const hdr = codec.parseHeader(llBytes);
assert.equal(hdr.width, W);
assert.equal(hdr.height, H);
const decLL = codec.decodeBuffer(llBytes);
assert.deepEqual(decLL.rgba, rgba);

// 2. Lossy + Adaptive Quant (--aq) + Progressive L0..L3 + Color DSP
const qBytes = codec.encodeRgba(rgba, W, H, { quality: 80, tileSize: 128, adaptiveQuant: true });
const levels = [...codec.progressiveBuffer(qBytes, { saturation: 1.25, temperature: 0.2 })];
assert.equal(levels.length, hdr.dwtLevels + 1);
assert.equal(levels[0].decodedLevel, 0);
assert.equal(levels.at(-1).width, W);
assert.equal(levels.at(-1).height, H);
assert.ok(levels[0].bytesFetched < qBytes.length, 'L0 must fetch less than full archive');

// 3. Grayscale + bicubic resize
const thumb = codec.decodeBuffer(qBytes, { targetWidth: 111, grayscale: true });
assert.equal(thumb.width, 111);
assert.equal(thumb.rgba[0], thumb.rgba[1]);
assert.equal(thumb.rgba[1], thumb.rgba[2]);

console.log(`@vimg/wasm Node test OK (lossless=${llBytes.length}B, q80_aq=${qBytes.length}B, L0=${levels[0].width}x${levels[0].height})`);
