// Zero-bindgen WebAssembly (`vxg.wasm`) integration example for Browser / Node.js / Bun / Edge Workers.
// Operates directly on Wasm linear memory (`exports.memory`) via `vxg_alloc` / `vxg_compress_into` /
// `vxg_decompress_into` with zero external JS packages and zero WASI imports (`wasm32-unknown-unknown`).
//
// Build & run:
//   rustup target add wasm32-unknown-unknown
//   cargo build --release --lib --target wasm32-unknown-unknown
//   node examples/ffi/wasm_example.mjs

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const wasmCandidates = [
  process.env.VXG_WASM_PATH,
  join(process.cwd(), "target", "wasm32-unknown-unknown", "release", "vxg.wasm"),
  join(homedir(), ".cargo", "target", "wasm32-unknown-unknown", "release", "vxg.wasm"),
];

const wasmPath = wasmCandidates.find((p) => p && existsSync(p));
if (!wasmPath) {
  throw new Error(
    "Could not find vxg.wasm; run `cargo build --release --lib --target wasm32-unknown-unknown` first.",
  );
}

const wasmBytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const vxg = instance.exports;

export function wasmCompress(inputBytes) {
  const srcLen = inputBytes.byteLength;
  const bound = vxg.vxg_compress_bound(srcLen, 0);
  if (bound < 0) throw new Error(`vxg_compress_bound failed: ${bound}`);

  const srcPtr = vxg.vxg_alloc(srcLen);
  const dstPtr = vxg.vxg_alloc(bound);
  try {
    new Uint8Array(vxg.memory.buffer, srcPtr, srcLen).set(inputBytes);
    const written = vxg.vxg_compress_into(srcPtr, srcLen, dstPtr, bound, 1, 0, 1);
    if (written < 0) throw new Error(`vxg_compress_into failed: ${written}`);
    return new Uint8Array(vxg.memory.buffer, dstPtr, written).slice();
  } finally {
    vxg.vxg_free(srcPtr, srcLen);
    vxg.vxg_free(dstPtr, bound);
  }
}

export function wasmDecompress(frameBytes) {
  const srcLen = frameBytes.byteLength;
  const srcPtr = vxg.vxg_alloc(srcLen);
  new Uint8Array(vxg.memory.buffer, srcPtr, srcLen).set(frameBytes);

  const rawLen = vxg.vxg_frame_raw_size(srcPtr, srcLen);
  if (rawLen < 0) {
    vxg.vxg_free(srcPtr, srcLen);
    throw new Error(`vxg_frame_raw_size failed: ${rawLen}`);
  }

  const dstPtr = vxg.vxg_alloc(rawLen);
  try {
    const written = vxg.vxg_decompress_into(srcPtr, srcLen, dstPtr, rawLen, 1, 1);
    if (written < 0) throw new Error(`vxg_decompress_into failed: ${written}`);
    return new Uint8Array(vxg.memory.buffer, dstPtr, written).slice();
  } finally {
    vxg.vxg_free(srcPtr, srcLen);
    vxg.vxg_free(dstPtr, rawLen);
  }
}

const sample = Array.from(
  { length: 1000 },
  (_, i) =>
    `{"ts":"2026-10-17T20:00:${String(i % 60).padStart(2, "0")}Z","seq":${i},"service":"edge-wasm","status":200}\n`,
).join("");
const raw = new TextEncoder().encode(sample);
const compressed = wasmCompress(raw);
const restored = wasmDecompress(compressed);

if (restored.byteLength !== raw.byteLength || Buffer.compare(restored, raw) !== 0) {
  throw new Error("WebAssembly roundtrip mismatch!");
}

console.log(
  `VXG WebAssembly (${wasmPath.split("/").slice(-3).join("/")}, ${(wasmBytes.byteLength / 1024).toFixed(1)} KiB binary) OK: ${raw.byteLength} B -> ${compressed.byteLength} B (ratio=${(compressed.byteLength / raw.byteLength).toFixed(4)})`,
);
