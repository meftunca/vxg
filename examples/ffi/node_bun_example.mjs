// Zero-copy Bun / TypeScript / JavaScript FFI integration for VXG v4.
// Passes `Uint8Array` pointers directly into `libvxg` with zero serialization overhead.
//
// Usage with Bun:
//   cargo build --release
//   bun run examples/ffi/node_bun_example.mjs

import { dlopen, FFIType, ptr, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const libName = process.platform === "win32" ? `vxg.${suffix}` : `libvxg.${suffix}`;
const libPath = [
  process.env.VXG_LIB_PATH,
  join(process.cwd(), "target", "release", libName),
  join(homedir(), ".cargo", "target", "release", libName),
].find((p) => p && existsSync(p));

if (!libPath) {
  throw new Error(`Could not find ${libName}; run 'cargo build --release' first.`);
}

const { symbols: vxg } = dlopen(libPath, {
  vxg_compress_bound: {
    args: [FFIType.usize, FFIType.usize],
    returns: FFIType.isize,
  },
  vxg_frame_raw_size: {
    args: [FFIType.ptr, FFIType.usize],
    returns: FFIType.isize,
  },
  vxg_compress_into: {
    args: [
      FFIType.ptr,
      FFIType.usize,
      FFIType.ptr,
      FFIType.usize,
      FFIType.usize,
      FFIType.usize,
      FFIType.i32,
    ],
    returns: FFIType.isize,
  },
  vxg_decompress_into: {
    args: [
      FFIType.ptr,
      FFIType.usize,
      FFIType.ptr,
      FFIType.usize,
      FFIType.usize,
      FFIType.i32,
    ],
    returns: FFIType.isize,
  },
});

export function compressVxg(input, threads = 0, blockSize = 0, checksum = true) {
  const bound = Number(vxg.vxg_compress_bound(input.byteLength, blockSize));
  if (bound < 0) throw new Error(`vxg_compress_bound failed: ${bound}`);
  const out = new Uint8Array(bound);
  const written = Number(
    vxg.vxg_compress_into(
      ptr(input),
      input.byteLength,
      ptr(out),
      out.byteLength,
      threads,
      blockSize,
      checksum ? 1 : 0,
    ),
  );
  if (written < 0) throw new Error(`vxg_compress_into failed: ${written}`);
  return out.subarray(0, written);
}

export function decompressVxg(frame, threads = 0, verifyChecksum = true) {
  const rawLen = Number(vxg.vxg_frame_raw_size(ptr(frame), frame.byteLength));
  if (rawLen < 0) throw new Error(`vxg_frame_raw_size failed: ${rawLen}`);
  const out = new Uint8Array(rawLen);
  const written = Number(
    vxg.vxg_decompress_into(
      ptr(frame),
      frame.byteLength,
      ptr(out),
      out.byteLength,
      threads,
      verifyChecksum ? 1 : 0,
    ),
  );
  if (written < 0) throw new Error(`vxg_decompress_into failed: ${written}`);
  return out;
}

if (import.meta.main) {
  const lines = Array.from(
    { length: 1000 },
    (_, i) => `{"ts":"2026-10-17T19:00:${String(i % 60).padStart(2, "0")}Z","seq":${i},"status":"ok"}\n`,
  ).join("");
  const raw = new TextEncoder().encode(lines);
  const compressed = compressVxg(raw);
  const restored = decompressVxg(compressed);
  console.log(
    `VXG JS/Bun FFI OK: ${raw.byteLength} B -> ${compressed.byteLength} B (ratio=${(compressed.byteLength / raw.byteLength).toFixed(4)})`,
  );
}
