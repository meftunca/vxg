/**
 * @vxg/wasm — Zero-WASI WebAssembly wrapper for VXG (block codec + compressed-domain SQL).
 * Works in Browser, Node.js 18+, Bun, and Edge Workers with 0 JS dependencies.
 */

const FORMAT_MAP = { tsv: 0, csv: 1, json: 2 };

export class VxgCodec {
  /** @param {WebAssembly.Exports} exports */
  constructor(exports) {
    this.w = /** @type {any} */ (exports);
    this._enc = new TextEncoder();
    this._dec = new TextDecoder();
  }

  /**
   * Initialize `VxgCodec` from a URL, `Response`, `ArrayBuffer`, or `Uint8Array`.
   * If omitted, loads the bundled sibling `./vxg.wasm`.
   * @param {string | URL | Response | ArrayBuffer | Uint8Array} [source]
   * @returns {Promise<VxgCodec>}
   */
  static async init(source) {
    let bytes;
    if (source === undefined) {
      const wasmUrl = new URL('./vxg.wasm', import.meta.url);
      if (wasmUrl.protocol === 'file:') {
        const fs = await import('node:fs/promises');
        bytes = await fs.readFile(wasmUrl);
      } else {
        const res = await fetch(wasmUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status} loading ${wasmUrl}`);
        bytes = await res.arrayBuffer();
      }
    } else if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
      bytes = source;
    } else if (typeof Response !== 'undefined' && source instanceof Response) {
      bytes = await source.arrayBuffer();
    } else {
      const url = typeof source === 'string' ? source : source.toString();
      if (url.startsWith('file://')) {
        const fs = await import('node:fs/promises');
        bytes = await fs.readFile(new URL(url));
      } else {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
        bytes = await res.arrayBuffer();
      }
    }
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new VxgCodec(instance.exports);
  }

  /**
   * Compress raw UTF-8 text or binary bytes into a self-contained `VXG1` frame.
   * @param {Uint8Array | string} input
   * @param {{ blockSize?: number; checksum?: boolean }} [options]
   * @returns {Uint8Array}
   */
  compress(input, options = {}) {
    const raw = typeof input === 'string' ? this._enc.encode(input) : input;
    const blockSize = options.blockSize ?? 0;
    const checksum = options.checksum === false ? 0 : 1;
    const bound = this.w.vxg_compress_bound(raw.byteLength, blockSize);
    if (bound < 0) throw new Error(`vxg_compress_bound failed (${bound})`);
    const srcPtr = this.w.vxg_alloc(Math.max(1, raw.byteLength));
    const dstPtr = this.w.vxg_alloc(Math.max(1, bound));
    try {
      new Uint8Array(this.w.memory.buffer, srcPtr, raw.byteLength).set(raw);
      const written = this.w.vxg_compress_into(srcPtr, raw.byteLength, dstPtr, bound, 1, blockSize, checksum);
      if (written < 0) throw new Error(`vxg_compress_into failed (${written})`);
      return new Uint8Array(this.w.memory.buffer, dstPtr, written).slice();
    } finally {
      this.w.vxg_free(srcPtr, Math.max(1, raw.byteLength));
      this.w.vxg_free(dstPtr, Math.max(1, bound));
    }
  }

  /**
   * Decompress a `VXG1` frame back into byte-exact original bytes.
   * @param {Uint8Array} frameBytes
   * @param {{ verifyChecksum?: boolean }} [options]
   * @returns {Uint8Array}
   */
  decompress(frameBytes, options = {}) {
    const verify = options.verifyChecksum === false ? 0 : 1;
    const srcPtr = this.w.vxg_alloc(Math.max(1, frameBytes.byteLength));
    try {
      new Uint8Array(this.w.memory.buffer, srcPtr, frameBytes.byteLength).set(frameBytes);
      const rawLen = this.w.vxg_frame_raw_size(srcPtr, frameBytes.byteLength);
      if (rawLen < 0) throw new Error(`vxg_frame_raw_size failed (${rawLen})`);
      const dstPtr = this.w.vxg_alloc(Math.max(1, rawLen));
      try {
        const written = this.w.vxg_decompress_into(srcPtr, frameBytes.byteLength, dstPtr, rawLen, 1, verify);
        if (written < 0) throw new Error(`vxg_decompress_into failed (${written})`);
        return new Uint8Array(this.w.memory.buffer, dstPtr, written).slice();
      } finally {
        this.w.vxg_free(dstPtr, Math.max(1, rawLen));
      }
    } finally {
      this.w.vxg_free(srcPtr, Math.max(1, frameBytes.byteLength));
    }
  }

  /**
   * Execute a compressed-domain SQL query (`SELECT`, `DISTINCT`, `WHERE`, `GROUP BY`, `HAVING`,
   * `COUNT(DISTINCT ...)`, `ORDER BY`, `LIMIT`, `OFFSET`) directly on a compressed `.vxg` frame!
   * @param {Uint8Array} frameBytes
   * @param {string} query
   * @param {{ format?: 'tsv' | 'csv' | 'json'; noHeader?: boolean }} [options]
   * @returns {string}
   */
  sql(frameBytes, query, options = {}) {
    const fmt = FORMAT_MAP[options.format ?? 'tsv'] ?? 0;
    const noHeader = options.noHeader ? 1 : 0;
    const qBytes = this._enc.encode(query + '\0');
    const fPtr = this.w.vxg_alloc(Math.max(1, frameBytes.byteLength));
    const qPtr = this.w.vxg_alloc(qBytes.byteLength);
    const lenPtr = this.w.vxg_alloc(4);
    try {
      new Uint8Array(this.w.memory.buffer, fPtr, frameBytes.byteLength).set(frameBytes);
      new Uint8Array(this.w.memory.buffer, qPtr, qBytes.byteLength).set(qBytes);
      new Uint32Array(this.w.memory.buffer, lenPtr, 1)[0] = 0;
      const outPtr = this.w.vxg_sql_alloc(fPtr, frameBytes.byteLength, qPtr, fmt, noHeader, 1, lenPtr);
      if (outPtr === 0) throw new Error(`VXG SQL query failed: ${query}`);
      const outLen = new Uint32Array(this.w.memory.buffer, lenPtr, 1)[0];
      const outBytes = new Uint8Array(this.w.memory.buffer, outPtr, outLen).slice();
      this.w.vxg_free(outPtr, outLen);
      return this._dec.decode(outBytes);
    } finally {
      this.w.vxg_free(fPtr, Math.max(1, frameBytes.byteLength));
      this.w.vxg_free(qPtr, qBytes.byteLength);
      this.w.vxg_free(lenPtr, 4);
    }
  }
}
