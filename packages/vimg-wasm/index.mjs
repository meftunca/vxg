/**
 * @vimg/wasm — Zero-WASI WebAssembly (+simd128) VIMG1 codec wrapper.
 * Works identically in browsers (with live HTTP/1.1 206 Range streaming) and Node.js 18+.
 */

export class VimgCodec {
  /** @param {WebAssembly.Exports} exports */
  constructor(exports) {
    this.w = /** @type {any} */ (exports);
  }

  /**
   * Initialize `VimgCodec` from a URL, `Response`, `ArrayBuffer`, or `Uint8Array`.
   * If omitted, loads the bundled `./vimg.wasm` sibling file automatically.
   * @param {string | URL | Response | ArrayBuffer | Uint8Array} [source]
   * @returns {Promise<VimgCodec>}
   */
  static async init(source) {
    let bytes;
    if (source === undefined) {
      const wasmUrl = new URL('./vimg.wasm', import.meta.url);
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
    return new VimgCodec(instance.exports);
  }

  _meta() {
    return new Uint32Array(this.w.memory.buffer, this.w.vimg_wasm_meta_ptr(), 12);
  }

  /**
   * Parse a 48+ byte `VIMG1` header prefix.
   * @param {Uint8Array} bytes
   */
  parseHeader(bytes) {
    const len = Math.min(bytes.length, 4096);
    const ptr = this.w.vimg_wasm_alloc(len);
    new Uint8Array(this.w.memory.buffer, ptr, len).set(bytes.subarray(0, len));
    const st = this.w.vimg_wasm_parse_header(ptr, len);
    const meta = this._meta();
    const info = {
      width: meta[1],
      height: meta[2],
      tilesTotal: meta[5],
      dirBytesNeeded: meta[8],
      totalBytes: meta[9],
      tileSize: meta[10],
      dwtLevels: meta[11],
    };
    this.w.vimg_wasm_free(ptr, len);
    if (st !== 0) throw new Error(`VIMG header parse error (${st})`);
    return info;
  }

  /**
   * Encode `RGBA8` pixels (`width * height * 4` bytes) into a `.vimg` buffer.
   * @param {Uint8Array | Uint8ClampedArray} rgba
   * @param {number} width
   * @param {number} height
   * @param {{ quality?: number; tileSize?: number; adaptiveQuant?: boolean }} [options]
   * @returns {Uint8Array}
   */
  encodeRgba(rgba, width, height, options = {}) {
    const quality = options.quality ?? 80;
    const tileSize = options.tileSize ?? 256;
    const aq = options.adaptiveQuant ? 1 : 0;
    const expected = width * height * 4;
    if (rgba.length !== expected) {
      throw new Error(`Expected ${expected} RGBA bytes, got ${rgba.length}`);
    }
    const inPtr = this.w.vimg_wasm_alloc(rgba.length);
    new Uint8Array(this.w.memory.buffer, inPtr, rgba.length).set(rgba);
    const st = typeof this.w.vimg_wasm_encode_rgba_v3 === 'function'
      ? this.w.vimg_wasm_encode_rgba_v3(inPtr, width, height, quality, tileSize, aq)
      : this.w.vimg_wasm_encode_rgba(inPtr, width, height, quality, tileSize);
    this.w.vimg_wasm_free(inPtr, rgba.length);
    if (st !== 0) throw new Error(`VIMG encode error (${st})`);
    const meta = this._meta();
    const outPtr = meta[6];
    const outLen = meta[7];
    const out = new Uint8Array(this.w.memory.buffer.slice(outPtr, outPtr + outLen));
    this.w.vimg_wasm_free(outPtr, outLen);
    return out;
  }

  /**
   * Plan the exact coalesced byte ranges `[start, end)` required for a query given the header+dir prefix.
   * @param {Uint8Array} prefix
   * @param {import('./index.d.ts').VimgQuery} [query]
   */
  planRanges(prefix, query = {}) {
    const c = query.crop ?? [0, 0, 0, 0];
    const tw = query.targetWidth ?? 0;
    const th = query.targetHeight ?? 0;
    const gray = query.grayscale ? 1 : 0;
    const sepia = query.sepia ? 1 : 0;
    const forceLevel = query.forceLevel ?? -1;

    const pPtr = this.w.vimg_wasm_alloc(prefix.length);
    new Uint8Array(this.w.memory.buffer, pPtr, prefix.length).set(prefix);
    const st = typeof this.w.vimg_wasm_plan_v3 === 'function'
      ? this.w.vimg_wasm_plan_v3(pPtr, prefix.length, c[0], c[1], c[2], c[3], tw, th, gray, sepia, forceLevel)
      : this.w.vimg_wasm_plan(pPtr, prefix.length, c[0], c[1], c[2], c[3], tw, th, gray, sepia);
    this.w.vimg_wasm_free(pPtr, prefix.length);
    if (st !== 0) throw new Error(`VIMG plan error (${st})`);

    const meta = this._meta();
    const rPtr = meta[6];
    const rCap = meta[7];
    const nRanges = rCap / 8;
    const flat = new Uint32Array(this.w.memory.buffer.slice(rPtr, rPtr + rCap));
    this.w.vimg_wasm_free(rPtr, rCap);

    /** @type {[number, number][]} */
    const ranges = [];
    for (let i = 0; i < nRanges; i++) {
      ranges.push([flat[i * 2], flat[i * 2 + 1]]);
    }
    return {
      ranges,
      level: meta[3],
      tilesFetched: meta[4],
      tilesTotal: meta[5],
      totalBytes: meta[9],
    };
  }

  /**
   * Decode from a prefix and fetched sparse `[offset, Uint8Array]` byte spans.
   * @param {Uint8Array} prefix
   * @param {Array<{ offset: number; data: Uint8Array }>} spans
   * @param {import('./index.d.ts').VimgQuery} [query]
   */
  decodeSparse(prefix, spans, query = {}) {
    const c = query.crop ?? [0, 0, 0, 0];
    const tw = query.targetWidth ?? 0;
    const th = query.targetHeight ?? 0;
    const gray = query.grayscale ? 1 : 0;
    const sepia = query.sepia ? 1 : 0;
    const blur = query.blur ?? 0;
    const sharpen = query.sharpen ?? 0;
    const brightness = query.brightness ?? 0;
    const contrast = query.contrast ?? 1.0;
    const saturation = query.saturation ?? 1.0;
    const temperature = query.temperature ?? 0.0;
    const rot = ((query.rotateDeg ?? 0) / 90) & 3;
    const forceLevel = query.forceLevel ?? -1;

    const nSpans = spans.length;
    const totalSpanBytes = spans.reduce((acc, s) => acc + s.data.length, 0);
    const pPtr = this.w.vimg_wasm_alloc(prefix.length);
    const offsPtr = this.w.vimg_wasm_alloc(Math.max(4, nSpans * 4));
    const lensPtr = this.w.vimg_wasm_alloc(Math.max(4, nSpans * 4));
    const dataPtr = this.w.vimg_wasm_alloc(Math.max(1, totalSpanBytes));

    new Uint8Array(this.w.memory.buffer, pPtr, prefix.length).set(prefix);
    const offsArr = new Uint32Array(this.w.memory.buffer, offsPtr, nSpans);
    const lensArr = new Uint32Array(this.w.memory.buffer, lensPtr, nSpans);
    const dataArr = new Uint8Array(this.w.memory.buffer, dataPtr, totalSpanBytes);
    let cursor = 0;
    for (let i = 0; i < nSpans; i++) {
      offsArr[i] = spans[i].offset;
      lensArr[i] = spans[i].data.length;
      dataArr.set(spans[i].data, cursor);
      cursor += spans[i].data.length;
    }

    const st = typeof this.w.vimg_wasm_decode_sparse_v3 === 'function'
      ? this.w.vimg_wasm_decode_sparse_v3(
          pPtr, prefix.length, offsPtr, lensPtr, nSpans, dataPtr, totalSpanBytes,
          c[0], c[1], c[2], c[3], tw, th, gray, sepia, blur, sharpen, brightness,
          contrast, saturation, temperature, rot, forceLevel
        )
      : this.w.vimg_wasm_decode_sparse(
          pPtr, prefix.length, offsPtr, lensPtr, nSpans, dataPtr, totalSpanBytes,
          c[0], c[1], c[2], c[3], tw, th, gray, sepia, blur, sharpen, brightness, contrast, rot
        );

    this.w.vimg_wasm_free(pPtr, prefix.length);
    this.w.vimg_wasm_free(offsPtr, Math.max(4, nSpans * 4));
    this.w.vimg_wasm_free(lensPtr, Math.max(4, nSpans * 4));
    this.w.vimg_wasm_free(dataPtr, Math.max(1, totalSpanBytes));
    if (st !== 0) throw new Error(`VIMG decode error (${st})`);

    const meta = this._meta();
    const width = meta[1];
    const height = meta[2];
    const outPtr = meta[6];
    const outCap = meta[7];
    const rgba = new Uint8ClampedArray(this.w.memory.buffer.slice(outPtr, outPtr + width * height * 4));
    this.w.vimg_wasm_free(outPtr, outCap);

    return {
      rgba,
      width,
      height,
      decodedLevel: meta[3],
      tilesFetched: meta[4],
      tilesTotal: meta[5],
      totalBytes: meta[9],
      bytesFetched: prefix.length + totalSpanBytes,
    };
  }

  /**
   * Convenience helper: run a compressed-domain query on an in-memory `.vimg` buffer
   * using the exact same sparse range planner (reporting accurate byte transfer stats!).
   * @param {Uint8Array} vimgBytes
   * @param {import('./index.d.ts').VimgQuery} [query]
   */
  decodeBuffer(vimgBytes, query = {}) {
    const hdr = this.parseHeader(vimgBytes);
    const plen = Math.min(vimgBytes.length, Math.max(4096, hdr.dirBytesNeeded));
    const prefix = vimgBytes.subarray(0, plen);
    const plan = this.planRanges(prefix, query);
    const spans = plan.ranges.map(([s, e]) => ({ offset: s, data: vimgBytes.subarray(s, e) }));
    return this.decodeSparse(prefix, spans, query);
  }

  /**
   * Fetch and decode a remote `.vimg` over HTTP/HTTPS `Range: bytes=...` requests.
   * @param {string} url
   * @param {import('./index.d.ts').VimgQuery} [query]
   */
  async decodeUrl(url, query = {}) {
    const fetchSlice = async (s, e) => {
      const res = await fetch(url, { headers: { Range: `bytes=${s}-${e - 1}` } });
      if (!res.ok) throw new Error(`HTTP ${res.status} for range ${s}-${e - 1}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      return res.status === 206 ? buf : buf.subarray(s, e);
    };
    let prefix = await fetchSlice(0, 4096);
    const hdr = this.parseHeader(prefix);
    if (hdr.dirBytesNeeded > prefix.length) {
      const extra = await fetchSlice(prefix.length, hdr.dirBytesNeeded);
      const merged = new Uint8Array(hdr.dirBytesNeeded);
      merged.set(prefix, 0);
      merged.set(extra, prefix.length);
      prefix = merged;
    }
    const plan = this.planRanges(prefix, query);
    const datas = await Promise.all(plan.ranges.map(([s, e]) => fetchSlice(s, e)));
    const spans = plan.ranges.map(([s], i) => ({ offset: s, data: datas[i] }));
    return this.decodeSparse(prefix, spans, query);
  }

  /**
   * Async generator yielding progressive frames `L0 -> L1 -> ... -> dwtLevels` from an in-memory buffer.
   * @param {Uint8Array} vimgBytes
   * @param {import('./index.d.ts').VimgQuery} [query]
   */
  *progressiveBuffer(vimgBytes, query = {}) {
    const hdr = this.parseHeader(vimgBytes);
    for (let lvl = 0; lvl <= hdr.dwtLevels; lvl++) {
      yield this.decodeBuffer(vimgBytes, { ...query, forceLevel: lvl });
    }
  }
}
