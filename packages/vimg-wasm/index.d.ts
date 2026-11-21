export interface VimgHeaderInfo {
  width: number;
  height: number;
  tilesTotal: number;
  dirBytesNeeded: number;
  totalBytes: number;
  tileSize: number;
  dwtLevels: number;
}

export interface VimgEncodeOptions {
  /** Quality 1..=99 (lossy CDF 9/7) or 100 (bit-exact lossless CDF 5/3 + Spatial Palette). Default: 80. */
  quality?: number;
  /** Tile size in pixels (64, 128, or 256). Default: 256. */
  tileSize?: number;
  /** Enable Tile-Adaptive Psychovisual Quantization (--aq). Default: false. */
  adaptiveQuant?: boolean;
}

export interface VimgQuery {
  /** Spatial crop box `[x, y, w, h]` in full-resolution coordinates. */
  crop?: [number, number, number, number];
  /** Target output width in pixels (stops pyramid decode at coarsest level >= target, then Catmull-Rom bicubic). */
  targetWidth?: number;
  /** Target output height in pixels. */
  targetHeight?: number;
  /** Skip Co/Cg chroma planes completely (0 chroma bytes fetched). */
  grayscale?: boolean;
  /** Compressed-domain sepia tone (0 chroma bytes fetched). */
  sepia?: boolean;
  /** Subband wavelet blur `0.0..=1.0`. */
  blur?: number;
  /** Subband wavelet sharpen `0.0..=2.0`. */
  sharpen?: number;
  /** LL luminance brightness offset `-128..=128`. */
  brightness?: number;
  /** Contrast multiplier `0.1..=4.0` (default `1.0`). */
  contrast?: number;
  /** Compressed-domain Co/Cg saturation `0.0..=3.0` (default `1.0`). */
  saturation?: number;
  /** Compressed-domain Co/Cg white balance `-1.0` (cool) .. `+1.0` (warm). */
  temperature?: number;
  /** Clockwise rotation in degrees (`0 | 90 | 180 | 270`). */
  rotateDeg?: 0 | 90 | 180 | 270;
  /** Explicit wavelet pyramid tier (`0..=dwtLevels`), or `-1` for automatic selection. */
  forceLevel?: number;
}

export interface VimgRangePlan {
  ranges: [number, number][];
  level: number;
  tilesFetched: number;
  tilesTotal: number;
  totalBytes: number;
}

export interface VimgDecodedFrame {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  decodedLevel: number;
  tilesFetched: number;
  tilesTotal: number;
  totalBytes: number;
  bytesFetched: number;
}

export class VimgCodec {
  static init(source?: string | URL | Response | ArrayBuffer | Uint8Array): Promise<VimgCodec>;
  parseHeader(bytes: Uint8Array): VimgHeaderInfo;
  encodeRgba(
    rgba: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    options?: VimgEncodeOptions
  ): Uint8Array;
  planRanges(prefix: Uint8Array, query?: VimgQuery): VimgRangePlan;
  decodeSparse(
    prefix: Uint8Array,
    spans: Array<{ offset: number; data: Uint8Array }>,
    query?: VimgQuery
  ): VimgDecodedFrame;
  decodeBuffer(vimgBytes: Uint8Array, query?: VimgQuery): VimgDecodedFrame;
  decodeUrl(url: string, query?: VimgQuery): Promise<VimgDecodedFrame>;
  progressiveBuffer(vimgBytes: Uint8Array, query?: VimgQuery): Generator<VimgDecodedFrame>;
}
