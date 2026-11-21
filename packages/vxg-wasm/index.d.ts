export type VxgSqlFormat = 'tsv' | 'csv' | 'json';

export interface VxgCompressOptions {
  /** Block size in bytes (`0` = default 64 KiB). */
  blockSize?: number;
  /** Embed hardware CRC32C per-block checksums. Default: `true`. */
  checksum?: boolean;
}

export interface VxgDecompressOptions {
  /** Verify per-block CRC32C checksums on decode. Default: `true`. */
  verifyChecksum?: boolean;
}

export interface VxgSqlOptions {
  /** Output serialization format: `'tsv'` | `'csv'` | `'json'`. Default: `'tsv'`. */
  format?: VxgSqlFormat;
  /** Omit header row for TSV/CSV outputs. Default: `false`. */
  noHeader?: boolean;
}

export class VxgCodec {
  static init(source?: string | URL | Response | ArrayBuffer | Uint8Array): Promise<VxgCodec>;
  compress(input: Uint8Array | string, options?: VxgCompressOptions): Uint8Array;
  decompress(frameBytes: Uint8Array, options?: VxgDecompressOptions): Uint8Array;
  sql(frameBytes: Uint8Array, query: string, options?: VxgSqlOptions): string;
}
