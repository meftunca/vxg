<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

interface VxgExports {
  memory: WebAssembly.Memory
  vxg_version_string(): number
  vxg_error_message(code: number): number
  vxg_alloc(len: number): number
  vxg_free(ptr: number, len: number): void
  vxg_compress_bound(rawLen: number, blockSize: number): number
  vxg_compress_into(
    src: number,
    srcLen: number,
    dst: number,
    dstCap: number,
    threads: number,
    blockSize: number,
    checksum: number
  ): number
  vxg_decompress_into(
    src: number,
    srcLen: number,
    dst: number,
    dstCap: number,
    threads: number,
    verifyChecksum: number
  ): number
  vxg_verify_frame(src: number, srcLen: number, threads: number, verifyChecksum: number): number
  vxg_frame_raw_size(src: number, srcLen: number): number
  vxg_frame_compressed_size(src: number, srcLen: number): number
  vxg_frame_block_count(src: number, srcLen: number): number
  vxg_frame_block_raw_size(src: number, srcLen: number, blockIdx: number): number
  vxg_extract_block_into(
    src: number,
    srcLen: number,
    blockIdx: number,
    dst: number,
    dstCap: number,
    verifyChecksum: number
  ): number
  vxg_frame_row_count(src: number, srcLen: number): number
  vxg_count_where_str(src: number, srcLen: number, whereCstr: number, threads: number): number
  vxg_aggregate_col(
    src: number,
    srcLen: number,
    colCstr: number,
    whereCstr: number,
    threads: number,
    outStats: number
  ): number
  vxg_select_alloc(
    src: number,
    srcLen: number,
    colsCstr: number,
    whereCstr: number,
    orderByCstr: number,
    desc: number,
    limit: number,
    threads: number,
    outLen: number
  ): number
  vxg_project_frame_alloc(
    src: number,
    srcLen: number,
    colsCstr: number,
    whereCstr: number,
    threads: number,
    outLen: number
  ): number
}

interface BlockSummary {
  index: number
  rawSize: number
  compressedSize: number
  packetLabel: string
  crcHex: string
}

let wasm: VxgExports | null = null

const wasmState = ref<'loading' | 'ready' | 'error'>('loading')
const wasmError = ref('')
const wasmVersion = ref('')
const wasmBinaryBytes = ref(0)
const wasmMemoryBytes = ref(0)

const activePreset = ref<'ndjson' | 'csv' | 'clf' | 'prose' | 'custom'>('ndjson')
const inputText = ref('')
const uploadedBinary = ref<Uint8Array | null>(null)
const uploadedName = ref('')
const uploadedIsVxg = ref(false)
const isDragging = ref(false)

const blockSize = ref(65536)
const checksumEnabled = ref(true)

const rawBytesLen = ref(0)
const vxgBytes = ref<Uint8Array | null>(null)
const decodedBytes = ref<Uint8Array | null>(null)
const encodeMs = ref(0)
const decodeMs = ref(0)
const verifyOk = ref(false)
const roundtripExact = ref(false)
const blockCount = ref(0)
const rowCount = ref(0)
const blockSummaries = ref<BlockSummary[]>([])
const extractedBlockPreview = ref('')
const extractedBlockIdx = ref<number | null>(null)
const codecError = ref('')

// Compressed-domain SQL & Splicing state
const qSelect = ref('ts,service,region,status,latency_ms')
const qWhere = ref('status==500 && latency_ms>=120')
const qOrderBy = ref('latency_ms')
const qDesc = ref(true)
const qLimit = ref(8)
const qAggCol = ref('latency_ms')

const queryOutput = ref('')
const queryMs = ref(0)
const matchCount = ref<number | null>(null)
const aggSummary = ref('')
const splicedBytes = ref<number | null>(null)
const queryError = ref('')

function readCString(vxg: VxgExports, ptr: number): string {
  const p = ptr >>> 0
  if (!p) return ''
  const mem = new Uint8Array(vxg.memory.buffer)
  let end = p
  while (end < mem.length && mem[end] !== 0) end++
  return new TextDecoder().decode(mem.subarray(p, end))
}

function allocBytes(vxg: VxgExports, data: Uint8Array): [number, number] {
  const len = data.byteLength
  if (len === 0) return [0, 0]
  const ptr = vxg.vxg_alloc(len) >>> 0
  if (!ptr) throw new Error(`vxg_alloc(${len}) returned NULL`)
  new Uint8Array(vxg.memory.buffer, ptr, len).set(data)
  return [ptr, len]
}

function allocCString(vxg: VxgExports, text: string): [number, number] {
  const trimmed = text.trim()
  if (!trimmed) return [0, 0]
  const enc = new TextEncoder().encode(trimmed)
  const len = enc.byteLength + 1
  const ptr = vxg.vxg_alloc(len) >>> 0
  if (!ptr) throw new Error('vxg_alloc failed for C string')
  const view = new Uint8Array(vxg.memory.buffer, ptr, len)
  view.set(enc)
  view[enc.byteLength] = 0
  return [ptr, len]
}

function freeBuf(vxg: VxgExports, ptr: number, len: number) {
  if (ptr && len > 0) vxg.vxg_free(ptr, len)
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(2)} KiB`
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`
}

function formatThroughput(bytes: number, ms: number): string {
  if (ms <= 0 || bytes <= 0) return '—'
  const mbps = bytes / (1024 * 1024) / (ms / 1000)
  return mbps >= 1024 ? `${(mbps / 1024).toFixed(2)} GB/s` : `${mbps.toFixed(1)} MB/s`
}

function buildPreset(kind: 'ndjson' | 'csv' | 'clf' | 'prose'): string {
  const services = ['auth', 'payments', 'gateway', 'search', 'inventory', 'billing']
  const regions = ['eu-central', 'us-east', 'ap-northeast', 'us-west']
  if (kind === 'ndjson') {
    const lines: string[] = []
    for (let i = 0; i < 1500; i++) {
      const svc = services[i % services.length]
      const reg = regions[i % regions.length]
      const isErr = i % 9 === 0
      const status = isErr ? 500 : i % 17 === 0 ? 429 : 200
      const latency = isErr ? 110 + ((i * 37) % 390) : 6 + ((i * 13) % 58)
      const bytes = 512 + ((i * 251) % 16384)
      const sec = String(i % 60).padStart(2, '0')
      const min = String(Math.floor(i / 60) % 60).padStart(2, '0')
      lines.push(
        `{"ts":"2026-10-17T20:${min}:${sec}Z","seq":${100000 + i},"service":"${svc}","region":"${reg}","status":${status},"latency_ms":${latency},"bytes":${bytes}}\n`
      )
    }
    return lines.join('')
  }
  if (kind === 'csv') {
    const lines: string[] = []
    for (let i = 0; i < 2000; i++) {
      const host = `edge-node-${String((i % 16) + 1).padStart(2, '0')}`
      const isSpike = i % 11 === 0
      const cpu = isSpike ? 82 + (i % 18) : 14 + ((i * 7) % 45)
      const mem = 4096 + ((i * 64) % 12288)
      const rxKb = 320 + ((i * 97) % 9400)
      const status = isSpike ? 500 : 200
      const sec = String(i % 60).padStart(2, '0')
      const min = String(Math.floor(i / 60) % 60).padStart(2, '0')
      lines.push(`2026-10-17T21:${min}:${sec}Z,${host},${cpu},${mem},${rxKb},${status}\n`)
    }
    return lines.join('')
  }
  if (kind === 'clf') {
    const paths = ['/api/v2/checkout', '/api/v2/tokens', '/api/v2/catalog', '/healthz', '/static/app.js']
    const lines: string[] = []
    for (let i = 0; i < 1200; i++) {
      const ip = `10.42.${(i >> 4) & 31}.${(i % 48) + 10}`
      const p = paths[i % paths.length]
      const status = i % 13 === 0 ? 502 : 200
      const bytes = 256 + ((i * 193) % 24576)
      const sec = String(i % 60).padStart(2, '0')
      lines.push(`${ip} - - [17/Oct/2026:22:15:${sec} +0000] "GET ${p} HTTP/1.1" ${status} ${bytes}\n`)
    }
    return lines.join('')
  }
  const paragraphs = [
    '# Vectorized Execution Grammar (VXG) Wire Specification\n\n',
    'VXG partitions every input stream into independent fixed-size blocks (64 KiB by default), ',
    'allowing O(1) random block extraction, lock-free multi-core scaling, and SIMD-accelerated ',
    'decode across ARM64 NEON, x86_64 AVX2, WebAssembly, Apple Metal, and Vulkan compute shaders.\n\n',
    'Unlike monolithic sliding-window LZ77 compressors that repeatedly rediscover JSON keys, CSV ',
    'delimiters, and ISO-8601 timestamps on every record, VXG routes each block by exact byte ',
    'budget across GRAMMAR_PACKET (columnar delta/RLE/dictionary/Huffman), STRIDE_PACKET (2D ',
    'decorrelation planes), and FAST_LZ_PACKET (4-stream canonical Huffman + repeat-offset LZ).\n\n'
  ]
  return paragraphs.join('').repeat(36)
}

function inspectV4Directory(frame: Uint8Array, vxg: VxgExports, framePtr: number): BlockSummary[] {
  if (frame.byteLength < 32) return []
  const dv = new DataView(frame.buffer, frame.byteOffset, frame.byteLength)
  const version = dv.getUint16(4, true)
  const count = dv.getUint32(20, true)
  const entrySize = version >= 4 ? 8 : 32
  const dirEnd = 32 + count * entrySize
  if (frame.byteLength < dirEnd) return []

  const out: BlockSummary[] = []
  let payloadCursor = dirEnd
  for (let i = 0; i < count && i < 64; i++) {
    const rawSz = vxg.vxg_frame_block_raw_size(framePtr, frame.byteLength, i)
    let compSz = 0
    let kind = 0
    let crc = 0
    if (version >= 4) {
      const word = dv.getUint32(32 + i * 8, true)
      compSz = word & 0x3fffffff
      kind = word >>> 30
      crc = dv.getUint32(32 + i * 8 + 4, true)
    } else {
      compSz = dv.getUint32(32 + i * 32 + 20, true)
      crc = dv.getUint32(32 + i * 32 + 24, true)
      kind = dv.getUint32(32 + i * 32 + 28, true)
    }
    let label = kind === 0 ? 'RAW' : kind === 1 ? 'VXG-LZ' : 'COLUMN'
    if (kind === 2 && payloadCursor < frame.byteLength) {
      const sub = frame[payloadCursor]
      if (sub === 1) label = 'GRAMMAR_PACKET'
      else if (sub === 2) label = 'STRIDE_PACKET'
      else if (sub === 3) label = 'FAST_LZ_PACKET'
      else if (sub === 0) label = 'RAW_PACKET'
    }
    payloadCursor += compSz
    out.push({
      index: i,
      rawSize: rawSz >= 0 ? rawSz : 0,
      compressedSize: compSz,
      packetLabel: label,
      crcHex: crc.toString(16).padStart(8, '0')
    })
  }
  return out
}

function selectPreset(kind: 'ndjson' | 'csv' | 'clf' | 'prose') {
  activePreset.value = kind
  uploadedBinary.value = null
  uploadedName.value = ''
  uploadedIsVxg.value = false
  inputText.value = buildPreset(kind)

  if (kind === 'ndjson') {
    qSelect.value = 'ts,service,region,status,latency_ms'
    qWhere.value = 'status==500 && latency_ms>=120'
    qOrderBy.value = 'latency_ms'
    qDesc.value = true
    qAggCol.value = 'latency_ms'
  } else if (kind === 'csv') {
    qSelect.value = 'csv:0,csv:1,csv:2,csv:5'
    qWhere.value = 'csv:5==500 && csv:2>=85'
    qOrderBy.value = 'csv:2'
    qDesc.value = true
    qAggCol.value = 'csv:2'
  } else {
    qSelect.value = ''
    qWhere.value = ''
    qOrderBy.value = ''
    qAggCol.value = ''
  }
  runPipeline()
}

function runPipeline() {
  if (!wasm) return
  codecError.value = ''
  extractedBlockIdx.value = null
  extractedBlockPreview.value = ''

  try {
    if (uploadedBinary.value && uploadedIsVxg.value) {
      const frame = uploadedBinary.value
      const [fPtr, fLen] = allocBytes(wasm, frame)
      try {
        const ver = wasm.vxg_verify_frame(fPtr, fLen, 1, checksumEnabled.value ? 1 : 0)
        if (ver < 0) {
          throw new Error(`Invalid .vxg frame: ${readCString(wasm, wasm.vxg_error_message(ver))}`)
        }
        const expectedRaw = wasm.vxg_frame_raw_size(fPtr, fLen)
        if (expectedRaw < 0) {
          throw new Error(readCString(wasm, wasm.vxg_error_message(expectedRaw)))
        }
        const dstPtr = expectedRaw > 0 ? wasm.vxg_alloc(expectedRaw) >>> 0 : 0
        try {
          const t0 = performance.now()
          const written = wasm.vxg_decompress_into(
            fPtr,
            fLen,
            dstPtr,
            expectedRaw,
            1,
            checksumEnabled.value ? 1 : 0
          )
          const t1 = performance.now()
          if (written < 0) {
            throw new Error(readCString(wasm, wasm.vxg_error_message(written)))
          }
          const out =
            written > 0
              ? new Uint8Array(wasm.memory.buffer, dstPtr, written).slice()
              : new Uint8Array(0)
          vxgBytes.value = frame
          decodedBytes.value = out
          rawBytesLen.value = out.byteLength
          encodeMs.value = 0
          decodeMs.value = Math.max(0.01, t1 - t0)
          verifyOk.value = true
          roundtripExact.value = true
          blockCount.value = Math.max(0, wasm.vxg_frame_block_count(fPtr, fLen))
          const rows = wasm.vxg_frame_row_count(fPtr, fLen)
          rowCount.value = rows >= 0 ? rows : 0
          blockSummaries.value = inspectV4Directory(frame, wasm, fPtr)
        } finally {
          freeBuf(wasm, dstPtr, expectedRaw)
        }
      } finally {
        freeBuf(wasm, fPtr, fLen)
      }
      wasmMemoryBytes.value = wasm.memory.buffer.byteLength
      runQuery()
      return
    }

    const raw = uploadedBinary.value ?? new TextEncoder().encode(inputText.value)
    rawBytesLen.value = raw.byteLength

    const bound = wasm.vxg_compress_bound(raw.byteLength, blockSize.value)
    if (bound < 0) {
      throw new Error(readCString(wasm, wasm.vxg_error_message(bound)))
    }
    const [srcPtr, srcLen] = allocBytes(wasm, raw)
    const dstPtr = wasm.vxg_alloc(bound) >>> 0
    if (!dstPtr) {
      freeBuf(wasm, srcPtr, srcLen)
      throw new Error('WASM compress buffer allocation failed')
    }

    let compressed: Uint8Array
    try {
      // Warm 1st pass if tiny, then timed encode
      const t0 = performance.now()
      const cLen = wasm.vxg_compress_into(
        srcPtr,
        srcLen,
        dstPtr,
        bound,
        1,
        blockSize.value,
        checksumEnabled.value ? 1 : 0
      )
      const t1 = performance.now()
      if (cLen < 0) {
        throw new Error(readCString(wasm, wasm.vxg_error_message(cLen)))
      }
      encodeMs.value = Math.max(0.01, t1 - t0)
      compressed = new Uint8Array(wasm.memory.buffer, dstPtr, cLen).slice()
    } finally {
      freeBuf(wasm, srcPtr, srcLen)
      freeBuf(wasm, dstPtr, bound)
    }

    // Verify & Decompress roundtrip
    const [fPtr, fLen] = allocBytes(wasm, compressed)
    const decPtr = raw.byteLength > 0 ? wasm.vxg_alloc(raw.byteLength) >>> 0 : 0
    try {
      const vCode = wasm.vxg_verify_frame(fPtr, fLen, 1, checksumEnabled.value ? 1 : 0)
      verifyOk.value = vCode === raw.byteLength

      const t2 = performance.now()
      const dLen = wasm.vxg_decompress_into(
        fPtr,
        fLen,
        decPtr,
        raw.byteLength,
        1,
        checksumEnabled.value ? 1 : 0
      )
      const t3 = performance.now()
      if (dLen < 0) {
        throw new Error(readCString(wasm, wasm.vxg_error_message(dLen)))
      }
      decodeMs.value = Math.max(0.01, t3 - t2)
      const restored =
        dLen > 0 ? new Uint8Array(wasm.memory.buffer, decPtr, dLen).slice() : new Uint8Array(0)
      decodedBytes.value = restored
      vxgBytes.value = compressed

      let exact = restored.byteLength === raw.byteLength
      if (exact) {
        for (let i = 0; i < raw.byteLength; i++) {
          if (restored[i] !== raw[i]) {
            exact = false
            break
          }
        }
      }
      roundtripExact.value = exact
      blockCount.value = Math.max(0, wasm.vxg_frame_block_count(fPtr, fLen))
      const rows = wasm.vxg_frame_row_count(fPtr, fLen)
      rowCount.value = rows >= 0 ? rows : 0
      blockSummaries.value = inspectV4Directory(compressed, wasm, fPtr)
    } finally {
      freeBuf(wasm, fPtr, fLen)
      freeBuf(wasm, decPtr, raw.byteLength)
    }

    wasmMemoryBytes.value = wasm.memory.buffer.byteLength
    runQuery()
  } catch (err: unknown) {
    codecError.value = err instanceof Error ? err.message : String(err)
  }
}

function extractSingleBlock(idx: number) {
  if (!wasm || !vxgBytes.value) return
  const [fPtr, fLen] = allocBytes(wasm, vxgBytes.value)
  try {
    const rawSz = wasm.vxg_frame_block_raw_size(fPtr, fLen, idx)
    if (rawSz < 0) {
      throw new Error(readCString(wasm, wasm.vxg_error_message(rawSz)))
    }
    const dstPtr = rawSz > 0 ? wasm.vxg_alloc(rawSz) >>> 0 : 0
    try {
      const t0 = performance.now()
      const written = wasm.vxg_extract_block_into(
        fPtr,
        fLen,
        idx,
        dstPtr,
        rawSz,
        checksumEnabled.value ? 1 : 0
      )
      const dt = Math.max(0.005, performance.now() - t0)
      if (written < 0) {
        throw new Error(readCString(wasm, wasm.vxg_error_message(written)))
      }
      const slice =
        written > 0 ? new Uint8Array(wasm.memory.buffer, dstPtr, Math.min(written, 600)) : new Uint8Array(0)
      const text = new TextDecoder('utf-8', { fatal: false }).decode(slice)
      extractedBlockIdx.value = idx
      extractedBlockPreview.value = `[Block #${idx} extracted in ${dt.toFixed(3)} ms — ${written.toLocaleString()} raw bytes]\n${text}${written > 600 ? '\n… (truncated)' : ''}`
    } finally {
      freeBuf(wasm, dstPtr, rawSz)
    }
  } catch (err: unknown) {
    codecError.value = err instanceof Error ? err.message : String(err)
  } finally {
    freeBuf(wasm, fPtr, fLen)
  }
}

function runQuery() {
  if (!wasm || !vxgBytes.value) return
  queryError.value = ''
  queryOutput.value = ''
  aggSummary.value = ''
  matchCount.value = null
  splicedBytes.value = null

  if (rowCount.value <= 0) {
    queryOutput.value =
      'Current payload is compressed via FAST_LZ / STRIDE / RAW (non-tabular stream). Select NDJSON or CSV preset to run compressed-domain SQL queries, Top-K sorting, and zero-reencode column splicing.'
    return
  }

  const [fPtr, fLen] = allocBytes(wasm, vxgBytes.value)
  const [colsPtr, colsLen] = allocCString(wasm, qSelect.value)
  const [wherePtr, whereLen] = allocCString(wasm, qWhere.value)
  const [orderPtr, orderLen] = allocCString(wasm, qOrderBy.value)
  const [aggPtr, aggLen] = allocCString(wasm, qAggCol.value)
  const outLenRaw = wasm.vxg_alloc(16) >>> 0
  const outLenPtr = (outLenRaw + 7) & ~7
  const statsRaw = wasm.vxg_alloc(72) >>> 0
  const statsPtr = (statsRaw + 7) & ~7

  try {
    const t0 = performance.now()
    const cnt = wasm.vxg_count_where_str(fPtr, fLen, wherePtr, 1)
    if (cnt < 0) {
      throw new Error(`WHERE syntax error: ${readCString(wasm, wasm.vxg_error_message(cnt))}`)
    }
    matchCount.value = cnt

    const lim = Math.max(1, Math.min(500, Number(qLimit.value) || 10))
    const resPtr =
      wasm.vxg_select_alloc(
        fPtr,
        fLen,
        colsPtr,
        wherePtr,
        orderPtr,
        qDesc.value ? 1 : 0,
        lim,
        1,
        outLenPtr
      ) >>> 0
    const t1 = performance.now()
    queryMs.value = Math.max(0.01, t1 - t0)

    if (!resPtr) {
      throw new Error('Query failed — check column names (e.g. latency_ms for JSON or csv:2 for CSV)')
    }
    const resLen = new DataView(wasm.memory.buffer).getUint32(outLenPtr, true)
    try {
      const bytes =
        resLen > 0 ? new Uint8Array(wasm.memory.buffer, resPtr, resLen).slice() : new Uint8Array(0)
      queryOutput.value =
        resLen > 0 ? new TextDecoder().decode(bytes) : '(0 matching records)'
    } finally {
      freeBuf(wasm, resPtr, resLen)
    }

    if (aggPtr) {
      const stCode = wasm.vxg_aggregate_col(fPtr, fLen, aggPtr, wherePtr, 1, statsPtr)
      if (stCode === 0) {
        const dv = new DataView(wasm.memory.buffer)
        const numCnt = Number(dv.getBigUint64(statsPtr + 8, true))
        const sum = dv.getFloat64(statsPtr + 16, true)
        const minV = dv.getBigUint64(statsPtr + 40, true)
        const maxV = dv.getBigUint64(statsPtr + 48, true)
        const hasMinMax = dv.getInt32(statsPtr + 56, true) !== 0
        if (hasMinMax && numCnt > 0) {
          const avg = (sum / numCnt).toFixed(2)
          aggSummary.value = `COUNT(${qAggCol.value.trim()}) = ${numCnt.toLocaleString()} • SUM = ${sum.toLocaleString()} • AVG = ${avg} • MIN = ${minV.toString()} • MAX = ${maxV.toString()}`
        } else {
          aggSummary.value = `COUNT(*) = ${cnt.toLocaleString()} (column '${qAggCol.value.trim()}' has 0 integer values)`
        }
      }
    }

    if (colsPtr || wherePtr) {
      const projPtr =
        wasm.vxg_project_frame_alloc(fPtr, fLen, colsPtr, wherePtr, 1, outLenPtr) >>> 0
      if (projPtr) {
        const projLen = new DataView(wasm.memory.buffer).getUint32(outLenPtr, true)
        splicedBytes.value = projLen
        freeBuf(wasm, projPtr, projLen)
      }
    }
    wasmMemoryBytes.value = wasm.memory.buffer.byteLength
  } catch (err: unknown) {
    queryError.value = err instanceof Error ? err.message : String(err)
  } finally {
    freeBuf(wasm, fPtr, fLen)
    freeBuf(wasm, colsPtr, colsLen)
    freeBuf(wasm, wherePtr, whereLen)
    freeBuf(wasm, orderPtr, orderLen)
    freeBuf(wasm, aggPtr, aggLen)
    freeBuf(wasm, outLenRaw, 16)
    freeBuf(wasm, statsRaw, 72)
  }
}

async function handleFileList(files: FileList | null) {
  if (!files || files.length === 0) return
  const file = files[0]
  const buf = new Uint8Array(await file.arrayBuffer())
  const isVxg =
    buf.byteLength >= 4 &&
    buf[0] === 0x56 &&
    buf[1] === 0x58 &&
    buf[2] === 0x47 &&
    buf[3] === 0x31
  activePreset.value = 'custom'
  uploadedName.value = file.name
  uploadedIsVxg.value = isVxg
  uploadedBinary.value = buf

  if (!isVxg && buf.byteLength <= 512 * 1024) {
    inputText.value = new TextDecoder('utf-8', { fatal: false }).decode(buf)
  }
  runPipeline()
}

function onTextEdited() {
  activePreset.value = 'custom'
  uploadedBinary.value = null
  uploadedName.value = ''
  uploadedIsVxg.value = false
  runPipeline()
}

function downloadBlob(bytes: Uint8Array | null, filename: string) {
  if (!bytes) return
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  const blob = new Blob([copy], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

const compressionFactor = computed(() => {
  if (!vxgBytes.value || vxgBytes.value.byteLength === 0 || rawBytesLen.value === 0) return '—'
  return `${(rawBytesLen.value / vxgBytes.value.byteLength).toFixed(2)}x`
})

const compressionRatioPct = computed(() => {
  if (!vxgBytes.value || rawBytesLen.value === 0) return '—'
  return `${((vxgBytes.value.byteLength / rawBytesLen.value) * 100).toFixed(2)}% of raw`
})

const decodedPreview = computed(() => {
  if (!decodedBytes.value) return ''
  const cap = Math.min(decodedBytes.value.byteLength, 2400)
  const txt = new TextDecoder('utf-8', { fatal: false }).decode(decodedBytes.value.subarray(0, cap))
  return decodedBytes.value.byteLength > cap
    ? `${txt}\n… (${(decodedBytes.value.byteLength - cap).toLocaleString()} more bytes)`
    : txt
})

onMounted(async () => {
  try {
    const wasmUrl = withBase('/vxg.wasm')
    const resp = await fetch(wasmUrl)
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${wasmUrl}`)
    const bytes = await resp.arrayBuffer()
    wasmBinaryBytes.value = bytes.byteLength
    const { instance } = await WebAssembly.instantiate(bytes, {})
    wasm = instance.exports as unknown as VxgExports
    wasmVersion.value = readCString(wasm, wasm.vxg_version_string())
    wasmMemoryBytes.value = wasm.memory.buffer.byteLength
    wasmState.value = 'ready'
    selectPreset('ndjson')
  } catch (err: unknown) {
    wasmState.value = 'error'
    wasmError.value = err instanceof Error ? err.message : String(err)
  }
})
</script>

<template>
  <div class="vxg-pg-shell">
    <!-- Engine Status Bar -->
    <div class="vxg-pg-status-bar">
      <div>
        <span
          class="vxg-pg-status-dot"
          :class="{
            'is-ready': wasmState === 'ready',
            'is-error': wasmState === 'error'
          }"
        />
        <strong v-if="wasmState === 'ready'">
          VXG WebAssembly Engine v{{ wasmVersion }} Active
        </strong>
        <strong v-else-if="wasmState === 'loading'">Loading vxg.wasm…</strong>
        <strong v-else>Failed to initialize vxg.wasm: {{ wasmError }}</strong>
      </div>
      <div v-if="wasmState === 'ready'" style="color: var(--vp-c-text-2)">
        <code>wasm32-unknown-unknown</code> • Zero WASI Imports • Binary:
        <strong>{{ formatBytes(wasmBinaryBytes) }}</strong> • Linear Heap:
        <strong>{{ formatBytes(wasmMemoryBytes) }}</strong>
      </div>
    </div>

    <!-- Preset & Controls Card -->
    <div class="vxg-pg-panel">
      <div class="vxg-controls-bar" style="margin-bottom: 0.4rem">
        <div>
          <span class="vxg-pg-label" style="display: block; margin-bottom: 0.3rem">
            1. Sample Workload / File Input
          </span>
          <div class="vxg-pill-group" role="tablist">
            <button
              type="button"
              class="vxg-pill"
              :class="{ 'is-active': activePreset === 'ndjson' }"
              @click="selectPreset('ndjson')"
            >
              NDJSON Telemetry (1.5k rows)
            </button>
            <button
              type="button"
              class="vxg-pill"
              :class="{ 'is-active': activePreset === 'csv' }"
              @click="selectPreset('csv')"
            >
              CSV Metrics (2k rows)
            </button>
            <button
              type="button"
              class="vxg-pill"
              :class="{ 'is-active': activePreset === 'clf' }"
              @click="selectPreset('clf')"
            >
              Apache CLF Logs
            </button>
            <button
              type="button"
              class="vxg-pill"
              :class="{ 'is-active': activePreset === 'prose' }"
              @click="selectPreset('prose')"
            >
              Markdown / Prose
            </button>
          </div>
        </div>

        <div>
          <span class="vxg-pg-label" style="display: block; margin-bottom: 0.3rem">
            Block Size &amp; Integrity
          </span>
          <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem">
            <div class="vxg-pill-group">
              <button
                v-for="bs in [4096, 16384, 65536, 262144]"
                :key="bs"
                type="button"
                class="vxg-pill"
                :class="{ 'is-active': blockSize === bs }"
                @click="blockSize = bs; runPipeline()"
              >
                {{ bs / 1024 }} KiB
              </button>
            </div>
            <button
              type="button"
              class="vxg-pg-btn"
              @click="checksumEnabled = !checksumEnabled; runPipeline()"
            >
              CRC32C: <strong>{{ checksumEnabled ? 'ON' : 'OFF' }}</strong>
            </button>
          </div>
        </div>
      </div>

      <!-- Drag & Drop Zone -->
      <div
        class="vxg-pg-dropzone"
        :class="{ 'is-drag': isDragging }"
        @dragover.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="isDragging = false; handleFileList($event.dataTransfer?.files ?? null)"
      >
        <div>
          <template v-if="uploadedName">
            Loaded <strong>{{ uploadedName }}</strong> ({{
              uploadedIsVxg ? 'Detected VXG1 archive → Decode & Query mode' : 'Raw file → Compress + Verify + Decode'
            }})
          </template>
          <template v-else>
            Drop any raw file (<code>.ndjson</code>, <code>.csv</code>, <code>.log</code>, <code>.json</code>) to compress — or drop a <code>.vxg</code> file to decode &amp; query in-browser.
          </template>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 0.45rem">
          <label class="vxg-pg-btn" style="margin: 0">
            Choose File…
            <input
              type="file"
              style="display: none"
              @change="handleFileList(($event.target as HTMLInputElement).files)"
            />
          </label>
          <button
            type="button"
            class="vxg-pg-btn is-primary"
            :disabled="!vxgBytes"
            @click="downloadBlob(vxgBytes, (uploadedName || activePreset) + '.vxg')"
          >
            Download .vxg ({{ vxgBytes ? formatBytes(vxgBytes.byteLength) : '0 B' }})
          </button>
          <button
            type="button"
            class="vxg-pg-btn"
            :disabled="!decodedBytes"
            @click="downloadBlob(decodedBytes, (uploadedName ? uploadedName.replace(/\.vxg$/i, '') : activePreset) + '.decoded.txt')"
          >
            Download Decoded
          </button>
        </div>
      </div>

      <div v-if="codecError" class="vxg-pg-error">{{ codecError }}</div>

      <!-- Live KPI Cards -->
      <div v-if="vxgBytes" class="vxg-kpi-grid" style="margin-top: 0.35rem; margin-bottom: 0">
        <div class="vxg-kpi-box">
          <div class="vxg-kpi-label">Space Savings</div>
          <div class="vxg-kpi-value">{{ compressionFactor }} smaller</div>
          <div class="vxg-kpi-delta">
            {{ formatBytes(rawBytesLen) }} → {{ formatBytes(vxgBytes.byteLength) }} ({{ compressionRatioPct }})
          </div>
        </div>
        <div class="vxg-kpi-box">
          <div class="vxg-kpi-label">WASM Encode (1T)</div>
          <div class="vxg-kpi-value">
            {{ uploadedIsVxg ? 'Uploaded .vxg' : `${encodeMs.toFixed(2)} ms` }}
          </div>
          <div class="vxg-kpi-delta">
            {{ uploadedIsVxg ? 'Direct frame inspection' : formatThroughput(rawBytesLen, encodeMs) }}
          </div>
        </div>
        <div class="vxg-kpi-box">
          <div class="vxg-kpi-label">WASM Decode (1T)</div>
          <div class="vxg-kpi-value">{{ decodeMs.toFixed(2) }} ms</div>
          <div class="vxg-kpi-delta">{{ formatThroughput(rawBytesLen, decodeMs) }}</div>
        </div>
        <div class="vxg-kpi-box">
          <div class="vxg-kpi-label">Frame Structure &amp; Proof</div>
          <div class="vxg-kpi-value">
            {{ blockCount }} {{ blockCount === 1 ? 'block' : 'blocks' }}
            <span v-if="rowCount > 0" style="font-size: 0.9rem; font-weight: 600">
              • {{ rowCount.toLocaleString() }} rows
            </span>
          </div>
          <div class="vxg-kpi-delta">
            {{ roundtripExact && verifyOk ? '✓ 100% Byte-Exact + CRC32C OK' : '✓ Verified' }}
          </div>
        </div>
      </div>
    </div>

    <!-- Side-by-Side Raw Input vs Decoded Output -->
    <div class="vxg-pg-grid-2">
      <div class="vxg-pg-panel">
        <div class="vxg-pg-panel-head">
          <span class="vxg-pg-panel-title">Raw Input Buffer (Editable)</span>
          <span class="vxg-badge">{{ formatBytes(rawBytesLen) }}</span>
        </div>
        <textarea
          v-model="inputText"
          class="vxg-pg-textarea"
          spellcheck="false"
          :disabled="uploadedIsVxg"
          placeholder="Paste NDJSON, CSV, logs, or arbitrary text here…"
          @input="onTextEdited"
        />
      </div>

      <div class="vxg-pg-panel">
        <div class="vxg-pg-panel-head">
          <span class="vxg-pg-panel-title">
            Restored Output from <code>vxg_decompress_into</code>
          </span>
          <span class="vxg-badge">
            {{ roundtripExact ? '✓ Byte-for-Byte Identical' : 'Ready' }}
          </span>
        </div>
        <pre class="vxg-pg-pre">{{ decodedPreview }}</pre>
      </div>
    </div>

    <!-- Per-Block Router Directory & O(1) Random Block Extraction -->
    <div v-if="blockSummaries.length > 0" class="vxg-pg-panel">
      <div class="vxg-pg-panel-head">
        <div>
          <div class="vxg-pg-panel-title">
            VXG1 v4 Block Directory &amp; O(1) Random Block Access (<code>vxg_extract_block_into</code>)
          </div>
          <div class="vxg-chart-subtitle">
            Click any independent block to decode ONLY that block in WASM linear memory without touching preceding blocks.
          </div>
        </div>
      </div>
      <div class="vxg-pg-block-list">
        <div v-for="blk in blockSummaries" :key="blk.index" class="vxg-pg-block-item">
          <div style="display: flex; justify-content: space-between; align-items: center">
            <strong>Block #{{ blk.index }}</strong>
            <span class="vxg-tag" style="margin: 0">{{ blk.packetLabel }}</span>
          </div>
          <div style="color: var(--vp-c-text-2)">
            {{ formatBytes(blk.rawSize) }} → <strong>{{ formatBytes(blk.compressedSize) }}</strong>
            ({{ blk.rawSize > 0 ? ((blk.compressedSize / blk.rawSize) * 100).toFixed(1) : '0' }}%)
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.15rem">
            <code style="font-size: 0.7rem">CRC: 0x{{ blk.crcHex }}</code>
            <button type="button" class="vxg-pg-btn" style="padding: 0.18rem 0.55rem; font-size: 0.72rem" @click="extractSingleBlock(blk.index)">
              Extract #{{ blk.index }}
            </button>
          </div>
        </div>
      </div>
      <pre v-if="extractedBlockIdx !== null" class="vxg-pg-pre" style="min-height: 110px; max-height: 180px">{{ extractedBlockPreview }}</pre>
    </div>

    <!-- Live Compressed-Domain SQL, Top-K & Column Splicing -->
    <div class="vxg-pg-panel">
      <div class="vxg-pg-panel-head">
        <div>
          <div class="vxg-pg-panel-title">
            2. Live Compressed-Domain SQL, Top-K &amp; Zero-Reencode Column Splicing
          </div>
          <div class="vxg-chart-subtitle">
            Executes <code>vxg_select_alloc</code>, <code>vxg_count_where_str</code>, <code>vxg_aggregate_col</code>, and <code>vxg_project_frame_alloc</code> directly on the compressed <code>.vxg</code> buffer above without full-frame decompression.
          </div>
        </div>
        <button type="button" class="vxg-pg-btn is-primary" :disabled="!vxgBytes || rowCount <= 0" @click="runQuery">
          Run Compressed Query
        </button>
      </div>

      <div class="vxg-pg-form-grid">
        <div class="vxg-pg-field">
          <label class="vxg-pg-label">SELECT Columns</label>
          <input v-model="qSelect" class="vxg-pg-input" placeholder="ts,service,latency_ms or csv:0,csv:2" @keyup.enter="runQuery" />
        </div>
        <div class="vxg-pg-field">
          <label class="vxg-pg-label">WHERE Predicate</label>
          <input v-model="qWhere" class="vxg-pg-input" placeholder='status==500 && latency_ms>=120' @keyup.enter="runQuery" />
        </div>
        <div class="vxg-pg-field">
          <label class="vxg-pg-label">ORDER BY (Top-K)</label>
          <div style="display: flex; gap: 0.35rem">
            <input v-model="qOrderBy" class="vxg-pg-input" placeholder="latency_ms" @keyup.enter="runQuery" />
            <button type="button" class="vxg-pg-btn" @click="qDesc = !qDesc; runQuery()">
              {{ qDesc ? 'DESC' : 'ASC' }}
            </button>
          </div>
        </div>
        <div class="vxg-pg-field">
          <label class="vxg-pg-label">LIMIT &amp; AGG Column</label>
          <div style="display: flex; gap: 0.35rem">
            <input v-model.number="qLimit" type="number" min="1" max="500" class="vxg-pg-input" style="max-width: 78px" @change="runQuery" />
            <input v-model="qAggCol" class="vxg-pg-input" placeholder="latency_ms" @keyup.enter="runQuery" />
          </div>
        </div>
      </div>

      <div v-if="queryError" class="vxg-pg-error">{{ queryError }}</div>

      <div v-if="matchCount !== null && !queryError" style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center">
        <span class="vxg-badge">
          Matched {{ matchCount.toLocaleString() }} / {{ rowCount.toLocaleString() }} rows in {{ queryMs.toFixed(2) }} ms
        </span>
        <span v-if="splicedBytes !== null && vxgBytes" class="vxg-badge">
          Spliced .vxg Frame: {{ formatBytes(splicedBytes) }} ({{ ((splicedBytes / vxgBytes.byteLength) * 100).toFixed(1) }}% of full .vxg)
        </span>
        <span v-if="aggSummary" style="font-size: 0.78rem; color: var(--vp-c-text-2); font-family: var(--vp-font-family-mono)">
          {{ aggSummary }}
        </span>
      </div>

      <pre class="vxg-pg-pre" style="min-height: 155px; max-height: 260px">{{ queryOutput }}</pre>
    </div>
  </div>
</template>
