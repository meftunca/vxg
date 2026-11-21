<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { withBase } from 'vitepress'

interface SampleItem {
  id: string
  label: string
  url: string | null
  bytes: Uint8Array | null
  width: number
  height: number
  tileSize: number
  tilesX: number
  tilesY: number
  dwtLevels: number
  totalBytes: number
  isRemote: boolean
}

interface LevelStepStat {
  level: number
  pct: number
  kib: number
  dims: string
  decMs: number
}

const BUILTIN_SAMPLES = [
  { id: 'landscape', label: 'Yosemite Landscape 3K (3200×2136, q=80 — Unsplash)', path: '/samples/landscape_3k_q80.vimg' },
  { id: 'portrait', label: 'Studio Portrait 11.3 MP (3000×3751, q=80 — Unsplash)', path: '/samples/portrait_3k_q80.vimg' },
]

const ready = ref(false)
const wasmSizeKiB = ref(0)
const statusMsg = ref('Initializing zero-WASI vimg.wasm (+simd128)...')
const samples = ref<SampleItem[]>([])
const activeId = ref('')
const encodeQuality = ref(80)
const encodeTile = ref(256)
const encodeAq = ref(true)
const encodeReport = ref('')
const encodeErr = ref(false)
const dragOver = ref(false)
const uploadOpen = ref(false)

// Compressed-domain controls
const targetW = ref(800)
const maxW = ref(3200)
const forceLevel = ref<number>(-1) // -1 = Auto, 0..3 = Explicit wavelet level
const blur = ref(0)
const sharpen = ref(0)
const brightness = ref(0)
const contrast = ref(1.0)
const saturation = ref(1.0)
const temperature = ref(0.0)
const grayscale = ref(false)
const sepia = ref(false)
const rot = ref(0)
const crop = ref<[number, number, number, number] | null>(null)

// Interactive view & progressive stream state
const zoomMode = ref<'fit' | '1:1'>('fit')
const streaming = ref(false)
const streamSteps = ref<LevelStepStat[]>([])
const hoverTile = ref<string>('')

// Live HUD metrics
const hud = ref({
  pct: 0,
  fetchedBytes: 0,
  totalBytes: 0,
  reqs: 0,
  lvl: 0,
  maxLvl: 3,
  tilesFetched: 0,
  tilesTotal: 0,
  outW: 0,
  outH: 0,
  totalMs: 0,
  decMs: 0,
  transport: 'HTTP/1.1 206 Range',
})

const navWrap = ref<HTMLDivElement | null>(null)
const navCanvas = ref<HTMLCanvasElement | null>(null)
const navGrid = ref<HTMLCanvasElement | null>(null)
const outCanvas = ref<HTMLCanvasElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

let wasm: any = null
const prefixCache = new Map<string, Uint8Array>()
let dragStart: { x0: number; y0: number } | null = null
let renderTimer: ReturnType<typeof setTimeout> | null = null
let streamAbort = 0

const activeItem = computed(() => samples.value.find(s => s.id === activeId.value))
const levelChoices = computed(() => {
  const maxL = activeItem.value?.dwtLevels ?? 3
  return Array.from({ length: maxL + 1 }, (_, i) => i)
})
const trafficMultiplier = computed(() => (hud.value.pct > 0.15 ? (100 / hud.value.pct).toFixed(1) + '×' : '—'))
const hasActiveFilters = computed(() =>
  blur.value > 0 ||
  sharpen.value > 0 ||
  brightness.value !== 0 ||
  Math.abs(contrast.value - 1.0) > 0.005 ||
  Math.abs(saturation.value - 1.0) > 0.005 ||
  Math.abs(temperature.value) > 0.005 ||
  grayscale.value ||
  sepia.value ||
  rot.value !== 0 ||
  forceLevel.value >= 0
)

function curSample(): SampleItem | undefined {
  return samples.value.find(s => s.id === activeId.value)
}

async function fetchRange(url: string, s: number, e: number): Promise<{ data: Uint8Array; is206: boolean }> {
  const res = await fetch(url, { headers: { Range: `bytes=${s}-${e - 1}` } })
  const buf = new Uint8Array(await res.arrayBuffer())
  if (res.status === 206) {
    return { data: buf, is206: true }
  }
  return { data: buf.subarray(s, e), is206: false }
}

async function loadPrefix(item: SampleItem): Promise<{ prefix: Uint8Array; reqs: number; fetched: number; is206: boolean }> {
  if (item.bytes) {
    const initLen = Math.min(item.bytes.length, 4096)
    const pPtr = wasm.vimg_wasm_alloc(initLen)
    new Uint8Array(wasm.memory.buffer, pPtr, initLen).set(item.bytes.subarray(0, initLen))
    wasm.vimg_wasm_parse_header(pPtr, initLen)
    const meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
    const need = meta[8]
    wasm.vimg_wasm_free(pPtr, initLen)
    const plen = Math.min(item.bytes.length, Math.max(initLen, need))
    return { prefix: item.bytes.subarray(0, plen), reqs: 1, fetched: plen, is206: false }
  }

  const cached = prefixCache.get(item.id)
  if (cached) {
    return { prefix: cached, reqs: 0, fetched: cached.length, is206: true }
  }

  const url = item.url!
  let reqs = 0
  let fetched = 0
  const r0 = await fetchRange(url, 0, 4096)
  reqs++
  fetched += r0.data.length
  const pPtr = wasm.vimg_wasm_alloc(r0.data.length)
  new Uint8Array(wasm.memory.buffer, pPtr, r0.data.length).set(r0.data)
  const st = wasm.vimg_wasm_parse_header(pPtr, r0.data.length)
  const meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
  const need = meta[8]
  const totalBytes = meta[9]
  wasm.vimg_wasm_free(pPtr, r0.data.length)
  if (st !== 0) throw new Error('Invalid VIMG1 header')

  item.width = meta[1]
  item.height = meta[2]
  item.tileSize = meta[10]
  item.tilesX = Math.ceil(item.width / item.tileSize)
  item.tilesY = Math.ceil(item.height / item.tileSize)
  item.dwtLevels = meta[11]
  item.totalBytes = totalBytes

  let prefix: Uint8Array
  if (need > r0.data.length) {
    const r1 = await fetchRange(url, r0.data.length, need)
    reqs++
    fetched += r1.data.length
    prefix = new Uint8Array(need)
    prefix.set(r0.data, 0)
    prefix.set(r1.data, r0.data.length)
  } else {
    prefix = r0.data
  }
  prefixCache.set(item.id, prefix)
  return { prefix, reqs, fetched, is206: r0.is206 }
}

async function decodeWasmQuery(
  item: SampleItem,
  opts: {
    crop: [number, number, number, number] | null
    tw: number
    gray: boolean
    sepia: boolean
    blur: number
    sharpen: number
    bright: number
    contrast: number
    saturation: number
    temperature: number
    rot: number
    forceLevel: number
  }
) {
  const t0 = performance.now()
  const { prefix, reqs: pReqs, fetched: pBytes, is206 } = await loadPrefix(item)
  let reqs = pReqs
  let fetchedBytes = pBytes

  const pPtr = wasm.vimg_wasm_alloc(prefix.length)
  new Uint8Array(wasm.memory.buffer, pPtr, prefix.length).set(prefix)
  const c = opts.crop || [0, 0, 0, 0]
  const stPlan = typeof wasm.vimg_wasm_plan_v3 === 'function'
    ? wasm.vimg_wasm_plan_v3(
        pPtr, prefix.length, c[0], c[1], c[2], c[3], opts.tw, 0,
        opts.gray ? 1 : 0, opts.sepia ? 1 : 0, opts.forceLevel
      )
    : wasm.vimg_wasm_plan(
        pPtr, prefix.length, c[0], c[1], c[2], c[3], opts.tw, 0,
        opts.gray ? 1 : 0, opts.sepia ? 1 : 0
      )
  if (stPlan !== 0) {
    wasm.vimg_wasm_free(pPtr, prefix.length)
    throw new Error('Range planning failed')
  }
  let meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
  const rPtr = meta[6]
  const rCap = meta[7]
  const nRanges = rCap / 8
  const rFlat = new Uint32Array(wasm.memory.buffer.slice(rPtr, rPtr + rCap))
  wasm.vimg_wasm_free(rPtr, rCap)

  const pairs: [number, number][] = []
  for (let i = 0; i < nRanges; i++) pairs.push([rFlat[i * 2], rFlat[i * 2 + 1]])

  let spans: Uint8Array[] = []
  let got206 = is206
  if (item.bytes) {
    spans = pairs.map(([s, e]) => item.bytes!.subarray(s, e))
    reqs += spans.length
  } else {
    const resList = await Promise.all(pairs.map(([s, e]) => fetchRange(item.url!, s, e)))
    spans = resList.map(r => r.data)
    got206 = resList.every(r => r.is206)
    reqs += spans.length
  }
  const totalSpanBytes = spans.reduce((a, b) => a + b.length, 0)
  fetchedBytes += totalSpanBytes

  const tDec0 = performance.now()
  const offsPtr = wasm.vimg_wasm_alloc(Math.max(4, nRanges * 4))
  const lensPtr = wasm.vimg_wasm_alloc(Math.max(4, nRanges * 4))
  const dataPtr = wasm.vimg_wasm_alloc(Math.max(1, totalSpanBytes))
  const offsArr = new Uint32Array(wasm.memory.buffer, offsPtr, nRanges)
  const lensArr = new Uint32Array(wasm.memory.buffer, lensPtr, nRanges)
  const dataArr = new Uint8Array(wasm.memory.buffer, dataPtr, totalSpanBytes)
  let curOff = 0
  for (let i = 0; i < nRanges; i++) {
    offsArr[i] = pairs[i][0]
    lensArr[i] = spans[i].length
    dataArr.set(spans[i], curOff)
    curOff += spans[i].length
  }

  const stDec = typeof wasm.vimg_wasm_decode_sparse_v3 === 'function'
    ? wasm.vimg_wasm_decode_sparse_v3(
        pPtr, prefix.length, offsPtr, lensPtr, nRanges, dataPtr, totalSpanBytes,
        c[0], c[1], c[2], c[3], opts.tw, 0,
        opts.gray ? 1 : 0, opts.sepia ? 1 : 0,
        opts.blur, opts.sharpen, opts.bright, opts.contrast,
        opts.saturation, opts.temperature, (opts.rot / 90) & 3, opts.forceLevel
      )
    : wasm.vimg_wasm_decode_sparse(
        pPtr, prefix.length, offsPtr, lensPtr, nRanges, dataPtr, totalSpanBytes,
        c[0], c[1], c[2], c[3], opts.tw, 0,
        opts.gray ? 1 : 0, opts.sepia ? 1 : 0,
        opts.blur, opts.sharpen, opts.bright, opts.contrast, (opts.rot / 90) & 3
      )
  wasm.vimg_wasm_free(pPtr, prefix.length)
  wasm.vimg_wasm_free(offsPtr, Math.max(4, nRanges * 4))
  wasm.vimg_wasm_free(lensPtr, Math.max(4, nRanges * 4))
  wasm.vimg_wasm_free(dataPtr, Math.max(1, totalSpanBytes))
  if (stDec !== 0) throw new Error('WASM decode error: ' + stDec)

  meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
  const ow = meta[1], oh = meta[2], lvl = meta[3], tf = meta[4], tt = meta[5]
  const outPtr = meta[6], outCap = meta[7]
  const rgba = new Uint8ClampedArray(wasm.memory.buffer.slice(outPtr, outPtr + ow * oh * 4))
  wasm.vimg_wasm_free(outPtr, outCap)
  const decMs = performance.now() - tDec0
  const totalMs = performance.now() - t0

  return { rgba, ow, oh, lvl, tf, tt, reqs, fetchedBytes, totalBytes: item.totalBytes, decMs, totalMs, got206 }
}

function drawNavGrid() {
  const item = curSample()
  const c = navGrid.value
  const bg = navCanvas.value
  if (!item || !c || !bg || !bg.clientWidth || !bg.clientHeight) return
  c.width = bg.clientWidth
  c.height = bg.clientHeight
  const ctx = c.getContext('2d')!
  ctx.clearRect(0, 0, c.width, c.height)
  const sx = c.width / item.width
  const sy = c.height / item.height
  const ts = item.tileSize
  const box = crop.value || [0, 0, item.width, item.height]
  const tx0 = Math.floor(box[0] / ts)
  const ty0 = Math.floor(box[1] / ts)
  const tx1 = Math.min(item.tilesX - 1, Math.floor((box[0] + box[2] - 1) / ts))
  const ty1 = Math.min(item.tilesY - 1, Math.floor((box[1] + box[3] - 1) / ts))

  for (let ty = 0; ty < item.tilesY; ty++) {
    for (let tx = 0; tx < item.tilesX; tx++) {
      const x = tx * ts * sx
      const y = ty * ts * sy
      const w = Math.min(ts, item.width - tx * ts) * sx
      const h = Math.min(ts, item.height - ty * ts) * sy
      const active = tx >= tx0 && tx <= tx1 && ty >= ty0 && ty <= ty1
      ctx.fillStyle = active ? 'rgba(16,185,129,0.14)' : 'rgba(4,7,14,0.62)'
      ctx.fillRect(x, y, w, h)
      ctx.strokeStyle = active ? 'rgba(52,211,153,0.72)' : 'rgba(148,163,184,0.18)'
      ctx.lineWidth = active ? 1.15 : 0.55
      ctx.strokeRect(x, y, w, h)
    }
  }
  if (crop.value) {
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 2
    ctx.strokeRect(box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
  }
}

function paintCanvas(rgba: Uint8ClampedArray, ow: number, oh: number) {
  const cv = outCanvas.value
  if (!cv) return
  cv.width = ow
  cv.height = oh
  cv.getContext('2d')!.putImageData(new ImageData(rgba, ow, oh), 0, 0)
}

function updateHud(item: SampleItem, r: Awaited<ReturnType<typeof decodeWasmQuery>>) {
  hud.value = {
    pct: (r.fetchedBytes * 100) / Math.max(1, r.totalBytes),
    fetchedBytes: r.fetchedBytes,
    totalBytes: r.totalBytes,
    reqs: r.reqs,
    lvl: r.lvl,
    maxLvl: item.dwtLevels,
    tilesFetched: r.tf,
    tilesTotal: r.tt,
    outW: r.ow,
    outH: r.oh,
    totalMs: r.totalMs,
    decMs: r.decMs,
    transport: item.bytes ? 'In-Browser RAM Range' : (r.got206 ? 'Live HTTP/1.1 206' : 'Static Slice Range'),
  }
}

async function renderMain() {
  const item = curSample()
  if (!item || !wasm) return
  try {
    const tw = forceLevel.value >= 0 ? 0 : (targetW.value || (crop.value ? 0 : Math.min(item.width, 1024)))
    const r = await decodeWasmQuery(item, {
      crop: crop.value,
      tw,
      gray: grayscale.value,
      sepia: sepia.value,
      blur: blur.value,
      sharpen: sharpen.value,
      bright: brightness.value,
      contrast: contrast.value,
      saturation: saturation.value,
      temperature: temperature.value,
      rot: rot.value,
      forceLevel: forceLevel.value,
    })
    paintCanvas(r.rgba, r.ow, r.oh)
    updateHud(item, r)
    statusMsg.value = `Decoded ${r.ow}×${r.oh} · L${r.lvl}/${item.dwtLevels} · ${r.tf}/${r.tt} tiles · ${r.decMs.toFixed(1)} ms`
  } catch (e: any) {
    statusMsg.value = `Error: ${e?.message || e}`
  }
}

async function runProgressiveStream() {
  const item = curSample()
  if (!item || !wasm) return
  const token = ++streamAbort
  streaming.value = true
  streamSteps.value = []
  const maxL = item.dwtLevels
  try {
    for (let lvl = 0; lvl <= maxL; lvl++) {
      if (token !== streamAbort) return
      forceLevel.value = lvl
      statusMsg.value = `Progressive HTTP 206 Stream · Fetching L${lvl}/${maxL}...`
      const r = await decodeWasmQuery(item, {
        crop: crop.value,
        tw: 0,
        gray: grayscale.value,
        sepia: sepia.value,
        blur: blur.value,
        sharpen: sharpen.value,
        bright: brightness.value,
        contrast: contrast.value,
        saturation: saturation.value,
        temperature: temperature.value,
        rot: rot.value,
        forceLevel: lvl,
      })
      if (token !== streamAbort) return
      paintCanvas(r.rgba, r.ow, r.oh)
      updateHud(item, r)
      streamSteps.value.push({
        level: lvl,
        pct: (r.fetchedBytes * 100) / Math.max(1, r.totalBytes),
        kib: r.fetchedBytes / 1024,
        dims: `${r.ow}×${r.oh}`,
        decMs: r.decMs,
      })
      if (lvl < maxL) {
        await new Promise(resolve => setTimeout(resolve, 420))
      }
    }
    statusMsg.value = `Progressive L0 → L${maxL} complete (${hud.value.outW}×${hud.value.outH} crisp full resolution)`
  } catch (e: any) {
    statusMsg.value = `Stream error: ${e?.message || e}`
  } finally {
    if (token === streamAbort) streaming.value = false
  }
}

function selectLevel(lvl: number) {
  streamAbort++
  streaming.value = false
  forceLevel.value = lvl
  scheduleRender()
}

function scheduleRender() {
  streamAbort++
  streaming.value = false
  drawNavGrid()
  if (renderTimer) clearTimeout(renderTimer)
  renderTimer = setTimeout(() => renderMain(), 30)
}

function resetFilters() {
  streamAbort++
  streaming.value = false
  forceLevel.value = -1
  blur.value = 0
  sharpen.value = 0
  brightness.value = 0
  contrast.value = 1.0
  saturation.value = 1.0
  temperature.value = 0.0
  grayscale.value = false
  sepia.value = false
  rot.value = 0
  scheduleRender()
}

function clearCrop() {
  crop.value = null
  const item = curSample()
  if (item && targetW.value === 0) {
    targetW.value = Math.min(item.width, Math.max(256, Math.round(item.width / 4)))
  }
  scheduleRender()
}

async function selectSample(id: string) {
  streamAbort++
  streaming.value = false
  streamSteps.value = []
  activeId.value = id
  const item = curSample()
  if (!item || !wasm) return
  crop.value = null
  forceLevel.value = -1
  blur.value = 0
  sharpen.value = 0
  brightness.value = 0
  contrast.value = 1.0
  saturation.value = 1.0
  temperature.value = 0.0
  grayscale.value = false
  sepia.value = false
  rot.value = 0
  statusMsg.value = `Fetching VIMG1 header & L0 preview for ${item.label}...`
  try {
    await loadPrefix(item)
    maxW.value = item.width
    targetW.value = Math.min(item.width, Math.max(256, Math.round(item.width / 4)))
    const thumb = await decodeWasmQuery(item, {
      crop: null, tw: Math.min(340, item.width), gray: false, sepia: false,
      blur: 0, sharpen: 0, bright: 0, contrast: 1, saturation: 1, temperature: 0, rot: 0, forceLevel: -1,
    })
    if (navCanvas.value) {
      navCanvas.value.width = thumb.ow
      navCanvas.value.height = thumb.oh
      navCanvas.value.getContext('2d')!.putImageData(new ImageData(thumb.rgba, thumb.ow, thumb.oh), 0, 0)
    }
    requestAnimationFrame(() => {
      drawNavGrid()
      renderMain()
    })
  } catch (e: any) {
    statusMsg.value = `Load failed: ${e?.message || e}`
  }
}

async function handleFile(file: File) {
  if (!wasm) return
  encodeErr.value = false
  encodeReport.value = `Encoding ${file.name} (${(file.size / 1024).toFixed(1)} KiB) in browser vimg.wasm...`
  try {
    const rawBuf = new Uint8Array(await file.arrayBuffer())
    let vimgBytes: Uint8Array
    let w = 0, h = 0, ch = 3, encMs = 0

    if (rawBuf.length >= 48 && rawBuf[0] === 86 && rawBuf[1] === 73 && rawBuf[2] === 77 && rawBuf[3] === 71 && rawBuf[4] === 49) {
      vimgBytes = rawBuf
      const pPtr = wasm.vimg_wasm_alloc(48)
      new Uint8Array(wasm.memory.buffer, pPtr, 48).set(rawBuf.subarray(0, 48))
      wasm.vimg_wasm_parse_header(pPtr, 48)
      const meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
      w = meta[1]; h = meta[2]
      wasm.vimg_wasm_free(pPtr, 48)
    } else {
      const bmp = await createImageBitmap(file)
      w = bmp.width; h = bmp.height
      const cv = document.createElement('canvas')
      cv.width = w; cv.height = h
      const ctx = cv.getContext('2d')!
      ctx.drawImage(bmp, 0, 0)
      bmp.close()
      const rgba = ctx.getImageData(0, 0, w, h).data
      const t0 = performance.now()
      const inPtr = wasm.vimg_wasm_alloc(rgba.length)
      new Uint8Array(wasm.memory.buffer, inPtr, rgba.length).set(rgba)
      const st = typeof wasm.vimg_wasm_encode_rgba_v3 === 'function'
        ? wasm.vimg_wasm_encode_rgba_v3(inPtr, w, h, encodeQuality.value, encodeTile.value, encodeAq.value ? 1 : 0)
        : wasm.vimg_wasm_encode_rgba(inPtr, w, h, encodeQuality.value, encodeTile.value)
      wasm.vimg_wasm_free(inPtr, rgba.length)
      if (st !== 0) throw new Error('WASM encode failed: ' + st)
      const meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
      ch = meta[3]
      const outPtr = meta[6], outLen = meta[7]
      vimgBytes = new Uint8Array(wasm.memory.buffer.slice(outPtr, outPtr + outLen))
      wasm.vimg_wasm_free(outPtr, outLen)
      encMs = performance.now() - t0
    }

    const pPtr = wasm.vimg_wasm_alloc(48)
    new Uint8Array(wasm.memory.buffer, pPtr, 48).set(vimgBytes.subarray(0, 48))
    wasm.vimg_wasm_parse_header(pPtr, 48)
    const meta = new Uint32Array(wasm.memory.buffer, wasm.vimg_wasm_meta_ptr(), 12)
    wasm.vimg_wasm_free(pPtr, 48)

    const stem = file.name.replace(/\.[^.]+$/, '')
    const suffix = encodeQuality.value >= 100 ? 'lossless' : `q${encodeQuality.value}${encodeAq.value ? '_aq' : ''}`
    const outName = file.name.endsWith('.vimg') ? file.name : `${stem}_${suffix}.vimg`
    const id = 'up_' + Date.now()
    const item: SampleItem = {
      id,
      label: `${outName} (${w}×${h}, ${(vimgBytes.length / 1024).toFixed(0)} KiB)`,
      url: null,
      bytes: vimgBytes,
      width: w,
      height: h,
      tileSize: meta[10],
      tilesX: Math.ceil(w / meta[10]),
      tilesY: Math.ceil(h / meta[10]),
      dwtLevels: meta[11],
      totalBytes: vimgBytes.length,
      isRemote: false,
    }
    samples.value.unshift(item)
    const bpp = (vimgBytes.length * 8) / Math.max(1, w * h)
    const diffPct = (1 - vimgBytes.length / Math.max(1, file.size)) * 100
    const diffStr = diffPct >= 0 ? `-${diffPct.toFixed(1)}%` : `+${(-diffPct).toFixed(1)}%`
    encodeReport.value = encMs > 0
      ? `✓ ${outName} (${w}×${h}×${ch}) · ${(file.size / 1024).toFixed(1)} → ${(vimgBytes.length / 1024).toFixed(1)} KiB (${diffStr}, ${bpp.toFixed(2)} bpp) in ${encMs.toFixed(0)} ms`
      : `✓ Loaded ${outName} (${w}×${h}, ${(vimgBytes.length / 1024).toFixed(1)} KiB)`
    await selectSample(id)
  } catch (e: any) {
    encodeErr.value = true
    encodeReport.value = `✗ ${e?.message || e}`
  }
}

async function downloadCurrentVimg() {
  const item = curSample()
  if (!item) return
  let blob: Blob
  if (item.bytes) {
    blob = new Blob([item.bytes as BlobPart], { type: 'application/octet-stream' })
  } else {
    const res = await fetch(item.url!)
    blob = await res.blob()
  }
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = item.label.split(' ')[0].endsWith('.vimg') ? item.label.split(' ')[0] : `${item.id}.vimg`
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
}

function downloadRenderedPng() {
  const cv = outCanvas.value
  const item = curSample()
  if (!cv || !item) return
  cv.toBlob(blob => {
    if (!blob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${item.id}_${hud.value.outW}x${hud.value.outH}.png`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 2000)
  }, 'image/png')
}

function applyPreset(k: string) {
  const item = curSample()
  if (!item) return
  streamAbort++
  streaming.value = false
  forceLevel.value = -1
  grayscale.value = false
  sepia.value = false
  blur.value = 0
  sharpen.value = 0
  brightness.value = 0
  contrast.value = 1.0
  saturation.value = 1.0
  temperature.value = 0.0
  if (k === 'full') { crop.value = null; targetW.value = Math.min(item.width, 1280) }
  else if (k === 'quarter') { crop.value = null; targetW.value = Math.max(64, Math.round(item.width / 4)) }
  else if (k === 'crop256') { const s = Math.min(256, item.width, item.height); crop.value = [Math.round((item.width - s) / 2), Math.round((item.height - s) / 2), s, s]; targetW.value = 0 }
  else if (k === 'crop512') { const s = Math.min(512, item.width, item.height); crop.value = [Math.round((item.width - s) / 2), Math.round((item.height - s) / 2), s, s]; targetW.value = 0 }
  else if (k === 'grayblur') { crop.value = null; targetW.value = Math.max(64, Math.round(item.width / 2)); grayscale.value = true; blur.value = 0.6 }
  else if (k === 'cinema') { crop.value = null; targetW.value = Math.max(64, Math.round(item.width / 2)); contrast.value = 1.18; saturation.value = 1.25; temperature.value = 0.35; sharpen.value = 0.4 }
  scheduleRender()
}

function onPointerDown(e: PointerEvent) {
  const item = curSample()
  const el = navWrap.value
  if (!item || !el) return
  const r = el.getBoundingClientRect()
  dragStart = { x0: (e.clientX - r.left) / r.width, y0: (e.clientY - r.top) / r.height }
  el.setPointerCapture(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  const item = curSample()
  const el = navWrap.value
  if (!item || !el) return
  const r = el.getBoundingClientRect()
  const nx = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
  const ny = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))
  const tx = Math.min(item.tilesX - 1, Math.floor((nx * item.width) / item.tileSize))
  const ty = Math.min(item.tilesY - 1, Math.floor((ny * item.height) / item.tileSize))
  hoverTile.value = `Tile (${tx},${ty})`
  if (!dragStart) return
  const x = Math.round(Math.min(dragStart.x0, nx) * item.width)
  const y = Math.round(Math.min(dragStart.y0, ny) * item.height)
  const w = Math.max(32, Math.round(Math.abs(nx - dragStart.x0) * item.width))
  const h = Math.max(32, Math.round(Math.abs(ny - dragStart.y0) * item.height))
  crop.value = [x, y, Math.min(w, item.width - x), Math.min(h, item.height - y)]
  forceLevel.value = -1
  targetW.value = 0
  drawNavGrid()
}

function onPointerUp() {
  if (dragStart) {
    dragStart = null
    scheduleRender()
  }
}

onMounted(async () => {
  try {
    const res = await fetch(withBase('/vimg.wasm'))
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching vimg.wasm`)
    const wasmBytes = await res.arrayBuffer()
    wasmSizeKiB.value = Math.round(wasmBytes.byteLength / 1024)
    const { instance } = await WebAssembly.instantiate(wasmBytes, {})
    wasm = instance.exports
    samples.value = BUILTIN_SAMPLES.map(s => ({
      id: s.id,
      label: s.label,
      url: withBase(s.path),
      bytes: null,
      width: 0,
      height: 0,
      tileSize: 256,
      tilesX: 1,
      tilesY: 1,
      dwtLevels: 3,
      totalBytes: 0,
      isRemote: true,
    }))
    ready.value = true
    await selectSample(samples.value[0].id)
  } catch (e: any) {
    statusMsg.value = `Failed to initialize vimg.wasm: ${e?.message || e}`
  }
})

onUnmounted(() => {
  streamAbort++
  if (renderTimer) clearTimeout(renderTimer)
})
</script>

<template>
  <div class="vimg-pg">
    <!-- Top Studio Header Bar -->
    <header class="vimg-top">
      <div class="vimg-top-left">
        <span class="vimg-badge">VIMG v0.4</span>
        <span class="vimg-simd-pill">WASM SIMD128 · {{ wasmSizeKiB || '—' }} KiB</span>
        <select class="vimg-sample-sel" v-model="activeId" @change="selectSample(activeId)" :disabled="!ready">
          <option v-for="s in samples" :key="s.id" :value="s.id">{{ s.label }}</option>
        </select>
      </div>
      <div class="vimg-top-right">
        <button
          type="button"
          class="vimg-btn-stream"
          :class="{ busy: streaming }"
          @click="runProgressiveStream"
          :disabled="!ready || streaming"
          title="Animate real HTTP 206 progressive wavelet streaming from L0 thumbnail to L3 full resolution"
        >
          <span class="dot" :class="{ pulse: streaming }"></span>
          {{ streaming ? 'Streaming L0 → L3...' : '▶ Progressive L0 → L3' }}
        </button>
        <button type="button" class="vimg-btn-ghost" @click="uploadOpen = !uploadOpen" :disabled="!ready" :class="{ active: uploadOpen }">
          ⬆ Encode Image
        </button>
        <button type="button" class="vimg-btn-ghost" @click="downloadRenderedPng" :disabled="!ready" title="Save rendered output as PNG">
          PNG
        </button>
        <button type="button" class="vimg-btn-ghost" @click="downloadCurrentVimg" :disabled="!ready" title="Download source .vimg archive">
          ⬇ .vimg
        </button>
      </div>
    </header>

    <!-- Collapsible Client-Side WASM Upload & Encode Drawer -->
    <div v-if="uploadOpen" class="vimg-upload-drawer">
      <input
        ref="fileInput"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,.vimg"
        style="display:none"
        @change="e => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); (e.target as HTMLInputElement).value = '' }"
      />
      <div
        class="vimg-drop"
        :class="{ over: dragOver }"
        @click="fileInput?.click()"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="e => { dragOver = false; const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f) }"
      >
        <b>+ Drop PNG / JPEG / WebP / .vimg or Click to Encode in Browser</b>
        <span>100% client-side in {{ wasmSizeKiB || '—' }} KiB zero-WASI WASM · Auto Spatial Palette for lossless UI screenshots</span>
      </div>
      <div class="vimg-upload-opts">
        <label>
          <span>Quality</span>
          <select v-model.number="encodeQuality">
            <option :value="75">Lossy q=75</option>
            <option :value="80">Lossy q=80 (WebP-class)</option>
            <option :value="85">Lossy q=85 (High)</option>
            <option :value="100">Lossless q=100</option>
          </select>
        </label>
        <label>
          <span>Tile Size</span>
          <select v-model.number="encodeTile">
            <option :value="128">128×128</option>
            <option :value="256">256×256</option>
          </select>
        </label>
        <label class="vimg-aq-toggle" title="Tile-Adaptive Psychovisual Quantization (--aq): boosts flat/skin tiles & trims dense foliage">
          <input type="checkbox" v-model="encodeAq" :disabled="encodeQuality >= 100" />
          <span>Adaptive Quant (<code>--aq</code>)</span>
        </label>
      </div>
      <div v-if="encodeReport" class="vimg-log" :class="{ err: encodeErr }">{{ encodeReport }}</div>
    </div>

    <!-- Main 2-Column Studio Workspace -->
    <div class="vimg-grid">
      <aside class="vimg-side">
        <!-- Interactive Tile Navigator -->
        <section class="vimg-card">
          <div class="vimg-ch">
            <span>Tile Navigator</span>
            <div class="vimg-ch-right">
              <span v-if="crop" class="vimg-crop-tag">{{ crop[2] }}×{{ crop[3] }}</span>
              <button v-if="crop" type="button" class="vimg-mini-btn" @click="clearCrop">Clear</button>
              <span v-else class="vimg-hint">{{ hoverTile || 'Drag to Crop' }}</span>
            </div>
          </div>
          <div
            ref="navWrap"
            class="vimg-nav"
            @pointerdown="onPointerDown"
            @pointermove="onPointerMove"
            @pointerup="onPointerUp"
            @pointerleave="hoverTile = ''"
          >
            <canvas ref="navCanvas" class="vimg-nav-bg"></canvas>
            <canvas ref="navGrid" class="vimg-nav-ov"></canvas>
          </div>
          <div class="vimg-presets">
            <button type="button" @click="applyPreset('full')">Full</button>
            <button type="button" @click="applyPreset('quarter')">1/4 Pyramid</button>
            <button type="button" @click="applyPreset('crop256')">256² Crop</button>
            <button type="button" @click="applyPreset('crop512')">512² Crop</button>
            <button type="button" @click="applyPreset('grayblur')">Gray+Blur</button>
            <button type="button" @click="applyPreset('cinema')">Warm Grade</button>
          </div>
        </section>

        <!-- Pyramid Level & Resolution Selector -->
        <section class="vimg-card">
          <div class="vimg-ch">
            <span>Wavelet Pyramid &amp; Scale</span>
            <span class="vimg-mono">{{ forceLevel >= 0 ? `Locked L${forceLevel}` : `Auto (L${hud.lvl})` }}</span>
          </div>
          <div class="vimg-lvl-pills">
            <button type="button" :class="{ active: forceLevel === -1 }" @click="selectLevel(-1)">Auto</button>
            <button
              v-for="l in levelChoices"
              :key="l"
              type="button"
              :class="{ active: forceLevel === l, current: forceLevel === -1 && hud.lvl === l }"
              @click="selectLevel(l)"
            >
              L{{ l }}
            </button>
          </div>
          <div class="vimg-ctrl" style="margin-top:10px; margin-bottom:0">
            <label>
              <span>Target Width <em v-if="forceLevel >= 0">(unlock Auto to resize)</em></span>
              <b>{{ targetW && forceLevel < 0 ? targetW + 'px' : 'Native' }}</b>
            </label>
            <input
              type="range"
              :min="0"
              :max="maxW"
              step="64"
              v-model.number="targetW"
              :disabled="forceLevel >= 0"
              @input="scheduleRender"
            />
          </div>
        </section>

        <!-- Compact 2-Column Compressed-Domain Controls -->
        <section class="vimg-card">
          <div class="vimg-ch">
            <span>Compressed-Domain DSP</span>
            <button v-if="hasActiveFilters" type="button" class="vimg-mini-btn" @click="resetFilters">Reset</button>
          </div>

          <div class="vimg-ctrl-grid">
            <div class="vimg-ctrl">
              <label><span>Subband Blur</span><b>{{ blur.toFixed(2) }}</b></label>
              <input type="range" min="0" max="1" step="0.05" v-model.number="blur" @input="scheduleRender" />
            </div>
            <div class="vimg-ctrl">
              <label><span>Sharpen</span><b>{{ sharpen.toFixed(2) }}</b></label>
              <input type="range" min="0" max="2" step="0.1" v-model.number="sharpen" @input="scheduleRender" />
            </div>
            <div class="vimg-ctrl">
              <label><span>Brightness</span><b>{{ brightness > 0 ? '+' + brightness : brightness }}</b></label>
              <input type="range" min="-80" max="80" step="4" v-model.number="brightness" @input="scheduleRender" />
            </div>
            <div class="vimg-ctrl">
              <label><span>Contrast</span><b>{{ contrast.toFixed(2) }}×</b></label>
              <input type="range" min="0.4" max="2.0" step="0.05" v-model.number="contrast" @input="scheduleRender" />
            </div>
            <div class="vimg-ctrl">
              <label><span>Saturation</span><b>{{ saturation.toFixed(2) }}×</b></label>
              <input type="range" min="0.0" max="2.0" step="0.05" v-model.number="saturation" :disabled="grayscale || sepia" @input="scheduleRender" />
            </div>
            <div class="vimg-ctrl">
              <label><span>Temp (Co/Cg)</span><b>{{ temperature > 0 ? '+' + temperature.toFixed(2) : temperature.toFixed(2) }}</b></label>
              <input type="range" min="-1.0" max="1.0" step="0.05" v-model.number="temperature" :disabled="grayscale || sepia" @input="scheduleRender" />
            </div>
          </div>

          <div class="vimg-toggles">
            <label :class="{ on: grayscale }">
              <input type="checkbox" v-model="grayscale" @change="scheduleRender" />
              <span>Grayscale <small>(0 chroma)</small></span>
            </label>
            <label :class="{ on: sepia }">
              <input type="checkbox" v-model="sepia" @change="scheduleRender" />
              <span>Sepia <small>(0 chroma)</small></span>
            </label>
          </div>

          <div class="vimg-rots">
            <span>Rotate</span>
            <div class="vimg-rot-grp">
              <button v-for="r in [0, 90, 180, 270]" :key="r" type="button" :class="{ active: rot === r }" @click="rot = r; scheduleRender()">
                {{ r }}°
              </button>
            </div>
          </div>
        </section>
      </aside>

      <!-- Main Preview & Telemetry Column -->
      <div class="vimg-main">
        <!-- Compact Telemetry HUD Strip -->
        <div class="vimg-hud">
          <div class="vimg-m accent-green">
            <div class="k">Wire Fetched</div>
            <div class="v good">{{ hud.pct.toFixed(1) }}%</div>
            <div class="s">{{ (hud.fetchedBytes / 1024).toFixed(1) }} / {{ (hud.totalBytes / 1024).toFixed(1) }} KiB</div>
          </div>
          <div class="vimg-m accent-cyan">
            <div class="k">Bandwidth Saved</div>
            <div class="v cyan">{{ Math.max(0, 100 - hud.pct).toFixed(1) }}%</div>
            <div class="s">{{ trafficMultiplier }} less · {{ hud.reqs }} reqs</div>
          </div>
          <div class="vimg-m">
            <div class="k">Pyramid Tier</div>
            <div class="v">L{{ hud.lvl }} <span class="dim">/ L{{ hud.maxLvl }}</span></div>
            <div class="s">{{ hud.outW }}×{{ hud.outH }} px</div>
          </div>
          <div class="vimg-m">
            <div class="k">Tiles Read</div>
            <div class="v">{{ hud.tilesFetched }} <span class="dim">/ {{ hud.tilesTotal }}</span></div>
            <div class="s">{{ hud.transport }}</div>
          </div>
          <div class="vimg-m">
            <div class="k">WASM SIMD128</div>
            <div class="v">{{ hud.decMs.toFixed(1) }} <span class="unit">ms</span></div>
            <div class="s">e2e {{ hud.totalMs.toFixed(1) }} ms</div>
          </div>
        </div>

        <!-- Wire Savings Gauge + Progressive Stream Timeline -->
        <div class="vimg-bar">
          <div class="vimg-bar-h">
            <span>HTTP 206 Byte Transfer vs 100% Full-File Download</span>
            <span class="vimg-bar-ratio">{{ hud.pct > 0.15 ? `${trafficMultiplier} smaller on the wire` : '—' }}</span>
          </div>
          <div class="vimg-bar-t">
            <div class="vimg-bar-f" :class="{ streaming }" :style="{ width: Math.min(100, hud.pct) + '%' }"></div>
          </div>
          <div v-if="streamSteps.length" class="vimg-stream-strip">
            <button
              v-for="st in streamSteps"
              :key="st.level"
              type="button"
              class="vimg-step-chip"
              :class="{ active: hud.lvl === st.level }"
              @click="selectLevel(st.level)"
            >
              <b>L{{ st.level }}</b> {{ st.dims }} · {{ st.pct.toFixed(1) }}% ({{ st.kib.toFixed(0) }} KiB) · {{ st.decMs.toFixed(1) }}ms
            </button>
          </div>
        </div>

        <!-- Interactive Image Stage -->
        <div class="vimg-stage" :class="{ 'zoom-native': zoomMode === '1:1' }">
          <div class="vimg-stage-bar">
            <span class="vimg-stage-pill">{{ hud.outW }}×{{ hud.outH }} · L{{ hud.lvl }} · {{ hud.decMs.toFixed(1) }} ms</span>
            <div class="vimg-zoom-grp">
              <button type="button" :class="{ active: zoomMode === 'fit' }" @click="zoomMode = 'fit'">Fit</button>
              <button type="button" :class="{ active: zoomMode === '1:1' }" @click="zoomMode = '1:1'">1:1 px</button>
            </div>
          </div>
          <div class="vimg-canvas-wrap">
            <canvas ref="outCanvas"></canvas>
          </div>
        </div>

        <div class="vimg-status">
          <span class="vimg-status-dot" :class="{ ok: ready }"></span>
          <span>{{ statusMsg }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.vimg-pg {
  margin: 20px 0;
  border: 1px solid #1f2b3e;
  border-radius: 16px;
  background: radial-gradient(circle at top right, #111c2e 0%, #090d16 58%);
  color: #e6edf7;
  overflow: hidden;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
}

.vimg-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border-bottom: 1px solid #1e2a3d;
  background: rgba(15, 22, 36, 0.92);
  backdrop-filter: blur(8px);
  flex-wrap: wrap;
}
.vimg-top-left, .vimg-top-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.vimg-badge {
  background: linear-gradient(135deg, #10b981, #06b6d4);
  color: #031512;
  font-weight: 800;
  font-size: 11px;
  padding: 4px 9px;
  border-radius: 6px;
  letter-spacing: 0.05em;
}
.vimg-simd-pill {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px;
  color: #6ee7b7;
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 3px 8px;
  border-radius: 99px;
}
.vimg-sample-sel {
  max-width: 310px;
  text-overflow: ellipsis;
}

.vimg-pg select, .vimg-pg button {
  background: #141e30;
  color: #e6edf7;
  border: 1px solid #283852;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease, color 140ms ease;
}
.vimg-pg select:hover, .vimg-pg button:hover:not(:disabled) {
  border-color: #38bdf8;
}
.vimg-pg button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.vimg-pg button.active {
  background: #0d2e25;
  border-color: #10b981;
  color: #6ee7b7;
}

.vimg-btn-stream {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(6, 182, 212, 0.22)) !important;
  border-color: #10b981 !important;
  color: #a7f3d0 !important;
  font-weight: 600;
}
.vimg-btn-stream:hover:not(:disabled) {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.34), rgba(6, 182, 212, 0.34)) !important;
}
.vimg-btn-stream .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #34d399;
}
.vimg-btn-stream .dot.pulse {
  animation: vimg-pulse 0.8s ease-in-out infinite;
}
@keyframes vimg-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.45); opacity: 0.45; }
}

.vimg-upload-drawer {
  padding: 12px 16px;
  background: #0d1422;
  border-bottom: 1px solid #1e2a3d;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.vimg-drop {
  border: 1.5px dashed #334155;
  border-radius: 10px;
  padding: 12px 14px;
  text-align: center;
  background: rgba(23, 32, 49, 0.65);
  cursor: pointer;
  font-size: 11.5px;
  color: #94a3b8;
  transition: border-color 140ms ease, background 140ms ease;
}
.vimg-drop:hover, .vimg-drop.over {
  border-color: #10b981;
  background: rgba(15, 36, 32, 0.85);
  color: #e6edf7;
}
.vimg-drop b {
  display: block;
  color: #6ee7b7;
  font-size: 12.5px;
  margin-bottom: 3px;
}
.vimg-upload-opts {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  font-size: 11.5px;
  color: #94a3b8;
}
.vimg-upload-opts label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.vimg-aq-toggle {
  background: #141e30;
  border: 1px solid #283852;
  padding: 5px 9px;
  border-radius: 8px;
  color: #cbd5e1;
  cursor: pointer;
}
.vimg-aq-toggle code {
  color: #38bdf8;
  font-size: 11px;
}
.vimg-log {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  color: #34d399;
  background: #081814;
  border: 1px solid #164e3f;
  border-radius: 8px;
  padding: 7px 10px;
}
.vimg-log.err {
  color: #fca5a5;
  background: #260d11;
  border-color: #7f1d1d;
}

.vimg-grid {
  display: grid;
  grid-template-columns: 332px 1fr;
  gap: 14px;
  padding: 14px;
}
@media (max-width: 960px) {
  .vimg-grid {
    grid-template-columns: 1fr;
  }
}

.vimg-side {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.vimg-card {
  background: rgba(15, 22, 36, 0.88);
  border: 1px solid #1f2c42;
  border-radius: 12px;
  padding: 12px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
.vimg-ch {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #94a3b8;
  margin-bottom: 9px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 700;
}
.vimg-ch-right {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.vimg-hint {
  color: #34d399;
  font-family: ui-monospace, monospace;
  text-transform: none;
  letter-spacing: 0;
}
.vimg-crop-tag {
  font-family: ui-monospace, monospace;
  color: #38bdf8;
  font-size: 10.5px;
  text-transform: none;
}
.vimg-mono {
  font-family: ui-monospace, monospace;
  color: #cbd5e1;
  text-transform: none;
  letter-spacing: 0;
}
.vimg-mini-btn {
  padding: 2px 7px !important;
  font-size: 10.5px !important;
  border-radius: 6px !important;
  text-transform: none;
}

.vimg-nav {
  position: relative;
  width: 100%;
  background: #06090f;
  border: 1px solid #223047;
  border-radius: 8px;
  overflow: hidden;
  cursor: crosshair;
  user-select: none;
}
.vimg-nav-bg {
  display: block;
  width: 100%;
  height: auto;
}
.vimg-nav-ov {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.vimg-presets {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
  margin-top: 8px;
}
.vimg-presets button {
  font-size: 11px;
  padding: 5px 6px;
  text-align: center;
}

.vimg-lvl-pills {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 5px;
}
.vimg-lvl-pills button {
  padding: 5px 4px;
  font-size: 11px;
  font-family: ui-monospace, monospace;
  font-weight: 600;
}
.vimg-lvl-pills button.current {
  border-color: rgba(56, 189, 248, 0.55);
  color: #7dd3fc;
}

.vimg-ctrl-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 10px;
}
.vimg-ctrl {
  margin-bottom: 2px;
}
.vimg-ctrl label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 10.5px;
  color: #94a3b8;
  margin-bottom: 3px;
}
.vimg-ctrl label em {
  font-style: normal;
  color: #64748b;
  font-size: 10px;
}
.vimg-ctrl label b {
  color: #e6edf7;
  font-family: ui-monospace, monospace;
  font-size: 10.5px;
}
.vimg-ctrl input[type="range"] {
  width: 100%;
  height: 5px;
  accent-color: #10b981;
  cursor: pointer;
}
.vimg-ctrl input[type="range"]:disabled {
  opacity: 0.38;
  cursor: not-allowed;
}

.vimg-toggles {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-top: 10px;
  font-size: 11px;
}
.vimg-toggles label {
  display: flex;
  align-items: center;
  gap: 6px;
  background: #141e30;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid #24334b;
  cursor: pointer;
}
.vimg-toggles label.on {
  border-color: #10b981;
  background: #0d2b23;
  color: #a7f3d0;
}
.vimg-toggles small {
  color: #64748b;
  font-size: 9.5px;
}

.vimg-rots {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-top: 9px;
  font-size: 11px;
  color: #94a3b8;
}
.vimg-rot-grp {
  display: flex;
  gap: 4px;
}
.vimg-rot-grp button {
  padding: 4px 8px;
  font-size: 11px;
  font-family: ui-monospace, monospace;
}

.vimg-main {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.vimg-hud {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
  gap: 8px;
}
.vimg-m {
  background: rgba(15, 22, 36, 0.88);
  border: 1px solid #1f2c42;
  border-radius: 10px;
  padding: 9px 11px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
.vimg-m.accent-green {
  border-color: rgba(16, 185, 129, 0.32);
}
.vimg-m.accent-cyan {
  border-color: rgba(56, 189, 248, 0.3);
}
.vimg-m .k {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #8fa3bf;
}
.vimg-m .v {
  font-size: 19px;
  font-weight: 700;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-top: 2px;
  line-height: 1.2;
}
.vimg-m .v .dim {
  font-size: 12px;
  font-weight: 500;
  color: #64748b;
}
.vimg-m .v .unit {
  font-size: 12px;
  font-weight: 500;
  color: #94a3b8;
}
.vimg-m .v.good { color: #34d399; }
.vimg-m .v.cyan { color: #38bdf8; }
.vimg-m .s {
  font-size: 10.5px;
  color: #8fa3bf;
  font-family: ui-monospace, monospace;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.vimg-bar {
  background: rgba(15, 22, 36, 0.88);
  border: 1px solid #1f2c42;
  border-radius: 10px;
  padding: 10px 12px;
}
.vimg-bar-h {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  margin-bottom: 6px;
  color: #cbd5e1;
  gap: 8px;
}
.vimg-bar-ratio {
  font-family: ui-monospace, monospace;
  color: #34d399;
  font-weight: 600;
}
.vimg-bar-t {
  height: 8px;
  background: #1a2436;
  border-radius: 99px;
  overflow: hidden;
}
.vimg-bar-f {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #06b6d4, #38bdf8);
  border-radius: 99px;
  transition: width 200ms cubic-bezier(0.22, 1, 0.36, 1);
}
.vimg-bar-f.streaming {
  box-shadow: 0 0 12px rgba(52, 211, 153, 0.65);
}

.vimg-stream-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 9px;
}
.vimg-step-chip {
  font-family: ui-monospace, monospace;
  font-size: 10.5px !important;
  padding: 3px 8px !important;
  border-radius: 99px !important;
  background: #131d2e !important;
  color: #cbd5e1 !important;
}
.vimg-step-chip.active {
  background: #0d2e25 !important;
  border-color: #10b981 !important;
  color: #6ee7b7 !important;
}

.vimg-stage {
  position: relative;
  flex: 1;
  min-height: 410px;
  background: repeating-conic-gradient(#0f1522 0% 25%, #090d16 0% 50%) 50% / 16px 16px;
  border: 1px solid #1f2c42;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.vimg-stage-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: rgba(9, 13, 22, 0.72);
  border-bottom: 1px solid rgba(31, 44, 66, 0.7);
}
.vimg-stage-pill {
  font-family: ui-monospace, monospace;
  font-size: 11px;
  color: #93c5fd;
}
.vimg-zoom-grp {
  display: flex;
  gap: 4px;
}
.vimg-zoom-grp button {
  padding: 3px 8px;
  font-size: 10.5px;
}
.vimg-canvas-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px;
  overflow: auto;
}
.vimg-stage canvas {
  max-width: 100%;
  max-height: 62vh;
  border-radius: 6px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.55);
}
.vimg-stage.zoom-native .vimg-canvas-wrap {
  align-items: flex-start;
  justify-content: flex-start;
}
.vimg-stage.zoom-native canvas {
  max-width: none;
  max-height: none;
  image-rendering: pixelated;
}

.vimg-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  color: #93c5fd;
  background: rgba(15, 22, 36, 0.88);
  border: 1px solid #1f2c42;
  border-radius: 9px;
  padding: 7px 11px;
}
.vimg-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #f59e0b;
  flex-shrink: 0;
}
.vimg-status-dot.ok {
  background: #10b981;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
}
</style>
