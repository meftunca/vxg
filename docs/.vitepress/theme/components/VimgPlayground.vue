<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
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

const BUILTIN_SAMPLES = [
  { id: 'landscape', label: 'Yosemite Landscape 3K (3200×2136, q=80 — Unsplash)', path: '/samples/landscape_3k_q80.vimg' },
  { id: 'portrait', label: 'Studio Portrait 11.3 MP (3000×3751, q=80 — Unsplash)', path: '/samples/portrait_3k_q80.vimg' },
]

const ready = ref(false)
const statusMsg = ref('Loading zero-WASI vimg.wasm (144 KiB)...')
const samples = ref<SampleItem[]>([])
const activeId = ref('')
const encodeQuality = ref(80)
const encodeTile = ref(256)
const encodeReport = ref('')
const encodeErr = ref(false)
const dragOver = ref(false)

// Compressed-domain controls
const targetW = ref(800)
const maxW = ref(3200)
const blur = ref(0)
const sharpen = ref(0)
const brightness = ref(0)
const contrast = ref(1.0)
const grayscale = ref(false)
const sepia = ref(false)
const rot = ref(0)
const crop = ref<[number, number, number, number] | null>(null)

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
let prefixCache = new Map<string, Uint8Array>()
let dragStart: { x0: number; y0: number } | null = null
let renderTimer: any = null

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
    rot: number
  }
) {
  const t0 = performance.now()
  const { prefix, reqs: pReqs, fetched: pBytes, is206 } = await loadPrefix(item)
  let reqs = pReqs
  let fetchedBytes = pBytes

  const pPtr = wasm.vimg_wasm_alloc(prefix.length)
  new Uint8Array(wasm.memory.buffer, pPtr, prefix.length).set(prefix)
  const c = opts.crop || [0, 0, 0, 0]
  const stPlan = wasm.vimg_wasm_plan(
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

  const stDec = wasm.vimg_wasm_decode_sparse(
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
      ctx.fillStyle = active ? 'rgba(16,185,129,0.16)' : 'rgba(5,8,14,0.58)'
      ctx.fillRect(x, y, w, h)
      ctx.strokeStyle = active ? 'rgba(52,211,153,0.75)' : 'rgba(148,163,184,0.2)'
      ctx.lineWidth = active ? 1.2 : 0.6
      ctx.strokeRect(x, y, w, h)
    }
  }
  if (crop.value) {
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 2
    ctx.strokeRect(box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
  }
}

async function renderMain() {
  const item = curSample()
  if (!item || !wasm) return
  try {
    const tw = targetW.value || (crop.value ? 0 : Math.min(item.width, 1024))
    const r = await decodeWasmQuery(item, {
      crop: crop.value,
      tw,
      gray: grayscale.value,
      sepia: sepia.value,
      blur: blur.value,
      sharpen: sharpen.value,
      bright: brightness.value,
      contrast: contrast.value,
      rot: rot.value,
    })
    const cv = outCanvas.value
    if (cv) {
      cv.width = r.ow
      cv.height = r.oh
      cv.getContext('2d')!.putImageData(new ImageData(r.rgba, r.ow, r.oh), 0, 0)
    }
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
      transport: item.bytes ? 'In-Browser RAM Range' : (r.got206 ? 'Live HTTP/1.1 206 Range' : 'Static Slice Range'),
    }
    statusMsg.value = `Decoded ${r.ow}×${r.oh} (L${r.lvl}, ${r.tf}/${r.tt} tiles) via vimg.wasm`
  } catch (e: any) {
    statusMsg.value = `Error: ${e?.message || e}`
  }
}

function scheduleRender() {
  drawNavGrid()
  clearTimeout(renderTimer)
  renderTimer = setTimeout(() => renderMain(), 35)
}

async function selectSample(id: string) {
  activeId.value = id
  const item = curSample()
  if (!item || !wasm) return
  crop.value = null
  blur.value = 0
  sharpen.value = 0
  brightness.value = 0
  contrast.value = 1.0
  grayscale.value = false
  sepia.value = false
  rot.value = 0
  statusMsg.value = `Fetching VIMG1 header & L0 preview for ${item.label}...`
  try {
    await loadPrefix(item)
    maxW.value = item.width
    targetW.value = Math.min(item.width, Math.max(256, Math.round(item.width / 4)))
    // Render crisp L0 thumbnail into Navigator canvas
    const thumb = await decodeWasmQuery(item, {
      crop: null, tw: Math.min(340, item.width), gray: false, sepia: false,
      blur: 0, sharpen: 0, bright: 0, contrast: 1, rot: 0,
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
      const st = wasm.vimg_wasm_encode_rgba(inPtr, w, h, encodeQuality.value, encodeTile.value)
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
    const suffix = encodeQuality.value >= 100 ? 'lossless' : `q${encodeQuality.value}`
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
      ? `✓ ${outName} (${w}×${h}×${ch}) · ${(file.size / 1024).toFixed(1)} KiB → ${(vimgBytes.length / 1024).toFixed(1)} KiB (${diffStr}, ${bpp.toFixed(2)} bpp) in ${encMs.toFixed(1)} ms (client WASM)`
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

function applyPreset(k: string) {
  const item = curSample()
  if (!item) return
  grayscale.value = false; sepia.value = false; blur.value = 0; sharpen.value = 0; brightness.value = 0; contrast.value = 1.0
  if (k === 'full') { crop.value = null; targetW.value = Math.min(item.width, 1280) }
  else if (k === 'quarter') { crop.value = null; targetW.value = Math.max(64, Math.round(item.width / 4)) }
  else if (k === 'crop256') { const s = Math.min(256, item.width, item.height); crop.value = [Math.round((item.width - s) / 2), Math.round((item.height - s) / 2), s, s]; targetW.value = 0 }
  else if (k === 'crop512') { const s = Math.min(512, item.width, item.height); crop.value = [Math.round((item.width - s) / 2), Math.round((item.height - s) / 2), s, s]; targetW.value = 0 }
  else if (k === 'grayblur') { crop.value = null; targetW.value = Math.max(64, Math.round(item.width / 2)); grayscale.value = true; blur.value = 0.6 }
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
  if (!dragStart || !item || !el) return
  const r = el.getBoundingClientRect()
  const x1 = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
  const y1 = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))
  const x = Math.round(Math.min(dragStart.x0, x1) * item.width)
  const y = Math.round(Math.min(dragStart.y0, y1) * item.height)
  const w = Math.max(32, Math.round(Math.abs(x1 - dragStart.x0) * item.width))
  const h = Math.max(32, Math.round(Math.abs(y1 - dragStart.y0) * item.height))
  crop.value = [x, y, Math.min(w, item.width - x), Math.min(h, item.height - y)]
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
    const { instance } = await WebAssembly.instantiateStreaming(res, {})
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

onUnmounted(() => clearTimeout(renderTimer))
</script>

<template>
  <div class="vimg-pg">
    <div class="vimg-top">
      <div class="vimg-top-left">
        <span class="vimg-badge">VIMG1 · WASM</span>
        <select v-model="activeId" @change="selectSample(activeId)" :disabled="!ready">
          <option v-for="s in samples" :key="s.id" :value="s.id">{{ s.label }}</option>
        </select>
      </div>
      <div class="vimg-top-right">
        <button type="button" @click="downloadCurrentVimg" :disabled="!ready">⬇ Download .vimg</button>
      </div>
    </div>

    <div class="vimg-grid">
      <aside class="vimg-side">
        <div class="vimg-card">
          <div class="vimg-ch">Upload &amp; Convert (100% Client WASM) <span>PNG · JPEG · WebP</span></div>
          <input ref="fileInput" type="file" accept="image/png,image/jpeg,image/webp,image/bmp,.vimg" style="display:none" @change="e => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); (e.target as HTMLInputElement).value = '' }" />
          <div
            class="vimg-drop"
            :class="{ over: dragOver }"
            @click="fileInput?.click()"
            @dragover.prevent="dragOver = true"
            @dragleave="dragOver = false"
            @drop.prevent="e => { dragOver = false; const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f) }"
          >
            <b>+ Drop Image or Click to Encode</b>
            Zero uploads — encodes straight in 144 KiB vimg.wasm
          </div>
          <div class="vimg-row">
            <select v-model.number="encodeQuality">
              <option :value="75">Lossy q=75</option>
              <option :value="80">Lossy q=80 (WebP-class)</option>
              <option :value="85">Lossy q=85 (High)</option>
              <option :value="100">Lossless q=100</option>
            </select>
            <select v-model.number="encodeTile">
              <option :value="128">Tile 128</option>
              <option :value="256">Tile 256</option>
            </select>
          </div>
          <div v-if="encodeReport" class="vimg-log" :class="{ err: encodeErr }">{{ encodeReport }}</div>
        </div>

        <div class="vimg-card">
          <div class="vimg-ch">Tile Navigator <span style="color:#10b981">Drag to Crop</span></div>
          <div
            ref="navWrap"
            class="vimg-nav"
            @pointerdown="onPointerDown"
            @pointermove="onPointerMove"
            @pointerup="onPointerUp"
          >
            <canvas ref="navCanvas" class="vimg-nav-bg"></canvas>
            <canvas ref="navGrid" class="vimg-nav-ov"></canvas>
          </div>
          <div class="vimg-presets">
            <button type="button" @click="applyPreset('full')">Full</button>
            <button type="button" @click="applyPreset('quarter')">1/4 Pyramid</button>
            <button type="button" @click="applyPreset('crop256')">256² Crop</button>
            <button type="button" @click="applyPreset('crop512')">512² Crop</button>
            <button type="button" @click="applyPreset('grayblur')">1/2 Gray+Blur</button>
          </div>
        </div>

        <div class="vimg-card">
          <div class="vimg-ch">Compressed-Domain Controls</div>
          <div class="vimg-ctrl">
            <label>Target Width <b>{{ targetW ? targetW + 'px' : 'Auto (1:1)' }}</b></label>
            <input type="range" :min="0" :max="maxW" step="64" v-model.number="targetW" @input="scheduleRender" />
          </div>
          <div class="vimg-ctrl">
            <label>Subband Blur <b>{{ blur.toFixed(2) }}</b></label>
            <input type="range" min="0" max="1" step="0.05" v-model.number="blur" @input="scheduleRender" />
          </div>
          <div class="vimg-ctrl">
            <label>Subband Sharpen <b>{{ sharpen.toFixed(2) }}</b></label>
            <input type="range" min="0" max="2" step="0.1" v-model.number="sharpen" @input="scheduleRender" />
          </div>
          <div class="vimg-ctrl">
            <label>LL Brightness <b>{{ brightness }}</b></label>
            <input type="range" min="-80" max="80" step="4" v-model.number="brightness" @input="scheduleRender" />
          </div>
          <div class="vimg-ctrl">
            <label>Contrast <b>{{ contrast.toFixed(2) }}</b></label>
            <input type="range" min="0.4" max="2.0" step="0.05" v-model.number="contrast" @input="scheduleRender" />
          </div>
          <div class="vimg-toggles">
            <label><input type="checkbox" v-model="grayscale" @change="scheduleRender" /> Grayscale (0 chroma)</label>
            <label><input type="checkbox" v-model="sepia" @change="scheduleRender" /> Sepia (0 chroma)</label>
          </div>
          <div class="vimg-rots">
            <span>Rotate:</span>
            <button v-for="r in [0, 90, 180, 270]" :key="r" type="button" :class="{ active: rot === r }" @click="rot = r; scheduleRender()">{{ r }}°</button>
          </div>
        </div>
      </aside>

      <div class="vimg-main">
        <div class="vimg-hud">
          <div class="vimg-m"><div class="k">Wire Fetched</div><div class="v good">{{ hud.pct.toFixed(1) }}%</div><div class="s">{{ (hud.fetchedBytes/1024).toFixed(1) }} / {{ (hud.totalBytes/1024).toFixed(1) }} KiB</div></div>
          <div class="vimg-m"><div class="k">Saved vs Full</div><div class="v cyan">{{ Math.max(0, 100 - hud.pct).toFixed(1) }}%</div><div class="s">{{ hud.reqs }} Range reqs</div></div>
          <div class="vimg-m"><div class="k">Pyramid Level</div><div class="v">L{{ hud.lvl }} / L{{ hud.maxLvl }}</div><div class="s">{{ hud.outW }}×{{ hud.outH }}</div></div>
          <div class="vimg-m"><div class="k">Tiles Fetched</div><div class="v">{{ hud.tilesFetched }} / {{ hud.tilesTotal }}</div><div class="s">{{ hud.transport }}</div></div>
          <div class="vimg-m"><div class="k">WASM Decode</div><div class="v">{{ hud.decMs.toFixed(1) }} ms</div><div class="s">total {{ hud.totalMs.toFixed(1) }} ms</div></div>
        </div>

        <div class="vimg-bar">
          <div class="vimg-bar-h">
            <span>Transferred vs 100% Full-Image Download</span>
            <span>{{ hud.pct > 0.1 ? (100 / hud.pct).toFixed(1) + '× less traffic' : '—' }}</span>
          </div>
          <div class="vimg-bar-t"><div class="vimg-bar-f" :style="{ width: Math.min(100, hud.pct) + '%' }"></div></div>
        </div>

        <div class="vimg-stage">
          <canvas ref="outCanvas"></canvas>
        </div>

        <div class="vimg-status">{{ statusMsg }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.vimg-pg { margin: 18px 0; border: 1px solid var(--vp-c-divider); border-radius: 14px; background: #0b0f17; color: #e6edf7; overflow: hidden; }
.vimg-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid #223047; background: #111723; flex-wrap: wrap; }
.vimg-top-left, .vimg-top-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.vimg-badge { background: linear-gradient(135deg, #10b981, #06b6d4); color: #041312; font-weight: 800; font-size: 11px; padding: 4px 8px; border-radius: 6px; letter-spacing: .05em; }
.vimg-pg select, .vimg-pg button { background: #172031; color: #e6edf7; border: 1px solid #2b3b55; border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
.vimg-pg button:hover { border-color: #38bdf8; }
.vimg-pg button.active { background: #0f2e26; border-color: #10b981; color: #6ee7b7; }
.vimg-grid { display: grid; grid-template-columns: 310px 1fr; gap: 14px; padding: 14px; }
@media (max-width: 900px) { .vimg-grid { grid-template-columns: 1fr; } }
.vimg-side { display: flex; flex-direction: column; gap: 12px; }
.vimg-card { background: #111723; border: 1px solid #223047; border-radius: 11px; padding: 12px; }
.vimg-ch { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8; margin-bottom: 8px; display: flex; justify-content: space-between; font-weight: 700; }
.vimg-drop { border: 1.5px dashed #334155; border-radius: 9px; padding: 10px; text-align: center; background: #172031; cursor: pointer; font-size: 11px; color: #94a3b8; }
.vimg-drop:hover, .vimg-drop.over { border-color: #10b981; background: #0f2420; color: #e6edf7; }
.vimg-drop b { display: block; color: #6ee7b7; font-size: 12px; margin-bottom: 2px; }
.vimg-row { display: flex; gap: 6px; margin-top: 8px; }
.vimg-row select { flex: 1; font-size: 11px; padding: 5px 6px; }
.vimg-log { margin-top: 8px; font-family: ui-monospace, monospace; font-size: 11px; color: #34d399; background: #091713; border: 1px solid #164e3f; border-radius: 7px; padding: 6px 8px; line-height: 1.4; }
.vimg-log.err { color: #fca5a5; background: #260d11; border-color: #7f1d1d; }
.vimg-nav { position: relative; width: 100%; background: #06090f; border: 1px solid #223047; border-radius: 8px; overflow: hidden; cursor: crosshair; user-select: none; }
.vimg-nav-bg { display: block; width: 100%; height: auto; }
.vimg-nav-ov { position: absolute; inset: 0; width: 100%; height: 100%; }
.vimg-presets { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.vimg-presets button { font-size: 11px; padding: 4px 7px; }
.vimg-ctrl { margin-bottom: 8px; }
.vimg-ctrl label { display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 3px; }
.vimg-ctrl label b { color: #e6edf7; font-family: ui-monospace, monospace; }
.vimg-ctrl input[type=range] { width: 100%; accent-color: #10b981; }
.vimg-toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; font-size: 11px; }
.vimg-toggles label { display: flex; align-items: center; gap: 5px; background: #172031; padding: 5px 7px; border-radius: 7px; border: 1px solid #223047; cursor: pointer; }
.vimg-rots { display: flex; gap: 5px; align-items: center; margin-top: 8px; font-size: 11px; color: #94a3b8; }
.vimg-rots button { padding: 4px 7px; font-size: 11px; }
.vimg-main { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.vimg-hud { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; }
.vimg-m { background: #111723; border: 1px solid #223047; border-radius: 9px; padding: 8px 10px; }
.vimg-m .k { font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: #94a3b8; }
.vimg-m .v { font-size: 18px; font-weight: 700; font-family: ui-monospace, monospace; margin-top: 2px; }
.vimg-m .v.good { color: #34d399; }
.vimg-m .v.cyan { color: #38bdf8; }
.vimg-m .s { font-size: 10px; color: #94a3b8; font-family: ui-monospace, monospace; }
.vimg-bar { background: #111723; border: 1px solid #223047; border-radius: 9px; padding: 9px 12px; }
.vimg-bar-h { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 5px; color: #cbd5e1; }
.vimg-bar-t { height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; }
.vimg-bar-f { height: 100%; background: linear-gradient(90deg, #10b981, #38bdf8); transition: width 160ms ease; }
.vimg-stage { flex: 1; min-height: 360px; background: repeating-conic-gradient(#101520 0% 25%, #0b0f17 0% 50%) 50% / 16px 16px; border: 1px solid #223047; border-radius: 11px; display: flex; align-items: center; justify-content: center; padding: 14px; overflow: auto; }
.vimg-stage canvas { max-width: 100%; max-height: 65vh; border-radius: 6px; box-shadow: 0 10px 32px rgba(0,0,0,.55); }
.vimg-status { font-family: ui-monospace, monospace; font-size: 11px; color: #93c5fd; background: #111723; border: 1px solid #223047; border-radius: 8px; padding: 7px 10px; }
</style>
