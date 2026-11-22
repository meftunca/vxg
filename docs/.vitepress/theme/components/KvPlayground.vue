<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'

type StorageBackend = 'indexeddb' | 'localstorage'

interface LogEntry {
  id: number
  pod: string
  cmd: string
  reply: string
  ok: boolean
  ms: number
  note?: string
}

interface PodStats {
  pod_id: string
  epoch: number
  dbsize: number
  mem_dirty: number
  wal_overlay: number
  active_wals: number
  history: number
  base_epoch: number
  base_keys: number
  base_raw_bytes: number
  base_compressed_bytes: number
  base_blocks: number
  range_requests: number
  range_bytes: number
  bloom_filtered: number
  cache_blocks: number
  cache_hits: number
}

interface StoreObj {
  key: string
  bytes: number
  etag: string
}

interface WasmExports {
  memory: WebAssembly.Memory
  vxg_kv_alloc: (len: number) => number
  vxg_kv_free: (ptr: number, len: number) => void
  vxg_kv_set_now_ms: (ms: number) => void
  vxg_kv_init: () => void
  vxg_kv_reset: () => void
  vxg_kv_reopen_pods: () => void
  vxg_kv_store_put: (kp: number, kl: number, vp: number, vl: number) => number
  vxg_kv_store_get: (kp: number, kl: number, outLenPtr: number) => number
  vxg_kv_store_list_json: (outLenPtr: number) => number
  vxg_kv_exec: (podIdx: number, cp: number, cl: number, outLenPtr: number) => number
  vxg_kv_pod_stats_json: (podIdx: number, outLenPtr: number) => number
}

const IDB_NAME = 'vxg_kv_playground_v1'
const IDB_STORE = 'objects'
const LS_PREFIX = 'vxg_kv_obj:'

const enc = new TextEncoder()
const dec = new TextDecoder()

const ready = ref(false)
const loadingMsg = ref('Loading vxg_kv.wasm (zero-WASI pure Rust)…')
const errorMsg = ref('')
const activePod = ref<0 | 1>(0)
const backend = ref<StorageBackend>('indexeddb')
const cmdInput = ref('')
const logs = ref<LogEntry[]>([])
const cmdHistory = ref<string[]>([])
const histIdx = ref(-1)
const stats = ref<PodStats | null>(null)
const storeObjects = ref<StoreObj[]>([])
const previewTitle = ref('')
const previewText = ref('')
const persistBanner = ref('')
const termEl = ref<HTMLElement | null>(null)

let wasm: WasmExports | null = null
let outLenCell = 0
let logSeq = 1

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`
}

function toBase64(bytes: Uint8Array): string {
  let s = ''
  const step = 0x8000
  for (let i = 0; i < bytes.length; i += step) {
    s += String.fromCharCode(...bytes.subarray(i, i + step))
  }
  return btoa(s)
}

function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

// ---- IndexedDB & localStorage persistence drivers ----
function openIdb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function idbLoadAll(): Promise<Array<[string, Uint8Array]>> {
  const db = await openIdb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly')
    const st = tx.objectStore(IDB_STORE)
    const keysReq = st.getAllKeys()
    const valsReq = st.getAll()
    tx.oncomplete = () => {
      const keys = keysReq.result as string[]
      const vals = valsReq.result as Uint8Array[]
      db.close()
      resolve(keys.map((k, i) => [k, vals[i] instanceof Uint8Array ? vals[i] : new Uint8Array(vals[i])]))
    }
    tx.onerror = () => {
      db.close()
      reject(tx.error)
    }
  })
}

async function idbSaveAll(entries: Array<[string, Uint8Array]>): Promise<void> {
  const db = await openIdb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite')
    const st = tx.objectStore(IDB_STORE)
    st.clear()
    for (const [k, v] of entries) st.put(v, k)
    tx.oncomplete = () => {
      db.close()
      resolve()
    }
    tx.onerror = () => {
      db.close()
      reject(tx.error)
    }
  })
}

function lsLoadAll(): Array<[string, Uint8Array]> {
  const out: Array<[string, Uint8Array]> = []
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)
    if (k && k.startsWith(LS_PREFIX)) {
      const raw = localStorage.getItem(k)
      if (raw !== null) {
        out.push([k.slice(LS_PREFIX.length), fromBase64(raw)])
      }
    }
  }
  return out
}

function lsSaveAll(entries: Array<[string, Uint8Array]>): void {
  const toDel: string[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)
    if (k && k.startsWith(LS_PREFIX)) toDel.push(k)
  }
  for (const k of toDel) localStorage.removeItem(k)
  for (const [k, v] of entries) {
    localStorage.setItem(LS_PREFIX + k, toBase64(v))
  }
}

// ---- WASM memory helpers ----
function readReturnedBytes(ptr: number): Uint8Array {
  if (!wasm || !ptr) return new Uint8Array(0)
  const len = new DataView(wasm.memory.buffer).getUint32(outLenCell, true)
  const copy = new Uint8Array(wasm.memory.buffer, ptr, len).slice()
  wasm.vxg_kv_free(ptr, len)
  return copy
}

function allocWrite(bytes: Uint8Array): number {
  const ptr = wasm!.vxg_kv_alloc(bytes.length || 1)
  if (bytes.length > 0) {
    new Uint8Array(wasm!.memory.buffer, ptr, bytes.length).set(bytes)
  }
  return ptr
}

function wasmGetObject(key: string): Uint8Array {
  const kb = enc.encode(key)
  const kp = allocWrite(kb)
  const ptr = wasm!.vxg_kv_store_get(kp, kb.length, outLenCell)
  wasm!.vxg_kv_free(kp, kb.length || 1)
  return readReturnedBytes(ptr)
}

function wasmPutObject(key: string, data: Uint8Array) {
  const kb = enc.encode(key)
  const kp = allocWrite(kb)
  const vp = allocWrite(data)
  wasm!.vxg_kv_store_put(kp, kb.length, vp, data.length)
  wasm!.vxg_kv_free(kp, kb.length || 1)
  wasm!.vxg_kv_free(vp, data.length || 1)
}

function refreshState() {
  if (!wasm) return
  wasm.vxg_kv_set_now_ms(Date.now())
  const sp = wasm.vxg_kv_pod_stats_json(activePod.value, outLenCell)
  stats.value = JSON.parse(dec.decode(readReturnedBytes(sp)))
  const lp = wasm.vxg_kv_store_list_json(outLenCell)
  storeObjects.value = JSON.parse(dec.decode(readReturnedBytes(lp)))
}

async function persistToBrowserStorage() {
  if (!wasm) return
  refreshState()
  const entries: Array<[string, Uint8Array]> = storeObjects.value.map((o) => [
    o.key,
    wasmGetObject(o.key),
  ])
  const totalBytes = entries.reduce((acc, [, v]) => acc + v.length, 0)
  try {
    if (backend.value === 'indexeddb') {
      await idbSaveAll(entries)
      persistBanner.value = `Persisted ${entries.length} object(s) (${fmtBytes(totalBytes)}) to IndexedDB`
    } else {
      lsSaveAll(entries)
      persistBanner.value = `Persisted ${entries.length} object(s) (${fmtBytes(totalBytes)}) to localStorage`
    }
  } catch (e: unknown) {
    persistBanner.value = `Storage write error: ${String(e)}`
  }
}

async function hydrateFromBrowserStorage() {
  if (!wasm) return
  wasm.vxg_kv_reset()
  const entries = backend.value === 'indexeddb' ? await idbLoadAll() : lsLoadAll()
  for (const [k, v] of entries) {
    wasmPutObject(k, v)
  }
  wasm.vxg_kv_reopen_pods()
  refreshState()
  if (entries.length > 0) {
    const bytes = entries.reduce((a, [, v]) => a + v.length, 0)
    persistBanner.value = `Restored ${entries.length} object(s) (${fmtBytes(bytes)}) from ${backend.value === 'indexeddb' ? 'IndexedDB' : 'localStorage'} — reload-safe!`
  } else {
    persistBanner.value = `${backend.value === 'indexeddb' ? 'IndexedDB' : 'localStorage'} is empty — run a preset or type a Redis command below.`
  }
}

function execRaw(podIdx: 0 | 1, cmd: string): { ok: boolean; reply: string; mutated_store: boolean; ms: number } {
  wasm!.vxg_kv_set_now_ms(Date.now())
  const cb = enc.encode(cmd)
  const cp = allocWrite(cb)
  const t0 = performance.now()
  const ptr = wasm!.vxg_kv_exec(podIdx, cp, cb.length, outLenCell)
  const ms = performance.now() - t0
  wasm!.vxg_kv_free(cp, cb.length || 1)
  const res = JSON.parse(dec.decode(readReturnedBytes(ptr)))
  return { ...res, ms }
}

async function scrollTerm() {
  await nextTick()
  if (termEl.value) termEl.value.scrollTop = termEl.value.scrollHeight
}

async function runCommand(cmdStr?: string, silentPersist = false) {
  if (!wasm) return
  const line = (cmdStr ?? cmdInput.value).trim()
  if (!line) return
  if (cmdStr === undefined) {
    cmdInput.value = ''
    cmdHistory.value.unshift(line)
    histIdx.value = -1
  }
  const res = execRaw(activePod.value, line)
  logs.value.push({
    id: logSeq++,
    pod: activePod.value === 0 ? 'pod-a' : 'pod-b',
    cmd: line,
    reply: res.reply,
    ok: res.ok,
    ms: res.ms,
  })
  if (res.mutated_store && !silentPersist) {
    await persistToBrowserStorage()
  } else {
    refreshState()
  }
  await scrollTerm()
}

function onKeyDown(e: KeyboardEvent) {
  if (e.key === 'ArrowUp') {
    if (cmdHistory.value.length > 0 && histIdx.value + 1 < cmdHistory.value.length) {
      histIdx.value++
      cmdInput.value = cmdHistory.value[histIdx.value]
      e.preventDefault()
    }
  } else if (e.key === 'ArrowDown') {
    if (histIdx.value > 0) {
      histIdx.value--
      cmdInput.value = cmdHistory.value[histIdx.value]
      e.preventDefault()
    } else if (histIdx.value === 0) {
      histIdx.value = -1
      cmdInput.value = ''
      e.preventDefault()
    }
  }
}

async function switchBackend(next: StorageBackend) {
  if (backend.value === next) return
  backend.value = next
  previewTitle.value = ''
  await hydrateFromBrowserStorage()
}

function switchPod(p: 0 | 1) {
  activePod.value = p
  refreshState()
}

async function simulateReload() {
  if (!wasm) return
  await hydrateFromBrowserStorage()
  logs.value.push({
    id: logSeq++,
    pod: 'system',
    cmd: 'COLD REOPEN PODS (simulating F5 page reload from browser storage)',
    reply: `Reopened pod-a & pod-b with 0 blocks in RAM from ${backend.value}. Next GET fetches strictly 1 VXG block!`,
    ok: true,
    ms: 0,
  })
  await scrollTerm()
}

async function clearAllStorage() {
  if (!wasm) return
  wasm.vxg_kv_reset()
  previewTitle.value = ''
  if (backend.value === 'indexeddb') {
    await idbSaveAll([])
  } else {
    lsSaveAll([])
  }
  refreshState()
  logs.value = []
  persistBanner.value = `Cleared ${backend.value === 'indexeddb' ? 'IndexedDB' : 'localStorage'} & both pods.`
}

// ---- One-Click Interactive Presets ----
async function runPresetJsonSql() {
  if (!wasm) return
  activePod.value = 0
  const plans = ['free', 'pro', 'team', 'enterprise']
  const regions = ['eu', 'us', 'ap', 'tr']
  const t0 = performance.now()
  for (let i = 0; i < 100; i++) {
    const plan = plans[i % plans.length]
    const region = regions[(i * 3) % regions.length]
    const spend = 20 + ((i * 17) % 240)
    execRaw(0, `JSON.SET user:${String(i).padStart(3, '0')} $ '{"plan":"${plan}","region":"${region}","spend":${spend}}'`)
  }
  const loadMs = performance.now() - t0
  logs.value.push({
    id: logSeq++,
    pod: 'pod-a',
    cmd: 'JSON.SET user:000 .. user:099 (100 JSON docs)',
    reply: 'OK (100 JSON documents ingested into MemTable)',
    ok: true,
    ms: loadMs,
  })
  await runCommand('VXG.COMPACT')
  await runCommand('VXG.SQL "SELECT data.plan, COUNT(*) AS users, SUM(data.spend) AS revenue GROUP BY data.plan ORDER BY revenue DESC" TSV')
  await runCommand('JSON.NUMINCRBY user:000 $.spend 50')
  await runCommand('JSON.GET user:000 $.spend')
}

async function runPresetMultiPodTimeTravel() {
  if (!wasm) return
  activePod.value = 0
  await runCommand('SET account:tier "starter"', true)
  await runCommand('SET account:quota "1000"', true)
  await runCommand('VXG.COMPACT')
  await runCommand('SET account:tier "enterprise"', true)
  await runCommand('SET account:quota "100000"', true)
  await runCommand('VXG.COMPACT')

  activePod.value = 1
  await runCommand('VXG.PULL')
  await runCommand('DBSIZE')
  await runCommand('GET account:tier')
  const ep = stats.value?.base_epoch ? Math.max(1, stats.value.base_epoch - 1) : 1
  await runCommand(`VXG.AT ${ep} GET account:tier`)
  await runCommand('GET nonexistent:bloom_demo_key')
  await runCommand('VXG.SNAPSHOTS')
}

async function runPresetGeoStreams() {
  if (!wasm) return
  activePod.value = 0
  await runCommand('GEOADD cities 28.9784 41.0082 istanbul 32.8597 39.9334 ankara 27.1428 38.4237 izmir 29.0609 40.1885 bursa', true)
  await runCommand('GEODIST cities istanbul ankara km', true)
  await runCommand('GEOSEARCH cities FROMMEMBER istanbul BYRADIUS 150 km ASC WITHDIST', true)
  await runCommand('XGROUP CREATE jobs workers 0 MKSTREAM', true)
  await runCommand('XADD jobs * task resize_img user 42', true)
  await runCommand('XADD jobs * task send_mail user 99', true)
  await runCommand('XREADGROUP GROUP workers c1 COUNT 1 STREAMS jobs >', true)
  await runCommand('XPENDING jobs workers', true)
  await runCommand('VXG.COMMIT')
}

async function runPresetRedisTypes() {
  if (!wasm) return
  activePod.value = 0
  await runCommand('HELLO 3', true)
  await runCommand('INCRBY visits 42', true)
  await runCommand('HSET profile:1 name "Ada" role "Systems Engineer"', true)
  await runCommand('HGETALL profile:1', true)
  await runCommand('ZADD leaderboard 1250 alice 1890 bob 1520 carol', true)
  await runCommand('ZRANGE leaderboard 0 -1 REV WITHSCORES', true)
  await runCommand('PFADD uniques u1 u2 u3 u4 u2 u1', true)
  await runCommand('PFCOUNT uniques', true)
  await runCommand('VXG.COMMIT')
}

function inspectObject(obj: StoreObj) {
  if (!wasm) return
  const raw = wasmGetObject(obj.key)
  previewTitle.value = `${obj.key} (${fmtBytes(raw.length)})`
  if (obj.key.endsWith('.json')) {
    try {
      previewText.value = JSON.stringify(JSON.parse(dec.decode(raw)), null, 2)
    } catch {
      previewText.value = dec.decode(raw)
    }
  } else {
    const hex = Array.from(raw.slice(0, 96))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join(' ')
    const magic = dec.decode(raw.slice(0, 4)).replace(/[^\x20-\x7e]/g, '.')
    previewText.value = `Binary compressed object (magic: "${magic}", total: ${raw.length} bytes)\nFirst 96 bytes (hex):\n${hex}${raw.length > 96 ? ' …' : ''}`
  }
}

function downloadObject(obj: StoreObj) {
  if (!wasm) return
  const raw = wasmGetObject(obj.key)
  const blob = new Blob([raw], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = obj.key.replace(/\//g, '_')
  a.click()
  URL.revokeObjectURL(url)
}

const totalStoreBytes = computed(() =>
  storeObjects.value.reduce((acc, o) => acc + o.bytes, 0)
)

const snapRatio = computed(() => {
  if (!stats.value || stats.value.base_raw_bytes === 0 || stats.value.base_compressed_bytes === 0) {
    return '—'
  }
  const x = stats.value.base_raw_bytes / stats.value.base_compressed_bytes
  return `${x.toFixed(1)}× (${fmtBytes(stats.value.base_compressed_bytes)})`
})

onMounted(async () => {
  try {
    const resp = await fetch('/vxg/vxg_kv.wasm')
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading /vxg/vxg_kv.wasm`)
    const bytes = await resp.arrayBuffer()
    const { instance } = await WebAssembly.instantiate(bytes, {})
    wasm = instance.exports as unknown as WasmExports
    outLenCell = wasm.vxg_kv_alloc(8)
    wasm.vxg_kv_set_now_ms(Date.now())
    wasm.vxg_kv_init()
    await hydrateFromBrowserStorage()
    ready.value = true
  } catch (e: unknown) {
    errorMsg.value = `Failed to initialize vxg_kv.wasm: ${String(e)}`
  }
})
</script>

<template>
  <div class="kvp-root">
    <div v-if="errorMsg" class="kvp-err">{{ errorMsg }}</div>
    <div v-else-if="!ready" class="kvp-loading">{{ loadingMsg }}</div>

    <template v-else>
      <!-- Top Control Bar: Pod Switcher + Browser Storage Switcher -->
      <div class="kvp-toolbar">
        <div class="kvp-group">
          <span class="kvp-lbl">Active Pod:</span>
          <button :class="['kvp-pill', { active: activePod === 0 }]" @click="switchPod(0)">
            Pod A (Writer)
          </button>
          <button :class="['kvp-pill', { active: activePod === 1 }]" @click="switchPod(1)">
            Pod B (Cold Reader)
          </button>
        </div>

        <div class="kvp-group">
          <span class="kvp-lbl">Persistence:</span>
          <button
            :class="['kvp-pill kvp-pill-em', { active: backend === 'indexeddb' }]"
            @click="switchBackend('indexeddb')"
          >
            IndexedDB
          </button>
          <button
            :class="['kvp-pill kvp-pill-em', { active: backend === 'localstorage' }]"
            @click="switchBackend('localstorage')"
          >
            localStorage
          </button>
        </div>
      </div>

      <!-- Persistence Status Banner -->
      <div class="kvp-status">
        <span>{{ persistBanner }}</span>
        <div class="kvp-status-actions">
          <button class="kvp-mini-btn" @click="simulateReload">Cold Reopen (F5 Sim)</button>
          <button class="kvp-mini-btn kvp-danger" @click="clearAllStorage">Clear Storage</button>
        </div>
      </div>

      <!-- One-Click Demo Presets -->
      <div class="kvp-presets">
        <span class="kvp-lbl">1-Click Demos:</span>
        <button class="kvp-preset" @click="runPresetJsonSql">1. 100 JSON + VXG.SQL</button>
        <button class="kvp-preset" @click="runPresetMultiPodTimeTravel">2. Multi-Pod &amp; Time-Travel</button>
        <button class="kvp-preset" @click="runPresetGeoStreams">3. Geo + Stream Groups</button>
        <button class="kvp-preset" @click="runPresetRedisTypes">4. RESP3 + Redis Types</button>
      </div>

      <!-- Live Metrics Strip -->
      <div v-if="stats" class="kvp-metrics">
        <div class="kvp-stat">
          <div class="kvp-stat-k">DBSIZE (Keys)</div>
          <div class="kvp-stat-v">{{ stats.dbsize }}</div>
          <div class="kvp-stat-s">mem: {{ stats.mem_dirty }} · wal: {{ stats.wal_overlay }}</div>
        </div>
        <div class="kvp-stat">
          <div class="kvp-stat-k">VXG Snapshot</div>
          <div class="kvp-stat-v">{{ snapRatio }}</div>
          <div class="kvp-stat-s">epoch {{ stats.epoch }} · {{ stats.base_blocks }} blocks</div>
        </div>
        <div class="kvp-stat">
          <div class="kvp-stat-k">Range I/O ({{ stats.pod_id }})</div>
          <div class="kvp-stat-v">{{ fmtBytes(stats.range_bytes) }}</div>
          <div class="kvp-stat-s">{{ stats.range_requests }} reqs · {{ stats.cache_blocks }} cached</div>
        </div>
        <div class="kvp-stat">
          <div class="kvp-stat-k">Bloom Filtered</div>
          <div class="kvp-stat-v">{{ stats.bloom_filtered }}</div>
          <div class="kvp-stat-s">0 Range bytes on miss</div>
        </div>
      </div>

      <!-- Main Split: Interactive Redis Terminal + Browser Object Store Inspector -->
      <div class="kvp-grid">
        <div class="kvp-panel kvp-term-panel">
          <div class="kvp-panel-hdr">
            <span>Redis CLI ({{ activePod === 0 ? 'pod-a' : 'pod-b' }} · WASM)</span>
            <span class="kvp-hint">Try: SET, JSON.SET, GEOADD, VXG.COMMIT, VXG.COMPACT, VXG.SQL</span>
          </div>
          <div ref="termEl" class="kvp-term-out">
            <div v-if="logs.length === 0" class="kvp-empty">
              Click any <strong>1-Click Demo</strong> above or type a Redis / VXG command below.
              Every <code>VXG.COMMIT</code> and <code>VXG.COMPACT</code> saves compressed <code>.vxg</code> objects directly to your browser's
              <strong>{{ backend === 'indexeddb' ? 'IndexedDB' : 'localStorage' }}</strong> — refresh the page to see them survive!
            </div>
            <div v-for="entry in logs" :key="entry.id" class="kvp-log">
              <div class="kvp-log-cmd">
                <span :class="['kvp-pod-tag', entry.pod]">{{ entry.pod }}&gt;</span>
                <span class="kvp-cmd-txt">{{ entry.cmd }}</span>
                <span v-if="entry.ms > 0" class="kvp-ms">{{ entry.ms.toFixed(2) }} ms</span>
              </div>
              <pre :class="['kvp-log-rep', { err: !entry.ok }]">{{ entry.reply }}</pre>
            </div>
          </div>
          <form class="kvp-input-row" @submit.prevent="runCommand()">
            <span :class="['kvp-pod-tag', activePod === 0 ? 'pod-a' : 'pod-b']">
              {{ activePod === 0 ? 'pod-a' : 'pod-b' }}&gt;
            </span>
            <input
              v-model="cmdInput"
              type="text"
              class="kvp-input"
              placeholder='e.g. JSON.SET item:1 $ &apos;{"price":42}&apos;  or  VXG.SQL "SELECT * LIMIT 5"'
              @keydown="onKeyDown"
            />
            <button type="submit" class="kvp-run-btn">Run</button>
          </form>
        </div>

        <div class="kvp-panel">
          <div class="kvp-panel-hdr">
            <span>{{ backend === 'indexeddb' ? 'IndexedDB' : 'localStorage' }} Object Store</span>
            <span class="kvp-hint">{{ storeObjects.length }} objects · {{ fmtBytes(totalStoreBytes) }}</span>
          </div>
          <div class="kvp-objs">
            <div v-if="storeObjects.length === 0" class="kvp-empty">
              No persisted objects yet. Run <code>VXG.COMMIT</code> or <code>VXG.COMPACT</code> to persist compressed VXG snapshots here.
            </div>
            <div v-for="o in storeObjects" :key="o.key" class="kvp-obj-row">
              <div class="kvp-obj-info" @click="inspectObject(o)">
                <div class="kvp-obj-key">{{ o.key }}</div>
                <div class="kvp-obj-meta">{{ fmtBytes(o.bytes) }} · ETag {{ o.etag }}</div>
              </div>
              <button class="kvp-dl-btn" title="Download raw file" @click="downloadObject(o)">↓</button>
            </div>
          </div>
          <div v-if="previewTitle" class="kvp-preview">
            <div class="kvp-preview-hdr">{{ previewTitle }}</div>
            <pre class="kvp-preview-body">{{ previewText }}</pre>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.kvp-root {
  margin: 1rem 0 2rem;
}

.kvp-loading,
.kvp-err {
  padding: 1.5rem;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  font-family: var(--vp-font-family-mono);
  font-size: 0.9rem;
}

.kvp-err {
  color: #ef4444;
  border-color: #ef4444;
}

.kvp-toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  margin-bottom: 0.65rem;
}

.kvp-group {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.kvp-lbl {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--vp-c-text-2);
}

.kvp-pill {
  padding: 0.35rem 0.78rem;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
}

.kvp-pill.active {
  background: #f97316;
  border-color: #f97316;
  color: #09090b;
}

.kvp-pill-em.active {
  background: #10b981;
  border-color: #10b981;
  color: #060a14;
}

.kvp-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.55rem 0.9rem;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.65rem;
}

.kvp-status-actions {
  display: flex;
  gap: 0.45rem;
}

.kvp-mini-btn {
  padding: 0.25rem 0.65rem;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.76rem;
  font-weight: 600;
  cursor: pointer;
}

.kvp-mini-btn.kvp-danger {
  color: #ef4444;
  border-color: rgba(239, 68, 68, 0.4);
}

.kvp-presets {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  margin-bottom: 0.75rem;
}

.kvp-preset {
  padding: 0.35rem 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(249, 115, 22, 0.45);
  background: rgba(249, 115, 22, 0.08);
  color: var(--vp-c-text-1);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
}

.kvp-preset:hover {
  background: rgba(249, 115, 22, 0.2);
}

.kvp-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.6rem;
  margin-bottom: 0.85rem;
}

@media (max-width: 760px) {
  .kvp-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.kvp-stat {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  padding: 0.7rem 0.85rem;
}

.kvp-stat-k {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--vp-c-text-2);
}

.kvp-stat-v {
  font-family: var(--vp-font-family-mono);
  font-size: 1.2rem;
  font-weight: 800;
  color: #f97316;
  margin: 0.15rem 0;
}

.kvp-stat-s {
  font-size: 0.74rem;
  color: var(--vp-c-text-2);
}

.kvp-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
  gap: 0.85rem;
}

@media (max-width: 900px) {
  .kvp-grid {
    grid-template-columns: 1fr;
  }
}

.kvp-panel {
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.kvp-panel-hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0.9rem;
  border-bottom: 1px solid var(--vp-c-divider);
  font-weight: 700;
  font-size: 0.84rem;
  background: var(--vp-c-bg-alt);
}

.kvp-hint {
  font-weight: 500;
  font-size: 0.74rem;
  color: var(--vp-c-text-2);
}

.kvp-term-out {
  height: 360px;
  overflow-y: auto;
  padding: 0.85rem;
  font-family: var(--vp-font-family-mono);
  font-size: 0.82rem;
  background: var(--vp-c-bg);
}

.kvp-empty {
  color: var(--vp-c-text-2);
  line-height: 1.55;
  font-size: 0.83rem;
  padding: 0.35rem;
}

.kvp-log {
  margin-bottom: 0.65rem;
}

.kvp-log-cmd {
  display: flex;
  align-items: baseline;
  gap: 0.45rem;
}

.kvp-pod-tag {
  font-weight: 800;
  font-family: var(--vp-font-family-mono);
  font-size: 0.78rem;
  white-space: nowrap;
}

.kvp-pod-tag.pod-a {
  color: #f97316;
}

.kvp-pod-tag.pod-b {
  color: #38bdf8;
}

.kvp-pod-tag.system {
  color: #10b981;
}

.kvp-cmd-txt {
  color: var(--vp-c-text-1);
  font-weight: 600;
  word-break: break-all;
}

.kvp-ms {
  margin-left: auto;
  font-size: 0.72rem;
  color: var(--vp-c-text-3);
  white-space: nowrap;
}

.kvp-log-rep {
  margin: 0.2rem 0 0 1.1rem;
  padding: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: #10b981;
  background: transparent;
  font-size: 0.8rem;
  line-height: 1.4;
}

.kvp-log-rep.err {
  color: #ef4444;
}

.kvp-input-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.75rem;
  border-top: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
}

.kvp-input {
  flex: 1;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 0.45rem 0.65rem;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.82rem;
}

.kvp-run-btn {
  padding: 0.45rem 0.9rem;
  border-radius: 8px;
  border: none;
  background: #f97316;
  color: #09090b;
  font-weight: 700;
  font-size: 0.82rem;
  cursor: pointer;
}

.kvp-objs {
  max-height: 230px;
  overflow-y: auto;
  padding: 0.55rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.kvp-obj-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.45rem 0.65rem;
  border-radius: 9px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.kvp-obj-info {
  cursor: pointer;
  min-width: 0;
}

.kvp-obj-key {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  overflow: hidden;
  text-overflow: ellipsis;
}

.kvp-obj-meta {
  font-family: var(--vp-font-family-mono);
  font-size: 0.7rem;
  color: var(--vp-c-text-2);
}

.kvp-dl-btn {
  padding: 0.25rem 0.55rem;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-weight: 700;
  cursor: pointer;
}

.kvp-preview {
  border-top: 1px solid var(--vp-c-divider);
  padding: 0.6rem 0.75rem;
  background: var(--vp-c-bg);
}

.kvp-preview-hdr {
  font-family: var(--vp-font-family-mono);
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
  margin-bottom: 0.35rem;
}

.kvp-preview-body {
  margin: 0;
  max-height: 155px;
  overflow: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 0.74rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--vp-c-text-1);
}
</style>
