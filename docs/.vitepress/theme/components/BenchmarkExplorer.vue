<script setup lang="ts">
import { computed, ref } from 'vue'

type ModuleId = 'vxg' | 'kv' | 'vxt' | 'vimg'

interface Kpi {
  label: string
  value: string
  sub: string
}

interface SubTab {
  id: string
  label: string
  env: string
  kpis: Kpi[]
  headers: string[]
  rows: { cells: string[]; highlight?: number[] }[]
  note: string
  linkText: string
  linkHref: string
}

interface ModuleSpec {
  id: ModuleId
  name: string
  version: string
  accent: string
  logo: string
  tabs: SubTab[]
}

const modules: ModuleSpec[] = [
  {
    id: 'vxg',
    name: 'VXG Core',
    version: 'v0.7.0',
    accent: '#10b981',
    logo: '/vxg/logos/vxg.svg',
    tabs: [
      {
        id: 'real-8t',
        label: 'Real Corpora (8T · 64 KiB)',
        env: 'Apple Silicon aarch64 NEON · VXG v0.7.0 (Format v5) vs Zstd 1.5.7 & LZ4 1.10.0',
        kpis: [
          { label: '6-Dataset Real Geomean', value: '0.1171', sub: 'Beats Zstd+Dict-9 (0.1181)' },
          { label: 'Covid-19 Time-Series', value: '0.0963', sub: 'Even smaller than zstd -19 --long' },
          { label: '8T Decode Throughput', value: '15.35 GB/s', sub: '38.59 GB/s on NYC Taxi NDJSON' },
        ],
        headers: ['Dataset (64 KiB blocks)', 'VXG Ratio', 'Zstd+Dict-9', 'LZ4 (a1)', 'VXG Enc / Dec (8T)'],
        rows: [
          { cells: ['nyc-taxi-2024-01.ndjson (52.3 MB)', '0.0593', '0.0784', '0.1899', '8.84 / 38.59 GB/s'], highlight: [1, 4] },
          { cells: ['nyc-taxi-2024-01.csv (23.0 MB)', '0.1829', '0.2141', '0.4241', '3.22 / 14.66 GB/s'], highlight: [1] },
          { cells: ['covid-19-combined.csv (8.6 MB)', '0.0963', '0.1495', '0.3254', '2.08 / 11.05 GB/s'], highlight: [1] },
          { cells: ['us_confirmed_33m.csv (35.2 MB)', '0.0456', '0.0938', '0.2363', '4.38 / 24.66 GB/s'], highlight: [1] },
          { cells: ['gharchive-2025.ndjson (50.3 MB)', '0.1181', '0.1072', '0.2256', '0.95 / 15.31 GB/s'], highlight: [2] },
          { cells: ['Text & Structured Geomean (6)', '0.1171', '0.1181', '0.2500', '1.52 / 15.35 GB/s'], highlight: [1] },
        ],
        note: 'Honest scope: Zstd+Dict-9 leads individual unstructured logs/prose/nested JSON; LZ4 leads unstructured encode GB/s.',
        linkText: 'Full Competitor Sweep →',
        linkHref: '/vxg/CHIMERA_BENCHMARKS',
      },
      {
        id: 'sql-duckdb',
        label: 'SQL vs DuckDB 2.0 (200k–40M)',
        env: 'Apple Silicon · VXG --record-aligned vs DuckDB 2.0 + Parquet (Snappy & Zstd)',
        kpis: [
          { label: 'In-Cache SQL (1T & 8T)', value: '15 / 15', sub: '0.03–0.23 ms (8T) · 0.12–1.07 ms (1T)' },
          { label: '10M NDJSON Warm Cache', value: '1.58–5.91 ms', sub: '5/5 wins (8T) · 140.8 MB vs 193.3 MB' },
          { label: '40M CSV Warm Cache', value: '11.1–27.8 ms', sub: '5/5 wins (8T) · 2.6× faster ingest' },
        ],
        headers: ['Workload (1T / 8T ms)', 'VXG --record-aligned', 'DuckDB Parquet-Snappy', 'DuckDB Parquet-Zstd', 'Lossless Raw'],
        rows: [
          { cells: ['200k NYC NDJSON · SUM/MIN/MAX', '0.16 / 0.04 ms', '1.07 / 0.84 ms', '1.36 / 1.03 ms', 'YES (2.86 MB)'], highlight: [1] },
          { cells: ['200k NYC NDJSON · GROUP BY', '0.54 / 0.10 ms', '1.31 / 0.93 ms', '1.59 / 1.17 ms', 'YES (2.86 MB)'], highlight: [1] },
          { cells: ['300k Varied NDJSON · GROUP BY', '0.86 / 0.18 ms', '2.77 / 1.61 ms', '4.00 / 2.17 ms', 'YES (4.70 MB)'], highlight: [1] },
          { cells: ['10M NYC NDJSON · GROUP BY (Warm)', '32.58 / 5.91 ms', '43.92 / 10.62 ms', '59.72 / 11.56 ms', 'YES (140.8 MB)'], highlight: [1] },
          { cells: ['40M NYC CSV · SUM/MIN/MAX (Warm)', '82.42 / 12.64 ms', '156.03 / 28.03 ms', '210.90 / 34.48 ms', 'YES (550.4 MB)'], highlight: [1] },
          { cells: ['40M NYC CSV · Cold NVMe (F_NOCACHE)', '370–643 ms (8T)', '30–63 ms (8T)', '27–63 ms (8T)', 'Parquet wins cold'], highlight: [2, 3] },
        ],
        note: 'Honest scope: On true Cold NVMe (F_NOCACHE evicted), DuckDB + Parquet wins 10/10 because Parquet reads contiguous column chunks.',
        linkText: '40M Scale & Cold-Disk Analysis →',
        linkHref: '/vxg/CHIMERA_BENCHMARKS',
      },
      {
        id: 'platforms',
        label: 'Cross-Platform CI (macOS / Linux / Win)',
        env: 'GitHub Actions CI · Native SIMD Dispatch (ARMv8 NEON · x86_64 AVX2 · wasm32 simd128)',
        kpis: [
          { label: 'macOS ARM64 (NEON)', value: '1.55 / 5.90 GB/s', sub: '1T Enc / Dec · 5.9× VPI vs LZ4' },
          { label: 'Linux x86_64 (AVX2)', value: '1.98 / 6.82 GB/s', sub: '1T Enc / Dec · 5.4× VPI vs LZ4' },
          { label: 'Windows x86_64 (AVX2)', value: '2.30 / 8.36 GB/s', sub: '1T Enc / Dec · 6.6× VPI vs LZ4' },
        ],
        headers: ['CI Platform & SIMD ISA', 'VXG Ratio', 'VXG 1T Enc / Dec', 'LZ4 1T Enc / Dec', 'Zstd-1 1T Enc / Dec'],
        rows: [
          { cells: ['macOS 14 · Apple Silicon (NEON)', '0.0249', '1.55 / 5.90 GB/s', '1.31 / 5.21 GB/s', '0.92 / 1.99 GB/s'], highlight: [1, 2] },
          { cells: ['Ubuntu Linux · x86_64 (AVX2+SSE2)', '0.0249', '1.98 / 6.82 GB/s', '2.04 / 5.99 GB/s', '1.31 / 2.70 GB/s'], highlight: [1] },
          { cells: ['Windows · x86_64 (AVX2+SSE2)', '0.0249', '2.30 / 8.36 GB/s', '1.52 / 7.52 GB/s', '1.11 / 3.08 GB/s'], highlight: [1, 2] },
        ],
        note: 'Measured on the reproducible --suite v4 CI matrix (bit-exact identical ratios across all 3 OS/arch targets).',
        linkText: 'Cross-Platform CI & Energy Details →',
        linkHref: '/vxg/BENCHMARKS',
      },
    ],
  },
  {
    id: 'kv',
    name: 'VXG-KV',
    version: 'v0.3.0',
    accent: '#f97316',
    logo: '/vxg/logos/vxg-kv.svg',
    tabs: [
      {
        id: 'kv-c16',
        label: 'Multi-Client (c=16, P=16)',
        env: 'Same Host · Official redis-benchmark · vxg-kv v0.3.0 (kqueue/epoll reactor) vs Redis 8.10.3',
        kpis: [
          { label: 'Win Rate vs Redis 8.10', value: '11 / 11', sub: 'Every benchmark command faster' },
          { label: 'Peak Throughput', value: '1.85M ops/s', sub: 'INCR / LPUSH / ZADD (up to 5.37× MSET)' },
          { label: 'Total Server CPU Time', value: '0.55 s', sub: '1.93× less CPU than Redis (1.06 s)' },
        ],
        headers: ['Command (c=16, P=16)', 'Redis 8.10.3', 'vxg-kv v0.3.0', 'Speedup'],
        rows: [
          { cells: ['INCR', '1,538,462 ops/s', '1,851,852 ops/s', '1.20×'], highlight: [2, 3] },
          { cells: ['LPUSH', '1,204,819 ops/s', '1,851,852 ops/s', '1.54×'], highlight: [2, 3] },
          { cells: ['ZADD', '1,190,476 ops/s', '1,851,852 ops/s', '1.56×'], highlight: [2, 3] },
          { cells: ['GET', '1,562,500 ops/s', '1,724,138 ops/s', '1.10×'], highlight: [2, 3] },
          { cells: ['SET', '1,265,823 ops/s', '1,694,915 ops/s', '1.34×'], highlight: [2, 3] },
          { cells: ['HSET', '1,234,568 ops/s', '1,694,915 ops/s', '1.37×'], highlight: [2, 3] },
          { cells: ['MSET (10 keys)', '310,559 ops/s', '1,666,667 ops/s', '5.37×'], highlight: [2, 3] },
        ],
        note: 'Zero external dependencies: native kqueue (macOS) / epoll (Linux) single-threaded lock-free reactor + zero-alloc RESP parser.',
        linkText: 'Explore VXG-KV Architecture →',
        linkHref: '/vxg/KV',
      },
      {
        id: 'kv-c1',
        label: 'Single-Client (c=1, P=16)',
        env: 'Same Host · Official redis-benchmark (c=1, P=16, n=100k) · vxg-kv v0.3.0 vs Redis 8.10.3',
        kpis: [
          { label: 'Win Rate vs Redis 8.10', value: '11 / 11', sub: '629k–725k ops/s single connection' },
          { label: 'CPU Efficiency', value: '2.22× less', sub: '0.58 s vs Redis 1.29 s total CPU' },
          { label: 'Idle Footprint (RSS)', value: '2.48 MiB', sub: 'vs Redis 6.56 MiB idle' },
        ],
        headers: ['Command (c=1, P=16)', 'Redis 8.10.3', 'vxg-kv v0.3.0', 'Speedup'],
        rows: [
          { cells: ['PING_INLINE', '561,798 ops/s', '724,638 ops/s', '1.29×'], highlight: [2, 3] },
          { cells: ['SET', '518,135 ops/s', '719,424 ops/s', '1.39×'], highlight: [2, 3] },
          { cells: ['GET', '591,716 ops/s', '714,286 ops/s', '1.21×'], highlight: [2, 3] },
          { cells: ['LPUSH', '500,000 ops/s', '704,225 ops/s', '1.41×'], highlight: [2, 3] },
          { cells: ['HSET / ZADD', '507,614 ops/s', '699,301 ops/s', '1.38×'], highlight: [2, 3] },
          { cells: ['MSET (10 keys)', '220,264 ops/s', '628,931 ops/s', '2.86×'], highlight: [2, 3] },
        ],
        note: 'Supports RESP2/RESP3 (HELLO 3), AUTH/ACL, JSON.*, GEO*, BLPOP/BRPOP, XGROUP/XREADGROUP, Time-Travel & --wal-shards N.',
        linkText: 'VXG-KV Guide & Commands →',
        linkHref: '/vxg/KV',
      },
      {
        id: 'kv-json',
        label: '100k JSON: RAM, Disk & SQL',
        env: '100,000 JSON User Records (12.0 MB raw) · Live HTTP/1.1 Range Origin vs Redis 8.10.3',
        kpis: [
          { label: 'Snapshot Size', value: '271.3 KiB', sub: '20.1× smaller than dump.rdb (5,458 KiB)' },
          { label: 'Cold Reader Pod RAM', value: '2.86 MiB', sub: '7.03× less RAM than Redis (20.12 MiB)' },
          { label: 'GROUP BY Analytics', value: '7.88 ms', sub: '26.8× faster than Redis MGET+JSON (211 ms)' },
        ],
        headers: ['Metric (100k JSON Keys)', 'Redis 8.10.3', 'vxg-kv v0.3.0', 'Advantage'],
        rows: [
          { cells: ['Load 100k JSON (P=500)', '190.7 ms', '152.6 ms (+60 ms compact)', '1.25× faster'], highlight: [2, 3] },
          { cells: ['Persisted Snapshot on Disk', '5,458 KiB (dump.rdb)', '271.3 KiB (snap.vxg)', '20.1× smaller'], highlight: [2, 3] },
          { cells: ['Cold Pod Idle RAM (RSS)', '20.12 MiB', '2.86 MiB (0 blocks in RAM)', '7.03× less RAM'], highlight: [2, 3] },
          { cells: ['Cold DBSIZE Network I/O', 'Full RAM dataset', '0 Range requests (0 B)', 'O(1) index-native'], highlight: [2, 3] },
          { cells: ['GROUP BY plan Analytics', '210.99 ms (MGET + JSON)', '7.88 ms (VXG.SQL)', '26.8× faster'], highlight: [2, 3] },
        ],
        note: 'Honest scope: VXG-KV is built for stateless pod/object-storage workloads and does not ship Lua scripting or Redis Cluster slots.',
        linkText: 'Read VXG-KV Spec →',
        linkHref: '/vxg/KV',
      },
    ],
  },
  {
    id: 'vxt',
    name: 'VXT-Turbo',
    version: 'v0.6.0',
    accent: '#a855f7',
    logo: '/vxg/logos/vxt-turbo.svg',
    tabs: [
      {
        id: 'vxt-cuda',
        label: 'NVIDIA CUDA (RTX 5090)',
        env: 'NVIDIA RTX 5090 · VXT-Turbo v0.6.0 VRAM-Resident E8 Lattice Kernels',
        kpis: [
          { label: 'Q3 Generation Speed', value: '962.9 tok/s', sub: 'Direct compressed-domain CUDA matvec' },
          { label: 'Q6 Generation Speed', value: '796.3 tok/s', sub: 'Higher accuracy than GGUF Q6_K' },
          { label: 'Model Architecture', value: 'LLM · VLM · MoE', sub: 'Zero-OOM streaming packer + OpenAI SSE' },
        ],
        headers: ['Preset / Workload', 'Throughput', 'Backend', 'Highlights'],
        rows: [
          { cells: ['VXT E8 Q3 (VRAM-Resident)', '962.86 tok/s', 'NVIDIA CUDA', 'Fused E8 dequant + GEMV kernel'], highlight: [1] },
          { cells: ['VXT E8-HQ Q6 (VRAM-Resident)', '796.28 tok/s', 'NVIDIA CUDA', 'Beats GGUF Q6_K quality & size'], highlight: [1] },
          { cells: ['OpenAI-Compatible Server', 'SSE + Metrics', 'CUDA / Metal', 'Speculative Decoding + Prometheus'], highlight: [1] },
        ],
        note: 'Packs both GGUF and SafeTensors models with zero-OOM streaming into self-contained .vxt (VXT1) containers.',
        linkText: 'Full VXT-Turbo Spec & Benchmarks →',
        linkHref: '/vxg/TENSOR',
      },
      {
        id: 'vxt-metal',
        label: 'Apple Silicon (M2 Max Metal)',
        env: 'Apple M2 Max · Native Metal Compressed-Domain Kernels + 3D M-RoPE Vision',
        kpis: [
          { label: '0.8B Model Decode', value: '298 tok/s', sub: 'Native Apple Metal compute' },
          { label: '2B Model Decode', value: '176 tok/s', sub: 'Unified memory zero-copy' },
          { label: '9B + Vision Decode', value: '47.2 tok/s', sub: '3D M-RoPE multimodal ready' },
        ],
        headers: ['Model Size (Apple M2 Max)', 'Metal Speed', 'Modality', 'Memory Path'],
        rows: [
          { cells: ['0.8B Parameter Model', '298 tok/s', 'Text LLM', 'Apple Unified Memory (mmap)'], highlight: [1] },
          { cells: ['2B Parameter Model', '176 tok/s', 'Text LLM', 'Apple Unified Memory (mmap)'], highlight: [1] },
          { cells: ['9B Parameter Model (Qwen3.5-9B)', '40.7 – 47.2 tok/s', 'LLM + Vision (3D M-RoPE)', 'Direct E8 Metal kernel'], highlight: [1] },
        ],
        note: 'Companion desktop app available in apps/vxt-studio for local chat, vision prompting, and live telemetry.',
        linkText: 'Apple Metal & Quality Report →',
        linkHref: '/vxg/TENSOR',
      },
      {
        id: 'vxt-quality',
        label: 'Quality vs GGUF (Qwen3.5-9B)',
        env: 'Qwen3.5-9B · E8 / E8-HQ Gosset Lattice Quantization vs GGUF Q6_K & Q8_0',
        kpis: [
          { label: 'vs GGUF Q6_K Size', value: '-0.43 GB', sub: 'And -32% lower KL divergence' },
          { label: 'vs GGUF Q8_0 Size', value: '-0.89 GB', sub: 'And -56% lower KL divergence' },
          { label: 'Quantization Lattice', value: 'E8 Gosset', sub: 'Optimal 8D sphere packing' },
        ],
        headers: ['Comparison (Qwen3.5-9B)', 'Size Savings', 'KL Divergence Improvement', 'Result'],
        rows: [
          { cells: ['VXT E8-HQ vs GGUF Q6_K', '-0.43 GB smaller', '-32% TR KLD', 'Smaller & more accurate'], highlight: [1, 2, 3] },
          { cells: ['VXT E8-HQ vs GGUF Q8_0', '-0.89 GB smaller', '-56% TR KLD', 'Smaller & more accurate'], highlight: [1, 2, 3] },
        ],
        note: 'Full perplexity, KLD tables, and reproducible packing commands are documented in docs/TENSOR.md.',
        linkText: 'Read Quality Benchmarks →',
        linkHref: '/vxg/TENSOR',
      },
    ],
  },
  {
    id: 'vimg',
    name: 'VIMG',
    version: 'v0.4.0',
    accent: '#38bdf8',
    logo: '/vxg/logos/vimg.svg',
    tabs: [
      {
        id: 'vimg-lossless',
        label: 'Lossless (UI & Photo)',
        env: 'Apple M2 Max NEON · Zero-Dep Pure Rust YCoCg-R + CDF 5/3 + Spatial Palette',
        kpis: [
          { label: '1080p UI Decode', value: '1.00 ms', sub: 'Faster than PNG (1.38 ms) at 0.20 bpp' },
          { label: 'Lossless Photo vs PNG', value: '-38.6%', sub: '60× faster encode than WebP Lossless' },
          { label: 'Multi-Core Throughput', value: '1.17 GP/s', sub: 'NEON / AVX2 / wasm32 +simd128' },
        ],
        headers: ['Workload', 'VIMG v0.4.0', 'Baseline', 'Advantage'],
        rows: [
          { cells: ['1080p UI Screenshot (Lossless)', '1.00 ms · 0.20 bpp (118×)', 'PNG 1.38 ms', 'Zero-IDWT Spatial Palette+RLE'], highlight: [1, 3] },
          { cells: ['Lossless Natural Photo (q=100)', '-38.6% smaller than PNG', 'PNG / WebP Lossless', '60× faster enc than WebP'], highlight: [1, 3] },
          { cells: ['16-bit Full-Precision HDR', '8 / 10 / 12 / 16-bit', 'Reversible YCoCg-R', 'Bit-exact lossless + Display P3 ICC'], highlight: [1, 3] },
        ],
        note: 'Core crate has 0 external dependencies and produces bit-exact identical output across ARM64, x86_64, and WASM.',
        linkText: 'Try Live VIMG WASM Studio →',
        linkHref: '/vxg/vimg-playground',
      },
      {
        id: 'vimg-range',
        label: 'Lossy, Animation & HTTP Range',
        env: 'VIMG1 / VIMA1 · HTTP/HTTPS 206 Partial Tile & Pyramid Level Queries',
        kpis: [
          { label: 'Lossy vs JPEG', value: '-46.8%', sub: 'CDF 9/7 + Tile-Adaptive AQ (--aq)' },
          { label: 'Remote Crop / Resize', value: '~1 ms', sub: 'Fetches only intersecting tiles/levels' },
          { label: 'WASM Bundle (+simd128)', value: '277 KiB', sub: 'Zero WASI · Browser & Edge Workers' },
        ],
        headers: ['Capability', 'Wire Behavior', 'Result'],
        rows: [
          { cells: ['Spatial Crop / Downscale', 'HTTP 206 intersecting tiles only', 'Zero-decode vimg → vimg repackage'], highlight: [1, 2] },
          { cells: ['ProgressiveDecoder (L0→L3)', 'Incremental wavelet levels', 'No re-download of coarser levels'], highlight: [1, 2] },
          { cells: ['VIMA1 Animated Container', '20 B/frame random-access directory', 'Scrub to frame #N via HTTP Range'], highlight: [1, 2] },
          { cells: ['Grayscale / Color Grading', 'Skip Co/Cg chroma planes (0 B)', 'Subband blur, sharpen, temperature'], highlight: [1, 2] },
        ],
        note: 'Includes optional pure-Rust TLS 1.3 (rustls) remote streaming, WebP Edge Proxy, and 12,000-case fuzz suite.',
        linkText: 'Read VIMG Architecture Spec →',
        linkHref: '/vxg/VIMG',
      },
    ],
  },
]

const activeModId = ref<ModuleId>('vxg')
const activeSubByMod = ref<Record<ModuleId, string>>({
  vxg: 'real-8t',
  kv: 'kv-c16',
  vxt: 'vxt-cuda',
  vimg: 'vimg-lossless',
})

const currentMod = computed(() => modules.find((m) => m.id === activeModId.value)!)
const currentTab = computed(
  () =>
    currentMod.value.tabs.find((t) => t.id === activeSubByMod.value[activeModId.value]) ??
    currentMod.value.tabs[0]
)

function selectMod(id: ModuleId) {
  activeModId.value = id
}

function selectSub(id: string) {
  activeSubByMod.value[activeModId.value] = id
}
</script>

<template>
  <div class="bx-wrap" :style="{ '--bx-accent': currentMod.accent }">
    <!-- Level 1: Module Switcher -->
    <div class="bx-mod-bar" role="tablist" aria-label="VXG Modules">
      <button
        v-for="m in modules"
        :key="m.id"
        role="tab"
        :aria-selected="activeModId === m.id"
        :class="['bx-mod-btn', { active: activeModId === m.id }]"
        :style="activeModId === m.id ? { borderColor: m.accent } : {}"
        @click="selectMod(m.id)"
      >
        <img :src="m.logo" :alt="m.name" class="bx-mod-logo" />
        <span class="bx-mod-name">{{ m.name }}</span>
        <span class="bx-mod-ver">{{ m.version }}</span>
      </button>
    </div>

    <!-- Level 2: Platform / Scenario Switcher -->
    <div class="bx-sub-bar" role="tablist" aria-label="Platform and Scenario">
      <button
        v-for="t in currentMod.tabs"
        :key="t.id"
        role="tab"
        :aria-selected="currentTab.id === t.id"
        :class="['bx-sub-btn', { active: currentTab.id === t.id }]"
        @click="selectSub(t.id)"
      >
        {{ t.label }}
      </button>
    </div>

    <!-- Environment Strip -->
    <div class="bx-env">{{ currentTab.env }}</div>

    <!-- 3 Headline KPI Cards -->
    <div class="bx-kpis">
      <div v-for="k in currentTab.kpis" :key="k.label" class="bx-kpi">
        <div class="bx-kpi-label">{{ k.label }}</div>
        <div class="bx-kpi-val">{{ k.value }}</div>
        <div class="bx-kpi-sub">{{ k.sub }}</div>
      </div>
    </div>

    <!-- Comparison Table -->
    <div class="bx-table-scroll">
      <table class="bx-table">
        <thead>
          <tr>
            <th v-for="h in currentTab.headers" :key="h">{{ h }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, ri) in currentTab.rows" :key="ri">
            <td
              v-for="(c, ci) in r.cells"
              :key="ci"
              :class="{ 'bx-win': r.highlight?.includes(ci), 'bx-first': ci === 0 }"
            >
              {{ c }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Footer Note + Deep Link -->
    <div class="bx-footer">
      <span class="bx-note">{{ currentTab.note }}</span>
      <a :href="currentTab.linkHref" class="bx-link">{{ currentTab.linkText }}</a>
    </div>
  </div>
</template>

<style scoped>
.bx-wrap {
  margin: 1.5rem 0 2.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  padding: 1.25rem;
}

.bx-mod-bar {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.65rem;
  margin-bottom: 0.9rem;
}

@media (max-width: 760px) {
  .bx-mod-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.bx-mod-btn {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.65rem 0.85rem;
  border-radius: 12px;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.16s ease;
  text-align: left;
}

.bx-mod-btn:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-3);
}

.bx-mod-btn.active {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

.bx-mod-logo {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
}

.bx-mod-name {
  font-weight: 700;
  font-size: 0.92rem;
  white-space: nowrap;
}

.bx-mod-ver {
  margin-left: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  opacity: 0.75;
}

.bx-sub-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 0.75rem;
}

.bx-sub-btn {
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  color: var(--vp-c-text-2);
  font-size: 0.83rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.bx-sub-btn:hover {
  color: var(--vp-c-text-1);
}

.bx-sub-btn.active {
  background: var(--bx-accent);
  border-color: var(--bx-accent);
  color: #060a14;
}

.bx-env {
  font-family: var(--vp-font-family-mono);
  font-size: 0.78rem;
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
}

.bx-kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}

@media (max-width: 720px) {
  .bx-kpis {
    grid-template-columns: 1fr;
  }
}

.bx-kpi {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
  padding: 0.85rem 1rem;
}

.bx-kpi-label {
  font-size: 0.76rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--vp-c-text-2);
}

.bx-kpi-val {
  font-size: 1.45rem;
  font-weight: 800;
  color: var(--bx-accent);
  margin: 0.2rem 0;
  font-family: var(--vp-font-family-mono);
}

.bx-kpi-sub {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.bx-table-scroll {
  overflow-x: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
}

.bx-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 !important;
  display: table !important;
  font-size: 0.86rem;
}

.bx-table th {
  text-align: left;
  padding: 0.65rem 0.9rem;
  font-weight: 700;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
}

.bx-table td {
  padding: 0.6rem 0.9rem;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

.bx-table tr:last-child td {
  border-bottom: none;
}

.bx-table td.bx-first {
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.bx-table td.bx-win {
  font-weight: 700;
  color: var(--bx-accent);
  font-family: var(--vp-font-family-mono);
}

.bx-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.85rem;
  font-size: 0.82rem;
}

.bx-note {
  color: var(--vp-c-text-2);
}

.bx-link {
  font-weight: 700;
  color: var(--bx-accent) !important;
  text-decoration: none !important;
  white-space: nowrap;
}

.bx-link:hover {
  text-decoration: underline !important;
}
</style>
