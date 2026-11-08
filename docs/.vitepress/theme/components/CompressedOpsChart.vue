<script setup lang="ts">
import { computed, ref } from 'vue'

type ViewMode = 'ms' | 'mrows'
type ThreadKey = '1t' | '8t'
type CatKey = 'all' | 'query' | 'analytics' | 'mutate'

interface OpRow {
  tag: string
  cat: Exclude<CatKey, 'all'>
  name: string
  ms1: [number, number] // [vxgMs, naiveMs] 1T
  ms8: [number, number] // [vxgMs, naiveMs] 8T
}

const MROWS = 0.3 // 300,000 rows = 0.300 million rows

const OPS: OpRow[] = [
  {
    tag: 'Filter',
    cat: 'query',
    name: 'filter_column: bytes >= 50000 (zone-map)',
    ms1: [2.36, 101.03],
    ms8: [0.93, 93.34]
  },
  {
    tag: 'Filter',
    cat: 'query',
    name: 'filter_column: latency_ms >= 200 (pushdown)',
    ms1: [4.56, 104.41],
    ms8: [1.81, 87.79]
  },
  {
    tag: 'Top-K',
    cat: 'query',
    name: 'top_k_by: ORDER BY latency_ms DESC LIMIT 10',
    ms1: [1.52, 93.91],
    ms8: [0.71, 113.43]
  },
  {
    tag: 'Top-K',
    cat: 'query',
    name: 'top_k_by: WHERE status==500 ORDER BY latency_ms DESC K=20',
    ms1: [2.48, 94.21],
    ms8: [0.84, 80.68]
  },
  {
    tag: 'Agg',
    cat: 'analytics',
    name: 'aggregate_column: SUM/MIN/MAX(bytes) (all 300k)',
    ms1: [1.70, 106.65],
    ms8: [0.57, 86.85]
  },
  {
    tag: 'Group By',
    cat: 'analytics',
    name: 'group_by(service): SUM/MIN/MAX(bytes) (8 groups)',
    ms1: [4.25, 171.08],
    ms8: [1.00, 158.21]
  },
  {
    tag: 'Select',
    cat: 'analytics',
    name: 'select_columns [service,status,bytes] (all 300k)',
    ms1: [8.61, 129.27],
    ms8: [3.07, 129.75]
  },
  {
    tag: 'Splice',
    cat: 'mutate',
    name: 'project_frame [service,status,bytes,lat] (VXG -> VXG)',
    ms1: [7.09, 165.26],
    ms8: [5.26, 171.48]
  },
  {
    tag: 'Update',
    cat: 'mutate',
    name: 'update_column: WHERE status==500 SET CRITICAL',
    ms1: [5.30, 174.79],
    ms8: [3.02, 120.22]
  },
  {
    tag: 'Map',
    cat: 'mutate',
    name: 'map_column: bytes -> bytes + 100 (1-col numeric splice)',
    ms1: [4.86, 156.97],
    ms8: [1.66, 103.03]
  }
]

const threads = ref<ThreadKey>('8t')
const view = ref<ViewMode>('ms')
const cat = ref<CatKey>('all')

const visibleRows = computed(() =>
  OPS.filter((r) => cat.value === 'all' || r.cat === cat.value)
)

function pairOf(r: OpRow): [number, number] {
  return threads.value === '1t' ? r.ms1 : r.ms8
}

function metricVal(ms: number): number {
  return view.value === 'mrows' ? MROWS / (ms / 1000) : ms
}

const maxMetric = computed(() => {
  let mx = 1e-9
  for (const r of visibleRows.value) {
    const [vms, nms] = pairOf(r)
    mx = Math.max(mx, metricVal(vms), metricVal(nms))
  }
  return mx
})

function barPct(ms: number): string {
  return `${Math.max(1.5, Math.min(100, (metricVal(ms) / maxMetric.value) * 100)).toFixed(1)}%`
}

function fmtMetric(ms: number): string {
  if (view.value === 'mrows') {
    return `${(MROWS / (ms / 1000)).toFixed(1)} M rows/s`
  }
  return `${ms.toFixed(2)} ms`
}

const minLatencyMs = computed(() => {
  let best = Infinity
  for (const r of OPS) {
    const [v] = pairOf(r)
    best = Math.min(best, v)
  }
  return best
})

const peakMrows = computed(() => {
  let best = 0
  for (const r of OPS) {
    const [v] = pairOf(r)
    best = Math.max(best, MROWS / (v / 1000))
  }
  return best
})
</script>

<template>
  <div class="vxg-chart-card">
    <div class="vxg-chart-header">
      <div>
        <h4 class="vxg-chart-title">Compressed-Domain Execution vs Full Decompress + Scan/Re-encode</h4>
        <p class="vxg-chart-subtitle">
          Synthetic multi-template <code>ndjson_varied.ndjson</code> (<code>gen_varied_ndjson.py</code> · 56.5 MB · 300,000 rows · 863 blocks · 9.14% VXG v4) ·
          {{ view === 'mrows' ? 'Longer bar = higher million rows/s' : 'Shorter bar = lower wall-clock latency (ms)' }}
        </p>
      </div>
      <div class="vxg-pill-group" role="tablist" aria-label="Operation category">
        <button type="button" class="vxg-pill" :class="{ 'is-active': cat === 'all' }" @click="cat = 'all'">All (10)</button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': cat === 'query' }" @click="cat = 'query'">Filter &amp; Top-K</button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': cat === 'analytics' }" @click="cat = 'analytics'">Agg &amp; Select</button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': cat === 'mutate' }" @click="cat = 'mutate'">Splice &amp; Mutate</button>
      </div>
    </div>

    <div class="vxg-controls-bar">
      <div class="vxg-pill-group" role="tablist" aria-label="View">
        <button type="button" class="vxg-pill" :class="{ 'is-active': view === 'ms' }" @click="view = 'ms'">
          Wall-Clock Latency (ms ↓)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': view === 'mrows' }" @click="view = 'mrows'">
          Row Throughput (M rows/s ↑)
        </button>
      </div>

      <div class="vxg-pill-group" role="tablist" aria-label="Threads">
        <button type="button" class="vxg-pill" :class="{ 'is-active': threads === '1t' }" @click="threads = '1t'">
          1 Thread
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': threads === '8t' }" @click="threads = '8t'">
          Multi-Thread (8T)
        </button>
      </div>
    </div>

    <div class="vxg-kpi-grid">
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Lowest Query Latency</div>
        <div class="vxg-kpi-value">{{ minLatencyMs.toFixed(2) }} ms</div>
        <div class="vxg-kpi-delta">300,000 rows · 100% byte-identical output</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Peak Row Scan Rate</div>
        <div class="vxg-kpi-value">{{ peakMrows.toFixed(1) }} M rows/s</div>
        <div class="vxg-kpi-delta">Zone-map + single-slot columnar pushdown</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Zero-Reencode Splicing</div>
        <div class="vxg-kpi-value">{{ pairOf(OPS[7])[0].toFixed(2) }} ms</div>
        <div class="vxg-kpi-delta">{{ (pairOf(OPS[7])[1] / pairOf(OPS[7])[0]).toFixed(1) }}× vs decode + re-encode</div>
      </div>
    </div>

    <div class="vxg-legend">
      <span class="vxg-legend-item"><i class="vxg-swatch is-vxg" /> VXG Compressed-Domain</span>
      <span class="vxg-legend-item"><i class="vxg-swatch is-naive" /> Full Decompress + Scan / Re-encode</span>
    </div>

    <div class="vxg-rows">
      <div v-for="row in visibleRows" :key="row.name" class="vxg-row-card">
        <div class="vxg-row-top">
          <span class="vxg-row-name">
            <span class="vxg-tag">{{ row.tag }}</span>{{ row.name }}
          </span>
          <span class="vxg-badge">{{ (pairOf(row)[1] / pairOf(row)[0]).toFixed(1) }}× vs naive</span>
        </div>
        <div class="vxg-bars">
          <div class="vxg-bar-line is-wide-label">
            <span class="vxg-bar-codec is-vxg">VXG Direct</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-vxg" :style="{ width: barPct(pairOf(row)[0]) }" /></div>
            <span class="vxg-bar-val is-vxg">{{ fmtMetric(pairOf(row)[0]) }}</span>
          </div>
          <div class="vxg-bar-line is-wide-label">
            <span class="vxg-bar-codec">Naive</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-naive" :style="{ width: barPct(pairOf(row)[1]) }" /></div>
            <span class="vxg-bar-val">{{ fmtMetric(pairOf(row)[1]) }}</span>
          </div>
        </div>
      </div>
    </div>

    <p class="vxg-chart-footnote">
      Reported as wall-clock latency (<code>ms</code>) and rows processed per second (<code>M rows/s</code> = 300k rows / latency) rather than dividing uncompressed file size by single-column touch time. For head-to-head benchmarks against <strong>DuckDB 2.0 (<code>read_json_auto</code>, <code>.zst</code>, <code>Parquet-Snappy</code>, <code>Parquet-Zstd</code>)</strong> on real NYC TLC Taxi corpora, see <a href="CHIMERA_BENCHMARKS#duckdb">Competitor Sweep &amp; DuckDB</a>.
    </p>
  </div>
</template>
