<script setup lang="ts">
import { computed, ref } from 'vue'

type EnergyMetric = 'dec_j' | 'enc_j' | 'gci'
type ThreadMode = '1t' | '8t'
type ScopeKey = 'structured' | 'media'

interface EnergyRow {
  name: string
  kind: 'geomean' | 'structured' | 'media'
  vxg: [number, number, number]   // [encJ, decJ, gci]
  zstd: [number, number, number]
  lz4: [number, number, number]
}

const DATA: Record<ThreadMode, EnergyRow[]> = {
  '1t': [
    { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [2.333, 0.665, 133.2], zstd: [2.868, 1.641, 15.5], lz4: [1.382, 0.619, 34.0] },
    { name: 'json-records.ndjson', kind: 'structured', vxg: [2.108, 0.466, 233.9], zstd: [2.615, 1.433, 21.8], lz4: [1.152, 0.507, 39.9] },
    { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [2.711, 0.614, 122.8], zstd: [3.670, 2.148, 7.9], lz4: [1.746, 0.781, 18.5] },
    { name: 'app-logs.logfmt', kind: 'structured', vxg: [2.683, 0.590, 100.6], zstd: [3.384, 1.851, 10.2], lz4: [1.453, 0.610, 18.4] },
    { name: 'events.tsv', kind: 'structured', vxg: [2.952, 0.653, 86.0], zstd: [3.486, 1.851, 9.9], lz4: [1.999, 0.816, 10.4] },
    { name: 'telemetry.csv', kind: 'structured', vxg: [2.921, 0.671, 70.2], zstd: [4.086, 2.176, 6.5], lz4: [1.472, 0.614, 19.9] },
    { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [3.777, 0.900, 33.1], zstd: [4.766, 2.451, 3.7], lz4: [2.384, 0.900, 7.9] },
    { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [3.914, 1.169, 22.2], zstd: [5.284, 2.945, 2.6], lz4: [2.216, 1.040, 7.8] },
    { name: 'source-and-docs.txt', kind: 'structured', vxg: [6.887, 1.718, 1.3], zstd: [8.527, 3.222, 0.6], lz4: [4.638, 1.158, 1.9] },
    { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.562, 0.345, 5530.4], zstd: [0.627, 0.544, 1055.6], lz4: [0.328, 0.202, 4527.0] },
    { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.631, 0.368, 19116.6], zstd: [0.604, 0.489, 1112.9], lz4: [0.368, 0.312, 1398.9] }
  ],
  '8t': [
    { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [2.274, 0.647, 728.7], zstd: [2.735, 1.558, 94.6], lz4: [1.389, 0.594, 184.1] },
    { name: 'json-records.ndjson', kind: 'structured', vxg: [2.028, 0.467, 1246.1], zstd: [2.441, 1.344, 123.9], lz4: [1.081, 0.478, 208.8] },
    { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [2.398, 0.584, 737.8], zstd: [3.442, 2.047, 54.4], lz4: [1.530, 0.759, 111.1] },
    { name: 'app-logs.logfmt', kind: 'structured', vxg: [2.461, 0.552, 579.2], zstd: [3.068, 1.743, 60.0], lz4: [1.415, 0.557, 101.1] },
    { name: 'events.tsv', kind: 'structured', vxg: [2.625, 0.623, 498.7], zstd: [3.111, 1.700, 58.1], lz4: [1.844, 0.753, 65.4] },
    { name: 'telemetry.csv', kind: 'structured', vxg: [2.639, 0.641, 432.4], zstd: [3.573, 2.008, 49.1], lz4: [1.343, 0.587, 130.9] },
    { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [3.968, 0.861, 167.9], zstd: [4.336, 2.346, 20.7], lz4: [2.196, 0.859, 46.1] },
    { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [3.697, 1.124, 149.4], zstd: [4.999, 2.777, 18.7], lz4: [2.231, 0.970, 52.1] },
    { name: 'source-and-docs.txt', kind: 'structured', vxg: [6.433, 1.638, 9.4], zstd: [7.848, 2.977, 5.0], lz4: [4.475, 1.051, 13.8] },
    { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.659, 0.359, 20873.8], zstd: [0.690, 0.540, 5090.2], lz4: [0.443, 0.220, 13196.8] },
    { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.717, 0.376, 74433.0], zstd: [0.695, 0.491, 5038.4], lz4: [0.474, 0.316, 5107.0] }
  ]
}

const threads = ref<ThreadMode>('1t')
const metric = ref<EnergyMetric>('gci')
const scope = ref<ScopeKey>('structured')

function valOf(t: [number, number, number], m: EnergyMetric): number {
  if (m === 'enc_j') return t[0]
  if (m === 'dec_j') return t[1]
  return t[2]
}

function fmt(t: [number, number, number], m: EnergyMetric): string {
  const v = valOf(t, m)
  if (m === 'gci') return v >= 1000 ? v.toFixed(0) : v.toFixed(1)
  return `${v.toFixed(3)} J/GB`
}

const geomean = computed(() => DATA[threads.value][0])
const visibleRows = computed(() =>
  DATA[threads.value].filter((r) =>
    scope.value === 'media' ? r.kind === 'geomean' || r.kind === 'media' : r.kind !== 'media'
  )
)

const maxVal = computed(() => {
  let mx = 1e-9
  for (const r of visibleRows.value) {
    mx = Math.max(mx, valOf(r.vxg, metric.value), valOf(r.zstd, metric.value), valOf(r.lz4, metric.value))
  }
  return mx
})

function pct(t: [number, number, number]): string {
  return `${Math.max(1.5, Math.min(100, (valOf(t, metric.value) / maxVal.value) * 100)).toFixed(1)}%`
}

function badgeText(r: EnergyRow): string {
  if (metric.value === 'gci') {
    return `${(r.vxg[2] / r.zstd[2]).toFixed(1)}× greener vs Zstd-1 · ${(r.vxg[2] / r.lz4[2]).toFixed(1)}× vs LZ4`
  }
  const idx = metric.value === 'enc_j' ? 0 : 1
  return `${(r.zstd[idx] / r.vxg[idx]).toFixed(2)}× less J/GB than Zstd-1`
}
</script>

<template>
  <div class="vxg-chart-card">
    <div class="vxg-chart-header">
      <div>
        <h4 class="vxg-chart-title">Hardware Energy Efficiency &amp; Green Codec Index (GCI)</h4>
        <p class="vxg-chart-subtitle">
          Apple Silicon per-process kernel energy counter (<code>ri_energy_nj</code>) ·
          {{ metric === 'gci' ? 'Longer bar = higher GCI (greener)' : 'Shorter bar = fewer Joules per GB' }}
        </p>
      </div>
      <div class="vxg-pill-group" role="tablist" aria-label="Dataset category">
        <button type="button" class="vxg-pill" :class="{ 'is-active': scope === 'structured' }" @click="scope = 'structured'">
          Structured &amp; Text
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': scope === 'media' }" @click="scope = 'media'">
          2D Media Frames
        </button>
      </div>
    </div>

    <div class="vxg-controls-bar">
      <div class="vxg-pill-group" role="tablist" aria-label="Energy Metric">
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'gci' }" @click="metric = 'gci'">
          Green Codec Index (GCI ↑)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'dec_j' }" @click="metric = 'dec_j'">
          Decode Energy (J/GB ↓)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'enc_j' }" @click="metric = 'enc_j'">
          Encode Energy (J/GB ↓)
        </button>
      </div>

      <div class="vxg-pill-group" role="tablist" aria-label="Threads">
        <button type="button" class="vxg-pill" :class="{ 'is-active': threads === '1t' }" @click="threads = '1t'">
          1 Thread
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': threads === '8t' }" @click="threads = '8t'">
          8 Threads
        </button>
      </div>
    </div>

    <div class="vxg-kpi-grid">
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">VXG Decode Energy (Geomean)</div>
        <div class="vxg-kpi-value">{{ geomean.vxg[1].toFixed(3) }} J/GB</div>
        <div class="vxg-kpi-delta">{{ (geomean.zstd[1] / geomean.vxg[1]).toFixed(2) }}× less energy than Zstd-1</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">VXG Encode Energy (Geomean)</div>
        <div class="vxg-kpi-value">{{ geomean.vxg[0].toFixed(3) }} J/GB</div>
        <div class="vxg-kpi-delta">{{ (geomean.zstd[0] / geomean.vxg[0]).toFixed(2) }}× less energy than Zstd-1</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Green Codec Index (GCI)</div>
        <div class="vxg-kpi-value">{{ geomean.vxg[2].toFixed(1) }}</div>
        <div class="vxg-kpi-delta">
          {{ (geomean.vxg[2] / geomean.zstd[2]).toFixed(2) }}× vs Zstd-1 · {{ (geomean.vxg[2] / geomean.lz4[2]).toFixed(2) }}× vs LZ4
        </div>
      </div>
    </div>

    <div class="vxg-legend">
      <span class="vxg-legend-item"><i class="vxg-swatch is-vxg" /> VXG v4</span>
      <span class="vxg-legend-item"><i class="vxg-swatch is-zstd" /> Zstd-1</span>
      <span class="vxg-legend-item"><i class="vxg-swatch is-lz4" /> LZ4</span>
    </div>

    <div class="vxg-rows">
      <div
        v-for="row in visibleRows"
        :key="row.name"
        class="vxg-row-card"
        :class="{ 'is-highlight': row.kind === 'geomean' }"
      >
        <div class="vxg-row-top">
          <span class="vxg-row-name" :class="{ 'is-sans': row.kind === 'geomean' }">{{ row.name }}</span>
          <span class="vxg-badge">{{ badgeText(row) }}</span>
        </div>
        <div class="vxg-bars">
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec is-vxg">VXG v4</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-vxg" :style="{ width: pct(row.vxg) }" /></div>
            <span class="vxg-bar-val is-vxg">{{ fmt(row.vxg, metric) }}</span>
          </div>
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec">Zstd-1</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-zstd" :style="{ width: pct(row.zstd) }" /></div>
            <span class="vxg-bar-val">{{ fmt(row.zstd, metric) }}</span>
          </div>
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec">LZ4</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-lz4" :style="{ width: pct(row.lz4) }" /></div>
            <span class="vxg-bar-val">{{ fmt(row.lz4, metric) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
