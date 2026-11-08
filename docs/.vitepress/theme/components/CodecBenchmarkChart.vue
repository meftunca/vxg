<script setup lang="ts">
import { computed, ref } from 'vue'

type MetricKey = 'factor' | 'decode' | 'encode' | 'vpi'
type ScopeKey = 'structured' | 'media'
type PlatformKey = 'm_1t' | 'm_8t' | 'ci_mac' | 'ci_linux' | 'ci_win'

interface RowEntry {
  name: string
  kind: 'geomean' | 'structured' | 'media'
  vxg: [number, number, number]   // [ratio, encGBs, decGBs]
  zstd: [number, number, number]
  lz4: [number, number, number]
}

const PLATFORMS: Record<PlatformKey, { label: string; sub: string; rows: RowEntry[] }> = {
  m_1t: {
    label: 'Apple Silicon (1T)',
    sub: 'aarch64 NEON · threads=1 · 16 MiB · 64 KiB blocks',
    rows: [
      { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [0.0162, 2.217, 10.532], zstd: [0.0615, 1.550, 2.657], lz4: [0.1286, 2.274, 6.937] },
      { name: 'source-and-docs.txt', kind: 'structured', vxg: [0.0036, 3.620, 62.686], zstd: [0.3133, 0.509, 1.365], lz4: [0.4286, 0.635, 3.701] },
      { name: 'json-records.ndjson', kind: 'structured', vxg: [0.0227, 2.077, 11.911], zstd: [0.0542, 1.695, 3.380], lz4: [0.1497, 2.455, 9.366] },
      { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [0.0248, 1.465, 7.914], zstd: [0.0779, 1.251, 2.373], lz4: [0.1545, 1.962, 5.706] },
      { name: 'events.tsv', kind: 'structured', vxg: [0.0326, 1.459, 8.642], zstd: [0.0741, 1.313, 2.593], lz4: [0.2223, 1.511, 5.828] },
      { name: 'app-logs.logfmt', kind: 'structured', vxg: [0.0353, 1.512, 9.012], zstd: [0.0695, 1.270, 2.464], lz4: [0.2007, 1.738, 7.028] },
      { name: 'telemetry.csv', kind: 'structured', vxg: [0.0407, 1.438, 7.935], zstd: [0.0841, 1.120, 2.161], lz4: [0.1994, 1.805, 7.245] },
      { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [0.0445, 1.139, 5.347], zstd: [0.1043, 0.982, 1.953], lz4: [0.2132, 1.377, 4.887] },
      { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [0.0480, 0.953, 3.984], zstd: [0.1196, 0.875, 1.736], lz4: [0.2065, 1.367, 4.401] },
      { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.0048, 9.041, 14.588], zstd: [0.0091, 6.528, 4.801], lz4: [0.0140, 12.657, 19.655] },
      { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.0012, 8.355, 14.792], zstd: [0.0119, 7.069, 7.120], lz4: [0.0227, 10.698, 10.397] }
    ]
  },
  m_8t: {
    label: 'Apple Silicon (8T)',
    sub: 'aarch64 NEON · threads=8 · work-stealing · 16 MiB',
    rows: [
      { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [0.0162, 8.512, 54.965], zstd: [0.0615, 7.731, 14.065], lz4: [0.1286, 10.957, 32.112] },
      { name: 'source-and-docs.txt', kind: 'structured', vxg: [0.0036, 4.358, 94.539], zstd: [0.3133, 3.254, 8.126], lz4: [0.4286, 4.013, 21.063] },
      { name: 'json-records.ndjson', kind: 'structured', vxg: [0.0227, 9.893, 69.601], zstd: [0.0542, 8.816, 16.191], lz4: [0.1497, 13.356, 37.503] },
      { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [0.0248, 6.465, 48.273], zstd: [0.0779, 7.320, 14.111], lz4: [0.1545, 9.696, 31.453] },
      { name: 'events.tsv', kind: 'structured', vxg: [0.0326, 6.982, 56.157], zstd: [0.0741, 6.141, 13.133], lz4: [0.2223, 7.658, 25.999] },
      { name: 'app-logs.logfmt', kind: 'structured', vxg: [0.0353, 7.336, 60.092], zstd: [0.0695, 6.566, 12.504], lz4: [0.2007, 9.327, 35.408] },
      { name: 'telemetry.csv', kind: 'structured', vxg: [0.0407, 6.648, 48.375], zstd: [0.0841, 6.945, 11.244], lz4: [0.1994, 9.660, 31.411] },
      { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [0.0445, 5.523, 32.666], zstd: [0.1043, 4.945, 11.862], lz4: [0.2132, 7.382, 26.519] },
      { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [0.0480, 4.997, 26.342], zstd: [0.1196, 4.418, 10.934], lz4: [0.2065, 7.476, 20.590] },
      { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.0048, 27.868, 76.950], zstd: [0.0091, 22.701, 20.078], lz4: [0.0140, 36.387, 66.873] },
      { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.0012, 27.348, 73.312], zstd: [0.0119, 26.166, 33.933], lz4: [0.0227, 34.627, 44.447] }
    ]
  },
  ci_mac: {
    label: 'macOS CI (1T)',
    sub: 'GitHub Actions macos-14 · NEON · 16 MiB',
    rows: [
      { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [0.0249, 1.554, 5.896], zstd: [0.0607, 0.920, 1.990], lz4: [0.1270, 1.307, 5.205] },
      { name: 'json-records.ndjson', kind: 'structured', vxg: [0.0227, 1.618, 9.242], zstd: [0.0542, 0.917, 2.811], lz4: [0.1497, 1.437, 6.288] },
      { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [0.0248, 1.281, 5.830], zstd: [0.0779, 0.770, 1.597], lz4: [0.1545, 1.260, 4.055] },
      { name: 'events.tsv', kind: 'structured', vxg: [0.0334, 1.187, 7.044], zstd: [0.0741, 0.751, 1.715], lz4: [0.2223, 0.952, 4.026] },
      { name: 'app-logs.logfmt', kind: 'structured', vxg: [0.0353, 1.614, 8.613], zstd: [0.0695, 0.770, 2.104], lz4: [0.2007, 1.205, 5.748] },
      { name: 'telemetry.csv', kind: 'structured', vxg: [0.0407, 1.296, 5.954], zstd: [0.0841, 0.730, 1.739], lz4: [0.1994, 0.982, 5.514] },
      { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [0.0445, 1.175, 4.980], zstd: [0.1043, 0.616, 1.348], lz4: [0.2132, 0.873, 3.274] },
      { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [0.0480, 0.940, 3.503], zstd: [0.1196, 0.671, 1.467], lz4: [0.2065, 0.985, 3.411] },
      { name: 'source-and-docs.txt', kind: 'structured', vxg: [0.2702, 0.374, 1.696], zstd: [0.2714, 0.390, 1.084], lz4: [0.3778, 0.496, 3.200] },
      { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.0048, 6.423, 9.170], zstd: [0.0091, 2.456, 2.719], lz4: [0.0140, 3.846, 14.402] },
      { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.0012, 6.025, 9.611], zstd: [0.0119, 3.669, 5.913], lz4: [0.0227, 4.357, 8.715] }
    ]
  },
  ci_linux: {
    label: 'Linux CI (AVX2)',
    sub: 'GitHub Actions ubuntu-latest · x86_64 AVX2 · 16 MiB',
    rows: [
      { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [0.0249, 1.984, 6.822], zstd: [0.0607, 1.308, 2.699], lz4: [0.1270, 2.038, 5.994] },
      { name: 'json-records.ndjson', kind: 'structured', vxg: [0.0227, 2.474, 10.906], zstd: [0.0542, 1.650, 4.085], lz4: [0.1497, 2.484, 9.651] },
      { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [0.0248, 1.824, 7.681], zstd: [0.0779, 1.145, 2.461], lz4: [0.1545, 1.836, 5.197] },
      { name: 'events.tsv', kind: 'structured', vxg: [0.0334, 1.460, 6.709], zstd: [0.0741, 1.034, 3.155], lz4: [0.2223, 1.281, 4.957] },
      { name: 'app-logs.logfmt', kind: 'structured', vxg: [0.0353, 1.885, 8.434], zstd: [0.0695, 1.219, 3.201], lz4: [0.2007, 1.871, 7.626] },
      { name: 'telemetry.csv', kind: 'structured', vxg: [0.0407, 1.706, 7.513], zstd: [0.0841, 0.737, 2.395], lz4: [0.1994, 1.437, 7.170] },
      { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [0.0445, 1.377, 5.483], zstd: [0.1043, 0.856, 2.113], lz4: [0.2132, 1.286, 4.587] },
      { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [0.0480, 1.032, 3.772], zstd: [0.1196, 0.801, 1.792], lz4: [0.2065, 1.300, 3.918] },
      { name: 'source-and-docs.txt', kind: 'structured', vxg: [0.2712, 0.513, 2.205], zstd: [0.2722, 0.416, 1.772], lz4: [0.3785, 0.661, 3.970] },
      { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.0048, 8.370, 14.818], zstd: [0.0091, 5.723, 4.797], lz4: [0.0140, 9.805, 21.433] },
      { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.0012, 7.311, 9.078], zstd: [0.0119, 5.124, 2.622], lz4: [0.0227, 7.260, 2.881] }
    ]
  },
  ci_win: {
    label: 'Windows CI (AVX2)',
    sub: 'GitHub Actions windows-latest · x86_64 AVX2 · 16 MiB',
    rows: [
      { name: '10-Dataset Geometric Mean', kind: 'geomean', vxg: [0.0249, 2.295, 8.362], zstd: [0.0606, 1.110, 3.080], lz4: [0.1268, 1.520, 7.519] },
      { name: 'json-records.ndjson', kind: 'structured', vxg: [0.0227, 2.748, 14.409], zstd: [0.0542, 1.261, 3.658], lz4: [0.1497, 1.776, 11.171] },
      { name: 'event-mix-identifiers.ndjson', kind: 'structured', vxg: [0.0248, 2.018, 9.347], zstd: [0.0779, 0.995, 2.339], lz4: [0.1545, 1.513, 4.840] },
      { name: 'events.tsv', kind: 'structured', vxg: [0.0334, 1.893, 8.583], zstd: [0.0741, 0.945, 2.917], lz4: [0.2223, 1.115, 6.385] },
      { name: 'app-logs.logfmt', kind: 'structured', vxg: [0.0353, 1.758, 9.922], zstd: [0.0695, 1.045, 2.667], lz4: [0.2007, 1.312, 8.689] },
      { name: 'telemetry.csv', kind: 'structured', vxg: [0.0407, 2.010, 8.915], zstd: [0.0841, 0.872, 2.481], lz4: [0.1994, 1.364, 8.835] },
      { name: 'field-order-optional.ndjson', kind: 'structured', vxg: [0.0445, 1.519, 6.405], zstd: [0.1043, 0.760, 1.946], lz4: [0.2132, 1.132, 4.697] },
      { name: 'bursty-variable-records.ndjson', kind: 'structured', vxg: [0.0480, 1.176, 4.472], zstd: [0.1196, 0.733, 1.670], lz4: [0.2065, 1.178, 3.965] },
      { name: 'source-and-docs.txt', kind: 'structured', vxg: [0.2678, 0.576, 2.962], zstd: [0.2672, 0.522, 1.805], lz4: [0.3721, 0.735, 5.253] },
      { name: 'ui-render-32bpp.bmp', kind: 'media', vxg: [0.0048, 10.452, 17.003], zstd: [0.0091, 3.133, 18.978], lz4: [0.0140, 3.523, 47.484] },
      { name: 'rgb-frame-24bpp.bmp', kind: 'media', vxg: [0.0012, 10.137, 11.329], zstd: [0.0119, 2.892, 4.180], lz4: [0.0227, 3.554, 4.692] }
    ]
  }
}

const props = withDefaults(defineProps<{ compact?: boolean }>(), {
  compact: false
})

const platform = ref<PlatformKey>('m_1t')
const metric = ref<MetricKey>('factor')
const scope = ref<ScopeKey>('structured')

function scoreOf(t: [number, number, number], m: MetricKey): number {
  const [ratio, enc, dec] = t
  if (m === 'factor') return 1 / ratio
  if (m === 'decode') return dec
  if (m === 'encode') return enc
  return Math.sqrt(enc * dec) / ratio
}

function fmtVal(t: [number, number, number], m: MetricKey): string {
  const [ratio, enc, dec] = t
  if (m === 'factor') return `${(1 / ratio).toFixed(1)}× (${(ratio * 100).toFixed(1)}%)`
  if (m === 'decode') return `${dec.toFixed(2)} GB/s`
  if (m === 'encode') return `${enc.toFixed(2)} GB/s`
  const vpi = Math.sqrt(enc * dec) / ratio
  return vpi >= 1000 ? vpi.toFixed(0) : vpi.toFixed(1)
}

const currentPlatform = computed(() => PLATFORMS[platform.value])
const geomean = computed(() => currentPlatform.value.rows[0])

const visibleRows = computed(() => {
  if (props.compact) {
    return currentPlatform.value.rows.filter(
      (r) => r.kind === 'geomean' || ['json-records.ndjson', 'app-logs.logfmt', 'telemetry.csv', 'source-and-docs.txt'].includes(r.name)
    )
  }
  if (scope.value === 'media') {
    return currentPlatform.value.rows.filter((r) => r.kind === 'geomean' || r.kind === 'media')
  }
  return currentPlatform.value.rows.filter((r) => r.kind !== 'media')
})

const maxScore = computed(() => {
  let mx = 1e-9
  for (const r of visibleRows.value) {
    mx = Math.max(mx, scoreOf(r.vxg, metric.value), scoreOf(r.zstd, metric.value), scoreOf(r.lz4, metric.value))
  }
  return mx
})

function pct(t: [number, number, number]): string {
  return `${Math.max(1.5, Math.min(100, (scoreOf(t, metric.value) / maxScore.value) * 100)).toFixed(1)}%`
}

function winBadge(r: RowEntry): string {
  const v = scoreOf(r.vxg, metric.value)
  const l = scoreOf(r.lz4, metric.value)
  const z = scoreOf(r.zstd, metric.value)
  if (metric.value === 'factor') {
    return `${(v / l).toFixed(1)}× smaller vs LZ4 · ${(v / z).toFixed(1)}× vs Zstd-1`
  }
  return `${(v / z).toFixed(2)}× vs Zstd-1 · ${(v / l).toFixed(2)}× vs LZ4`
}
</script>

<template>
  <div class="vxg-chart-card">
    <div class="vxg-chart-header">
      <div>
        <h4 class="vxg-chart-title">Interactive Codec Comparison — VXG v4 vs Zstd-1 &amp; LZ4</h4>
        <p class="vxg-chart-subtitle">{{ currentPlatform.sub }} · Longer bar is better</p>
      </div>
      <div v-if="!compact" class="vxg-pill-group" role="tablist" aria-label="Dataset category">
        <button type="button" class="vxg-pill" :class="{ 'is-active': scope === 'structured' }" @click="scope = 'structured'">
          Structured &amp; Text (8)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': scope === 'media' }" @click="scope = 'media'">
          2D Media Frames (BMP)
        </button>
      </div>
    </div>

    <div class="vxg-controls-bar">
      <div class="vxg-pill-group" role="tablist" aria-label="Metric">
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'factor' }" @click="metric = 'factor'">
          Compression (× smaller)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'decode' }" @click="metric = 'decode'">
          Decode (GB/s)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'encode' }" @click="metric = 'encode'">
          Encode (GB/s)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': metric === 'vpi' }" @click="metric = 'vpi'">
          Composite VPI
        </button>
      </div>

      <div class="vxg-pill-group" role="tablist" aria-label="Platform">
        <button
          v-for="(p, key) in PLATFORMS"
          :key="key"
          type="button"
          class="vxg-pill"
          :class="{ 'is-active': platform === key }"
          @click="platform = key"
        >
          {{ p.label }}
        </button>
      </div>
    </div>

    <div class="vxg-kpi-grid">
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">10-Dataset Compression</div>
        <div class="vxg-kpi-value">{{ (1 / geomean.vxg[0]).toFixed(1) }}× smaller</div>
        <div class="vxg-kpi-delta">
          {{ (geomean.lz4[0] / geomean.vxg[0]).toFixed(2) }}× smaller vs LZ4 · {{ (geomean.zstd[0] / geomean.vxg[0]).toFixed(2) }}× vs Zstd-1
        </div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Geomean Decode Speed</div>
        <div class="vxg-kpi-value">{{ geomean.vxg[2].toFixed(2) }} GB/s</div>
        <div class="vxg-kpi-delta">
          {{ (geomean.vxg[2] / geomean.zstd[2]).toFixed(2) }}× faster vs Zstd-1 · {{ (geomean.vxg[2] / geomean.lz4[2]).toFixed(2) }}× vs LZ4
        </div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">VXG Performance Index (VPI)</div>
        <div class="vxg-kpi-value">{{ scoreOf(geomean.vxg, 'vpi').toFixed(1) }}</div>
        <div class="vxg-kpi-delta">
          {{ (scoreOf(geomean.vxg, 'vpi') / scoreOf(geomean.lz4, 'vpi')).toFixed(2) }}× vs LZ4 · {{ (scoreOf(geomean.vxg, 'vpi') / scoreOf(geomean.zstd, 'vpi')).toFixed(2) }}× vs Zstd-1
        </div>
      </div>
    </div>

    <div class="vxg-legend">
      <span class="vxg-legend-item"><i class="vxg-swatch is-vxg" /> VXG v4</span>
      <span class="vxg-legend-item"><i class="vxg-swatch is-zstd" /> Zstd-1 (1.5.7)</span>
      <span class="vxg-legend-item"><i class="vxg-swatch is-lz4" /> LZ4 (1.10.0)</span>
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
          <span class="vxg-badge">{{ winBadge(row) }}</span>
        </div>
        <div class="vxg-bars">
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec is-vxg">VXG v4</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-vxg" :style="{ width: pct(row.vxg) }" /></div>
            <span class="vxg-bar-val is-vxg">{{ fmtVal(row.vxg, metric) }}</span>
          </div>
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec">Zstd-1</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-zstd" :style="{ width: pct(row.zstd) }" /></div>
            <span class="vxg-bar-val">{{ fmtVal(row.zstd, metric) }}</span>
          </div>
          <div class="vxg-bar-line">
            <span class="vxg-bar-codec">LZ4</span>
            <div class="vxg-bar-track"><div class="vxg-bar-fill is-lz4" :style="{ width: pct(row.lz4) }" /></div>
            <span class="vxg-bar-val">{{ fmtVal(row.lz4, metric) }}</span>
          </div>
        </div>
      </div>
    </div>

    <p class="vxg-chart-footnote">
      All codecs operate on identical independent 64 KiB blocks (checksums off for fair C baseline comparison; full byte-for-byte verification after timing).
    </p>
  </div>
</template>
