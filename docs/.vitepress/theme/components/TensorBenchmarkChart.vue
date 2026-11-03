<script setup lang="ts">
import { computed, ref } from 'vue'

type TabKey = 'e2e' | 'gemv'
type E2eMetric = 'ratio' | 'toks' | 'fidelity' | 'prefill'
type GemvMetric = 'eqbw' | 'ratio' | 'sqnr'
type TensorKey = 'embd' | 'qkv' | 'ffn'

interface E2eRow {
  name: string
  tag: string
  barClass: 'is-vxg' | 'is-zstd' | 'is-lz4' | 'is-naive'
  highlight?: boolean
  mib: number
  ratio: number
  kl: number
  cos: number
  top1: number
  prefill: number
  decode: number
  note: string
}

interface GemvRow {
  name: string
  barClass: 'is-vxg' | 'is-zstd' | 'is-lz4' | 'is-naive'
  highlight?: boolean
  bpw: number
  ratio: number
  sqnr: number // 45 = Bit-Exact visual cap
  bitExact?: boolean
  eqGbs: number
  actGbs: number
  speedup: number
}

const props = withDefaults(defineProps<{ compact?: boolean }>(), {
  compact: false
})

const tab = ref<TabKey>('e2e')
const e2eMetric = ref<E2eMetric>('toks')
const gemvMetric = ref<GemvMetric>('eqbw')
const tensorKey = ref<TensorKey>('embd')

const E2E_ROWS: E2eRow[] = [
  {
    name: 'VXT-CB4+Outlier',
    tag: '4.37 bpw',
    barClass: 'is-vxg',
    highlight: true,
    mib: 993.59,
    ratio: 3.62,
    kl: 0.030357,
    cos: 0.971171,
    top1: 93.8,
    prefill: 133.0,
    decode: 105.03,
    note: 'Smaller than Q4_0 · 1.76× lower KL divergence'
  },
  {
    name: 'VXT-CB3+Outlier',
    tag: '3.38 bpw',
    barClass: 'is-vxg',
    highlight: true,
    mib: 772.40,
    ratio: 4.66,
    kl: 0.111692,
    cos: 0.863824,
    top1: 100.0,
    prefill: 118.2,
    decode: 80.38,
    note: '4.66× smaller (772 MiB) · 100% Top-1 match'
  },
  {
    name: 'VXT-WHT2+Outlier (Hybrid)',
    tag: '2.65 bpw',
    barClass: 'is-vxg',
    mib: 645.96,
    ratio: 5.57,
    kl: 0.799811,
    cos: 0.646364,
    top1: 93.8,
    prefill: 140.9,
    decode: 114.17,
    note: '5.57× smaller (646 MiB) · FWHT_32 2-bit FFN'
  },
  {
    name: 'VXT-Lossless-BF16',
    tag: '11.35 bpw',
    barClass: 'is-zstd',
    mib: 2556.80,
    ratio: 1.41,
    kl: 0.0,
    cos: 1.0,
    top1: 100.0,
    prefill: 41.6,
    decode: 49.61,
    note: '100% Bit-Exact · 0.000000 nats KL'
  },
  {
    name: 'GGUF-Q4_0 (Ref)',
    tag: '4.50 bpw',
    barClass: 'is-lz4',
    mib: 1020.83,
    ratio: 3.53,
    kl: 0.053324,
    cos: 0.954276,
    top1: 87.5,
    prefill: 160.0,
    decode: 137.26,
    note: 'Uniform 4-bit baseline (higher KL error)'
  },
  {
    name: 'Raw BF16 (.gguf Ref)',
    tag: '16.00 bpw',
    barClass: 'is-naive',
    mib: 3600.80,
    ratio: 1.0,
    kl: 0.0,
    cos: 1.0,
    top1: 100.0,
    prefill: 40.2,
    decode: 38.4,
    note: 'Uncompressed 3.60 GiB ground truth'
  }
]

const GEMV_TENSORS: Record<TensorKey, { label: string; shape: string; rows: GemvRow[] }> = {
  embd: {
    label: 'token_embd [248K×2048]',
    shape: 'Qwen3.5-2B token_embd.weight · 248,320 × 2,048 · 970.00 MiB BF16',
    rows: [
      { name: 'VXT-CB4+Outlier', barClass: 'is-vxg', highlight: true, bpw: 4.40, ratio: 3.64, sqnr: 20.68, eqGbs: 710.5, actGbs: 195.4, speedup: 2.03 },
      { name: 'VXT-WHT2+Outlier', barClass: 'is-vxg', highlight: true, bpw: 2.37, ratio: 6.75, sqnr: 9.74, eqGbs: 720.0, actGbs: 106.7, speedup: 2.06 },
      { name: 'VXT-CB3+Outlier', barClass: 'is-vxg', bpw: 3.39, ratio: 4.73, sqnr: 15.40, eqGbs: 582.2, actGbs: 123.2, speedup: 1.66 },
      { name: 'VXT-Lossless-BF16', barClass: 'is-zstd', bpw: 11.35, ratio: 1.41, sqnr: 45.0, bitExact: true, eqGbs: 269.9, actGbs: 191.5, speedup: 0.77 },
      { name: 'GGUF-Q4_0-Ref', barClass: 'is-lz4', bpw: 4.50, ratio: 3.56, sqnr: 19.69, eqGbs: 995.8, actGbs: 280.1, speedup: 2.85 },
      { name: 'Raw BF16 Baseline', barClass: 'is-naive', bpw: 16.00, ratio: 1.00, sqnr: 45.0, bitExact: true, eqGbs: 349.7, actGbs: 349.7, speedup: 1.00 }
    ]
  },
  qkv: {
    label: 'attn_qkv [6144×2048]',
    shape: 'Qwen3.5-2B blk.0.attn_qkv.weight · 6,144 × 2,048 · 24.00 MiB BF16',
    rows: [
      { name: 'VXT-CB4+Outlier', barClass: 'is-vxg', highlight: true, bpw: 4.37, ratio: 3.66, sqnr: 20.82, eqGbs: 713.9, actGbs: 195.2, speedup: 1.45 },
      { name: 'VXT-CB3+Outlier', barClass: 'is-vxg', bpw: 3.38, ratio: 4.73, sqnr: 15.59, eqGbs: 400.9, actGbs: 84.8, speedup: 0.81 },
      { name: 'VXT-WHT2+Outlier', barClass: 'is-vxg', bpw: 2.31, ratio: 6.91, sqnr: 9.69, eqGbs: 312.2, actGbs: 45.2, speedup: 0.63 },
      { name: 'VXT-Lossless-BF16', barClass: 'is-zstd', bpw: 11.39, ratio: 1.40, sqnr: 45.0, bitExact: true, eqGbs: 255.6, actGbs: 181.9, speedup: 0.52 },
      { name: 'GGUF-Q4_0-Ref', barClass: 'is-lz4', bpw: 4.50, ratio: 3.56, sqnr: 19.76, eqGbs: 953.1, actGbs: 268.1, speedup: 1.94 },
      { name: 'Raw BF16 Baseline', barClass: 'is-naive', bpw: 16.00, ratio: 1.00, sqnr: 45.0, bitExact: true, eqGbs: 492.5, actGbs: 492.5, speedup: 1.00 }
    ]
  },
  ffn: {
    label: 'ffn_down [2048×6144]',
    shape: 'Qwen3.5-2B blk.0.ffn_down.weight · 2,048 × 6,144 · 24.00 MiB BF16',
    rows: [
      { name: 'VXT-CB4+Outlier', barClass: 'is-vxg', highlight: true, bpw: 4.33, ratio: 3.70, sqnr: 20.76, eqGbs: 280.3, actGbs: 75.8, speedup: 1.04 },
      { name: 'VXT-WHT2+Outlier', barClass: 'is-vxg', bpw: 2.30, ratio: 6.97, sqnr: 9.60, eqGbs: 206.6, actGbs: 29.7, speedup: 0.77 },
      { name: 'VXT-CB3+Outlier', barClass: 'is-vxg', bpw: 3.38, ratio: 4.74, sqnr: 15.56, eqGbs: 170.1, actGbs: 35.9, speedup: 0.63 },
      { name: 'VXT-Lossless-BF16', barClass: 'is-zstd', bpw: 11.32, ratio: 1.41, sqnr: 45.0, bitExact: true, eqGbs: 75.1, actGbs: 53.2, speedup: 0.28 },
      { name: 'GGUF-Q4_0-Ref', barClass: 'is-lz4', bpw: 4.50, ratio: 3.56, sqnr: 20.07, eqGbs: 460.8, actGbs: 129.6, speedup: 1.72 },
      { name: 'Raw BF16 Baseline', barClass: 'is-naive', bpw: 16.00, ratio: 1.00, sqnr: 45.0, bitExact: true, eqGbs: 268.4, actGbs: 268.4, speedup: 1.00 }
    ]
  }
}

const visibleE2e = computed(() =>
  props.compact ? E2E_ROWS.filter((r) => r.name !== 'Raw BF16 (.gguf Ref)') : E2E_ROWS
)

function e2eScore(r: E2eRow): number {
  if (e2eMetric.value === 'ratio') return r.ratio
  if (e2eMetric.value === 'toks') return r.decode
  if (e2eMetric.value === 'prefill') return r.prefill
  return 100 / (1 + r.kl * 4)
}

function e2eFmt(r: E2eRow): string {
  if (e2eMetric.value === 'ratio') return `${r.ratio.toFixed(2)}× (${r.mib.toFixed(0)} MiB)`
  if (e2eMetric.value === 'toks') return `${r.decode.toFixed(1)} tok/s`
  if (e2eMetric.value === 'prefill') return `${r.prefill.toFixed(1)} tok/s`
  return `KL ${r.kl.toFixed(4)} · Cos ${r.cos.toFixed(3)}`
}

const e2eMax = computed(() => Math.max(1e-6, ...visibleE2e.value.map(e2eScore)))
const e2ePct = (r: E2eRow) => `${Math.max(2, Math.min(100, (e2eScore(r) / e2eMax.value) * 100)).toFixed(1)}%`

const curTensor = computed(() => GEMV_TENSORS[tensorKey.value])

function gemvScore(r: GemvRow): number {
  if (gemvMetric.value === 'eqbw') return r.eqGbs
  if (gemvMetric.value === 'ratio') return r.ratio
  return r.sqnr
}

function gemvFmt(r: GemvRow): string {
  if (gemvMetric.value === 'eqbw') return `${r.eqGbs.toFixed(1)} GB/s (${r.speedup.toFixed(2)}×)`
  if (gemvMetric.value === 'ratio') return `${r.ratio.toFixed(2)}× (${r.bpw.toFixed(2)} bpw)`
  return r.bitExact ? 'Bit-Exact (∞ dB)' : `${r.sqnr.toFixed(2)} dB`
}

const gemvMax = computed(() => Math.max(1e-6, ...curTensor.value.rows.map(gemvScore)))
const gemvPct = (r: GemvRow) => `${Math.max(2, Math.min(100, (gemvScore(r) / gemvMax.value) * 100)).toFixed(1)}%`
</script>

<template>
  <div class="vxg-chart-card">
    <div class="vxg-chart-header">
      <div>
        <h4 class="vxg-chart-title">
          vxg-tensor (.vxt) — Qwen3.5-2B Compressed-Domain Apple Metal GPU Benchmarks
        </h4>
        <p class="vxg-chart-subtitle">
          {{ tab === 'e2e' ? 'Apple M2 Max · 1-CmdBuf Concurrent Metal Pipeline · 24 Layers (18 DeltaNet + 6 Gated-GQA)' : curTensor.shape }}
        </p>
      </div>
      <div class="vxg-pill-group" role="tablist" aria-label="Benchmark Mode">
        <button type="button" class="vxg-pill" :class="{ 'is-active': tab === 'e2e' }" @click="tab = 'e2e'">
          End-to-End LLM (Qwen3.5-2B)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': tab === 'gemv' }" @click="tab = 'gemv'">
          Fused GPU GEMV Kernels
        </button>
      </div>
    </div>

    <div v-if="tab === 'e2e'" class="vxg-controls-bar">
      <div class="vxg-pill-group" role="tablist" aria-label="LLM Metric">
        <button type="button" class="vxg-pill" :class="{ 'is-active': e2eMetric === 'toks' }" @click="e2eMetric = 'toks'">
          GPU Decode (tok/s)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': e2eMetric === 'ratio' }" @click="e2eMetric = 'ratio'">
          Compression (× smaller)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': e2eMetric === 'fidelity' }" @click="e2eMetric = 'fidelity'">
          Logit Fidelity (KL / Cos)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': e2eMetric === 'prefill' }" @click="e2eMetric = 'prefill'">
          GPU Prefill (tok/s)
        </button>
      </div>
    </div>

    <div v-else class="vxg-controls-bar">
      <div class="vxg-pill-group" role="tablist" aria-label="GEMV Metric">
        <button type="button" class="vxg-pill" :class="{ 'is-active': gemvMetric === 'eqbw' }" @click="gemvMetric = 'eqbw'">
          Effective BF16 BW (GB/s)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': gemvMetric === 'ratio' }" @click="gemvMetric = 'ratio'">
          Matrix Compression (×)
        </button>
        <button type="button" class="vxg-pill" :class="{ 'is-active': gemvMetric === 'sqnr' }" @click="gemvMetric = 'sqnr'">
          Weight SQNR (dB)
        </button>
      </div>
      <div class="vxg-pill-group" role="tablist" aria-label="Tensor Matrix">
        <button
          v-for="(t, k) in GEMV_TENSORS"
          :key="k"
          type="button"
          class="vxg-pill"
          :class="{ 'is-active': tensorKey === k }"
          @click="tensorKey = k"
        >
          {{ t.label }}
        </button>
      </div>
    </div>

    <div class="vxg-kpi-grid">
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">CB4+Outlier vs GGUF Q4_0</div>
        <div class="vxg-kpi-value">993.6 MiB · 105 tok/s</div>
        <div class="vxg-kpi-delta">27.2 MiB smaller &amp; 1.76× lower KL error vs Q4_0</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">Bit-Exact Lossless BF16</div>
        <div class="vxg-kpi-value">1.41× (2.50 GiB)</div>
        <div class="vxg-kpi-delta">0.000000 nats KL · 100.0% Top-1 identity</div>
      </div>
      <div class="vxg-kpi-box">
        <div class="vxg-kpi-label">CB3 / WHT2 Sub-800 MiB</div>
        <div class="vxg-kpi-value">4.66× – 5.57×</div>
        <div class="vxg-kpi-delta">772 MiB (100% Top-1) · 646 MiB (114.2 tok/s)</div>
      </div>
    </div>

    <div v-if="tab === 'e2e'" class="vxg-rows">
      <div
        v-for="row in visibleE2e"
        :key="row.name"
        class="vxg-row-card"
        :class="{ 'is-highlight': row.highlight }"
      >
        <div class="vxg-row-top">
          <span class="vxg-row-name">
            <span class="vxg-tag">{{ row.tag }}</span>{{ row.name }}
          </span>
          <span class="vxg-badge">{{ row.note }}</span>
        </div>
        <div class="vxg-bars">
          <div class="vxg-bar-line is-wide-label">
            <span class="vxg-bar-codec is-vxg">Top-1 {{ row.top1.toFixed(0) }}%</span>
            <div class="vxg-bar-track">
              <div class="vxg-bar-fill" :class="row.barClass" :style="{ width: e2ePct(row) }" />
            </div>
            <span class="vxg-bar-val is-vxg">{{ e2eFmt(row) }}</span>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="vxg-rows">
      <div
        v-for="row in curTensor.rows"
        :key="row.name"
        class="vxg-row-card"
        :class="{ 'is-highlight': row.highlight }"
      >
        <div class="vxg-row-top">
          <span class="vxg-row-name">
            <span class="vxg-tag">{{ row.bpw.toFixed(2) }} bpw</span>{{ row.name }}
          </span>
          <span class="vxg-badge">Actual DRAM {{ row.actGbs.toFixed(1) }} GB/s · {{ row.speedup.toFixed(2) }}× vs BF16</span>
        </div>
        <div class="vxg-bars">
          <div class="vxg-bar-line is-wide-label">
            <span class="vxg-bar-codec is-vxg">{{ row.ratio.toFixed(2) }}×</span>
            <div class="vxg-bar-track">
              <div class="vxg-bar-fill" :class="row.barClass" :style="{ width: gemvPct(row) }" />
            </div>
            <span class="vxg-bar-val is-vxg">{{ gemvFmt(row) }}</span>
          </div>
        </div>
      </div>
    </div>

    <p class="vxg-chart-footnote">
      Measured on Apple M2 Max Unified Memory using zero-dependency <code>vxg-tensor</code> (<code>VXT1</code> container, persistent compressed GPU planes, 1 concurrent <code>MTLCommandBuffer</code> per token).
    </p>
  </div>
</template>
