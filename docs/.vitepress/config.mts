import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'VXG Codec',
  description: 'Vectorized Execution Grammar — Multi-GB/s Block-Independent Lossless Compression for Structured Telemetry, Logs, Prose & Images',
  base: '/vxg/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: [
    /^https?:\/\/localhost/,
    /\.\.\/include\//,
    /\.\.\/src\//,
    /\.\.\/scripts\//,
    /\.\.\/examples\//,
    /\.\.\/reports\//,
    /\.\.\/tests\//,
    /\.\.\/archive\//,
    /\.\.\/\.github\//
  ],
  head: [
    ['meta', { name: 'theme-color', content: '#10b981' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'VXG — Vectorized Execution Grammar Codec' }],
    ['meta', { name: 'og:description', content: 'Store raw NDJSON/CSV verbatim, 0.1172 6-dataset real public geomean (beating Zstd-9 & Zstd+Dict-9 [32..256K] 0.1181), PFOR-Delta time-series (Covid-19 0.0963 vs Zstd-9 0.1415), SQL & Arrow/DuckDB scan up to 40M+ rows, 100% byte-exact reversible.' }]
  ],
  themeConfig: {
    siteTitle: 'VXG v0.5.2',
    nav: [
      { text: 'Playground (WASM)', link: '/playground' },
      { text: 'AI Tensor (.vxt)', link: '/TENSOR' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'API & FFI', link: '/guide/api-and-ffi' },
      { text: 'Benchmarks', link: '/BENCHMARKS' },
      { text: 'Energy (GCI)', link: '/ENERGY_BENCHMARKS' },
      {
        text: 'Specifications',
        items: [
          { text: 'Wire Format (v1–v5)', link: '/FORMAT' },
          { text: 'Column Models (v3/v4)', link: '/FORMAT_V3' },
          { text: 'AI Tensor Codec (VXT1)', link: '/TENSOR' },
          { text: 'Mathematical Proofs', link: '/PROOFS' },
          { text: 'Safety & Security', link: '/SAFETY' },
          { text: 'GPU Kernels', link: '/GPU' }
        ]
      }
    ],
    sidebar: [
      {
        text: 'Introduction & Guide',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Live WASM Playground', link: '/playground' },
          { text: 'AI Tensor & Metal LLM (.vxt)', link: '/TENSOR' },
          { text: 'Getting Started & CLI', link: '/guide/getting-started' },
          { text: 'Architecture & SIMD Internals', link: '/guide/architecture' },
          { text: 'Compressed-Domain Engine', link: '/STAGE2_DESIGN' },
          { text: 'Rust API, Analytics & FFI', link: '/guide/api-and-ffi' },
          { text: 'Cross-Language FFI Reference', link: '/FFI' },
          { text: 'Roadmap', link: '/ROADMAP' }
        ]
      },
      {
        text: 'Benchmarks & Energy',
        items: [
          { text: 'Cross-Platform Benchmarks', link: '/BENCHMARKS' },
          { text: 'Competitor Sweep & 40M DuckDB', link: '/CHIMERA_BENCHMARKS' },
          { text: 'Energy & Composite Indices (VPI / GCI)', link: '/ENERGY_BENCHMARKS' },
          { text: 'Validation & Reproducibility', link: '/VALIDATION' }
        ]
      },
      {
        text: 'Wire Format & Proofs',
        items: [
          { text: 'VXG1 Wire Specification (v1–v5)', link: '/FORMAT' },
          { text: 'Structured Column Models', link: '/FORMAT_V3' },
          { text: 'Mathematical Contracts & Proofs', link: '/PROOFS' },
          { text: 'Safety & Security Boundary', link: '/SAFETY' },
          { text: 'Native GPU Architecture', link: '/GPU' }
        ]
      },
      {
        text: 'Engineering History',
        collapsed: true,
        items: [
          { text: 'Release Progress Log', link: '/PROGRESS' },
          { text: 'References', link: '/REFERENCES' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/meftunca/vxg' }
    ],
    search: {
      provider: 'local'
    },
    footer: {
      message: 'Released under the MIT or Apache-2.0 License.',
      copyright: 'Copyright © 2026 VXG Contributors'
    }
  }
})
