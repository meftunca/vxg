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
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/vxg/logos/vxg.svg' }],
    ['meta', { name: 'theme-color', content: '#10b981' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'VXG — Unified Compressed-Domain Engine (Data/SQL · VXG-KV · VXT-Turbo · VIMG)' }],
    ['meta', { name: 'og:description', content: 'Zero-dependency Rust compressed-domain ecosystem: VXG Data/SQL (0.1171 ratio, 15/15 vs DuckDB), VXG-KV Stateless Redis (1.85M ops/s, 20x smaller snapshots), VXT-Turbo E8 LLM (963 tok/s CUDA), and VIMG HTTP-Range Wavelet Imaging.' }],
    ['meta', { name: 'og:image', content: 'https://meftunca.github.io/vxg/social-preview.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: 'https://meftunca.github.io/vxg/social-preview.png' }]
  ],
  themeConfig: {
    logo: '/logos/vxg.svg',
    siteTitle: 'VXG v0.7.0',
    nav: [
      {
        text: 'Modules',
        items: [
          { text: 'VXG Core (Data & SQL · v0.7.0)', link: '/guide/getting-started' },
          { text: 'VXG-KV (Stateless Redis · v0.3.0)', link: '/KV' },
          { text: 'VXT-Turbo (E8 AI LLM · v0.6.0)', link: '/TENSOR' },
          { text: 'VIMG (Wavelet Imaging · v0.4.0)', link: '/VIMG' }
        ]
      },
      {
        text: 'Playgrounds',
        items: [
          { text: 'VXG SQL Studio (WASM)', link: '/playground' },
          { text: 'VIMG Image Studio (WASM)', link: '/vimg-playground' }
        ]
      },
      { text: 'Quick Start', link: '/guide/getting-started' },
      { text: 'Benchmarks', link: '/CHIMERA_BENCHMARKS' },
      {
        text: 'Specs',
        items: [
          { text: 'Wire Format (v1–v5)', link: '/FORMAT' },
          { text: 'Architecture & SIMD', link: '/guide/architecture' },
          { text: 'Rust API & FFI / Arrow', link: '/guide/api-and-ffi' },
          { text: 'Safety & Security', link: '/SAFETY' },
          { text: 'Roadmap', link: '/ROADMAP' }
        ]
      }
    ],
    sidebar: [
      {
        text: 'Start Here',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Quick Start & CLI', link: '/guide/getting-started' },
          { text: 'SQL Playground (WASM)', link: '/playground' },
          { text: 'Image Studio (WASM)', link: '/vimg-playground' }
        ]
      },
      {
        text: 'Modules',
        items: [
          { text: 'VXG Core (Data, SQL & Range)', link: '/guide/getting-started' },
          { text: 'VXG-KV (Stateless Redis)', link: '/KV' },
          { text: 'VXT-Turbo (AI LLM / Metal / CUDA)', link: '/TENSOR' },
          { text: 'VIMG (Wavelet Image & Animation)', link: '/VIMG' }
        ]
      },
      {
        text: 'Benchmarks',
        items: [
          { text: 'Competitor Sweep & 40M DuckDB', link: '/CHIMERA_BENCHMARKS' },
          { text: 'Cross-Platform CI Baseline', link: '/BENCHMARKS' },
          { text: 'Energy & Green Codec Index', link: '/ENERGY_BENCHMARKS' },
          { text: 'Validation & Reproducibility', link: '/VALIDATION' }
        ]
      },
      {
        text: 'Specs & Deep Dive',
        collapsed: true,
        items: [
          { text: 'Architecture & SIMD Internals', link: '/guide/architecture' },
          { text: 'Rust API, Analytics & FFI', link: '/guide/api-and-ffi' },
          { text: 'Cross-Language FFI Reference', link: '/FFI' },
          { text: 'VXG1 Wire Specification (v1–v5)', link: '/FORMAT' },
          { text: 'Structured Column Models', link: '/FORMAT_V3' },
          { text: 'Compressed-Domain Engine', link: '/STAGE2_DESIGN' },
          { text: 'Mathematical Proofs', link: '/PROOFS' },
          { text: 'Safety & Security Boundary', link: '/SAFETY' },
          { text: 'Native GPU Architecture', link: '/GPU' },
          { text: 'Roadmap', link: '/ROADMAP' },
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
