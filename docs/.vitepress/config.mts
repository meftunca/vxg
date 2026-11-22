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
    ['meta', { name: 'og:description', content: 'Zero-dependency Rust compressed-domain ecosystem: VXG Data/SQL (0.1171 ratio, 15/15 vs DuckDB), VXG-KV Stateless Redis (1.85M ops/s, 20x smaller snapshots), VXT-Turbo E8 LLM (963 tok/s CUDA), and VIMG HTTP-Range Wavelet Imaging.' }]
  ],
  themeConfig: {
    logo: '/logos/vxg.svg',
    siteTitle: 'VXG v0.7.0',
    nav: [
      {
        text: 'Playgrounds (WASM)',
        items: [
          { text: 'VXG Data Studio (vxg.wasm)', link: '/playground' },
          { text: 'VIMG HTTP-Range Image Studio (vimg.wasm)', link: '/vimg-playground' }
        ]
      },
      { text: 'VXG-KV (Redis)', link: '/KV' },
      { text: 'Image (.vimg)', link: '/VIMG' },
      { text: 'AI Tensor (.vxt)', link: '/TENSOR' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Benchmarks', link: '/BENCHMARKS' },
      {
        text: 'Specifications',
        items: [
          { text: 'Wire Format (v1–v5)', link: '/FORMAT' },
          { text: 'Column Models (v3/v4)', link: '/FORMAT_V3' },
          { text: 'Stateless Redis (VXG-KV)', link: '/KV' },
          { text: 'Image Codec (VIMG1)', link: '/VIMG' },
          { text: 'AI Tensor Codec (VXT1)', link: '/TENSOR' },
          { text: 'Architecture & SIMD', link: '/guide/architecture' },
          { text: 'API & FFI', link: '/guide/api-and-ffi' },
          { text: 'Energy (GCI)', link: '/ENERGY_BENCHMARKS' },
          { text: 'Mathematical Proofs', link: '/PROOFS' },
          { text: 'Safety & Security', link: '/SAFETY' },
          { text: 'GPU Kernels', link: '/GPU' }
        ]
      }
    ],
    sidebar: [
      {
        text: 'Introduction & Ecosystem',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Live WASM Playground (VXG)', link: '/playground' },
          { text: 'Live WASM Playground (VIMG)', link: '/vimg-playground' },
          { text: 'Stateless Redis (VXG-KV)', link: '/KV' },
          { text: 'HTTP-Range Image Codec (.vimg)', link: '/VIMG' },
          { text: 'AI Tensor & Metal/CUDA LLM (.vxt)', link: '/TENSOR' },
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
