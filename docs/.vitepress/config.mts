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
    /\.\.\/examples\//,
    /\.\.\/reports\//,
    /\.\.\/\.github\//
  ],
  head: [
    ['meta', { name: 'theme-color', content: '#10b981' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'VXG — Vectorized Execution Grammar Codec' }],
    ['meta', { name: 'og:description', content: '5x smaller than LZ4, 2.4x smaller than Zstd, and up to 54 GB/s multi-threaded decompression with O(1) random block access.' }]
  ],
  themeConfig: {
    siteTitle: 'VXG v0.4',
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'API & FFI', link: '/guide/api-and-ffi' },
      { text: 'Benchmarks', link: '/BENCHMARKS' },
      { text: 'Energy (GCI)', link: '/ENERGY_BENCHMARKS' },
      {
        text: 'Specifications',
        items: [
          { text: 'Wire Format (v1–v4)', link: '/FORMAT' },
          { text: 'Column Models (v3/v4)', link: '/FORMAT_V3' },
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
          { text: 'Getting Started & CLI', link: '/guide/getting-started' },
          { text: 'Architecture & SIMD Internals', link: '/guide/architecture' },
          { text: 'Rust API, Streaming & FFI', link: '/guide/api-and-ffi' },
          { text: 'Cross-Language FFI Reference', link: '/FFI' }
        ]
      },
      {
        text: 'Benchmarks & Energy',
        items: [
          { text: 'Cross-Platform Benchmarks', link: '/BENCHMARKS' },
          { text: 'Energy & Composite Indices (VPI / GCI)', link: '/ENERGY_BENCHMARKS' },
          { text: 'Validation & Reproducibility', link: '/VALIDATION' }
        ]
      },
      {
        text: 'Wire Format & Proofs',
        items: [
          { text: 'VXG1 Wire Specification (v1–v4)', link: '/FORMAT' },
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
