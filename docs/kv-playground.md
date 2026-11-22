---
layout: page
title: VXG-KV WASM Playground (IndexedDB & localStorage)
description: Run real pure-Rust vxg-kv (RESP2/RESP3 Redis + JSON/Geo + Streams + Time-Travel + VXG.SQL) in your browser backed by IndexedDB or localStorage.
---

<div class="vp-doc" style="max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 3rem;">
  <div style="display:flex; align-items:center; gap:0.85rem; margin-bottom:0.5rem;">
    <img src="/logos/vxg-kv.svg" width="48" height="48" alt="VXG-KV" />
    <div>
      <h1 style="margin:0; border:none; padding:0;">VXG-KV Browser Playground (WASM)</h1>
      <p style="margin:0.2rem 0 0; color:var(--vp-c-text-2);">
        Real pure-Rust <code>vxg-kv v0.3.0</code> running locally in WebAssembly with two pods sharing an <strong>IndexedDB</strong> or <strong>localStorage</strong> VXG Object Store.
      </p>
    </div>
  </div>

  <KvPlayground />

  <h2>How Browser Persistence Works</h2>
  <p>
    Every <code>VXG.COMMIT</code> writes a compressed WAL segment and CAS-updates <code>manifest.json</code>; every <code>VXG.COMPACT</code> merges live keys into a sorted <code>VXG v5</code> snapshot (<code>snap_*.vxg</code>) plus a per-block 10-bit/key Bloom &amp; Fence index (<code>snap_*.vxi</code>).
    Those exact compressed objects are saved directly into your browser's <strong>IndexedDB</strong> or <strong>localStorage</strong>.
    Refresh the page (or click <strong>Cold Reopen</strong>) and switch to <strong>Pod B</strong> to watch a cold pod serve point reads with single-block range fetches, filter misses with 0-byte Bloom checks, query historical epochs with <code>VXG.AT</code>, and run <code>VXG.SQL</code> analytics without any server!
  </p>
</div>
