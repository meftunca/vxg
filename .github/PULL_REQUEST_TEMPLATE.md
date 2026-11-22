## Summary

<!-- Describe what this PR changes and why. -->

## Module(s) Touched

- [ ] `vxg` (Core codec / SQL / Arrow / HTTP Range)
- [ ] `vxg-kv` (Stateless snapshot Redis)
- [ ] `vxg-tensor` / `vxt-studio` (AI tensor codec & inference)
- [ ] `vimg` (Wavelet image / animation codec)
- [ ] `packages/*` (WASM) or `docs/`

## Verification Checklist

- [ ] Zero new external dependencies in core crates (`vxg`, `vxg-kv`, `vimg`)
- [ ] Modified `.rs` files remain `<= 600` lines
- [ ] `cargo clippy --no-deps --all-targets -- -D warnings` passes with 0 warnings
- [ ] Release tests pass (`cargo test --release`)
- [ ] `python3 scripts/render_tables.py --check` passes (if touching docs/reports)
