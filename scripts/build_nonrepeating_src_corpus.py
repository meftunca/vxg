#!/usr/bin/env python3
"""Builds `vxg-src-nonrepeating.rs` from unique `.rs` files in `src/` with zero repetition.

Produces the exact same byte-identical corpus as `vxg-bench --suite src-unique` so standalone
CLI tools (`vxg`, `zstd`, `lz4`) and `vxg-bench --file ...` measure non-repeating Rust source
without the synthetic `fill_repeated` periodicity of `--suite v4`'s `source-and-docs.txt`.
"""

from __future__ import annotations

import pathlib
import sys


def build_corpus(repo_root: pathlib.Path) -> bytes:
    src_dir = repo_root / "src"
    entries: list[tuple[str, pathlib.Path]] = []
    for p in src_dir.rglob("*.rs"):
        if p.is_file():
            rel = p.relative_to(repo_root).as_posix()
            entries.append((rel, p))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    chunks: list[bytes] = []
    for rel, path in entries:
        chunks.append(f"// === {rel} ===\n".encode("utf-8"))
        raw = path.read_bytes().replace(b"\r", b"")
        chunks.append(raw)
        if not raw.endswith(b"\n"):
            chunks.append(b"\n")
    return b"".join(chunks)


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_path = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else pathlib.Path("/private/tmp/vxg-covid-bench/vxg-src-nonrepeating.rs")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_corpus(repo_root)
    out_path.write_bytes(data)
    print(f"Wrote {out_path} ({len(data):,d} bytes, {len(data) / 1e6:.3f} MB, 0% repeated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
