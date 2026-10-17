#!/usr/bin/env python3
"""Deterministic robustness controls for the Q95-r2 members.

Builds two source-agnostic control inputs for each requested r2 member without
touching the frozen corpus or its manifest:

- ``*-renamed``: field names permuted to unrelated short names; values and rows
  stay byte-identical otherwise.
- ``*-shuffled``: rows deterministically shuffled with a fixed seed; field names
  and values stay identical.

The controls are inputs for the bench-only column evaluator. A model class that
only wins because of a field name or row order must fall back to RAW or pay its
own residuals on these files; the evaluator itself already enforces exact
reconstruction for every packet.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "corpus" / "qualification" / "r2"
DEFAULT_OUTPUT = ROOT / "corpus" / "qualification" / "r2-controls"

RENAMES = (
    (b'"request_id":', b'"rid":'),
    (b'"sequence":', b'"seq":'),
    (b'"message":', b'"msg":'),
    (b'"source":', b'"src":'),
    (b'"record":', b'"rec":'),
    (b'"event":', b'"evt":'),
    (b'"region":', b'"rgn":'),
)


def rename(block: bytes) -> bytes:
    for old, new in RENAMES:
        block = block.replace(old, new)
    return block


def shuffled(block: bytes, seed: int) -> bytes:
    lines = block.splitlines(keepends=True)
    random.Random(seed).shuffle(lines)
    return b"".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--member",
        action="append",
        default=None,
        help="member filename without extension; defaults to the two revised families at 16 MiB",
    )
    args = parser.parse_args()
    members = args.member or [
        "event-mix-identifiers-16MiB",
        "bursty-variable-records-16MiB",
    ]
    import hashlib

    args.output.mkdir(parents=True, exist_ok=True)
    for member in members:
        source = args.source / f"{member}.ndjson"
        data = source.read_bytes()
        for suffix, transform in (
            ("renamed", rename(data)),
            ("shuffled", shuffled(data, 0x5EED_1234)),
        ):
            path = args.output / f"{member}-{suffix}.ndjson"
            path.write_bytes(transform)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{path.relative_to(ROOT)} sha256={digest}")


if __name__ == "__main__":
    main()
