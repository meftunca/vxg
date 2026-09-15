#!/usr/bin/env python3
"""Verify the supplied source manifest. This does not compile or test the codec."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
count=0
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    expected, relative=line.split('  ',1)
    path=(root/relative).resolve()
    if root.resolve() not in path.parents:
        raise SystemExit(f'unsafe manifest path: {relative}')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'missing or changed: {relative}')
    count+=1
print(f'PASS: {count} file hashes. Integrity check only, not compilation or authenticity.')
