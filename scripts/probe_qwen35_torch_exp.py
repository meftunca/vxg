#!/usr/bin/env python3
"""Exact CPU torch float32 A_log oracle; never imports the VXG codec."""
import argparse
import hashlib
import json
import mmap
import platform
import struct
import sys
from pathlib import Path

import torch
from compare_qwen35_source import gguf, target_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument('official')
    p.add_argument('gguf')
    p.add_argument('output')
    a = p.parse_args()
    torch.set_num_threads(1)
    results = []
    with open(a.official, 'rb') as sf, open(a.gguf, 'rb') as gf:
        with mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ) as st, mmap.mmap(gf.fileno(), 0, access=mmap.ACCESS_READ) as gg:
            h = struct.unpack_from('<Q', st)[0]
            header = json.loads(st[8:8+h])
            targets = gguf(gg)
            for name, s in header.items():
                if not name.startswith('model.language_model.') or not name.endswith('.A_log'):
                    continue
                dest = target_name(name)
                t = targets[dest]
                assert s['dtype'] == 'F32' and t['kind'] == 0
                assert s['shape'] == t['shape'] == [16]
                start, end = s['data_offsets']
                raw = st[8+h+start:8+h+end]
                assert len(raw) == t['size'] == 64
                # Copy raw F32 bytes directly, then evaluate the whole tensor.
                x = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
                assert x.device.type == 'cpu' and x.dtype == torch.float32
                y = -torch.exp(x)
                bits = y.view(torch.int32).tolist()
                expected = b''.join(struct.pack('<I', b & 0xffffffff) for b in bits)
                actual = gg[t['start']:t['start']+t['size']]
                mismatches = [dict(index=i, source_hex=raw[4*i:4*i+4].hex(), torch_hex=expected[4*i:4*i+4].hex(), gguf_hex=actual[4*i:4*i+4].hex()) for i in range(16) if expected[4*i:4*i+4] != actual[4*i:4*i+4]]
                results.append(dict(source=name, target=dest, shape=s['shape'], numel=16, source_hex=raw.hex(), torch_hex=expected.hex(), gguf_hex=actual.hex(), mismatches=mismatches))
    assert len(results) == 18 and sum(r['numel'] for r in results) == 288
    report = dict(torch_version=torch.__version__, torch_git_version=torch.version.git_version, torch_build=torch.__config__.show(), python=sys.version, platform=platform.platform(), machine=platform.machine(), device='cpu', dtype='torch.float32', threads=torch.get_num_threads(), operation='-torch.exp(x), vectorized separately for each complete tensor', exact_comparison='all output bytes, no rounding or tolerance', official=str(Path(a.official).resolve()), gguf=str(Path(a.gguf).resolve()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), mapping_script_sha256=hashlib.sha256(Path(__file__).with_name('compare_qwen35_source.py').read_bytes()).hexdigest(), tensor_count=len(results), element_count=288, mismatch_tensors=sum(bool(r['mismatches']) for r in results), mismatch_elements=sum(len(r['mismatches']) for r in results), limitation='Reproducibility with this current torch build does not establish historical converter identity.', results=results)
    Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'torch_build')}, indent=2))
    return int(report['mismatch_elements'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
