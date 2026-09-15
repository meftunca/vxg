#!/usr/bin/env python3
"""Finite-model checks + cross-implementation tests. Does not execute Rust."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
from reference import Command, InvalidFrame, bound, commands_payload, crc32c, decode, decode_block, encode, encode_block, frame

ROOT = Path(__file__).resolve().parents[1]

def golden() -> dict[str, tuple[bytes, bytes | None]]:
    cases = {'empty': (frame([]), b'')}
    def add(name, commands, raw):
        payload = commands_payload(commands)
        assert decode_block(payload, 1, len(raw)) == raw
        cases[name] = (frame([(raw, 1, payload)]), raw)
    add('fill', [Command(3, run=4096, attribute=b'Z')], b'Z'*4096)
    cmds = [Command(0, b'hello'), Command(3, b'_', 257, b'A'),
            Command(4, b'?', 259, b'BC'), Command(5, b'!', 515, b'DEFG'),
            Command(6, b':', 513, b'HIJKLMNO'), Command(1, b'tail', 321, 37),
            Command(2, b'+', 323)]
    # Compute the oracle expected result with a simple direct construction.
    out = bytearray()
    recent = 0
    for c in cmds:
        out += c.literals
        if c.op in (1, 2):
            if c.op == 1: recent = c.attribute
            for _ in range(c.run): out.append(out[-recent])
        elif c.op >= 3:
            out += (c.attribute * ((c.run+len(c.attribute)-1)//len(c.attribute)))[:c.run]
    add('all-ops', cmds, bytes(out))
    add('match-overlap', [Command(1, b'abcdefg', 2001, 7)], (b'abcdefg'*287)[:2008])
    add('rep', [Command(1, b'abcdefghijk', 301, 11), Command(2, run=301)], (b'abcdefghijk'*56)[:613])
    raw = bytes(range(256))*16 + b'z'*271
    p = commands_payload([Command(3, run=271, attribute=b'z')])
    cases['multiblock'] = (frame([(raw[:4096], 0, raw[:4096]), (raw[4096:], 1, p)], checksum=True), raw)
    for name, dist in [('distance-zero', 0), ('distance-before-prefix', 19)]:
        p = commands_payload([Command(1, b'abc', 1000, dist)])
        cases['invalid-'+name] = (frame([(b'\0'*1003, 1, p)]), None)
    p = commands_payload([Command(3, run=1001, attribute=b'a')])
    cases['invalid-output-overrun'] = (frame([(b'a'*1000, 1, p)]), None)
    p = bytearray(commands_payload([Command(3, run=1000, attribute=b'a')]))
    struct.pack_into('<I', p, 0, 2)
    cases['invalid-command-count'] = (frame([(b'a'*1000, 1, bytes(p))]), None)
    folder = ROOT/'tests'/'vectors'
    folder.mkdir(parents=True, exist_ok=True)
    for name, (f, raw) in cases.items():
        (folder/(name+'.vxg')).write_bytes(f)
        if raw is not None: (folder/(name+'.raw')).write_bytes(raw)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emulator', type=Path)
    ap.add_argument('--report', type=Path, default=ROOT/'reports'/'model-validation.json')
    args = ap.parse_args()
    counts = {}
    assert crc32c(b'123456789') == 0xe3069283
    counts['crc32c_known_answer'] = 1
    # Exhaustive over every 16-bit word, including the reserved opcode.
    for word in range(65536):
        op, lc, rc = word & 7, (word >> 3) & 31, word >> 8
        assert word == op | (lc << 3) | (rc << 8)
    counts['complete_control_word_bit_domain'] = 65536
    # Structural identity: every LZ recurrence byte equals a prefix-seed byte.
    identities = 0
    for distance in range(1, 130):
        for length in [1, 7, 8, 15, 63, 64, 65, 127, 255, 256, 257, 1025]:
            seed = list(range(distance))
            recurrence = seed.copy()
            for _ in range(length): recurrence.append(recurrence[-distance])
            assert recurrence[distance:] == [i % distance for i in range(length)]
            # CPU doubling algorithm at the symbolic-index level.
            produced = seed[:min(distance, length)]
            while len(produced) < length:
                take = min(len(produced), length-len(produced))
                produced += produced[:take]
            assert produced == recurrence[distance:]
            identities += 1
    counts['symbolic_periodicity_and_doubling_cases'] = identities
    # Exact byte content on a finite exhaustive alphabet/domain.
    binary = 0
    for n in range(11):
        for values in itertools.product(range(2), repeat=n):
            data = bytes(values)
            packed = encode(data)
            assert decode(packed) == data and len(packed) <= bound(n)
            binary += 1
    counts['exhaustive_binary_inputs_length_0_to_10'] = binary
    rng = random.Random(0x565847)
    rounds = 0
    interop_data = []
    for n in [0, 1, 7, 8, 31, 63, 255, 256, 1025, 4095, 4096, 4097, 8193]:
        for period in [0, 1, 2, 3, 4, 7, 8, 17, 63, 65, 127]:
            seed = rng.randbytes(period) if period else b''
            raw = (seed*((n+period-1)//period))[:n] if period else rng.randbytes(n)
            packed = encode(raw, checksum=rounds % 2 == 0)
            assert decode(packed) == raw and len(packed) <= bound(n)
            for start in range(0, n, 4096):
                block = raw[start:start+4096]
                for acc in [1, 4, 16, 64]:
                    _, _, s = encode_block(block, acc)
                    assert s['probes'] <= len(block)
                    assert 8*s['matches'] <= s['matched_bytes'] <= len(block)
            if n in [0, 7, 4097, 8193] and period in [0, 1, 3, 8, 65]:
                interop_data.append(raw)
            rounds += 1
    counts['boundary_random_periodic_roundtrips_and_work_bounds'] = rounds
    vectors = golden()
    valid = invalid = truncations = 0
    for name, (packed, raw) in vectors.items():
        if raw is None:
            try: decode(packed)
            except InvalidFrame: invalid += 1
            else: raise AssertionError(name+' was accepted')
        else:
            assert decode(packed) == raw
            valid += 1
            for end in range(len(packed)):
                try: decode(packed[:end])
                except InvalidFrame: truncations += 1
                else: raise AssertionError('truncation accepted')
    counts.update(golden_valid=valid, golden_invalid=invalid, rejected_truncated_golden_prefixes=truncations)
    interoperability = 0
    if args.emulator:
        executable = args.emulator.resolve()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for raw in interop_data:
                source, packed, decoded = directory/'input', directory/'packed', directory/'decoded'
                source.write_bytes(raw)
                subprocess.run([str(executable), 'c', str(source), str(packed)], check=True)
                assert decode(packed.read_bytes()) == raw
                packed.write_bytes(encode(raw))
                subprocess.run([str(executable), 'd', str(packed), str(decoded)], check=True)
                assert decoded.read_bytes() == raw
                interoperability += 2
            for name, (packed_data, raw) in vectors.items():
                if raw is None: continue  # Never send unvalidated commands to a GPU kernel.
                packed.write_bytes(packed_data)
                subprocess.run([str(executable), 'd', str(packed), str(decoded)], check=True)
                assert decoded.read_bytes() == raw, name
                interoperability += 1
        counts['cross_implementation_directions_and_golden_gpu_decode'] = interoperability
    report = {
        'status': 'PASS', 'scope': 'Python reference model and optional shared-kernel CPU SIMT emulation ONLY',
        'rust_compiled_or_executed': False, 'actual_gpu_or_shader_compiler_executed': False,
        'unbounded_formal_proof_executed': False, 'counts': counts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
