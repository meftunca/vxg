"""Independent, deliberately slow VXG specification model. No native code imports.

This is a correctness oracle, not a production codec or performance benchmark.
Decode MATCH uses the bytewise LZ recurrence, unlike the optimized native kernels.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass


class InvalidFrame(ValueError):
    pass


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for x in data:
        crc ^= x
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def bound(n: int, block_size: int = 4096) -> int:
    if n < 0 or not 4096 <= block_size <= 1048576 or block_size % 64:
        raise ValueError("invalid size")
    return n + 32 + 32 * ((n + block_size - 1) // block_size)


def token(op: int, lit: int, run: int) -> bytes:
    assert 0 <= op < 7 and 0 <= lit < 2**32 and 0 <= run < 2**32
    out = struct.pack('<H', op | (min(lit, 31) << 3) | (min(run, 255) << 8))
    if lit >= 31:
        out += struct.pack('<I', lit)
    if run >= 255:
        out += struct.pack('<I', run)
    return out


@dataclass(frozen=True)
class Command:
    op: int
    literals: bytes = b''
    run: int = 0
    attribute: int | bytes = 0


def commands_payload(commands: list[Command]) -> bytes:
    control, attrs, literals = bytearray(), bytearray(), bytearray()
    for c in commands:
        control += token(c.op, len(c.literals), c.run)
        literals += c.literals
        if c.op == 1:
            attrs += struct.pack('<I', c.attribute)
        elif 3 <= c.op <= 6:
            period = 1 << (c.op - 3)
            seed = bytes(c.attribute)
            assert len(seed) == period
            attrs += seed.ljust(max(4, period), b'\0')
    return struct.pack('<IIII', len(commands), len(control), len(attrs), len(literals)) + control + attrs + literals


def decode_block(payload: bytes, kind: int, n: int) -> bytes:
    if kind == 0:
        if len(payload) != n:
            raise InvalidFrame('raw length')
        return payload
    if kind != 1 or len(payload) < 16 or len(payload) >= n:
        raise InvalidFrame('compressed length/kind')
    count, cs, ds, ls = struct.unpack_from('<IIII', payload)
    if 16 + cs + ds + ls != len(payload) or not 0 < count <= min(n, cs // 2):
        raise InvalidFrame('stream sizes/count')
    c, a, literals = payload[16:16+cs], payload[16+cs:16+cs+ds], payload[16+cs+ds:]
    ci = ai = li = recent = 0
    output = bytearray()
    def take32(stream: bytes, pos: int) -> tuple[int, int]:
        if pos + 4 > len(stream):
            raise InvalidFrame('truncated integer')
        return struct.unpack_from('<I', stream, pos)[0], pos + 4
    for _ in range(count):
        if ci + 2 > len(c):
            raise InvalidFrame('truncated token')
        word = struct.unpack_from('<H', c, ci)[0]
        ci += 2
        op, lc, rc = word & 7, (word >> 3) & 31, word >> 8
        if op == 7:
            raise InvalidFrame('reserved opcode')
        lit, run = lc, rc
        if lc == 31:
            lit, ci = take32(c, ci)
            if lit < 31:
                raise InvalidFrame('noncanonical literal')
        if rc == 255:
            run, ci = take32(c, ci)
            if run < 255:
                raise InvalidFrame('noncanonical run')
        if (op == 0 and (run != 0 or lit == 0)) or (op != 0 and run == 0):
            raise InvalidFrame('progress')
        if op in (1, 2) and run < 8:
            raise InvalidFrame('minmatch')
        if len(output) + lit + run > n or li + lit > len(literals):
            raise InvalidFrame('output/literal overflow')
        output += literals[li:li+lit]
        li += lit
        if op in (1, 2):
            if op == 1:
                recent, ai = take32(a, ai)
            if recent == 0 or recent > len(output):
                raise InvalidFrame('invalid distance')
            # The mathematical recurrence. No periodic shortcut or memcpy.
            for _ in range(run):
                output.append(output[-recent])
        elif 3 <= op <= 6:
            period = 1 << (op - 3)
            size = max(4, period)
            if ai + size > len(a):
                raise InvalidFrame('seed truncation')
            seed = a[ai:ai+period]
            if any(a[ai+period:ai+size]):
                raise InvalidFrame('seed padding')
            ai += size
            output += (seed * ((run + period - 1) // period))[:run]
    if (ci, ai, li, len(output)) != (len(c), len(a), len(literals), n):
        raise InvalidFrame('unconsumed streams/output')
    return bytes(output)


def frame(parts: list[tuple[bytes, int, bytes]], block_size: int = 4096, checksum: bool = False) -> bytes:
    """Parts are (uncompressed bytes, kind, payload). Caller controls validity."""
    n = sum(len(raw) for raw, _, _ in parts)
    header = struct.pack('<4sHHQIIQ', b'VXG1', 1, int(checksum), n, block_size, len(parts), 0)
    directory, chunks = bytearray(), bytearray()
    ro, po = 0, 32 + 32 * len(parts)
    for raw, kind, payload in parts:
        directory += struct.pack('<QQIIII', ro, po, len(raw), len(payload), crc32c(raw) if checksum else 0, kind)
        chunks += payload
        ro += len(raw)
        po += len(payload)
    return header + directory + chunks


def decode(data: bytes, max_output: int = 1 << 26) -> bytes:
    if len(data) < 32:
        raise InvalidFrame('header')
    magic, version, flags, n, bs, count, reserved = struct.unpack_from('<4sHHQIIQ', data)
    if magic != b'VXG1' or version != 1 or flags & ~1 or reserved:
        raise InvalidFrame('header fields')
    if n > max_output or not 4096 <= bs <= 1048576 or bs % 64:
        raise InvalidFrame('limits')
    if count != (n + bs - 1) // bs or 32 + count * 32 > len(data):
        raise InvalidFrame('directory count')
    output, next_payload = bytearray(), 32 + 32 * count
    for i in range(count):
        ro, po, rn, cn, crc, kind = struct.unpack_from('<QQIIII', data, 32 + i*32)
        if ro != len(output) or po != next_payload or rn != min(bs, n-len(output)):
            raise InvalidFrame('directory canonicality')
        if po + cn > len(data) or (not flags and crc):
            raise InvalidFrame('payload range/checksum flag')
        raw = decode_block(data[po:po+cn], kind, rn)
        if flags and crc32c(raw) != crc:
            raise InvalidFrame('checksum')
        output += raw
        next_payload += cn
    if next_payload != len(data) or len(output) != n:
        raise InvalidFrame('trailing bytes')
    return bytes(output)


def encode_block(data: bytes, acceleration: int = 1) -> tuple[int, bytes, dict]:
    """Reference greedy encoder, independently written; not a Rust execution."""
    if not 1 <= acceleration <= 64:
        raise ValueError('acceleration')
    table = {}
    pos = anchor = recent = misses = probes = matched = 0
    commands = []
    while pos + 8 <= len(data):
        word = int.from_bytes(data[pos:pos+8], 'little')
        h = ((word * 0x9e3779b185ebca87) & ((1 << 64) - 1)) >> 50
        candidate = table.get(h)
        table[h] = pos
        probes += 1
        if candidate is None or candidate >= pos or data[candidate:candidate+8] != data[pos:pos+8]:
            misses += 1
            pos += min(64, acceleration + (misses >> 6))
            continue
        run = 8
        while pos + run < len(data) and data[pos+run] == data[candidate+run]:
            run += 1
        distance = pos - candidate
        op = {1: 3, 2: 4, 4: 5, 8: 6}.get(distance, 2 if distance == recent else 1)
        attribute: int | bytes = 0
        if op == 1:
            attribute = recent = distance
        elif 3 <= op <= 6:
            attribute = data[pos:pos+(1 << (op-3))]
        commands.append(Command(op, data[anchor:pos], run, attribute))
        matched += run
        pos += run
        anchor = pos
        misses = 0
    match_count = len(commands)
    if anchor < len(data):
        commands.append(Command(0, data[anchor:]))
    encoded = commands_payload(commands)
    stats = {'probes': probes, 'matches': match_count, 'matched_bytes': matched}
    return (1, encoded, stats) if len(encoded) < len(data) else (0, data, stats)


def encode(data: bytes, block_size: int = 4096, checksum: bool = False, acceleration: int = 1) -> bytes:
    parts = []
    for start in range(0, len(data), block_size):
        raw = data[start:start+block_size]
        kind, payload, _ = encode_block(raw, acceleration)
        parts.append((raw, kind, payload))
    return frame(parts, block_size, checksum)
