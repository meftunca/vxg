#!/usr/bin/env python3
"""
Zero-dependency, zero-copy Python integration for VXG v4 via `ctypes`.
Operates directly on caller-allocated `bytearray` / `memoryview` buffers without intermediate copies.

Usage:
    cargo build --release
    python3 examples/ffi/python_example.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path


def _find_library() -> Path:
    root = Path(__file__).resolve().parents[2]
    system = platform.system()
    if system == "Darwin":
        name = "libvxg.dylib"
    elif system == "Windows":
        name = "vxg.dll"
    else:
        name = "libvxg.so"

    candidates = [
        Path(os.environ.get("VXG_LIB_PATH", "")),
        root / "target" / "release" / name,
        Path.home() / ".cargo" / "target" / "release" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {name}; run `cargo build --release` first.")


class VxgCodec:
    def __init__(self, lib_path: Path | None = None) -> None:
        lib = ctypes.CDLL(str(lib_path or _find_library()))

        lib.vxg_version_string.argtypes = []
        lib.vxg_version_string.restype = ctypes.c_char_p

        lib.vxg_error_message.argtypes = [ctypes.c_ssize_t]
        lib.vxg_error_message.restype = ctypes.c_char_p

        lib.vxg_compress_bound.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        lib.vxg_compress_bound.restype = ctypes.c_ssize_t

        lib.vxg_frame_raw_size.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.vxg_frame_raw_size.restype = ctypes.c_ssize_t

        lib.vxg_frame_block_count.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.vxg_frame_block_count.restype = ctypes.c_ssize_t

        lib.vxg_frame_block_raw_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.vxg_frame_block_raw_size.restype = ctypes.c_ssize_t

        lib.vxg_compress_into.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.vxg_compress_into.restype = ctypes.c_ssize_t

        lib.vxg_decompress_into.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.vxg_decompress_into.restype = ctypes.c_ssize_t

        lib.vxg_extract_block_into.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.vxg_extract_block_into.restype = ctypes.c_ssize_t

        self._lib = lib

    @property
    def version(self) -> str:
        return self._lib.vxg_version_string().decode("utf-8")

    def _check(self, code: int) -> int:
        if code < 0:
            msg = self._lib.vxg_error_message(code).decode("utf-8")
            raise RuntimeError(f"VXG error ({code}): {msg}")
        return code

    def compress(
        self, data: bytes | bytearray | memoryview, threads: int = 0, block_size: int = 0, checksum: bool = True
    ) -> memoryview:
        mv = memoryview(data).cast("B")
        bound = self._check(self._lib.vxg_compress_bound(len(mv), block_size))
        out = bytearray(bound)
        src_buf = (ctypes.c_uint8 * len(mv)).from_buffer_copy(mv) if mv.readonly else (ctypes.c_uint8 * len(mv)).from_buffer(mv)
        dst_buf = (ctypes.c_uint8 * bound).from_buffer(out)
        written = self._check(
            self._lib.vxg_compress_into(
                ctypes.addressof(src_buf),
                len(mv),
                ctypes.addressof(dst_buf),
                bound,
                threads,
                block_size,
                int(checksum),
            )
        )
        return memoryview(out)[:written]

    def decompress(self, frame: bytes | bytearray | memoryview, threads: int = 0, verify_checksum: bool = True) -> bytearray:
        mv = memoryview(frame).cast("B")
        src_buf = (ctypes.c_uint8 * len(mv)).from_buffer_copy(mv) if mv.readonly else (ctypes.c_uint8 * len(mv)).from_buffer(mv)
        raw_len = self._check(self._lib.vxg_frame_raw_size(ctypes.addressof(src_buf), len(mv)))
        out = bytearray(raw_len)
        dst_buf = (ctypes.c_uint8 * raw_len).from_buffer(out)
        written = self._check(
            self._lib.vxg_decompress_into(
                ctypes.addressof(src_buf),
                len(mv),
                ctypes.addressof(dst_buf),
                raw_len,
                threads,
                int(verify_checksum),
            )
        )
        assert written == raw_len
        return out


if __name__ == "__main__":
    codec = VxgCodec()
    payload = bytearray(
        "".join(
            f'{{"ts":"2026-10-17T19:00:{i % 60:02d}Z","seq":{i},"region":"eu-central-1","status":"ok"}}\n'
            for i in range(2000)
        ).encode("utf-8")
    )
    compressed = codec.compress(payload, threads=0)
    restored = codec.decompress(compressed, threads=0)
    assert restored == payload
    ratio = len(compressed) / len(payload)
    print(
        f"VXG v{codec.version} Python FFI OK: {len(payload)} B -> {len(compressed)} B (ratio={ratio:.4f})"
    )
