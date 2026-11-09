#!/usr/bin/env python3
"""Zero-dependency DuckDB / Polars / PyArrow scan bridge for `.vxg` frames via Arrow PyCapsule.

Exposes any `.vxg` file or in-memory frame via the standard Apache Arrow C Stream PyCapsule
protocol (`__arrow_c_stream__`), allowing `DuckDB >= 1.1`, `Polars`, and `PyArrow` to query
compressed `.vxg` archives directly with **zero `pyarrow` / `numpy` / `pandas` dependencies**.

Usage in Python + DuckDB:
    import duckdb
    from duckdb_read_vxg import read_vxg

    con = duckdb.connect()
    trips = read_vxg("nyc-taxi.ndjson.vxg", where="payment_type = 1 AND tip_cents >= 1000")
    print(con.execute("SELECT payment_type, COUNT(*), SUM(total_cents) FROM trips GROUP BY 1").fetchall())
"""

from __future__ import annotations

import ctypes
import mmap
import os
import platform
import sys
from pathlib import Path
from typing import Iterable


def _default_lib_path() -> str:
    env_override = os.environ.get("VXG_LIB_PATH")
    if env_override and os.path.exists(env_override):
        return env_override
    ext = {"Darwin": "libvxg.dylib", "Windows": "vxg.dll"}.get(platform.system(), "libvxg.so")
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    if os.environ.get("CARGO_TARGET_DIR"):
        candidates.append(Path(os.environ["CARGO_TARGET_DIR"]) / "release" / ext)
    candidates.extend([
        repo_root / "target" / "release" / ext,
        Path.home() / ".cargo" / "target" / "release" / ext,
    ])
    existing = [c for c in candidates if c.exists()]
    if existing:
        return str(max(existing, key=lambda p: p.stat().st_mtime))
    raise FileNotFoundError(
        f"Could not locate {ext}; run `cargo build --release --lib` or set VXG_LIB_PATH"
    )


_STREAM_RELEASE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _ArrowArrayStream(ctypes.Structure):
    _fields_ = [
        ("get_schema", ctypes.c_void_p),
        ("get_next", ctypes.c_void_p),
        ("get_last_error", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


# Keep live PyCapsule name, destructor callback, and ctypes definitions at module scope.
# IMPORTANT: `capsule_dealloc` invokes the destructor when `ob_refcnt == 0`, so the callback
# parameter MUST be `c_void_p` (never `py_object`, which would `Py_DECREF` a zero-refcnt object).
_CAPSULE_NAME_BUF = ctypes.create_string_buffer(b"arrow_array_stream")
_PYCAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _PYCAPSULE_DESTRUCTOR]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p

_ACTIVE_STREAMS: dict[int, _ArrowArrayStream] = {}


@_PYCAPSULE_DESTRUCTOR
def _release_stream_capsule(capsule_ptr: int) -> None:
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule_ptr, _CAPSULE_NAME_BUF)
    st = _ACTIVE_STREAMS.pop(ptr, None)
    if st is not None and st.release:
        rel_fn = _STREAM_RELEASE_FN(st.release)
        rel_fn(ctypes.addressof(st))


def _bind_vxg(lib_path: str | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path or _default_lib_path())
    lib.vxg_compress_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_compress_alloc.restype = ctypes.c_void_p
    lib.vxg_sql_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_sql_alloc.restype = ctypes.c_void_p
    lib.vxg_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vxg_free.restype = None
    lib.vxg_arrow_export_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(_ArrowArrayStream),
    ]
    lib.vxg_arrow_export_stream.restype = ctypes.c_ssize_t
    return lib


class VxgArrowDataset:
    """Zero-copy mmap / buffer wrapper implementing the Arrow PyCapsule `__arrow_c_stream__` protocol."""

    def __init__(
        self,
        source: str | os.PathLike | bytes,
        *,
        sql: str | None = None,
        columns: Iterable[str] | None = None,
        where: str | None = None,
        threads: int = 0,
        lib_path: str | None = None,
    ) -> None:
        self._lib = _bind_vxg(lib_path)
        self._threads = threads
        self._file = None
        self._mmap = None
        self._buf = None

        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
            self._buf = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
            self._ptr = ctypes.addressof(self._buf)
            self._len = len(raw)
        else:
            f = open(source, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
            buf = (ctypes.c_char * len(mm)).from_buffer(mm)
            self._file = f
            self._mmap = mm
            self._buf = buf
            self._ptr = ctypes.addressof(buf)
            self._len = len(mm)

        if sql is not None:
            self._sql_bytes = sql.encode("utf-8")
        elif columns is not None or where is not None:
            proj = ", ".join(columns) if columns is not None else "*"
            q = f"SELECT {proj} FROM frame"
            if where:
                q += f" WHERE {where}"
            self._sql_bytes = q.encode("utf-8")
        else:
            self._sql_bytes = None

    def close(self) -> None:
        self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "VxgArrowDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __arrow_c_stream__(self, requested_schema=None):
        _ = requested_schema
        st = _ArrowArrayStream()
        rc = self._lib.vxg_arrow_export_stream(
            self._ptr,
            self._len,
            self._sql_bytes,
            self._threads,
            ctypes.byref(st),
        )
        if rc != 0:
            raise RuntimeError(f"vxg_arrow_export_stream failed with status {rc}")
        addr = ctypes.addressof(st)
        _ACTIVE_STREAMS[addr] = st
        return ctypes.pythonapi.PyCapsule_New(
            addr, _CAPSULE_NAME_BUF, _release_stream_capsule
        )


def read_vxg(
    source: str | os.PathLike | bytes,
    *,
    sql: str | None = None,
    columns: Iterable[str] | None = None,
    where: str | None = None,
    threads: int = 0,
    lib_path: str | None = None,
) -> VxgArrowDataset:
    """Create an Arrow-stream-compatible relation over a `.vxg` path or byte buffer."""
    return VxgArrowDataset(
        source,
        sql=sql,
        columns=columns,
        where=where,
        threads=threads,
        lib_path=lib_path,
    )


def _self_test() -> None:
    import duckdb

    lib = _bind_vxg()

    # 1. Build an NDJSON frame with Int64, Float64, Boolean, Utf8, and NULL values
    ndjson_lines = []
    for i in range(600):
        pay = 1 + (i % 3)
        tip = "null" if (i % 7 == 0) else str((i * 25) % 900)
        total = 1000 + (i * 40) % 3500
        ratio = round((i % 19) * 0.125, 3)
        ok = "true" if (i % 2 == 0) else "false"
        zone_id = 10 + (i % 5)
        ndjson_lines.append(
            f'{{"id":{i},"zone_id":{zone_id},"payment_type":{pay},"tip_cents":{tip},"total_cents":{total},"surge":{ratio},"verified":{ok}}}\n'
        )
    ndjson_raw = "".join(ndjson_lines).encode("utf-8")

    out_len = ctypes.c_size_t(0)
    p1 = lib.vxg_compress_alloc(ndjson_raw, len(ndjson_raw), 2, 16384, 1, ctypes.byref(out_len))
    assert p1
    ndjson_vxg = ctypes.string_at(p1, out_len.value)
    lib.vxg_free(p1, out_len.value)

    # 2. Build a CSV lookup table frame (`zone_id,borough,tax_bps`)
    csv_raw = (
        "zone_id,borough,tax_bps\n"
        "10,Manhattan,888\n"
        "11,Queens,888\n"
        "12,Brooklyn,888\n"
        "13,Bronx,800\n"
        "14,Staten Island,800\n"
    ).encode("utf-8")
    p2 = lib.vxg_compress_alloc(csv_raw, len(csv_raw), 1, 16384, 1, ctypes.byref(out_len))
    assert p2
    csv_vxg = ctypes.string_at(p2, out_len.value)
    lib.vxg_free(p2, out_len.value)

    con = duckdb.connect()
    trips = read_vxg(ndjson_vxg, threads=2)
    zones = read_vxg(csv_vxg, threads=1)

    # Verify NULL preservation + Boolean + Int64 + Float64 + cross-archive SQL JOIN inside DuckDB!
    res_nulls = con.execute(
        "SELECT COUNT(*), COUNT(tip_cents), SUM(CASE WHEN verified THEN 1 ELSE 0 END) FROM trips"
    ).fetchone()
    expected_nulls = sum(1 for i in range(600) if i % 7 == 0)
    assert res_nulls == (600, 600 - expected_nulls, 300), f"unexpected null/bool counts: {res_nulls}"

    join_rows = con.execute(
        """
        SELECT z.borough, COUNT(*) AS n, SUM(t.total_cents)::BIGINT AS gross
        FROM trips t
        JOIN zones z ON t.zone_id = z.zone_id
        WHERE t.payment_type = 1
        GROUP BY z.borough
        ORDER BY gross DESC, z.borough ASC
        """
    ).fetchall()
    assert len(join_rows) == 5 and sum(r[1] for r in join_rows) == 200

    # Verify compressed-domain SQL pushdown -> Arrow -> DuckDB matches native `vxg_sql_alloc` 100%!
    pushed = read_vxg(
        ndjson_vxg,
        sql="SELECT payment_type, COUNT(*), SUM(total_cents) FROM frame WHERE total_cents >= 2000 GROUP BY payment_type ORDER BY payment_type ASC",
        threads=2,
    )
    duck_pushed = con.execute("SELECT * FROM pushed").fetchall()

    sql_q = b"SELECT payment_type, COUNT(*), SUM(total_cents) FROM frame WHERE total_cents >= 2000 GROUP BY payment_type ORDER BY payment_type ASC"
    s_len = ctypes.c_size_t(0)
    sp = lib.vxg_sql_alloc(ndjson_vxg, len(ndjson_vxg), sql_q, 0, 1, 2, ctypes.byref(s_len))
    assert sp
    native_tsv = ctypes.string_at(sp, s_len.value).decode("utf-8")
    lib.vxg_free(sp, s_len.value)
    native_rows = [tuple(int(x) for x in ln.split("\t")) for ln in native_tsv.strip().splitlines()]
    assert duck_pushed == native_rows, f"{duck_pushed} != {native_rows}"

    print(
        f"OK: DuckDB {duckdb.__version__} zero-copy Arrow C Stream scan verified "
        f"(600 NDJSON rows x 7 typed cols + 5 CSV rows JOIN + compressed SQL pushdown parity)."
    )


if __name__ == "__main__":
    _self_test()
