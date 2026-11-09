#!/usr/bin/env python3
"""Zero-dependency DuckDB / Polars / PyArrow scan bridge for `.vxg` frames via Arrow PyCapsule.

Exposes any `.vxg` file or in-memory frame via the standard Apache Arrow C Stream PyCapsule
protocol (`__arrow_c_stream__`), allowing `DuckDB >= 1.1`, `Polars`, and `PyArrow` to query
compressed `.vxg` archives directly with **zero `pyarrow` / `numpy` / `pandas` dependencies**,
with compressed-domain `columns` projection and `where` predicate pushdown (`vxg_arrow_query_stream`).

Usage in Python + DuckDB:
    import duckdb
    from duckdb_read_vxg import read_vxg

    con = duckdb.connect()
    trips = read_vxg(
        "nyc-taxi.ndjson.vxg",
        columns=["payment_type", "total_cents"],
        where="payment_type = 1 AND tip_cents >= 1000",
    )
    print(con.execute("SELECT payment_type, COUNT(*), SUM(total_cents) FROM trips GROUP BY 1").fetchall())
"""

from __future__ import annotations

import ctypes
import mmap
import os
import platform
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

_ACTIVE_STREAMS: dict[int, tuple[_ArrowArrayStream, object]] = {}


@_PYCAPSULE_DESTRUCTOR
def _release_stream_capsule(capsule_ptr: int) -> None:
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule_ptr, _CAPSULE_NAME_BUF)
    entry = _ACTIVE_STREAMS.pop(ptr, None)
    if entry is not None:
        st, _owner = entry
        if st.release:
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
    lib.vxg_arrow_query_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(_ArrowArrayStream),
    ]
    lib.vxg_arrow_query_stream.restype = ctypes.c_ssize_t
    return lib


class VxgArrowDataset:
    """Zero-copy mmap / buffer wrapper implementing the Arrow PyCapsule `__arrow_c_stream__` protocol."""

    def __init__(
        self,
        source: str | os.PathLike | bytes,
        *,
        sql: str | None = None,
        columns: Iterable[str] | str | None = None,
        where: str | None = None,
        threads: int = 0,
        lib_path: str | None = None,
    ) -> None:
        if sql is not None and (columns is not None or where is not None):
            raise ValueError("Specify either `sql=` OR (`columns=`, `where=`), not both")

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

        self._sql_bytes = sql.encode("utf-8") if sql is not None else None
        if columns is None:
            self._cols_bytes = None
        elif isinstance(columns, str):
            self._cols_bytes = columns.encode("utf-8")
        else:
            self._cols_bytes = ",".join(columns).encode("utf-8")
        self._where_bytes = where.encode("utf-8") if where is not None else None

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
        if self._sql_bytes is not None:
            rc = self._lib.vxg_arrow_export_stream(
                self._ptr,
                self._len,
                self._sql_bytes,
                self._threads,
                ctypes.byref(st),
            )
            if rc != 0:
                raise RuntimeError(f"vxg_arrow_export_stream failed with status {rc}")
        else:
            rc = self._lib.vxg_arrow_query_stream(
                self._ptr,
                self._len,
                self._cols_bytes,
                self._where_bytes,
                self._threads,
                ctypes.byref(st),
            )
            if rc != 0:
                raise RuntimeError(f"vxg_arrow_query_stream failed with status {rc}")
        addr = ctypes.addressof(st)
        _ACTIVE_STREAMS[addr] = (st, self)
        return ctypes.pythonapi.PyCapsule_New(
            addr, _CAPSULE_NAME_BUF, _release_stream_capsule
        )


def read_vxg(
    source: str | os.PathLike | bytes,
    *,
    sql: str | None = None,
    columns: Iterable[str] | str | None = None,
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

    # 1. Build an NDJSON frame (>70,000 rows to exercise multi-batch >65,536-row streaming!)
    n_rows = 72_000
    ndjson_lines = []
    for i in range(n_rows):
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
    p1 = lib.vxg_compress_alloc(ndjson_raw, len(ndjson_raw), 4, 65536, 1, ctypes.byref(out_len))
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
    trips = read_vxg(ndjson_vxg, threads=4)
    zones = read_vxg(csv_vxg, threads=1)

    # 3. Verify multi-batch NULL + Boolean + Int64 + Float64 + cross-archive SQL JOIN inside DuckDB
    res_nulls = con.execute(
        "SELECT COUNT(*), COUNT(tip_cents), SUM(CASE WHEN verified THEN 1 ELSE 0 END) FROM trips"
    ).fetchone()
    expected_nulls = sum(1 for i in range(n_rows) if i % 7 == 0)
    assert res_nulls == (n_rows, n_rows - expected_nulls, n_rows // 2), f"unexpected: {res_nulls}"

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
    assert len(join_rows) == 5 and sum(r[1] for r in join_rows) == n_rows // 3

    # 4. Verify compressed-domain `columns` + `where` pushdown (both SQL & VXG filter syntax!)
    filtered_sql = read_vxg(
        ndjson_vxg,
        columns=["payment_type", "total_cents", "surge"],
        where="payment_type IN (1, 2) AND total_cents >= 2500",
        threads=4,
    )
    filtered_vxg = read_vxg(
        ndjson_vxg,
        columns="payment_type,total_cents,surge",
        where="(payment_type==1 || payment_type==2) && total_cents>=2500",
        threads=4,
    )
    r_sql = con.execute(
        "SELECT payment_type, COUNT(*), SUM(total_cents)::BIGINT, ROUND(AVG(surge), 6) FROM filtered_sql GROUP BY 1 ORDER BY 1"
    ).fetchall()
    r_vxg = con.execute(
        "SELECT payment_type, COUNT(*), SUM(total_cents)::BIGINT, ROUND(AVG(surge), 6) FROM filtered_vxg GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert r_sql == r_vxg and len(r_sql) == 2, f"{r_sql} != {r_vxg}"

    # 5. Verify 0-match scan still exposes typed `Int64`/`Float64` schema so `SUM(...)` succeeds!
    empty_scan = read_vxg(
        ndjson_vxg,
        columns=["payment_type", "total_cents", "surge"],
        where="id < 0",
        threads=2,
    )
    empty_agg = con.execute(
        "SELECT COUNT(*), SUM(total_cents), AVG(surge) FROM empty_scan"
    ).fetchone()
    assert empty_agg == (0, None, None), f"unexpected empty_agg: {empty_agg}"

    # 6. Verify compressed-domain SQL GROUP BY pushdown -> Arrow -> DuckDB matches native `vxg_sql_alloc`
    pushed = read_vxg(
        ndjson_vxg,
        sql="SELECT payment_type, COUNT(*), SUM(total_cents) FROM frame WHERE total_cents >= 2000 GROUP BY payment_type ORDER BY payment_type ASC",
        threads=4,
    )
    duck_pushed = con.execute("SELECT * FROM pushed").fetchall()

    sql_q = b"SELECT payment_type, COUNT(*), SUM(total_cents) FROM frame WHERE total_cents >= 2000 GROUP BY payment_type ORDER BY payment_type ASC"
    s_len = ctypes.c_size_t(0)
    sp = lib.vxg_sql_alloc(ndjson_vxg, len(ndjson_vxg), sql_q, 0, 1, 4, ctypes.byref(s_len))
    assert sp
    native_tsv = ctypes.string_at(sp, s_len.value).decode("utf-8")
    lib.vxg_free(sp, s_len.value)
    native_rows = [tuple(int(x) for x in ln.split("\t")) for ln in native_tsv.strip().splitlines()]
    assert duck_pushed == native_rows, f"{duck_pushed} != {native_rows}"

    print(
        f"OK: DuckDB {duckdb.__version__} zero-copy Arrow C Stream scan verified "
        f"({n_rows} NDJSON rows multi-batch + columns/where pushdown + 0-match schema + CSV JOIN)."
    )


if __name__ == "__main__":
    _self_test()
