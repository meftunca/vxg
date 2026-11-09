#!/usr/bin/env python3
"""Head-to-head analytical benchmark: VXG Compressed-Domain vs DuckDB (Raw, .zst, Parquet).

Compares in-process libvxg (`ctypes` over `libvxg.dylib` / `libvxg.so`) against in-process
DuckDB across identical SQL queries, thread counts (`1T` and `8T`), and storage representations:
  1. `.vxg`                        — VXG Format v4/v5 (`64 KiB` independent blocks, 100% byte-exact lossless)
  2. `.aligned.vxg`                — VXG `--record-aligned` (100% byte-exact lossless)
  3. Raw `.ndjson/.csv`            — DuckDB `read_json_auto` / `read_csv`
  4. `.ndjson.zst`                 — DuckDB `read_json_auto` / `read_csv` over `zstd -3`
  5. `.parquet` (Snappy, default)  — DuckDB `read_parquet` (`ROW_GROUP_SIZE=122880`, NOT byte-reversible)
  6. `.parquet` (Zstd, default)    — DuckDB `read_parquet` (`ROW_GROUP_SIZE=122880`, NOT byte-reversible)
  7. `.parquet` (Snappy, RG=16384) — DuckDB `read_parquet` (`ROW_GROUP_SIZE=16384` for fair 8T parallelism)
  8. `.parquet` (Zstd, RG=16384)   — DuckDB `read_parquet` (`ROW_GROUP_SIZE=16384` for fair 8T parallelism)

Reports wall-clock latency (`ms`), throughput in million rows/sec (`M rows/s`), archive size on
disk (`MB` + ratio), one-time ingest/compression latency (`ms`), and exact output verification.
"""

import ctypes
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import duckdb


class VxgAggStats(ctypes.Structure):
    _fields_ = [
        ("row_count", ctypes.c_uint64),
        ("num_count", ctypes.c_uint64),
        ("sum", ctypes.c_double),
        ("sum_lo", ctypes.c_uint64),
        ("sum_hi", ctypes.c_uint64),
        ("min_val", ctypes.c_uint64),
        ("max_val", ctypes.c_uint64),
        ("has_minmax", ctypes.c_int32),
        ("_reserved", ctypes.c_int32),
    ]


def load_vxg_lib(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    lib.vxg_compress_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_compress_alloc.restype = ctypes.c_void_p
    lib.vxg_decompress_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_decompress_alloc.restype = ctypes.c_void_p
    lib.vxg_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vxg_free.restype = None

    lib.vxg_count_where_str.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.vxg_count_where_str.restype = ctypes.c_ssize_t

    lib.vxg_aggregate_col.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(VxgAggStats),
    ]
    lib.vxg_aggregate_col.restype = ctypes.c_ssize_t

    lib.vxg_select_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_select_alloc.restype = ctypes.c_void_p

    lib.vxg_group_by_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vxg_group_by_alloc.restype = ctypes.c_void_p
    return lib


def bench_median_ms(fn, samples: int = 7, warmups: int = 2):
    res = None
    for _ in range(warmups):
        res = fn()
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), res


def vxg_compress_bytes(lib, raw: bytes, threads: int) -> tuple[bytes, float]:
    raw_buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)

    def do_comp():
        out_len = ctypes.c_size_t(0)
        ptr = lib.vxg_compress_alloc(raw_buf, len(raw), threads, 65536, 0, ctypes.byref(out_len))
        assert ptr, "vxg_compress_alloc failed"
        b = ctypes.string_at(ptr, out_len.value)
        lib.vxg_free(ptr, out_len.value)
        return b

    ms, frame = bench_median_ms(do_comp, samples=5, warmups=1)
    # Verify 100% byte-exact roundtrip
    out_len = ctypes.c_size_t(0)
    f_buf = (ctypes.c_uint8 * len(frame)).from_buffer_copy(frame)
    dptr = lib.vxg_decompress_alloc(f_buf, len(frame), threads, 0, ctypes.byref(out_len))
    assert dptr, "vxg_decompress_alloc failed"
    restored = ctypes.string_at(dptr, out_len.value)
    lib.vxg_free(dptr, out_len.value)
    assert restored == raw, "VXG roundtrip mismatch!"
    return frame, ms


def resolve_vxg_artifact(env_key: str, filenames: list[str]) -> str:
    env_val = os.environ.get(env_key)
    if env_val and os.path.exists(env_val):
        return env_val
    dirs: list[str] = []
    if os.environ.get("CARGO_TARGET_DIR"):
        dirs.append(os.path.join(os.environ["CARGO_TARGET_DIR"], "release"))
    dirs.append(os.path.expanduser(os.path.join("~", ".cargo", "target", "release")))
    dirs.append(os.path.join("target", "release"))
    dirs.append("/tmp/vxg-target/release")
    existing: list[str] = []
    for d in dirs:
        for fn in filenames:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                existing.append(p)
    if not existing:
        raise FileNotFoundError(f"Could not find {filenames} in {dirs}; run `cargo build --release` first")
    return max(existing, key=lambda p: os.path.getmtime(p))


def run_dataset_suite(lib, vxg_bin: str, spec: dict, work_dir: str) -> dict:
    name = spec["name"]
    path = spec["path"]
    kind = spec["kind"]  # "ndjson" or "csv"
    raw = open(path, "rb").read()
    raw_mb = len(raw) / 1e6
    row_count = spec["rows"]

    ext = ".ndjson" if kind == "ndjson" else ".csv"
    zst_path = os.path.join(work_dir, f"{name}{ext}.zst")
    vxg_aln_path = os.path.join(work_dir, f"{name}.aligned.vxg")
    pq_zstd_path = os.path.join(work_dir, f"{name}.zstd.parquet")
    pq_snap_path = os.path.join(work_dir, f"{name}.snappy.parquet")
    pq_zstd_rg16k_path = os.path.join(work_dir, f"{name}.zstd.rg16k.parquet")
    pq_snap_rg16k_path = os.path.join(work_dir, f"{name}.snappy.rg16k.parquet")

    # Prepare Zstd -3 raw archive
    t0 = time.perf_counter()
    subprocess.run(["zstd", "-q", "-f", "-3", "-T8", path, "-o", zst_path], check=True)
    zst_enc_ms = (time.perf_counter() - t0) * 1000.0

    # Prepare Parquet (default ROW_GROUP_SIZE=122880 AND parallel-friendly ROW_GROUP_SIZE=16384) via DuckDB 8T
    con_prep = duckdb.connect()
    con_prep.execute("SET threads TO 8;")
    reader_expr = (
        f"read_json_auto('{path}')"
        if kind == "ndjson"
        else f"read_csv('{path}', header=true, auto_detect=true)"
    )
    pq_zstd_ms, _ = bench_median_ms(
        lambda: con_prep.execute(
            f"COPY (SELECT * FROM {reader_expr}) TO '{pq_zstd_path}' (FORMAT PARQUET, COMPRESSION ZSTD);"
        ),
        samples=3,
        warmups=1,
    )
    pq_snap_ms, _ = bench_median_ms(
        lambda: con_prep.execute(
            f"COPY (SELECT * FROM {reader_expr}) TO '{pq_snap_path}' (FORMAT PARQUET, COMPRESSION SNAPPY);"
        ),
        samples=3,
        warmups=1,
    )
    pq_zstd_rg16k_ms, _ = bench_median_ms(
        lambda: con_prep.execute(
            f"COPY (SELECT * FROM {reader_expr}) TO '{pq_zstd_rg16k_path}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384);"
        ),
        samples=3,
        warmups=1,
    )
    pq_snap_rg16k_ms, _ = bench_median_ms(
        lambda: con_prep.execute(
            f"COPY (SELECT * FROM {reader_expr}) TO '{pq_snap_rg16k_path}' "
            f"(FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 16384);"
        ),
        samples=3,
        warmups=1,
    )
    con_prep.close()

    _, vxg_1t_ms = vxg_compress_bytes(lib, raw, 1)
    frame, vxg_8t_ms = vxg_compress_bytes(lib, raw, 8)
    frame_buf = (ctypes.c_uint8 * len(frame)).from_buffer_copy(frame)
    frame_len = len(frame)

    # Prepare & verify VXG `--record-aligned`
    vxg_aln_8t_ms, _ = bench_median_ms(
        lambda: subprocess.run(
            [vxg_bin, "compress", path, vxg_aln_path, "-t", "8", "--record-aligned", "--no-checksum", "-f", "-q"],
            check=True,
        ),
        samples=3,
        warmups=1,
    )
    frame_aln = open(vxg_aln_path, "rb").read()
    frame_aln_buf = (ctypes.c_uint8 * len(frame_aln)).from_buffer_copy(frame_aln)
    frame_aln_len = len(frame_aln)
    out_len_aln = ctypes.c_size_t(0)
    dptr_aln = lib.vxg_decompress_alloc(frame_aln_buf, frame_aln_len, 8, 0, ctypes.byref(out_len_aln))
    assert dptr_aln, "vxg_decompress_alloc (--record-aligned) failed"
    restored_aln = ctypes.string_at(dptr_aln, out_len_aln.value)
    lib.vxg_free(dptr_aln, out_len_aln.value)
    assert restored_aln == raw, "VXG --record-aligned roundtrip mismatch!"

    storage = {
        "raw_bytes": len(raw),
        "raw_mb": raw_mb,
        "rows": row_count,
        "vxg_bytes": frame_len,
        "vxg_ratio": frame_len / len(raw),
        "vxg_enc_1t_ms": vxg_1t_ms,
        "vxg_enc_8t_ms": vxg_8t_ms,
        "vxg_aligned_bytes": frame_aln_len,
        "vxg_aligned_ratio": frame_aln_len / len(raw),
        "vxg_aligned_enc_8t_ms": vxg_aln_8t_ms,
        "zst_bytes": os.path.getsize(zst_path),
        "zst_ratio": os.path.getsize(zst_path) / len(raw),
        "zst_enc_8t_ms": zst_enc_ms,
        "pq_zstd_bytes": os.path.getsize(pq_zstd_path),
        "pq_zstd_ratio": os.path.getsize(pq_zstd_path) / len(raw),
        "pq_zstd_etl_8t_ms": pq_zstd_ms,
        "pq_snap_bytes": os.path.getsize(pq_snap_path),
        "pq_snap_ratio": os.path.getsize(pq_snap_path) / len(raw),
        "pq_snap_etl_8t_ms": pq_snap_ms,
        "pq_zstd_rg16k_bytes": os.path.getsize(pq_zstd_rg16k_path),
        "pq_zstd_rg16k_ratio": os.path.getsize(pq_zstd_rg16k_path) / len(raw),
        "pq_zstd_rg16k_etl_8t_ms": pq_zstd_rg16k_ms,
        "pq_snap_rg16k_bytes": os.path.getsize(pq_snap_rg16k_path),
        "pq_snap_rg16k_ratio": os.path.getsize(pq_snap_rg16k_path) / len(raw),
        "pq_snap_rg16k_etl_8t_ms": pq_snap_rg16k_ms,
    }

    results_by_threads = {}
    for threads in [1, 8]:
        con = duckdb.connect()
        con.execute(f"SET threads TO {threads};")

        # Measure pure Python + DuckDB parse/plan/fetch overhead (`SELECT 1` ad-hoc & prepared)
        sel1_adhoc_ms, _ = bench_median_ms(
            lambda: con.execute("SELECT 1").fetchall(), samples=11, warmups=3
        )
        con.execute("PREPARE sel1_stmt AS SELECT 1;")
        sel1_prep_ms, _ = bench_median_ms(
            lambda: con.execute("EXECUTE sel1_stmt").fetchall(), samples=11, warmups=3
        )

        src_raw = (
            f"read_json_auto('{path}')"
            if kind == "ndjson"
            else f"read_csv('{path}', header=true, auto_detect=true)"
        )
        src_zst = (
            f"read_json_auto('{zst_path}')"
            if kind == "ndjson"
            else f"read_csv('{zst_path}', header=true, auto_detect=true)"
        )
        src_pq_zstd = f"read_parquet('{pq_zstd_path}')"
        src_pq_snap = f"read_parquet('{pq_snap_path}')"
        src_pq_zstd_rg16k = f"read_parquet('{pq_zstd_rg16k_path}')"
        src_pq_snap_rg16k = f"read_parquet('{pq_snap_rg16k_path}')"

        q_rows = []
        for qi, q in enumerate(spec["queries"]):
            qtype = q["type"]
            label = q["label"]

            def make_vxg_runner(fbuf, flen: int):
                if qtype == "count_where":
                    w_b = q["vxg_where"].encode()

                    def _run():
                        n = lib.vxg_count_where_str(fbuf, flen, w_b, threads)
                        assert n >= 0
                        return int(n)

                    return _run
                if qtype == "agg":
                    c_b = q["vxg_col"].encode()
                    w_b = q["vxg_where"].encode() if q.get("vxg_where") else None

                    def _run():
                        st = VxgAggStats()
                        rc = lib.vxg_aggregate_col(fbuf, flen, c_b, w_b, threads, ctypes.byref(st))
                        assert rc == 0
                        total = st.sum_lo | (st.sum_hi << 64)
                        return (int(st.num_count), int(total), int(st.min_val), int(st.max_val))

                    return _run
                if qtype == "group_by":
                    g_b = q["vxg_group"].encode()
                    a_b = q["vxg_agg"].encode()
                    w_b = q["vxg_where"].encode() if q.get("vxg_where") else None

                    def _run():
                        out_len = ctypes.c_size_t(0)
                        ptr = lib.vxg_group_by_alloc(
                            fbuf, flen, g_b, a_b, w_b, threads, ctypes.byref(out_len)
                        )
                        assert ptr
                        txt = ctypes.string_at(ptr, out_len.value).decode()
                        lib.vxg_free(ptr, out_len.value)
                        out_map = {}
                        for ln in txt.splitlines():
                            k, rc_s, _, sum_s, _, _ = ln.split("\t")
                            out_map[str(k)] = (int(rc_s), int(sum_s))
                        return out_map

                    return _run
                if qtype == "top_k":
                    cols_b = q["vxg_cols"].encode()
                    ord_b = q["vxg_order"].encode()
                    w_b = q["vxg_where"].encode() if q.get("vxg_where") else None
                    desc_i = 1 if q.get("desc", True) else 0
                    lim = q["limit"]

                    def _run():
                        out_len = ctypes.c_size_t(0)
                        ptr = lib.vxg_select_alloc(
                            fbuf,
                            flen,
                            cols_b,
                            w_b,
                            ord_b,
                            desc_i,
                            lim,
                            threads,
                            ctypes.byref(out_len),
                        )
                        assert ptr
                        b = ctypes.string_at(ptr, out_len.value)
                        lib.vxg_free(ptr, out_len.value)
                        return b

                    return _run
                if qtype == "select_where":
                    cols_b = q["vxg_cols"].encode()
                    w_b = q["vxg_where"].encode()

                    def _run():
                        out_len = ctypes.c_size_t(0)
                        ptr = lib.vxg_select_alloc(
                            fbuf,
                            flen,
                            cols_b,
                            w_b,
                            None,
                            0,
                            (1 << 62),
                            threads,
                            ctypes.byref(out_len),
                        )
                        assert ptr
                        b = ctypes.string_at(ptr, out_len.value)
                        lib.vxg_free(ptr, out_len.value)
                        return b

                    return _run
                raise ValueError(qtype)

            vxg_ms, vxg_val = bench_median_ms(
                make_vxg_runner(frame_buf, frame_len), samples=7, warmups=2
            )
            vxg_aligned_ms, vxg_aln_val = bench_median_ms(
                make_vxg_runner(frame_aln_buf, frame_aln_len), samples=7, warmups=2
            )
            assert vxg_val == vxg_aln_val, f"{label}: aligned mismatch"

            def bench_duck(tbl_src: str):
                sql = q["sql"].format(tbl=tbl_src)
                return bench_median_ms(lambda: con.execute(sql).fetchall(), samples=5, warmups=1)

            def bench_duck_prepared(stmt_name: str, tbl_src: str):
                sql = q["sql"].format(tbl=tbl_src)
                con.execute(f"PREPARE {stmt_name} AS {sql};")
                return bench_median_ms(
                    lambda: con.execute(f"EXECUTE {stmt_name}").fetchall(), samples=7, warmups=2
                )

            raw_ms, d_raw = bench_duck(src_raw)
            zst_ms, d_zst = bench_duck(src_zst)
            pq_snap_ms_q, d_pq_s = bench_duck(src_pq_snap)
            pq_zstd_ms_q, d_pq_z = bench_duck(src_pq_zstd)
            pq_snap_prep_ms, d_pq_sp = bench_duck_prepared(f"pqs_{qi}", src_pq_snap)
            pq_zstd_prep_ms, d_pq_zp = bench_duck_prepared(f"pqz_{qi}", src_pq_zstd)
            pq_snap_rg16k_prep_ms, d_pq_s16 = bench_duck_prepared(
                f"pqs16_{qi}", src_pq_snap_rg16k
            )
            pq_zstd_rg16k_prep_ms, d_pq_z16 = bench_duck_prepared(
                f"pqz16_{qi}", src_pq_zstd_rg16k
            )
            if qtype == "group_by":
                assert (
                    sorted(d_pq_s) == sorted(d_pq_sp) == sorted(d_pq_s16)
                    and sorted(d_pq_z) == sorted(d_pq_zp) == sorted(d_pq_z16)
                )
            elif qtype == "top_k":
                assert [int(r[-1]) for r in d_pq_s] == [int(r[-1]) for r in d_pq_s16]
                assert [int(r[-1]) for r in d_pq_z] == [int(r[-1]) for r in d_pq_z16]
            else:
                assert d_pq_s == d_pq_sp == d_pq_s16 and d_pq_z == d_pq_zp == d_pq_z16

            # Strict semantic equivalence check between VXG and DuckDB!
            if qtype == "count_where":
                assert vxg_val == d_pq_z[0][0] == d_raw[0][0], f"{label}: {vxg_val} vs {d_pq_z}"
            elif qtype == "agg":
                d_tup = tuple(int(x) for x in d_pq_z[0])
                assert vxg_val == d_tup, f"{label}: {vxg_val} vs {d_tup}"
            elif qtype == "group_by":
                d_map = {str(r[0]): (int(r[1]), int(r[2])) for r in d_pq_z if r[0] is not None}
                assert vxg_val == d_map, f"{label}: group_by mismatch: {vxg_val} vs {d_map}"
            elif qtype == "top_k":
                vxg_lines = vxg_val.decode().splitlines()
                assert len(vxg_lines) == len(d_pq_z) == q["limit"]
                # Verify top-1 and top-K sort key values match DuckDB's sorted values!
                vxg_vals = [json.loads(ln)[q["vxg_order"]] for ln in vxg_lines]
                duck_vals = [int(r[-1]) for r in d_pq_z]
                assert vxg_vals == duck_vals, f"{label}: top_k values {vxg_vals} vs {duck_vals}"
            elif qtype == "select_where":
                vxg_n = vxg_val.count(b"\n")
                assert vxg_n == len(d_pq_z), f"{label}: row count {vxg_n} vs {len(d_pq_z)}"

            mrows = row_count / 1e6
            q_rows.append(
                {
                    "label": label,
                    "vxg_ms": vxg_ms,
                    "vxg_mrows_s": mrows / (vxg_ms / 1000.0),
                    "vxg_aligned_ms": vxg_aligned_ms,
                    "vxg_aligned_mrows_s": mrows / (vxg_aligned_ms / 1000.0),
                    "duck_raw_ms": raw_ms,
                    "duck_zst_ms": zst_ms,
                    "duck_pq_snap_ms": pq_snap_ms_q,
                    "duck_pq_snap_prep_ms": pq_snap_prep_ms,
                    "duck_pq_snap_rg16k_prep_ms": pq_snap_rg16k_prep_ms,
                    "duck_pq_zstd_ms": pq_zstd_ms_q,
                    "duck_pq_zstd_prep_ms": pq_zstd_prep_ms,
                    "duck_pq_zstd_rg16k_prep_ms": pq_zstd_rg16k_prep_ms,
                    "duck_pq_zstd_mrows_s": mrows / (pq_zstd_ms_q / 1000.0),
                    "duck_pq_zstd_prep_mrows_s": mrows / (pq_zstd_prep_ms / 1000.0),
                    "duck_pq_snap_rg16k_prep_mrows_s": mrows / (pq_snap_rg16k_prep_ms / 1000.0),
                    "duck_pq_zstd_rg16k_prep_mrows_s": mrows / (pq_zstd_rg16k_prep_ms / 1000.0),
                }
            )
            print(
                f"  [{name} {threads}T sel1={sel1_adhoc_ms:.2f}/{sel1_prep_ms:.2f}ms] {label:42s}  "
                f"VXG={vxg_ms:5.2f}ms (aln={vxg_aligned_ms:5.2f}ms) | "
                f"ParqSnap={pq_snap_prep_ms:5.2f}ms(rg16k={pq_snap_rg16k_prep_ms:5.2f}) | "
                f"ParqZstd={pq_zstd_prep_ms:5.2f}ms(rg16k={pq_zstd_rg16k_prep_ms:5.2f})",
                flush=True,
            )

        con.close()
        results_by_threads[str(threads)] = {
            "duck_select_1_adhoc_ms": sel1_adhoc_ms,
            "duck_select_1_prepared_ms": sel1_prep_ms,
            "queries": q_rows,
        }

    return {"name": name, "category": spec["category"], "storage": storage, "threads": results_by_threads}


def main():
    lib_path = resolve_vxg_artifact("VXG_LIB", ["libvxg.dylib", "libvxg.so", "vxg.dll"])
    vxg_bin = resolve_vxg_artifact("VXG_BIN", ["vxg", "vxg.exe"])
    print(f"Using libvxg={lib_path}, vxg_bin={vxg_bin}", flush=True)
    lib = load_vxg_lib(lib_path)

    suites = [
        {
            "name": "nyc-taxi-2024-01.ndjson",
            "category": "Real Public Corpus (NYC TLC Jan 2024, 200k trips, 52.3 MB)",
            "path": "/tmp/vxg-public-real/nyc-taxi-2024-01.ndjson",
            "kind": "ndjson",
            "rows": 200000,
            "queries": [
                {
                    "type": "count_where",
                    "label": "COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500",
                    "vxg_where": "payment_type==1 && tip_cents>=1000 && distance_centimiles>=500",
                    "sql": "SELECT COUNT(*) FROM {tbl} WHERE payment_type = 1 AND tip_cents >= 1000 AND distance_centimiles >= 500",
                },
                {
                    "type": "agg",
                    "label": "SUM/MIN/MAX(total_cents) [all 200k rows]",
                    "vxg_col": "total_cents",
                    "sql": "SELECT COUNT(total_cents), SUM(total_cents)::BIGINT, MIN(total_cents)::BIGINT, MAX(total_cents)::BIGINT FROM {tbl}",
                },
                {
                    "type": "agg",
                    "label": "SUM/MIN/MAX(fare_cents) WHERE payment_type=2",
                    "vxg_col": "fare_cents",
                    "vxg_where": "payment_type==2",
                    "sql": "SELECT COUNT(fare_cents), SUM(fare_cents)::BIGINT, MIN(fare_cents)::BIGINT, MAX(fare_cents)::BIGINT FROM {tbl} WHERE payment_type = 2",
                },
                {
                    "type": "group_by",
                    "label": "GROUP BY payment_type: COUNT(*), SUM(total)",
                    "vxg_group": "payment_type",
                    "vxg_agg": "total_cents",
                    "sql": "SELECT payment_type, COUNT(*), SUM(total_cents)::BIGINT FROM {tbl} GROUP BY payment_type",
                },
                {
                    "type": "top_k",
                    "label": "ORDER BY total_cents DESC LIMIT 10",
                    "vxg_cols": "pickup_ts,pu_loc,total_cents",
                    "vxg_order": "total_cents",
                    "desc": True,
                    "limit": 10,
                    "sql": "SELECT pickup_ts, pu_loc, total_cents FROM {tbl} ORDER BY total_cents DESC LIMIT 10",
                },
                {
                    "type": "select_where",
                    "label": "SELECT 4 cols WHERE tip_cents >= 2500",
                    "vxg_cols": "pickup_ts,pu_loc,do_loc,total_cents",
                    "vxg_where": "tip_cents>=2500",
                    "sql": "SELECT pickup_ts, pu_loc, do_loc, total_cents FROM {tbl} WHERE tip_cents >= 2500",
                },
            ],
        },
        {
            "name": "nyc-taxi-2024-01.csv",
            "category": "Real Public Corpus (NYC TLC Jan 2024 CSV, 300k trips, 23.0 MB)",
            "path": "/tmp/vxg-public-real/nyc-taxi-2024-01.csv",
            "kind": "csv",
            "rows": 300000,
            "queries": [
                {
                    "type": "count_where",
                    "label": "COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500",
                    "vxg_where": "payment_type==1 && tip_cents>=1000 && distance_centimiles>=500",
                    "sql": "SELECT COUNT(*) FROM {tbl} WHERE payment_type = 1 AND tip_cents >= 1000 AND distance_centimiles >= 500",
                },
                {
                    "type": "agg",
                    "label": "SUM/MIN/MAX(total_cents) [all 300k rows]",
                    "vxg_col": "total_cents",
                    "sql": "SELECT COUNT(total_cents), SUM(total_cents)::BIGINT, MIN(total_cents)::BIGINT, MAX(total_cents)::BIGINT FROM {tbl}",
                },
                {
                    "type": "agg",
                    "label": "SUM/MIN/MAX(fare_cents) WHERE payment_type=2",
                    "vxg_col": "fare_cents",
                    "vxg_where": "payment_type==2",
                    "sql": "SELECT COUNT(fare_cents), SUM(fare_cents)::BIGINT, MIN(fare_cents)::BIGINT, MAX(fare_cents)::BIGINT FROM {tbl} WHERE payment_type = 2",
                },
                {
                    "type": "group_by",
                    "label": "GROUP BY payment_type: COUNT(*), SUM(total)",
                    "vxg_group": "payment_type",
                    "vxg_agg": "total_cents",
                    "sql": "SELECT payment_type, COUNT(*), SUM(total_cents)::BIGINT FROM {tbl} GROUP BY payment_type",
                },
            ],
        },
        {
            "name": "ndjson_varied.ndjson",
            "category": "Synthetic Multi-Template Corpus (gen_varied_ndjson.py, 300k rows, 56.5 MB)",
            "path": "/tmp/vxgreal/ndjson_varied.ndjson",
            "kind": "ndjson",
            "rows": 300000,
            "queries": [
                {
                    "type": "count_where",
                    "label": "COUNT(*) WHERE fraud & status=500 & lat>=150",
                    "vxg_where": "service==\"fraud-engine\" && status==500 && latency_ms>=150",
                    "sql": "SELECT COUNT(*) FROM {tbl} WHERE service = 'fraud-engine' AND status = 500 AND latency_ms >= 150",
                },
                {
                    "type": "agg",
                    "label": "SUM/MIN/MAX(bytes) [all 300k rows]",
                    "vxg_col": "bytes",
                    "sql": "SELECT COUNT(bytes), SUM(bytes)::BIGINT, MIN(bytes)::BIGINT, MAX(bytes)::BIGINT FROM {tbl}",
                },
                {
                    "type": "group_by",
                    "label": "GROUP BY service: COUNT(*), SUM(bytes)",
                    "vxg_group": "service",
                    "vxg_agg": "bytes",
                    "sql": "SELECT service, COUNT(*), SUM(bytes)::BIGINT FROM {tbl} GROUP BY service",
                },
                {
                    "type": "top_k",
                    "label": "ORDER BY latency_ms DESC LIMIT 10",
                    "vxg_cols": "ts,service,latency_ms",
                    "vxg_order": "latency_ms",
                    "desc": True,
                    "limit": 10,
                    "sql": "SELECT ts, service, latency_ms FROM {tbl} ORDER BY latency_ms DESC LIMIT 10",
                },
                {
                    "type": "select_where",
                    "label": "SELECT 3 cols WHERE fraud & 500 & lat>=150",
                    "vxg_cols": "ts,req_id,latency_ms",
                    "vxg_where": "service==\"fraud-engine\" && status==500 && latency_ms>=150",
                    "sql": "SELECT ts, req_id, latency_ms FROM {tbl} WHERE service = 'fraud-engine' AND status = 500 AND latency_ms >= 150",
                },
            ],
        },
    ]

    out_json = sys.argv[1] if len(sys.argv) > 1 else "reports/duckdb-vs-vxg.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    all_out = {"duckdb_version": duckdb.__version__, "datasets": []}
    with tempfile.TemporaryDirectory(prefix="vxg_duck_") as wdir:
        for s in suites:
            if not os.path.exists(s["path"]):
                print(f"Skipping missing {s['path']}", file=sys.stderr)
                continue
            print(f"\n=== Benchmarking {s['name']} ({s['category']}) ===", flush=True)
            all_out["datasets"].append(run_dataset_suite(lib, vxg_bin, s, wdir))

    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved JSON report to {out_json}")


if __name__ == "__main__":
    main()
