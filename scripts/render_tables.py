#!/usr/bin/env python3
"""Single-source Markdown table generator and CI drift checker for VXG.

Reads benchmark CSV and JSON artifacts in `reports/` and renders deterministic
Markdown tables between `<!-- BEGIN GENERATED TABLE: <id> -->` and
`<!-- END GENERATED TABLE: <id> -->` markers in `README.md` and
`docs/CHIMERA_BENCHMARKS.md`.

Usage:
  python3 scripts/render_tables.py --write
  python3 scripts/render_tables.py --check
"""

import argparse
import csv
import difflib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REAL_PUBLIC_ORDER = [
    ("nyc-taxi-2024-01.ndjson", "NYC TLC, 200k trips"),
    ("nyc-taxi-2024-01.csv", "NYC TLC, 300k trips"),
    ("gharchive-2025-01-15.ndjson", "12k nested GitHub API"),
    ("apache-access.clf", "Elastic real HTTP CLF"),
    ("macos-install.log", "OS free-text multi-line log"),
    ("pg100-shakespeare.txt", "Gutenberg English prose"),
    ("kodak-truecolor-16.bmp", "16 real camera photos"),
]

COVID_SRC_ORDER = [
    ("time-series-19-covid-combined.csv", "datasets/covid-19 cumulative time series"),
    ("us_confirmed_33m.csv", "datasets/covid-19 US county cumulative prefix"),
    ("vxg-src-nonrepeating.rs", "All unique VXG src/*.rs files, 0% repeat"),
]


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_bench_csv(path: Path):
    """Returns dict[(basename, codec)] -> {'ratio': float, 'raw_bytes': int, 'enc': float, 'dec': float}."""
    data = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = os.path.basename(row["dataset"].strip())
            codec = row["codec"].strip()
            prof = row["profile"].strip()
            op = row["operation"].strip()
            key = (ds, codec)
            entry = data.setdefault(
                key,
                {
                    "ratio": float(row["ratio_encoded_over_raw"]),
                    "raw_bytes": int(row["raw_bytes"]),
                    "enc": 0.0,
                    "dec": 0.0,
                },
            )
            gbps = float(row["median_GBps"])
            if op == "compress" and prof in ("cpu_allocating_encode", "cpu_whole_file_encode"):
                entry["enc"] = gbps
            elif op == "decompress" and prof in ("cpu_preallocated_decode", "cpu_whole_file_decode"):
                entry["dec"] = gbps
    return data


def fmt_mb(raw_bytes: int) -> str:
    return f"{raw_bytes / 1_000_000.0:.1f} MB"


def bold_min(vals_dict, fmt="{:.4f}"):
    best_k = min(vals_dict, key=lambda k: vals_dict[k])
    return {
        k: (f"**{fmt.format(v)}**" if k == best_k else fmt.format(v))
        for k, v in vals_dict.items()
    }


def bold_max_pair(enc_dict, dec_dict, ratios_dict, fmt="{:.2f}"):
    eligible = [k for k, r in ratios_dict.items() if r < 0.90]
    best_enc = max(eligible, key=lambda k: enc_dict[k]) if eligible else None
    best_dec = max(eligible, key=lambda k: dec_dict[k]) if eligible else None
    out = {}
    for k in enc_dict:
        es = f"**{fmt.format(enc_dict[k])}**" if k == best_enc else fmt.format(enc_dict[k])
        ds = f"**{fmt.format(dec_dict[k])}**" if k == best_dec else fmt.format(dec_dict[k])
        out[k] = f"{es} / {ds}"
    return out


def bold_max_single(val_dict, ratios_dict, fmt="{:.2f}"):
    eligible = [k for k, r in ratios_dict.items() if r < 0.90]
    best_k = max(eligible, key=lambda k: val_dict[k]) if eligible else None
    return {
        k: (f"**{fmt.format(v)}**" if k == best_k else fmt.format(v))
        for k, v in val_dict.items()
    }


def render_real_public_summary_8t() -> str:
    d8 = load_bench_csv(ROOT / "reports/real-public-8t.csv")
    lines = [
        "| Real Public / OS Corpus (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | LZ4 (`a1`) Enc / Dec | Zstd-1 Enc / Dec | Zstd+D3 Enc / Dec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    text_ds = [name for name, _ in REAL_PUBLIC_ORDER if not name.endswith(".bmp")]
    for name, note in REAL_PUBLIC_ORDER:
        raw_b = d8[(name, "vxg")]["raw_bytes"]
        block_ratios = {
            "vxg": d8[(name, "vxg")]["ratio"],
            "lz4": d8[(name, "lz4-c-a1")]["ratio"],
            "z9": d8[(name, "zstd-c-l9")]["ratio"],
            "zd9": d8[(name, "zstd-dict-l9")]["ratio"],
        }
        r_fmt = bold_min(block_ratios)
        whole_r = f"{d8[(name, 'zstd-whole-l19-long')]['ratio']:.4f}"
        sp_codecs = {
            "vxg": "vxg",
            "lz4": "lz4-c-a1",
            "z1": "zstd-c-l1",
            "zd3": "zstd-dict-l3",
        }
        enc_map = {k: d8[(name, c)]["enc"] for k, c in sp_codecs.items()}
        dec_map = {k: d8[(name, c)]["dec"] for k, c in sp_codecs.items()}
        rat_map = {k: d8[(name, c)]["ratio"] for k, c in sp_codecs.items()}
        sp_fmt = bold_max_pair(enc_map, dec_map, rat_map, "{:.2f}")
        lines.append(
            f"| **`{name}`** *({note})* | {fmt_mb(raw_b)} | {r_fmt['vxg']} | {r_fmt['lz4']} | {r_fmt['z9']} | {r_fmt['zd9']} | {whole_r} | {sp_fmt['vxg']} | {sp_fmt['lz4']} | {sp_fmt['z1']} | {sp_fmt['zd3']} |"
        )

    # 6-dataset Text & Structured Geomean
    geo_ratios = {
        "vxg": geomean([d8[(n, "vxg")]["ratio"] for n in text_ds]),
        "lz4": geomean([d8[(n, "lz4-c-a1")]["ratio"] for n in text_ds]),
        "z9": geomean([d8[(n, "zstd-c-l9")]["ratio"] for n in text_ds]),
        "zd9": geomean([d8[(n, "zstd-dict-l9")]["ratio"] for n in text_ds]),
    }
    gr_fmt = bold_min(geo_ratios)
    whole_geo = f"{geomean([d8[(n, 'zstd-whole-l19-long')]['ratio'] for n in text_ds]):.4f}"
    sp_codecs = {"vxg": "vxg", "lz4": "lz4-c-a1", "z1": "zstd-c-l1", "zd3": "zstd-dict-l3"}
    g_enc = {k: geomean([d8[(n, c)]["enc"] for n in text_ds]) for k, c in sp_codecs.items()}
    g_dec = {k: geomean([d8[(n, c)]["dec"] for n in text_ds]) for k, c in sp_codecs.items()}
    g_rat = {k: geomean([d8[(n, c)]["ratio"] for n in text_ds]) for k, c in sp_codecs.items()}
    gsp_fmt = bold_max_pair(g_enc, g_dec, g_rat, "{:.2f}")
    lines.append(
        f"| **Text & Structured Geomean (6)** | — | {gr_fmt['vxg']} | {gr_fmt['lz4']} | {gr_fmt['z9']} | {gr_fmt['zd9']} | {whole_geo} | {gsp_fmt['vxg']} | {gsp_fmt['lz4']} | {gsp_fmt['z1']} | {gsp_fmt['zd3']} |"
    )
    return "\n".join(lines)


def render_real_public_full(threads: int) -> str:
    d = load_bench_csv(ROOT / f"reports/real-public-{threads}t.csv")
    lines = [
        f"| Dataset (`{threads}T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    text_ds = [name for name, _ in REAL_PUBLIC_ORDER if not name.endswith(".bmp")]
    for name, _ in REAL_PUBLIC_ORDER:
        raw_b = d[(name, "vxg")]["raw_bytes"]
        r_candidates = {
            "vxg": d[(name, "vxg")]["ratio"],
            "lz4": d[(name, "lz4-c-a1")]["ratio"],
            "z1": d[(name, "zstd-c-l1")]["ratio"],
            "z9": d[(name, "zstd-c-l9")]["ratio"],
            "zd3": d[(name, "zstd-dict-l3")]["ratio"],
            "zd9": d[(name, "zstd-dict-l9")]["ratio"],
        }
        rf = bold_min(r_candidates)
        whole_r = f"{d[(name, 'zstd-whole-l19-long')]['ratio']:.4f}"
        sp_c = {"vxg": "vxg", "lz4": "lz4-c-a1", "z1": "zstd-c-l1", "zd3": "zstd-dict-l3"}
        rat_c = {k: d[(name, c)]["ratio"] for k, c in sp_c.items()}
        ef = bold_max_single({k: d[(name, c)]["enc"] for k, c in sp_c.items()}, rat_c, "{:.2f}")
        df = bold_max_single({k: d[(name, c)]["dec"] for k, c in sp_c.items()}, rat_c, "{:.2f}")
        lines.append(
            f"| **`{name}`** | {fmt_mb(raw_b)} | {rf['vxg']} | {rf['lz4']} | {rf['z1']} | {rf['z9']} | {rf['zd3']} | {rf['zd9']} | {whole_r} | {ef['vxg']} | {ef['lz4']} | {ef['z1']} | {ef['zd3']} | {df['vxg']} | {df['lz4']} | {df['z1']} | {df['zd3']} |"
        )

    gr = {
        "vxg": geomean([d[(n, "vxg")]["ratio"] for n in text_ds]),
        "lz4": geomean([d[(n, "lz4-c-a1")]["ratio"] for n in text_ds]),
        "z1": geomean([d[(n, "zstd-c-l1")]["ratio"] for n in text_ds]),
        "z9": geomean([d[(n, "zstd-c-l9")]["ratio"] for n in text_ds]),
        "zd3": geomean([d[(n, "zstd-dict-l3")]["ratio"] for n in text_ds]),
        "zd9": geomean([d[(n, "zstd-dict-l9")]["ratio"] for n in text_ds]),
    }
    grf = bold_min(gr)
    whole_geo = f"{geomean([d[(n, 'zstd-whole-l19-long')]['ratio'] for n in text_ds]):.4f}"
    sp_c = {"vxg": "vxg", "lz4": "lz4-c-a1", "z1": "zstd-c-l1", "zd3": "zstd-dict-l3"}
    grat = {k: geomean([d[(n, c)]["ratio"] for n in text_ds]) for k, c in sp_c.items()}
    gef = bold_max_single({k: geomean([d[(n, c)]["enc"] for n in text_ds]) for k, c in sp_c.items()}, grat, "{:.2f}")
    gdf = bold_max_single({k: geomean([d[(n, c)]["dec"] for n in text_ds]) for k, c in sp_c.items()}, grat, "{:.2f}")
    lines.append(
        f"| **Text & Structured Geomean (6)** | — | {grf['vxg']} | {grf['lz4']} | {grf['z1']} | {grf['z9']} | {grf['zd3']} | {grf['zd9']} | {whole_geo} | {gef['vxg']} | {gef['lz4']} | {gef['z1']} | {gef['zd3']} | {gdf['vxg']} | {gdf['lz4']} | {gdf['z1']} | {gdf['zd3']} |"
    )
    return "\n".join(lines)


def render_covid_and_src() -> str:
    d8 = load_bench_csv(ROOT / "reports/covid-19-and-src-8t.csv")
    d1 = load_bench_csv(ROOT / "reports/covid-19-and-src-1t.csv")
    lines = [
        "| Cumulative Time-Series & Non-Repeating Source (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec (`GB/s`) | `8T` VXG Enc / Dec (`GB/s`) | `8T` LZ4 (`a1`) Enc / Dec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, note in COVID_SRC_ORDER:
        raw_b = d8[(name, "vxg")]["raw_bytes"]
        r_map = {
            "vxg": d8[(name, "vxg")]["ratio"],
            "lz4": d8[(name, "lz4-c-a1")]["ratio"],
            "z1": d8[(name, "zstd-c-l1")]["ratio"],
            "z9": d8[(name, "zstd-c-l9")]["ratio"],
            "zd9": d8[(name, "zstd-dict-l9")]["ratio"],
        }
        rf = bold_min(r_map)
        whole_r = f"{d8[(name, 'zstd-whole-l19-long')]['ratio']:.4f}"
        v1 = f"{d1[(name, 'vxg')]['enc']:.2f} / {d1[(name, 'vxg')]['dec']:.2f}"
        sp8 = bold_max_pair(
            {"vxg": d8[(name, "vxg")]["enc"], "lz4": d8[(name, "lz4-c-a1")]["enc"]},
            {"vxg": d8[(name, "vxg")]["dec"], "lz4": d8[(name, "lz4-c-a1")]["dec"]},
            {"vxg": d8[(name, "vxg")]["ratio"], "lz4": d8[(name, "lz4-c-a1")]["ratio"]},
            "{:.2f}",
        )
        lines.append(
            f"| **`{name}`** *({note})* | {fmt_mb(raw_b)} | {rf['vxg']} | {rf['lz4']} | {rf['z1']} | {rf['z9']} | {rf['zd9']} | {whole_r} | {v1} | {sp8['vxg']} | {sp8['lz4']} |"
        )
    return "\n".join(lines)


def fmt_1t_8t_pair(v1_dict, v8_dict):
    """Formats `1T / 8T` strings for each key, bolding the minimum on 1T and 8T independently."""
    b1 = min(v1_dict, key=lambda k: v1_dict[k])
    b8 = min(v8_dict, key=lambda k: v8_dict[k])
    out = {}
    for k in v1_dict:
        s1 = f"**{v1_dict[k]:.2f}**" if k == b1 else f"{v1_dict[k]:.2f}"
        s8 = f"**{v8_dict[k]:.2f}**" if k == b8 else f"{v8_dict[k]:.2f}"
        out[k] = f"{s1} / {s8}"
    return out


def render_duckdb_in_cache() -> str:
    doc = json.loads((ROOT / "reports/duckdb-vs-vxg.json").read_text(encoding="utf-8"))
    lines = [
        "| Corpus & Query (`1T / 8T` latency in `ms`, lower is faster) | `VXG -l 3` (`64K` std) | `VXG -l 3 --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* | DuckDB Raw |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ds in doc["datasets"]:
        st = ds["storage"]
        name = ds["name"]
        rows_k = f"{st['rows'] // 1000}k"
        raw_mb = f"{st['raw_mb']:.1f} MB"
        v_mb = st["vxg_bytes"] / 1_000_000.0
        va_mb = st["vxg_aligned_bytes"] / 1_000_000.0
        ps_mb = st["pq_snap_bytes"] / 1_000_000.0
        ps16_mb = st["pq_snap_rg16k_bytes"] / 1_000_000.0
        pz_mb = st["pq_zstd_bytes"] / 1_000_000.0
        sz = bold_min(
            {"v": v_mb, "va": va_mb, "ps": ps_mb, "ps16": ps16_mb, "pz": pz_mb},
            "{:.2f} MB",
        )
        lines.append(
            f"| **`{name}`** *({raw_mb}, {rows_k} rows)* | {sz['v']} (`{st['vxg_enc_8t_ms']:.1f} ms` `-l 3`, YES) | {sz['va']} (`{st['vxg_aligned_enc_8t_ms']:.1f} ms` `-l 3`, YES) | {sz['ps']} (`{st['pq_snap_etl_8t_ms']:.1f} ms` ETL, NO) | {sz['ps16']} (`{st['pq_snap_rg16k_etl_8t_ms']:.1f} ms`, NO) | {sz['pz']} (`{st['pq_zstd_etl_8t_ms']:.1f} ms` ETL, NO) | `{st['raw_mb']:.2f} MB` (YES) |"
        )
        q1_list = ds["threads"]["1"]["queries"]
        q8_list = ds["threads"]["8"]["queries"]
        for q1, q8 in zip(q1_list, q8_list):
            v1 = {
                "v": q1["vxg_ms"],
                "va": q1["vxg_aligned_ms"],
                "ps": q1["duck_pq_snap_prep_ms"],
                "ps16": q1["duck_pq_snap_rg16k_prep_ms"],
                "pz": q1["duck_pq_zstd_prep_ms"],
                "raw": q1["duck_raw_ms"],
            }
            v8 = {
                "v": q8["vxg_ms"],
                "va": q8["vxg_aligned_ms"],
                "ps": q8["duck_pq_snap_prep_ms"],
                "ps16": q8["duck_pq_snap_rg16k_prep_ms"],
                "pz": q8["duck_pq_zstd_prep_ms"],
                "raw": q8["duck_raw_ms"],
            }
            f = fmt_1t_8t_pair(v1, v8)
            lines.append(
                f"| ↳ `{q8['label']}` | {f['v']} | {f['va']} | {f['ps']} | {f['ps16']} | {f['pz']} | {f['raw']} |"
            )
    return "\n".join(lines)


def render_duckdb_scale_single(ds_name: str) -> str:
    doc = json.loads((ROOT / "reports/scale-duckdb-vs-vxg.json").read_text(encoding="utf-8"))
    ds = next(d for d in doc["datasets"] if d["name"] == ds_name)
    st = ds["storage"]
    rows_m = f"{st['rows'] // 1_000_000}M"
    lines = [
        f"| `{ds_name}` (`{rows_m}` rows, `{st['raw_gb']:.2f} GB` raw; **Warm OS Cache** `1T / 8T` in `ms`) | `VXG -l 3` (`64K` std) | `VXG -l 3 --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    sz = bold_min(
        {
            "v": st["vxg_bytes"] / 1_000_000.0,
            "va": st["vxg_aligned_bytes"] / 1_000_000.0,
            "ps": st["pq_snap_bytes"] / 1_000_000.0,
            "ps16": st["pq_snap_rg16k_bytes"] / 1_000_000.0,
            "pz": st["pq_zstd_bytes"] / 1_000_000.0,
        },
        "{:.1f} MB",
    )
    lines.append(
        f"| **Archive Size & `8T` Ingest (`-l 3`) / ETL** | {sz['v']} (`{st['vxg_enc_8t_ms']/1000:.2f} s` `-l 3`, YES) | {sz['va']} (`{st['vxg_aligned_enc_8t_ms']/1000:.2f} s` `-l 3`, YES) | {sz['ps']} (`{st['pq_snap_etl_8t_ms']/1000:.2f} s`, NO) | {sz['ps16']} (`{st['pq_snap_rg16k_etl_8t_ms']/1000:.2f} s`, NO) | {sz['pz']} (`{st['pq_zstd_etl_8t_ms']/1000:.2f} s`, NO) |"
    )
    for q1, q8 in zip(ds["threads"]["1"]["queries"], ds["threads"]["8"]["queries"]):
        v1 = {
            "v": q1["vxg_ms"],
            "va": q1["vxg_aligned_ms"],
            "ps": q1["duck_pq_snap_prep_ms"],
            "ps16": q1["duck_pq_snap_rg16k_prep_ms"],
            "pz": q1["duck_pq_zstd_prep_ms"],
        }
        v8 = {
            "v": q8["vxg_ms"],
            "va": q8["vxg_aligned_ms"],
            "ps": q8["duck_pq_snap_prep_ms"],
            "ps16": q8["duck_pq_snap_rg16k_prep_ms"],
            "pz": q8["duck_pq_zstd_prep_ms"],
        }
        f = fmt_1t_8t_pair(v1, v8)
        lines.append(
            f"| `{q8['label']}` | {f['v']} | {f['va']} | {f['ps']} | {f['ps16']} | {f['pz']} |"
        )
    return "\n".join(lines)


def render_duckdb_scale_cold(ds_name: str) -> str:
    doc = json.loads((ROOT / "reports/scale-duckdb-vs-vxg.json").read_text(encoding="utf-8"))
    ds = next(d for d in doc["datasets"] if d["name"] == ds_name)
    st = ds["storage"]
    rows_m = f"{st['rows'] // 1_000_000}M"
    lines = [
        f"| `{ds_name}` (`{rows_m}` rows; **Cold NVMe Cache** `8T` in `ms`, `F_NOCACHE` evicted) | `VXG -l 3` (`64K` std, cold) | `VXG -l 3 --record-aligned` (cold) | DuckDB `Parquet-Snappy` *(cold)* | DuckDB `Parquet-Snappy` *(RG=16k, cold)* | DuckDB `Parquet-Zstd` *(cold)* |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for q8 in ds["threads"]["8"]["queries"]:
        vc = {
            "v": q8["vxg_cold_ms"],
            "va": q8["vxg_aligned_cold_ms"],
            "ps": q8["duck_pq_snap_cold_ms"],
            "ps16": q8["duck_pq_snap_rg16k_cold_ms"],
            "pz": q8["duck_pq_zstd_cold_ms"],
        }
        fc = bold_min(vc, "{:.2f}")
        lines.append(
            f"| `{q8['label']}` | {fc['v']} | {fc['va']} | {fc['ps']} | {fc['ps16']} | {fc['pz']} |"
        )
    return "\n".join(lines)


GENERATORS = {
    "real-public-summary-8t": render_real_public_summary_8t,
    "real-public-full-1t": lambda: render_real_public_full(1),
    "real-public-full-8t": lambda: render_real_public_full(8),
    "covid-and-src": render_covid_and_src,
    "duckdb-in-cache": render_duckdb_in_cache,
    "duckdb-scale-40m-csv": lambda: render_duckdb_scale_single("nyc-taxi-2024-40m.csv"),
    "duckdb-scale-40m-csv-cold": lambda: render_duckdb_scale_cold("nyc-taxi-2024-40m.csv"),
    "duckdb-scale-10m-ndjson": lambda: render_duckdb_scale_single("nyc-taxi-2024-10m.ndjson"),
    "duckdb-scale-10m-ndjson-cold": lambda: render_duckdb_scale_cold("nyc-taxi-2024-10m.ndjson"),
}

TARGET_FILES = [
    ROOT / "README.md",
    ROOT / "docs/CHIMERA_BENCHMARKS.md",
]

MARKER_RE = re.compile(
    r"(<!-- BEGIN GENERATED TABLE: ([a-z0-9_-]+) -->\n)(?:.*?\n)?(<!-- END GENERATED TABLE: \2 -->)",
    re.DOTALL,
)


def process_files(write: bool) -> int:
    rendered_cache = {k: fn() for k, fn in GENERATORS.items()}
    drift_count = 0
    seen_markers = 0

    for path in TARGET_FILES:
        original = path.read_text(encoding="utf-8")

        def repl(match):
            nonlocal seen_markers
            prefix, name, suffix = match.groups()
            if name not in rendered_cache:
                raise KeyError(f"Unknown generated table marker '{name}' in {path}")
            seen_markers += 1
            return f"{prefix}{rendered_cache[name]}\n{suffix}"

        updated = MARKER_RE.sub(repl, original)
        if updated != original:
            if write:
                path.write_text(updated, encoding="utf-8")
                print(f"[render_tables] updated {path.relative_to(ROOT)}")
            else:
                drift_count += 1
                diff = "".join(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        updated.splitlines(keepends=True),
                        fromfile=str(path.relative_to(ROOT)),
                        tofile=f"{path.relative_to(ROOT)} (expected)",
                        n=2,
                    )
                )
                sys.stderr.write(f"[render_tables] DRIFT detected in {path.relative_to(ROOT)}:\n{diff}\n")

    if seen_markers == 0:
        sys.stderr.write("[render_tables] ERROR: zero GENERATED TABLE markers found in target files.\n")
        return 1
    if not write and drift_count == 0:
        print(f"[render_tables] OK: {seen_markers} generated table blocks match reports/ artifacts.")
    return 1 if drift_count > 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Rewrite generated table blocks in place")
    group.add_argument("--check", action="store_true", help="Verify generated table blocks have zero drift")
    args = parser.parse_args()
    sys.exit(process_files(write=args.write))


if __name__ == "__main__":
    main()
