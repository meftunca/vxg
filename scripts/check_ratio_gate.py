#!/usr/bin/env python3
"""CI & local compression ratio (`<= 1.0%`) and encode speed (`<= 10.0%`) regression gate.

Fails with exit code 1 if:
  1. Any checked dataset's VXG compression ratio exceeds its pinned baseline by more than
     `max_regression_pct` (default `1.0%`).
  2. Real-public 8T encode geomean (`--real-csv`, 6-set non-BMP or 7-set all, or per-dataset)
     or CI suite (`--ci-csv` hardware-normalized `vxg / zstd-c-l3` encode geomean) drops by
     more than `max_speed_regression_pct` (default `10.0%`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile

# Pinned v0.5.1/v0.5.2 real-world & CI suite ceilings (encoded_bytes / raw_bytes).
# Any regression > 1.0% relative (`ratio > baseline * 1.01`) fails the build.
CI_SUITE_BASELINES = {
    "telemetry.csv": 0.0408,
    "events.tsv": 0.0335,
    "app-logs.logfmt": 0.0354,
    "json-records.ndjson": 0.0228,
    "field-order-optional-escaping.ndjson": 0.0446,
    "event-mix-identifiers.ndjson": 0.0249,
    "bursty-variable-records.ndjson": 0.0481,
    "ui-render-32bpp.bmp": 0.0049,
    "rgb-frame-24bpp.bmp": 0.0013,
}

REAL_PUBLIC_BASELINES = {
    "nyc-taxi-2024-01.ndjson": 0.0593,
    "nyc-taxi-2024-01.csv": 0.1830,
    "gharchive-2025-01-15.ndjson": 0.1181,
    "apache-access.clf": 0.0976,
    "macos-install.log": 0.0549,
    "pg100-shakespeare.txt": 0.3786,
    "kodak-truecolor-16.bmp": 0.7987,
}

COVID_BASELINES = {
    "time-series-19-covid-combined.csv": 0.0963,
    "us_confirmed_33m.csv": 0.0456,
}

SRC_UNIQUE_BASELINES = {
    "vxg-src-nonrepeating.rs": 0.1950,
}

# Pinned v0.5.3 8T encode throughput baselines (GB/s) on the 7-file real public corpus.
# Any drop > max_speed_regression_pct (default 10.0%, i.e. < baseline * 0.90) fails the gate.
REAL_PUBLIC_8T_ENCODE_GBPS_BASELINES = {
    "nyc-taxi-2024-01.ndjson": 8.8361,
    "nyc-taxi-2024-01.csv": 3.2155,
    "gharchive-2025-01-15.ndjson": 0.9522,
    "apache-access.clf": 0.7541,
    "macos-install.log": 2.3375,
    "pg100-shakespeare.txt": 0.2567,
    "kodak-truecolor-16.bmp": 0.3909,
}
REAL_PUBLIC_8T_GEOMEAN_6SET_GBPS = 1.5181
REAL_PUBLIC_8T_GEOMEAN_7SET_GBPS = 1.2506

# Hardware-normalized CI suite (`--suite v4`) geomean encode speed ratio (`vxg / zstd-c-l3`)
# measured on the same runner so shared cloud VMs detect >10% VXG encode regressions reliably.
CI_SUITE_REL_ZSTD3_GEOMEAN_BASELINE = 0.7500


def resolve_vxg_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_bin = os.environ.get("VXG_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin
    exe = "vxg.exe" if sys.platform == "win32" else "vxg"
    candidates: list[str] = []
    if os.environ.get("CARGO_TARGET_DIR"):
        candidates.append(os.path.join(os.environ["CARGO_TARGET_DIR"], "release", exe))
    candidates.append(os.path.expanduser(os.path.join("~", ".cargo", "target", "release", exe)))
    candidates.append(os.path.join("target", "release", exe))
    existing = [c for c in candidates if os.path.exists(c)]
    if existing:
        return max(existing, key=lambda p: os.path.getmtime(p))
    return os.path.join("target", "release", exe)


def check_bench_csv(csv_path: str, baselines: dict[str, float], tol: float) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codec = row.get("codec", "")
            op = row.get("operation", "")
            ds = os.path.basename(row.get("dataset", ""))
            if codec != "vxg" or op != "compress" or ds not in baselines:
                continue
            seen.add(ds)
            raw_ratio = row.get("ratio_encoded_over_raw") or row.get("ratio")
            if raw_ratio is None:
                failures.append(f"{ds}: missing ratio column in {csv_path}")
                continue
            ratio = float(raw_ratio)
            base = baselines[ds]
            limit = base * (1.0 + tol / 100.0)
            delta_pct = (ratio / base - 1.0) * 100.0
            status = "PASS" if ratio <= limit else "FAIL"
            print(
                f"[{status}] {ds:38s} ratio={ratio:.6f} baseline={base:.6f} "
                f"delta={delta_pct:+.2f}% (limit=+{tol:.2f}%)"
            )
            if ratio > limit:
                failures.append(
                    f"{ds}: ratio {ratio:.6f} regressed {delta_pct:+.2f}% vs baseline {base:.6f}"
                )
    missing = sorted(set(baselines.keys()) - seen)
    for ds in missing:
        failures.append(f"{ds}: missing from benchmark CSV {csv_path}")
    return failures


def check_encode_speed_csv(
    csv_path: str,
    speed_baselines: dict[str, float] | None,
    geomean_6set_base: float | None,
    geomean_7set_base: float | None,
    rel_zstd3_geomean_base: float | None,
    speed_tol_pct: float,
) -> list[str]:
    failures: list[str] = []
    vxg_gbps: dict[str, float] = {}
    zstd3_gbps: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row.get("operation", "")
            if op != "compress":
                continue
            ds = os.path.basename(row.get("dataset", ""))
            codec = row.get("codec", "")
            raw_gbps = row.get("median_GBps") or row.get("encode_gbps")
            if raw_gbps is None:
                continue
            val = float(raw_gbps)
            if codec == "vxg":
                vxg_gbps[ds] = val
            elif codec == "zstd-c-l3":
                zstd3_gbps[ds] = val

    keep_factor = 1.0 - speed_tol_pct / 100.0
    if speed_baselines:
        for ds, base_g in speed_baselines.items():
            if ds not in vxg_gbps:
                failures.append(f"{ds}: missing VXG compress median_GBps in {csv_path}")
                continue
            cur_g = vxg_gbps[ds]
            floor_g = base_g * keep_factor
            delta_pct = (cur_g / base_g - 1.0) * 100.0
            status = "PASS" if cur_g >= floor_g else "FAIL"
            print(
                f"[{status}] {ds:38s} encode_GBps={cur_g:.4f} baseline={base_g:.4f} "
                f"delta={delta_pct:+.2f}% (floor=-{speed_tol_pct:.1f}%)"
            )
            if cur_g < floor_g:
                failures.append(
                    f"{ds}: encode speed {cur_g:.4f} GB/s regressed {delta_pct:+.2f}% "
                    f"vs baseline {base_g:.4f} GB/s (floor {floor_g:.4f} GB/s)"
                )

    if geomean_6set_base is not None:
        non_bmp = [g for ds, g in vxg_gbps.items() if not ds.endswith(".bmp") and g > 0.0]
        if non_bmp:
            geo6 = math.exp(sum(math.log(x) for x in non_bmp) / len(non_bmp))
            floor6 = geomean_6set_base * keep_factor
            delta6 = (geo6 / geomean_6set_base - 1.0) * 100.0
            st6 = "PASS" if geo6 >= floor6 else "FAIL"
            print(
                f"[{st6}] {'GEOMEAN (6-set non-BMP 8T encode)':38s} encode_GBps={geo6:.4f} "
                f"baseline={geomean_6set_base:.4f} delta={delta6:+.2f}% (floor=-{speed_tol_pct:.1f}%)"
            )
            if geo6 < floor6:
                failures.append(
                    f"6-set non-BMP 8T encode geomean {geo6:.4f} GB/s regressed {delta6:+.2f}% "
                    f"vs baseline {geomean_6set_base:.4f} GB/s"
                )

    if geomean_7set_base is not None and vxg_gbps:
        vals7 = [g for g in vxg_gbps.values() if g > 0.0]
        if vals7:
            geo7 = math.exp(sum(math.log(x) for x in vals7) / len(vals7))
            floor7 = geomean_7set_base * keep_factor
            delta7 = (geo7 / geomean_7set_base - 1.0) * 100.0
            st7 = "PASS" if geo7 >= floor7 else "FAIL"
            print(
                f"[{st7}] {'GEOMEAN (7-set all 8T encode)':38s} encode_GBps={geo7:.4f} "
                f"baseline={geomean_7set_base:.4f} delta={delta7:+.2f}% (floor=-{speed_tol_pct:.1f}%)"
            )
            if geo7 < floor7:
                failures.append(
                    f"7-set all 8T encode geomean {geo7:.4f} GB/s regressed {delta7:+.2f}% "
                    f"vs baseline {geomean_7set_base:.4f} GB/s"
                )

    if rel_zstd3_geomean_base is not None:
        rels = [
            vxg_gbps[ds] / zstd3_gbps[ds]
            for ds in CI_SUITE_BASELINES
            if ds in vxg_gbps and ds in zstd3_gbps and vxg_gbps[ds] > 0.0 and zstd3_gbps[ds] > 0.0
        ]
        if rels:
            geo_rel = math.exp(sum(math.log(x) for x in rels) / len(rels))
            floor_rel = rel_zstd3_geomean_base * keep_factor
            delta_rel = (geo_rel / rel_zstd3_geomean_base - 1.0) * 100.0
            st_rel = "PASS" if geo_rel >= floor_rel else "FAIL"
            print(
                f"[{st_rel}] {'CI v4 encode geomean (vxg/zstd-c-l3)':38s} rel_speed={geo_rel:.4f}x "
                f"baseline={rel_zstd3_geomean_base:.4f}x delta={delta_rel:+.2f}% (floor=-{speed_tol_pct:.1f}%)"
            )
            if geo_rel < floor_rel:
                failures.append(
                    f"CI v4 relative encode speed (vxg/zstd-c-l3) {geo_rel:.4f}x regressed "
                    f"{delta_rel:+.2f}% vs baseline {rel_zstd3_geomean_base:.4f}x"
                )
    return failures


def check_corpus_dir(vxg_bin: str, corpus_dir: str, baselines: dict[str, float], tol: float) -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        out_vxg = os.path.join(tmp, "gate.vxg")
        for ds, base in baselines.items():
            src = os.path.join(corpus_dir, ds)
            if not os.path.exists(src):
                failures.append(f"{ds}: missing corpus file at {src}")
                continue
            raw_sz = os.path.getsize(src)
            subprocess.run(
                [vxg_bin, "compress", src, out_vxg, "-t", "8", "-f", "-q"],
                check=True,
            )
            enc_sz = os.path.getsize(out_vxg)
            ratio = enc_sz / raw_sz
            limit = base * (1.0 + tol / 100.0)
            delta_pct = (ratio / base - 1.0) * 100.0
            status = "PASS" if ratio <= limit else "FAIL"
            print(
                f"[{status}] {ds:38s} ratio={ratio:.6f} baseline={base:.6f} "
                f"delta={delta_pct:+.2f}% (limit=+{tol:.2f}%)"
            )
            if ratio > limit:
                failures.append(
                    f"{ds}: ratio {ratio:.6f} regressed {delta_pct:+.2f}% vs baseline {base:.6f}"
                )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci-csv", help="Path to vxg-bench --suite v4 CSV")
    ap.add_argument("--src-csv", help="Path to vxg-bench --suite src-unique CSV")
    ap.add_argument("--real-csv", help="Path to vxg-bench real-public CSV")
    ap.add_argument("--real-dir", help="Directory containing real public corpora")
    ap.add_argument("--covid-dir", help="Directory containing datasets/covid-19 CSVs")
    ap.add_argument("--vxg-bin", default=None, help="Path to vxg CLI binary")
    ap.add_argument("--max-regression-pct", type=float, default=1.0)
    ap.add_argument(
        "--max-speed-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed encode throughput (GB/s) regression percentage (default: 10.0%%)",
    )
    ap.add_argument("--extra-json", help="Optional JSON mapping extra file paths -> baseline ratio")
    args = ap.parse_args()
    vxg_bin = resolve_vxg_bin(args.vxg_bin)

    if not any(
        [args.ci_csv, args.src_csv, args.real_csv, args.real_dir, args.covid_dir, args.extra_json]
    ):
        ap.error(
            "provide at least one of --ci-csv, --src-csv, --real-csv, --real-dir, --covid-dir, or --extra-json"
        )

    failures: list[str] = []
    if args.ci_csv:
        failures.extend(check_bench_csv(args.ci_csv, CI_SUITE_BASELINES, args.max_regression_pct))
        failures.extend(
            check_encode_speed_csv(
                args.ci_csv,
                None,
                None,
                None,
                CI_SUITE_REL_ZSTD3_GEOMEAN_BASELINE,
                args.max_speed_regression_pct,
            )
        )
    if args.src_csv:
        failures.extend(
            check_bench_csv(args.src_csv, SRC_UNIQUE_BASELINES, args.max_regression_pct)
        )
    if args.real_csv:
        failures.extend(check_bench_csv(args.real_csv, REAL_PUBLIC_BASELINES, args.max_regression_pct))
        failures.extend(
            check_encode_speed_csv(
                args.real_csv,
                REAL_PUBLIC_8T_ENCODE_GBPS_BASELINES,
                REAL_PUBLIC_8T_GEOMEAN_6SET_GBPS,
                REAL_PUBLIC_8T_GEOMEAN_7SET_GBPS,
                None,
                args.max_speed_regression_pct,
            )
        )
    if args.real_dir:
        failures.extend(
            check_corpus_dir(vxg_bin, args.real_dir, REAL_PUBLIC_BASELINES, args.max_regression_pct)
        )
    if args.covid_dir:
        failures.extend(
            check_corpus_dir(vxg_bin, args.covid_dir, COVID_BASELINES, args.max_regression_pct)
        )
    if args.extra_json:
        with open(args.extra_json, encoding="utf-8") as jf:
            extra: dict[str, float] = json.load(jf)
        with tempfile.TemporaryDirectory() as tmp:
            out_vxg = os.path.join(tmp, "extra.vxg")
            for path, base in extra.items():
                raw_sz = os.path.getsize(path)
                subprocess.run([vxg_bin, "compress", path, out_vxg, "-t", "8", "-f", "-q"], check=True)
                ratio = os.path.getsize(out_vxg) / raw_sz
                limit = base * (1.0 + args.max_regression_pct / 100.0)
                delta_pct = (ratio / base - 1.0) * 100.0
                status = "PASS" if ratio <= limit else "FAIL"
                print(f"[{status}] {os.path.basename(path):38s} ratio={ratio:.6f} baseline={base:.6f} delta={delta_pct:+.2f}%")
                if ratio > limit:
                    failures.append(f"{path}: ratio {ratio:.6f} > limit {limit:.6f}")

    if failures:
        print("\nRATIO REGRESSION GATE FAILED:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print("\nRatio regression gate passed (all checked corpora within tolerance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
