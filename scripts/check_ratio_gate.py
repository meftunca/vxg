#!/usr/bin/env python3
"""CI & local compression ratio regression gate (`<= 1.0%` tolerance).

Fails with exit code 1 if any checked dataset's VXG compression ratio exceeds its
pinned baseline by more than `max_regression_pct` (default `1.0%`).
"""

from __future__ import annotations

import argparse
import csv
import json
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
    if args.src_csv:
        failures.extend(
            check_bench_csv(args.src_csv, SRC_UNIQUE_BASELINES, args.max_regression_pct)
        )
    if args.real_csv:
        failures.extend(check_bench_csv(args.real_csv, REAL_PUBLIC_BASELINES, args.max_regression_pct))
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
