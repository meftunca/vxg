#!/usr/bin/env python3
"""Serial same-host CPU rail energy comparison; stdlib only.

powermetrics reports estimates, not wall-plug or per-process joules. Text mW units
are used deliberately: private plist '*_energy' fields vary across OS versions.
"""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODECS = ("vxg", "lz4-a1", "lz4-a4", "zstd-m3", "zstd-1", "zstd-3")
HEADER = re.compile(r"\*\*\* Sampled system activity \((.*?)\) \(([0-9.]+)ms elapsed\) \*\*\*")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_power(text):
    """Return sample intervals. Printed local timestamps have 1 s precision.

    Store the midpoint AND the uncertainty; edge windows are excluded later.
    All raw text is preserved by the runner for OS/schema review.
    """
    matches = list(HEADER.finditer(text))
    result = []
    for i, match in enumerate(matches):
        body = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        seconds = float(match[2]) / 1000
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("nonpositive/nonfinite power sample duration")
        # Recent macOS includes a numeric UTC offset (e.g. +0300). Preserve
        # that offset rather than interpreting the header in the reader's zone.
        # Older captures omit it and retain their original local-time semantics.
        stamp = match[1].strip()
        try:
            sampled_at = dt.datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y %z")
        except ValueError:
            sampled_at = dt.datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y")
        end = sampled_at.timestamp() + 0.5
        sample = {"start": end - seconds, "end": end, "seconds": seconds}
        for rail in ("CPU", "GPU", "ANE"):
            value = re.search(r"^" + rail + r" Power:\s*([0-9.]+)\s*mW\s*$", body, re.M)
            if value:
                watts = float(value[1]) / 1000
                if not math.isfinite(watts) or watts < 0:
                    raise ValueError("invalid power value")
                sample[rail.lower() + "_W"] = watts
        if "cpu_W" not in sample:
            raise ValueError("CPU Power in mW missing; do not treat an unavailable counter as zero")
        if all(rail + "_W" in sample for rail in ("cpu", "gpu", "ane")):
            sample["cpu_gpu_ane_W"] = sum(sample[rail + "_W"] for rail in ("cpu", "gpu", "ane"))
        pressure = re.search(r"(?:Current pressure level|Thermal pressure):\s*(.+)", body)
        sample["thermal_pressure"] = pressure[1].strip() if pressure else None
        result.append(sample)
    if not result:
        raise ValueError("no powermetrics samples; privilege or format failure")
    return result


def interpolate(points, when):
    times = [p[0] for p in points]
    i = bisect.bisect_right(times, when) - 1
    if i < 0 or when > times[-1]:
        raise ValueError("power interval is outside measured work")
    if i == len(points) - 1:
        return points[i][1]
    t0, b0 = points[i]
    t1, b1 = points[i + 1]
    return b0 + (b1 - b0) * (when - t0) / (t1 - t0)


def work_points(events):
    ticks = [e for e in events if e.get("event") in ("start", "progress", "end")]
    if not ticks or ticks[0]["event"] != "start" or ticks[-1]["event"] != "end":
        raise ValueError("incomplete workload timing")
    if not any(e.get("event") == "verified" and e.get("roundtrip") is True for e in events):
        raise ValueError("workload did not pass its post-measurement byte comparison")
    for left, right in zip(ticks, ticks[1:]):
        wall = right["unix_ns"] - left["unix_ns"]
        mono = right["elapsed_ns"] - left["elapsed_ns"]
        if wall <= 0 or mono <= 0 or abs(wall - mono) > 20_000_000:
            raise ValueError("nonmonotonic timestamps or wall-clock adjustment")
        if right["raw_bytes_processed"] <= left["raw_bytes_processed"]:
            # End may immediately follow an emitted progress record.
            if right["event"] != "end" or right["raw_bytes_processed"] != left["raw_bytes_processed"]:
                raise ValueError("nonmonotonic work counters")
    return [(e["unix_ns"] / 1e9, e["raw_bytes_processed"]) for e in ticks], ticks


def average_power(samples):
    seconds = sum(s["seconds"] for s in samples)
    return sum(s["cpu_W"] * s["seconds"] for s in samples) / seconds


def integrate(samples, events, idle_before, idle_after):
    points, ticks = work_points(events)
    # Whole interval +/-0.5 s must remain inside workload with an additional
    # 0.5 s warm/end guard. Second-resolution header timestamps are not exact.
    chosen = [s for s in samples if s["start"] >= points[0][0] + 1 and s["end"] <= points[-1][0] - 1]
    if len(chosen) < 3 or sum(s["seconds"] for s in chosen) < 2:
        raise ValueError("not enough interior power coverage; increase --seconds")
    # Consecutive text timestamps can round to equal seconds; relative intervals
    # may overlap by rounding. Avoid double counting overlapping windows.
    selected = []
    for sample in chosen:
        if not selected or sample["start"] >= selected[-1]["end"]:
            selected.append(sample)
    if len(selected) < 3:
        raise ValueError("too few nonoverlapping interior samples")
    duration = sum(s["seconds"] for s in selected)
    byte_count = sum(interpolate(points, s["end"]) - interpolate(points, s["start"]) for s in selected)
    if byte_count <= 0:
        raise ValueError("no work inside power intervals")
    gb = byte_count / 1e9
    joules = sum(s["cpu_W"] * s["seconds"] for s in selected)
    idle = (idle_before + idle_after) / 2
    incremental = joules - idle * duration
    idle_drift = abs(idle_after - idle_before) / max(idle, 0.001)
    return {
        "energy_status": "measured_cpu_rail_estimate",
        "cpu_rail_J": joules, "cpu_rail_J_per_GB": joules / gb,
        "cpu_rail_average_W": joules / duration,
        "sampled_raw_GB_estimate": gb, "sampled_seconds": duration,
        "sampled_GBps": gb / duration, "power_samples_used": len(selected),
        "power_samples_available": len(samples),
        "idle_before_W": idle_before, "idle_after_W": idle_after,
        "idle_drift_fraction": idle_drift,
        "incremental_cpu_J_per_GB": incremental / gb,
        "incremental_usable": incremental > 0 and idle_drift <= 0.2,
        "timestamp_uncertainty_seconds": 0.5,
        "work_interpolation": "linear between ~100ms completed-call counters; partial calls estimated",
        "thermal_pressure": sorted({s["thermal_pressure"] for s in selected if s["thermal_pressure"]}),
    }


def power_command(count, interval):
    return ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power,gpu_power,ane_power,thermal",
            "--format", "text", "--sample-rate", str(interval), "--sample-count", str(count),
            "--poweravg", "0", "--buffer-size", "0"]


def collect_idle(path, count, interval):
    with path.open("w") as stream:
        run = subprocess.run(power_command(count, interval), stdout=stream, stderr=subprocess.PIPE,
                             text=True, timeout=count * interval / 1000 + 20, env={**os.environ, "LC_ALL": "C"})
    if run.returncode:
        raise RuntimeError(run.stderr.strip())
    return average_power(parse_power(path.read_text()))


def phase(args, folder, codec, operation):
    command = [str(args.binary), str(args.input), codec, operation, str(args.seconds), "neon", "--wait"]
    monitor = None
    with (folder / "workload.stderr").open("w") as stderr:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(worker.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=60):
                    raise RuntimeError("workload readiness timeout")
            first = worker.stdout.readline()
            meta = json.loads(first)
            if meta.get("event") != "ready":
                raise RuntimeError("workload did not become ready")
            if not args.no_power:
                idle_before = collect_idle(folder / "idle-before.txt", 3, args.interval_ms)
            with (folder / "power.txt").open("w") as power, (folder / "power.stderr").open("w") as powererr:
                if not args.no_power:
                    count = math.ceil((args.seconds + 3) * 1000 / args.interval_ms)
                    monitor = subprocess.Popen(power_command(count, args.interval_ms), stdout=power, stderr=powererr,
                                               env={**os.environ, "LC_ALL": "C"})
                output, _ = worker.communicate("GO\n", timeout=args.seconds + 30)
                (folder / "workload.jsonl").write_text(first + output)
                if worker.returncode:
                    raise RuntimeError("workload failed; inspect workload.stderr")
                if monitor is not None and monitor.wait(timeout=30):
                    raise RuntimeError("power sampling failed; inspect power.stderr")
            events = [json.loads(line) for line in output.splitlines()]
            _, ticks = work_points(events)
            elapsed = (ticks[-1]["elapsed_ns"] - ticks[0]["elapsed_ns"]) / 1e9
            raw_done = ticks[-1]["raw_bytes_processed"]
            e0 = ticks[0].get("process_energy_nj", 0)
            e1 = ticks[-1].get("process_energy_nj", 0)
            i0 = ticks[0].get("process_instructions", 0)
            i1 = ticks[-1].get("process_instructions", 0)
            delta_enj = max(0, e1 - e0)
            delta_inst = max(0, i1 - i0)
            proc_j_gb = (delta_enj / raw_done) if (delta_enj > 0 and raw_done > 0) else None
            proc_w = (delta_enj / (elapsed * 1e9)) if (delta_enj > 0 and elapsed > 0) else None
            proc_ipb = (delta_inst / raw_done) if (delta_inst > 0 and raw_done > 0) else None
            result = {**meta, "elapsed_seconds": elapsed,
                      "GBps": raw_done / 1e9 / elapsed,
                      "energy_status": "measured_process_rusage_v6" if (args.no_power and proc_j_gb is not None) else "unavailable_no_power_mode",
                      "cpu_rail_J_per_GB": None,
                      "cpu_rail_average_W": None, "incremental_cpu_J_per_GB": None,
                      "process_cpu_J_per_GB": proc_j_gb,
                      "process_cpu_average_W": proc_w,
                      "process_instructions_per_byte": proc_ipb}
            if not args.no_power:
                idle_after = collect_idle(folder / "idle-after.txt", 3, args.interval_ms)
                result.update(integrate(parse_power((folder / "power.txt").read_text()), events, idle_before, idle_after))
            result.pop("event", None)
            write_json(folder / "result.json", result)
            return result
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
            if monitor is not None and monitor.poll() is None:
                # The privileged sampler is finite: allow it to exit naturally.
                monitor.wait(timeout=args.seconds + 30)


def summarize(rows):
    result = []
    for codec, operation in sorted({(r["codec"], r["operation"]) for r in rows}):
        group = [r for r in rows if r["codec"] == codec and r["operation"] == operation]
        item = {"codec": codec, "operation": operation, "profile": group[0]["profile"],
                "trials": len(group), "ratio": group[0]["ratio"],
                "incremental_usable_all_trials": all(r.get("incremental_usable", False) for r in group)}
        for key in ("GBps", "cpu_rail_J_per_GB", "cpu_rail_average_W", "incremental_cpu_J_per_GB",
                    "process_cpu_J_per_GB", "process_cpu_average_W", "process_instructions_per_byte"):
            values = [r[key] for r in group if r.get(key) is not None]
            item[key] = statistics.median(values) if len(values) == len(group) else None
            if values:
                item[key + "_min"] = min(values)
                item[key + "_max"] = max(values)
        item["energy_vs_lz4_a1"] = None
        item["process_energy_vs_lz4_a1"] = None
        result.append(item)
    for item in result:
        baseline = next((r for r in result if r["codec"] == "lz4-a1" and r["operation"] == item["operation"]), None)
        if baseline and baseline["cpu_rail_J_per_GB"] and item["cpu_rail_J_per_GB"] is not None:
            item["energy_vs_lz4_a1"] = item["cpu_rail_J_per_GB"] / baseline["cpu_rail_J_per_GB"]
        if baseline and baseline["process_cpu_J_per_GB"] and item["process_cpu_J_per_GB"] is not None:
            item["process_energy_vs_lz4_a1"] = item["process_cpu_J_per_GB"] / baseline["process_cpu_J_per_GB"]
    for codec in {r["codec"] for r in result}:
        enc = next((r for r in result if r["codec"] == codec and r["operation"] == "encode"), None)
        dec = next((r for r in result if r["codec"] == codec and r["operation"] == "decode"), None)
        if enc and dec and enc.get("GBps") and dec.get("GBps") and enc.get("ratio", 0) > 0:
            vpi = math.sqrt(enc["GBps"] * dec["GBps"]) / enc["ratio"]
            bci = ((enc["GBps"] * dec["GBps"]) / enc["ratio"]) ** (1.0 / 3.0)
            ej = enc.get("process_cpu_J_per_GB") or enc.get("cpu_rail_J_per_GB")
            dj = dec.get("process_cpu_J_per_GB") or dec.get("cpu_rail_J_per_GB")
            gci = (vpi / math.sqrt(ej * dj)) if (ej and dj and ej > 0 and dj > 0) else None
            for row in (enc, dec):
                row["vpi_index"] = vpi
                row["bci_index"] = bci
                row["gci_index"] = gci
        else:
            for row in [r for r in result if r["codec"] == codec]:
                row["vpi_index"] = None
                row["bci_index"] = None
                row["gci_index"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="Defaults to Cargo's release/vxg-energy-workload; build it first")
    parser.add_argument("--input", type=Path, default=ROOT / "corpus/qualification/json-records-16MiB.ndjson")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=list(CODECS))
    parser.add_argument("--authorize", action="store_true", help="Ask sudo for credentials in this local terminal; never run benchmark as root")
    parser.add_argument("--no-power", action="store_true", help="Test workloads without sensors; energy remains null")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not (0.05 <= args.seconds <= 3600) or not (1 <= args.trials <= 100):
        parser.error("invalid duration or trial count")
    if args.interval_ms != 1000:
        parser.error("text timestamp precision requires --interval-ms 1000")
    if not args.no_power and args.seconds < 10:
        parser.error("energy runs require at least ten seconds")
    if os.geteuid() == 0:
        parser.error("run as your normal user with --authorize; only powermetrics needs root")
    if args.binary is None:
        cargo = subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        args.binary = Path(json.loads(cargo.stdout)["target_directory"]) / "release/vxg-energy-workload"
    args.binary, args.input = args.binary.resolve(), args.input.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"schema": 1, "status": "running", "binary": str(args.binary), "input": str(args.input),
            "scope": "same-host system CPU rail estimates, not per-process or wall-plug joules",
            "timestamp_method": "powermetrics text local-second midpoint, +/-0.5s; guarded interior intervals; interpolated work counters",
            "profile_note": "VXG decode includes frame parsing; C reference blocks use prepared metadata and an accounting envelope as in BENCHMARKS.md",
            "seed": args.seed, "seconds_per_phase": args.seconds, "trials": args.trials,
            "power_enabled": not args.no_power, "os": os.uname().version}
    try:
        meta["binary_sha256"], meta["input_sha256"] = sha256(args.binary), sha256(args.input)
        manifest = json.loads((ROOT / "reports/qualification/manifest.json").read_text())
        member = next((m for m in manifest["members"] if (ROOT / m["path"]).resolve() == args.input), None)
        if member and (meta["input_sha256"] != member["sha256"] or args.input.stat().st_size != member["bytes"]):
            raise ValueError("frozen corpus SHA/size mismatch")
        meta["frozen_member"] = member["id"] if member else None
        if not args.no_power:
            if args.authorize:
                subprocess.run(["sudo", "-v"], check=True)
            collect_idle(args.output / "preflight-power.txt", 1, args.interval_ms)
        schedule = []
        rng = random.Random(args.seed)
        for trial in range(1, args.trials + 1):
            pairs = [(c, op) for c in dict.fromkeys(args.codecs) for op in ("encode", "decode")]
            rng.shuffle(pairs)
            schedule.extend((trial, c, op) for c, op in pairs)
        meta["schedule"] = schedule
        write_json(args.output / "run.json", meta)
        rows = []
        for order, (trial, codec, operation) in enumerate(schedule, 1):
            print(f"[{order}/{len(schedule)}] trial={trial} {codec} {operation}", flush=True)
            folder = args.output / f"{order:03d}-{codec}-{operation}"
            folder.mkdir()
            row = phase(args, folder, codec, operation)
            row.update(trial=trial, order=order)
            rows.append(row)
            write_json(args.output / "trials.json", rows)
        summary = summarize(rows)
        write_json(args.output / "summary.json", summary)
        keys = ["codec", "operation", "profile", "trials", "ratio", "GBps", "process_cpu_average_W", "process_cpu_J_per_GB", "process_instructions_per_byte", "process_energy_vs_lz4_a1", "vpi_index", "bci_index", "gci_index", "cpu_rail_average_W", "cpu_rail_J_per_GB", "incremental_cpu_J_per_GB", "incremental_usable_all_trials", "energy_vs_lz4_a1"]
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        meta["status"] = "workload_smoke_only" if args.no_power else "directional_energy_comparison"
        write_json(args.output / "run.json", meta)
        print(f"Completed: {args.output / 'summary.csv'}", flush=True)
    except Exception as error:
        meta.update(status="failed", error=str(error))
        write_json(args.output / "run.json", meta)
        raise


if __name__ == "__main__":
    main()
