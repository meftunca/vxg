#!/usr/bin/env python3
"""Generate structured-format and timestamp-layout benchmark files into OUT_DIR (default /tmp/vxgfmt).

Reproduces the deterministic CSV, TSV, logfmt, Apache CLF, pretty JSON, and 7 NDJSON
timestamp-layout corpora used in docs/BENCHMARKS.md.
"""
import datetime
import json
import os
import random
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vxgfmt"
os.makedirs(out_dir, exist_ok=True)
UTC = datetime.timezone.utc

# 1. Five structured formats (CSV, TSV, logfmt, CLF, pretty JSON)
random.seed(11)
svc = ["auth-gateway", "edge-router", "checkout-api", "catalog-index", "search-ranker", "fraud-engine"]
paths = ["/v1/items", "/v1/cart", "/v1/login", "/v2/search", "/health", "/v1/orders", "/v1/profile", "/static/app.js", "/static/logo.png"]
lvl = ["INFO"] * 6 + ["WARN", "ERROR", "DEBUG"]
agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "curl/8.4.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Go-http-client/2.0",
]
N = 250000
t = 1790000000
rows = []
for _ in range(N):
    t += random.choice([0, 0, 1, 1, 2, 5])
    rows.append(dict(
        ts=t,
        level=random.choice(lvl),
        service=random.choice(svc),
        req_id="%016x" % random.getrandbits(64),
        path=random.choice(paths),
        status=random.choices([200, 201, 204, 301, 404, 500], [70, 5, 5, 3, 12, 5])[0],
        latency_ms=int(random.lognormvariate(3, 0.8)),
        bytes=random.randint(100, 50000),
        user="u%06d" % random.randint(0, 99999),
        agent=random.choice(agents),
        ip="10.%d.%d.%d" % (random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)),
    ))
H = list(rows[0].keys())
with open(os.path.join(out_dir, "data.csv"), "w") as f:
    f.write(",".join(H) + "\n" + "\n".join(",".join(str(r[k]) for k in H) for r in rows) + "\n")
with open(os.path.join(out_dir, "data.tsv"), "w") as f:
    f.write("\t".join(H) + "\n" + "\n".join("\t".join(str(r[k]) for k in H) for r in rows) + "\n")
with open(os.path.join(out_dir, "data.logfmt"), "w") as f:
    f.write("\n".join(" ".join(f"{k}={r[k]}" if " " not in str(r[k]) else f'{k}="{r[k]}"' for k in H) for r in rows) + "\n")


def clf(r):
    d = datetime.datetime.fromtimestamp(r["ts"], UTC).strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{r["ip"]} - {r["user"]} [{d}] "GET {r["path"]} HTTP/1.1" {r["status"]} {r["bytes"]} "-" "{r["agent"]}"'


with open(os.path.join(out_dir, "data.clf"), "w") as f:
    f.write("\n".join(clf(r) for r in rows) + "\n")
with open(os.path.join(out_dir, "data.pretty.json"), "w") as f:
    f.write("\n".join(json.dumps(r, indent=2) for r in rows[:60000]) + "\n")

# 2. Seven timestamp-layout NDJSON files
random.seed(5)
svc2 = ["auth", "edge", "checkout", "catalog", "search", "fraud"]
lvl2 = ["INFO"] * 6 + ["WARN", "ERROR", "DEBUG"]


def fmt(kind, ts_val, ms):
    d = datetime.datetime.fromtimestamp(ts_val, UTC)
    if kind == "iso_z":
        return d.strftime("%Y-%m-%dT%H:%M:%SZ")
    if kind == "iso_ms":
        return d.strftime("%Y-%m-%dT%H:%M:%S") + ".%03dZ" % ms
    if kind == "iso_off":
        return d.strftime("%Y-%m-%dT%H:%M:%S") + "+01:00"
    if kind == "iso_space":
        return d.strftime("%Y-%m-%d %H:%M:%S")
    if kind == "syslog":
        return d.strftime("%b %d %H:%M:%S")
    if kind == "epoch_ms":
        return str(ts_val * 1000 + ms)
    if kind == "clf":
        return d.strftime("%d/%b/%Y:%H:%M:%S +0000")
    raise ValueError(kind)


for kind in ["iso_z", "iso_ms", "iso_off", "iso_space", "syslog", "epoch_ms", "clf"]:
    t = 1790000000
    out = []
    for _ in range(120000):
        t += random.choice([0, 0, 1, 1, 2, 5])
        ms = random.randint(0, 999)
        rec = {
            "ts": fmt(kind, t, ms) if kind != "epoch_ms" else int(fmt(kind, t, ms)),
            "level": random.choice(lvl2),
            "service": random.choice(svc2),
            "status": random.choices([200, 404, 500], [80, 15, 5])[0],
            "latency_ms": int(random.lognormvariate(3, 0.8)),
        }
        out.append(json.dumps(rec, separators=(",", ":")))
    with open(os.path.join(out_dir, f"ts_{kind}.ndjson"), "w") as f:
        f.write("\n".join(out) + "\n")
