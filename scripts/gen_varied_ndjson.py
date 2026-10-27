#!/usr/bin/env python3
"""Generate a varied-field NDJSON log (random ids, skewed enums, lognormal latency).

Synthetic, but unlike a repeated-line file every record differs.
Usage: gen_varied_ndjson.py [OUT] [ROWS]   (defaults: /tmp/vxgreal/ndjson_varied.ndjson 300000)
"""
import json, os, random, sys

out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vxgreal/ndjson_varied.ndjson"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
random.seed(7)
svc = ["auth-gateway", "edge-router", "checkout-api", "catalog-index",
       "search-ranker", "fraud-engine", "session-cache", "profile-store"]
reg = ["eu-central-1", "us-east-1", "ap-northeast-1", "us-west-2"]
lvl = ["INFO"] * 6 + ["WARN", "ERROR", "DEBUG"]
paths = ["/v1/items", "/v1/cart", "/v1/login", "/v2/search", "/health", "/v1/orders", "/v1/profile"]
t = 1790000000
lines = []
for _ in range(rows):
    t += random.choice([0, 0, 1, 1, 2, 5])
    rec = {"ts": t, "level": random.choice(lvl), "service": random.choice(svc),
           "region": random.choice(reg), "req_id": "%016x" % random.getrandbits(64),
           "path": random.choice(paths),
           "status": random.choices([200, 201, 204, 301, 404, 500], [70, 5, 5, 3, 12, 5])[0],
           "latency_ms": int(random.lognormvariate(3, 0.8)),
           "bytes": random.randint(100, 50000), "user": "u%06d" % random.randint(0, 99999)}
    lines.append(json.dumps(rec, separators=(",", ":")))
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
