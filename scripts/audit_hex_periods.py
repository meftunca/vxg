#!/usr/bin/env python3
"""Generic quoted-lowercase-hex per-value period opportunity audit, not a codec.

No field names or generator rules; retain block-local chunks and evaluate every
matching lexical string. Cut strings do not participate and their bytes are NOT
assumed removed. Savings are an optimistic opportunity bound, pending full packet.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def uleb_size(n):
    return max(1, (n.bit_length() + 6) // 7)


def period(value):
    for p in range(1, min(64, len(value)) + 1):
        if all(value[i] == value[i % p] for i in range(p, len(value))):
            return p
    return None


def audit(path):
    histogram = collections.Counter()
    raw_cost = variable_cost = common_cost = seed_cost = rows = blocks = 0
    common_histogram = collections.Counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(65536):
            digest.update(block)
            blocks += 1
            values = re.findall(rb'"([0-9a-f]+)"', block)
            eligible = []
            for value in values:
                p = period(value)
                if p is None:
                    continue
                histogram[p] += 1
                rows += 1
                eligible.append(value)
                raw_cost += uleb_size(len(value)) + (len(value) + 1) // 2
                variable_cost += uleb_size(len(value)) + uleb_size(p) + (p + 1) // 2
                seed_cost += (p + 1) // 2
            if eligible:
                options = []
                for p in range(1, 65):
                    if all(all(value[i] == value[i % p] for i in range(p, len(value))) for value in eligible):
                        cost = 1 + uleb_size(p) + sum(uleb_size(len(v)) + (min(p, len(v)) + 1) // 2 for v in eligible)
                        options.append((cost, p))
                if options:
                    cost, p = min(options)
                    common_cost += cost
                    common_histogram[p] += 1
                    variable_cost += 1  # per-block model selector
                else:
                    raise ValueError("no common <=64 period: report needs a RAW escape model")
    return {"input": str(path), "sha256": digest.hexdigest(), "blocks": blocks,
            "eligible_values": rows, "period_histogram": dict(sorted(histogram.items())),
            "raw_nibble_values_bytes": raw_cost, "per_value_period_bytes": variable_cost,
            "common_period_reference_bytes": common_cost,
            "common_period_block_histogram": dict(sorted(common_histogram.items())),
            "per_value_seed_bytes_only": seed_cost,
            "savings_vs_common_period_reference": common_cost - variable_cost,
            "optimistic_value_savings_bytes": raw_cost - variable_cost,
            "scope": "lexical opportunity audit; no complete frame ratio or codec-speed claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
