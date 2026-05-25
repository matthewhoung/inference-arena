#!/usr/bin/env python3
"""Compute statistical measures from experiment JSON files and produce stats_verification.csv.

Reads all 63 individual run JSON files (3 architectures × 7 load levels × 3 runs),
computes mean, standard deviation, and 95% confidence intervals for each metric,
and writes the results to analysis/tables/stats_verification.csv.

Usage:
    python -m analysis.utilities.compute_stats
"""

import json
import csv
import math
from pathlib import Path

# Navigate to project root from analysis/utilities/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "experiment" / "runs"
OUTPUT_CSV = PROJECT_ROOT / "analysis" / "tables" / "stats_verification.csv"

ARCH_MAP = {"mono": "Monolithic", "micro": "Microservices", "triton": "Triton"}
ARCHITECTURES = ["mono", "micro", "triton"]
CONCURRENCIES = [1, 5, 10, 25, 50, 75, 100]
RUNS = [1, 2, 3]

METRICS = [
    ("p50_ms", lambda d: d["client_latency"]["p50_ms"]),
    ("p99_ms", lambda d: d["client_latency"]["p99_ms"]),
    ("throughput_rps", lambda d: d["throughput_rps"]),
    ("cpu_percent", lambda d: d["resources"]["totals"]["cpu_avg_percent"]),
    ("memory_mb", lambda d: d["resources"]["totals"]["memory_avg_mb"]),
    ("network_rx_mbps", lambda d: d["resources"]["totals"]["network_rx_bytes_per_sec"] / 1048576),
]


def main():
    rows = []

    for arch in ARCHITECTURES:
        for n in CONCURRENCIES:
            # Load 3 runs
            run_data = []
            for r in RUNS:
                path = RESULTS_DIR / arch / f"users_{n}" / f"run_{r:03d}.json"
                with open(path) as f:
                    run_data.append(json.load(f))

            for metric_name, extractor in METRICS:
                values = [extractor(d) for d in run_data]
                mean = sum(values) / len(values)
                sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                margin = 1.96 * sd / math.sqrt(len(values))
                ci_lower = mean - margin
                ci_upper = mean + margin

                rows.append({
                    "architecture": ARCH_MAP[arch],
                    "concurrency": n,
                    "metric": metric_name,
                    "run1": round(values[0], 4),
                    "run2": round(values[1], 4),
                    "run3": round(values[2], 4),
                    "mean": round(mean, 4),
                    "sd": round(sd, 4),
                    "ci_lower": round(ci_lower, 4),
                    "ci_upper": round(ci_upper, 4),
                })

    fieldnames = ["architecture", "concurrency", "metric", "run1", "run2", "run3", "mean", "sd", "ci_lower", "ci_upper"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
