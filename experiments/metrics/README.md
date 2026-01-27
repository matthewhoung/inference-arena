# Metrics Collection Scripts

This directory contains scripts for collecting static code metrics and deployment measurements across all three architectures.

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `count_loc.py` | Count lines of code (application + configuration) | `results/metrics/loc_counts.csv` |
| `count_files.py` | Count files by extension | `results/metrics/file_counts.csv` |
| `count_api_endpoints.py` | Count HTTP/gRPC API endpoints | `results/metrics/api_endpoints.csv` |
| `measure_deployment.sh` | Measure cold-start deployment time | `results/metrics/deployment_times.csv` |
| `aggregate_metrics.py` | Aggregate all metrics into summary | `results/metrics/metrics_summary.csv` |

## Quick Start

```bash
# Generate all static metrics (fast, no Docker needed)
make metrics-static

# Measure deployment times (slow, ~40 min total)
make metrics-deploy-all

# Generate summary from all metrics
make metrics-summary

# Or run everything at once
make metrics-all
```

## Individual Commands

```bash
# LOC counts
make metrics-loc

# File counts
make metrics-files

# API endpoint counts
make metrics-endpoints

# Deployment time for specific architecture
make metrics-deploy ARCH=mono RUNS=3
make metrics-deploy ARCH=micro RUNS=3
make metrics-deploy ARCH=triton RUNS=3
```

## Output Location

All generated CSV files are written to `results/metrics/`:

```
results/metrics/
  loc_counts.csv          # LOC by architecture and category
  file_counts.csv         # File counts by extension
  api_endpoints.csv       # API endpoint counts
  deployment_times.csv    # Cold-start deployment times
  metrics_summary.csv     # Aggregated summary table
```

## Deployment Time Measurement

The `measure_deployment.sh` script measures worst-case cold-start deployment:

1. **Complete cleanup**: Removes architecture images and build cache
2. **Timing**: From `docker compose up --build` to HTTP health endpoint responding
3. **Repeats**: 3 runs per architecture (configurable)

**Important**: Triton takes ~10 minutes per run due to 8GB base image download and TensorRT model conversion.

## Dependencies

- `pygount` - For LOC counting (installed via `uv sync`)
- Docker and Docker Compose - For deployment measurements
- Infrastructure must be running for deployment measurements (`make start-infra`)
