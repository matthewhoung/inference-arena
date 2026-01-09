# Inference Arena - Load Testing Framework

This directory contains the Locust-based load testing framework for evaluating
the three ML serving architectures (Monolithic, Microservices, Triton).

## Overview

The framework implements a rigorous three-phase testing protocol:

1. **Warmup (60s)** - Prime JIT, ONNX optimizations, CPU caches
2. **Measurement (180s)** - Collect performance metrics under steady state
3. **Cooldown (30s)** - System reset, garbage collection

**Only metrics from the measurement phase are used for analysis.**

## Directory Structure

```
experiments/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for python -m experiments
├── config.py                # Configuration bridge to experiment.yaml
├── dataset.py               # Test image loader (preloaded)
├── metrics.py               # Thread-safe metrics collector
├── shapes.py                # ThreePhaseShape for Locust
├── locustfile.py            # Locust user behavior
├── runner.py                # CLI for experiment orchestration
├── results/
│   ├── __init__.py
│   ├── prometheus_client.py # Prometheus queries for CPU/memory
│   ├── collector.py         # Aggregate metrics
│   └── exporter.py          # CSV/JSON export
└── README.md
```

## Prerequisites

1. **Docker and docker-compose** installed
2. **Infrastructure running:**
   ```bash
   make start-infra
   ```
3. **Models initialized:**
   ```bash
   make models-init-minio
   ```
4. **Dependencies installed:**
   ```bash
   pip install -e ".[experiment]"
   ```

## Quick Start

### Run Full Experiment Matrix

```bash
# All 63 experiments (~4.7 hours)
python -m experiments.runner

# Or equivalently:
python -m experiments
```

### Run Single Architecture

```bash
# Monolithic only (~1.6 hours)
python -m experiments.runner -a monolithic

# Microservices only
python -m experiments.runner -a microservices

# Triton only
python -m experiments.runner -a triton
```

### Run Specific Configuration (Debugging)

```bash
# Single test: 10 users, 1 run, no docker orchestration
python -m experiments.runner -a monolithic -u 10 -r 1 --no-docker

# Multiple load levels
python -m experiments.runner -a triton -u 1 -u 10 -u 50
```

### Dry Run (Show Plan)

```bash
python -m experiments.runner --dry-run
```

## CLI Options

```
python -m experiments.runner [OPTIONS]

Options:
  -a, --architecture [monolithic|microservices|triton]
                                  Architecture(s) to test. Can be repeated.
                                  Default: all three
  -u, --users INTEGER             User level(s) to test. Can be repeated.
                                  Default: 1, 5, 10, 25, 50, 75, 100
  -r, --runs INTEGER              Runs per configuration.
                                  Default: 3 (from experiment.yaml)
  --no-docker                     Skip docker-compose orchestration.
                                  Assumes containers are already running.
  --dry-run                       Show experiment plan without executing.
  --output-dir PATH               Results output directory.
                                  Default: results/experiment/
  --no-prometheus                 Skip Prometheus resource metrics collection.
  -v, --verbose                   Enable verbose logging.
  --help                          Show this message and exit.
```

## Manual Testing with Locust

For interactive debugging, run Locust directly:

```bash
# Start architecture manually
make start-mono

# Run Locust with Web UI
locust -f experiments/locustfile.py --host=http://localhost:8100
# Open http://localhost:8089

# Run headless with custom parameters
locust -f experiments/locustfile.py \
    --host=http://localhost:8100 \
    --headless \
    -u 10 \
    -r 3 \
    -t 270s
```

## Output Format

Results are exported to `results/experiment/`:

```
results/experiment/
├── runs/
│   ├── monolithic_users1_run1_20260106T120000.json
│   ├── monolithic_users1_run2_20260106T121500.json
│   └── ...
├── experiment_summary_20260106.csv
└── experiment_aggregated_20260106.json
```

### CSV Columns

```
architecture, concurrent_users, run_number, timestamp,
total_requests, successful_requests, error_rate_percent, throughput_rps,
client_p50_ms, client_p95_ms, client_p99_ms,
server_p50_ms, server_p95_ms, server_p99_ms,
cpu_avg_percent, cpu_max_percent, memory_avg_mb, memory_max_mb
```

## Architecture Ports

| Architecture  | Port | Endpoint                |
|---------------|------|-------------------------|
| Monolithic    | 8100 | http://localhost:8100   |
| Microservices | 8200 | http://localhost:8200   |
| Triton        | 8300 | http://localhost:8300   |

## Load Levels

| Users | Spawn Rate |
|-------|------------|
| 1     | 1/sec      |
| 5     | 2/sec      |
| 10    | 3/sec      |
| 25    | 5/sec      |
| 50    | 10/sec     |
| 75    | 15/sec     |
| 100   | 20/sec     |

## Environment Variables

| Variable            | Default                  | Description                    |
|---------------------|--------------------------|--------------------------------|
| `PROMETHEUS_URL`    | http://localhost:9090    | Prometheus server URL          |
| `EXPERIMENT_OUTPUT_DIR` | results/experiment/  | Results output directory       |
| `LOCUST_USERS`      | 1                        | User count (set by runner)     |
| `LOG_REQUESTS`      | 0                        | Enable request logging (1=on)  |

## Troubleshooting

### Health Check Timeout

If the architecture doesn't respond:
```bash
# Check container status
docker ps

# View logs
docker logs inference-arena-monolithic

# Restart
make stop-mono && make start-mono
```

### No Test Images

If "No test images found" error:
```bash
python scripts/setup/download-data.py
```

### Prometheus Connection Failed

Ensure infrastructure is running:
```bash
make start-infra
curl http://localhost:9090/api/v1/query?query=up
```

## Author

Matthew Hong - National Chung Hsing University

## References

- [experiment.yaml](../experiment.yaml) - Experiment specification
- [Locust Documentation](https://docs.locust.io/)
