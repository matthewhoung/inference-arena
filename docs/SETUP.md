# Inference Arena - Setup Guide

Quick start guide for thesis committee members and collaborators.

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Setup (3 Commands)

```bash
# 1. Clone
git clone https://github.com/matthewhoung/inference-arena.git
cd inference-arena

# 2. Install dependencies
make install

# 3. Run tests
make test
```

That's it. You're ready to run experiments.

## Run Experiments

```bash
# Start infrastructure (MinIO, Prometheus, Grafana)
make start-infra

# Start an architecture
make start-mono     # Architecture A: Monolithic
make start-micro    # Architecture B: Microservices
make start-triton   # Architecture C: Triton

# Stop everything
make stop-all
```

## Service Endpoints

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| OTel Collector | http://localhost:8889/metrics | - |

## Pre-commit Hooks

Install pre-commit hooks for automated linting before push:

```bash
# Install pre-commit
pip install pre-commit

# Install pre-push hooks
pre-commit install --hook-type pre-push
```

The hooks run automatically when you `git push` and check:
- **ruff**: Linting and auto-formatting
- **mypy**: Static type checking

To run hooks manually:
```bash
pre-commit run --all-files --hook-stage pre-push
```

## Testing

### Development Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests with coverage (80% threshold) |
| `make test-fast` | Run tests without slow/load markers |
| `make test-unit` | Run unit tests only (no services needed) |
| `make test-load` | Run load tests (requires running services) |
| `make validate` | Validate infrastructure configuration |
| `make lint` | Run linters (ruff + mypy) |
| `make format` | Format code (black + ruff) |

### Shortcuts

| Shortcut | Command |
|----------|---------|
| `make t` | `make test` |
| `make tf` | `make test-fast` |
| `make tl` | `make test-load` |

### Coverage Information

- **Current coverage:** 88%
- **Minimum threshold:** 80% (enforced in pyproject.toml)
- **Coverage report:** `results/coverage_html/index.html`

### Running Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_specific.py -v

# Run with specific marker
uv run pytest tests/ -m "not slow" -v

# Run with load tests (skipped by default)
uv run pytest tests/ --load -v
```

For common testing issues (coverage below threshold, health check timeouts, tests hanging), see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#testing-issues).

## Available Make Commands

Run `make help` to see all commands:

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make test` | Run all tests with coverage |
| `make test-fast` | Run tests without slow markers |
| `make test-unit` | Run unit tests only |
| `make test-load` | Run load tests |
| `make validate` | Validate infrastructure configuration |
| `make lint` | Run linters (ruff + mypy) |
| `make format` | Format code (black + ruff fix) |
| `make start-infra` | Start infrastructure services |
| `make start-mono` | Start monolithic architecture |
| `make start-micro` | Start microservices architecture |
| `make start-triton` | Start Triton architecture |
| `make stop-all` | Stop all containers |
| `make clean` | Remove caches and build artifacts |

## Project Structure

```
inference-arena/
├── experiment.yaml              # Single source of truth
├── Makefile                     # Common commands
├── pyproject.toml               # Python dependencies
├── src/shared/
│   ├── config/                  # Loads experiment.yaml
│   ├── processing/              # Preprocessing pipelines
│   ├── model/                   # Model registry
│   └── triton/                  # Triton config & MinIO utilities
├── architectures/
│   ├── monolithic/              # Architecture A
│   ├── microservices/           # Architecture B
│   └── triton/                  # Architecture C
├── infrastructure/
│   ├── docker-compose.infra.yml # MinIO, Prometheus, Grafana
│   ├── grafana/                 # Dashboards
│   └── prometheus/              # Scrape config
├── scripts/
│   ├── setup/                   # Environment & proto setup
│   └── models/                  # Export & upload models
└── tests/                       # 100+ tests
```

## Configuration Philosophy

### `experiment.yaml` - Scientific Configuration
- Model specifications, preprocessing params, controlled variables
- Pre-registered hypotheses and predictions
- **Git-tracked** for reproducibility

See [EXPERIMENT_CONFIG.md](EXPERIMENT_CONFIG.md) for complete documentation.

### `.env` - Deployment Configuration
- Infrastructure credentials, port mappings
- **Git-ignored** (secrets)

**Key principle:** `experiment.yaml` defines **WHAT** you're testing, `.env` defines **WHERE** to run it.

## Troubleshooting

### "No module named 'shared'"

```bash
make install
```

### Docker services won't start

```bash
# Check Docker is running
docker info

# View logs
docker compose -f infrastructure/docker-compose.infra.yml logs
```

### Port conflicts

Edit `.env` to use different ports, then restart services.

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## For Thesis Committee Members

This project demonstrates:

1. **Reproducible Research** - All parameters in version-controlled `experiment.yaml`
2. **Pre-registered Hypotheses** - Defined before data collection
3. **Single Source of Truth** - No hardcoded values
4. **Comprehensive Testing** - 100+ tests with 80% coverage threshold
5. **Production-Grade Engineering** - Monitoring, infrastructure-as-code

To reproduce experiments:
1. Follow quick setup above
2. Review `experiment.yaml` for controlled variables
3. Run load tests (see thesis methodology chapter)

## Documentation

- **[EXPERIMENT_CONFIG.md](EXPERIMENT_CONFIG.md)** - Complete experiment.yaml reference
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Load testing framework
- **[ENVIRONMENT.md](ENVIRONMENT.md)** - Environment configuration details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[experiment.yaml](../experiment.yaml)** - Full experimental specification

## Questions?

See the thesis methodology chapter for detailed experimental design.
