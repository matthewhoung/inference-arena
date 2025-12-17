# Inference Arena - Setup Guide

Quick start guide for thesis committee members and collaborators.

## Prerequisites

- **Python 3.11+** (Required for all project components)
- **Docker & Docker Compose** (For infrastructure services)
- **Git** (For version control)

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/matthewhoung/inference-arena.git
cd inference-arena
```

### 2. Install Python Dependencies

```bash
# Install project in development mode with test dependencies
pip install -e ".[dev]"
```

This installs:
- Core dependencies: `pyyaml`, `opencv-python`, `numpy`
- Dev dependencies: `pytest`, `black`, `ruff`, `mypy`

### 3. Configure Environment

Run the cross-platform setup script:

```bash
python scripts/setup_env.py
```

**Choose configuration mode:**
- **Development (1):** Quick setup with default passwords - great for testing
- **Production (2):** Auto-generates secure passwords - use for actual experiments
- **Custom (3):** Enter your own credentials

The script creates `.env` from `.env.example` and ensures it's gitignored.

### 4. Verify Configuration

Check that experiment.yaml loads correctly:

```bash
python -c "from shared.config import get_config; print('✓ Config loaded successfully')"
```

Run tests to verify setup:

```bash
# Run all tests
pytest

# Or just config tests
pytest tests/shared/test_config.py -v
```

### 5. Start Infrastructure Services

```bash
docker compose -f infrastructure/docker-compose.infra.yml up -d
```

Verify services are running:

```bash
docker ps
```

You should see 4 containers:
- `inference-arena-minio` (port 9000, 9001)
- `inference-arena-cadvisor` (port 8080)
- `inference-arena-prometheus` (port 9090)
- `inference-arena-grafana` (port 3000)

### 6. Access Services

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin* |
| Grafana | http://localhost:3000 | admin / admin* |
| Prometheus | http://localhost:9090 | (no auth) |
| cAdvisor | http://localhost:8080 | (no auth) |

*\*If you used production mode, check the output from `setup_env.py` for generated passwords*

## Project Structure

```
inference-arena/
├── experiment.yaml          # Single source of truth for experimental config
├── .env.example             # Environment template (committed)
├── .env                     # Your secrets (gitignored, created by setup script)
├── pyproject.toml           # Python dependencies and project config
├── src/
│   └── shared/
│       ├── config.py        # Config loader (reads experiment.yaml)
│       ├── processing/      # Preprocessing pipelines
│       └── model/           # Model registry and exporters
├── infrastructure/
│   ├── docker-compose.infra.yml  # Infrastructure services
│   └── minio/               # MinIO initialization scripts
├── tests/                   # Comprehensive test suite
└── scripts/
    └── setup_env.py         # Environment setup utility
```

## Configuration Philosophy

This project uses **two complementary configuration systems**:

### `experiment.yaml` - Scientific Configuration
- **Purpose:** Reproducible experimental parameters
- **Contents:** Model specs, preprocessing params, controlled variables, hypotheses
- **Git Status:** ✅ Committed (version controlled)
- **Who reads it:** Python code via `shared.config` module

### `.env` - Deployment Configuration
- **Purpose:** Environment-specific secrets and ports
- **Contents:** Passwords, endpoints, local overrides
- **Git Status:** ❌ Never committed (in .gitignore)
- **Who reads it:** Docker Compose and infrastructure scripts

**Key principle:** `experiment.yaml` defines **WHAT** you're testing, `.env` defines **WHERE** and **HOW** to run it.

## Common Tasks

### Run All Tests
```bash
pytest -v
```

### Run Specific Test Suite
```bash
pytest tests/shared/test_config.py -v
pytest tests/shared/test_processing.py -v
pytest tests/infrastructure/ -v -m integration  # requires Docker
```

### Validate experiment.yaml
```bash
python -c "from shared.config import validate_config; validate_config(); print('✓ Valid')"
```

### Check Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Stop Infrastructure
```bash
docker compose -f infrastructure/docker-compose.infra.yml down

# Remove volumes (clean slate)
docker compose -f infrastructure/docker-compose.infra.yml down -v
```

## Troubleshooting

### "No module named 'yaml'" or "No module named 'shared'"

```bash
# Reinstall in editable mode
pip install -e ".[dev]"
```

### Docker services won't start

```bash
# Check Docker is running
docker info

# View logs
docker compose -f infrastructure/docker-compose.infra.yml logs

# Restart services
docker compose -f infrastructure/docker-compose.infra.yml restart
```

### Tests failing after config changes

```bash
# Reload Python to clear cached config
python -c "from shared.config import reload_config; reload_config()"

# Or just restart your Python interpreter
```

### Port conflicts (e.g., "port 9000 already in use")

Edit `.env` to use different ports:
```bash
MINIO_API_PORT=9002
GRAFANA_PORT=3001
# etc.
```

Then restart services.

## For Thesis Committee Members

This project demonstrates:

1. **Reproducible Research:** All experimental parameters in version-controlled `experiment.yaml`
2. **Pre-registered Hypotheses:** Hypotheses defined before data collection with changelog tracking
3. **Single Source of Truth:** No hardcoded values; all preprocessing/model specs from config
4. **Comprehensive Testing:** 100+ tests validating configuration consistency
5. **Production-Grade Engineering:** Proper secrets management, monitoring, and infrastructure-as-code

To reproduce the experiments:
1. Follow setup steps above
2. Review `experiment.yaml` to see controlled variables and hypotheses
3. Run experiment scripts (see methodology chapter in thesis)

## Documentation

- **[ENVIRONMENT.md](../ENVIRONMENT.md)** - Detailed environment configuration guide
- **[experiment.yaml](../experiment.yaml)** - Full experimental specification with inline docs
- **Test files** - Each test file has docstrings explaining what's being validated

## Questions?

See the thesis methodology chapter for detailed experimental design and rationale.
