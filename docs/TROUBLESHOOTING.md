# Troubleshooting Guide

Common issues and solutions for the Inference Arena project.

## Table of Contents

- [Quick Diagnostic Commands](#quick-diagnostic-commands)
- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [Infrastructure Issues](#infrastructure-issues)
- [Model Issues](#model-issues)
- [Testing Issues](#testing-issues)
- [Environment Issues](#environment-issues)
- [gRPC Issues](#grpc-issues)
- [Performance Issues](#performance-issues)

---

## Quick Diagnostic Commands

Run these first to assess system state:

```bash
# Check overall health
make health

# View all running containers
docker ps

# Check infrastructure services
docker compose -f infrastructure/docker-compose.infra.yml ps

# View recent logs
docker compose -f infrastructure/docker-compose.infra.yml logs --tail=50

# Verify Python environment
python -c "import shared; print('shared module OK')"
```

---

## Installation Issues

### "No module named 'shared'"

**Symptom:** Python imports fail with `ModuleNotFoundError: No module named 'shared'`

**Cause:** The shared package is not installed in editable mode.

**Solution:**
```bash
make install
```

This runs `uv sync` which installs the project in editable mode, making the `shared` package available.

---

### Python version mismatch

**Symptom:** Syntax errors or incompatible type hints

**Cause:** Project requires Python 3.11+

**Solution:**
```bash
# Check current version
python --version

# If below 3.11, install a newer version
# macOS with Homebrew:
brew install python@3.12

# Or use pyenv:
pyenv install 3.12.0
pyenv local 3.12.0
```

---

### uv installation failures

**Symptom:** `make install` fails because `uv` command not found

**Cause:** uv package manager not installed

**Solution:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# After installation, restart your terminal
```

---

### Dependencies fail to install

**Symptom:** `uv sync` fails with dependency resolution errors

**Cause:** Conflicting package versions or network issues

**Solution:**
```bash
# Clear uv cache
rm -rf ~/.cache/uv

# Try again
make install

# If still failing, check network and try with verbose output
uv sync -v
```

---

## Docker Issues

### Docker services won't start

**Symptom:** `make start-infra` fails or containers exit immediately

**Cause:** Docker Desktop not running, or containers from previous runs

**Solution:**
```bash
# 1. Verify Docker is running
docker info

# 2. Check for existing containers
docker ps -a | grep inference-arena

# 3. Clean up and restart
make stop-all
docker compose -f infrastructure/docker-compose.infra.yml down -v
make start-infra
```

---

### Port conflicts

**Symptom:** "port is already allocated" error

**Cause:** Another service using the same port (9000, 9090, 3000, etc.)

**Solution:**
```bash
# 1. Find what's using the port (example: 9000)
lsof -i :9000  # macOS/Linux
netstat -ano | findstr :9000  # Windows

# 2. Either stop the conflicting service, or:
# 3. Edit .env to use different ports
echo "MINIO_API_PORT=9100" >> .env
echo "MINIO_CONSOLE_PORT=9101" >> .env

# 4. Restart services
make stop-all && make start-infra
```

---

### Container name conflicts

**Symptom:** "container name already in use" error

**Cause:** Old containers with same names exist (stopped but not removed)

**Solution:**
```bash
# Remove specific containers
docker rm -f inference-arena-monolithic
docker rm -f inference-arena-detection
docker rm -f inference-arena-classification

# Or clean all project containers
docker ps -a | grep inference-arena | awk '{print $1}' | xargs docker rm -f
```

---

### Out of memory

**Symptom:** Containers killed with OOMKilled, or Docker becomes unresponsive

**Cause:** Docker Desktop memory limit too low

**Solution:**

**Docker Desktop (macOS/Windows):**
1. Open Docker Desktop > Settings > Resources
2. Increase Memory to at least 8GB
3. Click "Apply & Restart"

**Linux:**
```bash
# Check current memory
free -h

# Containers use cgroups limits; ensure system has 8GB+ available
```

---

### Docker Compose version issues

**Symptom:** Syntax errors in docker-compose files

**Cause:** Old docker-compose version (v1) instead of Docker Compose V2

**Solution:**
```bash
# Check version
docker compose version

# Should show Docker Compose version v2.x.x
# If using old 'docker-compose' command, upgrade Docker Desktop
```

---

## Infrastructure Issues

### MinIO connection refused

**Symptom:** "Connection refused" when accessing MinIO

**Cause:** MinIO container not running or wrong port

**Solution:**
```bash
# 1. Check MinIO is running
docker ps | grep minio

# 2. Check health endpoint
curl http://localhost:9000/minio/health/live

# 3. If not running, start infrastructure
make start-infra

# 4. Check logs for errors
docker logs inference-arena-minio
```

---

### Prometheus connection failed

**Symptom:** "Cannot connect to Prometheus" during experiments

**Cause:** Prometheus container not running

**Solution:**
```bash
# 1. Verify Prometheus is running
docker ps | grep prometheus

# 2. Test Prometheus API
curl http://localhost:9090/api/v1/query?query=up

# 3. Check if scrape targets are up
# Visit: http://localhost:9090/targets

# 4. Restart if needed
docker restart inference-arena-prometheus
```

---

### Grafana won't accept login

**Symptom:** Login fails with default credentials

**Cause:** Password was changed or Grafana data corrupted

**Solution:**
```bash
# 1. Check credentials in .env (or use defaults)
# Default: admin / admin

# 2. If changed, check .env file
grep GRAFANA .env

# 3. Reset Grafana (WARNING: deletes dashboards)
docker compose -f infrastructure/docker-compose.infra.yml down -v
make start-infra
```

---

### OTel Collector not exporting metrics

**Symptom:** Container metrics missing in Prometheus/Grafana

**Cause:** OTel Collector not running or misconfigured

**Solution:**
```bash
# 1. Check OTel Collector is running
docker ps | grep otel

# 2. Check health endpoint
curl http://localhost:13133/

# 3. Check metrics endpoint
curl http://localhost:8889/metrics

# 4. View logs for errors
docker logs inference-arena-otel-collector
```

---

## Model Issues

### Model not found in MinIO

**Symptom:** "Model not found" or "Object does not exist" errors

**Cause:** Models not exported/uploaded to MinIO

**Solution:**
```bash
# 1. Run full model setup
make models-setup

# 2. Verify models exist
make models-verify

# 3. Check MinIO console manually
# Visit: http://localhost:9001
# Login: minioadmin / minioadmin
# Browse "models" bucket
```

---

### ONNX export failures

**Symptom:** Model export fails with ONNX errors

**Cause:** PyTorch/ONNX version mismatch or missing dependencies

**Solution:**
```bash
# 1. Ensure all dependencies installed
make install

# 2. Check PyTorch and ONNX versions
python -c "import torch; print(torch.__version__)"
python -c "import onnx; print(onnx.__version__)"

# 3. Try export with verbose output
python scripts/models/export_yolo_onnx.py -v
```

See also: [ONNX_UPGRADE.md](ONNX_UPGRADE.md) for ONNX opset compatibility.

---

### Model verification fails

**Symptom:** `make models-verify` shows missing or corrupted models

**Cause:** Incomplete upload or storage corruption

**Solution:**
```bash
# 1. Re-upload models
make models-setup

# 2. If still failing, clear and retry
# Access MinIO console, delete models bucket
# Then: make models-setup
```

---

## Testing Issues

### Coverage below threshold

**Symptom:** `make test` fails with "FAIL Required test coverage of 80% not reached"

**Cause:** New code without tests, or excluded files changed

**Solution:**
```bash
# 1. Check current coverage
make test

# 2. View detailed coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# 3. Add tests for uncovered code, or update pyproject.toml exclusions
```

---

### Health check timeout

**Symptom:** Tests fail waiting for services to become healthy

**Cause:** Services slow to start or actually unhealthy

**Solution:**
```bash
# 1. Check service health manually
curl http://localhost:8100/health  # Monolithic
curl http://localhost:8200/health  # Microservices
curl http://localhost:8300/health  # Triton

# 2. Increase timeout (environment variable)
export HEALTH_CHECK_TIMEOUT=120

# 3. Check container logs for startup errors
docker logs inference-arena-monolithic
```

---

### No test images found

**Symptom:** "No test images found in data/thesis_test_set"

**Cause:** Test dataset not downloaded

**Solution:**
```bash
# Download and curate test set
make data-download

# Verify data exists
make data-verify

# Check directory
ls -la data/thesis_test_set/
```

---

### Tests hang indefinitely

**Symptom:** Tests don't complete, appear stuck

**Cause:** Network issues, deadlocks, or resource exhaustion

**Solution:**
```bash
# 1. Run with timeout
pytest --timeout=60

# 2. Run specific test file to isolate
pytest tests/unit/test_config.py -v

# 3. Check for Docker resource issues
docker stats
```

---

## Environment Issues

### Docker Compose doesn't read .env

**Symptom:** Environment variables not being applied

**Cause:** `.env` file in wrong location or wrong format

**Solution:**
```bash
# 1. Ensure .env is in project root (same directory as Makefile)
ls -la .env

# 2. Check file format (no spaces around =)
cat .env | head -5
# Good: MINIO_API_PORT=9000
# Bad:  MINIO_API_PORT = 9000

# 3. Preview what Docker Compose sees
docker compose -f infrastructure/docker-compose.infra.yml config
```

---

### Changes to .env not taking effect

**Symptom:** Updated .env values not reflected in running containers

**Cause:** Containers use values from when they started

**Solution:**
```bash
# Restart containers to pick up new values
docker compose -f infrastructure/docker-compose.infra.yml down
docker compose -f infrastructure/docker-compose.infra.yml up -d
```

---

### .env accidentally committed

**Symptom:** Git shows .env file in commits

**Cause:** .env was added before .gitignore entry

**Solution:**
```bash
# 1. Remove from git tracking (keeps local file)
git rm --cached .env
git commit -m "Remove .env from tracking"

# 2. Verify .gitignore includes .env
grep "^\.env$" .gitignore

# 3. ROTATE ALL PASSWORDS that were exposed!
```

---

### ENVIRONMENT variable not working

**Symptom:** Production mode not triggering despite ENVIRONMENT=production

**Cause:** Value must be exactly "production" (lowercase, no quotes in shell)

**Solution:**
```bash
# Correct:
export ENVIRONMENT=production

# Incorrect:
export ENVIRONMENT="production"  # May include quotes
export ENVIRONMENT=PRODUCTION    # Wrong case
export ENVIRONMENT=prod          # Wrong value
```

---

## gRPC Issues

### gRPC connection errors

**Symptom:** "failed to connect to all addresses" or similar gRPC errors

**Cause:** Classification service not running or unreachable

**Solution:**
```bash
# 1. Check both microservices containers
docker ps | grep inference-arena-detection
docker ps | grep inference-arena-classification

# 2. Check logs for gRPC errors
docker logs inference-arena-detection
docker logs inference-arena-classification

# 3. Verify gRPC port is correct
# Classification listens on 8201 internally
```

---

### Timeout errors during inference

**Symptom:** gRPC deadline exceeded

**Cause:** Classification service overloaded or network issues

**Solution:**
```bash
# 1. Check classification service health
docker logs inference-arena-classification --tail=50

# 2. Restart services
make stop-micro && make start-micro

# 3. Reduce concurrent load if testing
```

---

### Proto compilation errors

**Symptom:** "Import not found" or protobuf errors

**Cause:** Proto files not compiled or out of sync

**Solution:**
```bash
# Regenerate proto files
python scripts/setup/generate_proto.py

# Verify generated files
ls -la src/shared/proto/
```

---

## Performance Issues

### Experiments run slower than expected

**Symptom:** Full matrix takes much longer than ~4.7 hours

**Cause:** Resource contention or insufficient allocation

**Solution:**
```bash
# 1. Check Docker resource allocation
docker stats

# 2. Ensure Docker Desktop has enough resources
# Recommended: 8GB RAM, 4 CPUs

# 3. Close other resource-intensive applications

# 4. Run experiments on dedicated machine if possible
```

---

### High error rate during load tests

**Symptom:** Many failed requests during experiments

**Cause:** Services overwhelmed, resource exhaustion

**Solution:**
```bash
# 1. Check container logs during test
docker logs -f inference-arena-monolithic

# 2. Monitor resource usage
docker stats

# 3. Reduce concurrent users for debugging
python -m experiments.runner -a monolithic -u 1 -u 5 -r 1
```

---

## Getting More Help

If none of these solutions work:

1. **Check logs thoroughly:**
   ```bash
   docker compose -f infrastructure/docker-compose.infra.yml logs
   ```

2. **Search existing issues** in the project repository

3. **Collect diagnostic info:**
   ```bash
   docker version
   docker compose version
   python --version
   uv --version
   ```

---

## See Also

- **[SETUP.md](SETUP.md)** - Initial setup instructions
- **[ENVIRONMENT.md](ENVIRONMENT.md)** - Environment configuration
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Load testing framework

---

*Troubleshooting guide for Inference Arena v2.0*
