# Environment Configuration Guide

This guide explains how to configure the Inference Arena project using environment variables for secure, reproducible deployments across all platforms (Windows, macOS, Linux).

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
- [Environment Variables Reference](#environment-variables-reference)
- [Security Best Practices](#security-best-practices)
- [Docker Compose Integration](#docker-compose-integration)
- [Relationship with experiment.yaml](#relationship-with-experimentyaml)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option A: Automated Setup (Recommended)

Run the cross-platform Python setup script:

```bash
python scripts/setup_env.py
```

The script will:
- Guide you through configuration options (Development / Production / Custom)
- Generate cryptographically secure passwords (Production mode)
- Create `.env` from `.env.example`
- Verify `.gitignore` excludes secrets
- Work on Windows, macOS, and Linux

**For thesis development:** Choose option 1 (Development) for quick local testing.

### Option B: Manual Setup

1. **Copy the template:**
   - Copy `.env.example` to `.env` in the project root
   - Use your file manager or command line

2. **Edit `.env`:**
   - Open in any text editor (VS Code, Notepad++, nano, vim)
   - Update passwords and ports as needed

3. **Verify:**
   ```bash
   # Check .env is not tracked by git
   git status .env
   # Should show "Untracked" or nothing

   # Test Docker Compose reads your config
   docker compose -f infrastructure/docker-compose.infra.yml config
   ```

---

## Configuration Files

### `.env.example` ✅ Committed to Git
- **Purpose:** Template showing all available configuration options
- **Location:** Project root
- **Contents:** Safe default values
- **Audience:** Public (shared with committee, collaborators)

### `.env` ❌ Never Committed
- **Purpose:** Your actual configuration with secrets
- **Location:** Project root (same directory as `.env.example`)
- **Contents:** Passwords, custom ports, local settings
- **Status:** Listed in `.gitignore` to prevent accidental commits

### `experiment.yaml` ✅ Committed to Git
- **Purpose:** Single source of truth for experimental parameters
- **Location:** Project root
- **Contents:** Scientific configuration (models, preprocessing, hypotheses)
- **Why separate:** Scientific reproducibility requires version control; secrets don't

---

## Environment Variables Reference

### MinIO (Object Storage)

Used to store ONNX model files accessed by all three architectures.

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ROOT_USER` | `minioadmin` | Admin username |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | Admin password ⚠️ Change for production |
| `MINIO_API_PORT` | `9000` | S3 API port |
| `MINIO_CONSOLE_PORT` | `9001` | Web console port |
| `MINIO_BUCKET` | `models` | Default bucket name |

**Access:** http://localhost:9001 (login with credentials above)

### Grafana (Monitoring Dashboard)

Visualizes container metrics during experiments.

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFANA_ADMIN_USER` | `admin` | Dashboard admin username |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Dashboard password ⚠️ Change for production |
| `GRAFANA_PORT` | `3000` | Web interface port |

**Access:** http://localhost:3000 (login with credentials above)

### Prometheus (Metrics Database)

Collects time-series metrics from containers.

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_PORT` | `9090` | Web interface port |
| `PROMETHEUS_RETENTION_DAYS` | `15d` | How long to keep metrics |
| `PROMETHEUS_SCRAPE_INTERVAL` | `1s` | Metrics collection frequency |

**Note:** 1-second scrape interval matches thesis latency analysis requirements.

### cAdvisor (Container Metrics)

Exports container CPU, memory, and network metrics to Prometheus.

| Variable | Default | Description |
|----------|---------|-------------|
| `CADVISOR_PORT` | `8080` | Metrics endpoint port |

### Resource Limits

Should match `experiment.yaml` controlled variables for reproducibility.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_VCPU` | `2` | CPU cores per container |
| `CONTAINER_MEMORY` | `4096` | Memory limit (MB) per container |

---

## Security Best Practices

### 1. Never Commit Secrets

✅ **GOOD:**
```bash
git add .env.example          # Safe template
git add experiment.yaml       # Scientific config
```

❌ **BAD:**
```bash
git add .env                  # Contains passwords!
```

The `.env` file is automatically ignored via `.gitignore`.

### 2. Use Strong Passwords for Production

For local development (your laptop):
- Default passwords (`minioadmin` / `admin`) are fine
- Services only accessible from localhost

For shared servers or cloud deployment:
- Use production mode in `setup_env.py` (generates secure passwords)
- Or manually set strong passwords (16+ characters, mixed case, numbers, symbols)

### 3. Rotate Credentials

For thesis work, change passwords:
- After pilot runs (if shared infrastructure)
- Before final data collection
- If credentials are accidentally exposed

### 4. Verify .gitignore

```bash
# Ensure .env is listed in .gitignore
grep "^\.env$" .gitignore
# Should output: .env
```

If missing, the setup script adds it automatically.

---

## Docker Compose Integration

### How It Works

Docker Compose automatically reads `.env` from the project root:

```yaml
# docker-compose.infra.yml
ports:
  - "${MINIO_API_PORT:-9000}:9000"
  #      ^               ^
  #      |               └── Default if not set
  #      └── Reads from .env
```

### Variable Precedence

1. **Shell environment** (highest priority)
   ```bash
   MINIO_API_PORT=9999 docker compose up
   ```

2. **`.env` file** (middle priority)
   ```ini
   MINIO_API_PORT=8888
   ```

3. **Default values in compose file** (lowest priority)
   ```yaml
   ${MINIO_API_PORT:-9000}  # Uses 9000 if not set
   ```

### Testing Your Configuration

Preview the final configuration without starting services:

```bash
docker compose -f infrastructure/docker-compose.infra.yml config
```

This shows the actual values that will be used.

---

## Relationship with experiment.yaml

### Division of Concerns

**`experiment.yaml`** (Scientific Configuration) ✅ Git-tracked
- What models to use (YOLOv5n, MobileNetV2)
- Preprocessing parameters (input sizes, normalization)
- Resource constraints (2 vCPU, 4GB RAM)
- ONNX threading settings
- Load testing protocol
- Pre-registered hypotheses

**`.env`** (Deployment Configuration) ❌ Git-ignored
- Passwords for local services
- Port mappings (for avoiding conflicts)
- Infrastructure endpoints
- Local development overrides

**Key Principle:** `experiment.yaml` defines **WHAT** you're testing (reproducible science), `.env` defines **WHERE** to run it (deployment secrets).

### Example Alignment

```yaml
# experiment.yaml - Scientific
controlled_variables:
  resources:
    vcpu_per_container: 2
    memory_mb: 4096
```

```ini
# .env - Deployment
CONTAINER_VCPU=2
CONTAINER_MEMORY=4096
```

Both should match to ensure reproducibility.

---

## Troubleshooting

### Problem: Docker Compose doesn't read .env

**Cause:** `.env` must be in the same directory where you run `docker compose`.

**Solution:**
```bash
# Check .env location
ls -la .env  # Unix/Mac
dir .env     # Windows

# .env should be in project root, same directory as docker-compose.infra.yml is referenced from
```

### Problem: Changes to .env not taking effect

**Solution:** Restart services to pick up new values:

```bash
docker compose -f infrastructure/docker-compose.infra.yml down
docker compose -f infrastructure/docker-compose.infra.yml up -d
```

For code changes (not just `.env`), rebuild:
```bash
docker compose -f infrastructure/docker-compose.infra.yml up -d --build
```

### Problem: MinIO connection refused

**Check:**
1. Verify MinIO port in `.env`: `grep MINIO_API_PORT .env`
2. Check MinIO is running: `docker ps | grep minio`
3. Test connection: Visit http://localhost:9000/minio/health/live

### Problem: Grafana won't accept login

**Check:**
1. Verify credentials in `.env`: `grep GRAFANA .env`
2. Try default credentials: `admin` / `admin`
3. Reset Grafana data (⚠️ deletes dashboards):
   ```bash
   docker compose -f infrastructure/docker-compose.infra.yml down -v
   docker compose -f infrastructure/docker-compose.infra.yml up -d
   ```

### Problem: .env accidentally committed

**Fix immediately:**

1. Remove from git tracking:
   ```bash
   git rm --cached .env
   git add .gitignore  # Ensure .gitignore includes .env
   git commit -m "Remove .env from git tracking"
   ```

2. **Rotate all passwords** that were exposed!

3. Update `.env.example` if template needs changes (but never commit actual `.env`)

---

## Multiple Environments

### Development vs. Production

**Development (local laptop):**
```bash
python scripts/setup_env.py
# Choose option 1: Development
# Uses default passwords, localhost endpoints
```

**Production (shared server/cloud):**
```bash
python scripts/setup_env.py
# Choose option 2: Production
# Generates secure passwords, saves them for you to store in password manager
```

### Team Collaboration

```ini
# .env.example - Commit this (safe defaults)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# .env - Each team member creates their own (not committed)
MINIO_ROOT_USER=matthew_local
MINIO_ROOT_PASSWORD=my-secure-local-password-12345
```

---

## Checklist for Thesis Defense

- [ ] `.env.example` is committed and up-to-date
- [ ] `.env` is in `.gitignore` (never committed)
- [ ] All default passwords have been changed (if using production mode)
- [ ] Resource limits match `experiment.yaml`
- [ ] Port configuration is documented
- [ ] Can reproduce setup from `.env.example` alone
- [ ] No secrets in git history (`git log -p | grep password` shows nothing from `.env`)

---

## For Thesis Documentation

**Example methodology text:**

> "Infrastructure credentials were managed using environment variables stored in a `.env` file (excluded from version control). The `.env.example` template in the repository shows all configurable parameters with safe defaults. This separation ensures reproducibility while protecting sensitive information."

---

## See Also

- **[SETUP.md](SETUP.md)** - Complete setup guide with step-by-step instructions
- **[experiment.yaml](../experiment.yaml)** - Full experimental specification
- **[.env.example](../.env.example)** - Environment template with all options

---

## Questions?

For issues related to environment configuration:
1. Check this guide's troubleshooting section
2. Verify your `.env` file matches `.env.example` structure
3. Ensure Docker Compose can read `.env` (run `docker compose config`)
