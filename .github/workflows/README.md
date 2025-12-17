# GitHub Actions CI/CD Workflows

This directory contains automated workflows that ensure code quality, reproducibility, and security for the Inference Arena thesis project.

## Workflows

### 1. `ci.yml` - Main CI Pipeline

**Triggers:** Every push to `main` and all pull requests

**Purpose:** Comprehensive validation of code quality, tests, and configuration

**Jobs:**
- **Code Quality** - Black formatting, Ruff linting, MyPy type checking
- **Tests** - Run full test suite on Python 3.11 and 3.12
- **Config Validation** - Verify experiment.yaml loads and validates
- **Security** - Check for committed secrets and hardcoded passwords
- **Docker Validation** - Ensure docker-compose files are valid
- **Documentation** - Verify all docs exist and links work
- **Single Source of Truth** - Confirm no hardcoded values defeating experiment.yaml

**Status Badge:**
```markdown
![CI](https://github.com/matthewhoung/inference-arena/workflows/CI/badge.svg)
```

### 2. `validate-experiment-config.yml` - Experiment Config Guardian

**Triggers:** When `experiment.yaml` is modified

**Purpose:** Special validation for the single source of truth

**Jobs:**
- **Validate** - Deep validation of experiment.yaml structure
  - YAML syntax
  - Required sections present
  - Model configurations complete
  - Preprocessing parameters valid
  - Hypotheses properly formatted
  - Threading configs consistent (ONNX ↔ Triton)
  - Changelog updated

- **Notify** - Post PR comment summarizing changes (for review)

**Why separate:** experiment.yaml is critical for reproducibility. Changes require extra scrutiny and documentation.

---

## CI Requirements

### For PRs to be Mergeable

All of these must pass:
- ✅ Code formatted with Black
- ✅ No linting errors (Ruff)
- ✅ All tests pass
- ✅ experiment.yaml validates
- ✅ No committed secrets (.env, passwords)
- ✅ Docker Compose configs valid
- ✅ Documentation links not broken

### For experiment.yaml Changes

Additional requirements:
- ✅ All model configs have required fields
- ✅ Preprocessing params match expected structure
- ✅ Hypotheses have null/alternative/prediction
- ✅ Threading configs match (ONNX runtime = Triton)
- 📝 Changelog entry (recommended, not enforced)

---

## Local Development

### Run CI Checks Locally

**Before pushing**, run these locally to catch issues early:

```bash
# 1. Format code
black src/ tests/

# 2. Lint
ruff check src/ tests/

# 3. Type check
mypy src/

# 4. Run tests
pytest tests/ -v

# 5. Validate config
python -c "from shared.config import validate_config; validate_config()"

# 6. Check docker-compose
docker compose -f infrastructure/docker-compose.infra.yml config
```

### Pre-commit Hook (Recommended)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run quick checks before allowing commit

echo "Running pre-commit checks..."

# Format check
black --check src/ tests/ || {
    echo "❌ Code not formatted. Run: black src/ tests/"
    exit 1
}

# Quick lint
ruff check src/ tests/ || {
    echo "❌ Linting errors found"
    exit 1
}

# Run fast tests only
pytest tests/shared/test_config.py -q || {
    echo "❌ Config tests failed"
    exit 1
}

echo "✅ Pre-commit checks passed"
```

Make executable: `chmod +x .git/hooks/pre-commit`

---

## Workflow Best Practices

### For Thesis Work

1. **Never bypass CI**
   - Even small changes go through PR + CI
   - This creates audit trail for thesis committee

2. **Document experiment.yaml changes**
   - Always add changelog entry with rationale
   - Explain why controlled variables changed
   - Reference thesis sections if applicable

3. **Test locally first**
   - Don't use CI as your test runner
   - Push only when local tests pass
   - Saves GitHub Actions minutes

4. **Branch protection recommended**
   - Settings → Branches → Add rule for `main`
   - Require status checks to pass
   - Require PR reviews (if working with advisor)

### For Reproducibility

These workflows ensure:
- ✅ **No silent changes** - All config changes are visible in PRs
- ✅ **No hardcoded values** - CI fails if constants bypass experiment.yaml
- ✅ **No broken state** - Can't merge if tests fail
- ✅ **No missing docs** - README/SETUP/ENVIRONMENT checked
- ✅ **No committed secrets** - .env files blocked

---

## Adding New Workflows

### When to create a new workflow:

- Long-running tests (integration, load testing)
- Scheduled jobs (nightly builds, dependency updates)
- Deployment workflows (if deploying to cloud)
- Release automation (when publishing results)

### Workflow naming convention:

- `ci.yml` - Main continuous integration
- `validate-*.yml` - Validation-specific workflows
- `deploy-*.yml` - Deployment workflows
- `scheduled-*.yml` - Cron-based workflows

---

## Troubleshooting

### "CI failed but works locally"

**Common causes:**
1. **Missing dependency** - Add to `pyproject.toml`
2. **Path issues** - Use absolute paths or `Path(__file__).parent`
3. **Python version** - CI tests 3.11 and 3.12, check both locally

### "experiment.yaml validation failed"

**Check:**
1. YAML syntax: `python -c "import yaml; yaml.safe_load(open('experiment.yaml'))"`
2. Required sections: See `validate-experiment-config.yml` for list
3. Threading consistency: ONNX runtime must match Triton config

### "Security check failed"

**Common issues:**
1. `.env` committed - Run `git rm --cached .env` and add to `.gitignore`
2. Hardcoded password in code - Move to `.env.example` and reference via env var
3. API key in history - Consider BFG Repo Cleaner or new repo

### "Docker validation failed"

**Check:**
1. Syntax: `docker compose -f infrastructure/docker-compose.infra.yml config`
2. Environment variables: Ensure all `${VAR:-default}` syntax is correct
3. Network names: Must match between compose files

---

## GitHub Actions Configuration

### Secrets (if needed later)

For deployment workflows, add secrets in:
- Settings → Secrets and variables → Actions

**Never needed for this project:** All credentials are in `.env` (not committed).

### Caching

Workflows use `cache: 'pip'` for faster runs:
- Python dependencies cached between runs
- Saves ~30 seconds per job
- Automatically invalidated when `pyproject.toml` changes

### Concurrency

Default: Unlimited concurrent jobs
- Multiple PRs can run simultaneously
- Pushes to same branch cancel previous runs (saves minutes)

---

## Metrics

### CI Performance

Typical run times:
- Code Quality: ~45 seconds
- Tests (Python 3.11): ~1 minute
- Tests (Python 3.12): ~1 minute
- Config Validation: ~30 seconds
- Security: ~15 seconds
- Docker Validation: ~20 seconds
- Docs: ~10 seconds
- Total: **~4 minutes** for full CI

### Free Tier Limits

GitHub Actions free tier (public repos):
- Unlimited minutes
- 20 concurrent jobs

**This project well within limits.**

---

## For Thesis Committee

This CI infrastructure demonstrates:

1. **Automated Reproducibility**
   - All experimental parameters validated on every change
   - Single source of truth (experiment.yaml) is protected
   - Tests ensure config is actually used (not hardcoded values)

2. **Software Engineering Best Practices**
   - Code quality checks (formatting, linting, type safety)
   - Comprehensive test coverage
   - Security validation (no committed secrets)
   - Documentation validation

3. **Scientific Rigor**
   - Pre-registered hypotheses (changelog tracks when added)
   - Controlled variables enforced by CI
   - Git history provides audit trail
   - Changes to experiment config require PR review

---

## See Also

- [Main CI Workflow](ci.yml)
- [Experiment Config Validation](validate-experiment-config.yml)
- [Project README](../../README.md)
- [Setup Guide](../../docs/SETUP.md)
