# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Phase 3: Performance - 2026-01-23

#### Breaking Changes

**Module reorganization:** The following modules have been split into packages with subdirectories. The public API is preserved via `__init__.py` re-exports, so most imports should continue to work unchanged.

- `src/shared/config.py` -> `src/shared/config/` (loader.py, models.py, validators.py)
- `src/shared/data/curator.py` -> `src/shared/data/curator/` (types.py, sampling.py, manifest.py)
- `src/shared/model/exporter.py` -> `src/shared/model/exporter/` (types.py, utils.py, detection.py, classification.py)

**Import paths unchanged:** All existing imports like `from shared.config import get_config` continue to work.

#### Added

- **Parallel model downloads** - `init_microservices_models.py` now downloads models concurrently using ThreadPoolExecutor
  - Configurable via `experiment.yaml` `downloads.max_concurrent` (default: 3)
  - Configurable timeout via `downloads.timeout` (default: 300s)
  - Progress bars for each download
  - Fail-fast behavior: first failure cancels remaining downloads
  - Caching: skips already-downloaded files

- **Health check utility** - `src/shared/health.py` provides `wait_for_healthy()` with exponential backoff
  - Configurable initial delay, max wait, backoff multiplier, max interval
  - `HealthCheckTimeoutError` with descriptive messages including service name and last error

- **Config accessor functions** - New functions in `shared.config`:
  - `get_download_max_concurrent()` - returns max parallel downloads
  - `get_download_timeout()` - returns per-download timeout

#### Changed

- `experiments/runner.py` now uses `wait_for_healthy()` instead of fixed 2-second sleep loop for service readiness checks

#### Improved

- Code organization: Large modules split into focused submodules for better maintainability
- Each new package directory includes a README.md explaining its purpose

### Phase 2: Security - 2026-01-23

- Added credential security warnings (InsecureCredentialsError, W003)
- Added ENVIRONMENT production mode documentation

### Phase 1: Tech Debt - 2026-01-23

- Added exception hierarchy (InferenceArenaError, ConfigError, etc.)
- Added warning infrastructure (W001, W002 codes)
- Documented ONNX and YOLOv5 constraints
- Centralized port configuration in experiment.yaml
- Removed sys.path.insert() from test files
