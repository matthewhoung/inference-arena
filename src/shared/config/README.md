# Config Package

Python interface to `experiment.yaml` - the single source of truth for all experimental parameters.

## Structure

| File | Purpose |
|------|---------|
| `loader.py` | Configuration loading, caching, and accessor functions |
| `models.py` | Pydantic models for typed configuration (ServicePorts) |
| `validators.py` | Configuration validation logic |
| `__init__.py` | Public API re-exports |

## Usage

All public functions are re-exported from the package root:

```python
from shared.config import get_config, get_controlled_variable, ServicePorts
```

## Key Functions

- `get_config()` - Load and cache full configuration
- `reload_config()` - Force reload (clears cache)
- `get_controlled_variable(section, key)` - Get specific variable
- `get_model_config(name)` - Get model configuration
- `get_service_ports()` - Get validated service ports
- `validate_config()` - Validate configuration integrity

## Architecture

```
shared/config/
    __init__.py      # Re-exports all public API
    loader.py        # ~400 lines - config loading/access
    models.py        # ~100 lines - Pydantic models
    validators.py    # ~80 lines - validation logic
    README.md        # This file
```

Split from original 708-line `config.py` for maintainability.
