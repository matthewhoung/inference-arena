"""Centralized exception definitions for Inference Arena.

This module defines the exception hierarchy for the Inference Arena project.
All custom exceptions inherit from InferenceArenaError to allow catching
all project-specific exceptions with a single base class.

Usage:
    from shared.exceptions import ConfigError, ConfigNotFoundError

    try:
        config = load_config()
    except ConfigNotFoundError as e:
        print(f"Configuration file missing: {e}")
    except ConfigError as e:
        print(f"Configuration error: {e}")

Author: Matthew Hong
"""


class InferenceArenaError(Exception):
    """Base exception for all Inference Arena errors.

    All project-specific exceptions inherit from this class,
    allowing consumers to catch all project errors with a single
    except clause when desired.
    """

    pass


class ConfigError(InferenceArenaError):
    """Configuration-related errors.

    Base class for all configuration errors. Use specific subclasses
    (ConfigNotFoundError, ConfigParseError, ConfigKeyError) when the
    error type is known.
    """

    pass


class ConfigNotFoundError(ConfigError):
    """Experiment.yaml not found.

    Raised when the experiment configuration file cannot be located
    in any of the expected paths.
    """

    pass


class ConfigParseError(ConfigError):
    """YAML parsing failed.

    Raised when the configuration file exists but contains
    invalid YAML syntax.
    """

    pass


class ConfigKeyError(ConfigError):
    """Required configuration key missing.

    Raised when a required key or section is not present
    in the configuration file.
    """

    pass
