"""Configuration Validation Module.

This module provides validation functions for experiment configuration.

Functions:
    validate_config: Validate the experiment configuration

Author: Matthew Hong
"""

import yaml

from .loader import get_config

# =============================================================================
# Validation
# =============================================================================


def validate_config() -> list[str]:
    """Validate the experiment configuration.

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_config()
        >>> if errors:
        ...     print("Validation failed:", errors)
    """
    errors = []

    try:
        config = get_config()
    except FileNotFoundError as e:
        return [f"Configuration file not found: {e}"]
    except yaml.YAMLError as e:
        return [f"YAML parsing error: {e}"]

    # Check required sections
    required_sections = [
        "metadata",
        "research_questions",
        "hypotheses",
        "independent_variables",
        "controlled_variables",
        "infrastructure",
    ]

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Check controlled variables
    cv = config.get("controlled_variables", {})
    required_cv = [
        "models",
        "preprocessing",
        "resources",
        "onnx_runtime",
        "dataset",
        "load_testing",
    ]

    for section in required_cv:
        if section not in cv:
            errors.append(f"Missing controlled_variables section: {section}")

    # Check models
    models = cv.get("models", {})
    for model_name in ["yolov5n", "mobilenetv2"]:
        if model_name not in models:
            errors.append(f"Missing model configuration: {model_name}")
        else:
            model = models[model_name]
            for field in ["opset_version", "input", "output"]:
                if field not in model:
                    errors.append(f"Model {model_name} missing field: {field}")

    # Check ONNX runtime config
    onnx = cv.get("onnx_runtime", {})
    for field in ["intra_op_num_threads", "inter_op_num_threads"]:
        if field not in onnx:
            errors.append(f"Missing onnx_runtime field: {field}")

    # Check hypotheses
    hypotheses = config.get("hypotheses", {})
    for h_id, h_config in hypotheses.items():
        # Required fields for all hypotheses
        required_fields = ["category", "statement", "rationale"]
        for field in required_fields:
            if field not in h_config:
                errors.append(f"Hypothesis {h_id} missing required field: {field}")

        # Must have either 'testable_prediction' or 'prediction'
        if "testable_prediction" not in h_config and "prediction" not in h_config:
            errors.append(f"Hypothesis {h_id} missing testable_prediction or prediction")

    return errors
