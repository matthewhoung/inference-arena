# ONNX Model Exporter Package

This package exports PyTorch models to ONNX format with controlled parameters to ensure reproducibility and compatibility across all architectures.

## Structure

```
exporter/
  __init__.py       # Package API, export_all_models function
  types.py          # ExportResult dataclass, constants (ONNX_OPSET_VERSION, etc.)
  utils.py          # compute_checksum, verify_onnx_model
  detection.py      # export_yolov5n (YOLOv5n object detection)
  classification.py # export_mobilenetv2 (MobileNetV2 classification)
```

## Usage

```python
from shared.model.exporter import (
    export_yolov5n,
    export_mobilenetv2,
    export_all_models,
    ExportResult,
    ONNX_OPSET_VERSION,
)

# Export single model
result = export_yolov5n(Path("models/yolov5n.onnx"))

# Export all models
results = export_all_models(Path("models/"))
```

## Design Rationale

The exporter was split from a single 520-line file into focused modules:

- **types.py**: No dependencies, defines data structures
- **utils.py**: Depends only on types.py
- **detection.py / classification.py**: Model-specific export logic, isolated for maintainability
- **__init__.py**: Re-exports public API, contains export_all_models orchestration

This organization follows the principle of grouping by model family (detection vs classification) for better maintainability as more model types are added.
