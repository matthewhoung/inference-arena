# Dataset Curator Package

This package provides dataset curation utilities for creating thesis test
datasets from COCO val2017 with controlled detection counts (fan-out).

## Module Structure

```
curator/
  __init__.py   - Public API re-exports
  types.py      - Configuration constants and data classes
  sampling.py   - Detection counting using YOLOv5n ONNX model
  manifest.py   - Dataset curation orchestration
```

## Module Responsibilities

### types.py (~150 lines)
- Configuration constants loaded from experiment.yaml
- Data classes: CurationConfig, ImageRecord, CurationResult, DatasetManifest
- No internal imports (breaks circular dependency chain)

### sampling.py (~230 lines)
- DetectionCounter class for YOLO inference
- Handles YOLOv8-style output format
- Imports from .types only

### manifest.py (~350 lines)
- DatasetCurator class for curation orchestration
- Image scanning, selection, and manifest generation
- Imports from .types and .sampling

## Usage

```python
from shared.data.curator import (
    DatasetCurator,
    CurationConfig,
    DEFAULT_TARGET_COUNT,
)

curator = DatasetCurator(
    data_dir=Path("data/"),
    models_dir=Path("models/"),
    output_dir=Path("data/thesis_test_set/"),
)
result = curator.curate()
```

## Specification Reference

experiment.yaml controlled_variables.dataset
