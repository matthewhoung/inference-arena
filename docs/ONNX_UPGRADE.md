# ONNX Version Constraints

## Summary

The ONNX and ONNXRuntime versions are pinned to ensure exported models remain compatible with NVIDIA Triton Inference Server 24.08. These constraints prevent IR version incompatibilities that would cause model loading failures.

## Why Pinned

ONNX models contain an IR (Intermediate Representation) version that indicates the model format version. Triton Inference Server 24.08 supports IR versions up to version 10. The version constraints are:

- **onnx**: Versions 1.15.0 through 1.16.x produce IR version 10 or below. Version 1.17 and above produce IR version 11, which Triton 24.08 cannot load.

- **onnxruntime**: Versions 1.18.0 through 1.22.x work correctly. Version 1.23 and above contain a regression that increases IR version unexpectedly.

- **ml_dtypes**: Version 0.5.0 or above is required to provide float8_e8m0fnu support used by PyTorch during ONNX export.

Current constraints in pyproject.toml:
- onnx>=1.15.0,<1.17.0
- onnxruntime>=1.18.0,<1.23.0
- ml_dtypes>=0.5.0

## What Breaks

If the version constraints are violated:

1. **Models fail to load in Triton** with error messages indicating unsupported IR version or opset version incompatibility.

2. **Export succeeds but inference fails** because the model appears valid but uses features Triton 24.08 does not support.

3. **Inconsistent behavior between architectures** where monolithic and microservices work (using direct ONNXRuntime) but Triton architecture fails.

4. **Silent accuracy issues** if an incompatible IR version causes different operator implementations to be selected.

## Upgrade Steps

When Triton Inference Server releases a version supporting IR version 11:

1. Check the Triton release notes to confirm IR version 11 support.

2. Update pyproject.toml to relax the onnx constraint to allow version 1.17 or above.

3. Re-export all models to regenerate ONNX files with the new version.

4. Verify model checksums have changed (indicating new format).

5. Run the full test suite against all three architectures to confirm compatibility.

6. Update this document to reflect the new version requirements.

Until Triton adds IR version 11 support, maintain the current version constraints to ensure consistent behavior across all architectures.
