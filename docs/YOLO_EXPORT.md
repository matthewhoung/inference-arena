# YOLOv5 Export Settings

## Summary

YOLOv5 models are exported using the legacy PyTorch ONNX exporter (dynamo=False) to ensure proper support for dynamic axes. This allows the exported model to handle variable batch sizes during inference, which is essential for this project's load testing methodology.

## Why Required

PyTorch provides two ONNX export pathways:

1. **Legacy exporter** (torch.onnx.export with dynamo=False): Fully supports the dynamic_axes parameter, allowing batch dimension flexibility.

2. **New torch.export pathway** (dynamo=True): Uses TorchDynamo for export but may not fully honor the dynamic_axes parameter in all cases.

For inference serving, dynamic batch sizes are required because:

- Load testing sends requests at varying rates, requiring batching flexibility
- Different architectures may batch requests differently
- Triton Inference Server uses dynamic batching to optimize throughput

The legacy exporter reliably creates models where the batch dimension is marked as dynamic, ensuring consistent behavior across all inference scenarios.

## What Breaks

If dynamo=True is used during export:

1. **Fixed batch dimension** in the exported model causes inference failures when actual batch size differs from the export-time batch size.

2. **Triton dynamic batching fails** because the model cannot accept batches larger or smaller than the fixed size.

3. **Load testing produces errors** as concurrent requests result in varied batch sizes that the model rejects.

4. **Architecture comparison invalidated** because Triton uses different batching than monolithic/microservices, creating unfair comparisons.

## Migration Path

The long-term plan is to migrate to the torch.export pathway once it stabilizes:

1. Monitor PyTorch releases for torch.export API stability announcements.

2. When stable, test export with dynamo=True on a development branch.

3. Verify the exported model correctly handles dynamic batch dimension by inspecting ONNX graph inputs.

4. Run inference tests with varying batch sizes (1, 4, 8, 16) to confirm flexibility.

5. Execute the full test suite across all architectures.

6. If all tests pass, update exporter.py to use dynamo=True and update this document.

Until torch.export reliably supports dynamic_axes, continue using dynamo=False to ensure correct batch handling across all deployment scenarios.
