"""
Unit Tests for Proto Module

This module tests:
- Proto file existence and validity
- Message serialization/deserialization
- Service stub creation
- Type validation

Test Categories:
- Proto structure tests (no generation required)
- Generation tests (requires grpcio-tools)
- Message tests (requires generated files)
- Service tests (requires generated files)

Author: Matthew Hong
Specification Reference: Foundation Specification §6 gRPC Interface
"""

import tempfile
from pathlib import Path

import pytest

from shared.proto import (
    PROTO_FILE,
    get_proto_path,
    is_generated,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def proto_content() -> str:
    """Read actual proto file content."""
    return PROTO_FILE.read_text()


# =============================================================================
# Tests for Proto File Structure
# =============================================================================


class TestProtoFile:
    """Tests for inference.proto file structure."""

    def test_proto_file_exists(self) -> None:
        """Proto file should exist."""
        assert PROTO_FILE.exists()

    def test_proto_file_readable(self) -> None:
        """Proto file should be readable."""
        content = PROTO_FILE.read_text()
        assert len(content) > 0

    def test_proto_syntax_version(self, proto_content: str) -> None:
        """Proto should use syntax proto3."""
        assert 'syntax = "proto3"' in proto_content

    def test_proto_package_name(self, proto_content: str) -> None:
        """Proto should have package name."""
        assert "package inference" in proto_content

    def test_get_proto_path(self) -> None:
        """get_proto_path should return correct path."""
        path = get_proto_path()
        assert path == PROTO_FILE
        assert path.name == "inference.proto"


class TestProtoMessages:
    """Tests for message definitions in proto file."""

    def test_has_classification_request(self, proto_content: str) -> None:
        """Proto should define ClassificationRequest."""
        assert "message ClassificationRequest" in proto_content

    def test_has_classification_response(self, proto_content: str) -> None:
        """Proto should define ClassificationResponse."""
        assert "message ClassificationResponse" in proto_content

    def test_has_bounding_box(self, proto_content: str) -> None:
        """Proto should define BoundingBox."""
        assert "message BoundingBox" in proto_content

    def test_has_classification_result(self, proto_content: str) -> None:
        """Proto should define ClassificationResult."""
        assert "message ClassificationResult" in proto_content

    def test_has_timing_info(self, proto_content: str) -> None:
        """Proto should define TimingInfo."""
        assert "message TimingInfo" in proto_content

    def test_has_inference_request(self, proto_content: str) -> None:
        """Proto should define InferenceRequest."""
        assert "message InferenceRequest" in proto_content

    def test_has_inference_response(self, proto_content: str) -> None:
        """Proto should define InferenceResponse."""
        assert "message InferenceResponse" in proto_content

    def test_has_health_check(self, proto_content: str) -> None:
        """Proto should define HealthCheckRequest/Response."""
        assert "message HealthCheckRequest" in proto_content
        assert "message HealthCheckResponse" in proto_content


class TestProtoServices:
    """Tests for service definitions in proto file."""

    def test_has_classification_service(self, proto_content: str) -> None:
        """Proto should define ClassificationService."""
        assert "service ClassificationService" in proto_content

    def test_has_inference_service(self, proto_content: str) -> None:
        """Proto should define InferenceService."""
        assert "service InferenceService" in proto_content

    def test_has_health_service(self, proto_content: str) -> None:
        """Proto should define Health service."""
        assert "service Health" in proto_content

    def test_classification_service_has_classify(self, proto_content: str) -> None:
        """ClassificationService should have Classify RPC."""
        assert "rpc Classify(ClassificationRequest)" in proto_content

    def test_classification_service_has_batch(self, proto_content: str) -> None:
        """ClassificationService should have ClassifyBatch RPC."""
        assert "rpc ClassifyBatch(BatchClassificationRequest)" in proto_content

    def test_inference_service_has_infer(self, proto_content: str) -> None:
        """InferenceService should have Infer RPC."""
        assert "rpc Infer(InferenceRequest)" in proto_content


class TestProtoFields:
    """Tests for specific field definitions."""

    def test_bounding_box_has_coordinates(self, proto_content: str) -> None:
        """BoundingBox should have x1, y1, x2, y2 fields."""
        assert "float x1 = 1" in proto_content
        assert "float y1 = 2" in proto_content
        assert "float x2 = 3" in proto_content
        assert "float y2 = 4" in proto_content

    def test_bounding_box_has_confidence(self, proto_content: str) -> None:
        """BoundingBox should have confidence field."""
        assert "float confidence = 5" in proto_content

    def test_classification_request_has_image_crop(self, proto_content: str) -> None:
        """ClassificationRequest should have image_crop field."""
        assert "bytes image_crop" in proto_content

    def test_classification_request_has_request_id(self, proto_content: str) -> None:
        """ClassificationRequest should have request_id field."""
        assert "string request_id" in proto_content

    def test_timing_info_has_fields(self, proto_content: str) -> None:
        """TimingInfo should have latency fields."""
        assert "preprocessing_ms" in proto_content
        assert "inference_ms" in proto_content
        assert "total_ms" in proto_content


# =============================================================================
# Tests for Generated Files (requires grpcio-tools)
# =============================================================================


class TestProtoGeneration:
    """Tests for proto file generation."""

    def test_is_generated_returns_bool(self) -> None:
        """is_generated should return boolean."""
        result = is_generated()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not is_generated(),
        reason="Proto files not generated - run 'python scripts/generate_proto.py'",
    )
    def test_import_pb2(self) -> None:
        """Should be able to import inference_pb2."""
        from shared.proto import inference_pb2

        assert inference_pb2 is not None

    @pytest.mark.skipif(not is_generated(), reason="Proto files not generated")
    def test_import_pb2_grpc(self) -> None:
        """Should be able to import inference_pb2_grpc."""
        from shared.proto import inference_pb2_grpc

        assert inference_pb2_grpc is not None


# =============================================================================
# Tests for Message Classes (requires generated files)
# =============================================================================


@pytest.mark.skipif(
    not is_generated(), reason="Proto files not generated - run 'python scripts/generate_proto.py'"
)
class TestMessageClasses:
    """Tests for generated message classes."""

    def test_create_classification_request(self) -> None:
        """Should create ClassificationRequest message."""
        from shared.proto import inference_pb2

        request = inference_pb2.ClassificationRequest(
            request_id="test-123",
            image_crop=b"fake-image-data",
        )

        assert request.request_id == "test-123"
        assert request.image_crop == b"fake-image-data"

    def test_create_classification_response(self) -> None:
        """Should create ClassificationResponse message."""
        from shared.proto import inference_pb2

        response = inference_pb2.ClassificationResponse(
            request_id="test-123",
            error="",
        )

        assert response.request_id == "test-123"
        assert response.error == ""

    def test_create_bounding_box(self) -> None:
        """Should create BoundingBox message."""
        from shared.proto import inference_pb2

        box = inference_pb2.BoundingBox(
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=200.0,
            confidence=0.95,
            class_id=0,
        )

        assert box.x1 == 100.0
        assert box.confidence == pytest.approx(0.95)

    def test_create_classification_result(self) -> None:
        """Should create ClassificationResult message."""
        from shared.proto import inference_pb2

        result = inference_pb2.ClassificationResult(
            class_id=281,
            class_name="tabby cat",
            confidence=0.87,
        )

        assert result.class_id == 281
        assert result.class_name == "tabby cat"

    def test_create_timing_info(self) -> None:
        """Should create TimingInfo message."""
        from shared.proto import inference_pb2

        timing = inference_pb2.TimingInfo(
            preprocessing_ms=5.2,
            inference_ms=12.3,
            postprocessing_ms=1.1,
            total_ms=18.6,
        )

        assert timing.preprocessing_ms == 5.2
        assert timing.total_ms == 18.6

    def test_nested_message(self) -> None:
        """Should create nested messages."""
        from shared.proto import inference_pb2

        result = inference_pb2.ClassificationResult(
            class_id=281,
            class_name="tabby cat",
            confidence=0.87,
        )

        timing = inference_pb2.TimingInfo(
            total_ms=18.6,
        )

        response = inference_pb2.ClassificationResponse(
            request_id="test-123",
            result=result,
            timing=timing,
        )

        assert response.result.class_id == 281
        assert response.timing.total_ms == 18.6


# =============================================================================
# Tests for Serialization (requires generated files)
# =============================================================================


@pytest.mark.skipif(not is_generated(), reason="Proto files not generated")
class TestSerialization:
    """Tests for message serialization/deserialization."""

    def test_serialize_classification_request(self) -> None:
        """ClassificationRequest should serialize to bytes."""
        from shared.proto import inference_pb2

        request = inference_pb2.ClassificationRequest(
            request_id="test-123",
            image_crop=b"fake-image-data",
        )

        serialized = request.SerializeToString()

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_deserialize_classification_request(self) -> None:
        """ClassificationRequest should deserialize from bytes."""
        from shared.proto import inference_pb2

        original = inference_pb2.ClassificationRequest(
            request_id="test-456",
            image_crop=b"test-crop-data",
        )

        serialized = original.SerializeToString()

        restored = inference_pb2.ClassificationRequest()
        restored.ParseFromString(serialized)

        assert restored.request_id == "test-456"
        assert restored.image_crop == b"test-crop-data"

    def test_roundtrip_with_nested(self) -> None:
        """Complex message should roundtrip correctly."""
        from shared.proto import inference_pb2

        original = inference_pb2.ClassificationResponse(
            request_id="roundtrip-test",
            result=inference_pb2.ClassificationResult(
                class_id=999,
                class_name="test class",
                confidence=0.99,
            ),
            timing=inference_pb2.TimingInfo(
                preprocessing_ms=1.0,
                inference_ms=2.0,
                postprocessing_ms=0.5,
                total_ms=3.5,
            ),
        )

        serialized = original.SerializeToString()

        restored = inference_pb2.ClassificationResponse()
        restored.ParseFromString(serialized)

        assert restored.request_id == "roundtrip-test"
        assert restored.result.class_id == 999
        assert restored.result.confidence == pytest.approx(0.99)
        assert restored.timing.total_ms == pytest.approx(3.5)

    def test_empty_message_serialization(self) -> None:
        """Empty message should serialize."""
        from shared.proto import inference_pb2

        request = inference_pb2.ClassificationRequest()
        serialized = request.SerializeToString()

        assert isinstance(serialized, bytes)


# =============================================================================
# Tests for Service Stubs (requires generated files)
# =============================================================================


@pytest.mark.skipif(not is_generated(), reason="Proto files not generated")
class TestServiceStubs:
    """Tests for generated service stubs."""

    def test_classification_service_stub_exists(self) -> None:
        """ClassificationServiceStub should exist."""
        from shared.proto import inference_pb2_grpc

        assert hasattr(inference_pb2_grpc, "ClassificationServiceStub")

    def test_classification_service_servicer_exists(self) -> None:
        """ClassificationServiceServicer should exist."""
        from shared.proto import inference_pb2_grpc

        assert hasattr(inference_pb2_grpc, "ClassificationServiceServicer")

    def test_inference_service_stub_exists(self) -> None:
        """InferenceServiceStub should exist."""
        from shared.proto import inference_pb2_grpc

        assert hasattr(inference_pb2_grpc, "InferenceServiceStub")

    def test_health_stub_exists(self) -> None:
        """HealthStub should exist."""
        from shared.proto import inference_pb2_grpc

        assert hasattr(inference_pb2_grpc, "HealthStub")

    def test_add_servicer_function_exists(self) -> None:
        """add_*Servicer_to_server functions should exist."""
        from shared.proto import inference_pb2_grpc

        assert hasattr(inference_pb2_grpc, "add_ClassificationServiceServicer_to_server")
        assert hasattr(inference_pb2_grpc, "add_InferenceServiceServicer_to_server")


# =============================================================================
# Tests for Batch Messages (requires generated files)
# =============================================================================


@pytest.mark.skipif(not is_generated(), reason="Proto files not generated")
class TestBatchMessages:
    """Tests for batch message types."""

    def test_batch_classification_request(self) -> None:
        """Should create BatchClassificationRequest with multiple items."""
        from shared.proto import inference_pb2

        req1 = inference_pb2.ClassificationRequest(request_id="1")
        req2 = inference_pb2.ClassificationRequest(request_id="2")

        batch = inference_pb2.BatchClassificationRequest()
        batch.requests.append(req1)
        batch.requests.append(req2)

        assert len(batch.requests) == 2
        assert batch.requests[0].request_id == "1"
        assert batch.requests[1].request_id == "2"

    def test_batch_classification_response(self) -> None:
        """Should create BatchClassificationResponse with multiple items."""
        from shared.proto import inference_pb2

        resp1 = inference_pb2.ClassificationResponse(request_id="1")
        resp2 = inference_pb2.ClassificationResponse(request_id="2")

        batch = inference_pb2.BatchClassificationResponse()
        batch.responses.extend([resp1, resp2])

        assert len(batch.responses) == 2
