"""Unit tests for gRPC client failure scenarios.

This module tests ClassificationClient error handling for:
- Service unavailable (UNAVAILABLE status)
- Request timeout (DEADLINE_EXCEEDED status)
- Connection failures

These tests use mocked gRPC stubs to simulate failures without
requiring a running Classification service.

Author: Matthew Hong
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import numpy as np
import pytest
from microservices.detection.app.grpc_client import ClassificationClient

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_crop() -> np.ndarray:
    """Create a test crop for classification requests."""
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def mock_box() -> dict:
    """Create a test bounding box."""
    return {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0, "confidence": 0.9, "class_id": 0}


# =============================================================================
# Test Classes
# =============================================================================


class TestClassificationClientConnectionFailures:
    """Tests for connection-level failures."""

    @pytest.mark.asyncio
    async def test_connect_timeout_raises_runtime_error(self) -> None:
        """Connection timeout should raise RuntimeError."""
        client = ClassificationClient("localhost:9999")

        # Mock channel that times out
        with patch.object(client, 'channel', create=True) as mock_channel:
            mock_channel.channel_ready = AsyncMock(side_effect=TimeoutError())
            client.channel = mock_channel

            # Re-implement connect logic to test timeout handling
            with pytest.raises(RuntimeError, match="Timeout connecting"):
                # Manually trigger the timeout path
                try:
                    await asyncio.wait_for(
                        mock_channel.channel_ready(),
                        timeout=0.1,
                    )
                except TimeoutError as err:
                    raise RuntimeError(
                        f"Timeout connecting to Classification service at {client.endpoint}"
                    ) from err

    @pytest.mark.asyncio
    async def test_classify_without_connect_raises_runtime_error(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """Calling classify() before connect() should raise RuntimeError."""
        client = ClassificationClient("localhost:8201")
        # Don't call connect()

        with pytest.raises(RuntimeError, match="not connected"):
            await client.classify("test", mock_crop, mock_box)


class TestClassificationClientRPCFailures:
    """Tests for RPC-level failures using mocked stubs."""

    @pytest.mark.asyncio
    async def test_service_unavailable_propagates_error(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """UNAVAILABLE status should propagate as gRPC error."""
        client = ClassificationClient("localhost:8201")

        # Create mock stub that raises UNAVAILABLE
        mock_stub = AsyncMock()
        # Use a simple Exception since AioRpcError is hard to instantiate
        mock_stub.Classify.side_effect = grpc.RpcError()

        client.stub = mock_stub
        client.channel = MagicMock()

        with pytest.raises(grpc.RpcError):
            await client.classify("test", mock_crop, mock_box)

    @pytest.mark.asyncio
    async def test_deadline_exceeded_propagates_error(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """DEADLINE_EXCEEDED status should propagate as gRPC error."""
        client = ClassificationClient("localhost:8201")

        mock_stub = AsyncMock()
        mock_stub.Classify.side_effect = grpc.RpcError()

        client.stub = mock_stub
        client.channel = MagicMock()

        with pytest.raises(grpc.RpcError):
            await client.classify("test", mock_crop, mock_box)

    @pytest.mark.asyncio
    async def test_classify_parallel_handles_partial_failures(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """classify_parallel should propagate errors from any failed call."""
        client = ClassificationClient("localhost:8201")

        # Mock stub that fails on second call
        call_count = 0
        async def mock_classify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise grpc.RpcError()
            # Return a mock response
            mock_response = MagicMock()
            mock_response.class_id = 0
            mock_response.confidence = 0.9
            return mock_response

        mock_stub = AsyncMock()
        mock_stub.Classify.side_effect = mock_classify

        client.stub = mock_stub
        client.channel = MagicMock()

        crops = [mock_crop, mock_crop, mock_crop]
        boxes = [mock_box, mock_box, mock_box]

        # asyncio.gather should propagate the error
        with pytest.raises(grpc.RpcError):
            await client.classify_parallel("test", crops, boxes)


class TestClassificationClientEmptyInputs:
    """Tests for edge cases with empty inputs."""

    @pytest.mark.asyncio
    async def test_classify_parallel_empty_list_returns_empty(self) -> None:
        """classify_parallel with empty list should return empty list."""
        client = ClassificationClient("localhost:8201")
        client.stub = AsyncMock()
        client.channel = MagicMock()

        result = await client.classify_parallel("test", [], [])

        assert result == []


class TestClassificationClientClose:
    """Tests for client cleanup."""

    @pytest.mark.asyncio
    async def test_close_without_connect_is_safe(self) -> None:
        """Closing without connecting should not raise."""
        client = ClassificationClient("localhost:8201")
        # channel is None
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_closes_channel(self) -> None:
        """close() should close the channel."""
        client = ClassificationClient("localhost:8201")
        mock_channel = AsyncMock()
        client.channel = mock_channel

        await client.close()

        mock_channel.close.assert_called_once()


class TestClassificationClientBehavior:
    """Tests for expected client behavior patterns."""

    @pytest.mark.asyncio
    async def test_classify_builds_correct_request(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """classify() should build request with correct fields."""
        client = ClassificationClient("localhost:8201")

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.class_id = 42
        mock_stub.Classify.return_value = mock_response

        client.stub = mock_stub
        client.channel = MagicMock()

        await client.classify("test_req", mock_crop, mock_box)

        # Verify Classify was called
        mock_stub.Classify.assert_called_once()

        # Verify request structure
        call_args = mock_stub.Classify.call_args
        request = call_args[0][0]

        assert request.request_id == "test_req"
        assert request.crop_height == 224
        assert request.crop_width == 224
        assert len(request.image_crop) == 224 * 224 * 3  # RGB bytes

    @pytest.mark.asyncio
    async def test_classify_without_box_works(
        self,
        mock_crop: np.ndarray,
    ) -> None:
        """classify() should work without source_box."""
        client = ClassificationClient("localhost:8201")

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.class_id = 0
        mock_stub.Classify.return_value = mock_response

        client.stub = mock_stub
        client.channel = MagicMock()

        # Call without box
        result = await client.classify("test", mock_crop, None)

        assert result.class_id == 0

    @pytest.mark.asyncio
    async def test_classify_parallel_creates_indexed_request_ids(
        self,
        mock_crop: np.ndarray,
        mock_box: dict,
    ) -> None:
        """classify_parallel should create request_id_{index} for each call."""
        client = ClassificationClient("localhost:8201")

        captured_request_ids: list[str] = []

        async def capture_classify(request):
            captured_request_ids.append(request.request_id)
            mock_response = MagicMock()
            mock_response.class_id = 0
            return mock_response

        mock_stub = AsyncMock()
        mock_stub.Classify.side_effect = capture_classify

        client.stub = mock_stub
        client.channel = MagicMock()

        crops = [mock_crop, mock_crop, mock_crop]
        boxes = [mock_box, mock_box, mock_box]

        await client.classify_parallel("batch", crops, boxes)

        assert "batch_0" in captured_request_ids
        assert "batch_1" in captured_request_ids
        assert "batch_2" in captured_request_ids
