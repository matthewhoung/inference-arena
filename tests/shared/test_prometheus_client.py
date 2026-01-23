"""
Unit Tests for Prometheus Client Module

This module tests:
- PrometheusClient initialization
- Query error handling (URLError, timeouts)
- Empty result handling
- Data extraction from matrix/vector results
- Availability checking
- Container metrics aggregation

Test Categories:
- Init tests (URL configuration)
- Query error tests (mocked URLError, timeouts)
- Empty result tests (graceful degradation)
- Data extraction tests (matrix/vector parsing)
- Availability tests (is_available method)
- Aggregation tests (query_container_metrics)

All tests use mocks - no real network calls.

Author: Matthew Hong
Specification Reference: TEST-02 coverage gap
"""

import socket
from unittest.mock import MagicMock, Mock, patch

import pytest

from experiments.results.prometheus_client import (
    DEFAULT_PROMETHEUS_URL,
    PrometheusClient,
)
from urllib.error import URLError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> PrometheusClient:
    """Create a PrometheusClient with default URL."""
    return PrometheusClient()


@pytest.fixture
def custom_client() -> PrometheusClient:
    """Create a PrometheusClient with custom URL."""
    return PrometheusClient("http://prometheus.local:9090")


@pytest.fixture
def mock_success_response() -> MagicMock:
    """Create a mock successful response."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"status": "success", "data": {"resultType": "vector", "result": []}}'
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    return mock_response


@pytest.fixture
def mock_matrix_response() -> MagicMock:
    """Create a mock matrix (range query) response."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'''{
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"container_name": "test"},
                    "values": [[1000, "50.5"], [1005, "60.0"], [1010, "55.2"]]
                }
            ]
        }
    }'''
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    return mock_response


@pytest.fixture
def mock_vector_response() -> MagicMock:
    """Create a mock vector (instant query) response."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'''{
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"job": "prometheus"}, "value": [1000, "1"]}
            ]
        }
    }'''
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    return mock_response


# =============================================================================
# Tests for PrometheusClient Initialization
# =============================================================================


class TestPrometheusClientInit:
    """Tests for PrometheusClient initialization."""

    def test_default_url(self) -> None:
        """Should use DEFAULT_PROMETHEUS_URL when no URL provided."""
        client = PrometheusClient()
        assert client.url == DEFAULT_PROMETHEUS_URL
        assert client.url == "http://localhost:9090"

    def test_custom_url(self) -> None:
        """Should accept custom URL."""
        client = PrometheusClient("http://prometheus.example.com:9090")
        assert client.url == "http://prometheus.example.com:9090"

    @patch.dict("os.environ", {"PROMETHEUS_URL": "http://env-prom:9090"})
    def test_env_var_url(self) -> None:
        """Should read PROMETHEUS_URL environment variable."""
        client = PrometheusClient()
        assert client.url == "http://env-prom:9090"

    @patch.dict("os.environ", {"PROMETHEUS_URL": "http://env-prom:9090"})
    def test_explicit_url_overrides_env(self) -> None:
        """Explicit URL should override environment variable."""
        client = PrometheusClient("http://explicit:9090")
        assert client.url == "http://explicit:9090"

    def test_timeout_default(self) -> None:
        """Should have default timeout of 30 seconds."""
        client = PrometheusClient()
        assert client._timeout == 30


# =============================================================================
# Tests for Prometheus Query Errors
# =============================================================================


class TestPrometheusQueryErrors:
    """Tests for _query method error handling."""

    @patch("experiments.results.prometheus_client.urlopen")
    def test_connection_refused_raises_runtime_error(
        self, mock_urlopen: Mock
    ) -> None:
        """URLError with connection refused should raise RuntimeError."""
        mock_urlopen.side_effect = URLError("Connection refused")
        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="Failed to query"):
            client._query("up")

    @patch("experiments.results.prometheus_client.urlopen")
    def test_timeout_raises_runtime_error(self, mock_urlopen: Mock) -> None:
        """URLError with socket timeout should raise RuntimeError."""
        mock_urlopen.side_effect = URLError(socket.timeout("timed out"))
        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="Failed to query"):
            client._query("up")

    @patch("experiments.results.prometheus_client.urlopen")
    def test_http_error_raises_runtime_error(self, mock_urlopen: Mock) -> None:
        """URLError with HTTP error should raise RuntimeError."""
        mock_urlopen.side_effect = URLError("HTTP Error 503: Service Unavailable")
        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="Failed to query"):
            client._query("up")

    @patch("experiments.results.prometheus_client.urlopen")
    def test_query_returns_data_on_success(
        self, mock_urlopen: Mock, mock_vector_response: MagicMock
    ) -> None:
        """Successful query should return parsed data."""
        mock_urlopen.return_value = mock_vector_response
        client = PrometheusClient("http://localhost:9090")

        result = client._query("up")

        assert result["resultType"] == "vector"
        assert len(result["result"]) == 1

    @patch("experiments.results.prometheus_client.urlopen")
    def test_query_failed_status_raises_runtime_error(
        self, mock_urlopen: Mock
    ) -> None:
        """Non-success status should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "error", "error": "bad query"}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="query failed"):
            client._query("bad{query")


# =============================================================================
# Tests for Prometheus Range Query Errors
# =============================================================================


class TestPrometheusRangeQueryErrors:
    """Tests for _query_range method error handling."""

    @patch("experiments.results.prometheus_client.urlopen")
    def test_range_query_connection_error(self, mock_urlopen: Mock) -> None:
        """URLError on range query should raise RuntimeError."""
        mock_urlopen.side_effect = URLError("Connection refused")
        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="Failed to query"):
            client._query_range("up", start=0, end=100)

    @patch("experiments.results.prometheus_client.urlopen")
    def test_range_query_timeout(self, mock_urlopen: Mock) -> None:
        """Socket timeout on range query should raise RuntimeError."""
        mock_urlopen.side_effect = URLError(socket.timeout("timed out"))
        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="Failed to query"):
            client._query_range("up", start=0, end=100)

    @patch("experiments.results.prometheus_client.urlopen")
    def test_range_query_returns_data_on_success(
        self, mock_urlopen: Mock, mock_matrix_response: MagicMock
    ) -> None:
        """Successful range query should return parsed data."""
        mock_urlopen.return_value = mock_matrix_response
        client = PrometheusClient("http://localhost:9090")

        result = client._query_range("up", start=0, end=100)

        assert result["resultType"] == "matrix"
        assert len(result["result"]) == 1

    @patch("experiments.results.prometheus_client.urlopen")
    def test_range_query_failed_status_raises_runtime_error(
        self, mock_urlopen: Mock
    ) -> None:
        """Non-success status on range query should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "error", "errorType": "bad_data"}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient("http://localhost:9090")

        with pytest.raises(RuntimeError, match="range query failed"):
            client._query_range("bad{query", start=0, end=100)


# =============================================================================
# Tests for Prometheus Empty Results
# =============================================================================


class TestPrometheusEmptyResults:
    """Tests for handling empty Prometheus results."""

    @patch("experiments.results.prometheus_client.urlopen")
    def test_cpu_usage_empty_result_returns_zeros(
        self, mock_urlopen: Mock
    ) -> None:
        """Empty CPU result should return zero values."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {"resultType": "matrix", "result": []}
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient("http://localhost:9090")
        result = client.query_cpu_usage("nonexistent-container", 0, 100)

        assert result == {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

    @patch("experiments.results.prometheus_client.urlopen")
    def test_memory_usage_empty_result_returns_zeros(
        self, mock_urlopen: Mock
    ) -> None:
        """Empty memory result should return zero values."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {"resultType": "matrix", "result": []}
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient("http://localhost:9090")
        result = client.query_memory_usage("nonexistent-container", 0, 100)

        assert result == {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

    @patch("experiments.results.prometheus_client.urlopen")
    def test_network_io_empty_result_returns_zeros(
        self, mock_urlopen: Mock
    ) -> None:
        """Empty network I/O result should return zero values."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {"resultType": "matrix", "result": []}
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient("http://localhost:9090")
        result = client.query_network_io("nonexistent-container", 0, 100)

        assert result == {"rx_bytes_per_sec": 0.0, "tx_bytes_per_sec": 0.0}


# =============================================================================
# Tests for Prometheus Data Extraction
# =============================================================================


class TestPrometheusDataExtraction:
    """Tests for _extract_values method."""

    def test_extract_values_matrix_result(self, client: PrometheusClient) -> None:
        """Should extract values from matrix (range query) result."""
        result = {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"container_name": "test"},
                    "values": [[1000, "50.5"], [1005, "60.0"], [1010, "55.2"]],
                }
            ],
        }

        values = client._extract_values(result)

        assert len(values) == 3
        assert values[0] == 50.5
        assert values[1] == 60.0
        assert values[2] == 55.2

    def test_extract_values_vector_result(self, client: PrometheusClient) -> None:
        """Should extract values from vector (instant query) result."""
        result = {
            "resultType": "vector",
            "result": [
                {"metric": {"job": "prometheus"}, "value": [1000, "42.5"]},
                {"metric": {"job": "node"}, "value": [1000, "17.3"]},
            ],
        }

        values = client._extract_values(result)

        assert len(values) == 2
        assert values[0] == 42.5
        assert values[1] == 17.3

    def test_extract_values_handles_invalid_values(
        self, client: PrometheusClient
    ) -> None:
        """Should skip non-numeric values gracefully."""
        result = {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"container_name": "test"},
                    "values": [
                        [1000, "50.5"],
                        [1005, "NaN"],
                        [1010, "invalid"],
                        [1015, "30.0"],
                    ],
                }
            ],
        }

        values = client._extract_values(result)

        # NaN converts to float but "invalid" should be skipped
        # Actually NaN is valid float, so should be 3 values
        assert 50.5 in values
        assert 30.0 in values

    def test_extract_values_empty_result(self, client: PrometheusClient) -> None:
        """Should return empty list for empty result."""
        result = {"resultType": "matrix", "result": []}

        values = client._extract_values(result)

        assert values == []

    def test_extract_values_missing_values_key(
        self, client: PrometheusClient
    ) -> None:
        """Should handle missing 'values' key in series."""
        result = {
            "resultType": "matrix",
            "result": [{"metric": {"container_name": "test"}}],
        }

        values = client._extract_values(result)

        assert values == []

    def test_extract_values_unknown_result_type(
        self, client: PrometheusClient
    ) -> None:
        """Should return empty list for unknown result type."""
        result = {"resultType": "scalar", "result": [1000, "42"]}

        values = client._extract_values(result)

        assert values == []


# =============================================================================
# Tests for Prometheus Availability
# =============================================================================


class TestPrometheusAvailability:
    """Tests for is_available method."""

    @patch("experiments.results.prometheus_client.urlopen")
    def test_is_available_returns_true_on_success(
        self, mock_urlopen: Mock, mock_vector_response: MagicMock
    ) -> None:
        """Should return True when Prometheus responds."""
        mock_urlopen.return_value = mock_vector_response
        client = PrometheusClient("http://localhost:9090")

        assert client.is_available() is True

    @patch("experiments.results.prometheus_client.urlopen")
    def test_is_available_returns_false_on_error(
        self, mock_urlopen: Mock
    ) -> None:
        """Should return False when Prometheus is unreachable."""
        mock_urlopen.side_effect = URLError("Connection refused")
        client = PrometheusClient("http://localhost:9090")

        assert client.is_available() is False

    @patch("experiments.results.prometheus_client.urlopen")
    def test_is_available_returns_false_on_timeout(
        self, mock_urlopen: Mock
    ) -> None:
        """Should return False when query times out."""
        mock_urlopen.side_effect = URLError(socket.timeout("timed out"))
        client = PrometheusClient("http://localhost:9090")

        assert client.is_available() is False


# =============================================================================
# Tests for Query Container Metrics Aggregation
# =============================================================================


class TestQueryContainerMetrics:
    """Tests for query_container_metrics aggregation method."""

    @patch.object(PrometheusClient, "query_network_io")
    @patch.object(PrometheusClient, "query_memory_usage")
    @patch.object(PrometheusClient, "query_cpu_usage")
    def test_queries_all_containers(
        self, mock_cpu: Mock, mock_memory: Mock, mock_network: Mock
    ) -> None:
        """Should query metrics for all provided containers."""
        mock_cpu.return_value = {"avg_percent": 50.0, "max_percent": 80.0, "min_percent": 20.0}
        mock_memory.return_value = {"avg_mb": 512.0, "max_mb": 768.0, "min_mb": 256.0}
        mock_network.return_value = {"rx_bytes_per_sec": 1000.0, "tx_bytes_per_sec": 500.0}

        client = PrometheusClient()
        result = client.query_container_metrics(["container1", "container2"], 0, 100)

        assert len(result) == 2
        assert "container1" in result
        assert "container2" in result
        # Each method should be called twice (once per container)
        assert mock_cpu.call_count == 2
        assert mock_memory.call_count == 2
        assert mock_network.call_count == 2

    @patch.object(PrometheusClient, "query_network_io")
    @patch.object(PrometheusClient, "query_memory_usage")
    @patch.object(PrometheusClient, "query_cpu_usage")
    def test_returns_dict_per_container(
        self, mock_cpu: Mock, mock_memory: Mock, mock_network: Mock
    ) -> None:
        """Should return properly structured dict for each container."""
        mock_cpu.return_value = {"avg_percent": 50.0, "max_percent": 80.0, "min_percent": 20.0}
        mock_memory.return_value = {"avg_mb": 512.0, "max_mb": 768.0, "min_mb": 256.0}
        mock_network.return_value = {"rx_bytes_per_sec": 1000.0, "tx_bytes_per_sec": 500.0}

        client = PrometheusClient()
        result = client.query_container_metrics(["test-container"], 0, 100)

        container_metrics = result["test-container"]
        assert "cpu" in container_metrics
        assert "memory" in container_metrics
        assert "network" in container_metrics
        assert container_metrics["cpu"]["avg_percent"] == 50.0
        assert container_metrics["memory"]["avg_mb"] == 512.0
        assert container_metrics["network"]["rx_bytes_per_sec"] == 1000.0

    @patch.object(PrometheusClient, "query_network_io")
    @patch.object(PrometheusClient, "query_memory_usage")
    @patch.object(PrometheusClient, "query_cpu_usage")
    def test_handles_partial_failures(
        self, mock_cpu: Mock, mock_memory: Mock, mock_network: Mock
    ) -> None:
        """Should handle partial failures gracefully via method return values."""
        # First container succeeds, second returns zeros (as methods do on error)
        def cpu_side_effect(name: str, start: float, end: float) -> dict:
            if name == "good-container":
                return {"avg_percent": 50.0, "max_percent": 80.0, "min_percent": 20.0}
            return {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

        def memory_side_effect(name: str, start: float, end: float) -> dict:
            if name == "good-container":
                return {"avg_mb": 512.0, "max_mb": 768.0, "min_mb": 256.0}
            return {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

        def network_side_effect(name: str, start: float, end: float) -> dict:
            if name == "good-container":
                return {"rx_bytes_per_sec": 1000.0, "tx_bytes_per_sec": 500.0}
            return {"rx_bytes_per_sec": 0.0, "tx_bytes_per_sec": 0.0}

        mock_cpu.side_effect = cpu_side_effect
        mock_memory.side_effect = memory_side_effect
        mock_network.side_effect = network_side_effect

        client = PrometheusClient()
        result = client.query_container_metrics(
            ["good-container", "missing-container"], 0, 100
        )

        # Both containers should be present
        assert len(result) == 2
        # Good container has real data
        assert result["good-container"]["cpu"]["avg_percent"] == 50.0
        # Missing container has zero values
        assert result["missing-container"]["cpu"]["avg_percent"] == 0.0

    @patch.object(PrometheusClient, "query_network_io")
    @patch.object(PrometheusClient, "query_memory_usage")
    @patch.object(PrometheusClient, "query_cpu_usage")
    def test_empty_container_list(
        self, mock_cpu: Mock, mock_memory: Mock, mock_network: Mock
    ) -> None:
        """Should return empty dict for empty container list."""
        client = PrometheusClient()
        result = client.query_container_metrics([], 0, 100)

        assert result == {}
        mock_cpu.assert_not_called()
        mock_memory.assert_not_called()
        mock_network.assert_not_called()


# =============================================================================
# Tests for Metric Query Methods with Mocked Data
# =============================================================================


class TestMetricQueryMethods:
    """Tests for query_cpu_usage, query_memory_usage, query_network_io."""

    @patch("experiments.results.prometheus_client.urlopen")
    def test_cpu_usage_calculates_stats(self, mock_urlopen: Mock) -> None:
        """Should calculate avg/max/min from CPU data."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"container_name": "test"},
                    "values": [[1000, "50.0"], [1005, "60.0"], [1010, "40.0"]]
                }]
            }
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient()
        result = client.query_cpu_usage("test-container", 0, 100)

        assert result["avg_percent"] == 50.0  # (50+60+40)/3
        assert result["max_percent"] == 60.0
        assert result["min_percent"] == 40.0

    @patch("experiments.results.prometheus_client.urlopen")
    def test_memory_usage_calculates_stats(self, mock_urlopen: Mock) -> None:
        """Should calculate avg/max/min from memory data."""
        mock_response = MagicMock()
        # Values are in bytes, query divides by 1024*1024 to get MB
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"container_name": "test"},
                    "values": [[1000, "100.0"], [1005, "200.0"], [1010, "150.0"]]
                }]
            }
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient()
        result = client.query_memory_usage("test-container", 0, 100)

        assert result["avg_mb"] == 150.0  # (100+200+150)/3
        assert result["max_mb"] == 200.0
        assert result["min_mb"] == 100.0

    @patch("experiments.results.prometheus_client.urlopen")
    def test_network_io_calculates_rates(self, mock_urlopen: Mock) -> None:
        """Should calculate average rates from network data."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''{
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"container_name": "test"},
                    "values": [[1000, "1000.0"], [1005, "2000.0"]]
                }]
            }
        }'''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusClient()
        result = client.query_network_io("test-container", 0, 100)

        # Two queries: rx and tx, both get same mock data
        assert result["rx_bytes_per_sec"] == 1500.0  # (1000+2000)/2
        assert result["tx_bytes_per_sec"] == 1500.0

    @patch("experiments.results.prometheus_client.urlopen")
    def test_cpu_usage_handles_exception(self, mock_urlopen: Mock) -> None:
        """Should return zeros when query raises exception."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = PrometheusClient()
        result = client.query_cpu_usage("test-container", 0, 100)

        assert result == {"avg_percent": 0.0, "max_percent": 0.0, "min_percent": 0.0}

    @patch("experiments.results.prometheus_client.urlopen")
    def test_memory_usage_handles_exception(self, mock_urlopen: Mock) -> None:
        """Should return zeros when query raises exception."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = PrometheusClient()
        result = client.query_memory_usage("test-container", 0, 100)

        assert result == {"avg_mb": 0.0, "max_mb": 0.0, "min_mb": 0.0}

    @patch("experiments.results.prometheus_client.urlopen")
    def test_network_io_handles_exception(self, mock_urlopen: Mock) -> None:
        """Should return zeros when query raises exception."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = PrometheusClient()
        result = client.query_network_io("test-container", 0, 100)

        assert result == {"rx_bytes_per_sec": 0.0, "tx_bytes_per_sec": 0.0}
