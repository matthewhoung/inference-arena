"""
Pytest Configuration for Infrastructure Tests

This module provides shared fixtures and configuration for infrastructure tests.
"""

import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Generator

import pytest


class ConfigurableHealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler with configurable failure behavior."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress logging to keep test output clean."""
        pass

    def do_GET(self) -> None:
        """Handle GET requests with configurable failure behavior."""
        server = self.server
        if not isinstance(server, ConfigurableHealthServer):
            self.send_error(500, "Invalid server configuration")
            return

        with server.lock:
            server.request_count += 1
            current_request = server.request_count

        # Apply delay if configured (for timeout testing)
        if server.delay > 0:
            import time

            time.sleep(server.delay)

        # Fail if we haven't exceeded fail_count yet
        if current_request <= server.fail_count:
            self.send_response(server.fail_with)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Intentional failure for testing")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")


class ConfigurableHealthServer(HTTPServer):
    """HTTP server with configurable failure behavior for testing.

    Attributes:
        fail_count: Number of requests to fail before succeeding.
        fail_with: HTTP status code for failures (default: 503).
        delay: Optional delay in seconds before response (for timeout testing).
        request_count: Thread-safe counter of requests received.
    """

    def __init__(
        self,
        server_address: tuple[str, int],
        fail_count: int = 0,
        fail_with: int = 503,
        delay: float = 0.0,
    ) -> None:
        """Initialize configurable health server.

        Args:
            server_address: Tuple of (host, port) to bind to.
            fail_count: Number of requests to fail before succeeding.
            fail_with: HTTP status code for failures.
            delay: Delay in seconds before responding.
        """
        super().__init__(server_address, ConfigurableHealthHandler)
        self.fail_count = fail_count
        self.fail_with = fail_with
        self.delay = delay
        self.request_count = 0
        self.lock = threading.Lock()

    def reset(self) -> None:
        """Reset request counter for test reuse."""
        with self.lock:
            self.request_count = 0


@contextmanager
def run_health_server(
    fail_count: int = 0,
    fail_with: int = 503,
    delay: float = 0.0,
) -> Generator[tuple[ConfigurableHealthServer, str], None, None]:
    """Context manager to run a configurable health server in background thread.

    Args:
        fail_count: Number of requests to fail before succeeding.
        fail_with: HTTP status code for failures.
        delay: Delay in seconds before responding.

    Yields:
        Tuple of (server instance, server URL).
    """
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = ConfigurableHealthServer(
        ("localhost", port),
        fail_count=fail_count,
        fail_with=fail_with,
        delay=delay,
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield server, f"http://localhost:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


@pytest.fixture
def health_server() -> Generator[tuple[ConfigurableHealthServer, str], None, None]:
    """Fixture providing a configurable health server.

    The server starts with default settings (no failures).
    Tests can modify server.fail_count, server.fail_with, and server.delay
    as needed.

    Yields:
        Tuple of (server instance, server URL).
    """
    with run_health_server() as (server, url):
        yield server, url


@pytest.fixture
def unavailable_port() -> int:
    """Fixture providing a port number with no server listening.

    Finds an available port, binds briefly to reserve it, then releases.
    The port will have no listener when tests use it.

    Returns:
        Port number guaranteed to have no server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Docker)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless explicitly requested."""
    if config.getoption("-m") and "integration" in config.getoption("-m"):
        # Integration tests explicitly requested
        return

    skip_integration = pytest.mark.skip(
        reason="Integration tests skipped by default. Run with: pytest -m integration"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
