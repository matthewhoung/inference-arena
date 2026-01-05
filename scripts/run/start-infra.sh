#!/bin/bash
# =============================================================================
# Start Infrastructure - Inference Arena
# =============================================================================
#
# Starts the infrastructure services: MinIO, Prometheus, Grafana, cAdvisor.
#
# Usage: ./start-infra.sh
#
# Author: Matthew Hong
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Starting Inference Arena Infrastructure..."

# Start infrastructure
docker compose -f "$PROJECT_ROOT/infrastructure/docker-compose.infra.yml" up -d

echo ""
echo "Infrastructure started successfully!"
echo ""
echo "Services:"
echo "  MinIO API:      http://localhost:9000"
echo "  MinIO Console:  http://localhost:9001"
echo "  cAdvisor:       http://localhost:8080"
echo "  Prometheus:     http://localhost:9090"
echo "  Grafana:        http://localhost:3000"
echo ""
