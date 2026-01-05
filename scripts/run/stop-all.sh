#!/bin/bash
# =============================================================================
# Stop All Containers - Inference Arena
# =============================================================================
#
# Stops all running containers (infrastructure + architectures).
#
# Usage: ./stop-all.sh
#
# Author: Matthew Hong
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Stopping all Inference Arena containers..."

# Stop architectures (ignore errors if not running)
echo "Stopping Monolithic..."
cd "$PROJECT_ROOT/architectures/monolithic" && docker compose down 2>/dev/null || true

echo "Stopping Microservices..."
cd "$PROJECT_ROOT/architectures/microservices" && docker compose down 2>/dev/null || true

echo "Stopping Triton..."
cd "$PROJECT_ROOT/architectures/triton" && docker compose down 2>/dev/null || true

# Stop infrastructure
echo "Stopping Infrastructure..."
docker compose -f "$PROJECT_ROOT/infrastructure/docker-compose.infra.yml" down 2>/dev/null || true

echo ""
echo "All containers stopped."
