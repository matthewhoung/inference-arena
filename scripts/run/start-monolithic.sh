#!/bin/bash
# =============================================================================
# Auto-Start Script - Monolithic Architecture
# =============================================================================
#
# Starts monolithic architecture and automatically updates Grafana dashboards.
#
# Author: Matthew Hong
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting Monolithic Architecture..."
echo ""

# Start infrastructure if not running
if ! docker ps | grep -q "inference-arena-grafana"; then
    echo "Starting infrastructure services..."
    docker-compose -f "$PROJECT_ROOT/infrastructure/docker-compose.infra.yml" up -d
    echo "Infrastructure started"
    echo ""
fi

# Start monolithic architecture
echo "Starting monolithic container..."
docker-compose -f "$PROJECT_ROOT/architectures/monolithic/docker-compose.yml" up -d
echo "Monolithic started"
echo ""

# Wait for containers to be healthy
echo "Waiting for containers to be healthy..."
sleep 5
echo ""

# Auto-update dashboards
echo "Auto-updating Grafana dashboards..."
"$PROJECT_ROOT/infrastructure/scripts/update-dashboards.sh"
echo ""

# Restart Grafana to load updated dashboards
echo "Reloading Grafana..."
docker restart inference-arena-grafana > /dev/null 2>&1
sleep 3
echo "Grafana reloaded"
echo ""

echo "Monolithic architecture is ready!"
echo ""
echo "Grafana Dashboard: http://localhost:3000"
echo "   Dashboard: Inference Arena - Monolithic"
echo ""
echo "Service Health:"
docker ps --filter "name=inference-arena-monolithic" --format "   {{.Names}}: {{.Status}}"
echo ""
