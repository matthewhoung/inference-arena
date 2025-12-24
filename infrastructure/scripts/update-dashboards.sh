#!/bin/bash
# =============================================================================
# Dashboard Auto-Update Script - Inference Arena
# =============================================================================
#
# Automatically updates Grafana dashboards with current Docker container IDs.
# This ensures dashboards work correctly even after container restarts.
#
# Usage: ./update-dashboards.sh
#
# Author: Matthew Hong
# =============================================================================

set -e

DASHBOARD_DIR="/home/hong/projects/inference-arena/infrastructure/grafana/provisioning/dashboards"

echo "🔍 Discovering container IDs..."

# Get container IDs (12-char short hash)
MONO_ID=$(docker ps --filter "name=inference-arena-monolithic" --format "{{.ID}}" 2>/dev/null || echo "")
MICRO_DETECT_ID=$(docker ps --filter "name=micro-detect" --format "{{.ID}}" 2>/dev/null || echo "")
MICRO_CLASSIFY_ID=$(docker ps --filter "name=micro-classify" --format "{{.ID}}" 2>/dev/null || echo "")
TRITON_GATEWAY_ID=$(docker ps --filter "name=triton-gateway" --format "{{.ID}}" 2>/dev/null || echo "")
TRITON_SERVER_ID=$(docker ps --filter "name=triton-server" --format "{{.ID}}" 2>/dev/null || echo "")

echo ""
echo "Container ID Mapping:"
echo "  Monolithic:        ${MONO_ID:-[NOT RUNNING]}"
echo "  Micro-Detect:      ${MICRO_DETECT_ID:-[NOT RUNNING]}"
echo "  Micro-Classify:    ${MICRO_CLASSIFY_ID:-[NOT RUNNING]}"
echo "  Triton-Gateway:    ${TRITON_GATEWAY_ID:-[NOT RUNNING]}"
echo "  Triton-Server:     ${TRITON_SERVER_ID:-[NOT RUNNING]}"
echo ""

# Update Monolithic Dashboard
if [ -f "$DASHBOARD_DIR/infrastructure-mono.json" ]; then
    if [ -n "$MONO_ID" ]; then
        echo "Updating Monolithic dashboard..."
        sed -i "s/container_id=\\\\\"[a-f0-9]\\{12\\}\\\\\"/container_id=\\\\\"$MONO_ID\\\\\"/g" "$DASHBOARD_DIR/infrastructure-mono.json"
        echo "Updated to container_id=\"$MONO_ID\""
    else
        echo "Monolithic container not running - dashboard not updated"
    fi
fi

# Update Microservices Dashboard
if [ -f "$DASHBOARD_DIR/infrastructure-micro.json" ]; then
    if [ -n "$MICRO_DETECT_ID" ] && [ -n "$MICRO_CLASSIFY_ID" ]; then
        echo "Updating Microservices dashboard..."

        # Replace all occurrences of "000000000000" with micro-detect ID
        sed -i "s/container_id=\\\\\"000000000000\\\\\"/container_id=\\\\\"$MICRO_DETECT_ID\\\\\"/g" "$DASHBOARD_DIR/infrastructure-micro.json"

        # Replace all occurrences of "111111111111" with micro-classify ID
        sed -i "s/container_id=\\\\\"111111111111\\\\\"/container_id=\\\\\"$MICRO_CLASSIFY_ID\\\\\"/g" "$DASHBOARD_DIR/infrastructure-micro.json"

        echo "Updated micro-detect: $MICRO_DETECT_ID"
        echo "Updated micro-classify: $MICRO_CLASSIFY_ID"
    else
        echo "Microservices containers not running - dashboard not updated"
    fi
fi

# Update Triton Dashboard
if [ -f "$DASHBOARD_DIR/infrastructure-triton.json" ]; then
    if [ -n "$TRITON_GATEWAY_ID" ] && [ -n "$TRITON_SERVER_ID" ]; then
        echo "Updating Triton dashboard..."

        # Replace all occurrences of "222222222222" with triton-gateway ID
        sed -i "s/container_id=\\\\\"222222222222\\\\\"/container_id=\\\\\"$TRITON_GATEWAY_ID\\\\\"/g" "$DASHBOARD_DIR/infrastructure-triton.json"

        # Replace all occurrences of "333333333333" with triton-server ID
        sed -i "s/container_id=\\\\\"333333333333\\\\\"/container_id=\\\\\"$TRITON_SERVER_ID\\\\\"/g" "$DASHBOARD_DIR/infrastructure-triton.json"

        echo "Updated triton-gateway: $TRITON_GATEWAY_ID"
        echo "Updated triton-server: $TRITON_SERVER_ID"
    else
        echo "Triton containers not running - dashboard not updated"
    fi
fi

echo ""
echo "Dashboard update complete!"
echo ""
echo "Next steps:"
echo "   1. Restart Grafana: docker restart inference-arena-grafana"
echo "   2. Refresh browser to see updated dashboards"
echo ""
