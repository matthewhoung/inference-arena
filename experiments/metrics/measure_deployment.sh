#!/bin/bash
# =============================================================================
# Deployment Time Measurement Script
# =============================================================================
#
# Measures worst-case cold-start deployment time from docker build start to
# fully operational services responding to HTTP requests.
#
# Protocol:
#   1. Complete cleanup: docker system prune -af --volumes (removes all caches)
#   2. Time: docker compose up --build to all containers healthy + HTTP responds
#   3. Repeat 3 times per architecture (consistent with experiment protocol)
#
# Usage:
#   ./measure_deployment.sh <architecture> [runs]
#   ./measure_deployment.sh monolithic 3
#   ./measure_deployment.sh microservices 3
#   ./measure_deployment.sh triton 3
#
# Output:
#   Appends to analysis/rq3/deployment_times.csv
#   Format: architecture,run,total_time_seconds
#
# Notes:
#   - This will take significant time: Triton downloads 8GB base image each run
#   - Total runtime estimate: 1-2 hours for all architectures
#   - This is INTENTIONAL - measures real worst-case deployment cost
#
# Author: Claude (GSD Plan 03-02)
# Reference: .planning/phases/03-rq3-data-collection/03-CONTEXT.md
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

ARCH="${1:-monolithic}"  # Architecture to measure
RUNS="${2:-3}"           # Number of runs (default 3 per CONTEXT.md)
OUTPUT="results/metrics/deployment_times.csv"

# Port mapping for HTTP verification (per CONTEXT.md)
declare -A PORTS=(
    ["monolithic"]=8100
    ["microservices"]=8200
    ["triton"]=8300
)

# =============================================================================
# Setup
# =============================================================================

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# Write CSV header if file doesn't exist
if [ ! -f "$OUTPUT" ]; then
    echo "architecture,run,total_time_seconds" > "$OUTPUT"
fi

# Validate architecture argument
if [[ ! -v PORTS[$ARCH] ]]; then
    echo "Error: Invalid architecture '$ARCH'"
    echo "Valid options: monolithic, microservices, triton"
    exit 1
fi

PORT=${PORTS[$ARCH]}
COMPOSE_FILE="architectures/${ARCH}/docker-compose.yml"
INFRA_COMPOSE="infrastructure/docker-compose.infra.yml"

# Verify compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# =============================================================================
# Infrastructure Prerequisites
# =============================================================================
# Per CONTEXT.md: Infrastructure (MinIO, metrics) is operational overhead but
# separate from architecture deployment complexity. Ensure infrastructure is
# running as prerequisite, but don't include in timing measurement.
# =============================================================================

echo "=== Checking Infrastructure Prerequisites ==="

# Check if backend network exists (created by infrastructure)
if ! docker network inspect inference-arena-backend > /dev/null 2>&1; then
    echo "Infrastructure network not found. Starting infrastructure..."
    docker compose -f "$INFRA_COMPOSE" up -d

    # Wait for MinIO to be healthy (required by architectures)
    echo "Waiting for MinIO to be healthy..."
    until docker inspect --format='{{.State.Health.Status}}' inference-arena-minio 2>/dev/null | grep -q "healthy"; do
        echo "  Waiting for MinIO..."
        sleep 2
    done
    echo "Infrastructure ready"
fi

# Note: Models will be downloaded by init containers from MinIO
# Deployment timing includes model download time as part of operational complexity

echo "Infrastructure prerequisites satisfied"
echo ""

# =============================================================================
# Measurement Loop
# =============================================================================

echo "=== Measuring Deployment Time: ${ARCH} (${RUNS} runs) ==="
echo "=== Output: ${OUTPUT}"
echo "=== HTTP verification port: ${PORT}"
echo ""

for run in $(seq 1 "$RUNS"); do
    echo ""
    echo "--- Run ${run}/${RUNS} ---"
    echo ""

    # -------------------------------------------------------------------------
    # Step 1: Complete Cleanup (Cold-Start Protocol)
    # -------------------------------------------------------------------------
    # Per CONTEXT.md: "clear Docker cache between runs" for true cold start
    # This removes architecture-specific resources:
    # - Architecture containers and volumes
    # - Architecture images (monolithic/microservices/triton)
    # - All build cache (forces complete rebuild)
    # - Base images (python, nvidia/tritonserver)
    # Note: Infrastructure kept running (MinIO, etc.) as operational prerequisite
    # -------------------------------------------------------------------------
    echo "Cleaning up (this may take 1-2 minutes)..."

    # Stop and remove containers/volumes for this architecture
    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

    # Remove architecture-specific images to force base image pulls
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "inference-arena-(${ARCH}|monolithic|detection|classification|triton)" | xargs -r docker rmi -f 2>/dev/null || true

    # Remove base images used by architectures (forces fresh pulls)
    # - python:3.11-slim (used by all for init containers and custom builds)
    # - nvcr.io/nvidia/tritonserver:24.08-py3 (Triton only, 8GB)
    docker rmi -f python:3.11-slim 2>/dev/null || true
    if [ "$ARCH" = "triton" ]; then
        docker rmi -f nvcr.io/nvidia/tritonserver:24.08-py3 2>/dev/null || true
    fi

    # Clear all build cache (forces complete rebuild, not just layer reuse)
    docker builder prune -af > /dev/null 2>&1

    # Small delay to ensure cleanup completes
    sleep 5

    echo "Cleanup complete. Starting measurement..."

    # -------------------------------------------------------------------------
    # Step 2: Measure Deployment Time
    # -------------------------------------------------------------------------
    # Per CONTEXT.md time boundaries:
    # - Start: docker compose up --build execution begins
    # - Stop: All containers healthy AND HTTP health endpoint responds
    # - Includes: base image pulls + Docker build + container start + warmup
    # -------------------------------------------------------------------------

    START=$(date +%s.%N)

    # Build and start all services
    echo "Building and starting services..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    # -------------------------------------------------------------------------
    # Step 3: Wait for Container Health Checks
    # -------------------------------------------------------------------------
    # Per CONTEXT.md: "All services healthy and responding to requests"
    # Filter logic: Count only running containers with healthy status
    # Excludes: Init containers that exit successfully
    # -------------------------------------------------------------------------
    echo "Waiting for services to be healthy..."

    while true; do
        # Count running containers with healthcheck (excludes exited init containers)
        RUNNING=$(docker compose -f "$COMPOSE_FILE" ps --status running --format json 2>/dev/null | grep -c '"Health"' || echo "0")
        RUNNING=$(echo "$RUNNING" | tr -d '\n')

        # Count healthy containers by inspecting docker ps output
        HEALTHY=$(docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -c "(healthy)" || echo "0")
        HEALTHY=$(echo "$HEALTHY" | tr -d '\n')

        # All running containers with healthchecks must be healthy
        if [ "$RUNNING" -gt 0 ] && [ "$HEALTHY" -eq "$RUNNING" ]; then
            echo "All services healthy (${HEALTHY}/${RUNNING})"
            break
        fi

        echo "  Waiting... (${HEALTHY}/${RUNNING} healthy)"
        sleep 2
    done

    # -------------------------------------------------------------------------
    # Step 4: Verify HTTP Endpoint (Pitfall #5 Protection)
    # -------------------------------------------------------------------------
    # Per CONTEXT.md and RESEARCH.md Pitfall #5:
    # Docker health check may pass but service still warming up
    # Verify actual HTTP connectivity before stopping timer
    # -------------------------------------------------------------------------
    echo "Verifying HTTP endpoint responds..."

    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        echo "  Waiting for HTTP response..."
        sleep 2
    done

    END=$(date +%s.%N)
    TOTAL_TIME=$(echo "$END - $START" | bc)

    echo "Deployment complete: ${TOTAL_TIME}s"

    # -------------------------------------------------------------------------
    # Step 5: Record Measurement
    # -------------------------------------------------------------------------
    # CSV format: architecture,run,total_time_seconds
    # -------------------------------------------------------------------------
    echo "${ARCH},${run},${TOTAL_TIME}" >> "$OUTPUT"

    echo "Run ${run} complete"
done

# =============================================================================
# Summary Statistics
# =============================================================================

echo ""
echo "=== Measurement Complete ==="
echo "Results saved to: ${OUTPUT}"
echo ""
echo "Summary for ${ARCH}:"

# Calculate mean using awk
awk -F, -v arch="$ARCH" '
    NR>1 && $1==arch {
        sum+=$3
        count++
        times[count]=$3
    }
    END {
        if (count > 0) {
            mean = sum/count
            printf "  Runs: %d\n", count
            printf "  Mean: %.1fs\n", mean
            printf "  Times: "
            for (i=1; i<=count; i++) {
                printf "%.1fs", times[i]
                if (i < count) printf ", "
            }
            printf "\n"

            # Calculate std deviation
            sum_sq = 0
            for (i=1; i<=count; i++) {
                sum_sq += (times[i] - mean)^2
            }
            std = sqrt(sum_sq/count)
            printf "  Std Dev: %.1fs\n", std
        }
    }
' "$OUTPUT"

echo ""
echo "=== Next Steps ==="
echo "To measure all architectures:"
echo "  for arch in monolithic microservices triton; do"
echo "    bash experiments/metrics/measure_deployment.sh \$arch 3"
echo "  done"
echo ""
