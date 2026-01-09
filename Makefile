# =============================================================================
# Inference Arena - Makefile
# =============================================================================
# ML Serving Architecture Benchmark - Compare Monolithic, Microservices, and Triton
#
# Quick Start:
#   make setup      # Complete setup (install deps, start infra, download data, upload models)
#   make start-mono # Start monolithic architecture
#   make test-quick # Run a quick load test
#
# Reset Everything:
#   make reset      # Stop all, clean results, prune Docker, start fresh
#
# Author: Matthew Hong
# =============================================================================

.PHONY: help setup reset \
        install test test-fast lint format clean clean-results clean-all \
        docker-build docker-build-mono docker-build-micro docker-build-triton docker-prune \
        start-infra stop-infra start-mono start-micro start-triton \
        stop-mono stop-micro stop-triton stop-all \
        proto models-export models-init-minio models-init-minio-batched models-setup models-verify \
        data-download data-curate data-verify \
        update-dashboards restart-grafana refresh-dashboards \
        test-quick test-arch test-matrix test-dry-run test-web

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Project paths
PROJECT_ROOT := $(shell pwd)
VENV := $(PROJECT_ROOT)/.venv/bin

# Default variables (ARCH: mono, micro, triton)
ARCH ?= mono
USERS ?= 10
RUNS ?= 1

# Map ARCH names to full names for the experiment runner
# Supports both short (mono, micro, triton) and full names (monolithic, microservices, triton)
ARCH_MAP_mono := monolithic
ARCH_MAP_micro := microservices
ARCH_MAP_triton := triton
ARCH_MAP_monolithic := monolithic
ARCH_MAP_microservices := microservices
ARCH_FULL := $(ARCH_MAP_$(ARCH))

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║$(NC)  $(GREEN)Inference Arena$(NC) - ML Serving Architecture Benchmark          $(BLUE)║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)🚀 Quick Start:$(NC)"
	@echo "  $(BLUE)make setup$(NC)        Complete setup (install, infra, data, models)"
	@echo "  $(BLUE)make start-mono$(NC)   Start monolithic architecture"
	@echo "  $(BLUE)make test-quick$(NC)   Run a quick load test (10 users)"
	@echo ""
	@echo "$(YELLOW)🔄 Reset:$(NC)"
	@echo "  $(BLUE)make reset$(NC)        Stop all, clean results, prune Docker"
	@echo "  $(BLUE)make clean-all$(NC)    Remove caches, artifacts, and results"
	@echo ""
	@echo "$(YELLOW)🏗️  Setup & Infrastructure:$(NC)"
	@echo "  $(BLUE)make install$(NC)      Install Python dependencies (uv sync)"
	@echo "  $(BLUE)make start-infra$(NC)  Start MinIO, Prometheus, Grafana, cAdvisor"
	@echo "  $(BLUE)make stop-infra$(NC)   Stop infrastructure containers"
	@echo ""
	@echo "$(YELLOW)🐳 Architectures:$(NC)"
	@echo "  $(BLUE)make start-mono$(NC)   Start monolithic      (http://localhost:8100)"
	@echo "  $(BLUE)make start-micro$(NC)  Start microservices   (http://localhost:8200)"
	@echo "  $(BLUE)make start-triton$(NC) Start Triton          (http://localhost:8300)"
	@echo "  $(BLUE)make stop-all$(NC)     Stop all containers"
	@echo ""
	@echo "$(YELLOW)📦 Data & Models:$(NC)"
	@echo "  $(BLUE)make data-download$(NC)    Download COCO test images"
	@echo "  $(BLUE)make models-setup$(NC)     Export ONNX models + upload to MinIO (with batched)"
	@echo "  $(BLUE)make data-verify$(NC)      Verify data setup"
	@echo ""
	@echo "$(YELLOW)🧪 Load Testing:$(NC)"
	@echo "  $(BLUE)make test-quick$(NC)   Quick test (10 users, 1 run)"
	@echo "  $(BLUE)make test-arch$(NC)    Test one architecture (ARCH=mono|micro|triton)"
	@echo "  $(BLUE)make test-matrix$(NC)  Full experiment matrix (63 tests, ~4.7h)"
	@echo "  $(BLUE)make test-web$(NC)     Start Locust web UI (http://localhost:8089)"
	@echo ""
	@echo "$(YELLOW)🔧 Development:$(NC)"
	@echo "  $(BLUE)make test$(NC)         Run unit tests with coverage"
	@echo "  $(BLUE)make lint$(NC)         Run linters (ruff + mypy)"
	@echo "  $(BLUE)make format$(NC)       Format code (black + ruff)"
	@echo ""

# =============================================================================
# 🚀 Quick Start Commands
# =============================================================================

setup: ## Complete setup: install deps, start infra, download data, upload models
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Setting up Inference Arena...$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/5]$(NC) Installing Python dependencies..."
	@uv sync --all-extras
	@echo ""
	@echo "$(YELLOW)[2/5]$(NC) Starting infrastructure (MinIO, Prometheus, Grafana)..."
	@docker compose -f infrastructure/docker-compose.infra.yml up -d
	@echo "  Waiting for services to be ready..."
	@sleep 5
	@echo ""
	@echo "$(YELLOW)[3/5]$(NC) Exporting models to ONNX..."
	@$(VENV)/python scripts/models/export.py
	@echo ""
	@echo "$(YELLOW)[4/5]$(NC) Downloading test data and curating thesis test set..."
	@$(VENV)/python scripts/setup/download-data.py
	@echo ""
	@echo "$(YELLOW)[5/5]$(NC) Uploading models to MinIO (with batched variants)..."
	@$(VENV)/python scripts/models/init-minio.py --include-batched
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Setup complete! $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  $(BLUE)MinIO Console:$(NC)  http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  $(BLUE)Prometheus:$(NC)     http://localhost:9090"
	@echo "  $(BLUE)Grafana:$(NC)        http://localhost:3000 (admin/admin)"
	@echo ""
	@echo "  $(YELLOW)Next steps:$(NC)"
	@echo "    make docker-build   # Build architecture images"
	@echo "    make start-mono     # Start monolithic architecture"
	@echo "    make test-quick     # Run a quick load test"
	@echo ""

reset: ## Stop all containers, clean results, prune Docker
	@echo "$(RED)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(RED)  Resetting Inference Arena...$(NC)"
	@echo "$(RED)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)[1/4]$(NC) Stopping all containers..."
	@cd architectures/monolithic && docker compose down 2>/dev/null || true
	@cd architectures/microservices && docker compose down 2>/dev/null || true
	@cd architectures/triton && docker compose down 2>/dev/null || true
	@docker compose -f infrastructure/docker-compose.infra.yml down 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)[2/4]$(NC) Cleaning experiment results..."
	@rm -rf results/experiment/ 2>/dev/null || true
	@rm -rf results/tmp/ 2>/dev/null || true
	@rm -rf results/processed/*.csv results/processed/*.json 2>/dev/null || true
	@rm -rf results/coverage_html/ 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)[3/4]$(NC) Cleaning Python caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)[4/4]$(NC) Pruning Docker resources..."
	@docker system prune -f 2>/dev/null || true
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Reset complete!$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  $(YELLOW)To start fresh:$(NC)"
	@echo "    make setup          # Full setup"
	@echo "    make start-infra    # Just infrastructure"
	@echo ""

health: ## Check health of all running services
	@echo "$(YELLOW)Checking service health...$(NC)"
	@echo ""
	@echo "$(BLUE)Infrastructure:$(NC)"
	@curl -s http://localhost:9000/minio/health/live > /dev/null 2>&1 && echo "  ✅ MinIO" || echo "  ❌ MinIO"
	@curl -s http://localhost:9090/-/healthy > /dev/null 2>&1 && echo "  ✅ Prometheus" || echo "  ❌ Prometheus"
	@curl -s http://localhost:3000/api/health > /dev/null 2>&1 && echo "  ✅ Grafana" || echo "  ❌ Grafana"
	@curl -s http://localhost:8080/metrics > /dev/null 2>&1 && echo "  ✅ cAdvisor" || echo "  ❌ cAdvisor"
	@echo ""
	@echo "$(BLUE)Architectures:$(NC)"
	@curl -s http://localhost:8100/health > /dev/null 2>&1 && echo "  ✅ Monolithic (8100)" || echo "  ❌ Monolithic (8100)"
	@curl -s http://localhost:8200/health > /dev/null 2>&1 && echo "  ✅ Microservices (8200)" || echo "  ❌ Microservices (8200)"
	@curl -s http://localhost:8300/health > /dev/null 2>&1 && echo "  ✅ Triton (8300)" || echo "  ❌ Triton (8300)"
	@echo ""

# =============================================================================
# 🏗️ Installation & Development
# =============================================================================

install: ## Install all Python dependencies using uv
	uv sync --all-extras
	@echo "$(GREEN)Dependencies installed$(NC)"

test: ## Run all tests with coverage
	$(VENV)/pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:results/coverage_html

test-fast: ## Run tests without slow markers
	$(VENV)/pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing

lint: ## Run linters (ruff + mypy)
	$(VENV)/ruff check src/ tests/
	$(VENV)/mypy src/ --ignore-missing-imports

format: ## Format code (black + ruff fix)
	$(VENV)/black src/ tests/ experiments/
	$(VENV)/ruff check src/ tests/ experiments/ --fix

clean: ## Remove Python caches and build artifacts
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "$(GREEN)Caches cleaned$(NC)"

clean-results: ## Remove experiment results
	@rm -rf results/experiment/ 2>/dev/null || true
	@rm -rf results/tmp/ 2>/dev/null || true
	@rm -rf results/processed/*.csv results/processed/*.json 2>/dev/null || true
	@rm -rf results/coverage_html/ 2>/dev/null || true
	@echo "$(GREEN)Results cleaned$(NC)"

clean-all: clean clean-results ## Remove all caches, artifacts, and results
	@echo "$(GREEN)All cleaned$(NC)"

proto: ## Generate gRPC protocol buffer files
	$(VENV)/python scripts/setup/generate-proto.py

# =============================================================================
# 🐳 Docker Build
# =============================================================================

docker-build: docker-build-mono docker-build-micro docker-build-triton ## Build all Docker images
	@echo "$(GREEN)All Docker images built$(NC)"

docker-build-mono: ## Build monolithic architecture image
	@echo "$(YELLOW)Building monolithic image...$(NC)"
	docker build -f architectures/monolithic/Dockerfile -t inference-arena-monolithic:latest .

docker-build-micro: ## Build microservices architecture images
	@echo "$(YELLOW)Building microservices images...$(NC)"
	docker build -f architectures/microservices/detection/Dockerfile -t inference-arena-detection:latest .
	docker build -f architectures/microservices/classification/Dockerfile -t inference-arena-classification:latest .

docker-build-triton: ## Build Triton gateway image
	@echo "$(YELLOW)Building Triton gateway image...$(NC)"
	docker build -f architectures/triton/gateway/Dockerfile -t inference-arena-triton-gateway:latest .

docker-prune: ## Prune unused Docker resources (images, containers, volumes)
	@echo "$(YELLOW)Pruning Docker resources...$(NC)"
	docker system prune -af --volumes
	@echo "$(GREEN)Docker pruned$(NC)"

# =============================================================================
# 🏢 Infrastructure Services
# =============================================================================

start-infra: ## Start infrastructure (MinIO, Prometheus, Grafana, cAdvisor)
	docker compose -f infrastructure/docker-compose.infra.yml up -d
	@echo ""
	@echo "$(GREEN)Infrastructure started$(NC)"
	@echo "  MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  Grafana:       http://localhost:3000 (admin/admin)"
	@echo "  cAdvisor:      http://localhost:8080"
	@echo ""

stop-infra: ## Stop infrastructure containers
	docker compose -f infrastructure/docker-compose.infra.yml down
	@echo "$(GREEN)Infrastructure stopped$(NC)"

# =============================================================================
# 🚀 Architecture Services
# =============================================================================

start-mono: ## Start monolithic architecture
	cd architectures/monolithic && docker compose up -d
	@echo "$(GREEN)Monolithic started$(NC) - http://localhost:8100"
	@sleep 2 && $(VENV)/python scripts/utils/update-dashboards.py 2>/dev/null || true

start-micro: ## Start microservices architecture
	cd architectures/microservices && docker compose up -d
	@echo "$(GREEN)Microservices started$(NC) - http://localhost:8200"
	@sleep 2 && $(VENV)/python scripts/utils/update-dashboards.py 2>/dev/null || true

start-triton: ## Start Triton architecture
	cd architectures/triton && docker compose up -d
	@echo "$(GREEN)Triton started$(NC) - http://localhost:8300"
	@sleep 2 && $(VENV)/python scripts/utils/update-dashboards.py 2>/dev/null || true

stop-mono: ## Stop monolithic architecture
	cd architectures/monolithic && docker compose down

stop-micro: ## Stop microservices architecture
	cd architectures/microservices && docker compose down

stop-triton: ## Stop Triton architecture
	cd architectures/triton && docker compose down

stop-all: ## Stop all containers (infrastructure + architectures)
	@cd architectures/monolithic && docker compose down 2>/dev/null || true
	@cd architectures/microservices && docker compose down 2>/dev/null || true
	@cd architectures/triton && docker compose down 2>/dev/null || true
	@docker compose -f infrastructure/docker-compose.infra.yml down 2>/dev/null || true
	@echo "$(GREEN)All containers stopped$(NC)"

# =============================================================================
# 📦 Models
# =============================================================================

models-export: ## Export models to ONNX format
	$(VENV)/python scripts/models/export.py

models-init-minio: ## Upload models to MinIO with Triton structure
	$(VENV)/python scripts/models/init-minio.py

models-init-minio-batched: ## Upload models to MinIO (including batched variants for Triton experiments)
	$(VENV)/python scripts/models/init-minio.py --include-batched

models-setup: models-export models-init-minio-batched ## Export ONNX models and upload to MinIO (with batched variants)
	@echo "$(GREEN)✓ Models exported and uploaded to MinIO$(NC)"

models-verify: ## Verify models in MinIO
	$(VENV)/python scripts/models/init-minio.py --verify

# =============================================================================
# 📊 Data
# =============================================================================

data-download: ## Download COCO and curate thesis test dataset (100 images)
	$(VENV)/python scripts/setup/download-data.py

data-verify: ## Verify data setup (COCO + thesis test set)
	$(VENV)/python scripts/setup/download-data.py --verify

# =============================================================================
# 📈 Grafana Dashboards
# =============================================================================

update-dashboards: ## Update Grafana dashboards with current container IDs
	$(VENV)/python scripts/utils/update-dashboards.py

restart-grafana: ## Restart Grafana to reload dashboard changes
	docker restart inference-arena-grafana
	@echo "$(GREEN)Grafana restarted$(NC) - http://localhost:3000"

refresh-dashboards: update-dashboards restart-grafana ## Update dashboards AND restart Grafana

# =============================================================================
# 🧪 Load Testing (ARCH=mono|micro|triton, USERS=10, RUNS=1)
# =============================================================================

test-quick: ## Quick load test: 10 users, 1 run (ARCH=mono|micro|triton)
	@echo "$(YELLOW)Running quick load test: $(ARCH_FULL) - 10 users - 1 run$(NC)"
	$(VENV)/python -m experiments -a $(ARCH_FULL) -u 10 -r 1 --no-docker

test-arch: ## Test one architecture with all load levels (ARCH=mono|micro|triton)
	@echo "$(YELLOW)Running all load levels for $(ARCH_FULL)$(NC)"
	$(VENV)/python -m experiments -a $(ARCH_FULL) --no-docker

test-matrix: ## Run full experiment matrix (63 experiments, ~4.7 hours)
	@echo "$(YELLOW)Running full experiment matrix$(NC)"
	@echo "  3 architectures × 7 load levels × 3 runs = 63 experiments"
	@echo ""
	$(VENV)/python -m experiments

test-dry-run: ## Preview experiment plan without executing
	$(VENV)/python -m experiments --dry-run

test-web: ## Start Locust web UI for manual testing (http://localhost:8089)
	@echo "$(GREEN)Starting Locust Web UI$(NC)"
	@echo "  Open: http://localhost:8089"
	@echo "  Target: http://localhost:8100 (change in UI)"
	$(VENV)/locust -f experiments/locustfile.py --host=http://localhost:8100

test-single: ## Single test run (ARCH=mono|micro|triton, USERS, RUNS)
	@echo "$(YELLOW)Running: $(ARCH_FULL) - $(USERS) users - $(RUNS) runs$(NC)"
	$(VENV)/python -m experiments -a $(ARCH_FULL) -u $(USERS) -r $(RUNS) --no-docker
