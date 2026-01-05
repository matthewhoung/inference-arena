# =============================================================================
# Inference Arena - Makefile
# =============================================================================
# Common development and deployment commands for the ML serving benchmark project.
#
# Usage: make <target>
#
# Author: Matthew Hong
# =============================================================================

.PHONY: help install test test-fast test-cov lint format clean \
        docker-build docker-build-mono docker-build-micro docker-build-triton \
        start-infra start-mono start-micro start-triton stop-all \
        proto models-export models-init-minio models-generate-pbtxt \
        data-download data-verify

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Project paths
PROJECT_ROOT := $(shell pwd)
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Inference Arena$(NC) - ML Serving Architecture Benchmark"
	@echo ""
	@echo "$(GREEN)Usage:$(NC) make <target>"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' | grep -E "(install|test|lint|format|clean|proto)"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' | grep -E "docker"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' | grep -E "(start|stop)"
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' | grep -E "models"
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' | grep -E "data"

# =============================================================================
# Development
# =============================================================================

install: ## Install all dependencies using uv
	uv sync --all-extras

test: ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:results/coverage_html

test-fast: ## Run tests without slow markers
	pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing

test-cov: ## Run tests and check coverage threshold (80%)
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

lint: ## Run linters (ruff + mypy)
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Format code (black + ruff fix)
	black src/ tests/
	ruff check src/ tests/ --fix

clean: ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true

proto: ## Generate gRPC protocol buffer files
	python scripts/setup/generate-proto.py

# =============================================================================
# Docker Build
# =============================================================================

docker-build: docker-build-mono docker-build-micro docker-build-triton ## Build all Docker images

docker-build-mono: ## Build monolithic architecture image
	docker build -f architectures/monolithic/Dockerfile -t inference-arena-monolithic:latest .

docker-build-micro: ## Build microservices architecture images
	docker build -f architectures/microservices/detection/Dockerfile -t inference-arena-detection:latest .
	docker build -f architectures/microservices/classification/Dockerfile -t inference-arena-classification:latest .

docker-build-triton: ## Build Triton gateway image
	docker build -f architectures/triton/gateway/Dockerfile -t inference-arena-triton-gateway:latest .

# =============================================================================
# Services - Infrastructure
# =============================================================================

start-infra: ## Start infrastructure (MinIO, Prometheus, Grafana, cAdvisor)
	docker compose -f infrastructure/docker-compose.infra.yml up -d
	@echo "$(GREEN)Infrastructure started$(NC)"
	@echo "  MinIO Console: http://localhost:9001"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  Grafana:       http://localhost:3000"

stop-infra: ## Stop infrastructure containers
	docker compose -f infrastructure/docker-compose.infra.yml down

# =============================================================================
# Services - Architectures
# =============================================================================

start-mono: ## Start monolithic architecture
	cd architectures/monolithic && docker compose up -d
	@echo "$(GREEN)Monolithic started$(NC) - http://localhost:8100"

start-micro: ## Start microservices architecture
	cd architectures/microservices && docker compose up -d
	@echo "$(GREEN)Microservices started$(NC) - http://localhost:8200"

start-triton: ## Start Triton architecture
	cd architectures/triton && docker compose up -d
	@echo "$(GREEN)Triton started$(NC) - http://localhost:8300"

stop-mono: ## Stop monolithic architecture
	cd architectures/monolithic && docker compose down

stop-micro: ## Stop microservices architecture
	cd architectures/microservices && docker compose down

stop-triton: ## Stop Triton architecture
	cd architectures/triton && docker compose down

stop-all: ## Stop all containers (infrastructure + architectures)
	cd architectures/monolithic && docker compose down 2>/dev/null || true
	cd architectures/microservices && docker compose down 2>/dev/null || true
	cd architectures/triton && docker compose down 2>/dev/null || true
	docker compose -f infrastructure/docker-compose.infra.yml down 2>/dev/null || true
	@echo "$(GREEN)All containers stopped$(NC)"

# =============================================================================
# Models
# =============================================================================

models-export: ## Export models to ONNX format
	python scripts/models/export.py

models-init-minio: ## Upload models to MinIO with Triton structure
	python scripts/models/init-minio.py

models-generate-pbtxt: ## Generate Triton config.pbtxt files
	python scripts/models/generate-pbtxt.py --print

# =============================================================================
# Utilities
# =============================================================================

update-dashboards: ## Update Grafana dashboards with current container IDs
	bash scripts/utils/update-dashboards.sh

# =============================================================================
# Data
# =============================================================================

data-download: ## Download COCO and curate thesis test dataset (100 images)
	python scripts/setup/download-data.py

data-verify: ## Verify data setup (COCO + thesis test set)
	python scripts/setup/download-data.py --verify
