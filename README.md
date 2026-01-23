# Inference Arena

**A Comparative Study of ML Model Serving Architectures: Monolithic vs Microservices vs NVIDIA Triton**

[![CI](https://github.com/matthewhoung/inference-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/matthewhoung/inference-arena/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/matthewhoung/inference-arena/branch/main/graph/badge.svg)](https://codecov.io/gh/matthewhoung/inference-arena)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Master's Thesis Project
> Author: Matthew Hong
> Single Source of Truth: [experiment.yaml](experiment.yaml)

---

## Overview

This repository contains the experimental artifacts for a master's thesis comparing three ML model serving architectures under controlled conditions. The study evaluates **performance**, **resource efficiency**, and **operational complexity** trade-offs to help practitioners make informed architectural decisions.

### Key Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | How do latency (P50, P99) and throughput differ across architectures under varying load? |
| **RQ2** | What is the resource consumption and cost-per-request for each architecture? |
| **RQ3** | What is the deployment complexity (configuration LOC, deployment time) for each? |
| **RQ4** | Under what conditions is each architecture optimal? |

See [experiment.yaml](experiment.yaml) for full hypotheses and pre-registered experimental design.

---

## Architectures Under Test

### Architecture A: Monolithic
Single container consolidating preprocessing, detection, and classification.

```
[Client] → HTTP POST /predict → [FastAPI + YOLO + MobileNet]
                                        ↓
                                 [1 Container: 2 vCPU, 4GB]
```

### Architecture B: Microservices
Two independent services communicating via gRPC with async fan-out.

```
[Client] → HTTP → [Detection Service + YOLO]
                            ↓ gRPC (async fan-out)
                  [Classification Service + MobileNet]
                            ↓
                  [2 Containers: 4 vCPU, 8GB total]
```

### Architecture C: NVIDIA Triton
FastAPI gateway with preprocessing, forwarding to Triton Inference Server.

```
[Client] → HTTP → [FastAPI Gateway + preprocessing]
                            ↓ gRPC
                  [Triton Server (YOLO + MobileNet)]
                            ↓
                  [2 Containers: 4 vCPU, 8GB total]
```

---

## Workload: Multi-Model Pipeline

A two-stage computer vision pipeline with controlled fan-out:

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| Detection | YOLOv5n (ONNX) | 640×640 image | Bounding boxes |
| Classification | MobileNetV2 (ONNX) | 224×224 crops | 1000-class probabilities |

**Fan-out Factor:** Each image produces 3-5 detections (μ=4, σ≈0.71), curated from COCO val2017.

**All parameters defined in [experiment.yaml](experiment.yaml)** - the single source of truth for reproducibility.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup (3 Commands)

```bash
# 1. Clone
git clone https://github.com/matthewhoung/inference-arena.git
cd inference-arena

# 2. Install dependencies
make install

# 3. Run tests
make test
```

### Run Experiments

```bash
# Start infrastructure (MinIO, Prometheus, Grafana)
make start-infra

# Start an architecture
make start-mono     # Monolithic
make start-micro    # Microservices
make start-triton   # Triton

# Stop everything
make stop-all
```

### Service Endpoints

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

**For detailed setup, see [docs/SETUP.md](docs/SETUP.md)**

---

## Project Structure

```
inference-arena/
├── experiment.yaml              # Single source of truth (all experimental params)
├── Makefile                     # Common commands (make help)
├── pyproject.toml               # Python dependencies
├── src/shared/
│   ├── config.py                # Loads experiment.yaml
│   ├── processing/              # Preprocessing pipelines
│   ├── model/                   # Model registry
│   └── triton/                  # Triton config & MinIO utilities
├── architectures/
│   ├── monolithic/              # Architecture A
│   ├── microservices/           # Architecture B
│   └── triton/                  # Architecture C
├── infrastructure/
│   ├── docker-compose.infra.yml # MinIO, Prometheus, Grafana
│   ├── grafana/                 # Dashboards and provisioning
│   └── prometheus/              # Scrape configuration
├── scripts/
│   ├── setup/                   # Environment & proto setup
│   ├── models/                  # Export & upload models
│   └── utils/                   # Dashboard utilities
├── tests/                       # 100+ tests
└── docs/                        # Documentation
```

---

## Configuration Philosophy

This project uses **two complementary configuration systems**:

### `experiment.yaml` - Scientific Configuration ✅ Git-tracked
- Model specifications (architectures, opset versions, I/O shapes)
- Preprocessing parameters (input sizes, normalization)
- Controlled variables (resources, threading, dataset)
- Pre-registered hypotheses and predictions
- Load testing protocol

### `.env` - Deployment Configuration ❌ Git-ignored
- Infrastructure credentials (MinIO, Grafana passwords)
- Port mappings (for local development)
- Environment-specific overrides

**Key Principle:** `experiment.yaml` defines **WHAT** you're testing (reproducible science), `.env` defines **WHERE** to run it (local secrets).

---

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** - Complete setup guide for committee members
- **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)** - Detailed environment configuration
- **[experiment.yaml](experiment.yaml)** - Full experimental specification with inline documentation

---

## Test Coverage

All preprocessing pipelines have comprehensive test coverage:

| Module | Coverage |
|--------|----------|
| `transforms.py` | 88% |
| `yolo_preprocess.py` | 96% |
| `mobilenet_preprocess.py` | 95% |
| `config.py` | 100% |
| **Total** | **93%** |

Run tests with:
```bash
pytest tests/ --cov=src/shared --cov-report=html
```

---

## Controlled Variables

All architectures use **identical** configurations (defined in `experiment.yaml`):

| Variable | Value | Purpose |
|----------|-------|---------|
| ML Models | YOLOv5n + MobileNetV2 (ONNX) | Byte-identical weights |
| Preprocessing | Shared preprocessing module | Eliminates variance |
| Model Source | MinIO (S3-compatible) | Single source for all architectures |
| Container Resources | 2 vCPU, 4GB per container | Fair comparison |
| ONNX Threading | `intra_op=2`, `inter_op=1` | Optimal for 2 vCPU |
| Test Dataset | 100 COCO images (curated) | Controlled fan-out factor |

---

## Reproducibility

This project demonstrates research best practices:

- ✅ **Pre-registered Hypotheses** - All hypotheses defined before data collection
- ✅ **Single Source of Truth** - All parameters in version-controlled `experiment.yaml`
- ✅ **No Hardcoding** - Configuration values loaded from centralized config
- ✅ **Comprehensive Testing** - 100+ tests validate configuration consistency
- ✅ **Git Changelog** - All experiment.yaml changes tracked with rationale
- ✅ **Cross-Platform** - Python-based setup works on Windows/Mac/Linux

---

## Known Limitations

This project has documented version constraints and workarounds:

- **ONNX Version Pinning** - IR version compatibility with Triton. See [docs/ONNX_UPGRADE.md](docs/ONNX_UPGRADE.md)
- **YOLOv5 Legacy Exporter** - Dynamic axes support requirement. See [docs/YOLO_EXPORT.md](docs/YOLO_EXPORT.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NVIDIA Triton Inference Server team
- Ultralytics YOLOv5 team
- COCO dataset maintainers
- Anthropic Claude for development assistance

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{hong2025inference,
  title={Characterizing ML Serving Architectures in CPU-Constrained Environments},
  author={Hong, Matthew},
  year={2025},
  school={[National Chung Hsing University]},
  type={Master's Thesis}
}
```
