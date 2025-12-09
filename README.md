# Inference Arena

**A Comparative Study of ML Model Serving Architectures: Monolithic vs Microservices vs NVIDIA Triton**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains the experimental artifacts for a master's thesis comparing three ML model serving architectures under controlled conditions. The study evaluates **performance**, **resource efficiency**, and **operational complexity** trade-offs to help practitioners make informed architectural decisions.

### Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | How do latency (P50, P99) and throughput differ across architectures under varying load? |
| **RQ2** | What is the resource consumption and cost-per-request for each architecture? |
| **RQ3** | What is the deployment complexity (configuration LOC, deployment time) for each? |
| **RQ4** | Under what conditions is each architecture optimal? |

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

A two-stage computer vision pipeline with controlled fan-out behavior:

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| Detection | YOLOv5n (ONNX) | 640×640 image | Bounding boxes |
| Classification | MobileNetV2 (ONNX) | 224×224 crops | 1000-class probabilities |

**Fan-out Factor:** Each image produces 3-5 detections (μ=4, σ≈0.8), curated from COCO val2017.

---

## Repository Structure

```
inference-arena/
├── common/                     # Shared controlled variables
│   ├── preprocessing.py        # Identical preprocessing across all architectures
│   └── proto/
│       └── inference.proto     # gRPC service definition
│
├── models/                     # ONNX models (downloaded, not committed)
│   └── download_models.py
│
├── data/                       # Test dataset
│   ├── curate_dataset.py       # Fan-out controlled dataset generator
│   └── thesis_test_set/        # 100 curated COCO images
│
├── architectures/
│   ├── monolithic/             # Architecture A
│   ├── microservices/          # Architecture B
│   └── triton/                 # Architecture C
│
├── infrastructure/             # MinIO, Prometheus, cAdvisor
│   └── docker-compose.infra.yml
│
├── experiments/                # Load testing & data collection
│   ├── locustfile.py
│   └── run_matrix.py
│
├── analysis/                   # Results processing
│   └── analyze_results.ipynb
│
└── results/raw/                # Experiment outputs (not committed)
```

---

## Controlled Variables

All architectures share these identical configurations to isolate architectural overhead:

| Variable | Value | Purpose |
|----------|-------|---------|
| ML Models | YOLOv5n + MobileNetV2 (ONNX) | Byte-identical weights |
| Preprocessing | `common/preprocessing.py` | Eliminates preprocessing variance |
| Model Source | MinIO (S3-compatible) | All architectures download from same source |
| Container Resources | 2 vCPU, 4GB per container | Fair compute allocation |
| ONNX Threading | `intra_op=2`, `inter_op=1` | Optimal for 2 vCPU |
| Test Dataset | 100 COCO images (3-5 detections each) | Controlled fan-out |

---

## Quick Start

### Prerequisites

- Python 3.11 (managed via asdf)
- uv (Python package manager)
- Docker & Docker Compose

### Environment Setup

```bash
# Clone repository
git clone https://github.com/matthewhoung/inference-arena.git
cd inference-arena

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync

# Initialize project structure
python scripts/init_project.py
```

### Download Models

```bash
# Download ONNX models to models/ directory
python models/download_models.py
```

### Start Infrastructure

```bash
# Start MinIO + Prometheus + cAdvisor
docker compose -f infrastructure/docker-compose.infra.yml up -d

# Upload models to MinIO
python infrastructure/minio/init_models.py
```

### Run Single Architecture

```bash
# Example: Run monolithic architecture
docker compose -f architectures/monolithic/docker-compose.yml up -d

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -F "file=@data/thesis_test_set/000000001234.jpg"
```

### Run Experiments

```bash
# Run full experiment matrix (3 architectures × 7 load levels × 3 runs)
python experiments/run_matrix.py

# Results saved to results/raw/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NVIDIA Triton Inference Server team
- Ultralytics YOLOv5 team
- COCO dataset maintainers