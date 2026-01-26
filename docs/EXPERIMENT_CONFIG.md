# Experiment Configuration Reference

Complete documentation for `experiment.yaml` - the single source of truth for all experimental parameters.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Section Reference](#section-reference)
  - [metadata](#metadata)
  - [research_questions](#research_questions)
  - [hypotheses](#hypotheses)
  - [independent_variables](#independent_variables)
  - [controlled_variables](#controlled_variables)
  - [services](#services)
  - [downloads](#downloads)
  - [infrastructure](#infrastructure)
  - [triton](#triton)
  - [changelog](#changelog)
- [Usage Examples](#usage-examples)
- [Modifying Configuration](#modifying-configuration)
- [Relationship with .env](#relationship-with-env)

---

## Overview

The `experiment.yaml` file serves as the **single source of truth** for all experimental parameters in the Inference Arena project. This design ensures:

1. **Reproducibility** - All experimental parameters are version-controlled in Git
2. **Pre-registration** - Hypotheses are defined before experiments run
3. **Consistency** - Code imports values from this file rather than hardcoding
4. **Traceability** - Git history proves when parameters were defined

**Key Principle:** No hardcoded experimental values in code. All models, preprocessing parameters, resource constraints, and test protocols are defined here and imported programmatically.

---

## File Structure

The configuration file is organized into logical sections:

```
experiment.yaml
├── metadata              # Thesis identification
├── research_questions    # RQ1-RQ4 definitions
├── hypotheses           # Pre-registered H1a-H3c
├── independent_variables # Variables being manipulated
├── controlled_variables  # Variables held constant
├── services             # Infrastructure port configuration
├── downloads            # Model download settings
├── infrastructure       # Docker/MinIO/network config
├── triton               # Triton-specific settings
└── changelog            # Version history
```

---

## Section Reference

### metadata

Identifies the thesis and author information.

| Field | Description |
|-------|-------------|
| `title` | Full thesis title |
| `subtitle` | Thesis subtitle |
| `author` | Researcher name |
| `institution` | University/organization |
| `degree` | Academic program |
| `created_at` | Initial creation date |
| `spec_version` | Specification version (semver) |

**Purpose:** Provides context for anyone reviewing the experimental design and ties the configuration to the academic work.

---

### research_questions

Defines the four research questions (RQ1-RQ4) that guide the experiments.

Each research question includes:

| Field | Description |
|-------|-------------|
| `title` | Short name (e.g., "Performance") |
| `question` | Full research question text |
| `metrics` | List of metrics that answer this question |

**Research Question Overview:**

| RQ | Focus | Key Metrics |
|----|-------|-------------|
| RQ1 | Performance | Latency (P50, P99), throughput, error rate |
| RQ2 | Resource Efficiency | CPU utilization, memory usage, cost |
| RQ3 | Operational Complexity | Code LOC, config LOC, deployment time |
| RQ4 | Decision Framework | Crossover points, decision matrix |

---

### hypotheses

Contains pre-registered hypotheses (H1a through H3c) that are tested by the experiments.

Each hypothesis includes:

| Field | Description |
|-------|-------------|
| `category` | Research area (performance, resource_efficiency, operational_complexity) |
| `statement` | Hypothesis claim |
| `rationale` | Why this is expected |
| `testable_prediction` | Concrete, measurable prediction |
| `conditions` | When this hypothesis applies |
| `null_hypothesis` | Alternative if hypothesis is rejected |

**Important:** The specific hypothesis content is protected thesis material. The structure documents that pre-registration occurred with Git commit history serving as proof.

---

### independent_variables

Variables being manipulated during experiments.

**Architecture (Categorical)**

The three ML serving architectures under evaluation:

| Level | Description |
|-------|-------------|
| `monolithic` | Single container with both detection and classification |
| `microservices` | Separate containers communicating via gRPC |
| `triton` | NVIDIA Triton Inference Server with gateway |

**Concurrent Users (Ordinal)**

Load levels tested: `[1, 5, 10, 25, 50, 75, 100]`

Represents simulated concurrent Locust users sending requests.

---

### controlled_variables

Variables held constant to ensure fair comparison. This is the most detailed section.

#### models

Specifies the two ML models used across all architectures:

**YOLOv5n (Object Detection)**
- Input shape: `[1, 3, 640, 640]`
- Output shape: `[1, 84, 8400]`
- ONNX opset version: 17
- Normalization: divide by 255

**MobileNetV2 (Image Classification)**
- Input shape: `[1, 3, 224, 224]`
- Output shape: `[1, 1000]`
- ONNX opset version: 17
- Normalization: ImageNet mean/std

Both models use identical ONNX exports across all architectures.

#### preprocessing

Defines preprocessing pipelines for each model type:

| Model | Method | Target Size |
|-------|--------|-------------|
| YOLO | letterbox | 640 |
| MobileNet | resize_and_normalize | 224 |

Implemented in `shared.processing` module.

#### resources

Container resource allocation (uniform across architectures):

| Resource | Value |
|----------|-------|
| vCPU per container | 2 |
| Memory per container | 4 GB |

Architecture-specific totals:

| Architecture | Containers | Total vCPU | Total Memory |
|--------------|------------|------------|--------------|
| Monolithic | 1 | 2 | 4 GB |
| Microservices | 2 | 4 | 8 GB |
| Triton | 2 | 4 | 8 GB |

#### onnx_runtime

ONNX Runtime configuration for consistent inference:

| Setting | Value |
|---------|-------|
| `intra_op_num_threads` | 2 |
| `inter_op_num_threads` | 1 |
| `graph_optimization_level` | ORT_ENABLE_ALL |
| `execution_mode` | ORT_SEQUENTIAL |
| `enable_cpu_mem_arena` | true |

#### dataset

Test dataset configuration:

| Field | Value |
|-------|-------|
| Source | COCO val2017 |
| Total images | 5000 |
| Sample size | 100 |
| Random seed | 42 |

The curated test set ensures consistent workload across experiments.

#### load_testing

Three-phase testing protocol:

| Phase | Duration | Purpose |
|-------|----------|---------|
| Warmup | 60s | Prime JIT, caches, optimizations |
| Measurement | 180s | Collect metrics under steady state |
| Cooldown | 30s | System reset, garbage collection |

**Runs per configuration:** 3 (statistical validity)

**Total experiment time:** ~4.7 hours for full matrix (3 architectures x 7 loads x 3 runs)

#### monitoring

Prometheus and OpenTelemetry configuration:

| Setting | Value |
|---------|-------|
| Scrape interval | 1s |
| Retention | 15 days |
| OTel port | 8889 |

#### container_names

Maps architecture names to container names for metric queries:

| Architecture | Containers |
|--------------|------------|
| Monolithic | inference-arena-monolithic |
| Microservices | inference-arena-detection, inference-arena-classification |
| Triton | inference-arena-triton-server, inference-arena-triton-gateway |

---

### services

Port configuration for infrastructure services.

| Service | Port | Purpose |
|---------|------|---------|
| MinIO API | 9000 | S3-compatible storage |
| MinIO Console | 9001 | Web interface |
| Prometheus | 9090 | Metrics database |
| Grafana | 3000 | Dashboards |
| OTel Collector | 8889 | Container metrics |
| OTel Health | 13133 | Health endpoint |

Used by test files and validation utilities.

---

### downloads

Parallel model download configuration:

| Setting | Value | Purpose |
|---------|-------|---------|
| `max_concurrent` | 3 | Maximum parallel downloads |
| `timeout` | 300 | Timeout per download (seconds) |

---

### infrastructure

Docker and network configuration:

**MinIO:**
- Internal endpoint: `minio:9000` (container-to-container)
- External endpoint: `localhost:9000` (host access)
- Default bucket: `models`

**Networks:**
- `inference-arena-backend` - Architecture containers
- `inference-arena-infra` - Infrastructure services

**Container Images:**
Lists specific versions for reproducibility (e.g., `minio/minio:RELEASE.2024-01-18T22-51-28Z`)

---

### triton

NVIDIA Triton Inference Server specific settings:

| Setting | Value |
|---------|-------|
| Model repository | S3://minio:9000/models |
| Instance count | 1 |
| Instance kind | KIND_CPU |
| Intra-op threads | 2 |
| Inter-op threads | 1 |

**Dynamic Batching:**

| Setting | Value |
|---------|-------|
| Enabled | false (default) |
| Max batch size | 8 |
| Preferred batch sizes | [4, 8] |
| Max queue delay | 5000 microseconds |

Batching is disabled by default to measure baseline overhead, then enabled for comparative analysis.

---

### changelog

Version history with justifications for any changes.

Each changelog entry includes:

| Field | Description |
|-------|-------------|
| `date` | When change was made |
| `version` | New version number |
| `change` | What changed |
| `justification` | Why it was necessary |
| `author` | Who made the change |
| `impact` | Effect on experiments |

**Important:** Any change after initial pre-registration requires a documented justification in the changelog.

---

## Usage Examples

### Python: Loading Configuration

```python
from shared.config import get_config, get_controlled_variable

# Load entire config
config = get_config()

# Access specific values
threads = get_controlled_variable("onnx_runtime", "intra_op_num_threads")
sample_size = get_controlled_variable("dataset", "sample_size")

# Access service ports
from shared.config import get_service_ports
ports = get_service_ports()
prometheus_port = ports.prometheus
```

### Python: Container Names

```python
from shared.config import get_container_names

containers = get_container_names("microservices")
# Returns: ["inference-arena-detection", "inference-arena-classification"]
```

### Verifying Configuration

```bash
# View modification history
git log --follow experiment.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('experiment.yaml'))"
```

---

## Modifying Configuration

### What CAN Be Changed

- **Infrastructure ports** (if conflicts occur locally)
- **Docker network names** (for isolation)
- **Prometheus retention** (for storage constraints)

### What Should NOT Be Changed

- **Model specifications** - Would invalidate comparisons
- **Preprocessing parameters** - Affects input consistency
- **Resource constraints** - Core controlled variables
- **Load testing protocol** - Statistical validity depends on consistency
- **ONNX runtime settings** - Affects inference behavior

### Change Protocol

1. Document in `changelog` section with justification
2. Commit with descriptive message
3. Note impact on any completed experiments

---

## Relationship with .env

| File | Purpose | Git Status |
|------|---------|------------|
| `experiment.yaml` | **What** to test (science) | Tracked |
| `.env` | **Where** to run (deployment) | Ignored |

**Example:**

```yaml
# experiment.yaml (tracked)
controlled_variables:
  resources:
    vcpu_per_container: 2
```

```ini
# .env (not tracked)
CONTAINER_VCPU=2
MINIO_ROOT_PASSWORD=my-secret-password
```

Both should align, but `.env` contains secrets that must not be committed.

---

## See Also

- **[ENVIRONMENT.md](ENVIRONMENT.md)** - Environment variable reference
- **[SETUP.md](SETUP.md)** - Quick start guide
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Load testing framework
- **[experiment.yaml](../experiment.yaml)** - The actual configuration file

---

*Documentation for Inference Arena v2.0*
*See experiment.yaml for authoritative configuration*
