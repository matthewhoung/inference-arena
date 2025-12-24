# Quick Start Guide - Inference Arena

## Starting Experiments (One-Command)

### Monolithic Architecture
```bash
./scripts/start-monolithic.sh
```

### Microservices Architecture
```bash
./scripts/start-microservices.sh
```

### Triton Architecture
```bash
./scripts/start-triton.sh
```

**What these scripts do:**
1. ✅ Start infrastructure (Prometheus, Grafana, cAdvisor)
2. ✅ Start your architecture containers
3. ✅ Wait for containers to be healthy
4. ✅ **Automatically update Grafana dashboards** with current container IDs
5. ✅ Restart Grafana to load updates
6. ✅ Display status and URLs

---

## 📊 Container Naming Conventions

### Monolithic
- `inference-arena-monolithic` - Single unified service

### Microservices
- `micro-detect` - Detection service (YOLOv5n)
- `micro-classify` - Classification service (MobileNetV2)

### Triton
- `triton-gateway` - API Gateway (HTTP → gRPC)
- `triton-server` - Triton Inference Server

**These names are:**
- ✅ **Thesis-appropriate**: Professional academic naming
- ✅ **Self-documenting**: Component role is clear
- ✅ **Consistent**: `architecture-component` pattern
- ✅ **Scalable**: Easy to add components

---

## 🎯 Access Your Dashboards

**Grafana**: http://localhost:3000 (admin/admin)

**Available Dashboards:**
- Inference Arena - Monolithic
- Inference Arena - Microservices
- Inference Arena - Triton

## 📝 Example Experiment Workflow

```bash
# 1. Start architecture
./scripts/start-monolithic.sh

# Output:
# Starting Monolithic Architecture...
# Starting infrastructure services...
# Starting monolithic container...
# Auto-updating Grafana dashboards...
# Monolithic architecture is ready!
# Grafana Dashboard: http://localhost:3000

# 2. Run load tests (Locust, etc.)
# ...

# 3. View real-time metrics in Grafana
open http://localhost:3000

# 4. Stop when done
docker-compose -f architectures/monolithic/docker-compose.yml down
```

---

## 🛑 Stopping Experiments

### Stop Specific Architecture
```bash
# Monolithic
docker-compose -f architectures/monolithic/docker-compose.yml down

# Microservices
docker-compose -f architectures/microservices/docker-compose.yml down

# Triton
docker-compose -f architectures/triton/docker-compose.yml down
```

### Stop Everything (including infrastructure)
```bash
docker-compose -f infrastructure/docker-compose.infra.yml down
docker-compose -f architectures/*/docker-compose.yml down 2>/dev/null || true
```

**Author**: Matthew Hong
**Project**: Inference Arena - Master's Thesis
**Last Updated**: 2025-12-23
