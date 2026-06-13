# syntax=docker/dockerfile:1
#
# One codebase, two build targets (see design §11.2):
#   dev → slim multi-arch base for the macOS laptop. Skeleton + LocalStack + tests.
#         No CUDA, no models.
#   gpu → CUDA runtime base for the EC2 g5.xlarge. Real detectors + vLLM sidecar.
#         Model weights are NOT baked in; they are mounted from an EBS volume.
#
# Build dev:  docker build --target dev -t cxpoc:dev .
# Build gpu:  docker build --target gpu -t cxpoc:gpu .

# ---------------------------------------------------------------------------
# dev target — runs on the developer laptop (multi-arch: arm64 + amd64)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS dev

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests ./tests

# VLMClient talks to a mock in dev — no model server required.
ENV VLM_BASE_URL=mock:// \
    VLM_MODEL=mock

ENTRYPOINT ["python", "-m", "pipeline.run"]

# ---------------------------------------------------------------------------
# gpu target — runs on the EC2 VM. CUDA base; heavy deps; models on a volume.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[gpu]"

# Model weights live on a mounted volume, not in the image (see §11.1/§11.2).
ENV MODELS_DIR=/models \
    VLM_BASE_URL=http://vllm:8000/v1 \
    VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

ENTRYPOINT ["python3", "-m", "pipeline.run"]
