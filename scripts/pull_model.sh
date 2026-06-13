#!/usr/bin/env bash
# Pull a VLM's weights to the EBS model volume on first VM boot (see §11.1/§11.2).
# Weights are NOT baked into the image; this runs once per VM, then the snapshot
# is reused. Swapping the model under test = pull a second model here and flip
# the MODEL env var in docker-compose.gpu.yml — no image rebuild.
#
# Usage: ./scripts/pull_model.sh Qwen/Qwen2.5-VL-7B-Instruct
set -euo pipefail

MODEL="${1:?usage: pull_model.sh <hf-model-id>}"
MODELS_DIR="${MODELS_DIR:-/models}"

mkdir -p "$MODELS_DIR"
echo "Pulling $MODEL into $MODELS_DIR ..."

# huggingface-cli is provided by the vllm image; run it there to avoid host deps.
docker run --rm \
    -v "$MODELS_DIR":/models \
    -e HF_HOME=/models/.hf \
    vllm/vllm-openai:latest \
    huggingface-cli download "$MODEL" --local-dir "/models/$(echo "$MODEL" | tr '/' '_')"

echo "Done. Set MODEL=$MODEL when launching docker-compose.gpu.yml."
