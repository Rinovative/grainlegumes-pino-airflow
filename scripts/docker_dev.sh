#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="grainlegumes-pino-airflow"
CONTAINER_NAME="grainlegumes-pino-airflow-dev"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="$(cd "${PROJECT_DIR}/../storage" && pwd)"

mkdir -p "${PROJECT_DIR}/logs/docker"

docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e DATA_ROOT=/workspace/storage/data \
  -e GEN_ROOT=/workspace/storage/data_generation \
  -e TRAIN_ROOT=/workspace/storage/data_training \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${IMAGE_NAME}" \
  bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Stop with: docker stop ${CONTAINER_NAME}"