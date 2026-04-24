#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SCRIPT_NAME="$2"
LOG_BASENAME="$3"
shift 3

IMAGE_NAME="grainlegumes-pino-airflow"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="$(cd "${PROJECT_DIR}/../storage" && pwd)"
LOG_FILE="/workspace/repo/logs/${LOG_BASENAME}"

SCRIPT_RELATIVE_PATH="$(find "${PROJECT_DIR}" -path "${PROJECT_DIR}/.git" -prune -o -type f -name "${SCRIPT_NAME}" -print | head -n 1)"
if [ -z "${SCRIPT_RELATIVE_PATH}" ]; then
  echo "Script not found by name: ${SCRIPT_NAME}"
  exit 1
fi

SCRIPT_RELATIVE_PATH="${SCRIPT_RELATIVE_PATH#${PROJECT_DIR}/}"

WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY_VALUE="$(cat "${HOME}/wandb_key.txt")"
fi

docker run --rm \
  --gpus "\"device=${GPU_ID}\"" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e DATA_ROOT=/workspace/storage/data \
  -e GEN_ROOT=/workspace/storage/data_generation \
  -e TRAIN_ROOT=/workspace/storage/data_training \
  -e WANDB_API_KEY="${WANDB_API_KEY_VALUE}" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${IMAGE_NAME}" \
  bash -lc "mkdir -p /workspace/repo/logs && python '/workspace/repo/${SCRIPT_RELATIVE_PATH}' \"\$@\" > '${LOG_FILE}' 2>&1" -- "$@"