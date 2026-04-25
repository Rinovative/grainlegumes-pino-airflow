#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SCRIPT_NAME="$2"
LOG_BASENAME="$3"
shift 3

IMAGE_NAME="grainlegumes-pino-airflow"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="$(cd "${PROJECT_DIR}/../storage" && pwd)"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"
LOG_FILE="/workspace/repo/logs/${LOG_BASENAME}"

mkdir -p \
  "${PROJECT_DIR}/logs" \
  "${PROJECT_DIR}/data" \
  "${PROJECT_DIR}/data_generation/data" \
  "${PROJECT_DIR}/model_training/data" \
  "${STORAGE_DIR}/data" \
  "${STORAGE_DIR}/data_generation" \
  "${STORAGE_DIR}/data_training" \
  "${DOCKER_HOME}"

# ----------------------------------------------------------------------
# Create runtime user mapping for container
# ----------------------------------------------------------------------
cat > "${DOCKER_HOME}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
EOF

cat > "${DOCKER_HOME}/group" <<EOF
root:x:0:
rino:x:$(id -g):
EOF

chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

# ----------------------------------------------------------------------
# Resolve script by filename
# ----------------------------------------------------------------------
SCRIPT_ABS_PATH="$(
  find "${PROJECT_DIR}" \
    -path "${PROJECT_DIR}/.git" -prune -o \
    -path "${PROJECT_DIR}/logs" -prune -o \
    -type f -name "${SCRIPT_NAME}" -print \
    | head -n 1
)"

if [ -z "${SCRIPT_ABS_PATH}" ]; then
  echo "Script not found by name: ${SCRIPT_NAME}"
  exit 1
fi

SCRIPT_RELATIVE_PATH="${SCRIPT_ABS_PATH#${PROJECT_DIR}/}"

# ----------------------------------------------------------------------
# Load W&B key if available
# ----------------------------------------------------------------------
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY_VALUE="$(cat "${HOME}/wandb_key.txt")"
fi

# ----------------------------------------------------------------------
# Optional SSH mount for Git operations
# ----------------------------------------------------------------------
SSH_ARGS=()
if [ -d "${HOME}/.ssh" ]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

# ----------------------------------------------------------------------
# Run queued job inside Docker
# ----------------------------------------------------------------------
docker run --rm \
  --gpus "\"device=${GPU_ID}\"" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
  -e HOME=/workspace/storage/.docker_home \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e DATA_ROOT=/workspace/storage/data \
  -e GEN_ROOT=/workspace/storage/data_generation \
  -e TRAIN_ROOT=/workspace/storage/data_training \
  -e WANDB_API_KEY="${WANDB_API_KEY_VALUE}" \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  -v "${STORAGE_DIR}/data:/workspace/repo/data:rw" \
  -v "${STORAGE_DIR}/data_generation:/workspace/repo/data_generation/data:rw" \
  -v "${STORAGE_DIR}/data_training:/workspace/repo/model_training/data:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -lc "mkdir -p /workspace/repo/logs && python '/workspace/repo/${SCRIPT_RELATIVE_PATH}' \"\$@\" > '${LOG_FILE}' 2>&1" -- "$@"