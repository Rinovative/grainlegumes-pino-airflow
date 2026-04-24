#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="${1:-model_training/src/util/util_gpu_test.py}"
shift || true

SCRIPT_HOST_PATH="${PROJECT_DIR}/${SCRIPT_PATH}"
if [ ! -f "${SCRIPT_HOST_PATH}" ]; then
  echo "Script not found: ${SCRIPT_HOST_PATH}"
  exit 1
fi

SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"

echo "Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
echo "------------------------------------------------------------"

AUTO_GPU=$(nvidia-smi --query-gpu=index,memory.used \
  --format=csv,noheader,nounits \
  | sort -t, -k2 -n \
  | head -n1 \
  | cut -d',' -f1 \
  | xargs)

read -r -p "Select GPU (0-3, press Enter for ${AUTO_GPU}): " GPU_ID
GPU_ID="${GPU_ID:-$AUTO_GPU}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_TAG="$(basename "${SCRIPT_NAME}" .py)"
LOG_BASENAME="${TIMESTAMP}__${SCRIPT_TAG}__gpu${GPU_ID}.log"

cd "${PROJECT_DIR}"
runTSGPU.py -g"${GPU_ID}" -- scripts/docker_run.sh \
  "${GPU_ID}" \
  "${SCRIPT_NAME}" \
  "${LOG_BASENAME}" \
  "$@"

echo "Queued Docker job on GPU ${GPU_ID}: ${SCRIPT_NAME}"
echo "Queue: runTSGPU.py -g${GPU_ID} -s"
echo "Tail:  tail -f ${LOG_DIR}/${LOG_BASENAME}"