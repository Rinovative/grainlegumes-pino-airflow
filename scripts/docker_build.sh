#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="grainlegumes-pino-airflow"

cd "$(dirname "$0")/.."

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .