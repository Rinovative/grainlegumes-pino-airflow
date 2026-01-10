#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash hpc_train_nodocker.sh <script.py>"
    exit 1
fi

SCRIPT_FILE="$1"

echo "📄 Training script: model_training/training/${SCRIPT_FILE}"
echo ""

# ------------------------------------------------------------
# Activate environment
# ------------------------------------------------------------
source ~/workspace/grainlegumes-pino/mlenv/bin/activate

# ------------------------------------------------------------
# Load W&B key
# ------------------------------------------------------------
export WANDB_API_KEY=$(cat ~/wandb_key.txt)

# ------------------------------------------------------------
# Move into correct project root for imports
# ------------------------------------------------------------
cd ~/workspace/grainlegumes-pino/model_training

# ------------------------------------------------------------
# Fix Python import root
# ------------------------------------------------------------
export PYTHONPATH=$(pwd)

# ------------------------------------------------------------
# Set dataset root (external storage)
# ------------------------------------------------------------
export DATA_ROOT=/home/rino.albertin/workspace/tmp_data/model_training/data/raw

# ------------------------------------------------------------
# Show GPU status
# ------------------------------------------------------------
echo "📊 Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
           --format=csv,noheader,nounits
echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Select GPU automatically
# ------------------------------------------------------------
auto_gpu=$(nvidia-smi --query-gpu=index,memory.used \
                      --format=csv,noheader,nounits \
                      | sort -t, -k2 -n | head -n1 | cut -d',' -f1)

read -p "Select GPU (0–3, press Enter for ${auto_gpu}): " GPU_ID
GPU_ID=${GPU_ID:-$auto_gpu}

echo ""
echo "➡️  Starting training on GPU $GPU_ID (queued via runTSGPU)"
echo ""

# ------------------------------------------------------------
# Launch via GPU queue
# ------------------------------------------------------------
runTSGPU.py -g$GPU_ID -- python training/"$SCRIPT_FILE"
