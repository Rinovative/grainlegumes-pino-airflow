"""Configuration paths and constants for model training."""

from pathlib import Path

# ==============================
# Project root and data paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = PROJECT_ROOT / "model_training" / "data_temp"
TRAIN_DATA_RAW = PROJECT_ROOT / "model_training" / "data" / "raw"

# ==== TEMPORARY: DOCKER-GPU NOT AVAILABLE ========================
# ==== TEMPORARY: DOCKER-GPU NOT AVAILABLE ========================
TMP_ROOT = Path("/home/rino.albertin/workspace/tmp_data/temp_data_training")
TRAIN_DATA_RAW = TMP_ROOT / "temp_data" / "temp_raw"
# ==== TEMPORARY: DOCKER-GPU NOT AVAILABLE ========================
# ==== TEMPORARY: DOCKER-GPU NOT AVAILABLE ========================


# ==============================
# W&B configuration
# ==============================
WANDB_PROJECT = "grainlegumes_pino"
WANDB_ENTITY = "Rinovative-Hub"

WANDB_BASE_DIR = DATA_PROCESSED
