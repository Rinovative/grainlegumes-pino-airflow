"""Configuration paths and constants for model training."""

from pathlib import Path

from src import common

# ==============================
# Project root and data paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths resolved through common.paths for reproducibility
DATA_PROCESSED = common.paths.get_train_root() / "runs"
TRAIN_DATA_RAW = common.paths.get_train_root()


# ==============================
# W&B configuration
# ==============================
WANDB_PROJECT = "grainlegumes_pino"
WANDB_ENTITY = "Rinovative-Hub"

WANDB_BASE_DIR = DATA_PROCESSED
