"""Optuna objective function for FNO model training."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from training.optuna.optuna_setup import (
    run_optuna_study,
    run_staged_optuna_training,
)
from training.train_fno import CONFIG as BASE_CONFIG
from training.train_fno import run_fno

if TYPE_CHECKING:
    from optuna import Trial


# ================================================================
# 🏷️ FNO model naming
# ================================================================
def build_fno_run_name_from_config(CONFIG: dict) -> str:
    """Build a descriptive FNO run name based on the configuration."""
    n_modes = CONFIG["n_modes"]
    hidden = CONFIG["hidden_channels"]
    layers = CONFIG["n_layers"]

    parts = [
        "FNO",
        f"m{n_modes[0]}x{n_modes[1]}",
        f"h{hidden}",
        f"l{layers}",
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# ================================================================
# 🎯 Optuna objective
# ================================================================
def objective(trial: Trial) -> float:
    """Optuna objective function for FNO model training."""
    _raw_config = copy.deepcopy(BASE_CONFIG)
    CONFIG: dict[str, object] = dict(_raw_config)

    # ------------------------------------------------------------
    # Trial identity
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = trial.study.study_name
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

    # ------------------------------------------------------------
    # 🚑 Bootstrap trial
    # ------------------------------------------------------------
    if trial.number == 0:
        CONFIG["n_modes"] = (128, 160)
        CONFIG["hidden_channels"] = 64
        CONFIG["n_layers"] = 3
        CONFIG["batch_size"] = 32
        CONFIG["lr"] = 5e-3
    else:
        # ------------------------------------------------------------
        # 🔍 Hyperparameter search space
        # ------------------------------------------------------------

        # --- Spectral resolution ---
        base_modes = trial.suggest_categorical("base_modes", [32, 48, 64, 96, 128])
        mode_ratio = trial.suggest_categorical("mode_ratio", [1.0, 1.25, 1.5])
        CONFIG["n_modes"] = (base_modes, int(base_modes * mode_ratio))

        # --- Model capacity ---
        CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 96, 128])
        CONFIG["n_layers"] = trial.suggest_categorical("n_layers", [3, 4, 5, 6])

        # --- Optimization ---
        CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.5e-2, log=True)

        # --- Batch size ---
        CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # ------------------------------------------------------------
    # 🏷️ FINAL run name
    # ------------------------------------------------------------
    CONFIG["model_name"] = build_fno_run_name_from_config(CONFIG)

    # ------------------------------------------------------------
    # 🚀 Run staged Optuna training
    # ------------------------------------------------------------
    return run_staged_optuna_training(
        trial=trial,
        config=CONFIG,
        run_fn=run_fno,
        metric_key="eval_overall_rmse",
        budgets=[300, 400, 600],
    )


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    run_optuna_study(
        objective=objective,
        study_name="optuna_fno",
        n_trials=30,
    )
