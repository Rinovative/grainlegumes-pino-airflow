"""Optuna hyperparameter optimisation for Physics-Informed FNO (PI-FNO)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

from training.optuna.optuna_setup import (
    run_optuna_study,
    run_staged_optuna_training,
)
from training.train_pi_fno import CONFIG as BASE_CONFIG
from training.train_pi_fno import run_pi_fno

if TYPE_CHECKING:
    from optuna import Trial


# ================================================================
# 🏷️ PI-FNO model naming
# ================================================================
def build_pi_fno_run_name_from_config(CONFIG: dict) -> str:
    """Build PI-FNO model run name from configuration."""
    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    loss_tag = "PI-FNO-FFT" if CONFIG.get("pino_loss_type") == "fft" else "PI-FNO-PS"

    parts = [
        loss_tag,
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# ================================================================
# 🎯 Optuna objective
# ================================================================
def objective(trial: Trial) -> float:
    """Optuna objective function for PI-FNO hyperparameter optimization."""
    CONFIG: dict[str, object] = dict(copy.deepcopy(BASE_CONFIG))
    # fft or ps
    CONFIG["pino_loss_type"] = "fft"  # Set loss type for PI-FNO

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
        CONFIG["modes_x"] = 64
        CONFIG["modes_y"] = 64
        CONFIG["hidden_channels"] = 64
        CONFIG["n_layers"] = 4
        CONFIG["batch_size"] = 32
        CONFIG["lr"] = 1e-2
        CONFIG["lambda_phys"] = 1e-5
        CONFIG["lambda_p"] = 1e-4
        CONFIG["phys_warmup_epochs"] = 300
    else:
        # ------------------------------------------------------------
        # 🔍 Hyperparameter search space
        # ------------------------------------------------------------

        # --- Spectral resolution ---
        CONFIG["modes_x"] = trial.suggest_categorical("modes_x", [24, 32, 48, 64, 96, 128])
        CONFIG["modes_y"] = trial.suggest_categorical("modes_y", [24, 32, 48, 64, 96, 128])

        # --- Model capacity ---
        CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 96, 128])
        CONFIG["n_layers"] = trial.suggest_categorical("n_layers", [3, 4, 5, 6])

        # --- Optimisation ---
        CONFIG["lr"] = trial.suggest_float("lr", 1e-3, 1.5e-2, log=True)

        # --- Batch size ---
        CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

        # --- Physics-informed weights ---
        n_layers = cast("int", CONFIG["n_layers"])
        hidden = cast("int", CONFIG["hidden_channels"])
        m_x = cast("int", CONFIG["modes_x"])
        m_y = cast("int", CONFIG["modes_y"])
        capacity = n_layers * hidden * ((m_x + m_y) / 2)

        phys_ratio = trial.suggest_float("phys_ratio", 0.01, 100.0, log=True)
        p_ratio = trial.suggest_float("p_ratio", 0.01, 100.0, log=True)

        lambda_phys = float(phys_ratio) / float(capacity)
        lambda_p = float(p_ratio) * lambda_phys

        CONFIG["lambda_phys"] = lambda_phys
        CONFIG["lambda_p"] = lambda_p

        # 🔥 Physics warmup: capacity-aware (deterministic)
        warmup = int(30 * capacity**0.5)
        CONFIG["phys_warmup_epochs"] = int(min(max(warmup, 50), 600))

    # ------------------------------------------------------------
    # 🏷️ FINAL run name
    # ------------------------------------------------------------
    CONFIG["model_name"] = build_pi_fno_run_name_from_config(CONFIG)

    # ------------------------------------------------------------
    # 🚀 Run staged Optuna training
    # ------------------------------------------------------------
    return run_staged_optuna_training(
        trial=trial,
        config=CONFIG,
        run_fn=run_pi_fno,
        metric_key="eval_overall_rmse",
        budgets=[100, 200, 300, 400],
    )


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    run_optuna_study(
        objective=objective,
        study_name="optuna_PI-FNO-FFT",
        n_trials=30,
    )
