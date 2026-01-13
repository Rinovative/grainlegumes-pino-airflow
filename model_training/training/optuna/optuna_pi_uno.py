"""Optuna hyperparameter optimisation for Physics-Informed U-shaped Neural Operator (PI-UNO)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

from training.optuna.optuna_setup import (
    run_optuna_study,
    run_staged_optuna_training,
)
from training.train_pi_uno import CONFIG as BASE_CONFIG
from training.train_pi_uno import run_pi_uno

if TYPE_CHECKING:
    from optuna import Trial


# ================================================================
# 🏷️ PI-UNO model naming
# ================================================================
def build_pi_uno_run_name_from_config(CONFIG: dict) -> str:
    """Build a descriptive PI-UNO run name based on the configuration."""
    scaling_tag = "-".join(str(int(s[0]) if float(s[0]).is_integer() else s[0]).replace(".", "") for s in CONFIG["uno_scalings"])

    parts = [
        "PI-UNO",
        f"m{CONFIG['base_modes']}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# ================================================================
# Grid-safe UNO architecture presets
# ================================================================
UNO_PRESETS: dict[str, list[list[float]]] = {
    "l4_s1-05-1-2": [
        [1.0, 1.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 2.0],
    ],
    "l5_s1-05-1-1-2": [
        [1.0, 1.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ],
    "l6_s1-05-05-1-2-2": [
        [1.0, 1.0],
        [0.5, 0.5],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0],
    ],
}


# ================================================================
# 🎯 Optuna objective
# ================================================================
def objective(trial: Trial) -> float:
    """Optuna objective function for PI-UNO hyperparameter optimization."""
    CONFIG: dict[str, object] = dict(copy.deepcopy(BASE_CONFIG))

    # ------------------------------------------------------------
    # Trial identity
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = trial.study.study_name
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

    # ------------------------------------------------------------
    # 🚑 Bootstrap trial (stabile Referenz gegen Serien-OOM)
    # ------------------------------------------------------------
    if trial.number == 0:
        CONFIG["uno_scalings"] = UNO_PRESETS["l5_s1-05-1-1-2"]
        CONFIG["n_layers"] = 5
        CONFIG["base_modes"] = 48
        CONFIG["hidden_channels"] = 64
        CONFIG["lr"] = 5e-3
        CONFIG["batch_size"] = 32
        CONFIG["lambda_phys"] = 1e-4
        CONFIG["lambda_p"] = 1e-4
        CONFIG["phys_warmup_epochs"] = 300
    else:
        # ------------------------------------------------------------
        # 🔍 Hyperparameter search space
        # ------------------------------------------------------------

        # --- Grid-safe UNO presets ---
        preset = trial.suggest_categorical("uno_preset", list(UNO_PRESETS))
        CONFIG["uno_scalings"] = UNO_PRESETS[preset]
        CONFIG["n_layers"] = len(UNO_PRESETS[preset])

        CONFIG["base_modes"] = trial.suggest_categorical("base_modes", [32, 48, 64, 96, 128])
        CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])

        # --- Optimisation ---
        CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.2e-2, log=True)

        # --- Batch size ---
        CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

        # --- Physics-informed weights ---
        n_layers = cast("int", CONFIG["n_layers"])
        hidden = cast("int", CONFIG["hidden_channels"])
        base_modes = cast("int", CONFIG["base_modes"])

        # → capacity-aware lambda_phys
        capacity = n_layers * hidden * base_modes
        phys_ratio = trial.suggest_float("phys_ratio", 0.5, 2.0, log=True)
        p_ratio = trial.suggest_float("p_ratio", 0.5, 20.0, log=True)

        lambda_phys = phys_ratio / capacity
        lambda_p = p_ratio * lambda_phys

        CONFIG["lambda_phys"] = lambda_phys
        CONFIG["lambda_p"] = lambda_p

        # 🔥 Physics warmup: capacity-aware (deterministic)
        warmup = int(30 * capacity**0.5)
        CONFIG["phys_warmup_epochs"] = int(min(max(warmup, 50), 300))

    # ------------------------------------------------------------
    # 🏷️ FINAL run name (must be set BEFORE wandb.init)
    # ------------------------------------------------------------
    CONFIG["model_name"] = build_pi_uno_run_name_from_config(CONFIG)

    # ------------------------------------------------------------
    # 🚀 Run staged Optuna training
    # ------------------------------------------------------------
    return run_staged_optuna_training(
        trial=trial,
        config=CONFIG,
        run_fn=run_pi_uno,
        metric_key="eval_overall_rmse",
        budgets=[50, 100, 200, 300, 400, 600],
    )


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    run_optuna_study(
        objective=objective,
        study_name="optuna_pi_uno",
        n_trials=30,
    )
