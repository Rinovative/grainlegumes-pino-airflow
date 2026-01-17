"""Optuna objective function for UNO model training."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from training.optuna.optuna_setup import (
    run_optuna_study,
    run_staged_optuna_training,
)
from training.train_uno import CONFIG as BASE_CONFIG
from training.train_uno import run_uno

if TYPE_CHECKING:
    from optuna import Trial


# ================================================================
# 🏷️ UNO model naming
# ================================================================
def build_uno_run_name_from_config(CONFIG: dict) -> str:
    """Build a descriptive model name based on the configuration."""
    scaling_tag = "-".join(str(int(s[0]) if float(s[0]).is_integer() else s[0]).replace(".", "") for s in CONFIG["uno_scalings"])

    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    parts = [
        "UNO",
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"mr{CONFIG.get('mode_ratio', 0.5):.3g}".replace(".", "p"),
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
    """Optuna objective function for UNO model training."""
    CONFIG: dict[str, object] = dict(copy.deepcopy(BASE_CONFIG))

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
        CONFIG["uno_scalings"] = UNO_PRESETS["l5_s1-05-1-1-2"]
        CONFIG["n_layers"] = 5
        CONFIG["modes_x"] = 48
        CONFIG["modes_y"] = 48
        CONFIG["hidden_channels"] = 64
        CONFIG["lr"] = 5e-3
        CONFIG["batch_size"] = 32
        CONFIG["mode_ratio"] = 0.5
    else:
        # ------------------------------------------------------------
        # Search space
        # ------------------------------------------------------------
        preset = trial.suggest_categorical("uno_preset", list(UNO_PRESETS))
        CONFIG["uno_scalings"] = UNO_PRESETS[preset]
        CONFIG["n_layers"] = len(UNO_PRESETS[preset])
        CONFIG["modes_x"] = trial.suggest_categorical("modes_x", [24, 32, 48, 64, 96, 128])
        CONFIG["modes_y"] = trial.suggest_categorical("modes_y", [24, 32, 48, 64, 96, 128])
        CONFIG["mode_ratio"] = trial.suggest_float("mode_ratio", 0.1, 0.9)
        CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])
        CONFIG["lr"] = trial.suggest_float("lr", 1e-3, 1.5e-2, log=True)
        CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # ------------------------------------------------------------
    # 🏷️ FINAL run name
    # ------------------------------------------------------------
    CONFIG["model_name"] = build_uno_run_name_from_config(CONFIG)

    # ------------------------------------------------------------
    # 🚀 Staged Optuna training
    # ------------------------------------------------------------
    return run_staged_optuna_training(
        trial=trial,
        config=CONFIG,
        run_fn=run_uno,
        metric_key="eval_overall_rmse",
        budgets=[300, 400],
    )


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    run_optuna_study(
        objective=objective,
        study_name="optuna_UNO",
        n_trials=30,
    )
