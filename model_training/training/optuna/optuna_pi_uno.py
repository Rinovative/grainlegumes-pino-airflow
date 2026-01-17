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

    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    loss_tag = "PI-UNO-FFT" if CONFIG.get("pino_loss_type") == "fft" else "PI-UNO-PS"

    parts = [
        loss_tag,
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"mr{CONFIG.get('mode_ratio', 0.5):.3g}".replace(".", "p"),
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
        CONFIG["train_dataset_name"],
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# ================================================================
# Grid-safe UNO architecture presets
# ================================================================
UNO_PRESETS: dict[str, list[list[float]]] = {
    "l5_s1-05-1-1-2": [
        [1.0, 1.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ],
    "l7_s1-05-05-1-1-2-2": [
        [1.0, 1.0],
        [0.5, 0.5],
        [0.5, 0.5],
        [1.0, 1.0],
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
    # fft or ps
    CONFIG["pino_loss_type"] = "ps"  # Set loss type for PI-FNO

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
        CONFIG["n_layers"] = 7
        CONFIG["hidden_channels"] = 32
        CONFIG["modes_x"] = 64
        CONFIG["modes_y"] = 64
        CONFIG["mode_ratio"] = 0.4951318201313778
        CONFIG["uno_scalings"] = [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ]

        CONFIG["lr"] = 0.008511542981479816
        CONFIG["batch_size"] = 32

        CONFIG["lambda_phys"] = 0.0001
        CONFIG["lambda_p"] = 0.0005
        CONFIG["phys_warmup_epochs"] = 300
    else:
        # --- Fixe Architektur (identisch zu Bootstrap) ---
        CONFIG["n_layers"] = 7
        CONFIG["hidden_channels"] = 32
        CONFIG["modes_x"] = 64
        CONFIG["modes_y"] = 64
        CONFIG["mode_ratio"] = 0.4951318201313778
        CONFIG["uno_scalings"] = [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ]

        CONFIG["lr"] = 0.008511542981479816
        CONFIG["batch_size"] = 32

        # --- Physics-informed weights ONLY ---
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
    CONFIG["model_name"] = build_pi_uno_run_name_from_config(CONFIG)

    # ------------------------------------------------------------
    # 🚀 Run staged Optuna training
    # ------------------------------------------------------------
    return run_staged_optuna_training(
        trial=trial,
        config=CONFIG,
        run_fn=run_pi_uno,
        metric_key="eval_overall_rmse",
        budgets=[100, 200, 300, 400],
    )


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    run_optuna_study(
        objective=objective,
        study_name="optuna_PI-UNO-PS",
        n_trials=30,
    )
