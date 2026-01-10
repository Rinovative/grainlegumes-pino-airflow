"""Optuna hyperparameter optimisation for Physics-Informed U-shaped Neural Operator (PI-UNO)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import torch
import wandb
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import create_study

from training.train_pi_uno import CONFIG as BASE_CONFIG
from training.train_pi_uno import run_pi_uno

if TYPE_CHECKING:
    from optuna.trial import Trial


# ================================================================
# Helper: build flexible symmetric U-shape scalings
# ================================================================
def build_uno_scalings(
    n_layers: int,
    bottleneck_scale: float,
    depth_steps: int,
    max_upscale: float,
) -> list[list[float]]:
    """Build U-shape scalings for UNO based on the number of layers and bottleneck parameters."""
    if n_layers < 3:  # noqa: PLR2004
        msg = "UNO needs at least 3 layers"
        raise ValueError(msg)

    mid = n_layers // 2

    # --- Down path ---
    down = [1.0]
    for i in range(1, mid + 1):
        factor = bottleneck_scale ** (i / depth_steps)
        down.append(factor)
    down[-1] = bottleneck_scale

    # --- Up path (mirror) ---
    up = down[:-1][::-1] if n_layers % 2 == 0 else down[::-1]
    scalings = down + up[1:]

    # --- Limit upscaling ---
    max_val = max(scalings)
    if max_val > 1.0:
        scale = min(max_upscale / max_val, 1.0)
        scalings = [s * scale for s in scalings]

    return [[float(s), float(s)] for s in scalings[:n_layers]]


# ================================================================
# 🎯 Optuna objective for PI-UNO
# ================================================================
def objective(trial: Trial) -> float:
    """Optuna objective function for PI-UNO hyperparameter optimization."""
    base_config = copy.deepcopy(BASE_CONFIG)
    CONFIG: dict[str, object] = dict(base_config)

    # ------------------------------------------------------------
    # Trial identity
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = "optuna_pi_uno"
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

    # ------------------------------------------------------------
    # 🔍 Hyperparameter search space
    # ------------------------------------------------------------

    # --- Architecture ---
    CONFIG["n_layers"] = trial.suggest_categorical("n_layers", [4, 5, 6])
    CONFIG["base_modes"] = trial.suggest_categorical("base_modes", [32, 48, 64])
    CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 96])

    # --- U-shape control ---
    CONFIG["bottleneck_scale"] = trial.suggest_float("bottleneck_scale", 0.25, 0.6)
    CONFIG["depth_steps"] = trial.suggest_int(
        "depth_steps",
        1,
        max(1, cast("int", CONFIG["n_layers"]) // 2),
    )
    CONFIG["max_upscale"] = trial.suggest_float("max_upscale", 1.0, 1.5)

    # --- Physics-informed weights ---
    CONFIG["lambda_phys"] = trial.suggest_float("lambda_phys", 1e-6, 5e-3, log=True)
    CONFIG["lambda_p"] = trial.suggest_float("lambda_p", 1e-4, 1e-2, log=True)

    # --- Optimisation ---
    CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.2e-2, log=True)

    # --- Batch size ---
    CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [8, 16])

    # ------------------------------------------------------------
    # 🔧 Derived architecture
    # ------------------------------------------------------------
    CONFIG["uno_scalings"] = build_uno_scalings(
        n_layers=cast("int", CONFIG["n_layers"]),
        bottleneck_scale=cast("float", CONFIG["bottleneck_scale"]),
        depth_steps=cast("int", CONFIG["depth_steps"]),
        max_upscale=cast("float", CONFIG["max_upscale"]),
    )

    # ------------------------------------------------------------
    # ⏱️ Staged training (cheap → expensive)
    # ------------------------------------------------------------
    budgets = [300, 400, 600]
    resume_dir: str | None = None
    last_value: float | None = None

    try:
        for n_epochs in budgets:
            CONFIG["n_epochs"] = n_epochs
            CONFIG["resume_from_dir"] = resume_dir

            # ------------------------
            # Run training stage
            # ------------------------
            try:
                run_pi_uno(CONFIG, manage_wandb=False)
            except RuntimeError as err:
                msg = str(err).lower()

                if "out of memory" in msg:
                    torch.cuda.empty_cache()
                    raise TrialPruned from None

                if "size of tensor" in msg and "must match the size of tensor" in msg:
                    raise TrialPruned from None

                raise

            # ------------------------
            # Read metric
            # ------------------------
            run = wandb.run
            if run is None:
                msg = "wandb.run is None after run_pi_uno"
                raise RuntimeError(msg)

            value = run.summary.get("eval_overall_rmse")
            if value is None:
                msg = "eval_overall_rmse not found in wandb summary"
                raise RuntimeError(msg)

            value = float(value)
            last_value = value

            # ------------------------
            # Report & prune
            # ------------------------
            trial.report(value, step=n_epochs)

            if trial.should_prune():
                raise TrialPruned

            resume_dir = "latest"

        if last_value is None:
            msg = "No valid objective value produced"
            raise RuntimeError(msg)

        return last_value

    finally:
        # --------------------------------------------------------
        # GUARANTEED cleanup (no zombie runs)
        # --------------------------------------------------------
        if wandb.run is not None:
            wandb.finish()


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    study = create_study(
        study_name="optuna_pi_uno",
        direction="minimize",
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    study.optimize(
        objective,
        n_trials=25,
        show_progress_bar=False,
    )

    print("Best trial:")
    print(study.best_trial)
