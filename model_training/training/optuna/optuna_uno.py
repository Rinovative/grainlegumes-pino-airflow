"""Optuna hyperparameter optimisation for U-shaped Neural Operator (UNO)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import wandb
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import create_study

from training.train_uno import CONFIG as BASE_CONFIG
from training.train_uno import run_uno

if TYPE_CHECKING:
    from optuna.trial import Trial


# ================================================================
# Fixed, grid-safe UNO architecture presets
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
    """Optuna objective function for UNO hyperparameter optimization."""
    base_config = copy.deepcopy(BASE_CONFIG)
    CONFIG: dict[str, object] = dict(base_config)

    # ------------------------------------------------------------
    # Trial identity
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = "optuna_uno"
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

    # ------------------------------------------------------------
    # Hyperparameter search space
    # ------------------------------------------------------------
    preset = trial.suggest_categorical(
        "uno_preset",
        list(UNO_PRESETS.keys()),
    )

    CONFIG["uno_scalings"] = UNO_PRESETS[preset]
    CONFIG["n_layers"] = len(UNO_PRESETS[preset])

    CONFIG["base_modes"] = trial.suggest_categorical("base_modes", [32, 48, 64, 96, 128])
    CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])

    CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.2e-2, log=True)
    CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # ------------------------------------------------------------
    # Staged training (cheap → expensive)
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
                run_uno(CONFIG, manage_wandb=False)
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
                msg = "wandb.run is None after run_uno"
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

            # allow resume for next stage
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
        study_name="optuna_uno",
        direction="minimize",
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    study.optimize(
        objective,
        n_trials=30,
        show_progress_bar=False,
    )

    print("Best trial:")
    print(study.best_trial)
