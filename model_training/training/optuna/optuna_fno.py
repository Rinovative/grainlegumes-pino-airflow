"""Optuna hyperparameter optimisation for Fourier Neural Operator (FNO)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import wandb
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import create_study

from training.train_fno import CONFIG as BASE_CONFIG
from training.train_fno import run_fno

if TYPE_CHECKING:
    from optuna.trial import Trial


# ================================================================
# 🎯 Optuna objective for FNO
# ================================================================
def objective(trial: Trial) -> float:
    """Optuna objective function for FNO hyperparameter optimization."""
    base_config = copy.deepcopy(BASE_CONFIG)
    CONFIG: dict[str, object] = dict(base_config)

    # ------------------------------------------------------------
    # Trial identity
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = "optuna_fno"
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

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

    # --- Optimisation ---
    CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.5e-2, log=True)

    # --- Batch size ---
    CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

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
                run_fno(CONFIG, manage_wandb=False)
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
                msg = "wandb.run is None after run_fno"
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
        study_name="optuna_fno",
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
