from __future__ import annotations

import copy

import torch
import wandb

import optuna
from model_training.training.train_fno import CONFIG as BASE_CONFIG
from model_training.training.train_fno import run_fno


# ================================================================
# 🎯 Optuna objective for FNO
# ================================================================
def objective(trial: optuna.Trial) -> float:
    CONFIG = copy.deepcopy(BASE_CONFIG)

    # ------------------------------------------------------------
    # Trial identity & output folder
    # ------------------------------------------------------------
    CONFIG["optuna_study_name"] = "optuna_fno"
    CONFIG["run_suffix"] = f"trial{trial.number}"
    CONFIG["resume_from_dir"] = None

    # ------------------------------------------------------------
    # 🔍 Hyperparameter search space
    # ------------------------------------------------------------

    # --- Spectral resolution (KEY parameter) ---
    base_modes = trial.suggest_categorical("base_modes", [32, 48, 64, 96, 128])
    mode_ratio = trial.suggest_categorical("mode_ratio", [1.0, 1.25, 1.5])

    CONFIG["n_modes"] = (
        base_modes,
        int(base_modes * mode_ratio),
    )

    # --- Model capacity (secondary) ---
    CONFIG["hidden_channels"] = trial.suggest_categorical("hidden_channels", [64, 96, 128])

    CONFIG["n_layers"] = trial.suggest_categorical("n_layers", [4, 5, 6])

    # --- Optimisation ---
    CONFIG["lr"] = trial.suggest_float("lr", 3e-3, 1.5e-2, log=True)

    # --- Batch size (discrete, memory-aware) ---
    CONFIG["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # ------------------------------------------------------------
    # ⏱️ Staged training budgets
    # ------------------------------------------------------------
    budgets = [300, 600, 1000]

    best_value: float | None = None
    resume_dir: str | None = None

    for n_epochs in budgets:
        CONFIG["n_epochs"] = n_epochs
        CONFIG["resume_from_dir"] = resume_dir

        try:
            run_fno(CONFIG)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        # --------------------------------------------------------
        # Read metric from wandb
        # --------------------------------------------------------
        value = wandb.run.summary.get("eval_overall_rmse")
        if value is None:
            msg = "eval_overall_rmse not found in wandb summary"
            raise RuntimeError(msg)

        value = float(value)

        # --------------------------------------------------------
        # Early stopping logic
        # --------------------------------------------------------
        if best_value is None:
            best_value = value
        else:
            self_improvement = (best_value - value) / max(best_value, 1e-8)
            global_best = trial.study.best_value

            gap_to_best = (value - global_best) / max(abs(global_best), 1e-8) if global_best is not None else 0.0

            # schwach + klar schlechter -> abbrechen
            if self_improvement < 0.02 and gap_to_best > 0.10:  # noqa: PLR2004
                break

            best_value = value

        # --------------------------------------------------------
        # Continue same run (single GPU safe)
        # --------------------------------------------------------
        resume_dir = "latest"
        trial.report(value, step=n_epochs)

    assert best_value is not None
    return best_value


# ================================================================
# 🧪 Study launcher
# ================================================================
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="optuna_fno",
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
    )

    study.optimize(
        objective,
        n_trials=25,
        show_progress_bar=True,
    )

    print("Best trial:")
    print(study.best_trial)
