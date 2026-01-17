"""Module for setting up Optuna training with pruning support."""

from __future__ import annotations

import contextlib
import gc
import json
import multiprocessing as mp
import os
from typing import TYPE_CHECKING

import torch
import wandb
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.study import Study, create_study

from training import DATA_PROCESSED, WANDB_BASE_DIR, WANDB_ENTITY, WANDB_PROJECT

if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna import Trial

# ================================================================
# 🧮 GPU memory estimation limit
# ================================================================

GPU_LIMIT_MB = 30_000


# ================================================================
# 🧮 Deterministic OOM precheck (FNO / PI-FNO)
# ================================================================
def estimate_fno_memory_mb(*, config: dict, safety: float = 2.0) -> float:
    """
    Estimate the GPU memory consumption of an FNO / PI-FNO model in MB.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing model parameters.
    safety : float, optional
        A safety factor to account for overhead. Default is 2.0.

    Returns
    -------
    float
        Estimated memory consumption in MB.

    """
    batch = int(config["batch_size"])
    hidden = int(config["hidden_channels"])
    layers = int(config["n_layers"])
    mx = int(config["modes_x"])
    my = int(config["modes_y"])

    bytes_per_float = 4  # float32

    # activation memory (dominant term)
    activations = batch * hidden * mx * my * layers

    # parameter memory (rough, conservative)
    params = hidden * hidden * layers

    total_bytes = (activations + params) * bytes_per_float
    return safety * total_bytes / 1024**2  # MB


def _finalize_wandb_run(
    *,
    status: str,
    resume_dir: str | None,
    study_name: str,
) -> None:
    """
    Finalize the W&B run by updating the config with the true configuration.

    Parameters
    ----------
    status : str
        The final status to set in the W&B config.
    resume_dir : str | None
        The directory name of the run to resume from.
    study_name : str
        The name of the Optuna study.

    """
    if wandb.run is None or resume_dir is None:
        return

    try:
        run_dir = DATA_PROCESSED / study_name / resume_dir
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                true_cfg = json.load(f)
            wandb.run.config.update(true_cfg, allow_val_change=True)

        wandb.run.config.update({"status": status}, allow_val_change=True)

    except Exception:  # noqa: BLE001, S110
        pass


# ================================================================
# 🚀 Staged Optuna training (1 W&B run per trial)
# ================================================================
def run_staged_optuna_training(
    *,
    trial: Trial,
    config: dict,
    run_fn: Callable[..., None],
    metric_key: str,
    budgets: list[int],
) -> float:
    """
    Run staged training with Optuna pruning support.

    Parameters
    ----------
    trial : Trial
        The Optuna trial object.
    config : dict
        The configuration dictionary for training.
    run_fn : Callable[..., None]
        The training function to be called.
    metric_key : str
        The key of the metric to monitor for pruning.
    budgets : list[int]
        List of epoch budgets for each training stage.

    Returns
    -------
    float
        The final metric value after training.

    """
    # ------------------------------------------------------------
    # 🧮 Capacity + memory logging (for DB analysis)
    # ------------------------------------------------------------
    capacity = config["batch_size"] * config["hidden_channels"] * config["n_layers"] * config["modes_x"] * config["modes_y"]

    trial.set_user_attr("capacity", int(capacity))

    est_mem = estimate_fno_memory_mb(config=config)
    trial.set_user_attr("est_mem_mb", float(est_mem))

    # ------------------------------------------------------------
    # 🚨 OOM PRE-CHECK (before wandb / cuda / training)
    # ------------------------------------------------------------
    if est_mem > GPU_LIMIT_MB:
        msg = f"OOM precheck: ~{est_mem:.0f} MB"
        raise TrialPruned(msg)

    # ------------------------
    # Trial grouping metadata
    # ------------------------
    trial.set_user_attr("study_name", config["optuna_study_name"])

    # ------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY

    wandb_dir = WANDB_BASE_DIR / config["optuna_study_name"]
    wandb_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(wandb_dir)
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=config["model_name"],
        dir=os.environ["WANDB_DIR"],
        reinit=True,
        tags=[config["optuna_study_name"].removeprefix("optuna_"), "optuna"],
    )

    wandb.config.update({"status": "initialised"}, allow_val_change=True)

    resume_dir: str | None = None
    last_value: float | None = None

    try:
        for stage_idx, n_epochs in enumerate(budgets):
            is_last_stage = stage_idx == len(budgets) - 1

            config["n_epochs"] = n_epochs
            config["resume_from_dir"] = resume_dir

            # --------------------------------------------
            # Checkpointing policy (STAGE-DEPENDENT)
            # --------------------------------------------
            if is_last_stage:
                config["save_every"] = None
                config["save_best"] = metric_key
            else:
                config["save_every"] = 5  # eval_interval
                config["save_best"] = None

            # ------------------------
            # Run training stage
            # ------------------------
            try:
                run_fn(config, manage_wandb=False)
            except RuntimeError as err:
                msg = str(err).lower()

                if "out of memory" in msg:
                    _finalize_wandb_run(
                        status="oom_pruned",
                        resume_dir=resume_dir,
                        study_name=config["optuna_study_name"],
                    )
                    torch.cuda.empty_cache()
                    raise TrialPruned from None

                raise

            # ------------------------
            # Read metric from W&B
            # ------------------------
            run = wandb.run
            if run is None:
                msg = "wandb.run is None after training"
                raise RuntimeError(msg)

            value = run.summary.get(metric_key)
            if value is None:
                msg = f"{metric_key} not found in wandb summary"
                raise RuntimeError(msg)

            value = float(value)
            if not torch.isfinite(torch.tensor(value)):
                msg = "NaN or Inf detected after stage"
                _finalize_wandb_run(
                    status="nan_pruned",
                    resume_dir=resume_dir,
                    study_name=config["optuna_study_name"],
                )
                raise TrialPruned(msg)

            last_value = value

            # ------------------------
            # Report to Optuna
            # ------------------------
            trial.report(value, step=stage_idx)

            # ------------------------------------------------------------
            # 💥 Hard prune on sudden spike
            # ------------------------------------------------------------
            prev = trial.user_attrs.get("last_stage_metric", None)

            spike_factor = 8.0 if stage_idx == 0 else 4.0

            if prev is not None and value > spike_factor * prev:
                msg = f"Stage exploded: {value:.3e} > {spike_factor} x prev {prev:.3e}"
                _finalize_wandb_run(
                    status="hard_pruned",
                    resume_dir=resume_dir,
                    study_name=config["optuna_study_name"],
                )
                raise TrialPruned(msg)

            trial.set_user_attr("last_stage_metric", value)

            # ------------------------------------------------------------
            # 🔪 Standard Optuna pruning
            # ------------------------------------------------------------
            if trial.should_prune():
                msg = f"Pruned by Optuna at stage {stage_idx}"
                _finalize_wandb_run(
                    status="optuna_pruned",
                    resume_dir=resume_dir,
                    study_name=config["optuna_study_name"],
                )
                raise TrialPruned(msg)

            # ------------------------------------------------------------
            # 🔁 Set resume_dir to the actual run directory of THIS trial
            # ------------------------------------------------------------
            base = DATA_PROCESSED / config["optuna_study_name"]

            runs = sorted(p for p in base.iterdir() if p.is_dir() and p.name != "wandb")

            if not runs:
                msg = "No run directory found after training stage"
                raise RuntimeError(msg)

            latest_run = max(runs, key=lambda p: p.stat().st_mtime)
            resume_dir = latest_run.name

            # ------------------------------------------------------------
            # 🔄 After stage 0: rewrite W&B config with final config
            # ------------------------------------------------------------
            if stage_idx == 0 and wandb.run is not None:
                _finalize_wandb_run(
                    status="running",
                    resume_dir=resume_dir,
                    study_name=config["optuna_study_name"],
                )

        # ------------------------------------------------------------
        # 🔁 Final W&B config sync (BOUND TO THIS TRIAL)
        # ------------------------------------------------------------
        if is_last_stage:
            _finalize_wandb_run(
                status="finished",
                resume_dir=resume_dir,
                study_name=config["optuna_study_name"],
            )

        if last_value is None:
            msg = "No valid objective value produced"
            raise RuntimeError(msg)

        return last_value

    finally:
        if wandb.run is not None:
            wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()

        # kill any leaked worker processes
        for p in mp.active_children():
            with contextlib.suppress(Exception):
                p.terminate()


# ================================================================
# 🧪 Optuna study launcher
# ================================================================
def run_optuna_study(
    *,
    objective: Callable,
    study_name: str,
    n_trials: int,
    direction: str = "minimize",
    pruner: str = "median",
    show_progress_bar: bool = False,
) -> Study:
    """
    Run an Optuna study with the specified configuration.

    Parameters
    ----------
    objective : Callable
        The Optuna objective function to optimize.
    study_name : str
        The name of the Optuna study.
    n_trials : int
        The number of trials to run.
    direction : str, optional
        The optimization direction, either "minimize" or "maximize". Default is "minimize".
    pruner : str, optional
        The pruner type to use. Currently supports "median". Default is "median".
    show_progress_bar : bool, optional
        Whether to show a progress bar during optimization. Default is False.

    Returns
    -------
    Study
        The completed Optuna study object.

    """
    if pruner == "median":
        pruner_obj = MedianPruner(
            n_startup_trials=0,
            n_warmup_steps=0,
            interval_steps=1,
        )
    else:
        msg = f"Unknown pruner: {pruner}"
        raise ValueError(msg)

    study_dir = DATA_PROCESSED / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    study = create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner_obj,
        storage=f"sqlite:///{study_dir}/{study_name}.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    print("Best trial:")
    print(study.best_trial)

    return study
