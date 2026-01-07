"""
Model training pipeline with dataset creation, trainer setup, and logging.

This module defines the `train_base` function which orchestrates the entire
model training process, including dataset loading, dataloader creation,
trainer setup, checkpointing, and logging with wandb.
"""

import inspect
import json
import os
import random
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from neuralop import Trainer
from src import dataset


# ================================================================
# 🧭 Utilities
# ================================================================
def set_seed(seed: int = 9) -> None:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to set.

    Returns
    -------
    None

    """
    random.seed(seed)
    _rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_init_params(obj: Any) -> dict[str, Any]:
    """
    Extract initialisation parameters of an object for logging.

    Parameters
    ----------
    obj : Any
        Object instance to extract parameters from.

    Returns
    -------
    dict[str, Any]
        Dictionary of initialisation parameters and their values.

    """
    try:
        sig = inspect.signature(obj.__class__.__init__)
    except (TypeError, ValueError):
        return {}

    params: dict[str, Any] = {}
    for name in sig.parameters:
        if name == "self":
            continue
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if not callable(value):
            params[name] = value

    return params


def make_json_safe(obj: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON serialisable structures.

    Parameters
    ----------
    obj : Any
        Input object to convert.

    Returns
    -------
    Any
        JSON serialisable representation of the input object.

    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    return str(obj)


def build_wandb_config(
    CONFIG: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    train_loss: Any | None,
    eval_losses: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a wandb configuration dictionary for logging.

    Parameters
    ----------
    CONFIG : Mapping[str, Any]
        Global configuration dictionary.
    model : Any
        Neural operator model instance.
    optimizer : Any
        Initialised optimizer.
    scheduler : Any | None
        Scheduler instance or None.
    train_loss : Any | None
        Training loss function.
    eval_losses : Mapping[str, Any]
        Evaluation losses.

    Returns
    -------
    dict[str, Any]
        Wandb configuration dictionary.

    """
    return {
        "general": {
            "run_name": CONFIG["model_name"],
            "seed": CONFIG["seed"],
            "device": CONFIG["device"],
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "data": {
            "train_dataset": CONFIG["train_dataset_name"],
            "ood_dataset": CONFIG["ood_dataset_name"],
            "train_ratio": CONFIG["train_ratio"],
            "ood_fraction": CONFIG["ood_fraction"],
            "batch_size": CONFIG["batch_size"],
            "num_workers": CONFIG["num_workers"],
        },
        "training": {
            "n_epochs": CONFIG["n_epochs"],
            "eval_interval": CONFIG["eval_interval"],
            "mixed_precision": CONFIG["mixed_precision"],
        },
        "model": {
            "architecture": type(model).__name__,
            "model_params": extract_init_params(model),
            "optimizer": type(optimizer).__name__,
            "optimizer_params": extract_init_params(optimizer),
            "scheduler": type(scheduler).__name__ if scheduler else None,
            "scheduler_params": extract_init_params(scheduler) if scheduler else None,
            "train_loss": type(train_loss).__name__ if train_loss else None,
            "eval_losses": {k: type(v).__name__ for k, v in eval_losses.items()},
        },
    }


# ================================================================
# 🚀 Main Training Pipeline
# ================================================================
def train_base(
    CONFIG: dict[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    train_loss: Any | None = None,
    eval_losses: dict[str, Any] | None = None,
    spectral_hook: Any | None = None,
) -> None:
    """
    Execute the complete model training pipeline.

    This includes:
    - Seeding
    - Resume logic
    - Dataset and dataloader creation
    - Trainer setup and execution
    - Checkpoint handling
    - wandb logging

    Parameters
    ----------
    CONFIG : dict[str, Any]
        Global configuration dictionary.
    model : Any
        Neural operator model instance.
    optimizer : Any
        Initialised optimizer.
    scheduler : Any | None
        Scheduler instance or None.
    train_loss : Any | None
        Training loss function.
    eval_losses : dict[str, Any] | None
        Evaluation losses.
    spectral_hook : Any | None
        Spectral energy hook for spectral diagnostics.

    Returns
    -------
    None

    """
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    data_root = Path("model_training/data/raw")
    train_dataset = data_root / CONFIG["train_dataset_name"] / f"{CONFIG['train_dataset_name']}.pt"
    ood_dataset = data_root / CONFIG["ood_dataset_name"] / f"{CONFIG['ood_dataset_name']}.pt"

    # ------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------
    resume_from = CONFIG.get("resume_from_dir", None)  # noqa: SIM910
    base = Path("model_training/data/processed")

    if resume_from:
        if resume_from == "latest":
            runs = sorted([p for p in base.iterdir() if p.is_dir()])
            if not runs:
                msg = "No previous runs for latest."
                raise FileNotFoundError(msg)
            resume_from = runs[-1]
        else:
            resume_from = base / resume_from

        if not resume_from.is_dir():
            msg = f"resume_from_dir not found: {resume_from}"
            raise FileNotFoundError(msg)

        print(f"[RUN] Resuming from checkpoint: {resume_from}")

        run_name = resume_from.name
        save_dir = resume_from

    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        # W&B / semantic name
        run_name = f"{CONFIG['model_name']}_{CONFIG['train_dataset_name']}"
        # Local save directory
        save_dir = base / f"{run_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {run_name} → {save_dir}")

    # ------------------------------------------------------------
    # W&B Setup
    # ------------------------------------------------------------
    os.environ["WANDB_PROJECT"] = "grainlegumes_pino"
    os.environ["WANDB_ENTITY"] = "Rinovative-Hub"
    os.environ["WANDB_DIR"] = "model_training/training/wandb"

    wandb_cfg = build_wandb_config(CONFIG, model, optimizer, scheduler, train_loss, eval_losses or {})

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=run_name,
        dir=os.environ["WANDB_DIR"],
        config=wandb_cfg,
        reinit=True,
    )

    config_path = save_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(wandb_cfg), f, indent=2)
    wandb.save(str(config_path))

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    dataloader_cfg = {
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["num_workers"],
        "pin_memory": CONFIG["pin_memory"],
        "persistent_workers": CONFIG["persistent_workers"],
    }

    train_loader, test_loaders, data_processor = dataset.dataset_base.create_dataloaders(
        dataset_cls=dataset.dataset_simulation.PermeabilityFlowDataset,
        path_train=str(train_dataset),
        path_test_ood=str(ood_dataset),
        train_ratio=CONFIG["train_ratio"],
        ood_fraction=CONFIG["ood_fraction"],
        **dataloader_cfg,
    )

    data_processor = data_processor.to(device)

    # Save normalizer for inference
    normalizer_path = save_dir / "normalizer.pt"
    torch.save(data_processor.state_dict(), normalizer_path)

    # ------------------------------------------------------------
    # Trainer Setup
    # ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        n_epochs=CONFIG["n_epochs"],
        wandb_log=True,
        device=device,
        mixed_precision=CONFIG["mixed_precision"],
        data_processor=data_processor,
        eval_interval=CONFIG["eval_interval"],
        verbose=True,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    save_best = CONFIG["save_best"] if CONFIG.get("save_best") is not None else None

    if resume_from is None:
        # → NO RESUME
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=train_loss,
            eval_losses=eval_losses,
            save_dir=str(save_dir),
            save_best=save_best,  # pyright: ignore[reportArgumentType]
        )
    else:
        # → RESUME
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=train_loss,
            eval_losses=eval_losses,
            save_dir=str(save_dir),
            save_best=save_best,  # pyright: ignore[reportArgumentType]
            resume_from_dir=str(resume_from),
        )

    # ------------------------------------------------------------
    # Spectral diagnostics save
    # ------------------------------------------------------------
    if spectral_hook is not None:
        spectral_agg = spectral_hook.aggregate()
        torch.save(spectral_agg, save_dir / "spectral_energy_aggregated.pt")

    # ------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------
    wandb.finish()
