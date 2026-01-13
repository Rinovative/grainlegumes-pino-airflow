"""
Model training script for Physics-Informed U-shaped Neural Operator (PI-U-NO).

This script sets up the PI-U-NO model, optimizer, scheduler, loss functions,
and launches the training process using the base training function.
"""

from pathlib import Path

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO
from neuralop.training import AdamW
from src.util.util_metrics import RMSEOverall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from training.tools.pino_loss import PINOLoss
from training.tools.spectral_hook import SpectralEnergyHook
from training.train_base import train_base

# ================================================================
# ⚙️ 1) Base configuration
# ================================================================
CONFIG = {
    # --- General ---
    "run_suffix": None,
    "seed": 9,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # --- Physics-informed training ---
    "lambda_phys": 1e-4,
    "lambda_p": 5e-4,
    # --- Spectral diagnostics ---
    "enable_spectral_hooks": True,
    # --- Dataset ---
    "train_dataset_name": "lhs_var80_seed3001",
    "ood_dataset_name": "lhs_var120_seed4001",
    "train_ratio": 0.8,  # fraction of dataset used for training
    "ood_fraction": 0.2,  # fraction of OOD data for evaluation
    # --- Dataloader ---
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    # --- Training ---
    "n_epochs": 1_000,
    "eval_interval": 5,  # evaluate every N epochs
    "mixed_precision": False,  # enables AMP on modern GPUs
    # --- Checkpointing & Resume ---
    # "resume_from_dir": "FNO_samples_uniform_var10_N1000_20251112_153012",
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_overall_rmse",  # metric key to monitor for best checkpoint
    "log_physical_rmse": True,  # log RMSE in physical units for each channel
    "save_every": None,  # optional periodic checkpoint saving
}


def finalize_config(CONFIG: dict) -> None:
    """Finalize the configuration by setting default values for missing keys."""
    # Architecture
    CONFIG.setdefault("n_layers", 5)
    CONFIG.setdefault("hidden_channels", 32)
    CONFIG.setdefault("base_modes", 128)
    CONFIG.setdefault(
        "uno_scalings",
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
    )

    # Physics-informed weights
    _ = CONFIG["lambda_phys"]
    _ = CONFIG["lambda_p"]


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
class UNOWithCheckpoint(UNO):
    """U-NO model with added checkpoint saving functionality."""

    def save_checkpoint(self, save_dir: str, save_name: str = "model") -> None:
        """Save the model checkpoint to the specified directory."""
        torch.save(self.state_dict(), Path(save_dir) / f"{save_name}_state_dict.pt")

        metadata = {"model_class": self.__class__.__name__, "architecture": "UNO"}
        torch.save(metadata, Path(save_dir) / f"{save_name}_metadata.pkl")


def build_model(CONFIG: dict) -> UNOWithCheckpoint:
    """Build the U-NO model based on the configuration."""
    n_layers = CONFIG["n_layers"]
    hidden = CONFIG["hidden_channels"]
    base_modes = CONFIG["base_modes"]
    uno_scalings = CONFIG["uno_scalings"]

    uno_n_modes = [[base_modes, base_modes]] * n_layers
    uno_out_channels = [hidden] * n_layers

    return UNOWithCheckpoint(
        in_channels=7,
        out_channels=3,
        hidden_channels=hidden,
        n_layers=n_layers,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        channel_mlp_skip="linear",
    ).to(CONFIG["device"])


# 🏷️ --- Model naming ---
def build_model_name(CONFIG: dict, model: UNO) -> str:
    """Build a descriptive model name based on the configuration."""
    scaling_tag = "-".join(str(int(s[0]) if float(s[0]).is_integer() else s[0]).replace(".", "") for s in model.uno_scalings)

    parts = [
        "PI-UNO",
        f"m{CONFIG['base_modes']}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"s{scaling_tag}",
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
        CONFIG["train_dataset_name"],
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


def build_optimizer(CONFIG: dict, model: UNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 5e-3),
        weight_decay=CONFIG.get("weight_decay", 1e-4),
    )


# --- Scheduler ---
def build_scheduler(optimizer: Optimizer) -> ReduceLROnPlateau:
    """Build the ReduceLROnPlateau scheduler for the optimizer."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-5,
    )


# --- Losses ---
def build_train_loss(CONFIG: dict) -> PINOLoss:
    """Build the PINO loss function based on the configuration."""
    return PINOLoss(
        data_loss=H1Loss(d=2),
        lambda_phys=CONFIG["lambda_phys"],
        lambda_p=CONFIG["lambda_p"],
        in_normalizer=None,
        out_normalizer=None,
    )


eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": RMSEOverall(),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
def run_pi_uno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the PI-U-NO training process."""
    finalize_config(CONFIG)
    model = build_model(CONFIG)
    CONFIG["model_name"] = build_model_name(CONFIG, model)

    spectral_hook = None
    if CONFIG.get("enable_spectral_hooks", False):
        spectral_hook = SpectralEnergyHook()
        for module in model.modules():
            if isinstance(module, SpectralConv):
                module.register_forward_hook(spectral_hook.hook)

    optimizer = build_optimizer(CONFIG, model)
    scheduler = build_scheduler(optimizer)
    train_loss = build_train_loss(CONFIG)

    train_base(
        CONFIG,
        model,
        optimizer,
        scheduler,
        train_loss,
        eval_losses,
        spectral_hook,
        manage_wandb,
    )


if __name__ == "__main__":
    run_pi_uno(CONFIG)
