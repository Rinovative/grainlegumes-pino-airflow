"""
Model training script for U-shaped Neural Operator (U-NO).

This script sets up the U-NO model, optimizer, scheduler, loss functions,
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
from training.tools.spectral_hook import SpectralEnergyHook
from training.train_base import train_base

# ================================================================
# 0) Change Architecture
# ================================================================
SMALL = True

# ================================================================
# ⚙️ 1) Base configuration
# ================================================================
CONFIG = {
    # --- General ---
    "run_suffix": None,
    "seed": 9,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # --- Spectral diagnostics ---
    "enable_spectral_hooks": True,
    # --- Dataset ---
    "train_dataset_name": "lhs_var80_seed3001",
    "ood_dataset_name": "lhs_var120_seed4001",
    "train_ratio": 0.8,  # fraction of dataset used for training
    "ood_fraction": 0.2,  # fraction of OOD data for evaluation
    # --- Dataloader ---
    "batch_size": 32 if SMALL else 16,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    # --- Training ---
    "n_epochs": 1_000 if SMALL else 1_500,
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


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
class UNOWithCheckpoint(UNO):
    """U-NO model with added checkpoint saving functionality."""

    def save_checkpoint(self, save_dir: str, save_name: str = "model") -> None:
        """Save the model state dictionary as a checkpoint."""
        torch.save(self.state_dict(), Path(save_dir) / f"{save_name}.pt")


if SMALL:
    # --- Model small ---
    n_layers = 4
    model = UNOWithCheckpoint(
        in_channels=7,
        out_channels=3,
        hidden_channels=24,
        uno_out_channels=[24, 24, 24, 24],
        uno_n_modes=[[12, 12]] * n_layers,
        uno_scalings=[
            [1.0, 1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        channel_mlp_skip="linear",
    ).to(CONFIG["device"])
else:
    # --- Model big ---
    n_layers = 6
    model = UNOWithCheckpoint(
        in_channels=7,
        out_channels=3,
        hidden_channels=96,
        n_layers=n_layers,
        uno_out_channels=[96, 96, 96, 96, 96, 96],
        uno_n_modes=[[24, 24]] * n_layers,
        uno_scalings=[
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        channel_mlp_skip="linear",
    ).to(CONFIG["device"])

# 🏷️ --- Model naming ---
scaling_tag = "-".join(str(int(s[0]) if s[0].is_integer() else s[0]).replace(".", "") for s in model.uno_scalings)

parts: list[str] = [
    "U-NO",
    f"m{model.uno_n_modes[0][0]}x{model.uno_n_modes[0][1]}",
    f"h{model.hidden_channels}",
    f"l{model.n_layers}",
    f"s{scaling_tag}",
    str(CONFIG["train_dataset_name"]),
]

if CONFIG.get("run_suffix") is not None:
    parts.append(str(CONFIG["run_suffix"]))

CONFIG["model_name"] = "_".join(parts)


# --- Optional: spectral diagnostics ---
spectral_hook = None
if CONFIG.get("enable_spectral_hooks", False):
    spectral_hook = SpectralEnergyHook()
    for module in model.modules():
        if isinstance(module, SpectralConv):
            module.register_forward_hook(spectral_hook.hook)

if SMALL:  # noqa: SIM108
    # --- Optimizer model small ---
    optimizer = AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-4,
    )
else:
    # --- Optimizer model big ---
    optimizer = AdamW(
        model.parameters(),
        lr=5e-3,
        weight_decay=1e-4,
    )

# --- Scheduler ---
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",  # reduce when validation loss plateaus
    factor=0.5,  # halve learning rate
    patience=20,  # number of evals to wait before reducing
    min_lr=1e-5,  # do not go below this lr
)

# --- Losses ---
train_loss = H1Loss(d=2)
eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": RMSEOverall(),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
train_base(CONFIG, model, optimizer, scheduler, train_loss, eval_losses)
