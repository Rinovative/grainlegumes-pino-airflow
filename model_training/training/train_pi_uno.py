"""
Model training script for Physics-Informed U-shaped Neural Operator (PI-U-NO).

This script sets up the PI-U-NO model, optimizer, scheduler, loss functions,
and launches the training process using the base training function.
"""

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO
from neuralop.training import AdamW
from src.util.util_metrics import RelRMSEChannel, RMSEOverall
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    "lambda_phys": 1e-2,
    # --- Spectral diagnostics ---
    "enable_spectral_hooks": False,
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
    "n_epochs": 1000,
    "eval_interval": 5,  # evaluate every N epochs
    "mixed_precision": False,  # enables AMP on modern GPUs
    # --- Checkpointing & Resume ---
    # "resume_from_dir": "FNO_samples_uniform_var10_N1000_20251112_153012",
    # "resume_from_dir": "latest",
    # --- Logging ---
    "save_best": "eval_overall_rmse",  # metric key to monitor for best checkpoint
    "save_every": None,  # optional periodic checkpoint saving
}


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
# --- Model ---
n_layers = 4

model = UNO(
    in_channels=7,
    out_channels=3,
    hidden_channels=24,
    n_layers=n_layers,
    uno_out_channels=[24, 24, 24, 24],
    uno_n_modes=[[12, 12]] * n_layers,
    uno_scalings=[
        [1.0, 1.0],  # L0: original
        [0.5, 0.5],  # L1: downsample
        [1.0, 1.0],  # L2: process coarse
        [2.0, 2.0],  # L3: upsample
    ],
).to(CONFIG["device"])


# 🏷️ --- Model naming ---
scaling_tag = "-".join(str(int(s[0]) if s[0].is_integer() else s[0]).replace(".", "") for s in model.uno_scalings)

parts: list[str] = [
    "PI-U-NO",
    f"m{model.uno_n_modes[0][0]}x{model.uno_n_modes[0][1]}",
    f"h{model.hidden_channels}",
    f"l{model.n_layers}",
    f"s{scaling_tag}",
    f"lam{CONFIG['lambda_phys']:.0e}",
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

# --- Optimizer ---
optimizer = AdamW(
    model.parameters(),
    lr=1e-2,
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
train_loss = PINOLoss(
    data_loss=H1Loss(d=2),
    lambda_phys=CONFIG["lambda_phys"],
)
eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": RMSEOverall(),
    "rel_rmse_p": RelRMSEChannel(0),
    "rel_rmse_u": RelRMSEChannel(1),
    "rel_rmse_v": RelRMSEChannel(2),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
train_base(CONFIG, model, optimizer, scheduler, train_loss, eval_losses)
