"""
Model training script for Physics-Informed Fourier Neural Operator (PI-FNO).

This script sets up the FNO model, optimizer, scheduler, physics-informed loss functions,
and launches the training process using the base training function.
"""

import torch
from neuralop import H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import FNO
from neuralop.training import AdamW
from src.util.util_metrics import RMSEOverall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from training.tools.pino_loss_physical import PINOPhysicalLoss
from training.tools.pino_loss_spectral import PINOSpectralLoss
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
    # "fft" -> PI-FNO-FFT (spectral derivatives)
    # "ps"  -> PI-FNO-PS  (physical space derivatives)
    "pino_loss_type": "fft",
    "lambda_phys": 1e-3,
    "lambda_p": 5e-3,
    "phys_warmup_epochs": 0,
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
    "n_epochs": 1000,
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
    """Finalize and validate the configuration dictionary."""
    # Architecture
    CONFIG.setdefault("modes_x", 12)
    CONFIG.setdefault("modes_y", 12)
    CONFIG.setdefault("hidden_channels", 24)
    CONFIG.setdefault("n_layers", 4)

    # Physics-informed weights (nur Existenz erzwingen)
    _ = CONFIG["lambda_phys"]
    _ = CONFIG["lambda_p"]


# ================================================================
# 🧠 2) Model, hooks, optimizer, scheduler, and losses
# ================================================================
# --- Model ---
def build_model(CONFIG: dict) -> FNO:
    """Build the FNO model based on the configuration."""
    m_x = int(CONFIG["modes_x"])
    m_y = int(CONFIG["modes_y"])

    CONFIG["n_modes_xy"] = (m_x, m_y)

    return FNO(
        n_modes=(m_x, m_y),
        hidden_channels=CONFIG["hidden_channels"],
        in_channels=7,
        out_channels=3,
        n_layers=CONFIG["n_layers"],
    ).to(CONFIG["device"])


# 🏷️ --- Model naming ---
def build_model_name(CONFIG: dict) -> str:
    """Build a descriptive FFT-PI-FNO model name based on the configuration."""
    m_x, m_y = CONFIG.get("n_modes_xy", (CONFIG["modes_x"], CONFIG["modes_y"]))

    loss_tag = "PI-FNO-FFT" if CONFIG.get("pino_loss_type") == "fft" else "PI-FNO-PS"

    parts = [
        loss_tag,
        f"m{m_x}x{m_y}",
        f"h{CONFIG['hidden_channels']}",
        f"l{CONFIG['n_layers']}",
        f"lamPhys{CONFIG['lambda_phys']:.0e}",
        f"lamP{CONFIG['lambda_p']:.0e}",
    ]

    if CONFIG.get("run_suffix") is not None:
        parts.append(str(CONFIG["run_suffix"]))

    return "_".join(parts)


# --- Optimizer ---
def build_optimizer(CONFIG: dict, model: FNO) -> AdamW:
    """Build the AdamW optimizer for the model."""
    return AdamW(
        model.parameters(),
        lr=CONFIG.get("lr", 0.01),
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
def build_train_loss(CONFIG: dict) -> PINOPhysicalLoss | PINOSpectralLoss:
    """Build the physics-informed loss function for training."""
    loss_type = CONFIG.get("pino_loss_type", "fft")

    if loss_type == "fft":
        return PINOSpectralLoss(
            data_loss=H1Loss(d=2),
            lambda_phys=CONFIG["lambda_phys"],
            lambda_p=CONFIG["lambda_p"],
            in_normalizer=None,
            out_normalizer=None,
        )

    if loss_type == "ps":
        return PINOPhysicalLoss(
            data_loss=H1Loss(d=2),
            lambda_phys=CONFIG["lambda_phys"],
            lambda_p=CONFIG["lambda_p"],
            in_normalizer=None,
            out_normalizer=None,
        )

    msg = f"Unknown pino_loss_type: {loss_type}"
    raise ValueError(msg)


eval_losses = {
    "h1": H1Loss(d=2),
    "l2": LpLoss(d=2, p=2),
    "overall_rmse": RMSEOverall(),
}


# ================================================================
# 🚀 3) Launch training
# ================================================================
def run_pi_fno(CONFIG: dict, manage_wandb: bool = True) -> None:
    """Run the training process for the PI-FNO model."""
    finalize_config(CONFIG)
    model = build_model(CONFIG)
    CONFIG["model_name"] = build_model_name(CONFIG)

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
    run_pi_fno(CONFIG)
