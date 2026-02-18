"""
Create persistent evaluation artifacts for PINO and FNO models.

This module runs deterministic inference on simulation datasets and stores
reusable artifacts for all downstream evaluation and visualisation modules.

Artifacts
---------
Parquet (global, per case):
    - case_index   : integer case id
    - npz_path     : path to the corresponding NPZ artifact
    - l2           : global L2 error over the full domain
    - rel_l2       : global relative L2 error
    - kappa_names  : list of available permeability tensor components
    - meta         : JSON-safe metadata dictionary

NPZ (local, full fields per case):
    - pred         : (C_out, H, W) model prediction
    - gt           : (C_out, H, W) ground truth
    - err          : (C_out, H, W) prediction error (pred - gt)
    - kappa_log    : (C_kappa, H, W) log10-permeability components
    - kappa        : (C_kappa, H, W) physical permeability components
    - kappa_names  : list[str], same order as kappa channels
    - p_bc         : (1, H, W) pressure boundary condition
    - meta         : JSON string with full metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from training.tools.pino_brinkman_losses import (
    PINOPhysicalLossDiv,
    PINOPhysicalLossPhi,
    PINOSpectralLossDiv,
    PINOSpectralLossPhi,
)

from src.schema.schema_kappa import INTERNAL_KAPPA_2D_ORDER, INTERNAL_KAPPA_3D_ORDER
from src.schema.schema_training import DEFAULT_INPUTS_2D

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(INTERNAL_KAPPA_2D_ORDER) | set(INTERNAL_KAPPA_3D_ORDER)
MU_AIR = 1.8139e-5  # must be consistent with training

# =============================================================================
# JSON / type normalisation utilities
# =============================================================================


NUMPY_INT_TYPES = (
    np.int_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)

NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
)


def meta_to_jsonable(obj: Any) -> Any:
    """
    Convert tensors, numpy values and nested structures into JSON-safe types.

    Rules
    -----
    - torch.Tensor      -> float (0-d) or list
    - numpy.ndarray     -> float (0-d) or list
    - numpy scalar      -> Python scalar
    - dict / list       -> recursively processed

    This function guarantees that the returned object can be safely
    serialised via `json.dumps`.
    """
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return float(arr) if arr.ndim == 0 else arr.tolist()

    if isinstance(obj, np.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()

    if isinstance(obj, NUMPY_INT_TYPES):  # pyright: ignore[reportArgumentType]
        return int(obj)

    if isinstance(obj, NUMPY_FLOAT_TYPES):  # pyright: ignore[reportArgumentType]
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


# =============================================================================
# Kappa field utilities (fields only, no scalar statistics)
# =============================================================================


def detect_kappa_channels_from_inputs(include_inputs: list[str]) -> list[str]:
    """
    Detect permeability-related input channels based on their names.

    Parameters
    ----------
    include_inputs : list[str]
        List of canonical input channel names.

    Returns
    -------
    list[str]
        Names of all channels that represent permeability components.

    """
    return [name for name in include_inputs if name in INTERNAL_KAPPA_NAMES]


def extract_kappa(
    x_tensor: torch.Tensor,
    *,
    input_fields: list[str],
    kappa_names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract log-kappa and physical kappa fields from the input tensor.

    This function handles the case where no permeability channels are
    present by returning empty tensors with the correct shape.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input tensor of shape (B, C_in, H, W).
    input_fields : list[str]
        Canonical list of input channel names.
    kappa_names : list[str]
        Names of kappa components to extract.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
            - "kappa_log"
            - "kappa"

    """
    if not kappa_names:
        return {
            "kappa_log": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
            "kappa": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
        }

    index_map = {name: i for i, name in enumerate(input_fields)}
    kappa_indices = [index_map[name] for name in kappa_names]

    # --------------------------------------------------
    # log-kappa (as stored in dataset)
    # --------------------------------------------------
    kappa_log = x_tensor[:, kappa_indices, :, :]

    # --------------------------------------------------
    # physical kappa reconstruction
    # --------------------------------------------------
    kappa_phys = torch.zeros_like(kappa_log)

    name_to_pos = {name: i for i, name in enumerate(kappa_names)}

    # kxx, kyy (always log10-physical)
    kxx = torch.pow(10.0, kappa_log[:, name_to_pos["kxx"]])
    kyy = torch.pow(10.0, kappa_log[:, name_to_pos["kyy"]])

    kappa_phys[:, name_to_pos["kxx"]] = kxx
    kappa_phys[:, name_to_pos["kyy"]] = kyy

    # kxy is stored as kxy_hat (dimensionless). Keep it as is.
    if "kxy" in name_to_pos:
        kxy_hat = kappa_log[:, name_to_pos["kxy"]]
        kappa_phys[:, name_to_pos["kxy"]] = kxy_hat

    return {
        "kappa_log": kappa_log,
        "kappa": kappa_phys,
    }


# ============================================================================
# Physics term computations
# ============================================================================


def _infer_variant_from_run_dir(run_dir_name: str) -> tuple[str, str]:
    """
    Return (deriv_backend, continuity).

        PS -> physical derivatives
        SP -> spectral derivatives (reflect-FFT)
        divu -> div(u)
        divepsu / divεsu -> div(phi u)
    """
    name = run_dir_name.lower()

    if "-ps" in name or "_ps" in name:
        deriv_backend = "physical"
    elif "-sp" in name or "_sp" in name:
        deriv_backend = "spectral"
    else:
        deriv_backend = "physical"

    # --- continuity: divu vs divepsu (hart) ---
    if "divu" in name:
        continuity = "div"
    elif "divepsu" in name or "divεsu" in name:
        continuity = "phi"
    else:
        # fallback
        continuity = "phi"

    return deriv_backend, continuity


# =============================================================================
# Main artifact generator
# =============================================================================


def generate_artifacts(  # noqa: PLR0915
    *,
    model: Any,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Run inference on all cases and generate persistent evaluation artifacts.

    For each case:
        - perform a forward pass with the trained model
        - compute global error metrics
        - store full spatial fields in an NPZ file
        - store scalar metrics and metadata in a Parquet table

    Parameters
    ----------
    model : Any
        Trained neural operator model (FNO, PINO, etc.).
    loader : DataLoader
        Deterministic evaluation DataLoader.
    processor : Any
        Normalisation processor used during training.
    device : torch.device
        Device used for inference.
    save_root : str or Path
        Root directory for all generated artifacts.
    dataset_name : str
        Base name for the Parquet summary file.
    max_cases : int or None, optional
        Maximum number of cases to process. If None, process all cases.

    Returns
    -------
    df : pandas.DataFrame
        Per-case summary table.
    parquet_path : pathlib.Path
        Path to the written Parquet file.

    """
    model.eval()

    save_root = Path(save_root)

    # Infer run_dir from save_root (.../processed/<run>/analysis/...)
    run_dir = save_root
    while run_dir.name not in {"processed", ""} and not (run_dir / "best_model_state_dict.pt").exists():
        if run_dir.parent == run_dir:
            break
        run_dir = run_dir.parent

    run_name = run_dir.name
    deriv_backend, continuity = _infer_variant_from_run_dir(run_name)

    ps_sp = "PS" if deriv_backend == "physical" else "SP"
    cont_tag = "divu" if continuity == "div" else "divepsu"
    physics_variant = f"{cont_tag}-{ps_sp}"

    print(
        "[ARTIFACTS]",
        f"save_root={save_root}",
        f"run_dir={run_dir}",
        f"run_name={run_name}",
        f"variant={physics_variant}",
        sep="\n  - ",
    )

    # --------------------------------------------------
    # Build exact residual calculator (same as training loss)
    # --------------------------------------------------
    def _zero_data_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        return pred.new_zeros(())

    loss_cls: type[nn.Module]

    if deriv_backend == "physical" and continuity == "div":
        loss_cls = PINOPhysicalLossDiv
    elif deriv_backend == "physical" and continuity == "phi":
        loss_cls = PINOPhysicalLossPhi
    elif deriv_backend == "spectral" and continuity == "div":
        loss_cls = PINOSpectralLossDiv
    else:  # spectral + phi
        loss_cls = PINOSpectralLossPhi
    print(f"[ARTIFACTS] Using loss_cls={loss_cls.__name__}")

    loss_obj = loss_cls(
        data_loss=_zero_data_loss,
        lambda_phys=1.0,
        lambda_p=0.0,
        in_normalizer=processor.in_normalizer,
        out_normalizer=processor.out_normalizer,
    ).to(device)
    loss_obj.eval()

    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation tensors
    in_mean = processor.in_normalizer.mean.to(device)
    in_std = processor.in_normalizer.std.to(device)
    out_mean = processor.out_normalizer.mean.to(device)
    out_std = processor.out_normalizer.std.to(device)

    rows: list[dict[str, Any]] = []

    # Detect available kappa channels from schema
    kappa_names = detect_kappa_channels_from_inputs(DEFAULT_INPUTS_2D)

    for idx, batch in enumerate(loader):
        if max_cases is not None and idx >= max_cases:
            break
        case_id = idx + 1

        x = batch["x"].to(device)
        y = batch["y"].to(device)

        # Metadata (generator-side)
        meta_clean = meta_to_jsonable(batch.get("meta", {}))

        # Pressure boundary condition (stored for diagnostics)
        p_bc_idx = DEFAULT_INPUTS_2D.index("p_bc")
        p_bc = x[:, p_bc_idx : p_bc_idx + 1].detach().cpu()

        # Permeability fields (no scalar stats here)
        kappa_info = extract_kappa(
            x,
            input_fields=DEFAULT_INPUTS_2D,
            kappa_names=kappa_names,
        )

        # Forward pass
        # --------------------------
        # Inference timing per sample
        # --------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda" and start_time is not None:
            start_time.record(torch.cuda.current_stream())

        with torch.no_grad():
            x_norm = (x - in_mean) / (in_std + 1e-12)
            y_hat_norm = model(x_norm)
            y_hat = y_hat_norm * (out_std + 1e-12) + out_mean

        # --------------------------------------------------
        # Exact training-consistent physics diagnostics
        # (works in normalized space)
        # --------------------------------------------------
        with torch.no_grad():
            diag = loss_obj.compute_diagnostics(y_hat_norm, x=x_norm)

        mom_mse = float(diag["mom_mse"].detach().cpu().item())
        cont_mse = float(diag["cont_mse"].detach().cpu().item())
        bc_mse = float(diag["bc_mse"].detach().cpu().item())

        # Optional extra diagnostics (often useful)
        mom_mse_full = float(diag["mom_mse_full"].detach().cpu().item())
        cont_mse_full = float(diag["cont_mse_full"].detach().cpu().item())

        Rx_np = diag["Rx"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Ry_np = diag["Ry"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Rc_np = diag["Rc"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divu_np = diag["div_u"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divphiu_np = diag["div_phi_u"].detach().cpu().squeeze(0).squeeze(0).numpy()
        cont_mse_divu = float(np.mean(divu_np**2))
        cont_mse_divepsu = float(np.mean(divphiu_np**2))

        if device.type == "cuda" and end_time is not None:
            end_time.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            inference_time_ms = start_time.elapsed_time(end_time) if start_time is not None else None
        else:
            inference_time_ms = None

        # Outputs
        p, u, v = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]
        U = torch.sqrt(u**2 + v**2)

        p_gt, u_gt, v_gt = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        U_gt = torch.sqrt(u_gt**2 + v_gt**2)

        y_hat_ext = torch.cat([p, u, v, U], dim=1)
        y_ext = torch.cat([p_gt, u_gt, v_gt, U_gt], dim=1)

        err = y_hat_ext - y_ext
        l2 = torch.linalg.norm(err).item()
        rel_l2 = l2 / (torch.linalg.norm(y_ext).item() + 1e-12)

        idx_x = DEFAULT_INPUTS_2D.index("x")
        idx_y = DEFAULT_INPUTS_2D.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())

        derr_dy, derr_dx = torch.gradient(err, spacing=(dy, dx), dim=(2, 3))
        h1_sq = (err.pow(2) + derr_dx.pow(2) + derr_dy.pow(2)).mean()
        h1 = h1_sq.sqrt().item()

        dy_ref, dx_ref = torch.gradient(y_ext, spacing=(dy, dx), dim=(2, 3))
        h1_ref_sq = (y_ext.pow(2) + dx_ref.pow(2) + dy_ref.pow(2)).mean()
        rel_h1 = h1 / (h1_ref_sq.sqrt().item() + 1e-12)

        # Write NPZ artifact
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        x_raw = x.squeeze(0).detach().cpu().numpy()  # (C_in,H,W)
        y_raw = y.squeeze(0).detach().cpu().numpy()  # (C_out,H,W)

        np.savez_compressed(
            npz_path,
            pred=y_hat_ext.squeeze(0).cpu().numpy(),
            gt=y_ext.squeeze(0).cpu().numpy(),
            err=err.squeeze(0).cpu().numpy(),
            kappa_log=kappa_info["kappa_log"].squeeze(0).cpu().numpy(),
            kappa=kappa_info["kappa"].squeeze(0).cpu().numpy(),
            kappa_names=np.array(kappa_names, dtype=object),
            p_bc=p_bc.squeeze(0).numpy(),
            meta=json.dumps(meta_clean),
            x_raw=x_raw,
            y_raw=y_raw,
            input_fields=np.array(DEFAULT_INPUTS_2D, dtype=object),
            physics_variant=np.array(physics_variant, dtype=object),
            Rx=Rx_np,
            Ry=Ry_np,
            Rc=Rc_np,
            div_u=divu_np,
            div_phi_u=divphiu_np,
        )

        # Parquet row (scalar metrics + metadata only)
        rows.append(
            {
                "inference_time_ms": inference_time_ms,
                "case_index": case_id,
                "npz_path": str(npz_path),
                "l2": l2,
                "rel_l2": rel_l2,
                "h1": h1,
                "rel_h1": rel_h1,
                "kappa_names": kappa_names,
                "physics_variant": physics_variant,
                "mom_mse": mom_mse,
                "cont_mse": cont_mse,
                "cont_mse_divu": cont_mse_divu,
                "cont_mse_divepsu": cont_mse_divepsu,
                "mom_mse_full": mom_mse_full,
                "cont_mse_full": cont_mse_full,
                "bc_mse": bc_mse,
                "meta": json.dumps(meta_clean),
            }
        )
    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
