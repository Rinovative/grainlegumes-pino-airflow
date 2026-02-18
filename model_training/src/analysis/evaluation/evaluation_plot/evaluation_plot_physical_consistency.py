"""
Physical consistency plots for PINO/FNO evaluation.

This module provides interactive plots and summary tables that assess
physical admissibility of model predictions independently of classical
error metrics. It is intentionally "no-raise" and "no try/except":

    - Missing columns, missing NPZ keys, invalid shapes or non-finite values
      are handled by returning NaN, skipping affected samples, or disabling
      individual plot elements.
    - The goal is robustness when working with mixed artifact versions.

Implemented physical checks
---------------------------
    1. Mass conservation (continuity)
       - Scalar CDF: cont_mse_divu  (MSE of div(u) over the interior)
       - Mean map:   mean |div(u)|  from stored div_u fields

    2. Pressure boundary condition consistency
       - Scalar CDF: bc_mse (MSE of (p - p_bc) on boundary support)
         Uses a threshold mask derived from max(|p_bc|) to avoid noise.

    3. Darcy-Brinkman momentum residual consistency
       - Scalar CDF: mom_mse (MSE of Rx^2 + Ry^2 computed from gt + inputs)
       - Mean map:   mean sqrt(Rx^2 + Ry^2) from stored Rx/Ry fields

Data assumptions
----------------
Evaluation DataFrames are case-level tables. The plots require:
    - npz_path (path to evaluation artifact .npz)
    - geometry_Lx, geometry_Ly (for physical coordinate axes in maps)
    - metric columns (cont_mse_divu, bc_mse, mom_mse) for model curves

NPZ artifacts are expected to contain (if present):
    - gt:  (C, ny, nx) with channels [p, u, v, ...]
    - p_bc: (1, ny, nx) or (ny, nx)
    - x_raw: (Cin, ny, nx) and input_fields describing Cin
    - physics_variant: one of {divu-PS, divu-SP, divepsu-PS, divepsu-SP}
    - div_u, div_phi_u, Rx, Ry (for mean maps, when available)

Interior padding rule
--------------------
The interior crop padding is inferred as:
    - If meta contains {"interior_pad": ...} as a dict, use it.
    - Else infer from physics_variant suffix:
        ...-PS -> pad = 0
        ...-SP -> pad = 2
    - Fallback: pad = 0

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

from src import util

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


# =============================================================================
# CONSTANTS
# =============================================================================
EPS = 1e-12
CLIP_Q = 99.5
MU_AIR = 1.8139e-5

BC_THR_REL = 1e-6
BC_THR_ABS = 1e-12


# =============================================================================
# SCHEMAS
# =============================================================================
DF_REQUIRED_COLS = ("npz_path", "geometry_Lx", "geometry_Ly")
NPZ_REQUIRED_KEYS = ("gt", "p_bc", "x_raw", "input_fields", "meta", "physics_variant")
NPZ_REQUIRED_INPUT_FIELDS = ("x", "y", "phi", "kxx", "kyy")


# =============================================================================
# TYPES
# =============================================================================
class _FieldAcc(TypedDict):
    loaded_until: int
    sum_field: np.ndarray | None
    count: int


# =============================================================================
# SAFE HELPERS (no raise, no try/except)
# =============================================================================
def _nan2d() -> np.ndarray:
    """
    Create a minimal 2D NaN array placeholder.

    Returns
    -------
    numpy.ndarray
        Array of shape (1, 1) filled with NaN.

    """
    return np.full((1, 1), np.nan, dtype=float)


def _has_df_schema(df: pd.DataFrame) -> bool:
    """
    Check whether an evaluation DataFrame contains the required base columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Case-level evaluation table.

    Returns
    -------
    bool
        True if required columns exist, else False.

    """
    return all(c in df.columns for c in DF_REQUIRED_COLS)


def _has_npz_keys(data: Any) -> bool:
    """
    Check whether a loaded NPZ object contains required keys.

    Parameters
    ----------
    data : Any
        np.load(...) return value (NpzFile-like).

    Returns
    -------
    bool
        True if all required keys exist, else False.

    """
    if not hasattr(data, "files"):
        return False
    return all(k in data.files for k in NPZ_REQUIRED_KEYS)


def _npz_get(data: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a key from a loaded NPZ object.

    Parameters
    ----------
    data : Any
        np.load(...) return value (NpzFile-like).
    key : str
        NPZ key.
    default : Any, optional
        Fallback value if key does not exist.

    Returns
    -------
    Any
        The NPZ entry if present, otherwise default.

    """
    if not hasattr(data, "files"):
        return default
    if key not in data.files:
        return default
    return data[key]


def _take_2d(a: Any) -> np.ndarray:
    """
    Coerce an array-like to a 2D float array.

    Accepts (ny, nx) or (1, ny, nx). Any other shape returns a NaN placeholder.

    Parameters
    ----------
    a : Any
        Array-like object.

    Returns
    -------
    numpy.ndarray
        2D float array or NaN placeholder.

    """
    if a is None:
        return _nan2d()
    a2 = np.asarray(a)
    if a2.ndim == 3 and a2.shape[0] == 1:  # noqa: PLR2004
        a2 = a2[0]
    if a2.ndim != 2:  # noqa: PLR2004
        return _nan2d()
    return np.asarray(a2, dtype=float)


def _infer_interior_pad(data: Any) -> int:
    """
    Infer interior padding from NPZ data.

    The function is robust to mixed artifact versions and never raises.

    Rules
    -----
    1) If meta is a dict and contains "interior_pad", use it.
    2) Otherwise infer from physics_variant suffix:
         ...-PS -> 0
         ...-SP -> 2
    3) Fallback: 0

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    int
        Interior crop padding in pixels.

    """
    if not _has_npz_keys(data):
        return 0

    meta_obj = _npz_get(data, "meta", default=None)
    meta_val: Any = meta_obj

    if isinstance(meta_obj, np.ndarray):
        if meta_obj.shape == ():
            meta_val = meta_obj.item()
        elif meta_obj.size == 1:
            meta_val = meta_obj.flat[0]
        else:
            meta_val = None

    if isinstance(meta_val, dict):
        v = meta_val.get("interior_pad", None)
        if isinstance(v, (int, np.integer)):
            return max(0, int(v))
        if isinstance(v, float) and np.isfinite(v):
            return max(0, int(v))

    pv = _npz_get(data, "physics_variant", default="")
    if isinstance(pv, np.ndarray) and pv.shape == ():
        pv = pv.item()

    backend = str(pv).strip().upper().split("-")[-1]
    if backend == "PS":
        return 0
    if backend == "SP":
        return 2

    return 0


def _crop_interior(a: Any, pad: int) -> np.ndarray:
    """
    Crop a 2D field to its interior, removing a constant padding border.

    If the crop is invalid (wrong dims, pad too large), a NaN placeholder
    is returned.

    Parameters
    ----------
    a : Any
        Array-like object, expected 2D.
    pad : int
        Pixels removed on each side.

    Returns
    -------
    numpy.ndarray
        Cropped 2D array or NaN placeholder.

    """
    if a is None:
        return _nan2d()
    a2 = np.asarray(a, dtype=float)
    if a2.ndim != 2:  # noqa: PLR2004
        return _nan2d()
    if pad <= 0:
        return a2
    if a2.shape[0] <= 2 * pad or a2.shape[1] <= 2 * pad:
        return _nan2d()
    return a2[pad:-pad, pad:-pad]


def _npz_input_fields(data: Any) -> list[str]:
    """
    Read input_fields from NPZ and return as list[str].

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    list[str]
        List of input field names. Empty if missing.

    """
    fields_raw = _npz_get(data, "input_fields", default=[])
    fields = fields_raw.tolist() if isinstance(fields_raw, np.ndarray) else list(fields_raw)
    return [str(s) for s in fields]


def _npz_get_input_field(data: Any, name: str) -> np.ndarray:
    """
    Extract a specific input field from x_raw by name.

    If the field is unavailable or inconsistent, a NaN placeholder is returned.

    Parameters
    ----------
    data : Any
        Loaded NPZ object.
    name : str
        Input field name to extract.

    Returns
    -------
    numpy.ndarray
        2D array (ny, nx) or NaN placeholder.

    """
    x_raw = _npz_get(data, "x_raw", default=None)
    if x_raw is None:
        return _nan2d()

    x_raw2 = np.asarray(x_raw)
    if x_raw2.ndim != 3:  # noqa: PLR2004
        return _nan2d()

    fields = _npz_input_fields(data)
    if len(fields) != x_raw2.shape[0]:
        return _nan2d()

    if name not in fields:
        return _nan2d()

    idx = int(fields.index(name))
    return np.asarray(x_raw2[idx], dtype=float)


def _infer_dx_dy_from_npz(data: Any) -> tuple[float, float]:
    """
    Infer grid spacing (dx, dy) from x and y input fields.

    If inference fails, (1.0, 1.0) is returned.

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    tuple[float, float]
        (dx, dy) grid spacing.

    """
    x = _npz_get_input_field(data, "x")
    y = _npz_get_input_field(data, "y")

    if x.ndim != 2 or y.ndim != 2:  # noqa: PLR2004
        return 1.0, 1.0
    if x.shape[1] <= 1 or y.shape[0] <= 1:
        return 1.0, 1.0

    dx = np.nanmedian(np.diff(x, axis=1))
    dy = np.nanmedian(np.diff(y, axis=0))

    if not np.isfinite(dx) or not np.isfinite(dy):
        return 1.0, 1.0

    dx_f = float(abs(float(dx)))
    dy_f = float(abs(float(dy)))

    if dx_f <= 0.0 or dy_f <= 0.0:
        return 1.0, 1.0

    return dx_f, dy_f


def _infer_geom(df: pd.DataFrame) -> tuple[float, float]:
    """
    Infer (Lx, Ly) from an evaluation DataFrame.

    If missing or invalid, (1.0, 1.0) is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation table.

    Returns
    -------
    tuple[float, float]
        (Lx, Ly) geometry extents.

    """
    if not _has_df_schema(df):
        return 1.0, 1.0

    Lx = pd.to_numeric(df["geometry_Lx"].iloc[0], errors="coerce")
    Ly = pd.to_numeric(df["geometry_Ly"].iloc[0], errors="coerce")

    Lx_f = float(Lx) if np.isfinite(Lx) else 1.0
    Ly_f = float(Ly) if np.isfinite(Ly) else 1.0

    if Lx_f <= 0.0 or Ly_f <= 0.0:
        return 1.0, 1.0

    return Lx_f, Ly_f


def _setup_xy_axes(
    ax: Any,
    *,
    LxLy: tuple[float, float],
    ny: int,
    nx: int,
    is_left: bool,
    is_bottom: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Configure axes to show physical coordinates (x, y) in meters.

    Only the left-most subplot shows y-labels, and only the bottom row shows
    x-labels.

    Parameters
    ----------
    ax : Any
        Matplotlib Axes.
    LxLy : tuple[float, float]
        Physical domain size (Lx, Ly).
    ny : int
        Number of grid points in y.
    nx : int
        Number of grid points in x.
    is_left : bool
        Whether this axis is on the left-most column.
    is_bottom : bool
        Whether this axis is on the bottom row.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Coordinate vectors (x, y).

    """
    ax.set_aspect("equal", adjustable="box")

    Lx, Ly = LxLy
    Lx = float(Lx) if np.isfinite(Lx) and Lx > 0.0 else 1.0
    Ly = float(Ly) if np.isfinite(Ly) and Ly > 0.0 else 1.0

    nx_i = int(nx) if int(nx) > 1 else 2
    ny_i = int(ny) if int(ny) > 1 else 2

    x = np.linspace(0.0, Lx, nx_i)
    y = np.linspace(0.0, Ly, ny_i)

    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)

    if is_left:
        ax.set_ylabel("y [m]")
        ax.set_yticks([0.0, Ly / 2.0, Ly])
    else:
        ax.set_yticks([])

    if is_bottom:
        ax.set_xlabel("x [m]")
        ax.set_xticks([0.0, Lx / 2.0, Lx])
    else:
        ax.set_xticks([])

    return x, y


# =============================================================================
# GT METRICS (NaN on failure, no try/except)
# =============================================================================
def _compute_gt_cont_mse_divu_from_npz(data: Any) -> float:
    """
    Compute GT continuity metric: MSE(div(u)) over the cropped interior.

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    float
        Mean squared divergence, or NaN if not computable.

    """
    if not _has_npz_keys(data):
        return float("nan")

    gt = _npz_get(data, "gt", default=None)
    if gt is None:
        return float("nan")

    gt2 = np.asarray(gt, dtype=float)
    if gt2.ndim != 3 or gt2.shape[0] < 3:  # noqa: PLR2004
        return float("nan")

    pad = _infer_interior_pad(data)
    dx, dy = _infer_dx_dy_from_npz(data)

    u = _crop_interior(gt2[1], pad)
    v = _crop_interior(gt2[2], pad)

    if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape:  # noqa: PLR2004
        return float("nan")
    if u.shape[0] < 2 or u.shape[1] < 2:  # noqa: PLR2004
        return float("nan")

    _, du_dx = np.gradient(u, dy, dx, edge_order=1)
    dv_dy, _ = np.gradient(v, dy, dx, edge_order=1)

    div_u = du_dx + dv_dy
    v_mse = float(np.mean(div_u**2))
    return v_mse if np.isfinite(v_mse) else float("nan")


def _compute_gt_bc_mse_from_npz(data: Any) -> float:
    """
    Compute GT pressure boundary mismatch metric: MSE(p - p_bc) on boundary support.

    Boundary support is defined as points where |p_bc| exceeds a threshold:
        thr = max(BC_THR_ABS, BC_THR_REL * max(|p_bc|))

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    float
        Boundary MSE or NaN if not computable.

    """
    if not _has_npz_keys(data):
        return float("nan")

    gt = _npz_get(data, "gt", default=None)
    p_bc = _npz_get(data, "p_bc", default=None)
    if gt is None or p_bc is None:
        return float("nan")

    gt2 = np.asarray(gt, dtype=float)
    bc2 = _take_2d(p_bc)

    if gt2.ndim != 3 or gt2.shape[0] < 1:  # noqa: PLR2004
        return float("nan")

    pad = _infer_interior_pad(data)

    p = _crop_interior(gt2[0], pad)
    bc = _crop_interior(bc2, pad)

    if p.ndim != 2 or bc.ndim != 2 or p.shape != bc.shape:  # noqa: PLR2004
        return float("nan")

    finite = np.isfinite(bc)
    if not np.any(finite):
        return float("nan")

    bc_abs = np.abs(bc)
    bc_max = float(np.nanmax(bc_abs[finite]))
    if not np.isfinite(bc_max) or bc_max <= 0.0:
        return float("nan")

    thr = max(BC_THR_ABS, BC_THR_REL * bc_max)
    mask = finite & (bc_abs > thr)
    if not np.any(mask):
        return float("nan")

    diff = p[mask] - bc[mask]
    v_mse = float(np.mean(diff**2))
    return v_mse if np.isfinite(v_mse) else float("nan")


def _compute_gt_mom_mse_from_npz(data: Any) -> float:
    """
    Compute GT Darcy-Brinkman momentum residual MSE.

    The residual is computed from ground-truth (p, u, v) and input fields
    (phi, kxx, kyy) using finite differences.

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    float
        MSE(Rx^2 + Ry^2) over the interior, or NaN if not computable.

    """
    if not _has_npz_keys(data):
        return float("nan")

    fields = _npz_input_fields(data)
    if any(req not in fields for req in NPZ_REQUIRED_INPUT_FIELDS):
        return float("nan")

    gt = _npz_get(data, "gt", default=None)
    if gt is None:
        return float("nan")

    gt2 = np.asarray(gt, dtype=float)
    if gt2.ndim != 3 or gt2.shape[0] < 3:  # noqa: PLR2004
        return float("nan")

    pad = _infer_interior_pad(data)
    dx, dy = _infer_dx_dy_from_npz(data)

    p = _crop_interior(gt2[0], pad)
    u = _crop_interior(gt2[1], pad)
    v = _crop_interior(gt2[2], pad)

    phi = _crop_interior(_npz_get_input_field(data, "phi"), pad)
    kxx_log2 = _crop_interior(_npz_get_input_field(data, "kxx"), pad)
    kyy_log2 = _crop_interior(_npz_get_input_field(data, "kyy"), pad)

    if p.ndim != 2:  # noqa: PLR2004
        return float("nan")
    if not (u.shape == v.shape == p.shape == phi.shape == kxx_log2.shape == kyy_log2.shape):
        return float("nan")
    if p.shape[0] < 2 or p.shape[1] < 2:  # noqa: PLR2004
        return float("nan")

    phi2 = np.maximum(phi, 1e-6)
    kxx = np.maximum(10.0**kxx_log2, 1e-30)
    kyy = np.maximum(10.0**kyy_log2, 1e-30)

    mu = MU_AIR

    dp_dy, dp_dx = np.gradient(p, dy, dx, edge_order=1)
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=1)
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=1)

    div_u = du_dx + dv_dy

    fac = mu / phi2
    tau_xx = fac * (2.0 * du_dx - (2.0 / 3.0) * div_u)
    tau_yy = fac * (2.0 * dv_dy - (2.0 / 3.0) * div_u)
    tau_xy = fac * (du_dy + dv_dx)

    _, dtau_xx_dx = np.gradient(tau_xx, dy, dx, edge_order=1)
    dtau_xy_dy, _ = np.gradient(tau_xy, dy, dx, edge_order=1)
    _, dtau_xy_dx = np.gradient(tau_xy, dy, dx, edge_order=1)
    dtau_yy_dy, _ = np.gradient(tau_yy, dy, dx, edge_order=1)

    div_tau_x = dtau_xx_dx + dtau_xy_dy
    div_tau_y = dtau_xy_dx + dtau_yy_dy

    drag_x = mu * (u / kxx)
    drag_y = mu * (v / kyy)

    Rx = -dp_dx + div_tau_x - drag_x
    Ry = -dp_dy + div_tau_y - drag_y

    v_mse = float(np.mean(Rx**2 + Ry**2))
    return v_mse if np.isfinite(v_mse) else float("nan")


def _compute_gt_metric_from_npz(npz_path: str | Path, *, metric: str) -> float:
    """
    Compute a GT metric from an NPZ artifact path.

    Parameters
    ----------
    npz_path : str | pathlib.Path
        Path to a case NPZ artifact.
    metric : str
        Metric identifier:
            - cont_mse_divu
            - bc_mse
            - mom_mse

    Returns
    -------
    float
        Computed metric or NaN if not computable.

    """
    p = Path(str(npz_path))
    if not p.exists():
        return float("nan")

    with np.load(p, allow_pickle=True) as z:
        if metric == "cont_mse_divu":
            return _compute_gt_cont_mse_divu_from_npz(z)
        if metric == "bc_mse":
            return _compute_gt_bc_mse_from_npz(z)
        if metric == "mom_mse":
            return _compute_gt_mom_mse_from_npz(z)

    return float("nan")


def _infer_inlet_outlet_masks_from_p_bc(bc: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer inlet/outlet masks from a p_bc field on the grid.

    Assumption:
        - p_bc is non-zero on the inlet boundary
        - outlet pressure is 0 on the opposite boundary
        - elsewhere p_bc may be 0 (interior), so we must use geometry (sides)

    Parameters
    ----------
    bc : numpy.ndarray
        2D pressure boundary condition field.
    thr : float
        Threshold for inlet support: |p_bc| > thr.

    Returns
    -------
    (inlet_mask, outlet_mask) as boolean arrays, or all-False masks on failure.

    """
    if bc is None:
        z = np.zeros((1, 1), dtype=bool)
        return z, z

    bc2 = np.asarray(bc, dtype=float)
    if bc2.ndim != 2:  # noqa: PLR2004
        z = np.zeros((1, 1), dtype=bool)
        return z, z

    finite = np.isfinite(bc2)
    if not np.any(finite):
        z = np.zeros_like(bc2, dtype=bool)
        return z, z

    inlet_support = finite & (np.abs(bc2) > float(thr))
    if not np.any(inlet_support):
        z = np.zeros_like(bc2, dtype=bool)
        return z, z

    ny, nx = bc2.shape

    c_bottom = int(np.sum(inlet_support[0, :]))
    c_top = int(np.sum(inlet_support[ny - 1, :]))
    c_left = int(np.sum(inlet_support[:, 0]))
    c_right = int(np.sum(inlet_support[:, nx - 1]))

    counts = np.asarray([c_bottom, c_top, c_left, c_right], dtype=int)
    side = int(np.argmax(counts))
    if counts[side] == 0:
        z = np.zeros_like(bc2, dtype=bool)
        return z, z

    inlet_mask = np.zeros_like(bc2, dtype=bool)
    if side == 0:
        inlet_mask[0, :] = inlet_support[0, :]
    elif side == 1:
        inlet_mask[ny - 1, :] = inlet_support[ny - 1, :]
    elif side == 2:  # noqa: PLR2004
        inlet_mask[:, 0] = inlet_support[:, 0]
    else:
        inlet_mask[:, nx - 1] = inlet_support[:, nx - 1]

    outlet_mask = np.zeros_like(bc2, dtype=bool)
    # opposite side: bottom <-> top, left <-> right
    if side == 0:
        outlet_mask[ny - 1, :] = finite[ny - 1, :]
    elif side == 1:
        outlet_mask[0, :] = finite[0, :]
    elif side == 2:  # noqa: PLR2004
        outlet_mask[:, nx - 1] = finite[:, nx - 1]
    else:
        outlet_mask[:, 0] = finite[:, 0]

    outlet_mask &= ~inlet_mask

    if not np.any(outlet_mask):
        z = np.zeros_like(bc2, dtype=bool)
        return z, z

    return inlet_mask, outlet_mask


# =============================================================================
# PRESSURE DROP (Delta-p) CONSISTENCY (NPZ-based)
# =============================================================================
def _compute_dp_rel_err_from_npz(data: Any) -> float:  # noqa: PLR0911
    """
    Relative pressure-drop mismatch.

    Uses inlet = boundary side where p_bc is non-zero,
    outlet = opposite boundary side (can be p_bc == 0).

    rel = |dp_pred - dp_bc| / (|dp_bc| + EPS)

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    float
        Relative error or NaN if not computable.

    """
    if not hasattr(data, "files"):
        return float("nan")

    pred = _npz_get(data, "pred", default=None)
    p_bc = _npz_get(data, "p_bc", default=None)
    if pred is None or p_bc is None:
        return float("nan")

    pr = np.asarray(pred, dtype=float)
    if pr.ndim != 3 or pr.shape[0] < 1:  # noqa: PLR2004
        return float("nan")

    pad = _infer_interior_pad(data)

    p_pred = _crop_interior(pr[0], pad)
    bc = _crop_interior(_take_2d(p_bc), pad)

    if p_pred.ndim != 2 or bc.ndim != 2 or p_pred.shape != bc.shape:  # noqa: PLR2004
        return float("nan")

    finite = np.isfinite(bc)
    if not np.any(finite):
        return float("nan")

    bc_abs = np.abs(bc)
    bc_max = float(np.nanmax(bc_abs[finite]))
    if not np.isfinite(bc_max) or bc_max <= 0.0:
        return float("nan")

    thr = max(BC_THR_ABS, BC_THR_REL * bc_max)

    inlet_mask, outlet_mask = _infer_inlet_outlet_masks_from_p_bc(bc, thr)
    if inlet_mask.shape != bc.shape or outlet_mask.shape != bc.shape:
        return float("nan")
    if not (np.any(inlet_mask) and np.any(outlet_mask)):
        return float("nan")

    dp_bc = float(np.nanmean(bc[inlet_mask]) - np.nanmean(bc[outlet_mask]))
    if not np.isfinite(dp_bc) or abs(dp_bc) <= EPS:
        return float("nan")

    dp_pred = float(np.nanmean(p_pred[inlet_mask]) - np.nanmean(p_pred[outlet_mask]))
    if not np.isfinite(dp_pred):
        return float("nan")

    rel = abs(dp_pred - dp_bc) / (abs(dp_bc) + EPS)
    return float(rel) if np.isfinite(rel) else float("nan")


def _compute_dp_rel_err_gt_from_npz(data: Any) -> float:
    """
    GT reference for relative pressure-drop mismatch.

    rel = |dp_gt - dp_bc| / (|dp_bc| + EPS)

    inlet: boundary side where p_bc is non-zero
    outlet: opposite boundary side (can be p_bc == 0)

    Parameters
    ----------
    data : Any
        Loaded NPZ object.

    Returns
    -------
    float
        Relative error or NaN if not computable.

    """
    if not hasattr(data, "files"):
        return float("nan")

    gt = _npz_get(data, "gt", default=None)
    p_bc = _npz_get(data, "p_bc", default=None)
    if gt is None or p_bc is None:
        return float("nan")

    gt2 = np.asarray(gt, dtype=float)
    if gt2.ndim != 3 or gt2.shape[0] < 1:  # noqa: PLR2004
        return float("nan")

    pad = _infer_interior_pad(data)

    p_gt = _crop_interior(gt2[0], pad)
    bc = _crop_interior(_take_2d(p_bc), pad)

    if p_gt.ndim != 2 or bc.ndim != 2 or p_gt.shape != bc.shape:  # noqa: PLR2004
        return float("nan")

    finite = np.isfinite(bc)
    if not np.any(finite):
        return float("nan")

    bc_abs = np.abs(bc)
    bc_max = float(np.nanmax(bc_abs[finite]))
    if not np.isfinite(bc_max) or bc_max <= 0.0:
        return float("nan")

    thr = max(BC_THR_ABS, BC_THR_REL * bc_max)

    inlet_mask, outlet_mask = _infer_inlet_outlet_masks_from_p_bc(bc, thr)
    if not (np.any(inlet_mask) and np.any(outlet_mask)):
        return float("nan")

    dp_bc = float(np.nanmean(bc[inlet_mask]) - np.nanmean(bc[outlet_mask]))
    if not np.isfinite(dp_bc) or abs(dp_bc) <= EPS:
        return float("nan")

    dp_gt = float(np.nanmean(p_gt[inlet_mask]) - np.nanmean(p_gt[outlet_mask]))
    if not np.isfinite(dp_gt):
        return float("nan")

    rel = abs(dp_gt - dp_bc) / (abs(dp_bc) + EPS)
    return float(rel) if np.isfinite(rel) else float("nan")


def _plot_npz_scalar_cdf_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    compute_scalar: Callable[[Any], float],
    xlabel: str,
    title: str,
    gt_compute_scalar: Callable[[Any], float] | None = None,
    gt_label: str = "GT (reference)",
) -> widgets.VBox:
    """
    Create an interactive CDF viewer for a scalar metric computed from NPZ artifacts.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping of dataset names to evaluation DataFrames.
    compute_scalar : Callable[[Any], float]
        Function that computes the scalar metric from a loaded NPZ object.
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    gt_compute_scalar : Callable[[Any], float] | None, optional
        Optional function to compute GT reference scalar from NPZ. If provided,
        the GT curve is plotted alongside model curves.
    gt_label : str, optional
        Legend label for the GT reference curve.

    Returns
    -------
    ipywidgets.VBox
        Interactive CDF viewer widget.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)

    cache: dict[str, dict[str, Any]] = {name: {"loaded_until": 0, "vals": []} for name in names}
    gt_cache: dict[str, Any] = {"loaded_until": 0, "vals": []}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles: list[Any] = []
        labels: list[str] = []

        # model curves
        for name in names:
            df = datasets[name].reset_index(drop=True)

            entry = cache[name]
            loaded = int(entry["loaded_until"])

            if max_cases > loaded and "npz_path" in df.columns:
                vals_list: list[float] = list(entry["vals"])
                nmax = min(max_cases, len(df))

                for i in range(loaded, nmax):
                    npz_path = df.loc[i, "npz_path"]
                    if not isinstance(npz_path, str) or not npz_path:
                        continue

                    p = Path(npz_path)
                    if not p.exists():
                        continue

                    with np.load(p, allow_pickle=True) as z:
                        v = compute_scalar(z)

                    if np.isfinite(v):
                        vals_list.append(float(v))

                entry["vals"] = vals_list
                entry["loaded_until"] = max_cases

            vals = np.asarray(entry["vals"], dtype=float)
            line = _plot_cdf(ax, vals, label=name)
            if line is not None:
                handles.append(line)
                labels.append(name)

        # GT reference curve
        if gt_compute_scalar is not None and "npz_path" in df_ref.columns:
            loaded = int(gt_cache["loaded_until"])
            if max_cases > loaded:
                vals_list = list(gt_cache["vals"])
                nmax = min(max_cases, len(df_ref))

                for i in range(loaded, nmax):
                    npz_path = df_ref.loc[i, "npz_path"]
                    if not isinstance(npz_path, str) or not npz_path:
                        continue

                    p = Path(npz_path)
                    if not p.exists():
                        continue

                    with np.load(p, allow_pickle=True) as z:
                        v = gt_compute_scalar(z)

                    if np.isfinite(v):
                        vals_list.append(float(v))

                gt_cache["vals"] = vals_list
                gt_cache["loaded_until"] = max_cases

            gt_vals = np.asarray(gt_cache["vals"], dtype=float)
            gt_line = _plot_cdf(ax, gt_vals, label=gt_label, color="black", ls="-", zorder=10)
            if gt_line is not None:
                handles.append(gt_line)
                labels.append(gt_label)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if len(handles) > 0:
            ax_leg.legend(handles, labels, loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================
def _get_scalar_vals(df: pd.DataFrame, *, col: str, max_cases: int) -> np.ndarray:
    """
    Extract finite scalar values from a metric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation table.
    col : str
        Column name to extract.
    max_cases : int
        Maximum number of rows used from the beginning of the table.

    Returns
    -------
    numpy.ndarray
        1D array of finite values. Empty if missing or invalid.

    """
    if col not in df.columns:
        return np.asarray([], dtype=float)
    s = pd.to_numeric(df.reset_index(drop=True)[col].iloc[: int(max_cases)], errors="coerce")
    arr = s.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _plot_cdf(
    ax: Any,
    vals: np.ndarray,
    *,
    label: str,
    color: str | None = None,
    ls: str = "-",
    zorder: int = 1,
) -> Any | None:
    """
    Plot a CDF curve on an axis.

    Parameters
    ----------
    ax : Any
        Matplotlib Axes.
    vals : numpy.ndarray
        1D array of values to plot.
    label : str
        Legend label.
    color : str | None, optional
        Line color.
    ls : str, optional
        Line style.
    zorder : int, optional
        Matplotlib z-order.

    Returns
    -------
    Any | None
        Matplotlib Line2D if plotted, otherwise None.

    """
    vals2 = np.asarray(vals, dtype=float)
    vals2 = vals2[np.isfinite(vals2)]
    if vals2.size == 0:
        return None
    x = np.maximum(vals2, EPS)
    x = np.sort(x)
    y = np.linspace(0.0, 1.0, x.size)
    (line,) = ax.plot(x, y, lw=2, label=label, color=color, linestyle=ls, zorder=zorder)
    return line


def _load_npz_fields(npz_path: str | Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """
    Load selected 2D fields from an NPZ artifact.

    Missing files or keys are filled with a NaN placeholder.

    Parameters
    ----------
    npz_path : str | pathlib.Path
        Path to NPZ file.
    keys : tuple[str, ...]
        Keys to load.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping key -> array.

    """
    p = Path(str(npz_path))
    out: dict[str, np.ndarray] = {}
    if not p.exists():
        for k in keys:
            out[k] = _nan2d()
        return out

    with np.load(p, allow_pickle=True) as data:
        for k in keys:
            if hasattr(data, "files") and k in data.files:
                out[k] = np.asarray(data[k], dtype=float)
            else:
                out[k] = _nan2d()
    return out


# =============================================================================
# GENERIC VIEWERS (CDF + mean maps)
# =============================================================================
def _plot_metric_cdf_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    col: str,
    xlabel: str,
    title: str,
    gt_label: str = "GT (reference)",
) -> widgets.VBox:
    """
    Interactive CDF viewer for a scalar physical metric.

    For each dataset (model), a CDF curve is plotted from the evaluation
    DataFrame column `col`. Additionally, a black "GT (reference)" curve
    is computed from the first dataset's NPZ artifacts and shown as an
    independent reference.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - npz_path (for GT reference computation)
            - col (metric column for model curves)
    col : str
        Metric column name (and GT metric identifier).
    xlabel : str
        X-axis label for the CDF plot.
    title : str
        Plot title.
    gt_label : str, optional
        Label for GT reference curve.

    Returns
    -------
    ipywidgets.VBox
        Interactive viewer with a case-count slider.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)
    gt_cache: dict[str, Any] = {"loaded_until": 0, "vals": []}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles: list[Any] = []
        labels: list[str] = []

        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col=col, max_cases=max_cases)
            line = _plot_cdf(ax, vals, label=name)
            if line is not None:
                handles.append(line)
                labels.append(name)

        loaded = int(gt_cache["loaded_until"])
        if max_cases > loaded and _has_df_schema(df_ref):
            vals_list: list[float] = list(gt_cache["vals"])
            nmax = min(max_cases, len(df_ref))
            for i in range(loaded, nmax):
                npz_path = df_ref.loc[i, "npz_path"]
                if isinstance(npz_path, str) and npz_path:
                    v = _compute_gt_metric_from_npz(npz_path, metric=col)
                    if np.isfinite(v):
                        vals_list.append(float(v))
            gt_cache["vals"] = vals_list
            gt_cache["loaded_until"] = max_cases

        gt_vals = np.asarray(gt_cache["vals"], dtype=float)
        gt_line = _plot_cdf(ax, gt_vals, label=gt_label, color="black", ls="-", zorder=10)
        if gt_line is not None:
            handles.append(gt_line)
            labels.append(gt_label)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if len(handles) > 0:
            ax_leg.legend(handles, labels, loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


def _plot_mean_field_map_viewer(
    *,
    datasets: dict[str, pd.DataFrame],
    npz_keys: tuple[str, ...],
    reducer: Callable[[dict[str, np.ndarray]], np.ndarray],
    cbar_label: str,
    title: str,
) -> widgets.VBox:
    """
    Interactive mean-field map viewer based on NPZ field accumulation.

    For each dataset, the viewer loads NPZ fields for an increasing number
    of cases and accumulates a running sum. The displayed map is the mean
    of the accumulated field.

    A robust percentile clipping is applied:
        vmax = percentile(mean_field, CLIP_Q)

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - npz_path
            - geometry_Lx
            - geometry_Ly
    npz_keys : tuple[str, ...]
        NPZ keys required by the reducer.
    reducer : Callable[[dict[str, numpy.ndarray]], numpy.ndarray]
        Function mapping loaded fields to a single 2D field to accumulate.
    cbar_label : str
        Colorbar label.
    title : str
        Figure title.

    Returns
    -------
    ipywidgets.VBox
        Interactive viewer with a case-count slider.

    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown(f"## {title}"))
            print("No datasets.")
        return widgets.VBox([out])

    cache: dict[str, _FieldAcc] = {name: {"loaded_until": 0, "sum_field": None, "count": 0} for name in names}
    geom_cache: dict[str, tuple[float, float]] = {}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        max_cases = int(max_cases)

        fig = plt.figure(figsize=(6 * len(names), 6))
        gs = fig.add_gridspec(1, len(names), wspace=0.25, hspace=0.0)

        for c, name in enumerate(names):
            ax = fig.add_subplot(gs[0, c])
            df = datasets[name].reset_index(drop=True)

            if name not in geom_cache:
                geom_cache[name] = _infer_geom(df)

            entry = cache[name]
            loaded = int(entry["loaded_until"])

            if max_cases > loaded and _has_df_schema(df):
                nmax = min(max_cases, len(df))
                for _, row in df.iloc[loaded:nmax].iterrows():
                    npz_path = row.get("npz_path", None)
                    if not isinstance(npz_path, str) or not npz_path:
                        continue

                    p = Path(npz_path)
                    if not p.exists():
                        continue

                    fields = _load_npz_fields(npz_path, npz_keys)
                    field = np.asarray(reducer(fields), dtype=float)

                    if field.ndim != 2 or not np.any(np.isfinite(field)):  # noqa: PLR2004
                        continue

                    if entry["sum_field"] is None:
                        entry["sum_field"] = np.zeros_like(field, dtype=float)
                    elif entry["sum_field"].shape != field.shape:
                        continue

                    entry["sum_field"] += np.nan_to_num(field, nan=0.0)
                    entry["count"] += 1

                entry["loaded_until"] = max_cases

            if entry["sum_field"] is None or entry["count"] == 0:
                ax.set_title(name)
                ax.axis("off")
                continue

            mean_field = entry["sum_field"] / float(entry["count"])

            vmax = float(np.nanpercentile(mean_field, CLIP_Q))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = float(np.nanmax(mean_field))
                vmax = vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0

            mean_plot = np.ma.masked_greater(mean_field, vmax)

            ny, nx = mean_field.shape
            x, y = _setup_xy_axes(
                ax,
                LxLy=geom_cache[name],
                ny=ny,
                nx=nx,
                is_left=(c == 0),
                is_bottom=True,
            )
            X, Y = np.meshgrid(x, y)

            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("white")

            levels = np.linspace(0.0, vmax, 11)
            im = ax.contourf(X, Y, mean_plot, levels=levels, cmap=cmap)
            ax.set_title(name)

            cb = fig.colorbar(im, ax=ax, fraction=0.025)
            cb.ax.yaxis.set_major_formatter(util.util_plot_components.choose_colorbar_formatter(0.0, vmax))
            cb.set_label(cbar_label)

        fig.suptitle(title, y=0.85)
        fig.subplots_adjust(top=0.95, bottom=0.12, left=0.06, right=0.98, wspace=0.25)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# TABLE
# =============================================================================
def _style_numeric_block_blue(block: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Create a Styler background-color map for numeric columns.

    Values are normalized per column using the 5% and 95% quantiles to
    reduce sensitivity to outliers.

    Parameters
    ----------
    block : pandas.DataFrame
        DataFrame to style.
    columns : list[str]
        Subset of columns to style.

    Returns
    -------
    pandas.DataFrame
        DataFrame of CSS strings matching block shape.

    """
    styled = pd.DataFrame("", index=block.index, columns=block.columns)
    cmap = plt.get_cmap("Blues")

    for col in columns:
        vals_all = pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
        vals = vals_all[np.isfinite(vals_all)]
        if vals.size == 0:
            continue

        qlo, qhi = np.nanquantile(vals, 0.05), np.nanquantile(vals, 0.95)
        vmin, vmax = float(qlo), float(qhi)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax == vmin:
            continue

        for i in block.index:
            v = pd.to_numeric(block.loc[i, col], errors="coerce")
            if not np.isfinite(v):
                continue
            t = float(np.clip((float(v) - vmin) / (vmax - vmin), 0.0, 1.0))
            lo, hi = 0.05, 0.95
            r, g, b, _ = cmap(lo + (hi - lo) * t)
            alpha = 0.55
            styled.loc[i, col] = f"background-color: rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    return styled


def build_physical_consistency_summary_table(
    datasets_eval: dict[str, pd.DataFrame],
    *,
    metrics: tuple[str, ...] = ("cont_mse_divu", "mom_mse", "bc_mse"),
    stats: tuple[str, ...] = ("median", "mean", "q90", "q95"),
    sort_by: str = "cont_mse_divu_median",
) -> pd.DataFrame:
    """
    Build a model-level physical consistency summary table.

    Aggregates per-case scalar metrics into robust summary statistics.

    Parameters
    ----------
    datasets_eval : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Each DataFrame should contain the requested metric columns.
    metrics : tuple[str, ...], optional
        Metric column names to aggregate.
    stats : tuple[str, ...], optional
        Statistics to compute per metric:
            - median
            - mean
            - q90
            - q95
    sort_by : str, optional
        Output column name to sort by (ascending).

    Returns
    -------
    pandas.DataFrame
        Summary table with index "model" and columns like:
            cont_mse_divu_median, mom_mse_q95, ...

    """
    stat_fns: dict[str, Callable[[np.ndarray], float]] = {
        "median": lambda a: float(np.nanmedian(a)),
        "mean": lambda a: float(np.nanmean(a)),
        "q90": lambda a: float(np.nanquantile(a, 0.90)),
        "q95": lambda a: float(np.nanquantile(a, 0.95)),
    }

    rows: list[dict[str, float | str]] = []
    for name, df in datasets_eval.items():
        row: dict[str, float | str] = {"model": name}
        for m in metrics:
            arr = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float) if m in df.columns else np.asarray([], dtype=float)
            for s in stats:
                fn = stat_fns.get(s)
                row[f"{m}_{s}"] = fn(arr) if fn is not None else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows).set_index("model")
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=True)
    return out


def plot_physical_consistency_summary_table(
    *,
    datasets: dict[str, pd.DataFrame],
    title: str = "Physical consistency summary",
    metrics: tuple[str, ...] = ("cont_mse_divu", "mom_mse", "bc_mse"),
    stats: tuple[str, ...] = ("median", "mean", "q90", "q95"),
    sort_by: str = "cont_mse_divu_median",
) -> widgets.VBox:
    """
    Render a styled physical consistency summary table.

    The table is computed by `build_physical_consistency_summary_table`
    and displayed using a blue intensity background per numeric column.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
    title : str, optional
        Title shown above the table.
    metrics : tuple[str, ...], optional
        Metrics to include in the table.
    stats : tuple[str, ...], optional
        Summary statistics per metric.
    sort_by : str, optional
        Column name used for ascending sort.

    Returns
    -------
    ipywidgets.VBox
        Widget containing the rendered table.

    """
    summary = build_physical_consistency_summary_table(
        datasets_eval=datasets,
        metrics=metrics,
        stats=stats,
        sort_by=sort_by,
    )

    out = widgets.Output()
    with out:
        display(Markdown(f"## {title}"))
        cols_to_style = [c for c in summary.columns if pd.api.types.is_numeric_dtype(summary[c])]
        style_df = _style_numeric_block_blue(summary, cols_to_style)
        display(summary.style.format("{:.4g}").apply(lambda _: style_df, axis=None))

    return widgets.VBox([out])


# =============================================================================
# PLOTS (thin wrappers)
# =============================================================================
def plot_velocity_divergence(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive CDF plot of mass conservation residual MSE(div(u)).

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - cont_mse_divu

    Returns
    -------
    ipywidgets.VBox
        Interactive CDF viewer.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="cont_mse_divu",
        xlabel=r"$\mathrm{MSE}(\nabla \cdot \mathbf{u})$",
        title="Mass conservation residual distribution",
    )


def plot_brinkman_residual(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive CDF plot of Darcy-Brinkman momentum residual MSE.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - mom_mse

    Returns
    -------
    ipywidgets.VBox
        Interactive CDF viewer.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="mom_mse",
        xlabel=r"$\mathrm{MSE}(\mathbf{r}_{\mathrm{mom}})$",
        title="Darcy-Brinkman momentum residual distribution",
    )


def plot_pressure_bc_consistency(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive CDF plot of pressure boundary mismatch MSE.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - bc_mse

    Returns
    -------
    ipywidgets.VBox
        Interactive CDF viewer.

    """
    return _plot_metric_cdf_viewer(
        datasets=datasets,
        col="bc_mse",
        xlabel=r"$\mathrm{MSE}(p_{\Gamma} - p_{\mathrm{bc}})$",
        title="Pressure boundary consistency",
    )


def plot_mass_conservation_error_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Mean absolute mass conservation error map.

    Displays, per dataset, the mean of |div_u| accumulated from NPZ artifacts.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - npz_path
            - geometry_Lx
            - geometry_Ly

    Returns
    -------
    ipywidgets.VBox
        Interactive mean map viewer.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        npz_keys=("div_u",),
        reducer=lambda f: np.abs(f["div_u"]),
        cbar_label=r"mean $|\nabla \cdot \mathbf{u}|$",
        title="Mean absolute mass conservation error map",
    )


def plot_brinkman_momentum_residual_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Mean Darcy-Brinkman momentum residual magnitude map.

    Displays, per dataset, the mean of sqrt(Rx^2 + Ry^2) accumulated from NPZ artifacts.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - npz_path
            - geometry_Lx
            - geometry_Ly

    Returns
    -------
    ipywidgets.VBox
        Interactive mean map viewer.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        npz_keys=("Rx", "Ry"),
        reducer=lambda f: np.sqrt(f["Rx"] ** 2 + f["Ry"] ** 2),
        cbar_label=r"mean $\left\|\mathbf{r}_{\mathrm{mom}}\right\|$",
        title="Mean Darcy-Brinkman momentum residual map",
    )


def plot_pressure_drop_consistency(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive CDF plot of relative pressure-drop mismatch.

    inlet: boundary side where p_bc is non-zero
    outlet: opposite boundary side (can be p_bc == 0)

    rel = |dp_pred - dp_bc| / (|dp_bc| + EPS)

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain NPZ artifacts with:
            - p_bc
            - p_model_in
            - p_model_out
            - p_bc_in
            - p_bc_out

    Returns
    -------
    ipywidgets.VBox
        Interactive CDF viewer.

    """
    return _plot_npz_scalar_cdf_viewer(
        datasets=datasets,
        compute_scalar=_compute_dp_rel_err_from_npz,
        gt_compute_scalar=_compute_dp_rel_err_gt_from_npz,
        gt_label="GT (reference)",
        xlabel=r"$\frac{\left|(\bar p_{\mathrm{model,in}}-\bar p_{\mathrm{model,out}})-(\bar p_{\mathrm{BC,in}}-\bar p_{\mathrm{BC,out}})\right|}{\left|(\bar p_{\mathrm{BC,in}}-\bar p_{\mathrm{BC,out}})\right|+\epsilon}$",  # noqa: E501
        title="Pressure drop consistency (relative mismatch)",
    )


def plot_div_phi_u_error_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Mean absolute porosity-weighted continuity residual map.

    Displays, per dataset, the mean of |div_phi_u| accumulated from NPZ artifacts.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping label -> evaluation DataFrame.
        Must contain:
            - npz_path
            - geometry_Lx
            - geometry_Ly

    Returns
    -------
    ipywidgets.VBox
        Interactive mean map viewer.

    """
    return _plot_mean_field_map_viewer(
        datasets=datasets,
        npz_keys=("div_phi_u",),
        reducer=lambda f: np.abs(f["div_phi_u"]),
        cbar_label=r"mean $|\nabla \cdot (\varepsilon \mathbf{u})|$",
        title="Mean absolute porosity-weighted continuity residual map",
    )


def plot_physical_consistency_cdf_grid(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:  # noqa: C901, PLR0915
    """
    2x2 grid of CDF plots with a single legend on the right (outside).

    Panels:
        (0,0) Mass conservation residual (cont_mse_divu)
        (0,1) Darcy-Brinkman momentum residual (mom_mse)
        (1,0) Pressure drop consistency (relative mismatch, NPZ-based)
        (1,1) Pressure boundary consistency (bc_mse)
    """
    names = list(datasets.keys())
    if len(names) == 0:
        out = widgets.Output()
        with out:
            display(Markdown("## Physical consistency (CDF grid)"))
            print("No datasets.")
        return widgets.VBox([out])

    df_ref = datasets[names[0]].reset_index(drop=True)

    # per-model cache for NPZ-based dp metric
    dp_cache: dict[str, dict[str, Any]] = {name: {"loaded_until": 0, "vals": []} for name in names}

    # GT cache (computed once from df_ref NPZs)
    gt_cache: dict[str, Any] = {
        "loaded_until": 0,
        "cont": [],
        "mom": [],
        "bc": [],
        "dp": [],
    }

    # stable color mapping across all 4 panels
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = []
    if cycle is not None:
        colors = cycle.by_key().get("color", [])
    if len(colors) == 0:
        colors = [f"C{i}" for i in range(10)]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(names)}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        2x2 grid of CDF plots with a single legend on the right (outside).

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to include from the start of each dataset.
        datasets : dict[str, pandas.DataFrame]
            Mapping label -> evaluation DataFrame.

        Returns
        -------
        Figure
            Matplotlib Figure with the 2x2 CDF grid.

        """
        max_cases = int(max_cases)

        # ----------------------------
        # update per-model dp caches
        # ----------------------------
        for name in names:
            df = datasets[name].reset_index(drop=True)
            entry = dp_cache[name]
            loaded = int(entry["loaded_until"])

            if max_cases > loaded and "npz_path" in df.columns:
                vals_list: list[float] = list(entry["vals"])
                nmax = min(max_cases, len(df))

                for i in range(loaded, nmax):
                    npz_path = df.loc[i, "npz_path"]
                    if not isinstance(npz_path, str) or not npz_path:
                        continue
                    p = Path(npz_path)
                    if not p.exists():
                        continue

                    with np.load(p, allow_pickle=True) as z:
                        v = _compute_dp_rel_err_from_npz(z)

                    if np.isfinite(v):
                        vals_list.append(float(v))

                entry["vals"] = vals_list
                entry["loaded_until"] = max_cases

        # ----------------------------
        # update GT cache (single pass)
        # ----------------------------
        loaded = int(gt_cache["loaded_until"])
        if max_cases > loaded and "npz_path" in df_ref.columns:
            cont_list: list[float] = list(gt_cache["cont"])
            mom_list: list[float] = list(gt_cache["mom"])
            bc_list: list[float] = list(gt_cache["bc"])
            dp_list: list[float] = list(gt_cache["dp"])

            nmax = min(max_cases, len(df_ref))
            for i in range(loaded, nmax):
                npz_path = df_ref.loc[i, "npz_path"]
                if not isinstance(npz_path, str) or not npz_path:
                    continue
                p = Path(npz_path)
                if not p.exists():
                    continue

                with np.load(p, allow_pickle=True) as z:
                    v_cont = _compute_gt_cont_mse_divu_from_npz(z)
                    v_mom = _compute_gt_mom_mse_from_npz(z)
                    v_bc = _compute_gt_bc_mse_from_npz(z)
                    v_dp = _compute_dp_rel_err_gt_from_npz(z)

                if np.isfinite(v_cont):
                    cont_list.append(float(v_cont))
                if np.isfinite(v_mom):
                    mom_list.append(float(v_mom))
                if np.isfinite(v_bc):
                    bc_list.append(float(v_bc))
                if np.isfinite(v_dp):
                    dp_list.append(float(v_dp))

            gt_cache["cont"] = cont_list
            gt_cache["mom"] = mom_list
            gt_cache["bc"] = bc_list
            gt_cache["dp"] = dp_list
            gt_cache["loaded_until"] = max_cases

            # ----------------------------
        # build 2x2 + legend column (right, legend at top)
        # ----------------------------
        fig = plt.figure(figsize=(15.5, 8.5))
        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.0, 1.0, 0.30],
            wspace=0.25,
            hspace=0.30,
        )

        ax_mass = fig.add_subplot(gs[0, 0])
        ax_mom = fig.add_subplot(gs[0, 1])
        ax_dp = fig.add_subplot(gs[1, 0])
        ax_bc = fig.add_subplot(gs[1, 1])

        ax_leg = fig.add_subplot(gs[:, 2])
        ax_leg.axis("off")

        # helper to format axes uniformly
        def _format_ax(ax: Any, *, xlabel: str, title: str) -> None:
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CDF")
            ax.set_title(title)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # (0,0) mass conservation (cont_mse_divu)
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="cont_mse_divu", max_cases=max_cases)
            _plot_cdf(ax_mass, vals, label=name, color=color_map[name])
        _plot_cdf(
            ax_mass,
            np.asarray(gt_cache["cont"], dtype=float),
            label="GT (reference)",
            color="black",
            ls="-",
            zorder=10,
        )
        _format_ax(
            ax_mass,
            xlabel=r"$\mathrm{MSE}(\nabla \cdot \mathbf{u})$",
            title="Mass conservation residual distribution",
        )

        # (0,1) momentum residual (mom_mse)
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="mom_mse", max_cases=max_cases)
            _plot_cdf(ax_mom, vals, label=name, color=color_map[name])
        _plot_cdf(
            ax_mom,
            np.asarray(gt_cache["mom"], dtype=float),
            label="GT (reference)",
            color="black",
            ls="-",
            zorder=10,
        )
        _format_ax(
            ax_mom,
            xlabel=r"$\mathrm{MSE}(\mathbf{r}_{\mathrm{mom}})$",
            title="Darcy-Brinkman momentum residual distribution",
        )

        # (1,0) pressure drop consistency (NPZ-based)
        for name in names:
            vals = np.asarray(dp_cache[name]["vals"], dtype=float)
            _plot_cdf(ax_dp, vals, label=name, color=color_map[name])
        _plot_cdf(
            ax_dp,
            np.asarray(gt_cache["dp"], dtype=float),
            label="GT (reference)",
            color="black",
            ls="-",
            zorder=10,
        )
        _format_ax(
            ax_dp,
            xlabel=r"$\frac{\left|(\bar p_{\mathrm{model,in}}-\bar p_{\mathrm{model,out}})-(\bar p_{\mathrm{BC,in}}-\bar p_{\mathrm{BC,out}})\right|}{\left|(\bar p_{\mathrm{BC,in}}-\bar p_{\mathrm{BC,out}})\right|+\epsilon}$",  # noqa: E501
            title="Pressure drop consistency (relative mismatch)",
        )

        # (1,1) pressure BC consistency (bc_mse)
        for name in names:
            df = datasets[name]
            vals = _get_scalar_vals(df, col="bc_mse", max_cases=max_cases)
            _plot_cdf(ax_bc, vals, label=name, color=color_map[name])
        _plot_cdf(
            ax_bc,
            np.asarray(gt_cache["bc"], dtype=float),
            label="GT (reference)",
            color="black",
            ls="-",
            zorder=10,
        )
        _format_ax(
            ax_bc,
            xlabel=r"$\mathrm{MSE}(p_{\Gamma} - p_{\mathrm{bc}})$",
            title="Pressure boundary consistency",
        )

        # ----------------------------
        # legend (right, top)
        # ----------------------------
        legend_handles = [Line2D([0], [0], color=color_map[name], lw=2, label=name) for name in names]
        legend_handles.append(Line2D([0], [0], color="black", lw=2, label="GT (reference)"))
        legend_labels: list[str] = [str(h.get_label()) for h in legend_handles]
        ax_leg.legend(legend_handles, legend_labels, loc="upper left", frameon=True)

        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )
