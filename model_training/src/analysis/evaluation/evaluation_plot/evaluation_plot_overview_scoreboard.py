"""
Overview and model-level comparison plots for PINO/FNO/UNO evaluation.

This module provides high-level summary plots intended for:
    - model comparison
    - tradeoff analysis (accuracy vs physics)
    - decision support (which model class dominates)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from src.schema.schema_fields import OUTPUT_FIELDS

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

# =============================================================================
# Channels
# =============================================================================
CHANNELS = list(OUTPUT_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

# =============================================================================
# Constants (MUST match training & physics plots)
# =============================================================================
MU_AIR = 1.8139e-5
EPS_DET = 1e-4
EPS = 1e-12


# =============================================================================
# NPZ loading (local copy)
# =============================================================================
def _load_npz(row: pd.Series) -> dict[str, Any]:
    """
    Load prediction data from NPZ file.

    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing 'npz_path'.

    Returns
    -------
    dict
        Dictionary with pred, gt, kappa, kappa_names, p_bc, meta.

    """
    data = np.load(row["npz_path"], allow_pickle=True)

    return {
        "pred": np.asarray(data["pred"]),
        "gt": np.asarray(data["gt"]),
        "kappa": np.asarray(data["kappa"]),
        "kappa_names": [str(n) for n in data["kappa_names"]],
        "p_bc": np.asarray(data["p_bc"]),
        "meta": data.get("meta", {}),
    }


# =============================================================================
# Physics: Darcy-Brinkman residual (NumPy, identical to PINOLoss)
# =============================================================================
def _compute_brinkman_residual(
    *,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
    kappa_names: list[str],
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """
    Compute Darcy-Brinkman residual magnitude |R|.

    R = -∇p + div(mu * viscous stress) - mu * K^{-1} u
    """
    ny, nx = u.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # --------------------------------------------------
    # Gradients
    # --------------------------------------------------
    dpdx = np.gradient(p, dx, axis=1)
    dpdy = np.gradient(p, dy, axis=0)

    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)

    # --------------------------------------------------
    # Viscous stress (phi = 1)
    # --------------------------------------------------
    mu = MU_AIR

    Kxx = mu * (2.0 * dudx) - (2.0 / 3.0) * mu * (dudx + dvdy)
    Kyy = mu * (2.0 * dvdy) - (2.0 / 3.0) * mu * (dudx + dvdy)
    Kxy = mu * (dudy + dvdx)

    divKx = np.gradient(Kxx, dx, axis=1) + np.gradient(Kxy, dy, axis=0)
    divKy = np.gradient(Kxy, dx, axis=1) + np.gradient(Kyy, dy, axis=0)

    # --------------------------------------------------
    # Permeability tensor handling
    # --------------------------------------------------
    idx = {name: i for i, name in enumerate(kappa_names)}

    kxx = kappa[idx["kxx"]]
    kyy = kappa[idx["kyy"]]
    kxy_rel = kappa[idx["kxy"]] if "kxy" in idx else 0.0

    K0 = np.sqrt(np.maximum(kxx * kyy, 1e-30))

    kxx_hat = kxx / K0
    kyy_hat = kyy / K0
    kxy_hat = np.clip(kxy_rel, -0.99, 0.99)

    det_hat = kxx_hat * kyy_hat - kxy_hat**2
    det_hat = np.maximum(det_hat, EPS_DET)

    inv_xx = (kyy_hat / det_hat) / K0
    inv_xy = (-kxy_hat / det_hat) / K0
    inv_yy = (kxx_hat / det_hat) / K0

    drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
    drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

    # --------------------------------------------------
    # Residual
    # --------------------------------------------------
    Rx = -dpdx + divKx - drag_x
    Ry = -dpdy + divKy - drag_y

    return np.sqrt(Rx**2 + Ry**2)


# =============================================================================
# Scalar physics metric
# =============================================================================
def _median_physics_residual(df: pd.DataFrame, max_cases: int = 100) -> float:
    """
    Compute median normalised physics residual over cases.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing evaluation cases.
    max_cases : int, optional
        Maximum number of cases to consider, by default 100.

    Returns
    -------
    float
        Median normalised physics residual.

    """
    vals: list[float] = []

    df_i = df.reset_index(drop=True).iloc[:max_cases]

    for _, row in df_i.iterrows():
        data = _load_npz(row)

        pred = data["pred"]
        kappa = data["kappa"]
        kappa_names = data["kappa_names"]

        Lx = float(row["geometry_Lx"])
        Ly = float(row["geometry_Ly"])
        L = max(Lx, Ly)

        p = pred[CHANNEL_INDICES["p"]]
        u = pred[CHANNEL_INDICES["u"]]
        v = pred[CHANNEL_INDICES["v"]]

        R = _compute_brinkman_residual(
            p=p,
            u=u,
            v=v,
            kappa=kappa,
            kappa_names=kappa_names,
            Lx=Lx,
            Ly=Ly,
        )

        U = np.sqrt(u**2 + v**2)
        denom = max(np.mean(U) / (L**2), 1e-6)

        Rnorm = np.mean(R) / denom
        if np.isfinite(Rnorm):
            vals.append(float(Rnorm))

    return float(np.median(vals)) if vals else float("nan")


# =============================================================================
# Overview plots
# =============================================================================
def plot_overview_scoreboard(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Overview scoreboard plot comparing multiple evaluation groups.

    Each entry in `datasets` represents one comparison item
    (e.g. model, dataset).

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.

    Returns
    -------
    matplotlib.figure.Figure
        The generated scoreboard figure.

    """
    names = list(datasets.keys())

    rel_l2 = []
    l2 = []
    phys = []

    for df in datasets.values():
        rel_l2.append(float(np.median(df["rel_l2"])))
        l2.append(float(np.median(df["l2"])))
        phys.append(_median_physics_residual(df))

    metrics = {
        "rel_l2": np.asarray(rel_l2),
        "l2": np.asarray(l2),
        "physics": np.asarray(phys),
    }

    norm = {}
    for k, arr in metrics.items():
        ref = np.nanmin(arr)  # best value (lower is better)
        norm[k] = arr / (ref + EPS)

    fig_width = max(10.0, 3.5 * len(names))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(len(names))
    w = 0.25

    ax.bar(x - w, norm["rel_l2"], w, label="rel L2")
    ax.bar(x, norm["l2"], w, label="L2")
    ax.bar(x + w, norm["physics"], w, label="Physics residual")

    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Relative score (x best, lower is better)")
    ax.set_title("Global comparison scoreboard\n(relative to best entry)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.30,
        top=0.90,
    )
    return fig


def plot_overview_pareto_error_vs_physics(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Overview Pareto plot: accuracy vs physics consistency.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Dictionary mapping model names to their evaluation DataFrames.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Pareto figure.

    """
    fig, ax = plt.subplots(figsize=(16, 8))

    for name, df in datasets.items():
        x = float(np.median(df["rel_l2"]))
        y = _median_physics_residual(df)

        ax.scatter(x, y, s=80)
        ax.text(x, y, f" {name}", va="center", fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median relative L2 error")
    ax.set_ylabel("Median physics residual")
    ax.set_title("Pareto: accuracy vs physics consistency")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.subplots_adjust(
        left=0.15,
        right=0.75,
        bottom=0.15,
        top=0.90,
    )

    return fig
