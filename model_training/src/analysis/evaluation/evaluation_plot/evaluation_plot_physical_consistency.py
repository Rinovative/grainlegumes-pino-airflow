"""
Physical consistency plots for PINO/FNO evaluation.

This module checks whether model predictions are physically admissible,
independently of classical error metrics.

Implemented checks:
    3-1) Velocity divergence ∇·u (mass conservation)
    3-2) Mean mass conservation error map
    3-3) Pressure BC mismatch p|_Γ - p_bc
    3-4) Darcy-Brinkman operator residual
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np

from src import util
from src.schema.schema_fields import OUTPUT_FIELDS

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure

# =============================================================================
# IMPORTS
# ============================================================================
CHANNELS = list(OUTPUT_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

# =============================================================================
# TYPES
# =============================================================================


class MassDivCacheEntry(TypedDict):
    """
    Cache entry for velocity divergence computation.

    Attributes
    ----------
    loaded_until : int
        Number of cases loaded so far.
    sum_div : np.ndarray | None
        Accumulated sum of divergence fields.
    count : int
        Number of cases accumulated.

    """

    loaded_until: int
    sum_div: np.ndarray | None
    count: int


class NPZEntry(TypedDict):
    """
    Loaded data from .npz file.

    Attributes
    ----------
    pred : np.ndarray
        Predicted field.
    gt : np.ndarray
        Ground truth field.
    kappa : np.ndarray
        Permeability tensor components.
    kappa_names : list[str]
        Names of kappa components.
    p_bc : np.ndarray
        Pressure boundary condition.
    meta : Any
        Additional metadata.

    """

    pred: np.ndarray
    gt: np.ndarray
    kappa: np.ndarray
    kappa_names: list[str]
    p_bc: np.ndarray
    meta: Any


class ResidualCacheEntry(TypedDict):
    """
    Cache entry for residual computation.

    Attributes
    ----------
    loaded_until : int
        Number of cases loaded so far.
    vals : list[float]
        List of computed residuals.

    """

    loaded_until: int
    vals: list[float]


# =============================================================================
# CONSTANTS
# =============================================================================

EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================


def _load_npz(row: pd.Series) -> NPZEntry:
    """
    Load prediction data from .npz file specified in the DataFrame row.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing 'npz_path'.

    Returns
    -------
    NPZEntry
        Loaded data.

    """
    data = np.load(row["npz_path"], allow_pickle=True)

    pred = np.asarray(data["pred"])
    gt = np.asarray(data["gt"])
    kappa = np.asarray(data["kappa"])
    p_bc = np.asarray(data["p_bc"])

    return {
        "pred": pred,
        "gt": gt,
        "kappa": kappa,
        "kappa_names": [str(n) for n in data["kappa_names"]],
        "p_bc": p_bc,
        "meta": data.get("meta", {}),
    }


def _compute_divergence(u: np.ndarray, v: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Compute discrete divergence ∇·u = du/dx + dv/dy using central differences.

    Parameters
    ----------
    u, v : np.ndarray
        Velocity components [H, W]
    Lx, Ly : float
        Domain size

    Returns
    -------
    np.ndarray
        Divergence field [H, W]

    """
    ny, nx = u.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    return du_dx + dv_dy


# =============================================================================
# 3-1. VELOCITY DIVERGENCE
# =============================================================================


def plot_velocity_divergence(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot distribution of normalised velocity divergence for different datasets.

    Normalised divergence:  ⟨|∇·u|⟩  /  (⟨|u|⟩ / L)
    where L = max(Lx, Ly)

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to evaluate.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """
    names = list(datasets.keys())

    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "vals_pred": [],
            "vals_gt": [],
        }
        for name in names
    }

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Plot normalised velocity divergence CDF.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to load.
        datasets : dict[str, pd.DataFrame]
            Datasets to evaluate.

        Returns
        -------
        Figure
            Matplotlib figure.

        """
        max_cases = int(max_cases)
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]
                gt = data["gt"]
                kappa = data["kappa"]
                kappa_names = data["kappa_names"]

                Lx = float(row["geometry_Lx"])
                Ly = float(row["geometry_Ly"])
                L = max(Lx, Ly)

                # ======================
                # PRED
                # ======================
                p_p = pred[CHANNEL_INDICES["p"]]
                u_p = pred[CHANNEL_INDICES["u"]]
                v_p = pred[CHANNEL_INDICES["v"]]

                R_p = _compute_brinkman_residual(
                    p=p_p,
                    u=u_p,
                    v=v_p,
                    kappa=kappa,
                    kappa_names=kappa_names,
                    Lx=Lx,
                    Ly=Ly,
                )

                U_p = np.sqrt(u_p**2 + v_p**2)
                denom_p = max(np.mean(U_p) / (L**2), 1e-6)
                Rnorm_p = np.mean(R_p) / denom_p

                if np.isfinite(Rnorm_p):
                    entry["vals_pred"].append(float(Rnorm_p))

                # ======================
                # GT
                # ======================
                p_g = gt[CHANNEL_INDICES["p"]]
                u_g = gt[CHANNEL_INDICES["u"]]
                v_g = gt[CHANNEL_INDICES["v"]]

                R_g = _compute_brinkman_residual(
                    p=p_g,
                    u=u_g,
                    v=v_g,
                    kappa=kappa,
                    kappa_names=kappa_names,
                    Lx=Lx,
                    Ly=Ly,
                )

                U_g = np.sqrt(u_g**2 + v_g**2)
                denom_g = max(np.mean(U_g) / (L**2), 1e-6)
                Rnorm_g = np.mean(R_g) / denom_g

                if np.isfinite(Rnorm_g):
                    entry["vals_gt"].append(float(Rnorm_g))

            entry["loaded_until"] = max_cases

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.0, 0.35],
            wspace=0.25,
        )

        ax = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis("off")

        handles = []
        labels = []

        for name in names:
            vals_p = np.asarray(cache[name]["vals_pred"], dtype=float)
            vals_g = np.asarray(cache[name]["vals_gt"], dtype=float)

            if vals_p.size > 0:
                s = np.sort(vals_p)
                y = np.linspace(0, 1, len(s))
                (lp,) = ax.plot(s, y, lw=2)
                handles.append(lp)
                labels.append(f"{name} (pred)")

            if vals_g.size > 0:
                s = np.sort(vals_g)
                y = np.linspace(0, 1, len(s))
                (lg,) = ax.plot(s, y, lw=2, ls="--")
                handles.append(lg)
                labels.append(f"{name} (gt)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\langle |\nabla \cdot \mathbf{u}| \rangle \,/\, (\langle |\mathbf{u}| \rangle / L)$")
        ax.set_ylabel("CDF")
        ax.set_title("Normalised velocity divergence distribution")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        ax_legend.legend(
            handles,
            labels,
            title="Dataset",
            loc="upper left",
        )

        fig.subplots_adjust(
            top=0.90,
            bottom=0.15,
            left=0.05,
            right=0.98,
        )

        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 3-2. MASS CONSERVATION ERROR MAP
# =============================================================================
def plot_mass_conservation_error_map(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot mean mass conservation error map (mean ∇·u) for different datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to evaluate.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """
    names = list(datasets.keys())

    cache: dict[str, MassDivCacheEntry] = {
        name: {
            "loaded_until": 0,
            "sum_div": None,
            "count": 0,
        }
        for name in names
    }

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Plot mean mass conservation error maps.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to load.
        datasets : dict[str, pd.DataFrame]
            Datasets to evaluate.

        Returns
        -------
        Figure
            Matplotlib figure.

        """
        max_cases = int(max_cases)
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), squeeze=False)
        axes = axes[0]

        for ax, (name, df) in zip(axes, datasets.items(), strict=False):
            entry = cache[name]
            loaded = entry["loaded_until"]

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]

                u = pred[CHANNEL_INDICES["u"]]
                v = pred[CHANNEL_INDICES["v"]]

                Lx = float(row["geometry_Lx"])
                Ly = float(row["geometry_Ly"])

                div = _compute_divergence(u, v, Lx, Ly)

                if entry["sum_div"] is None:
                    entry["sum_div"] = np.zeros_like(div)

                entry["sum_div"] += div
                entry["count"] += 1

            entry["loaded_until"] = max_cases

            if entry["sum_div"] is None or entry["count"] == 0:
                continue

            mean_div = entry["sum_div"] / entry["count"]

            im = ax.imshow(
                mean_div,
                cmap="RdBu_r",
                origin="lower",
            )
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            cb = fig.colorbar(im, ax=ax, fraction=0.03)
            cb.ax.yaxis.set_major_formatter(util.util_plot_components.choose_colorbar_formatter(*im.get_clim()))
            cb.set_label(r"mean $\nabla \cdot \mathbf{u}$")

        fig.suptitle("Mean mass conservation error map", y=0.85)
        fig.subplots_adjust(
            top=0.95,
            bottom=0.07,
            left=0.001,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 3-3. PRESSURE BOUNDARY CONSISTENCY
# =============================================================================


def plot_pressure_bc_consistency(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDF of pressure boundary condition consistency error.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to evaluate.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """
    names = list(datasets.keys())

    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "vals": [],
        }
        for name in names
    }

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Plot pressure boundary consistency CDF.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to load.
        datasets : dict[str, pd.DataFrame]
            Datasets to evaluate.

        Returns
        -------
        Figure
            Matplotlib figure.

        """
        max_cases = int(max_cases)
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]

                p = pred[0]
                p_bc = data["p_bc"]

                p_in = p[0, :]
                bc_in = p_bc[0, :]

                mismatch = np.mean(np.abs(p_in - bc_in))

                entry["vals"].append(mismatch)

            entry["loaded_until"] = max_cases

        # --------------------------------------------------
        # Plot (legend LEFT)
        # --------------------------------------------------
        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.0, 0.35],
            wspace=0.25,
        )

        ax = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis("off")

        legend_handles = []

        for name in names:
            vals = np.array(cache[name]["vals"], dtype=float)
            if vals.size == 0:
                continue

            s = np.sort(vals)
            y = np.linspace(0, 1, len(s))

            (line,) = ax.plot(s, y, lw=2)
            legend_handles.append(line)

        ax.set_xscale("log")
        ax.set_xlabel("|p_pred(inlet) - p_bc|")
        ax.set_ylabel("CDF")
        ax.set_title("Pressure boundary consistency")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        ax_legend.legend(
            legend_handles,
            names,
            title="Dataset",
            loc="upper left",
        )

        fig.subplots_adjust(
            top=0.90,
            bottom=0.15,
            left=0.05,
            right=0.98,
        )

        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 3-4. DARCY-BRINKMAN OPERATOR RESIDUAL (CONSISTENT WITH PINOLoss)
# =============================================================================

MU_AIR = 1.8139e-5  # Pa*s, MUSS identisch zum Training sein
EPS_DET = 1e-4
EPS = 1e-12


def _compute_brinkman_residual(
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
    kappa_names: list[str],
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """
    Darcy-Brinkman residual consistent with PINOLoss (NumPy version).

    R = -∇p + div(μ * viscous stress) - μ * K^{-1} u
    """
    ny, nx = u.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # ------------------------------------------------------------
    # 1) Gradients
    # ------------------------------------------------------------
    dpdx = np.gradient(p, dx, axis=1)
    dpdy = np.gradient(p, dy, axis=0)

    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)

    # ------------------------------------------------------------
    # 2) Viscous stress tensor (phi = 1)
    # ------------------------------------------------------------
    mu = MU_AIR

    Kxx = mu * (2.0 * dudx) - (2.0 / 3.0) * mu * (dudx + dvdy)
    Kyy = mu * (2.0 * dvdy) - (2.0 / 3.0) * mu * (dudx + dvdy)
    Kxy = mu * (dudy + dvdx)

    divKx = np.gradient(Kxx, dx, axis=1) + np.gradient(Kxy, dy, axis=0)
    divKy = np.gradient(Kxy, dx, axis=1) + np.gradient(Kyy, dy, axis=0)

    # ------------------------------------------------------------
    # 3) Kappa handling
    # ------------------------------------------------------------
    idx = {name: i for i, name in enumerate(kappa_names)}

    kxx = kappa[idx["kxx"]]
    kyy = kappa[idx["kyy"]]
    kxy_rel = kappa[idx["kxy"]] if "kxy" in idx else 0.0

    K0 = np.sqrt(np.maximum(kxx * kyy, 1e-30))

    kxx_hat = kxx / K0
    kyy_hat = kyy / K0
    kxy_hat = np.clip(kxy_rel, -0.99, 0.99)

    det_hat = kxx_hat * kyy_hat - kxy_hat**2
    det_hat_safe = np.maximum(det_hat, EPS_DET)

    invhat_xx = kyy_hat / det_hat_safe
    invhat_xy = -kxy_hat / det_hat_safe
    invhat_yy = kxx_hat / det_hat_safe

    inv_xx = invhat_xx / K0
    inv_xy = invhat_xy / K0
    inv_yy = invhat_yy / K0

    # ------------------------------------------------------------
    # 4) Darcy drag
    # ------------------------------------------------------------
    drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
    drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

    # ------------------------------------------------------------
    # 5) Residual
    # ------------------------------------------------------------
    Rx = -dpdx + divKx - drag_x
    Ry = -dpdy + divKy - drag_y

    return np.sqrt(Rx**2 + Ry**2)


def plot_brinkman_residual(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDF of normalised Darcy-Brinkman residual for different datasets.

    Normalised residual:
        ⟨R⟩ / (⟨|u|⟩ / L²)

    Pred vs GT are shown explicitly.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to evaluate.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """
    names = list(datasets.keys())

    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "vals_pred": [],
            "vals_gt": [],
        }
        for name in names
    }

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        """
        Plot normalised Darcy-Brinkman residual CDF.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to load.
        datasets : dict[str, pd.DataFrame]
            Datasets to evaluate.

        Returns
        -------
        Figure
            Matplotlib figure.

        """
        max_cases = int(max_cases)

        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)

                pred = data["pred"]
                gt = data["gt"]
                kappa = data["kappa"]
                kappa_names = data["kappa_names"]

                Lx = float(row["geometry_Lx"])
                Ly = float(row["geometry_Ly"])
                L = max(Lx, Ly)

                # ======================
                # PRED
                # ======================
                p_p = pred[CHANNEL_INDICES["p"]]
                u_p = pred[CHANNEL_INDICES["u"]]
                v_p = pred[CHANNEL_INDICES["v"]]

                R_p = _compute_brinkman_residual(
                    p=p_p,
                    u=u_p,
                    v=v_p,
                    kappa=kappa,
                    kappa_names=kappa_names,
                    Lx=Lx,
                    Ly=Ly,
                )

                U_p = np.sqrt(u_p**2 + v_p**2)
                denom_p = max(np.mean(U_p) / (L**2), 1e-6)
                Rnorm_p = np.mean(R_p) / denom_p

                if np.isfinite(Rnorm_p):
                    entry["vals_pred"].append(float(Rnorm_p))

                # ======================
                # GT
                # ======================
                p_g = gt[CHANNEL_INDICES["p"]]
                u_g = gt[CHANNEL_INDICES["u"]]
                v_g = gt[CHANNEL_INDICES["v"]]

                R_g = _compute_brinkman_residual(
                    p=p_g,
                    u=u_g,
                    v=v_g,
                    kappa=kappa,
                    kappa_names=kappa_names,
                    Lx=Lx,
                    Ly=Ly,
                )

                U_g = np.sqrt(u_g**2 + v_g**2)
                denom_g = max(np.mean(U_g) / (L**2), 1e-6)
                Rnorm_g = np.mean(R_g) / denom_g

                if np.isfinite(Rnorm_g):
                    entry["vals_gt"].append(float(Rnorm_g))

            entry["loaded_until"] = max_cases

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles = []
        labels = []

        for name in names:
            vals_p = np.asarray(cache[name]["vals_pred"], dtype=float)
            vals_g = np.asarray(cache[name]["vals_gt"], dtype=float)

            if vals_p.size > 0:
                s = np.sort(vals_p)
                y = np.linspace(0, 1, len(s))
                (lp,) = ax.plot(s, y, lw=2)
                handles.append(lp)
                labels.append(f"{name} (pred)")

            if vals_g.size > 0:
                s = np.sort(vals_g)
                y = np.linspace(0, 1, len(s))
                (lg,) = ax.plot(s, y, lw=2, ls="--")
                handles.append(lg)
                labels.append(f"{name} (gt)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\langle R \rangle \,/\, (\langle |\mathbf{u}| \rangle / L^2)$")
        ax.set_ylabel("CDF")
        ax.set_title("Normalised Darcy-Brinkman residual distribution")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        ax_leg.legend(handles, labels, title="Dataset", loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )
