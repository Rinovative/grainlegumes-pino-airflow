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

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure

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
    kappa : np.ndarray
        Permeability tensor components.
    kappa_names : list[str]
        Names of kappa components.
    meta : Any
        Additional metadata.

    """

    pred: np.ndarray
    kappa: np.ndarray
    kappa_names: list[str]
    meta: Any


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
        DataFrame row containing 'npz_path' column.

    Returns
    -------
    dict[str, np.ndarray]
        Loaded data including 'pred' and optional 'meta'.

    """
    data = np.load(row["npz_path"], allow_pickle=True)
    return {
        "pred": np.asarray(data["pred"][0]),
        "kappa": np.asarray(data["kappa"][0]),
        "kappa_names": [str(n) for n in data["kappa_names"]],
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


def _compute_brinkman_residual(
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
    kappa_names: list[str],
    Lx: float,
    Ly: float,
    mu_eff: float = 1.0,
) -> np.ndarray:
    """
    Compute tensor-consistent Darcy-Brinkman residual magnitude.

    Residual:
        R = -∇p + μ Δu - κ^{-1} · u

    Parameters
    ----------
    p : np.ndarray
        Pressure field [H, W]
    u, v : np.ndarray
        Velocity components [H, W]
    kappa : np.ndarray
        Permeability tensor components [C_kappa, H, W]
    kappa_names : list[str]
        Names of kappa components (e.g. kappaxx, kappaxy, kappayx, kappayy)
    Lx, Ly : float
        Domain size
    mu_eff : float
        Effective viscosity (scaling only)

    Returns
    -------
    np.ndarray
        Residual magnitude field [H, W]

    """
    ny, nx = u.shape
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # --------------------------------------------------
    # Gradients
    # --------------------------------------------------
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)

    d2u_dx2 = np.gradient(np.gradient(u, dx, axis=1), dx, axis=1)
    d2u_dy2 = np.gradient(np.gradient(u, dy, axis=0), dy, axis=0)
    d2v_dx2 = np.gradient(np.gradient(v, dx, axis=1), dx, axis=1)
    d2v_dy2 = np.gradient(np.gradient(v, dy, axis=0), dy, axis=0)

    lap_u = d2u_dx2 + d2u_dy2
    lap_v = d2v_dx2 + d2v_dy2

    # --------------------------------------------------
    # Assemble κ tensor (2D)
    # --------------------------------------------------
    name_to_idx = {n.lower(): i for i, n in enumerate(kappa_names)}

    kxx = kappa[name_to_idx["kappaxx"]]
    kyy = kappa[name_to_idx["kappayy"]]

    kxy = kappa[name_to_idx.get("kappaxy", name_to_idx["kappaxx"])]
    kyx = kappa[name_to_idx.get("kappayx", name_to_idx["kappayy"])]

    # Determinant
    det = kxx * kyy - kxy * kyx
    det = np.maximum(det, EPS)

    # Inverse tensor components
    inv_kxx = kyy / det
    inv_kyy = kxx / det
    inv_kxy = -kxy / det
    inv_kyx = -kyx / det

    # --------------------------------------------------
    # Darcy term: κ^{-1} · u
    # --------------------------------------------------
    darcy_x = inv_kxx * u + inv_kxy * v
    darcy_y = inv_kyx * u + inv_kyy * v

    # --------------------------------------------------
    # Residual
    # --------------------------------------------------
    Rx = -dp_dx + mu_eff * lap_u - darcy_x
    Ry = -dp_dy + mu_eff * lap_v - darcy_y

    return np.sqrt(Rx**2 + Ry**2)


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
            "vals": [],
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
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]

                # channels: p, u, v
                u = pred[1]
                v = pred[2]

                Lx = float(row["geom_Lx"])
                Ly = float(row["geom_Ly"])

                div = _compute_divergence(u, v, Lx, Ly)

                L = max(Lx, Ly)
                U_mean = np.mean(np.sqrt(u**2 + v**2))

                div_rel = np.abs(div).mean() / (U_mean / L + EPS)

                entry["vals"].append(div_rel)

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
        ax.set_xlabel(r"$\langle |\nabla \cdot \mathbf{u}| \rangle \,/\, (\langle |\mathbf{u}| \rangle / L)$")
        ax.set_ylabel("CDF")
        ax.set_title("Normalised velocity divergence distribution")
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
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), squeeze=False)
        axes = axes[0]

        for ax, (name, df) in zip(axes, datasets.items(), strict=False):
            entry = cache[name]
            loaded = entry["loaded_until"]

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]

                u = pred[1]
                v = pred[2]

                Lx = float(row["geom_Lx"])
                Ly = float(row["geom_Ly"])

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
            top=0.97,
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
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]

                p = pred[0]  # pressure field
                p_bc = float(row.get("par_p_bc", np.nan))

                # inlet boundary at y = 0 (bottom boundary)
                p_in = p[0, :]

                mismatch = np.abs(p_in.mean() - p_bc)
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
# 3-4. DARCY-BRINKMAN OPERATOR RESIDUAL
# =============================================================================


def plot_brinkman_residual(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot CDF of the normalised Darcy-Brinkman residual magnitude.

    Residual definition:
        R = || -∇p + μ Δu - κ^{-1} · u ||

    Normalisation:
        R* = ⟨R⟩ / (⟨|u|⟩ / L²)

    This directly checks operator consistency with the PDE solved in COMSOL.

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
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_i = df.reset_index(drop=True)

            for _, row in df_i.iloc[loaded:max_cases].iterrows():
                data = _load_npz(row)
                pred = data["pred"]
                kappa = data["kappa"]
                kappa_names = data["kappa_names"]

                # channels: p, u, v
                p = pred[0]
                u = pred[1]
                v = pred[2]

                Lx = float(row["geom_Lx"])
                Ly = float(row["geom_Ly"])
                L = max(Lx, Ly)

                R = _compute_brinkman_residual(
                    p=p,
                    u=u,
                    v=v,
                    kappa=kappa,
                    kappa_names=kappa_names,
                    Lx=Lx,
                    Ly=Ly,
                )

                U_mean = np.mean(np.sqrt(u**2 + v**2))
                R_norm = np.mean(R) / (U_mean / (L**2) + EPS)

                entry["vals"].append(R_norm)

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
        ax.set_xlabel(r"$\langle R \rangle \,/\, (\langle |\mathbf{u}| \rangle / L^2)$")
        ax.set_ylabel("CDF")
        ax.set_title("Normalised Darcy-Brinkman residual distribution")
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
