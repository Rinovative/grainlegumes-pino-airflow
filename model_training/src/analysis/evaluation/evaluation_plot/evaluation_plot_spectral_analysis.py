"""
Spectral analysis plots for PINO / FNO / UNO evaluation.

This module provides post-hoc spectral diagnostics to analyse how neural
operators distribute and utilise spectral energy across layers and how
this relates to observed prediction errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def plot_learned_spectral_energy_per_layer(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Visualise learned spectral energy distribution per layer.

    For each dataset (model), the mean spectral energy across spatial
    frequencies is shown per layer. This plot highlights where spectral
    information is concentrated along the network depth.

    Expected input:
        - 'layer_id' : int
        - 'spectral_energy' : np.ndarray of shape (K,) per row

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset name → evaluation DataFrame.

    Returns
    -------
    matplotlib.figure.Figure
        Line plot of mean spectral energy per layer.

    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, df in datasets.items():
        layers = sorted(df["layer_id"].unique())

        energies: list[float] = []

        for layer in layers:
            series = cast(
                "pd.Series[Any]",
                df.loc[df["layer_id"] == layer, "spectral_energy"],
            )

            arrs = [np.asarray(v, dtype=float) for v in series]

            if not arrs:
                energies.append(float("nan"))
                continue

            stacked = np.stack(arrs, axis=0)
            energies.append(float(stacked.mean()))

        ax.plot(layers, energies, marker="o", lw=2, label=name)

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Mean spectral energy")
    ax.set_title("Learned spectral energy per layer")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig


def plot_spectral_centroid_vs_depth(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse how the spectral centroid shifts with network depth.

    The spectral centroid is defined as the energy-weighted mean frequency:
        k_c = sum(k * E_k) / sum(E_k)

    This plot shows whether higher layers focus on lower or higher
    spatial frequencies.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset name → evaluation DataFrame.
        Required columns:
            - 'layer_id'
            - 'spectral_energy' : array-like
            - 'spectral_freq'   : array-like (same length as energy)

    Returns
    -------
    matplotlib.figure.Figure
        Line plot of spectral centroid versus layer depth.

    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, df in datasets.items():
        layers = sorted(df["layer_id"].unique())
        centroids = []

        for layer in layers:
            rows = df[df["layer_id"] == layer]

            E = np.stack(rows["spectral_energy"].to_numpy())
            k = np.stack(rows["spectral_freq"].to_numpy())

            num = np.sum(k * E)
            den = np.sum(E) + 1e-12
            centroids.append(float(num / den))

        ax.plot(layers, centroids, marker="o", lw=2, label=name)

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Spectral centroid")
    ax.set_title("Spectral centroid vs depth")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig


def plot_learned_spectrum_vs_error_spectrum(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Compare learned spectral energy with observed error spectrum.

    This plot overlays:
        - mean learned spectral energy (averaged over layers)
        - mean prediction error spectrum

    It highlights whether the model allocates spectral capacity to the
    frequencies where errors actually occur.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset name → evaluation DataFrame.
        Required columns:
            - 'spectral_energy' : np.ndarray (K,)
            - 'spectral_freq'   : np.ndarray (K,)
            - 'error_spectrum'  : np.ndarray (K,)

    Returns
    -------
    matplotlib.figure.Figure
        Log-log comparison plot of learned vs error spectra.

    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, df in datasets.items():
        # -------------------------------
        # Safe extraction (pylance-clean)
        # -------------------------------
        energy_series = cast("pd.Series[Any]", df["spectral_energy"])
        freq_series = cast("pd.Series[Any]", df["spectral_freq"])
        error_series = cast("pd.Series[Any]", df["error_spectrum"])

        E_list = [np.asarray(v, dtype=float) for v in energy_series]
        k_list = [np.asarray(v, dtype=float) for v in freq_series]
        Err_list = [np.asarray(v, dtype=float) for v in error_series]

        if not E_list or not Err_list or not k_list:
            continue

        E = np.stack(E_list, axis=0)
        Err = np.stack(Err_list, axis=0)
        k = np.stack(k_list, axis=0)

        # -------------------------------
        # Mean spectra
        # -------------------------------
        E_mean = E.mean(axis=0)
        Err_mean = Err.mean(axis=0)
        k_mean = k.mean(axis=0)

        ax.plot(k_mean, E_mean, lw=2, label=f"{name} learned")
        ax.plot(k_mean, Err_mean, lw=2, ls="--", label=f"{name} error")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Spectral power")
    ax.set_title("Learned spectrum vs error spectrum")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig
