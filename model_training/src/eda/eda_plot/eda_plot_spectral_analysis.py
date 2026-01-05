"""
Exploratory spectral analysis tools for porous-media simulation data.

This module provides utilities to compute spectral quantities
(two-dimensional PSDs and radial energy spectra) and to build
interactive spectral visualisations for simulation fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure


# ======================================================================
# Internal DataFrame utilities
# ======================================================================
def _infer_field_keys(df: pd.DataFrame) -> list[str]:
    """
    Infer 2D field keys from a DataFrame by inspecting the first row.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing simulation case data.

    Returns
    -------
    list[str]
        List of column names corresponding to 2D numeric fields.

    """
    keys: list[str] = []

    sample = df.iloc[0]
    for col in df.columns:
        if col in {"x", "y", "meta"}:
            continue

        arr = np.asarray(sample[col])
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            keys.append(col)

    return keys


def _infer_ncols(n_items: int, *, max_cols: int = 4) -> int:
    """
    Infer a suitable number of subplot columns for a given number of items.

    Parameters
    ----------
    n_items : int
        Total number of items to plot.
    max_cols : int
        Maximum number of columns allowed.

    Returns
    -------
    int
        Number of columns to use for subplots.

    """
    return min(max_cols, max(1, int(np.ceil(np.sqrt(n_items)))))


# ======================================================================
# Internal spectral utilities
# ======================================================================


def _hann2d(ny: int, nx: int) -> np.ndarray:
    """
    Create a two-dimensional Hann window.

    A separable Hann weighting is applied in both spatial directions
    to reduce edge effects before computing the FFT.

    Args:
        ny (int): Number of points in the vertical direction.
        nx (int): Number of points in the horizontal direction.

    Returns:
        np.ndarray: Hann window of shape (ny, nx).

    """
    return np.outer(np.hanning(ny), np.hanning(nx))


def _fft2_psd(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the two-dimensional FFT and power spectral density.

    The input field is mean-subtracted and Hann-windowed. The result
    is returned in centred form (fftshifted), together with the
    corresponding wavenumber grids.

    Args:
        field (np.ndarray): Two-dimensional scalar field.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.

    Returns:
        tuple:
            np.ndarray: Centred power spectral density.
            np.ndarray: Centred wavenumber grid kx.
            np.ndarray: Centred wavenumber grid ky.

    """
    a = field.astype(float, copy=True)
    a -= np.mean(a)
    a *= _hann2d(*a.shape)

    F = np.fft.fft2(a)
    PSD = np.abs(F) ** 2 / a.size

    kx = np.fft.fftfreq(a.shape[1], d=dx)
    ky = np.fft.fftfreq(a.shape[0], d=dy)
    kx, ky = np.meshgrid(kx, ky)

    return np.fft.fftshift(PSD), np.fft.fftshift(kx), np.fft.fftshift(ky)


def _radial_spectrum(
    PSD: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the isotropic radial energy spectrum E(k).

    The two-dimensional PSD is binned by radial wavenumber distance.
    The result is the mean spectral energy in each radial band.

    Args:
        PSD (np.ndarray): Power spectral density.
        kx (np.ndarray): Wavenumber grid in the x direction.
        ky (np.ndarray): Wavenumber grid in the y direction.
        n_bins (int): Number of radial bins.

    Returns:
        tuple:
            np.ndarray: Radial wavenumber centres.
            np.ndarray: Energy density E(k).

    """
    kr = np.sqrt(kx**2 + ky**2).ravel()
    ps = PSD.ravel()

    mask = np.isfinite(kr) & np.isfinite(ps)
    kr, ps = kr[mask], ps[mask]

    edges = np.linspace(kr.min(), kr.max(), n_bins + 1)
    idx = np.digitize(kr, edges) - 1

    sums = np.bincount(idx, weights=ps)
    counts = np.bincount(idx)

    n = min(len(sums), len(edges) - 1)
    E = sums[:n] / np.maximum(counts[:n], 1)

    k_centres = 0.5 * (edges[:-1] + edges[1:])
    return k_centres[:n], E


# ======================================================================
# Interactive 2D spectral overview
# ======================================================================


def plot_spectral_overview(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Build an interactive viewer for 2D spectral maps (PSD).

    Each case is visualised by transforming the fields listed in
    `ALL_KEYS` into log-scaled 2D PSD maps using FFT. A separate subplot
    is shown for each field, including an individual colourbar.

    Navigation across cases is handled by the generic util_plot navigator.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping: dataset_name -> DataFrame
        Each DataFrame must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy

    Returns
    -------
    widgets.VBox
        Fully interactive spectral viewer with next/previous navigation.


    """
    # Navigator übernimmt Auswahl → kein df oder dataset_name direkt nötig

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot a multi-field 2D PSD overview for a single simulation case."""
        row = df.iloc[idx]

        field_keys = _infer_field_keys(df)
        fields = {key: np.asarray(row[key], float) for key in field_keys}

        x = np.asarray(row["x"])
        y = np.asarray(row["y"])
        dx = float(np.nanmedian(np.diff(np.unique(x))))
        dy = float(np.nanmedian(np.diff(np.unique(y))))

        n_items = len(fields)
        ncols = _infer_ncols(n_items, max_cols=4)
        nrows = int(np.ceil(n_items / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, (label, field) in zip(axes[: len(fields)], fields.items(), strict=True):
            PSD, kx, ky = _fft2_psd(field, dx, dy)
            logPSD = np.log10(PSD + 1e-20)

            vmin, vmax = np.nanpercentile(logPSD, [2, 98])

            im = ax.pcolormesh(
                kx,
                ky,
                logPSD,
                cmap="inferno",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label} spectrum")
            ax.set_xlabel(r"Wavenumber $k_x$ [$\frac{1}{m}$]")
            ax.set_ylabel(r"Wavenumber $k_y$ [$\frac{1}{m}$]")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

        for ax in axes[len(fields) :]:
            ax.axis("off")

        fig.suptitle(f"2D spectral maps: {dataset_name} - Case {idx + 1}", fontsize=12, y=0.98)
        fig.subplots_adjust(top=0.97, wspace=0.5, hspace=0.01)
        return fig

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=False,
    )


# ======================================================================
# Interactive vertical spectral evolution
# ======================================================================


def plot_spectral_vertical(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Build an interactive viewer for vertical spectral evolution.

    For each field in `ALL_KEYS`, two radial spectra are computed:
    one from a thin slice near the bottom of the domain and one
    from a slice near mid-height. This highlights changes in spatial
    structure across the domain height.

    Navigation across cases is handled by the generic util_plot navigator.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping: dataset_name -> DataFrame
        Each DataFrame must contain:
        - the fields defined in ALL_KEYS
        - spatial coordinates "x" and "y" for computing dx, dy

    Returns
    -------
    widgets.VBox
        Fully interactive viewer with per-field vertical line spectra.

    """

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str) -> Figure:
        """Plot vertical (bottom/mid) radial spectra for a single case."""
        row = df.iloc[idx]
        field_keys = _infer_field_keys(df)
        fields = {key: np.asarray(row[key], float) for key in field_keys}

        x_arr = np.asarray(row["x"])
        y_arr = np.asarray(row["y"])
        dx = float(np.nanmedian(np.diff(np.unique(x_arr))))
        dy = float(np.nanmedian(np.diff(np.unique(y_arr))))

        n_items = len(fields)
        ncols = _infer_ncols(n_items, max_cols=4)
        nrows = int(np.ceil(n_items / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        field_keys = list(fields.keys())
        y_coords = np.linspace(0, 1, fields[field_keys[0]].shape[0])
        y_low, y_high = 0.05, 0.70
        win = 0.01

        mask_low = (y_coords >= y_low - win) & (y_coords <= y_low + win)
        mask_high = (y_coords >= y_high - win) & (y_coords <= y_high + win)

        for ax, (label, field) in zip(axes[: len(fields)], fields.items(), strict=True):
            seg_low = np.atleast_2d(field[mask_low].mean(axis=0))
            seg_high = np.atleast_2d(field[mask_high].mean(axis=0))

            PSD_low, kx_low, ky_low = _fft2_psd(seg_low, dx, dy)
            PSD_high, kx_high, ky_high = _fft2_psd(seg_high, dx, dy)

            k_low, E_low = _radial_spectrum(PSD_low, kx_low, ky_low)
            k_high, E_high = _radial_spectrum(PSD_high, kx_high, ky_high)

            mask_l = (k_low > 0) & (E_low > 0)
            mask_h = (k_high > 0) & (E_high > 0)

            if np.count_nonzero(mask_l) > 1:
                ax.loglog(k_low[mask_l], E_low[mask_l], lw=1.4, label=f"y={y_low:.2f}")

            if np.count_nonzero(mask_h) > 1:
                ax.loglog(k_high[mask_h], E_high[mask_h], lw=1.4, label=f"y={y_high:.2f}")
            ax.set_title(label)
            ax.set_xlabel(r"Wavenumber $k$ [$\frac{1}{m}$]")
            ax.set_ylabel(r"Spectral energy $E(k)$")
            ax.grid(True, which="both", ls=":")
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

        for ax in axes[len(fields) :]:
            ax.axis("off")

        fig.suptitle(f"Vertical spectral profiles: {dataset_name} - Case {idx + 1}", fontsize=12, y=0.98)
        fig.subplots_adjust(top=0.94, wspace=0.40, hspace=0.3)
        return fig

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_idx=0,
        enable_dataset_dropdown=False,
    )
