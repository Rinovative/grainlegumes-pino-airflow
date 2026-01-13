"""
Spectral analysis plots for PINO/FNO evaluation.

This module analyses:
    - what spatial frequencies the model learns,
    - whether learned spectral capacity matches physical demand (GT),
    - and where spectral errors remain.

Implemented analyses:
    5-1 + 5-3) Combined spectral analysis:
        - learned spectrum (model)
        - GT spectral demand
        - error spectrum (filled)
    5-2) Learned spectral energy per layer
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src import util
from src.schema.schema_fields import OUTPUT_FIELDS

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure


# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = list(OUTPUT_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}


# =============================================================================
# TYPES
# =============================================================================


class SpectralCacheEntry(TypedDict):
    learned_rad: np.ndarray | None
    learned_layer_energy: np.ndarray | None
    loaded_until: int
    gt_spectra: list[np.ndarray]
    err_spectra: list[np.ndarray]


# =============================================================================
# HELPERS
# =============================================================================


def _infer_run_dir_from_df(df: pd.DataFrame) -> Path:
    if "npz_path" not in df.columns:
        raise ValueError("DataFrame has no 'npz_path' column")

    npz_path = Path(df["npz_path"].iloc[0])
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ path does not exist: {npz_path}")

    return npz_path.parents[3]


def _load_spectral_energy(run_dir: Path) -> dict[int, np.ndarray]:
    pt_path = run_dir / "spectral_energy_aggregated.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing spectral file: {pt_path}")

    spec = torch.load(pt_path, map_location="cpu", weights_only=True)
    return {int(lid): np.asarray(t.cpu().numpy()) for lid, t in spec.items()}


def _radial_frequency_spectrum(field: np.ndarray) -> np.ndarray:
    ny, nx = field.shape

    fhat = np.fft.fft2(field)
    power2d = np.abs(fhat) ** 2

    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)

    kr_flat = kr.ravel()
    p_flat = power2d.ravel()

    nbins = max(1, min(ny, nx) // 2)
    bins = np.linspace(0.0, kr_flat.max(), nbins + 1)
    idx = np.digitize(kr_flat, bins) - 1

    spec = np.zeros(nbins)
    cnt = np.zeros(nbins)

    for i in range(nbins):
        mask = idx == i
        if np.any(mask):
            spec[i] = p_flat[mask].mean()
            cnt[i] = mask.sum()

    return spec[cnt > 0]


def _radialise_learned_spectrum(E: np.ndarray) -> np.ndarray:
    ny, nxh = E.shape
    nx = (nxh - 1) * 2

    ky = np.fft.fftfreq(ny)
    kx = np.fft.rfftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)

    kr_flat = kr.ravel()
    E_flat = E.ravel()

    nbins = max(1, min(ny, nx) // 2)
    bins = np.linspace(0.0, kr_flat.max(), nbins + 1)
    idx = np.digitize(kr_flat, bins) - 1

    spec = np.zeros(nbins)
    cnt = np.zeros(nbins)

    for i in range(nbins):
        mask = idx == i
        if np.any(mask):
            spec[i] = E_flat[mask].mean()
            cnt[i] = mask.sum()

    return spec[cnt > 0]


def _init_cache(names: list[str]) -> dict[str, SpectralCacheEntry]:
    return {
        name: {
            "learned_rad": None,
            "learned_layer_energy": None,
            "loaded_until": 0,
            "gt_spectra": [],
            "err_spectra": [],
        }
        for name in names
    }


def _ensure_learned_cached(entry: SpectralCacheEntry, run_dir: Path) -> None:
    if entry["learned_rad"] is not None:
        return

    spec = _load_spectral_energy(run_dir)
    layer_ids = sorted(spec.keys())

    learned_layers = [_radialise_learned_spectrum(spec[lid]) for lid in layer_ids]
    entry["learned_rad"] = np.mean(np.vstack(learned_layers), axis=0)
    entry["learned_layer_energy"] = np.array([spec[lid].mean() for lid in layer_ids])


def _accumulate_case_spectra(entry: SpectralCacheEntry, df: pd.DataFrame, max_cases: int) -> None:
    loaded = entry["loaded_until"]
    max_cases = int(max_cases)

    if max_cases <= loaded:
        return

    df_i = df.reset_index(drop=True)

    for path in df_i["npz_path"].iloc[loaded:max_cases]:
        data = np.load(path)

        gt = data["gt"]
        u = gt[CHANNEL_INDICES["u"]]
        v = gt[CHANNEL_INDICES["v"]]
        speed = np.sqrt(u**2 + v**2)
        entry["gt_spectra"].append(_radial_frequency_spectrum(speed))

        err = data["err"]
        err_field = np.linalg.norm(err, axis=0)
        entry["err_spectra"].append(_radial_frequency_spectrum(err_field))

    entry["loaded_until"] = max_cases


# =============================================================================
# 1-6. ERROR FREQUENCY SPECTRUM
# =============================================================================


def plot_global_error_frequency_spectrum(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Global error frequency spectrum across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.
        Must contain:
            - 'npz_path' : str (path to .npz with 'err' array)

    Returns
    -------
    matplotlib.figure.Figure
        Figure with global error frequency spectrum plot.

    """
    palette = sns.color_palette("tab10", len(datasets))

    spectra: dict[str, list[np.ndarray]] = {name: [] for name in datasets}
    freqs: dict[str, np.ndarray] = {}

    for name, df in datasets.items():
        df_i = df.reset_index(drop=True)

        for path in df_i["npz_path"]:
            data = np.load(path)
            err = data["err"]

            # combine all channels into a single error magnitude field
            field = np.linalg.norm(err, axis=0)

            k, p = _radial_frequency_spectrum(field)
            spectra[name].append(p)
            freqs[name] = k

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, color in zip(datasets.keys(), palette, strict=False):
        arr = np.vstack(spectra[name])
        mean_spec = np.mean(arr, axis=0)

        ax.plot(freqs[name], mean_spec, lw=2, label=name, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Error power")
    ax.set_title("Global error frequency spectrum")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    return fig


# =============================================================================
# 5-1 + 5-3 COMBINED SPECTRAL ANALYSIS
# =============================================================================


def plot_combined_spectral_analysis(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Combined spectral plot:
        - solid line   : learned spectrum (model)
        - dashed line  : GT spectral demand
        - filled area  : error spectrum

    Same color per model. Works for multiple models.
    """
    names = list(datasets.keys())
    cache = _init_cache(names)

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:
        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles = []

        for name in names:
            df = datasets[name]
            run_dir = _infer_run_dir_from_df(df)

            _ensure_learned_cached(cache[name], run_dir)
            _accumulate_case_spectra(cache[name], df, max_cases)

            learned = cache[name]["learned_rad"]
            if learned is None:
                continue

            color = None

            # --------------------
            # MODEL (solid)
            # --------------------
            k = np.arange(len(learned))
            (h_model,) = ax.plot(k, learned, lw=2)
            color = h_model.get_color()
            handles.append(h_model)

            # --------------------
            # GT (dashed)
            # --------------------
            if cache[name]["gt_spectra"]:
                gt = np.mean(np.vstack(cache[name]["gt_spectra"]), axis=0)
                n = min(len(k), len(gt))
                ax.plot(k[:n], gt[:n], lw=2, ls="--", color=color)

            # --------------------
            # ERROR (filled)
            # --------------------
            if cache[name]["err_spectra"]:
                err = np.mean(np.vstack(cache[name]["err_spectra"]), axis=0)
                n = min(len(k), len(err))
                ax.fill_between(
                    k[:n],
                    learned[:n],
                    gt[:n],
                    where=gt[:n] > learned[:n],
                    color=color,
                    alpha=0.25,
                )
                ax.fill_between(
                    k[:n],
                    learned[:n],
                    gt[:n],
                    where=gt[:n] < learned[:n],
                    color=color,
                    alpha=0.25,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Spatial frequency (radial index)")
        ax.set_ylabel("Spectral energy")
        ax.set_title("Spectral capacity, demand and error")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # --------------------
        # Legend (models only)
        # --------------------
        ax_leg.legend(handles, names, title="Model", loc="upper left")

        # --------------------
        # Meaning annotation
        # --------------------
        ax.text(
            0.02,
            0.02,
            "Solid: model   Dashed: GT   Filled: error",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 5-2 LEARNED SPECTRAL ENERGY PER LAYER
# =============================================================================


def plot_learned_spectral_energy_per_layer(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    names = list(datasets.keys())
    cache = _init_cache(names)

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame]) -> Figure:  # noqa: ARG001
        fig = plt.figure(figsize=(9.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.35], wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles = []

        for name in names:
            df = datasets[name]
            run_dir = _infer_run_dir_from_df(df)
            _ensure_learned_cached(cache[name], run_dir)

            energy = cache[name]["learned_layer_energy"]
            if energy is None:
                continue

            (h,) = ax.plot(range(len(energy)), energy, marker="o", lw=2)
            handles.append(h)

        ax.set_yscale("log")
        ax.set_xlabel("Layer depth")
        ax.set_ylabel("Mean spectral energy")
        ax.set_title("Learned spectral energy per layer")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax_leg.legend(handles, names, title="Model", loc="upper left")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.98)
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )
