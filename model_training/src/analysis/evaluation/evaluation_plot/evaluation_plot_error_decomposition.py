"""
Error decomposition plots for PINO/FNO evaluation.

This module focuses on *where* prediction errors occur by decomposing them
with respect to:
    1) distance from the domain boundary
    2) output magnitude |GT|
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure

    from src.util.util_plot_components import CheckboxGroup


# =============================================================================
# 2-1. ERROR VS |GT| MAGNITUDE (OVERLAYED DATASETS, CASECOUNT)
# =============================================================================


def plot_error_vs_gt_magnitude(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Error decomposition with respect to output magnitude |GT|.

    One subplot per channel (p, u, v, U).
    All datasets are overlaid within each subplot.

    Legend is shown in a dedicated column on the right,
    consistent with plot_global_error_metrics (1-1).

    Uses casecount viewer to incrementally load NPZ files.
    """
    channels = ["p", "u", "v", "U"]
    channel_idx = {"p": 0, "u": 1, "v": 2, "U": 3}

    n_bins = 15
    eps = 1e-12
    names = list(datasets.keys())

    # ------------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------------
    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "gt": {ch: [] for ch in channels},
            "err": {ch: [] for ch in channels},
        }
        for name in names
    }

    # ------------------------------------------------------------------
    # INTERNAL PLOT FUNCTION
    # ------------------------------------------------------------------
    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        dataset_selector: CheckboxGroup,
    ) -> Figure:
        """
        Plot error vs |GT| magnitude.

        Parameters
        ----------
        max_cases : int
            Number of cases to include from each dataset.
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name → evaluation DataFrame.
        dataset_selector : CheckboxGroup
            Dataset selection checkbox widget.

        Returns
        -------
        matplotlib.figure.Figure
            Overlayed error vs |GT| magnitude plots.

        """
        # --------------------------------------------------------------
        # Incremental NPZ loading
        # --------------------------------------------------------------
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            for path in df.iloc[loaded:max_cases]["npz_path"]:
                data = np.load(path)
                pred = data["pred"][0]
                gt = data["gt"][0]

                for ch in channels:
                    k = channel_idx[ch]
                    g = np.abs(gt[k]).ravel()

                    e = (np.abs(pred[k] - gt[k]) / (np.abs(gt[k]) + eps)).ravel()

                    q_cut = 0.02  # 2 % quantile cut to remove extreme outliers
                    g_min = np.nanquantile(g, q_cut)
                    mask = g > g_min
                    entry["gt"][ch].append(g[mask])
                    entry["err"][ch].append(e[mask])

            entry["loaded_until"] = max_cases

        # --------------------------------------------------------------
        # Active datasets
        # --------------------------------------------------------------
        active = [n for n, cb in dataset_selector.boxes.items() if cb.value]

        if not active:
            msg = "Select at least one dataset."
            raise ValueError(msg)

        # --------------------------------------------------------------
        # FIGURE + GRID (right column for legend)
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(9.5, 9))
        gs = fig.add_gridspec(
            len(channels),
            2,
            width_ratios=[1.0, 0.35],
            hspace=0.35,
            wspace=0.25,
        )

        axes = [fig.add_subplot(gs[r, 0]) for r in range(len(channels))]
        ax_legend = fig.add_subplot(gs[:, 1])
        ax_legend.axis("off")

        legend_handles = []

        # --------------------------------------------------------------
        # PLOTS
        # --------------------------------------------------------------
        for ax, ch in zip(axes, channels, strict=False):
            for name in active:
                g = np.concatenate(cache[name]["gt"][ch])
                e = np.concatenate(cache[name]["err"][ch])

                if g.size == 0:
                    continue

                bins = np.linspace(g.min(), g.max(), n_bins + 1) if ch == "p" else np.logspace(np.log10(g.min()), np.log10(g.max()), n_bins + 1)

                centers = 0.5 * (bins[:-1] + bins[1:]) if ch == "p" else np.sqrt(bins[:-1] * bins[1:])

                vals_per_bin = [e[(g >= lo) & (g < hi)] for lo, hi in itertools.pairwise(bins)]

                center = np.array(
                    [np.nanmedian(v) if v.size > 0 else np.nan for v in vals_per_bin],
                    dtype=float,
                )

                q25 = np.array(
                    [np.nanquantile(v, 0.25) if v.size > 0 else np.nan for v in vals_per_bin],
                    dtype=float,
                )

                q75 = np.array(
                    [np.nanquantile(v, 0.75) if v.size > 0 else np.nan for v in vals_per_bin],
                    dtype=float,
                )

                (line,) = ax.plot(
                    centers,
                    center,
                    marker="o",
                    markersize=4,
                    label=name,
                    alpha=0.9,
                )

                ax.fill_between(
                    centers,
                    q25,
                    q75,
                    alpha=0.25,
                )

                if ch == channels[0]:
                    legend_handles.append(line)

            # Axis scaling
            if ch == "p":
                ax.set_xscale("linear")
                ax.set_yscale("linear")
            else:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.set_title(f"{ch}: Relative MAE vs |GT|")
            ax.set_ylabel("Relative MAE  |e| / |GT|")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

        axes[-1].set_xlabel("|GT| (bin center)")

        # --------------------------------------------------------------
        # LEGEND (RIGHT COLUMN)
        # --------------------------------------------------------------
        ax_legend.legend(
            legend_handles,
            active,
            title="Dataset",
            loc="upper left",
        )

        fig.subplots_adjust(
            top=0.95,
            bottom=0.07,
            left=0.10,
            right=0.97,
        )
        return fig

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    dataset_selector = util.util_plot_components.ui_checkbox_datasets(
        dataset_names=names,
    )

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[dataset_selector],
        dataset_selector=dataset_selector,
    )


# =============================================================================
# 2-2. ERROR VS DISTANCE FROM BOUNDARY (CASECOUNT, BARPLOT)
# =============================================================================


def plot_error_vs_boundary_distance(
    *,
    datasets: dict[str, pd.DataFrame],
) -> widgets.VBox:
    """
    Error decomposition with respect to distance from domain boundary. One subplot per dataset.

    Uses casecount viewer to incrementally load NPZ files.
    Plots MAE against distance bands from the boundary for all output channels separately.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name → evaluation DataFrame.

    Returns
    -------
    ipywidgets.VBox
        Interactive casecount viewer widget.

    """
    channels = ["p", "u", "v", "U"]
    channel_idx = {"p": 0, "u": 1, "v": 2, "U": 3}

    # Distance bands (normalised distance 0..1)
    bands = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.40),
        (0.40, None),
    ]
    band_labels = ["0-5 %", "5-10 %", "10-20 %", "20-40 %", ">40 %"]

    # ------------------------------------------------------------------
    # CACHE (incremental, per dataset)
    # ------------------------------------------------------------------
    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "count": 0,
            "sum": {ch: np.zeros(len(bands), dtype=float) for ch in channels},
        }
        for name in datasets
    }

    # ------------------------------------------------------------------
    # INTERNAL PLOT FUNCTION (CASECOUNT)
    # ------------------------------------------------------------------
    def _plot(
        *,
        datasets: dict[str, pd.DataFrame],
        max_cases: int,
        channel_selector: CheckboxGroup,
    ) -> Figure:
        """
        Plot boundary error vs distance from boundary.

        Parameters
        ----------
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name → evaluation DataFrame.
        max_cases : int
            Number of cases to include from each dataset.
        channel_selector : CheckboxGroup
            Channel selection checkbox widget.

        Returns
        -------
        matplotlib.figure.Figure
            Bar plot figure.

        """
        # --------------------------------------------------------------
        # Active channels from checkbox widget
        # --------------------------------------------------------------
        active_channels = [name for name, cb in channel_selector.boxes.items() if cb.value]

        if not active_channels:
            msg = "At least one channel must be selected."
            raise ValueError(msg)

        # --------------------------------------------------------------
        # Incremental NPZ loading
        # --------------------------------------------------------------
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            df_new = df.iloc[loaded:max_cases]

            for path in df_new["npz_path"]:
                data = np.load(path)
                pred = data["pred"][0]
                gt = data["gt"][0]

                ny, nx = gt.shape[1:]

                # Distance to boundary (cells)
                yy, xx = np.meshgrid(
                    np.arange(ny),
                    np.arange(nx),
                    indexing="ij",
                )

                dist_cells = np.minimum.reduce([xx, yy, (nx - 1) - xx, (ny - 1) - yy])

                # Normalised distance [0, 1]
                d_max = min(nx, ny) / 2.0
                dist = dist_cells / d_max

                for ch in active_channels:
                    k = channel_idx[ch]
                    err = np.abs(pred[k] - gt[k])

                    for i, (lo, hi) in enumerate(bands):
                        mask = dist >= lo if hi is None else (dist >= lo) & (dist < hi)

                        if np.any(mask):
                            entry["sum"][ch][i] += float(np.nanmean(err[mask]))

                entry["count"] += 1

            entry["loaded_until"] = max_cases

        # --------------------------------------------------------------
        # BUILD BARPLOT
        # --------------------------------------------------------------
        names = list(datasets.keys())
        n_data = len(names)

        fig = plt.figure(figsize=(6 * n_data + 2.5, 5))
        gs = fig.add_gridspec(
            1,
            n_data + 1,
            width_ratios=[1] * n_data + [0.35],
            wspace=0.25,
        )

        axes = [fig.add_subplot(gs[0, i]) for i in range(n_data)]
        ax_legend = fig.add_subplot(gs[0, -1])
        ax_legend.axis("off")

        x = np.arange(len(active_channels))
        width = 0.8 / len(bands)

        bar_handles: list[Any] = []

        for ax, name in zip(axes, names, strict=False):
            entry = cache[name]

            for i, _label in enumerate(band_labels):
                eps = 1e-12

                # MAE pro Channel und Band
                mae_by_ch = {ch: np.array(entry["sum"][ch], dtype=float) / max(entry["count"], 1) for ch in active_channels}

                # Referenz: Interior-Band (>40 %) = letztes Band
                ref_idx = -1

                # Band-Ratio
                y = [mae_by_ch[ch][i] / (mae_by_ch[ch][ref_idx] + eps) for ch in active_channels]

                bars = ax.bar(
                    x + (i - (len(bands) - 1) / 2) * width,
                    y,
                    width,
                )

                if len(bar_handles) < len(bands):
                    bar_handles.append(bars[0])

            ax.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(active_channels)
            ax.set_yscale("linear")
            ax.set_ylabel("Boundary error ratio (MAE / interior MAE)")
            ax.set_xlabel("Channel")
            ax.set_title(f"Dataset: {name}")
            ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        ax_legend.legend(
            bar_handles,
            band_labels,
            title="Distance from boundary",
            loc="upper left",
        )

        fig.suptitle(f"Boundary error ratio vs distance from domain boundary (interior reference: >40 %, first {max_cases} cases)")
        fig.subplots_adjust(
            top=0.87,
            bottom=0.07,
            left=0.001,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )
        return fig

    # ------------------------------------------------------------------
    # UI COMPONENTS
    # ------------------------------------------------------------------
    channel_selector = util.util_plot_components.ui_checkbox_channels(default_on=["p", "u", "v", "U"])

    # ------------------------------------------------------------------
    # CASECOUNT VIEWER
    # ------------------------------------------------------------------
    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[channel_selector],
        channel_selector=channel_selector,
    )
