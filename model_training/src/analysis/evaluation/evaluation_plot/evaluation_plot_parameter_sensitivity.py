"""
Error sensitivity analysis for PINO/FNO evaluation.

3.1  Parameter-error correlation — HEATMAP

Design rules:
- Channel selector affects ONLY:
    - which per-channel error is used
    - title / focus information
- ALL par_* columns are used (no hardcoding)
- Per-channel errors are computed explicitly from NPZ files
- No heuristics, no guessing
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    from matplotlib.figure import Figure


class _CheckboxGroupProto(Protocol):
    boxes: dict[str, Any]


# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = ["p", "u", "v", "U"]
EPS = 1e-12


# =============================================================================
# GLOBAL CACHE
# =============================================================================

_npz_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


# =============================================================================
# NPZ LOADING
# =============================================================================


def _load_err_gt(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Load error and ground-truth arrays from NPZ file, with caching.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing the "npz_path" column.

    Returns
    -------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground-truth array.

    """
    key = str(Path(row["npz_path"]))
    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key)
    err = data["err"][0]
    gt = data["gt"][0]

    _npz_cache[key] = (err, gt)
    return err, gt


# =============================================================================
# METRIC
# =============================================================================


def _rel_l2(err: np.ndarray, gt: np.ndarray) -> float:
    """
    Relative L2 error metric.

    Parameters
    ----------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground-truth array.

    Returns
    -------
    float
        Relative L2 error.

    """
    return float(np.linalg.norm(err) / (np.linalg.norm(gt) + EPS))


# =============================================================================
# 3.1 PARAMETER-ERROR CORRELATION — HEATMAP (ALL CHANNELS)
# =============================================================================


def plot_parameter_error_heatmap(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Parameter-error correlation heatmap.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to analyze.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """

    def _plot(
        *,
        datasets: dict[str, pd.DataFrame],
        max_cases: int,
    ) -> Figure:
        """
        Plot parameter-error correlation heatmap.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            Datasets to analyze.
        max_cases : int
            Maximum number of cases to consider.

        Returns
        -------
        fig : Figure
            Generated figure.

        """
        # --------------------------------------------------
        # common parameter columns
        # --------------------------------------------------
        par_cols = sorted(set.intersection(*[{c for c in df.columns if c.startswith("par_") and c != "par_seed"} for df in datasets.values()]))

        if not par_cols:
            msg = "No common par_* columns found across datasets."
            raise ValueError(msg)

        # --------------------------------------------------
        # collect data
        # --------------------------------------------------
        data: dict[str, dict[str, Any]] = {}

        for name, df in datasets.items():
            err_data: dict[str, list[float]] = {ch: [] for ch in CHANNELS}
            par_data: dict[str, list[float]] = {p: [] for p in par_cols}

            for _, row in df.iloc[:max_cases].iterrows():
                err, gt = _load_err_gt(row)

                for ci, ch in enumerate(CHANNELS):
                    err_data[ch].append(_rel_l2(err[ci], gt[ci]))

                for p in par_cols:
                    par_data[p].append(float(row[p]))

            data[name] = {"err": err_data, "par": par_data}

        # --------------------------------------------------
        # correlation matrix: (par, channel)
        # --------------------------------------------------
        corrs: dict[str, pd.DataFrame] = {}

        for name, d in data.items():
            corr = pd.DataFrame(index=par_cols, columns=CHANNELS, dtype=float)

            for p in par_cols:
                x = np.asarray(d["par"][p])
                for ch in CHANNELS:
                    y = np.asarray(d["err"][ch])

                    if np.std(x) < EPS or np.std(y) < EPS:
                        corr.loc[p, ch] = np.nan
                    else:
                        corr.loc[p, ch] = np.corrcoef(x, y)[0, 1]

            corrs[name] = corr

        # --------------------------------------------------
        # plotting: single heatmap
        # --------------------------------------------------
        n = len(corrs)
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(n * (1.4 * len(CHANNELS) + 2), 0.25 * len(par_cols) + 2),
            squeeze=False,
            constrained_layout=True,
        )

        for ax, (name, corr) in zip(axes[0], corrs.items(), strict=False):
            im = ax.imshow(
                corr.to_numpy(),
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
            )

            ax.set_xticks(np.arange(len(CHANNELS)))
            ax.set_xticklabels([f"rel_l2[{ch}]" for ch in CHANNELS])
            ax.set_yticks(np.arange(len(par_cols)))
            ax.set_yticklabels(par_cols)
            ax.set_title(name)

            corr_values = corr.to_numpy(dtype=float)
            for i in range(corr_values.shape[0]):
                for j in range(corr_values.shape[1]):
                    v = corr_values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Pearson correlation")
        fig.suptitle("Parameter-error correlation per dataset")
        return fig

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
    )


# =============================================================================
# 3.2 PARAMETER-ERROR TREND (SORTED BY EFFECT, TOP-K, PER DATASET)
# =============================================================================


def plot_error_vs_parameter_trend(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Parameter-error trend plots.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Datasets to analyze.

    Returns
    -------
    widgets.VBox
        Interactive plot widget.

    """
    # --------------------------------------------------
    # parameter columns (ORDER AS IN DF)
    # --------------------------------------------------
    first_df = next(iter(datasets.values()))
    par_cols = [c for c in first_df.columns if c.startswith("par_") and c != "par_seed"]
    if not par_cols:
        msg = "No par_* columns found."
        raise ValueError(msg)

    names = list(datasets.keys())

    n_bins = 12
    min_pairs_per_param = 20
    min_per_bin = 2

    # --------------------------------------------------
    # CACHE
    # --------------------------------------------------
    cache: dict[str, dict[str, Any]] = {
        name: {
            "loaded_until": 0,
            "x": {p: [] for p in par_cols},
            "y": {p: {ch: [] for ch in CHANNELS} for p in par_cols},
        }
        for name in names
    }

    # --------------------------------------------------
    # sensitivity computation
    # --------------------------------------------------
    def _sensitivity_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute sensitivity from (x, y) pairs.

        Sensitivity is defined as the difference between the 90th and 10th percentiles
        of the median y values in bins of x.

        Parameters
        ----------
        x : np.ndarray
            Parameter values.
        y : np.ndarray
            Error values.

        Returns
        -------
        float
            Sensitivity value.

        """
        if x.size < min_pairs_per_param or y.size < min_pairs_per_param:
            return float("nan")

        if np.nanstd(x) < EPS or np.nanstd(y) < EPS:
            return float("nan")

        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, q)

        medians: list[float] = []
        for lo, hi in itertools.pairwise(edges):
            mask = (x >= lo) & (x < hi)
            yy = y[mask]
            if yy.size >= min_per_bin:
                medians.append(float(np.nanmedian(yy)))

        if len(medians) < 3:  # noqa: PLR2004
            return float("nan")

        lo = np.nanpercentile(medians, 10)
        hi = np.nanpercentile(medians, 90)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return float("nan")

        return float(hi - lo)

    # --------------------------------------------------
    # CASECOUNT plot
    # --------------------------------------------------
    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        channel_selector: widgets.VBox,
    ) -> Figure:
        """
        Plot parameter sensitivity figure.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to consider.
        datasets : dict[str, pd.DataFrame]
            Datasets to analyze.
        channel_selector : widgets.VBox
            Channel selection widget.

        Returns
        -------
        fig : Figure
            Generated figure.

        """
        for name, df in datasets.items():
            entry = cache[name]
            loaded = entry["loaded_until"]

            if max_cases <= loaded:
                continue

            for _, row in df.iloc[loaded:max_cases].iterrows():
                err, gt = _load_err_gt(row)

                for p in par_cols:
                    entry["x"][p].append(float(row[p]))
                    for ci, ch in enumerate(CHANNELS):
                        entry["y"][p][ch].append(_rel_l2(err[ci], gt[ci]))

            entry["loaded_until"] = max_cases

        group = channel_selector  # runtime object
        group = cast("_CheckboxGroupProto", group)
        active_channels = [ch for ch, cb in group.boxes.items() if cb.value]
        if not active_channels:
            msg = "Select at least one channel."
            raise ValueError(msg)

        shown_params = par_cols
        y_pos = np.arange(len(shown_params))

        fig = plt.figure(figsize=(8.0 * len(names) + 2.5, 0.55 * len(shown_params) + 2.0))
        gs = fig.add_gridspec(
            1,
            len(names) + 1,
            width_ratios=[1.0] * len(names) + [0.35],
            wspace=0.35,
        )

        axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
        ax_legend = fig.add_subplot(gs[0, -1])
        ax_legend.axis("off")

        legend_handles = []

        for ax, name in zip(axes, names, strict=False):
            for ch in active_channels:
                xs_plot = []

                for p in shown_params:
                    x = np.asarray(cache[name]["x"][p], dtype=float)
                    y = np.asarray(cache[name]["y"][p][ch], dtype=float)
                    xs_plot.append(_sensitivity_from_pairs(x, y))

                (line,) = ax.plot(
                    xs_plot,
                    y_pos,
                    marker="o",
                    linestyle="-",
                    alpha=0.9,
                    label=ch,
                )

                if name == names[0]:
                    legend_handles.append(line)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(shown_params)
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlabel("Sensitivity on rel_l2 (P90 minus P10 of binned medians, log scale)")
            ax.set_title(f"Dataset: {name}")
            ax.grid(True, which="both", axis="x", alpha=0.3)

        ax_legend.legend(
            legend_handles,
            active_channels,
            title="Channel",
            loc="upper left",
        )

        fig.suptitle(
            "Parameter sensitivity (original parameter order)",
            fontsize=14,
        )
        fig.subplots_adjust(
            top=0.90,
            bottom=0.08,
            left=0.08,
            right=0.96,
        )
        return fig

    channel_selector = util.util_plot_components.ui_checkbox_channels(
        default_on=CHANNELS,
    )

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=100,
        step_size=50,
        extra_widgets=[channel_selector],
        channel_selector=channel_selector,
    )
