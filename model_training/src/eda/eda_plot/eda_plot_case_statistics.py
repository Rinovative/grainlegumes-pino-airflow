"""
Case-level statistics plots for PINO/FNO EDA.

Distributions derived from:
    1) meta["statistics"]
    2) meta["parameters"]
    3) reduced field statistics (min/mean/max per case)

Compatible with casecount viewer + dataset toggle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde

from src import util

if TYPE_CHECKING:
    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.util.util_plot_components import CheckboxGroup

# ============================================================================
# TYPES
# ============================================================================


class _StatCache(TypedDict):
    """
    Cache structure for meta statistics.

    Attributes
    ----------
    loaded : int
        Number of loaded cases.
    cols : dict[str, list[float]]
        Dictionary mapping statistic keys to lists of float values.

    """

    loaded: int
    cols: dict[str, list[float]]


class _ParamCache(TypedDict):
    """
    Cache structure for meta parameters.

    Attributes
    ----------
    loaded : int
        Number of loaded cases.
    cols : dict[str, list[float]]
        Dictionary mapping parameter keys to lists of float values.

    """

    loaded: int
    cols: dict[str, list[float]]


class _FieldCache(TypedDict):
    """
    Cache structure for field value statistics.

    Attributes
    ----------
    loaded : int
        Number of loaded cases.
    data : dict[str, dict[str, list[float]]]
        Dictionary mapping field names to their statistics (min, mean, max) lists.

    """

    loaded: int
    data: dict[str, dict[str, list[float]]]


# =============================================================================
# CONSTANTS
# =============================================================================

FIELDS = ["kappaxx", "kappayy", "p", "u", "v", "U"]
FIELD_STATS = ["min", "mean", "max"]

META_BINS = 40
META_NCOLS = 4

PARAMETER_ORDER = [
    "k_mean",
    "var_rel",
    "corr_len_rel",
    "ms_scale",
    "coupling",
    "volume_fraction",
    "lognormal",
    "aniso_major",
    "aniso_minor",
    "aniso_ratio",
    "ms_w1",
    "ms_w2",
]

# Parameter where log10 is meaningful
LOG_PARAMETER_KEYS = {
    "k_mean",
    "var_rel",
    "corr_len_rel",
    "ms_scale",
    "aniso_ratio",
}


# =============================================================================
# HELPERS
# =============================================================================


def _as_float(x: Any) -> float:
    """
    Convert x to float.

    If x is None, return nan.
    If x is a list/array with one element, return that element as float.

    Parameters
    ----------
    x : Any
        Input value to convert.

    Returns
    -------
        float
            Converted float value.

    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return float(np.asarray(x).item())


def _flatten_dict_raw(dct: dict[str, Any]) -> dict[str, float]:
    """
    Flatten a nested dictionary into a single-level dictionary with compound keys.

    Parameters
    ----------
    dct : dict[str, Any]
        Input nested dictionary.

    Returns
    -------
    dict[str, float]
        Flattened dictionary with compound keys and float values.

    """
    out: dict[str, float] = {}

    def _rec(key: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _rec(f"{key}_{k}", v)
            return
        if isinstance(obj, (list, tuple, np.ndarray)) and not np.isscalar(obj):
            for i, v in enumerate(obj):
                _rec(f"{key}_{i}", v)
            return
        out[key] = _as_float(obj)

    for k, v in dct.items():
        _rec(k, v)

    return out


def _flatten_parameters_ordered(dct: dict[str, Any]) -> dict[str, float]:
    """
    Flatten parameters dictionary into a single-level dictionary with specific keys.

    Parameters
    ----------
    dct : dict[str, Any]
        Input parameters dictionary.

    Returns
    -------
    dict[str, float]
        Flattened dictionary with specific keys and float values.

    """
    out: dict[str, float] = {}

    out["k_mean"] = _as_float(dct.get("k_mean"))
    out["var_rel"] = _as_float(dct.get("var_rel"))
    out["corr_len_rel"] = _as_float(dct.get("corr_len_rel"))
    out["ms_scale"] = _as_float(dct.get("ms_scale"))
    out["coupling"] = _as_float(dct.get("coupling"))
    out["volume_fraction"] = _as_float(dct.get("volume_fraction"))
    out["lognormal"] = _as_float(dct.get("lognormal"))

    aniso = dct.get("anisotropy", [np.nan, np.nan])
    out["aniso_major"] = _as_float(aniso[0])
    out["aniso_minor"] = _as_float(aniso[1])
    out["aniso_ratio"] = out["aniso_major"] / out["aniso_minor"] if np.isfinite(out["aniso_minor"]) and out["aniso_minor"] != 0 else np.nan

    msw = dct.get("ms_weight", [np.nan, np.nan])
    out["ms_w1"] = _as_float(msw[0])
    out["ms_w2"] = _as_float(msw[1])

    return out


def _selected_datasets(dataset_selector: CheckboxGroup) -> list[str]:
    """
    Get the list of selected datasets from the dataset selector.

    Parameters
    ----------
    dataset_selector : CheckboxGroup
        Dataset selector widget.

    Returns
    -------
    list[str]
        List of selected dataset names.

    """
    active = [n for n, cb in dataset_selector.boxes.items() if cb.value]
    if not active:
        msg = "Select at least one dataset."
        raise ValueError(msg)
    return active


def _maybe_log_parameter(vals: np.ndarray, *, key: str, use_log: bool) -> np.ndarray:
    """
    Apply log10 transformation to parameter values if applicable.

    Parameters
    ----------
    vals : np.ndarray
        Array of parameter values.
    key : str
        Parameter key.
    use_log : bool
        Whether to apply log10 transformation.

    Returns
    -------
    np.ndarray
        Transformed array of parameter values.

    """
    if not use_log or key not in LOG_PARAMETER_KEYS:
        return vals

    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return vals
    return np.log10(vals)


# =============================================================================
# HIST GRID (USED BY 1.1 + 1.2)
# =============================================================================


def _hist_grid(
    *,
    data_by_dataset: dict[str, dict[str, np.ndarray]],
    active_datasets: list[str],
    columns: list[str],
    title: str,
    bins: int,
    ncols: int,
    use_log: bool = False,
) -> Figure:
    """
    Create a grid of histograms for the specified columns.

    - One fixed color per dataset
    - Histogram and KDE share EXACT same distribution + color
    - Stable legend
    """
    # ------------------------------------------------------------------
    # Fixed, high-contrast colors (dataset-level)
    # ------------------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    dataset_colors = {name: cmap(i % 10) for i, name in enumerate(active_datasets)}

    n = len(columns)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(4.0 * ncols + 2.0, 2.8 * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.35],
        hspace=0.35,
        wspace=0.25,
    )

    axes: list[Axes] = []
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            if idx < n:
                axes.append(fig.add_subplot(gs[r, c]))

    # ------------------------------------------------------------------
    # Legend axis
    # ------------------------------------------------------------------
    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")

    legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active_datasets]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    for ax, key in zip(axes, columns, strict=False):
        for name in active_datasets:
            vals = data_by_dataset[name].get(key, np.array([]))
            vals = vals[np.isfinite(vals)]
            vals = _maybe_log_parameter(vals, key=key, use_log=use_log)

            if vals.size < 2:  # noqa: PLR2004
                continue

            color = dataset_colors[name]

            # Histogram
            _, bin_edges, _ = ax.hist(
                vals,
                bins=bins,
                color=color,
                alpha=0.35,
            )

            # KDE (same distribution, same color)
            if np.nanstd(vals) > 0:
                try:
                    kde = gaussian_kde(vals)
                    x = np.linspace(vals.min(), vals.max(), 300)
                    bw = bin_edges[1] - bin_edges[0]
                    ax.plot(
                        x,
                        kde(x) * vals.size * bw,
                        color=color,
                        lw=1.6,
                    )
                except LinAlgError:
                    pass

        suffix = " (log10)" if use_log and key in LOG_PARAMETER_KEYS else ""
        ax.set_title(f"{key}{suffix}", fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)

    # ------------------------------------------------------------------
    # Legend + finishing
    # ------------------------------------------------------------------
    ax_leg.legend(
        legend_handles,
        active_datasets,
        title="Dataset",
        loc="upper left",
    )

    fig.suptitle(title)
    fig.subplots_adjust(
        top=0.93,
        bottom=0.06,
        left=0.06,
        right=0.98,
    )

    return fig


# =============================================================================
# 1-1. META STATISTICS (LINEAR)
# =============================================================================


def plot_meta_statistics(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot meta statistics distributions.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets with their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the plot and controls.

    """
    names = list(datasets.keys())
    cache: dict[str, _StatCache] = {n: {"loaded": 0, "cols": {}} for n in names}

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame], dataset_selector: CheckboxGroup) -> Figure:
        """
        Plot meta statistics distributions for the selected datasets.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to include in the plot.
        datasets : dict[str, pd.DataFrame]
            Dictionary of datasets with their corresponding DataFrames.
        dataset_selector : CheckboxGroup
            Dataset selector widget.

        Returns
        -------
            Figure
                A Matplotlib Figure object containing the plot.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for m in df.iloc[entry["loaded"] : max_cases]["meta"]:
                flat = _flatten_dict_raw(m["statistics"])
                for k, v in flat.items():
                    entry["cols"].setdefault(k, []).append(v)

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)
        keys = sorted({k for n in active for k in cache[n]["cols"]})

        data = {n: {k: np.asarray(cache[n]["cols"].get(k, [])) for k in keys} for n in active}

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=keys,
            title=f"Meta statistics distributions (first {max_cases} cases)",
            bins=META_BINS,
            ncols=META_NCOLS,
        )

    ds = util.util_plot_components.ui_checkbox_datasets(dataset_names=names)
    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )


# =============================================================================
# 1-2. META PARAMETERS (OPTIONAL LOG)
# =============================================================================


def plot_meta_parameters(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot meta parameters distributions.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets with their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the plot and controls.

    """
    names = list(datasets.keys())
    cache: dict[str, _ParamCache] = {n: {"loaded": 0, "cols": {}} for n in names}

    log_cb = util.util_plot_components.ui_checkbox_log_scale()

    def _plot(max_cases: int, *, datasets: dict[str, pd.DataFrame], dataset_selector: CheckboxGroup, use_log: widgets.Checkbox) -> Figure:
        """
        Plot meta parameters distributions for the selected datasets.

        Parameters
        ----------
        max_cases : int
            Maximum number of cases to include in the plot.
        datasets : dict[str, pd.DataFrame]
            Dictionary of datasets with their corresponding DataFrames.
        dataset_selector : CheckboxGroup
            Dataset selector widget.
        use_log : widgets.Checkbox
            Checkbox widget to indicate whether to use log scale.

        Returns
        -------
            Figure
                A Matplotlib Figure object containing the plot.

        """
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for m in df.iloc[entry["loaded"] : max_cases]["meta"]:
                flat = _flatten_parameters_ordered(m["parameters"])
                for k in PARAMETER_ORDER:
                    entry["cols"].setdefault(k, []).append(flat.get(k, np.nan))

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)
        keys = [k for k in PARAMETER_ORDER if any(k in cache[n]["cols"] for n in active)]

        data = {n: {k: np.asarray(cache[n]["cols"].get(k, [])) for k in keys} for n in active}

        return _hist_grid(
            data_by_dataset=data,
            active_datasets=active,
            columns=keys,
            title=f"Meta parameter distributions (first {max_cases} cases)",
            bins=META_BINS,
            ncols=META_NCOLS,
            use_log=use_log.value,
        )

    ds = util.util_plot_components.ui_checkbox_datasets(dataset_names=names)
    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds, log_cb],
        dataset_selector=ds,
        use_log=log_cb,
    )


# =============================================================================
# 1-3. FIELD VALUE DISTRIBUTIONS (LINEAR)
# =============================================================================


def plot_field_value_distributions(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot field value distributions for the given datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets with their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the plot and controls.

    """
    names = list(datasets.keys())

    cache: dict[str, _FieldCache] = {
        n: {
            "loaded": 0,
            "data": {f: {s: [] for s in FIELD_STATS} for f in FIELDS},
        }
        for n in names
    }

    # ------------------------------------------------------------------
    # FIXED, HIGH-CONTRAST COLORS (one per dataset)
    # ------------------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    dataset_colors = {name: cmap(i % 10) for i, name in enumerate(names)}

    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        dataset_selector: CheckboxGroup,
    ) -> Figure:
        # ------------------------------------------------------------
        # Cache update
        # ------------------------------------------------------------
        for name, df in datasets.items():
            entry = cache[name]
            if max_cases <= entry["loaded"]:
                continue

            for _, row in df.iloc[entry["loaded"] : max_cases].iterrows():
                for f in FIELDS:
                    arr = np.asarray(row[f])
                    if arr.size == 0:
                        continue
                    entry["data"][f]["min"].append(float(np.nanmin(arr)))
                    entry["data"][f]["mean"].append(float(np.nanmean(arr)))
                    entry["data"][f]["max"].append(float(np.nanmax(arr)))

            entry["loaded"] = max_cases

        active = _selected_datasets(dataset_selector)

        # ------------------------------------------------------------
        # Legend handles (dataset-level, fixed color)
        # ------------------------------------------------------------
        legend_handles = [Line2D([], [], lw=6, color=dataset_colors[name], alpha=0.6) for name in active]

        # ------------------------------------------------------------
        # Figure layout
        # ------------------------------------------------------------
        nrows = len(FIELDS)
        ncols = len(FIELD_STATS)

        fig = plt.figure(figsize=(12.5, 2.4 * nrows))
        gs = fig.add_gridspec(
            nrows,
            ncols + 1,
            width_ratios=[1.0, 1.0, 1.0, 0.35],
            hspace=0.35,
            wspace=0.25,
        )

        axes = [[fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)]

        ax_leg = fig.add_subplot(gs[:, -1])
        ax_leg.axis("off")

        # ------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------
        for i, f in enumerate(FIELDS):
            for j, s in enumerate(FIELD_STATS):
                ax = axes[i][j]

                for name in active:
                    vals = np.asarray(cache[name]["data"][f][s])
                    vals = vals[np.isfinite(vals)]

                    if vals.size < 2:  # noqa: PLR2004
                        continue

                    color = dataset_colors[name]

                    # Histogram
                    _, bin_edges, _ = ax.hist(
                        vals,
                        bins=META_BINS,
                        color=color,
                        alpha=0.35,
                    )

                    # KDE with EXACT SAME DISTRIBUTION + COLOR
                    if np.nanstd(vals) > 0:
                        try:
                            kde = gaussian_kde(vals)
                            x = np.linspace(vals.min(), vals.max(), 300)
                            bw = bin_edges[1] - bin_edges[0]
                            ax.plot(
                                x,
                                kde(x) * vals.size * bw,
                                color=color,
                                lw=1.6,
                            )
                        except LinAlgError:
                            pass

                # Column titles
                if i == 0:
                    ax.set_title(s)

                # Row labels
                if j == 0:
                    ax.set_ylabel("Count")
                    ax.annotate(
                        f,
                        xy=(-0.25, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=11,
                        fontweight="bold",
                    )

                ax.set_xlabel("Value")
                ax.grid(True, linestyle="--", alpha=0.25)

        # ------------------------------------------------------------
        # Legend + finishing
        # ------------------------------------------------------------
        ax_leg.legend(
            legend_handles,
            active,
            title="Dataset",
            loc="upper left",
        )

        fig.suptitle(f"Field value distributions per channel (first {max_cases} cases)")
        fig.subplots_adjust(
            top=0.95,
            bottom=0.06,
            left=0.06,
            right=0.97,
        )

        return fig

    ds = util.util_plot_components.ui_checkbox_datasets(dataset_names=names)

    return util.util_plot.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=250,
        step_size=50,
        extra_widgets=[ds],
        dataset_selector=ds,
    )
