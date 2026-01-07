"""
Outlier and extreme case analysis plots for PINO/FNO evaluation.

This module provides functions to analyze and visualize outlier cases
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib import cm

from src import util

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = ["p", "u", "v", "U"]
UNIT_MAP = {"p": "Pa", "u": "m/s", "v": "m/s", "U": "m/s"}

N_LEVELS = 10
MASK_THRESHOLD = 1e-4

# ============================================================================
# HELPERS
# ============================================================================


def _as_series(row: pd.Series | pd.DataFrame) -> pd.Series:
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _scalar(x: Any) -> float:
    if isinstance(x, pd.Series):
        x = x.iloc[0]

    if x is None or pd.isna(x):
        return float("nan")

    try:
        xf = float(x)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(xf):
        return float("nan")

    return xf


def _fmt_num(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(xf):
        return ""
    if 1e-4 <= abs(xf) < 1e4:  # noqa: PLR2004
        return f"{xf:.3g}".rstrip("0").rstrip(".")
    return f"{xf:.2e}"


# =============================================================================
# GLOBAL CACHES
# =============================================================================

_npz_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
_case_map_cache: dict[str, dict[str, list[int]]] = {}
_ref_case_cache: dict[str, int] = {}


# =============================================================================
# NPZ LOADING
# =============================================================================


def _load_npz(row: pd.Series | pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load prediction, ground truth, error, and kappa from NPZ file.

    Uses a global cache to avoid redundant loading.

    Parameters
    ----------
    row : pd.Series
        DataFrame row containing the 'npz_path' column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
        Tuple containing prediction, ground truth, error, kappa, and kappa names.

    """
    row = _as_series(row)
    key = str(Path(row["npz_path"]))
    if key in _npz_cache:
        return _npz_cache[key]

    data = np.load(key, allow_pickle=True)
    pred = data["pred"][0]
    gt = data["gt"][0]
    err = data["err"][0]
    kappa = data["kappa"][0]
    kappa_names = list(data["kappa_names"])

    _npz_cache[key] = (pred, gt, err, kappa, kappa_names)
    return pred, gt, err, kappa, kappa_names


# =============================================================================
# METRICS
# =============================================================================


def _rel_l2(err: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute relative L2 error.

    Parameters
    ----------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground truth array.

    Returns
    -------
    float
        Relative L2 error.

    """
    return float(np.linalg.norm(err) / (np.linalg.norm(gt) + 1e-12))


def _rel_l2_per_channel(err: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Compute relative L2 error per channel.

    Parameters
    ----------
    err : np.ndarray
        Error array.
    gt : np.ndarray
        Ground truth array.

    Returns
    -------
    dict[str, float]
        Relative L2 error per channel.

    """
    return {ch: _rel_l2(err[i], gt[i]) for i, ch in enumerate(CHANNELS)}


# =============================================================================
# REFERENCE CASE (PARAMETER CENTER)
# =============================================================================


def _select_reference_case(df: pd.DataFrame) -> int:
    """
    Select the reference case based on the parameter center (median).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cases.

    Returns
    -------
    int
        Index of the reference case.

    """
    par_cols = [c for c in df.columns if c.startswith("par_") and c != "par_seed"]

    if not par_cols:
        return int(df.index[len(df) // 2])

    P = df[par_cols].astype(float)
    center = P.median()
    dist = ((P - center) / (P.std() + 1e-12)).pow(2).sum(axis=1)
    return int(dist.idxmin())


# =============================================================================
# WORST-CASE SELECTION (PER CHANNEL)
# =============================================================================


def _select_topk_per_channel(*, datasets: dict[str, pd.DataFrame], k: int) -> dict[str, dict[str, list[int]]]:
    """
    Select the top-k worst cases per channel based on relative L2 error.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of datasets.
    k : int
        Number of top cases to select per channel.

    Returns
    -------
    dict[str, dict[str, list[int]]]
        Dictionary mapping dataset names to channel names to lists of case indices.

    """
    out: dict[str, dict[str, list[int]]] = {}

    for name, df in datasets.items():
        out[name] = {}
        for ci, ch in enumerate(CHANNELS):
            scores = []
            for idx, row in df.iterrows():
                _, gt, err, _, _ = _load_npz(row)
                scores.append((idx, _rel_l2(err[ci], gt[ci])))
            scores.sort(key=lambda x: x[1], reverse=True)
            out[name][ch] = [i for i, _ in scores[:k]]

    return out


# =============================================================================
# KAPPA AGGREGATION
# =============================================================================


def _aggregate_kappa(kappa: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Aggregate kappa values into a single field.

    Parameters
    ----------
    kappa : np.ndarray
        Kappa values with shape (K, ny, nx).
    names : list[str]
        Names of the kappa components.

    Returns
    -------
        np.ndarray
            Aggregated kappa field.

    """
    K = kappa.shape[0]
    name_to_idx = {name.lower(): i for i, name in enumerate(names)}

    diag_keys = ["kappaxx", "kappayy", "kappazz"]
    diag_indices = [name_to_idx[k] for k in diag_keys if k in name_to_idx]

    if K == 1:
        return kappa[0]

    if K == 2 and len(diag_indices) == 2:  # noqa: PLR2004
        ixx, iyy = diag_indices
        return (kappa[ixx] + kappa[iyy]) / 2.0

    if K == 4 and len(diag_indices) >= 2:  # noqa: PLR2004
        ixx = name_to_idx.get("kappaxx", diag_indices[0])
        iyy = name_to_idx.get("kappayy", diag_indices[1])
        return (kappa[ixx] + kappa[iyy]) / 2.0

    if K == 9 and len(diag_indices) == 3:  # noqa: PLR2004
        ixx = name_to_idx["kappaxx"]
        iyy = name_to_idx["kappayy"]
        izz = name_to_idx["kappazz"]
        return (kappa[ixx] + kappa[iyy] + kappa[izz]) / 3.0

    return kappa.mean(axis=0)


# =============================================================================
# SHARED 4x4 PLOT KERNEL
# =============================================================================


def _plot_prediction_overview_case(
    *,
    row: pd.Series,
    dataset_name: str,
    case_label: str,
    error_mode: widgets.ValueWidget,
) -> Figure:
    """
    Plot a 4x4 overview of prediction, ground truth, error, and kappa for a given case.

    Parameters
    ----------
    row : pd.Series
        The DataFrame row containing the case data.
    dataset_name : str
        The name of the dataset.
    case_label : str
        A label for the case (e.g., "Case 42 | channel=u").
    error_mode : widgets.ValueWidget
        A widget indicating the error mode ("MAE" or "RelErr").

    Returns
    -------
        Figure
            The generated matplotlib Figure.

    """
    pred, gt, err, kappa, kappa_names = _load_npz(row)

    Lx = _scalar(row["geom_Lx"])
    Ly = _scalar(row["geom_Ly"])
    ny, nx = pred[0].shape
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    fig, axes = plt.subplots(4, 4, figsize=(20, 9))

    kappa_field = _aggregate_kappa(kappa, kappa_names)
    kappa_levels = util.util_plot_components.compute_levels(kappa_field, N_LEVELS)

    kappa_log_field = np.log10(np.maximum(kappa_field, 1e-30))
    kappa_log_levels = util.util_plot_components.compute_levels(kappa_log_field, N_LEVELS)

    nrows = 4  # fixed layout

    for r, ch in enumerate(CHANNELS):
        is_last = r == nrows - 1

        # Prediction
        ax = axes[r, 0]
        im = ax.contourf(X, Y, pred[r], levels=util.util_plot_components.compute_levels(pred[r]), cmap="turbo")
        if ch in {"u", "v", "U"}:
            util.util_plot_components.overlay_streamlines(ax, X, Y, pred[1], pred[2])
        ax.set_title(f"{ch} pred [{UNIT_MAP[ch]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(util.util_plot_components.choose_colorbar_formatter(*im.get_clim()))
        util.util_plot_components.apply_axis_labels(ax, 0, Lx, Ly, is_last_row=is_last)

        # Ground truth
        ax = axes[r, 1]
        im = ax.contourf(X, Y, gt[r], levels=util.util_plot_components.compute_levels(gt[r]), cmap="turbo")
        if ch in {"u", "v", "U"}:
            util.util_plot_components.overlay_streamlines(ax, X, Y, gt[1], gt[2])
        ax.set_title(f"{ch} true [{UNIT_MAP[ch]}]")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(util.util_plot_components.choose_colorbar_formatter(*im.get_clim()))
        util.util_plot_components.apply_axis_labels(ax, 1, Lx, Ly, is_last_row=is_last)

        # Error
        ax = axes[r, 2]
        if error_mode.value == "MAE":
            field = np.abs(err[r])
            field = np.nan_to_num(field, nan=0.0)
            levels = np.linspace(
                0.0,
                np.nanquantile(field, 0.99),
                N_LEVELS,
            )
            title = f"{ch} MAE [{UNIT_MAP[ch]}]"
        else:
            abs_err = np.abs(err[r])
            true_abs = np.abs(gt[r])
            field = np.asarray(abs_err / (true_abs + 1e-12) * 100.0, dtype=float)
            field[true_abs < MASK_THRESHOLD] = np.nan
            levels = np.linspace(
                0.0,
                np.nanquantile(field, 0.99),
                N_LEVELS,
            )
            title = f"{ch} rel err [%]"

        im = ax.contourf(X, Y, field, levels=levels, cmap="Blues")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.ax.yaxis.set_major_formatter(util.util_plot_components.choose_colorbar_formatter(*im.get_clim()))
        util.util_plot_components.apply_axis_labels(ax, 2, Lx, Ly, is_last_row=is_last)

        # Kappa
        ax = axes[r, 3]
        if r == 0:
            im = ax.contourf(
                X,
                Y,
                kappa_field,
                levels=kappa_levels,
                cmap="viridis",
            )
            ax.set_title("kappa [m²]")

        elif r == 1:
            im = ax.contourf(
                X,
                Y,
                kappa_log_field,
                levels=kappa_log_levels,
                cmap="viridis",
            )
            ax.set_title("log10(kappa) [m²]")
        else:
            ax.axis("off")
            continue

        # HIER:
        fig.colorbar(im, ax=ax, fraction=0.04)
        util.util_plot_components.apply_axis_labels(ax, 3, Lx, Ly, is_last_row=is_last)

    fig.suptitle(f"{dataset_name} — {case_label}", fontsize=14)
    fig.tight_layout()
    return fig


# =============================================================================
# 5.1 TABLE VIEW — REFERENCE INCLUDED IN TABLE
# =============================================================================


def _style_rank_diverging(df: pd.DataFrame, columns: list[str], kind: pd.Series) -> pd.DataFrame:
    """
    Style function to highlight cells based on their rank diverging from the reference case.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be styled.
    columns : list[str]
        The list of columns to apply the styling to.
    kind : pd.Series
        A Series indicating the kind of each row (e.g., "reference" or "worst").

    Returns
    -------
        pd.DataFrame
            A DataFrame with the same shape as `df`, containing the style strings for each cell.

    """
    out = {}

    cmap_low = cm.get_cmap("Blues")
    cmap_high = cm.get_cmap("Purples")

    ref_idx = kind[kind == "reference"].index[0]

    for c in columns:
        col = df[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue

        ref_val = col.loc[ref_idx]

        lower = col[col < ref_val]
        higher = col[col > ref_val]

        # Ranks getrennt
        r_low = lower.rank(method="average", ascending=True)
        r_high = higher.rank(method="average", ascending=True)

        def _cell(
            idx: int,
            *,
            col: pd.Series = col,
            r_low: pd.Series = r_low,
            r_high: pd.Series = r_high,
        ) -> str:
            if idx == ref_idx or not np.isfinite(col.loc[idx]):
                return "background-color: white"

            # ---------- smaller than reference ----------
            if idx in r_low.index:
                t = (r_low.loc[idx] - 1) / max(len(r_low) - 1, 1)
                x = 0.75 - 0.5 * t  # [0.75 → 0.25]
                cr, cg, cb, _ = cmap_low(x)

            # ---------- greater than reference ----------
            elif idx in r_high.index:
                t = (r_high.loc[idx] - 1) / max(len(r_high) - 1, 1)
                x = 0.25 + 0.5 * t  # [0.25 → 0.75]
                cr, cg, cb, _ = cmap_high(x)

            else:
                return "background-color: white"

            return f"background-color: rgba({int(cr * 255)}, {int(cg * 255)}, {int(cb * 255)}, 0.65)"

        out[c] = [_cell(i) for i in df.index]

    return pd.DataFrame(out)


def plot_outlier_tables_per_channel(*, datasets: dict[str, pd.DataFrame], k: int = 5) -> widgets.VBox:
    """
    Plot tables showing the worst k cases per channel, including the reference case.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    k : int, optional
        The number of worst cases to select per channel, by default 5.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the tables for each channel.

    """
    run_btn = widgets.Button(description="Compute tables", button_style="primary", icon="play")
    dataset_sel = widgets.Dropdown(options=list(datasets), description="Dataset:")
    output = widgets.Output()

    def _parameter_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith("par_") and c != "par_seed"]

    def _render(_: Any = None) -> None:
        output.clear_output(wait=True)
        name = dataset_sel.value
        df = datasets[name]
        par_cols = _parameter_columns(df)

        if name not in _case_map_cache:
            _case_map_cache[name] = _select_topk_per_channel(datasets={name: df}, k=k)[name]
            _ref_case_cache[name] = _select_reference_case(df)

        with output:
            display(Markdown(f"### Dataset: `{name}`"))

            for ch in CHANNELS:
                display(Markdown(f"#### Channel `{ch}`"))

                rows = []

                # Worst cases
                for idx in _case_map_cache[name][ch]:
                    row = _as_series(df.loc[idx])
                    _, gt, err, _, _ = _load_npz(row)
                    rels = _rel_l2_per_channel(err, gt)

                    r = {
                        "case_index": int(row["case_index"]),
                        f"rel_l2[{ch}]": rels[ch],
                        "rel_l2_global": _scalar(row["rel_l2"]),
                        "l2_global": _scalar(row["l2"]),
                        "__kind__": "worst",
                    }
                    for p in par_cols:
                        r[p] = _scalar(row[p])
                    rows.append(r)

                # Reference case (APPENDED)
                ref_idx = _ref_case_cache[name]
                row = _as_series(df.loc[ref_idx])
                _, gt, err, _, _ = _load_npz(row)
                rels = _rel_l2_per_channel(err, gt)

                r = {
                    "case_index": f"{int(row['case_index'])} (Ref)",
                    f"rel_l2[{ch}]": rels[ch],
                    "rel_l2_global": _scalar(row["rel_l2"]),
                    "l2_global": _scalar(row["l2"]),
                    "__kind__": "reference",
                }
                for p in par_cols:
                    r[p] = _scalar(row[p])
                rows.append(r)

                df_out = pd.DataFrame(rows)
                kind = df_out.pop("__kind__")

                display(
                    df_out.style.format({c: _fmt_num for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])})
                    .apply(
                        lambda _, df_out=df_out, kind=kind: _style_rank_diverging(
                            df_out,
                            [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c]) and c != "case_index"],
                            kind,
                        ),
                        axis=None,
                    )
                    .apply(
                        lambda _, df_out=df_out, kind=kind: ["" if kind.loc[i] == "reference" else "background-color: white" for i in df_out.index],
                        axis=0,
                        subset=["case_index"],
                    )
                )

    run_btn.on_click(_render)
    return widgets.VBox([widgets.HBox([dataset_sel, run_btn]), output])


# =============================================================================
# 5.2 FIELD PLOT — WORST PER-CHANNEL (REFERENCE LAST, LAZY)
# =============================================================================


def _num_cases_outlier_viewer(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    k: int,
    channel: str,
) -> int:
    """
    Determine the number of cases to display in the outlier viewer.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    df : pd.DataFrame
        The DataFrame containing the cases.
    k : int
        The number of worst cases to select per channel.
    channel : str
        The channel to focus on.

    Returns
    -------
        int
            The number of cases to display (worst cases + 1 reference).

    """
    # Lazy cache init
    if dataset_name not in _case_map_cache:
        _case_map_cache[dataset_name] = _select_topk_per_channel(datasets={dataset_name: df}, k=k)[dataset_name]
        _ref_case_cache[dataset_name] = _select_reference_case(df)

    # worst cases + 1 reference
    return len(_case_map_cache[dataset_name][channel]) + 1


def plot_outlier_cases_per_channel(*, datasets: dict[str, pd.DataFrame], k: int = 5) -> widgets.VBox:
    """
    Plot an interactive viewer for the worst k cases per channel, including the reference case.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    k : int, optional
        The number of worst cases to select per channel, by default 5.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the interactive case viewer.

    """
    ch_sel = util.util_plot_components.ui_dropdown_channel()
    err_sel = util.util_plot_components.ui_radio_error_mode()

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str, focus_channel: widgets.ValueWidget, error_mode: widgets.ValueWidget) -> Figure:
        """
        Plot function for the interactive case viewer.

        Parameters
        ----------
        idx : int
            The index of the case to plot.
        df : pd.DataFrame
            The DataFrame containing the cases.
        dataset_name : str
            The name of the dataset.
        focus_channel : widgets.ValueWidget
            The widget indicating the focus channel.
        error_mode : widgets.ValueWidget
            The widget indicating the error mode.

        Returns
        -------
            Figure
                The generated matplotlib Figure.

        """
        if dataset_name not in _case_map_cache:
            _case_map_cache[dataset_name] = _select_topk_per_channel(datasets={dataset_name: df}, k=k)[dataset_name]
            _ref_case_cache[dataset_name] = _select_reference_case(df)

        ch = focus_channel.value
        worst = _case_map_cache[dataset_name][ch]
        n = len(worst)

        if idx < n:
            df_idx = worst[idx]
            row = _as_series(df.loc[df_idx])
            case_id = row.get("case_index", df_idx)
            label = f"Case {case_id} | channel={ch}"
        else:
            ref_idx = _ref_case_cache[dataset_name]
            row = _as_series(df.loc[ref_idx])
            case_id = row.get("case_index", ref_idx)
            label = f"Case {case_id} | channel={ch} (reference)"

        return _plot_prediction_overview_case(
            row=row,
            dataset_name=dataset_name,
            case_label=label,
            error_mode=error_mode,
        )

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        enable_dataset_dropdown=True,
        extra_widgets=[ch_sel, err_sel],
        focus_channel=ch_sel,
        error_mode=err_sel,
        n_cases_fn=lambda name, df: _num_cases_outlier_viewer(
            name,
            df,
            k=k,
            channel=str(ch_sel.value),
        ),
    )


# =============================================================================
# 5.3 TABLE VIEW — EXTREME INPUT PARAMETERS
# =============================================================================


def plot_extreme_input_table(*, datasets: dict[str, pd.DataFrame]) -> widgets.HBox:
    """
    Plot tables summarizing extreme input parameter values for each dataset.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the summary tables.

    """
    tables: list[widgets.Output] = []

    for name, df in datasets.items():
        out = widgets.Output()

        with out:
            par_cols = [c for c in df.columns if c.startswith("par_") and c != "par_seed"]
            ref_idx = _select_reference_case(df)
            ref_row = _as_series(df.loc[ref_idx])

            rows = []
            for p in par_cols:
                col = df[p].astype(float)
                rows.append(
                    {
                        "parameter": p,
                        "min": col.min(),
                        "ref": ref_row[p],
                        "max": col.max(),
                        "min/ref": col.min() / ref_row[p] if ref_row[p] != 0 else np.nan,
                        "max/ref": col.max() / ref_row[p] if ref_row[p] != 0 else np.nan,
                    }
                )

            df_out = pd.DataFrame(rows).set_index("parameter")

            display(Markdown(f"### Dataset: `{name}`"))
            display(df_out.style.format(dict.fromkeys(df_out.columns, _fmt_num)))

        tables.append(out)

    return widgets.HBox(tables)


# =============================================================================
# 5.4 FIELD PLOT — EXTREME INPUT CASES
# =============================================================================


def _candidate_input_columns(*, datasets: dict[str, pd.DataFrame]) -> list[str]:
    """
    Identify candidate input parameter columns from the datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
    list[str]
            A list of candidate input parameter column names.

    """
    first_df = next(iter(datasets.values()))
    return [c for c in first_df.columns if c.startswith("par_") and c != "par_seed"]


def _select_extreme_inputs(*, datasets: dict[str, pd.DataFrame], column: str) -> dict[str, pd.DataFrame]:
    """
    Select cases with extreme values for a given input parameter column.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.
    column : str
        The input parameter column to analyze.

    Returns
    -------
        dict[str, pd.DataFrame]
            A dictionary mapping dataset names to DataFrames containing the extreme cases.

    """

    def _to_scalar(v: Any) -> float:
        if v is None or pd.isna(v):
            return float("nan")

        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return float("nan")

        xf = float(arr) if arr.ndim == 0 else float(np.nanmean(arr))

        if not np.isfinite(xf):
            return float("nan")

        return xf

    out = {}
    for name, df in datasets.items():
        v = df[column].apply(_to_scalar)
        out[name] = df.loc[[v.idxmin(), v.idxmax()]].reset_index(drop=True)
    return out


def plot_extreme_input_cases(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Plot an interactive viewer for cases with extreme input parameter values.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary mapping dataset names to their corresponding DataFrames.

    Returns
    -------
        widgets.VBox
            A VBox widget containing the interactive case viewer.

    """
    inp_sel = util.util_plot_components.ui_dropdown_input_parameter(parameters=_candidate_input_columns(datasets=datasets))
    err_sel = util.util_plot_components.ui_radio_error_mode()

    def _plot(idx: int, *, df: pd.DataFrame, dataset_name: str, error_mode: widgets.ValueWidget) -> Figure:
        """
        Plot function for the interactive extreme input case viewer.

        Parameters
        ----------
        idx : int
            The index of the case to plot.
        df : pd.DataFrame
            The DataFrame containing the cases.
        dataset_name : str
            The name of the dataset.
        error_mode : widgets.ValueWidget
            The widget indicating the error mode.

        Returns
        -------
            Figure
                The generated matplotlib Figure.

        """
        column = inp_sel.value
        if column is None:
            msg = "No input parameter selected"
            raise RuntimeError(msg)

        extremes = _select_extreme_inputs(
            datasets={dataset_name: df},
            column=column,
        )[dataset_name]

        row = extremes.iloc[idx]

        kind = "MIN" if idx == 0 else "MAX"
        value = row[column]

        # Case-Nummer sauber bestimmen
        case_id = row.get("case_index", idx)

        case_label = f"Case {case_id} | {inp_sel.value} = {value:.3g} ({kind})"

        return _plot_prediction_overview_case(
            row=row,
            dataset_name=dataset_name,
            case_label=case_label,
            error_mode=error_mode,
        )

    return util.util_plot.make_interactive_case_viewer(
        plot_func=_plot,
        datasets=datasets,
        enable_dataset_dropdown=True,
        extra_widgets=[inp_sel, err_sel],
        error_mode=err_sel,
        n_cases_fn=lambda *_: 2,
    )
