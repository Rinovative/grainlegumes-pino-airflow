"""
Architecture sensitivity plots for PINO/FNO/UNO evaluation.

This module analyses how architectural design choices influence model performance.
All plots operate purely on aggregated evaluation DataFrames and are intended for:

    - architecture comparison
    - scaling behaviour analysis
    - efficiency tradeoff studies
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def plot_error_vs_architecture_parameters(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse error sensitivity with respect to individual architecture parameters.

    For each architecture parameter, the relationship between the parameter value
    and the relative L2 error is visualised. Raw samples are shown as scatter points,
    while the median trend is overlaid for clarity.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping from dataset name to aggregated evaluation DataFrame.
        Each DataFrame must contain the following columns:
            - 'rel_l2'
            - 'arch_hidden_channels'
            - 'arch_n_layers'
            - 'arch_n_modes'

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one subplot per architecture parameter.

    """
    names = list(datasets.keys())

    arch_params = [
        "arch_hidden_channels",
        "arch_n_layers",
        "arch_n_modes",
    ]

    fig, axes = plt.subplots(
        1,
        len(arch_params),
        figsize=(5.5 * len(arch_params), 4.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, param in zip(axes, arch_params, strict=False):
        for name in names:
            df = datasets[name]

            if param not in df.columns:
                continue

            x = df[param].to_numpy(dtype=float)
            y = df["rel_l2"].to_numpy(dtype=float)

            ax.scatter(x, y, alpha=0.4, s=20)

            uniq = np.unique(x)
            med = [np.median(y[x == u]) for u in uniq]
            ax.plot(uniq, med, lw=2)

        ax.set_xlabel(param.replace("arch_", ""))
        ax.set_ylabel("Relative L2")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.suptitle("Error vs architecture parameters")
    fig.tight_layout()
    return fig


def plot_capacity_vs_performance(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse the tradeoff between model capacity and predictive performance.

    A simple capacity proxy is used:
        capacity = arch_hidden_channels x arch_n_layers x arch_n_modes

    This plot helps identify diminishing returns or inefficient over-parameterisation.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping from dataset name to aggregated evaluation DataFrame.
        Each DataFrame must contain:
            - 'arch_hidden_channels'
            - 'arch_n_layers'
            - 'arch_n_modes'
            - 'rel_l2'

    Returns
    -------
    matplotlib.figure.Figure
        Log-log scatter plot of capacity versus relative L2 error.

    """
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for name, df in datasets.items():
        h = df["arch_hidden_channels"].to_numpy(dtype=float)
        layer = df["arch_n_layers"].to_numpy(dtype=float)
        m = df["arch_n_modes"].to_numpy(dtype=float)

        capacity = h * layer * m
        error = df["rel_l2"].to_numpy(dtype=float)

        ax.scatter(capacity, error, alpha=0.45, s=25, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model capacity proxy")
    ax.set_ylabel("Relative L2")
    ax.set_title("Capacity vs performance")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig


def plot_parameter_efficiency(*, datasets: dict[str, pd.DataFrame]) -> Figure:
    """
    Analyse parameter efficiency across architectures.

    Efficiency is defined as:
        efficiency = rel_l2 x num_parameters

    Lower values indicate better use of parameters for achieving accuracy.
    This plot is particularly useful for comparing architectures such as
    FNO vs UNO under similar error regimes.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping from dataset name to aggregated evaluation DataFrame.
        Each DataFrame must contain:
            - 'rel_l2'
            - 'num_parameters'

    Returns
    -------
    matplotlib.figure.Figure
        Log-log scatter plot of parameter count versus efficiency metric.

    """
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for name, df in datasets.items():
        npar = df["num_parameters"].to_numpy(dtype=float)
        err = df["rel_l2"].to_numpy(dtype=float)

        efficiency = err * npar

        ax.scatter(npar, efficiency, alpha=0.45, s=25, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Relative L2 x parameters")
    ax.set_title("Parameter efficiency")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    return fig
