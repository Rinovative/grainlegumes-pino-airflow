"""
Project source code organization.

Core modules: common, domain, datasets, learning, analysis, experiments.
Legacy modules (in transition): eda, util.
"""

from . import (
    analysis,
    common,
    datasets,
    domain,
    experiments,
    learning,
)

__all__ = [
    "analysis",
    "common",
    "datasets",
    "domain",
    "experiments",
    "learning",
]
