"""
Project source code organization.

Core modules: common, domain, data, learning, analysis, experiments.
Legacy modules (in transition): eda, util.
"""

from . import (
    analysis,
    common,
    data,
    domain,
    experiments,
    learning,
)

__all__ = [
    "analysis",
    "common",
    "data",
    "domain",
    "experiments",
    "learning",
]
