"""Expose concise package aliases for dataset abstractions."""

from . import dataset_base as base
from . import dataset_modules as modules
from . import dataset_simulation as simulation

__all__ = [
    "base",
    "modules",
    "simulation",
]
