"""
===============================================================================
 schema_training.
===============================================================================
Author:  Rino M. Albertin
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Training schema defining which fields are used as model inputs and outputs.

This module defines:
  - default input channel selections for training
  - default output channel selections for training
  - canonical ordering rules for training tensors

Design principles:
  - purely declarative (no numerical logic)
  - independent of data loading and preprocessing
  - independent of model architecture (FNO / PINO / UNO)
  - uses canonical field names defined elsewhere

This module answers ONLY:
  "Which fields does the model see during training?"
===============================================================================
"""  # noqa: D205

from __future__ import annotations

# ---------------------------------------------------------------------
# Default training inputs
# ---------------------------------------------------------------------
# These are INTERNAL, CANONICAL field names
# (after kappa_schema + build_batch_dataset)
# ---------------------------------------------------------------------

DEFAULT_INPUTS_2D = [
    "x",
    "y",
    "kxx",
    "kyy",
    "kxy",
    "phi",
    "p_bc",
]

DEFAULT_INPUTS_3D = [
    "x",
    "y",
    "z",
    "kxx",
    "kyy",
    "kzz",
    "kxy",
    "kxz",
    "kyz",
    "phi",
    "p_bc",
]

# ---------------------------------------------------------------------
# Default training outputs
# ---------------------------------------------------------------------

DEFAULT_OUTPUTS_2D = [
    "p",
    "u",
    "v",
]

DEFAULT_OUTPUTS_3D = [
    "p",
    "u",
    "v",
    "w",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def default_training_inputs(dim: int) -> list[str]:
    """
    Return default input field names for the given problem dimension.

    Parameters
    ----------
    dim : int
        Problem dimension (2 or 3).

    Returns
    -------
    list[str]
        Canonical input field names for training.

    """
    if dim == 2:  # noqa: PLR2004
        return DEFAULT_INPUTS_2D
    if dim == 3:  # noqa: PLR2004
        return DEFAULT_INPUTS_3D
    msg = f"Unsupported dimension: {dim}"
    raise ValueError(msg)


def default_training_outputs(dim: int) -> list[str]:
    """
    Return default output field names for the given problem dimension.

    Parameters
    ----------
    dim : int
        Problem dimension (2 or 3).

    Returns
    -------
    list[str]
        Canonical output field names for training.

    """
    if dim == 2:  # noqa: PLR2004
        return DEFAULT_OUTPUTS_2D
    if dim == 3:  # noqa: PLR2004
        return DEFAULT_OUTPUTS_3D
    msg = f"Unsupported dimension: {dim}"
    raise ValueError(msg)
