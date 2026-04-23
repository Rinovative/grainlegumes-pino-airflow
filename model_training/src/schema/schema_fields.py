"""
===============================================================================
 schema_fields.
===============================================================================
Author:  Rino M. Albertin
Project: GrainLegumes_PINO_project

DESCRIPTION
-----------
Single source of truth for non-kappa input and output field definitions.

Defines:
  - coordinate input fields (x, y)
  - scalar volume input fields with explicit COMSOL column mapping
  - output field mapping from internal names to COMSOL variables
  - canonical, deterministicD deterministic input channel ordering logic

Design principles:
  - kappa (permeability tensor) is handled elsewhere (kappa_schema)
  - this module is strictly declarative (names, order, mapping)
  - no torch / numpy / numerical logic allowed
  - deterministic channel ordering is enforced centrally

The canonical input order is:
  [ coordinates | kappa components | scalar inputs ]

This module does NOT decide:
  - how kappa is computed, transformed, or normalised
  - how fields are loaded or interpolated
  - how tensors are stacked or consumed
===============================================================================
"""  # noqa: D205

from __future__ import annotations

# ---------------------------------------------------------------------
# Coordinate fields (always present)
# ---------------------------------------------------------------------
COORD_FIELDS = ["x", "y"]

# ---------------------------------------------------------------------
# Scalar field inputs (material properties, boundary conditions, etc.)
# All are represented as volume fields in COMSOL
# ---------------------------------------------------------------------
SCALAR_INPUT_FIELDS = {
    "eps": "int4(x,y)",  # porosity (material field)
    "p_bc": "int5(x,y)",  # pressure boundary condition (volume-encoded)
}
# ---------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------
OUTPUT_FIELDS = ["p", "u", "v", "U"]


# ---------------------------------------------------------------------
# Canonical input field order (without kappa!)
# kappa is inserted dynamically between coords and scalars
# ---------------------------------------------------------------------
def canonical_input_order(kappa_fields: list[str]) -> list[str]:
    """
    Get the canonical input field order, given the kappa component names.

    Parameters
    ----------
    kappa_fields : list[str]
        List of kappa component names to include.

    Returns
    -------
    list[str]
        Canonical input field order.

    """
    return [
        *COORD_FIELDS,
        *kappa_fields,
        *SCALAR_INPUT_FIELDS.keys(),
    ]
