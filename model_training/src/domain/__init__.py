"""
Domain-specific definitions and semantic contracts.

Submodules exposed through clean aliases:
- fields: field names, coordinates, inputs and outputs
- permeability: permeability tensor naming, ordering and mapping
- field_sets: default field selections for model inputs and outputs
"""

from . import domain_field_sets as field_sets
from . import domain_fields as fields
from . import domain_permeability as permeability

__all__ = [
    "field_sets",
    "fields",
    "permeability",
]
