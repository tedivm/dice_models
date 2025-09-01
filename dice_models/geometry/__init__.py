"""Dice geometry module for generating 3D dice models."""

from .compatibility import PolyhedronGeometry
from .dice import DiceGeometry  # This needs to import from dice.py, not dice/
from .factory import DiceFactory
from .layouts import get_standard_number_layout
from .polyhedra import PolyhedronType

__all__ = [
    "DiceGeometry",
    "PolyhedronType",
    "PolyhedronGeometry",  # Backward compatibility
    "get_standard_number_layout",  # Backward compatibility
    "DiceFactory",  # New interface
]
