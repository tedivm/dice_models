"""Standard number layouts for dice types."""

from typing import List

from .factory import DiceFactory
from .polyhedra import PolyhedronType


def get_standard_number_layout(polyhedron_type: PolyhedronType) -> List[int]:
    """
    Get the standard number layout for a dice type.

    Args:
        polyhedron_type: The type of polyhedron

    Returns:
        List of numbers in face order
    """
    dice = DiceFactory.create_dice(polyhedron_type, radius=1.0)
    return dice.get_standard_number_layout()
