"""Standard number layouts for dice types."""

from enum import Enum
from typing import List

from .factory import DiceFactory
from .polyhedra import PolyhedronType


class LayoutType(Enum):
    """
    Enumeration of different layout strategies for dice numbers.

    NAIVE: Simple sequential placement (current behavior).
    OPPOSING_BALANCED: Opposing faces sum to same number, balanced distribution.
    OPPOSING_WEIGHTED: Opposing faces sum to same number, high/low clustering.
    """

    NAIVE = "naive"
    OPPOSING_BALANCED = "opposing-balanced"
    OPPOSING_WEIGHTED = "opposing-weighted"


def get_standard_number_layout(
    polyhedron_type: PolyhedronType,
    layout_type: LayoutType = LayoutType.OPPOSING_BALANCED,
) -> List[int]:
    """
    Get the standard number layout for a dice type.

    Args:
        polyhedron_type: The type of polyhedron
        layout_type: The layout strategy to use

    Returns:
        List of numbers in face order
    """
    dice = DiceFactory.create_dice(polyhedron_type, radius=1.0)
    return dice.get_layout(layout_type)


def get_layout_description(layout_type: LayoutType) -> str:
    """
    Get a human-readable description of a layout type.

    Args:
        layout_type: The layout type to describe

    Returns:
        Description string
    """
    descriptions = {
        LayoutType.NAIVE: "Simple sequential placement (numbers placed adjacently)",
        LayoutType.OPPOSING_BALANCED: "Opposing faces sum consistently, balanced high/low distribution",
        LayoutType.OPPOSING_WEIGHTED: "Opposing faces sum consistently, high numbers clustered together",
    }
    return descriptions.get(layout_type, "Unknown layout type")
