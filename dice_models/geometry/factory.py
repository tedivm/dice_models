"""Factory for creating dice geometry instances."""

from typing import Dict, Type

from .base.polyhedron import BasePolyhedron
from .polyhedra import PolyhedronType
from .types import D4, D6, D8, D10, D12, D20


class DiceFactory:
    """Factory class for creating dice geometry instances."""

    # Mapping from PolyhedronType enum to dice classes
    _DICE_CLASSES: Dict[PolyhedronType, Type[BasePolyhedron]] = {
        PolyhedronType.TETRAHEDRON: D4,
        PolyhedronType.CUBE: D6,
        PolyhedronType.OCTAHEDRON: D8,
        PolyhedronType.PENTAGONAL_TRAPEZOHEDRON: D10,
        PolyhedronType.DODECAHEDRON: D12,
        PolyhedronType.ICOSAHEDRON: D20,
    }

    # Mapping from number of sides to dice classes
    _SIDES_TO_DICE: Dict[int, Type[BasePolyhedron]] = {
        4: D4,
        6: D6,
        8: D8,
        10: D10,
        12: D12,
        20: D20,
    }

    @classmethod
    def create_dice(cls, polyhedron_type: PolyhedronType, radius: float = 1.0) -> BasePolyhedron:
        """
        Create a dice instance from a PolyhedronType enum.

        Args:
            polyhedron_type: The type of polyhedron to create
            radius: The radius of the circumscribed sphere

        Returns:
            Instance of the appropriate dice class

        Raises:
            ValueError: If polyhedron_type is not supported
        """
        if polyhedron_type not in cls._DICE_CLASSES:
            raise ValueError(f"Unsupported polyhedron type: {polyhedron_type}")

        dice_class = cls._DICE_CLASSES[polyhedron_type]
        return dice_class(radius=radius)

    @classmethod
    def create_dice_by_sides(cls, sides: int, radius: float = 1.0) -> BasePolyhedron:
        """
        Create a dice instance from the number of sides.

        Args:
            sides: The number of sides (4, 6, 8, 10, 12, or 20)
            radius: The radius of the circumscribed sphere

        Returns:
            Instance of the appropriate dice class

        Raises:
            ValueError: If sides is not a valid dice type
        """
        if sides not in cls._SIDES_TO_DICE:
            valid_sides = list(cls._SIDES_TO_DICE.keys())
            raise ValueError(f"Invalid number of sides: {sides}. Valid options: {valid_sides}")

        dice_class = cls._SIDES_TO_DICE[sides]
        return dice_class(radius=radius)

    @classmethod
    def get_supported_types(cls) -> list[PolyhedronType]:
        """
        Get list of all supported polyhedron types.

        Returns:
            List of supported PolyhedronType enum values
        """
        return list(cls._DICE_CLASSES.keys())

    @classmethod
    def get_supported_sides(cls) -> list[int]:
        """
        Get list of all supported side counts.

        Returns:
            List of supported side counts
        """
        return list(cls._SIDES_TO_DICE.keys())
