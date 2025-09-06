"""D20 (Icosahedron) dice geometry implementation."""

import math
from typing import Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D20(BasePolyhedron):
    """
    Twenty-sided dice (icosahedron) implementation.

    The D20 is a regular icosahedron with 20 triangular faces.
    """

    # 1 - 20 - A
    # 2 - 16 - B
    # 3 - 17 - C
    # 4 - 18 - D
    # 5 - 19 - E
    # 6 - 11 - F
    # 7 - 12 - G
    # 8 - 13 - H
    # 9 - 14 - I
    # 10 - 15 - J

    @property
    def layouts(self) -> dict:
        """Return the available layouts for D20."""
        from ..layouts import LayoutType

        return {
            LayoutType.NAIVE: list(range(1, 21)),  # [1, 2, 3, ..., 20]
            LayoutType.OPPOSING_BALANCED: [
                1,  # A
                19,  # B
                3,  # C
                17,  # D
                5,  # E
                15,  # F
                7,  # G
                13,  # H
                9,  # I
                11,  # J
                6,  # F
                14,  # G
                8,  # H
                12,  # I
                10,  # J
                2,  # B
                18,  # C
                4,  # D
                16,  # E
                20,  # A
            ],  # Balanced arrangement
            LayoutType.OPPOSING_WEIGHTED: [
                1,  # A
                2,  # B
                8,  # C
                5,  # D
                4,  # E
                6,  # F
                7,  # G
                3,  # H
                9,  # I
                10,  # J
                15,  # F
                14,  # G
                18,  # H
                12,  # I
                11,  # J
                19,  # B
                13,  # C
                16,  # D
                17,  # E
                20,  # A
            ],  # High numbers clustered
        }

    @property
    def sides(self) -> int:
        """Return the number of sides (20) for an icosahedron."""
        return 20

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "ICOSAHEDRON"

    def _generate_vertices_and_faces(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate icosahedron vertices and faces.

        Args:
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2

        # Scale to achieve desired radius
        scale = radius / math.sqrt(1 + phi**2)

        # Icosahedron vertices
        vertices = (
            np.array(
                [
                    [0, 1, phi],
                    [0, -1, phi],
                    [0, 1, -phi],
                    [0, -1, -phi],
                    [1, phi, 0],
                    [-1, phi, 0],
                    [1, -phi, 0],
                    [-1, -phi, 0],
                    [phi, 0, 1],
                    [-phi, 0, 1],
                    [phi, 0, -1],
                    [-phi, 0, -1],
                ]
            )
            * scale
        )

        # Icosahedron faces (20 triangular faces)
        faces = np.array(
            [
                [0, 1, 8],
                [0, 8, 4],
                [0, 4, 5],
                [0, 5, 9],
                [0, 9, 1],
                [1, 9, 7],
                [1, 7, 6],
                [1, 6, 8],
                [8, 6, 10],
                [8, 10, 4],
                [4, 10, 2],
                [4, 2, 5],
                [5, 2, 11],
                [5, 11, 9],
                [9, 11, 7],
                [7, 11, 3],
                [7, 3, 6],
                [6, 3, 10],
                [10, 3, 2],
                [2, 3, 11],
            ]
        )

        return vertices, faces
