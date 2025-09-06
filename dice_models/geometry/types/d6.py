"""D6 (Cube) dice geometry implementation."""

import math
from typing import Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D6(BasePolyhedron):
    """
    Six-sided dice (cube) implementation.

    The D6 is a regular cube with 6 square faces.
    """

    @property
    def layouts(self) -> dict:
        """Return the available layouts for D6."""
        from ..layouts import LayoutType

        return {
            LayoutType.NAIVE: list(range(1, 7)),  # [1, 2, 3, 4, 5, 6]
            LayoutType.OPPOSING_BALANCED: [1, 6, 2, 5, 3, 4],  # Balanced arrangement
            LayoutType.OPPOSING_WEIGHTED: [6, 1, 5, 2, 4, 3],  # High numbers clustered
        }

    @property
    def sides(self) -> int:
        """Return the number of sides (6) for a cube."""
        return 6

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "CUBE"

    def _generate_vertices_and_faces(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cube vertices and faces.

        Args:
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Cube vertices (inscribed in sphere of given radius)
        a = radius / math.sqrt(3)
        vertices = np.array(
            [
                [-a, -a, -a],
                [a, -a, -a],
                [a, a, -a],
                [-a, a, -a],
                [-a, -a, a],
                [a, -a, a],
                [a, a, a],
                [-a, a, a],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2, 3],  # Bottom
                [4, 7, 6, 5],  # Top
                [0, 4, 5, 1],  # Front
                [2, 6, 7, 3],  # Back
                [0, 3, 7, 4],  # Left
                [1, 5, 6, 2],  # Right
            ]
        )

        return vertices, faces
