"""D6 (Cube) dice geometry implementation."""

import math
from typing import List, Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D6(BasePolyhedron):
    """
    Six-sided dice (cube) implementation.

    The D6 is a regular cube with 6 square faces.
    """

    @property
    def sides(self) -> int:
        """Return the number of sides (6) for a cube."""
        return 6

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "CUBE"

    def get_standard_number_layout(self) -> List[int]:
        """
        Get the standard number layout for a D6.

        Returns:
            List of numbers [1, 2, 3, 4, 5, 6] in face order
        """
        return [1, 2, 3, 4, 5, 6]

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
