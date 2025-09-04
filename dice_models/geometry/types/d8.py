"""D8 (Octahedron) dice geometry implementation."""

from typing import Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D8(BasePolyhedron):
    """
    Eight-sided dice (octahedron) implementation.

    The D8 is a regular octahedron with 8 triangular faces.
    """

    @property
    def layouts(self) -> dict:
        """Return the available layouts for D8."""
        from ..layouts import LayoutType

        return {
            LayoutType.NAIVE: list(range(1, 9)),  # [1, 2, 3, 4, 5, 6, 7, 8]
            LayoutType.OPPOSING_BALANCED: [
                1,
                8,
                2,
                7,
                3,
                6,
                4,
                5,
            ],  # Balanced arrangement
            LayoutType.OPPOSING_WEIGHTED: [
                8,
                1,
                7,
                2,
                6,
                3,
                5,
                4,
            ],  # High numbers clustered
        }

    @property
    def sides(self) -> int:
        """Return the number of sides (8) for an octahedron."""
        return 8

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "OCTAHEDRON"

    def _generate_vertices_and_faces(
        self, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate octahedron vertices and faces.

        Args:
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        vertices = np.array(
            [
                [radius, 0, 0],  # +X
                [-radius, 0, 0],  # -X
                [0, radius, 0],  # +Y
                [0, -radius, 0],  # -Y
                [0, 0, radius],  # +Z
                [0, 0, -radius],  # -Z
            ]
        )

        faces = np.array(
            [
                [0, 2, 4],  # +X+Y+Z
                [0, 4, 3],  # +X+Z-Y
                [0, 3, 5],  # +X-Y-Z
                [0, 5, 2],  # +X-Z+Y
                [1, 4, 2],  # -X+Z+Y
                [1, 3, 4],  # -X-Y+Z
                [1, 5, 3],  # -X-Z-Y
                [1, 2, 5],  # -X+Y-Z
            ]
        )

        return vertices, faces
