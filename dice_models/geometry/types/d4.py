"""D4 (Tetrahedron) dice geometry implementation."""

import math
from typing import List, Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D4(BasePolyhedron):
    """
    Four-sided dice (tetrahedron) implementation.
    
    The D4 is a regular tetrahedron with 4 triangular faces.
    """

    @property
    def sides(self) -> int:
        """Return the number of sides (4) for a tetrahedron."""
        return 4

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "TETRAHEDRON"

    def get_standard_number_layout(self) -> List[int]:
        """
        Get the standard number layout for a D4.
        
        Returns:
            List of numbers [1, 2, 3, 4] in face order
        """
        return [1, 2, 3, 4]

    def _generate_vertices_and_faces(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate tetrahedron vertices and faces.
        
        Args:
            radius: The radius of the circumscribed sphere
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Regular tetrahedron vertices - corrected for proper geometry
        h = radius * math.sqrt(2 / 3)  # Height from base to top
        a = radius * math.sqrt(8 / 9)  # Side length of base triangle

        vertices = np.array([
            [a / 2, -a / (2 * math.sqrt(3)), -h / 3],  # Vertex 0
            [-a / 2, -a / (2 * math.sqrt(3)), -h / 3],  # Vertex 1
            [0, a / math.sqrt(3), -h / 3],  # Vertex 2
            [0, 0, 2 * h / 3],  # Vertex 3 (top)
        ])

        faces = np.array([
            [0, 2, 1],  # Bottom face (clockwise from outside)
            [0, 1, 3],  # Side face 1
            [1, 2, 3],  # Side face 2
            [2, 0, 3],  # Side face 3
        ])

        return vertices, faces
