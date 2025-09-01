"""D10 (Pentagonal Trapezohedron) dice geometry implementation."""

import math
from typing import List, Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D10(BasePolyhedron):
    """
    Ten-sided dice (pentagonal trapezohedron) implementation.

    The D10 is a pentagonal trapezohedron with 10 kite-shaped faces.
    """

    @property
    def sides(self) -> int:
        """Return the number of sides (10) for a pentagonal trapezohedron."""
        return 10

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "PENTAGONAL_TRAPEZOHEDRON"

    def get_standard_number_layout(self) -> List[int]:
        """
        Get the standard number layout for a D10.

        Returns:
            List of numbers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in face order
        """
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def _generate_vertices_and_faces(
        self, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pentagonal trapezohedron (D10) vertices and faces.

        Creates a proper pentagonal trapezohedron with 10 kite-shaped faces that form
        a watertight mesh. The vertex positions are carefully tuned to make the
        triangle pairs that form each kite face as coplanar as possible.

        Args:
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        vertices = []

        # Polar vertices - stretch vertically to make die taller
        polar_height = radius * 1.2  # Increased from 0.7 to make it much taller
        vertices.append([0, 0, polar_height])  # vertex 0: top pole
        vertices.append([0, 0, -polar_height])  # vertex 1: bottom pole

        # Adjust the ring parameters to optimize edge length ratios for coplanarity
        # The key is to break the symmetry between radial and diagonal edge lengths

        # Keep symmetry: both rings same radius, but dramatically adjust heights
        ring1_radius = (
            radius * 1.1
        )  # Much larger rings to make ring-to-ring edges much longer
        ring2_radius = radius * 1.1  # Same radius to maintain top/bottom symmetry

        # Move rings MUCH closer to center to make pole-to-ring edges much shorter
        ring1_height = polar_height * 0.1  # Upper ring very close to center
        ring2_height = (
            -polar_height * 0.1
        )  # Lower ring very close to center (symmetric)

        # Ring 1: 5 vertices (upper ring) - same radius for symmetry
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = ring1_radius * math.cos(angle)
            y = ring1_radius * math.sin(angle)
            vertices.append([x, y, ring1_height])  # vertices 2-6

        # Ring 2: 5 vertices (lower ring) - same radius for symmetry
        for i in range(5):
            angle = 2 * math.pi * i / 5 + math.pi / 5  # 36-degree twist (standard)
            x = ring2_radius * math.cos(angle)
            y = ring2_radius * math.sin(angle)
            vertices.append([x, y, ring2_height])  # vertices 7-11

        vertices = np.array(vertices)

        # Create faces to form a watertight pentagonal trapezohedron
        faces = []

        # Top cap: connect pole 0 to upper ring
        for i in range(5):
            v1 = 2 + i  # current vertex in upper ring
            v2 = 2 + (i + 1) % 5  # next vertex in upper ring
            faces.append([0, v1, v2])  # Triangle from pole to ring (corrected winding)

        # Bottom cap: connect pole 1 to lower ring
        for i in range(5):
            v1 = 7 + i  # current vertex in lower ring
            v2 = 7 + (i + 1) % 5  # next vertex in lower ring
            faces.append([1, v2, v1])  # Triangle from pole to ring (corrected winding)

        # Middle band: connect upper ring to lower ring to form kite faces
        # Each kite face is formed by 4 vertices: 2 from upper ring, 2 from lower ring
        for i in range(5):
            upper1 = 2 + i  # current vertex in upper ring
            upper2 = 2 + (i + 1) % 5  # next vertex in upper ring
            lower1 = 7 + i  # corresponding vertex in lower ring (twisted)
            lower2 = 7 + (i + 1) % 5  # next vertex in lower ring

            # Each kite face needs to be split into 2 triangles
            # First triangle: upper1, lower1, upper2 (corrected winding)
            faces.append([upper1, lower1, upper2])
            # Second triangle: upper2, lower1, lower2 (corrected winding)
            faces.append([upper2, lower1, lower2])

        return vertices, np.array(faces)
