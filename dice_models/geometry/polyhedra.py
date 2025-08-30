"""Polyhedra definitions for dice geometry."""

import math
from enum import Enum
from typing import List, Tuple

import numpy as np


class PolyhedronType(Enum):
    """Supported polyhedron types for dice."""

    TETRAHEDRON = 4  # D4
    CUBE = 6  # D6
    OCTAHEDRON = 8  # D8
    PENTAGONAL_TRAPEZOHEDRON = 10  # D10
    DODECAHEDRON = 12  # D12
    ICOSAHEDRON = 20  # D20


class PolyhedronGeometry:
    """Base class for polyhedron geometry calculations."""

    @staticmethod
    def get_vertices_and_faces(
        polyhedron_type: PolyhedronType, radius: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get vertices and faces for a given polyhedron type.

        Args:
            polyhedron_type: The type of polyhedron to generate
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        if polyhedron_type == PolyhedronType.TETRAHEDRON:
            return PolyhedronGeometry._tetrahedron(radius)
        elif polyhedron_type == PolyhedronType.CUBE:
            return PolyhedronGeometry._cube(radius)
        elif polyhedron_type == PolyhedronType.OCTAHEDRON:
            return PolyhedronGeometry._octahedron(radius)
        elif polyhedron_type == PolyhedronType.PENTAGONAL_TRAPEZOHEDRON:
            return PolyhedronGeometry._pentagonal_trapezohedron(radius)
        elif polyhedron_type == PolyhedronType.DODECAHEDRON:
            return PolyhedronGeometry._dodecahedron(radius)
        elif polyhedron_type == PolyhedronType.ICOSAHEDRON:
            return PolyhedronGeometry._icosahedron(radius)
        else:
            raise ValueError(f"Unsupported polyhedron type: {polyhedron_type}")

    @staticmethod
    def get_face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Calculate the center points of all faces.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face vertex indices

        Returns:
            Array of face center coordinates
        """
        face_centers = []
        for face in faces:
            face_vertices = vertices[face]
            center = np.mean(face_vertices, axis=0)
            face_centers.append(center)
        return np.array(face_centers)

    @staticmethod
    def get_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Calculate outward-pointing normal vectors for all faces.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face vertex indices

        Returns:
            Array of normalized face normal vectors
        """
        normals = []
        for face in faces:
            # Get three vertices of the face
            v0, v1, v2 = vertices[face[:3]]
            # Calculate two edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            # Cross product gives normal vector
            normal = np.cross(edge1, edge2)
            # Normalize
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return np.array(normals)

    @staticmethod
    def _tetrahedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tetrahedron vertices and faces."""
        # Regular tetrahedron vertices - corrected for proper geometry
        h = radius * math.sqrt(2 / 3)  # Height from base to top
        a = radius * math.sqrt(8 / 9)  # Side length of base triangle

        vertices = np.array(
            [
                [a / 2, -a / (2 * math.sqrt(3)), -h / 3],  # Vertex 0
                [-a / 2, -a / (2 * math.sqrt(3)), -h / 3],  # Vertex 1
                [0, a / math.sqrt(3), -h / 3],  # Vertex 2
                [0, 0, 2 * h / 3],  # Vertex 3 (top)
            ]
        )

        faces = np.array(
            [
                [0, 2, 1],  # Bottom face (clockwise from outside)
                [0, 1, 3],  # Side face 1
                [1, 2, 3],  # Side face 2
                [2, 0, 3],  # Side face 3
            ]
        )

        return vertices, faces

    @staticmethod
    def _cube(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate cube vertices and faces."""
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

    @staticmethod
    def _octahedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate octahedron vertices and faces."""
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

    @staticmethod
    def _pentagonal_trapezohedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pentagonal trapezohedron (D10) vertices and faces."""
        # Simplified D10 using a double pyramid over a pentagon
        height = radius * 0.9
        base_radius = radius * 0.8

        vertices = []

        # Top point
        vertices.append([0, 0, height])

        # Pentagon vertices in the middle
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = base_radius * math.cos(angle)
            y = base_radius * math.sin(angle)
            vertices.append([x, y, 0])

        # Bottom point
        vertices.append([0, 0, -height])

        vertices = np.array(vertices)

        # Create 10 triangular faces
        faces = []

        # Upper faces (5 triangular faces from top point to pentagon)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([0, 1 + i, 1 + next_i])

        # Lower faces (5 triangular faces from pentagon to bottom point)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([1 + i, 6, 1 + next_i])  # Note: vertex 6 is bottom point

        faces = np.array(faces)
        return vertices, faces

    @staticmethod
    def _dodecahedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dodecahedron vertices and faces."""
        # Very simplified dodecahedron - using a truncated icosahedron approach
        # This creates a sphere-like shape with 12 faces

        vertices = []
        faces = []

        # Create vertices around a sphere in a roughly dodecahedral pattern
        # 3 rings of vertices plus top and bottom points

        # Top vertex
        vertices.append([0, 0, radius])

        # Upper ring (5 vertices)
        upper_radius = radius * 0.8
        upper_height = radius * 0.4
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = upper_radius * math.cos(angle)
            y = upper_radius * math.sin(angle)
            vertices.append([x, y, upper_height])

        # Lower ring (5 vertices, rotated)
        lower_radius = radius * 0.8
        lower_height = -radius * 0.4
        for i in range(5):
            angle = 2 * math.pi * i / 5 + math.pi / 5  # Rotated by 36 degrees
            x = lower_radius * math.cos(angle)
            y = lower_radius * math.sin(angle)
            vertices.append([x, y, lower_height])

        # Bottom vertex
        vertices.append([0, 0, -radius])

        vertices = np.array(vertices)

        # Create faces - 12 triangular faces
        # Top cap (5 faces)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([0, 1 + i, 1 + next_i])

        # Middle band (5 faces)
        for i in range(5):
            next_i = (i + 1) % 5
            # Connect upper ring to lower ring
            faces.append([1 + i, 6 + i, 1 + next_i])
            faces.append([1 + next_i, 6 + i, 6 + next_i])

        # Bottom cap would be 2 more faces, but we'll use just one
        # Simplified bottom (1 face connecting to center)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([6 + i, 11, 6 + next_i])

        return vertices, np.array(faces)

    @staticmethod
    def _icosahedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate icosahedron vertices and faces."""
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


def get_standard_number_layout(polyhedron_type: PolyhedronType) -> List[int]:
    """
    Get the standard number layout for a dice type.

    Args:
        polyhedron_type: The type of polyhedron

    Returns:
        List of numbers in face order
    """
    layouts = {
        PolyhedronType.TETRAHEDRON: [1, 2, 3, 4],
        PolyhedronType.CUBE: [1, 2, 3, 4, 5, 6],
        PolyhedronType.OCTAHEDRON: [1, 2, 3, 4, 5, 6, 7, 8],
        PolyhedronType.PENTAGONAL_TRAPEZOHEDRON: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        PolyhedronType.DODECAHEDRON: list(range(1, 13)),
        PolyhedronType.ICOSAHEDRON: list(range(1, 21)),
    }
    return layouts.get(polyhedron_type, list(range(1, polyhedron_type.value + 1)))
