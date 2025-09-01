"""Compatibility layer for the refactored polyhedra system."""

from typing import List, Tuple

import numpy as np

from .factory import DiceFactory
from .polyhedra import PolyhedronType


class PolyhedronGeometry:
    """
    Compatibility class maintaining the original static method interface.

    This class provides backward compatibility by delegating to the new
    dice-specific classes while maintaining the same API.
    """

    @staticmethod
    def get_vertices_and_faces(polyhedron_type: PolyhedronType, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get vertices and faces for a given polyhedron type.

        Args:
            polyhedron_type: The type of polyhedron to generate
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        dice = DiceFactory.create_dice(polyhedron_type, radius)
        return dice.get_vertices_and_faces()

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

        # Calculate polyhedron center
        polyhedron_center = np.mean(vertices, axis=0)

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

            # Calculate face center
            face_center = np.mean([v0, v1, v2], axis=0)

            # Vector from polyhedron center to face center
            center_to_face = face_center - polyhedron_center

            # Check if normal points outward (same direction as center_to_face)
            # If dot product is negative, normal points inward, so flip it
            if np.dot(normal, center_to_face) < 0:
                normal = -normal

            normals.append(normal)
        return np.array(normals)

    @staticmethod
    def get_dodecahedron_logical_face_centers_and_normals(
        vertices: np.ndarray, faces: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get face centers and normals for the 12 logical pentagonal faces of a dodecahedron.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)
            radius: Radius of the dodecahedron

        Returns:
            Tuple of (face_centers, face_normals) for the 12 logical pentagonal faces
        """
        from .types.d12 import D12

        d12 = D12(radius=radius)
        return d12.get_logical_face_centers_and_normals(vertices, faces)

    @staticmethod
    def get_dodecahedron_logical_face_vertices(
        vertices: np.ndarray, faces: np.ndarray, radius: float
    ) -> List[np.ndarray]:
        """
        Get the vertices for the 12 logical pentagonal faces of a dodecahedron.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)
            radius: Radius of the dodecahedron

        Returns:
            List of 12 arrays, each containing 5 vertices of a pentagonal face
        """
        from .types.d12 import D12

        d12 = D12(radius=radius)
        return d12.get_logical_face_vertices(vertices, faces)
