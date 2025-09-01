"""Base polyhedron class providing common functionality for all dice types."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BasePolyhedron(ABC):
    """
    Abstract base class for all polyhedron dice types.

    This class defines the interface that all dice implementations must follow
    and provides common functionality shared across all dice types.
    """

    def __init__(self, radius: float = 1.0):
        """
        Initialize the polyhedron with a given radius.

        Args:
            radius: The radius of the circumscribed sphere
        """
        self.radius = radius

    @property
    @abstractmethod
    def sides(self) -> int:
        """Return the number of sides/faces of this polyhedron."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        pass

    @abstractmethod
    def _generate_vertices_and_faces(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the vertices and faces for this polyhedron type.

        Args:
            radius: The radius of the circumscribed sphere

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        pass

    @abstractmethod
    def get_standard_number_layout(self) -> List[int]:
        """
        Get the standard number layout for this dice type.

        Returns:
            List of numbers in face order
        """
        pass

    def get_vertices_and_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get vertices and faces for this polyhedron.

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        return self._generate_vertices_and_faces(self.radius)

    def get_face_centers(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
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

    def get_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
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

    def validate_geometry(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        """
        Validate that the generated geometry is correct.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face vertex indices

        Returns:
            True if geometry is valid, False otherwise
        """
        # Check that we have the expected number of faces
        if len(faces) < self.sides:
            return False

        # Check that all vertices are roughly on the circumsphere
        distances = np.linalg.norm(vertices, axis=1)
        expected_distance = self.radius
        tolerance = expected_distance * 0.1  # 10% tolerance

        return all(abs(dist - expected_distance) < tolerance for dist in distances)
