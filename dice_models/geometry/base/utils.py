"""Utility functions for polyhedron geometry operations."""

import numpy as np


class PolyhedronUtils:
    """Utility class containing static methods for polyhedron operations."""

    @staticmethod
    def calculate_face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
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
    def calculate_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
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
    def ensure_outward_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Ensure face normals point outward from the polyhedron center.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face vertex indices

        Returns:
            Array of faces with corrected winding order
        """
        centroid = np.mean(vertices, axis=0)
        corrected_faces = []

        for face in faces:
            # Calculate face normal
            v0, v1, v2 = vertices[face[:3]]
            normal = np.cross(v1 - v0, v2 - v0)

            # Calculate face center
            face_center = np.mean([v0, v1, v2], axis=0)

            # Vector from centroid to face center
            to_face = face_center - centroid

            # If normal points inward (dot product negative), flip the face
            if np.dot(normal, to_face) < 0:
                corrected_faces.append(
                    [face[0], face[2], face[1]]
                )  # Flip winding order
            else:
                corrected_faces.append(face)

        return np.array(corrected_faces)

    @staticmethod
    def validate_mesh_integrity(
        vertices: np.ndarray, faces: np.ndarray, radius: float
    ) -> bool:
        """
        Validate that a mesh has the expected properties.

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face vertex indices
            radius: Expected radius of circumscribed sphere

        Returns:
            True if mesh is valid, False otherwise
        """
        # Check that all vertices are roughly on the circumsphere
        distances = np.linalg.norm(vertices, axis=1)
        tolerance = radius * 0.1  # 10% tolerance

        vertices_on_sphere = all(abs(dist - radius) < tolerance for dist in distances)

        # Check that we have reasonable face connectivity
        has_faces = len(faces) > 0

        return vertices_on_sphere and has_faces
