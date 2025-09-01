"""D12 (Dodecahedron) dice geometry implementation."""

from typing import List, Tuple

import numpy as np

from ..base.polyhedron import BasePolyhedron


class D12(BasePolyhedron):
    """
    Twelve-sided dice (dodecahedron) implementation.
    
    The D12 is a regular dodecahedron with 12 pentagonal faces (triangulated).
    """

    @property
    def sides(self) -> int:
        """Return the number of sides (12) for a dodecahedron."""
        return 12

    @property
    def name(self) -> str:
        """Return the name of this polyhedron type."""
        return "DODECAHEDRON"

    def get_standard_number_layout(self) -> List[int]:
        """
        Get the standard number layout for a D12.
        
        Returns:
            List of numbers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in face order
        """
        return list(range(1, 13))

    def _generate_vertices_and_faces(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dodecahedron vertices and faces with pentagonal faces triangulated.
        
        Args:
            radius: The radius of the circumscribed sphere
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Create a dodecahedron using the dual of an icosahedron
        # This ensures we get exactly 12 pentagonal faces (triangulated) and 20 vertices

        # Start with icosahedron face centers as dodecahedron vertices
        import trimesh
        from scipy.spatial import ConvexHull

        icosahedron = trimesh.creation.icosahedron(radius=1.0)

        # Create dodecahedron vertices from icosahedron face centers
        vertices = []
        for face in icosahedron.faces:
            # Calculate face center
            center = np.mean(icosahedron.vertices[face], axis=0)
            # Project to sphere surface and scale to desired radius
            center = center / np.linalg.norm(center) * radius
            vertices.append(center)

        vertices = np.array(vertices)

        # Use convex hull to create triangulated surface
        # This automatically creates a watertight triangulated mesh
        hull = ConvexHull(vertices)
        faces = hull.simplices

        # Fix face orientation - convex hull may have inward-facing normals
        # Ensure faces are oriented outward by checking against centroid
        centroid = np.mean(vertices, axis=0)
        corrected_faces = []

        for face in faces:
            # Calculate face normal
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)

            # Calculate face center
            face_center = np.mean([v0, v1, v2], axis=0)

            # Vector from centroid to face center
            to_face = face_center - centroid

            # If normal points inward (dot product negative), flip the face
            if np.dot(normal, to_face) < 0:
                corrected_faces.append([face[0], face[2], face[1]])  # Flip winding order
            else:
                corrected_faces.append(face)

        return vertices, np.array(corrected_faces)

    def get_logical_face_centers_and_normals(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get face centers and normals for the 12 logical pentagonal faces of a dodecahedron.

        The dodecahedron is triangulated into 36 triangular faces (12 pentagons × 3 triangles each),
        but for text engraving we need the centers and normals of the 12 logical pentagonal faces.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)

        Returns:
            Tuple of (face_centers, face_normals) for the 12 logical pentagonal faces
        """
        import trimesh

        # Create the actual dodecahedron mesh to work with
        dodecahedron_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Get the icosahedron vertices which correspond to dodecahedron face centers
        icosahedron = trimesh.creation.icosahedron(radius=1.0)

        logical_face_centers = []
        logical_face_normals = []

        # Use icosahedron vertices as the directions for dodecahedron face centers
        for vertex in icosahedron.vertices:
            # Get the direction from origin to this vertex
            direction = vertex / np.linalg.norm(vertex)

            # Project this direction onto the dodecahedron surface
            # We'll cast a ray from inside the mesh outward to find the surface
            ray_origin = direction * (self.radius * 0.1)  # Start from inside
            ray_direction = direction

            # Find intersection with dodecahedron surface
            locations, ray_indices, triangle_indices = (
                dodecahedron_mesh.ray.intersects_location(
                    ray_origins=[ray_origin], ray_directions=[ray_direction]
                )
            )

            if len(locations) > 0:
                # Use the first intersection (should be the surface)
                face_center = locations[0]
                face_normal = direction  # Normal points outward from center

                logical_face_centers.append(face_center)
                logical_face_normals.append(face_normal)

        # If we didn't get exactly 12 faces, fall back to a simpler approach
        if len(logical_face_centers) != 12:
            logical_face_centers = []
            logical_face_normals = []

            # Use icosahedron vertices projected to the correct radius
            for vertex in icosahedron.vertices:
                # Project to surface of dodecahedron
                direction = vertex / np.linalg.norm(vertex)
                face_center = direction * self.radius
                face_normal = direction

                logical_face_centers.append(face_center)
                logical_face_normals.append(face_normal)

        return np.array(logical_face_centers), np.array(logical_face_normals)

    def get_logical_face_vertices(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> List[np.ndarray]:
        """
        Get the vertices for the 12 logical pentagonal faces of a dodecahedron.

        The dodecahedron is triangulated, but we need to reconstruct the original
        pentagonal faces for edge alignment.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)

        Returns:
            List of 12 arrays, each containing 5 vertices of a pentagonal face
        """
        import trimesh
        logical_face_centers, _ = self.get_logical_face_centers_and_normals(vertices, faces)

        logical_face_vertices = []

        for face_center in logical_face_centers:
            # Find vertices that are part of this logical face
            # For a dodecahedron, each pentagonal face has 5 vertices
            # We'll find the 5 vertices closest to this face center

            # Calculate distances from all vertices to this face center
            distances = np.linalg.norm(vertices - face_center, axis=1)

            # Get the 5 closest vertices (these form the pentagon)
            closest_indices = np.argsort(distances)[:5]
            face_verts = vertices[closest_indices]

            # Sort vertices in pentagonal order around the face center
            # Project vertices to a 2D plane centered at face_center
            face_normal = face_center / np.linalg.norm(face_center)  # Points outward from origin

            # Create a coordinate system for the face plane
            # Use the first vertex as reference for the X direction
            to_first_vertex = face_verts[0] - face_center
            to_first_vertex = (
                to_first_vertex - np.dot(to_first_vertex, face_normal) * face_normal
            )
            x_axis = to_first_vertex / np.linalg.norm(to_first_vertex)
            y_axis = np.cross(face_normal, x_axis)

            # Project all vertices to 2D coordinates in this plane
            angles = []
            for vertex in face_verts:
                to_vertex = vertex - face_center
                to_vertex = to_vertex - np.dot(to_vertex, face_normal) * face_normal
                x_coord = np.dot(to_vertex, x_axis)
                y_coord = np.dot(to_vertex, y_axis)
                angle = np.arctan2(y_coord, x_coord)
                angles.append(angle)

            # Sort vertices by angle to get proper pentagonal order
            sorted_indices = np.argsort(angles)
            ordered_face_verts = face_verts[sorted_indices]

            logical_face_vertices.append(ordered_face_verts)

        return logical_face_vertices
