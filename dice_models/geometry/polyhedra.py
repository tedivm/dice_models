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
        """
        Generate pentagonal trapezohedron (D10) vertices and faces.

        Creates a proper pentagonal trapezohedron with 10 kite-shaped faces that form
        a watertight mesh. The vertex positions are carefully tuned to make the
        triangle pairs that form each kite face as coplanar as possible.
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
        # The key is to ensure every edge is shared by exactly 2 faces
        faces = []

        # Top cap: connect top pole to upper ring (5 triangular faces)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([0, 2 + i, 2 + next_i])  # top pole to upper ring

        # Bottom cap: connect bottom pole to lower ring (5 triangular faces)
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append(
                [1, 7 + next_i, 7 + i]
            )  # bottom pole to lower ring (reversed winding)

        # Side faces: connect upper ring to lower ring (10 triangular faces)
        # These form the "kite" sides of the trapezohedron
        for i in range(5):
            next_i = (i + 1) % 5

            # Each kite side is made of 2 triangles connecting upper and lower rings
            upper_curr = 2 + i
            upper_next = 2 + next_i
            lower_curr = 7 + i
            lower_next = 7 + next_i

            # First triangle of the kite side
            faces.append([upper_curr, lower_curr, upper_next])
            # Second triangle of the kite side
            faces.append([lower_curr, lower_next, upper_next])

        faces = np.array(faces)
        return vertices, faces

    @staticmethod
    def _dodecahedron(radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dodecahedron vertices and faces with pentagonal faces triangulated."""
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
                corrected_faces.append(
                    [face[0], face[2], face[1]]
                )  # Flip winding order
            else:
                corrected_faces.append(face)

        return vertices, np.array(corrected_faces)

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

    @staticmethod
    def get_dodecahedron_logical_face_centers_and_normals(
        vertices: np.ndarray, faces: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get face centers and normals for the 12 logical pentagonal faces of a dodecahedron.

        The dodecahedron is triangulated into 36 triangular faces (12 pentagons × 3 triangles each),
        but for text engraving we need the centers and normals of the 12 logical pentagonal faces.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)
            radius: Radius of the dodecahedron

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
            ray_origin = direction * (radius * 0.1)  # Start from inside
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
                face_center = direction * radius
                face_normal = direction

                logical_face_centers.append(face_center)
                logical_face_normals.append(face_normal)

        return np.array(logical_face_centers), np.array(logical_face_normals)

    @staticmethod
    def get_dodecahedron_logical_face_vertices(
        vertices: np.ndarray, faces: np.ndarray, radius: float
    ) -> List[np.ndarray]:
        """
        Get the vertices for the 12 logical pentagonal faces of a dodecahedron.

        The dodecahedron is triangulated, but we need to reconstruct the original
        pentagonal faces for edge alignment.

        Args:
            vertices: Array of dodecahedron vertices (20 vertices)
            faces: Array of triangulated faces (36 triangular faces)
            radius: Radius of the dodecahedron

        Returns:
            List of 12 arrays, each containing 5 vertices of a pentagonal face
        """
        import trimesh

        # Create the actual dodecahedron mesh
        dodecahedron_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Get logical face centers first
        logical_face_centers, _ = (
            PolyhedronGeometry.get_dodecahedron_logical_face_centers_and_normals(
                vertices, faces, radius
            )
        )

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
            face_normal = face_center / np.linalg.norm(
                face_center
            )  # Points outward from origin

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
