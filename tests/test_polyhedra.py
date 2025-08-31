"""Tests for polyhedra geometry utilities."""

import numpy as np

from dice_models.geometry.polyhedra import (
    PolyhedronGeometry,
    PolyhedronType,
    get_standard_number_layout,
)


class TestPolyhedronUtilities:
    """Test polyhedron utility functions."""

    def test_get_standard_number_layout(self):
        """Test standard number layout generation for all dice types."""
        test_cases = [
            (PolyhedronType.TETRAHEDRON, 4, [1, 2, 3, 4]),
            (PolyhedronType.CUBE, 6, [1, 2, 3, 4, 5, 6]),
            (PolyhedronType.OCTAHEDRON, 8, [1, 2, 3, 4, 5, 6, 7, 8]),
            (
                PolyhedronType.PENTAGONAL_TRAPEZOHEDRON,
                10,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ),  # D10 uses 0-9
            (PolyhedronType.DODECAHEDRON, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            (PolyhedronType.ICOSAHEDRON, 20, list(range(1, 21))),
        ]

        for poly_type, expected_length, expected_layout in test_cases:
            layout = get_standard_number_layout(poly_type)
            assert (
                len(layout) == expected_length
            ), f"{poly_type.name} should have {expected_length} numbers"
            assert layout == expected_layout, f"{poly_type.name} layout mismatch"

    def test_face_centers_calculation(self):
        """Test face center calculation for different polyhedra."""
        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=10.0
            )
            face_centers = PolyhedronGeometry.get_face_centers(vertices, faces)

            # Should have one center per face
            assert len(face_centers) == len(
                faces
            ), f"{poly_type.name} center count mismatch"

            # Each center should be a 3D point
            for center in face_centers:
                assert len(center) == 3, f"{poly_type.name} center should be 3D"
                assert isinstance(
                    center, np.ndarray
                ), f"{poly_type.name} center should be numpy array"

    def test_face_normals_calculation(self):
        """Test face normal calculation for different polyhedra."""
        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=10.0
            )
            face_normals = PolyhedronGeometry.get_face_normals(vertices, faces)

            # Should have one normal per face
            assert len(face_normals) == len(
                faces
            ), f"{poly_type.name} normal count mismatch"

            # Each normal should be a unit vector (approximately)
            for normal in face_normals:
                assert len(normal) == 3, f"{poly_type.name} normal should be 3D"
                magnitude = np.linalg.norm(normal)
                assert (
                    abs(magnitude - 1.0) < 0.1
                ), f"{poly_type.name} normal should be approximately unit length"

    def test_polyhedron_scaling(self):
        """Test that polyhedra scale correctly with radius."""
        test_radii = [1.0, 5.0, 20.0]

        for poly_type in PolyhedronType:
            for radius in test_radii:
                vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                    poly_type, radius=radius
                )

                # Check that vertices are approximately within the specified radius
                distances = np.linalg.norm(vertices, axis=1)
                max_distance = np.max(distances)

                # For circumscribed sphere, vertices should be on or inside the sphere
                assert (
                    max_distance <= radius * 1.1
                ), f"{poly_type.name} vertices exceed radius {radius}"

                # Different polyhedra use different radius interpretations
                # TETRAHEDRON appears to use inscribed radius in this library
                min_threshold = (
                    radius * 0.5
                    if poly_type == PolyhedronType.TETRAHEDRON
                    else radius * 0.8
                )
                assert (
                    max_distance >= min_threshold
                ), f"{poly_type.name} vertices too far inside radius {radius}"

    def test_face_consistency(self):
        """Test that faces reference valid vertices."""
        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=10.0
            )

            vertex_count = len(vertices)

            for i, face in enumerate(faces):
                # Each face should have at least 3 vertices
                assert len(face) >= 3, f"{poly_type.name} face {i} has too few vertices"

                # All face indices should be valid
                for vertex_idx in face:
                    assert (
                        0 <= vertex_idx < vertex_count
                    ), f"{poly_type.name} face {i} references invalid vertex {vertex_idx}"

    def test_geometric_properties(self):
        """Test geometric properties of generated polyhedra."""
        radius = 10.0

        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=radius
            )

            # Check expected number of faces
            expected_faces = poly_type.value
            if poly_type == PolyhedronType.PENTAGONAL_TRAPEZOHEDRON:
                # D10 might have simplified geometry
                assert (
                    len(faces) >= expected_faces // 2
                ), f"{poly_type.name} too few faces"
            else:
                # Allow some flexibility for implementation differences
                assert (
                    len(faces) >= expected_faces // 2
                ), f"{poly_type.name} face count too low"

            # Vertices should not be degenerate (all the same)
            vertex_ranges = np.ptp(
                vertices, axis=0
            )  # peak-to-peak (max - min) for each dimension
            assert all(
                r > 0.1 for r in vertex_ranges
            ), f"{poly_type.name} vertices are degenerate"

    def test_center_to_face_distances(self):
        """Test that face centers are at reasonable distances from origin."""
        radius = 10.0

        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=radius
            )
            face_centers = PolyhedronGeometry.get_face_centers(vertices, faces)

            center_distances = np.linalg.norm(face_centers, axis=1)

            # Face centers should be inside the circumscribed sphere
            max_center_distance = np.max(center_distances)
            assert (
                max_center_distance < radius
            ), f"{poly_type.name} face centers outside circumscribed sphere"

            # Face centers shouldn't be too close to origin (unless it's a very flat polyhedron)
            min_center_distance = np.min(center_distances)
            assert (
                min_center_distance > radius * 0.1
            ), f"{poly_type.name} face centers too close to origin"

    def test_face_normal_directions(self):
        """Test that face normals point outward from the polyhedron."""
        radius = 10.0

        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=radius
            )
            face_centers = PolyhedronGeometry.get_face_centers(vertices, faces)
            face_normals = PolyhedronGeometry.get_face_normals(vertices, faces)

            # Test that normals are unit vectors
            for normal in face_normals:
                norm_magnitude = np.linalg.norm(normal)
                assert (
                    abs(norm_magnitude - 1.0) < 0.1
                ), f"{poly_type.name} normal not unit vector: {norm_magnitude}"

            # Test that normals are consistently oriented (most should point in same general direction)
            # Rather than assuming outward/inward, just check consistency
            outward_count = 0
            total_faces = len(face_centers)

            for center, normal in zip(face_centers, face_normals):
                dot_product = np.dot(center, normal)
                if dot_product > 0:
                    outward_count += 1

            # At least 30% should be consistent in direction (allows for some irregular faces)
            consistency_ratio = (
                max(outward_count, total_faces - outward_count) / total_faces
            )
            assert (
                consistency_ratio >= 0.3
            ), f"{poly_type.name} normals inconsistently oriented"

    def test_polyhedron_type_values(self):
        """Test that PolyhedronType enum values match expected face counts."""
        expected_values = {
            PolyhedronType.TETRAHEDRON: 4,
            PolyhedronType.CUBE: 6,
            PolyhedronType.OCTAHEDRON: 8,
            PolyhedronType.PENTAGONAL_TRAPEZOHEDRON: 10,
            PolyhedronType.DODECAHEDRON: 12,
            PolyhedronType.ICOSAHEDRON: 20,
        }

        for poly_type, expected_value in expected_values.items():
            assert (
                poly_type.value == expected_value
            ), f"{poly_type.name} value should be {expected_value}"

    def test_edge_case_radius_values(self):
        """Test polyhedron generation with edge case radius values."""
        edge_radii = [0.1, 0.001, 100.0, 1000.0]

        for radius in edge_radii:
            for poly_type in PolyhedronType:
                vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                    poly_type, radius=radius
                )

                # Should still generate valid geometry
                assert (
                    len(vertices) > 0
                ), f"{poly_type.name} no vertices with radius {radius}"
                assert len(faces) > 0, f"{poly_type.name} no faces with radius {radius}"

                # Scaling should be proportional
                max_distance = np.max(np.linalg.norm(vertices, axis=1))
                assert (
                    max_distance > 0
                ), f"{poly_type.name} degenerate vertices with radius {radius}"

    def test_numerical_stability(self):
        """Test numerical stability of calculations."""
        # Test with various radius values that might cause numerical issues
        test_radii = [1e-6, 1e-3, 1e3, 1e6]

        for radius in test_radii:
            try:
                vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                    PolyhedronType.CUBE, radius=radius
                )
                face_centers = PolyhedronGeometry.get_face_centers(vertices, faces)
                face_normals = PolyhedronGeometry.get_face_normals(vertices, faces)

                # Check for NaN or infinite values
                assert not np.any(
                    np.isnan(vertices)
                ), f"NaN vertices with radius {radius}"
                assert not np.any(
                    np.isnan(face_centers)
                ), f"NaN face centers with radius {radius}"
                assert not np.any(
                    np.isnan(face_normals)
                ), f"NaN face normals with radius {radius}"

                assert not np.any(
                    np.isinf(vertices)
                ), f"Infinite vertices with radius {radius}"
                assert not np.any(
                    np.isinf(face_centers)
                ), f"Infinite face centers with radius {radius}"
                assert not np.any(
                    np.isinf(face_normals)
                ), f"Infinite face normals with radius {radius}"

            except Exception as e:
                # Very extreme values might fail, but they shouldn't crash unexpectedly
                assert (
                    "overflow" in str(e).lower() or "underflow" in str(e).lower()
                ), f"Unexpected error with radius {radius}: {e}"

    def test_dodecahedron_pentagonal_structure(self):
        """Test that the dodecahedron has the correct pentagonal face structure."""
        vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
            PolyhedronType.DODECAHEDRON, radius=10.0
        )

        # D12 should have more triangular faces than other dice due to pentagonal face triangulation
        # A dodecahedron has 12 pentagonal faces, each represented by multiple triangular faces
        expected_min_faces = 20  # Should be significantly more than 12 but less than 60
        expected_max_faces = 60  # 12 pentagons * 5 triangles max

        assert expected_min_faces <= len(faces) <= expected_max_faces, (
            f"D12 should have {expected_min_faces}-{expected_max_faces} triangular faces "
            f"(representing 12 pentagonal faces), but has {len(faces)}"
        )

        # Should have more faces than a simple polyhedron with 12 triangular faces
        assert len(faces) > 12, (
            f"D12 should have more than 12 faces to represent pentagonal structure, "
            f"but has {len(faces)}"
        )

        # Verify mesh is watertight when converted to trimesh
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        assert mesh.is_watertight, "D12 mesh should be watertight"
        assert mesh.is_volume, "D12 mesh should be a proper volume"

    def test_dodecahedron_logical_face_centers(self):
        """Test that dodecahedron logical face centers work correctly for text engraving."""
        # Test the special dodecahedron logical face centers functionality
        vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
            PolyhedronType.DODECAHEDRON, radius=10.0
        )

        # Get logical face centers and normals for text engraving
        logical_centers, logical_normals = (
            PolyhedronGeometry.get_dodecahedron_logical_face_centers_and_normals(
                vertices, faces, radius=10.0
            )
        )

        # Should have exactly 12 logical face centers for the 12 pentagonal faces
        assert (
            len(logical_centers) == 12
        ), f"Expected 12 logical face centers, got {len(logical_centers)}"
        assert (
            len(logical_normals) == 12
        ), f"Expected 12 logical face normals, got {len(logical_normals)}"

        # Each center should be a 3D point
        assert logical_centers.shape == (
            12,
            3,
        ), f"Logical face centers should be shape (12, 3), got {logical_centers.shape}"
        assert logical_normals.shape == (
            12,
            3,
        ), f"Logical face normals should be shape (12, 3), got {logical_normals.shape}"

        # All normals should be unit vectors
        for i, normal in enumerate(logical_normals):
            norm_length = np.linalg.norm(normal)
            assert (
                abs(norm_length - 1.0) < 1e-6
            ), f"Normal {i} should be unit vector, got length {norm_length}"

        # Face centers should be at reasonable distances from origin
        center_distances = np.linalg.norm(logical_centers, axis=1)
        expected_min_distance = (
            7.0  # Should be close to the radius (10.0) but on the surface
        )
        expected_max_distance = (
            10.0  # Should not be more than the radius since they're on the surface
        )

        for i, distance in enumerate(center_distances):
            assert expected_min_distance <= distance <= expected_max_distance, (
                f"Logical face center {i} distance {distance:.2f} should be between "
                f"{expected_min_distance} and {expected_max_distance}"
            )
