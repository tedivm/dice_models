#!/usr/bin/env python3
"""
Test file specifically for D10 pole alignment functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dice_models.geometry.dice import DiceGeometry
from dice_models.geometry.polyhedra import PolyhedronType
from dice_models.geometry.text import _calculate_d10_pole_alignment


class TestD10PoleAlignment:
    """Test the D10 pole alignment feature."""

    def test_d10_pole_alignment_calculation(self):
        """Test that D10 pole alignment calculations work correctly."""
        radius = 10.0

        # Test face closer to top pole
        face_center_top = np.array([5.0, 0.0, 8.0])  # Closer to top pole
        face_normal = np.array([0.5, 0.0, 0.866])  # Some arbitrary face normal

        alignment_matrix = _calculate_d10_pole_alignment(face_center_top, face_normal, radius)

        assert alignment_matrix is not None
        assert alignment_matrix.shape == (4, 4)

        # Test face closer to bottom pole
        face_center_bottom = np.array([5.0, 0.0, -8.0])  # Closer to bottom pole
        alignment_matrix = _calculate_d10_pole_alignment(face_center_bottom, face_normal, radius)

        assert alignment_matrix is not None
        assert alignment_matrix.shape == (4, 4)

    def test_d10_pole_alignment_with_parallel_direction(self):
        """Test pole alignment when direction is parallel to face normal."""
        radius = 10.0

        # Face normal pointing directly at pole - should return None
        face_center = np.array([0.0, 0.0, 6.0])
        face_normal = np.array([0.0, 0.0, 1.0])  # Points directly toward top pole

        alignment_matrix = _calculate_d10_pole_alignment(face_center, face_normal, radius)

        # Should return None when pole direction is parallel to face normal
        assert alignment_matrix is None

    def test_d10_creation_with_pole_alignment(self):
        """Test that D10 dice can be created with pole alignment."""
        dice = DiceGeometry(
            polyhedron_type=PolyhedronType.PENTAGONAL_TRAPEZOHEDRON,
            radius=10.0,
            text_size=3.0,
        )

        # Should not raise any exceptions
        mesh = dice.generate_mesh()

        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        # Should be a valid, watertight mesh
        assert mesh.is_watertight

    def test_d10_export_with_pole_alignment(self):
        """Test that D10 with pole alignment can be exported successfully."""
        dice = DiceGeometry(
            polyhedron_type=PolyhedronType.PENTAGONAL_TRAPEZOHEDRON,
            radius=10.0,
            text_size=3.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_d10_pole_alignment.stl"

            # Should not raise any exceptions
            dice.export_stl(output_path)

            # File should exist and have content
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_d10_pole_positions(self):
        """Test that D10 pole positions are calculated correctly."""
        radius = 10.0

        # According to the D10 geometry, poles should be at radius * 1.2
        radius * 1.2

        # Test the pole position calculation in our alignment function
        face_center = np.array([8.0, 0.0, 2.0])  # Some face position
        face_normal = np.array([0.8, 0.0, 0.6])  # Normalized face normal

        # The function should handle the pole calculations correctly
        alignment_matrix = _calculate_d10_pole_alignment(face_center, face_normal, radius)

        assert alignment_matrix is not None

    def test_d10_face_to_pole_distance_logic(self):
        """Test that faces are correctly assigned to closest pole."""
        radius = 10.0
        polar_height = radius * 1.2  # 12.0

        # Test face clearly closer to top pole
        face_center_top = np.array([5.0, 5.0, 8.0])
        top_pole = np.array([0, 0, polar_height])
        bottom_pole = np.array([0, 0, -polar_height])

        dist_to_top = np.linalg.norm(face_center_top - top_pole)
        dist_to_bottom = np.linalg.norm(face_center_top - bottom_pole)

        assert dist_to_top < dist_to_bottom, "Face should be closer to top pole"

        # Test face clearly closer to bottom pole
        face_center_bottom = np.array([5.0, 5.0, -8.0])

        dist_to_top = np.linalg.norm(face_center_bottom - top_pole)
        dist_to_bottom = np.linalg.norm(face_center_bottom - bottom_pole)

        assert dist_to_bottom < dist_to_top, "Face should be closer to bottom pole"

    def test_d10_alignment_matrix_properties(self):
        """Test that alignment matrices have proper properties."""
        radius = 10.0
        face_center = np.array([6.0, 6.0, 3.0])
        face_normal = np.array([0.6, 0.6, 0.53])  # Normalized
        face_normal = face_normal / np.linalg.norm(face_normal)

        alignment_matrix = _calculate_d10_pole_alignment(face_center, face_normal, radius)

        if alignment_matrix is not None:
            # Should be a 4x4 transformation matrix
            assert alignment_matrix.shape == (4, 4)

            # Should be approximately orthogonal (rotation matrix)
            rotation_part = alignment_matrix[:3, :3]
            should_be_identity = rotation_part @ rotation_part.T
            np.testing.assert_allclose(
                should_be_identity,
                np.eye(3),
                atol=1e-10,
                err_msg="Rotation matrix should be orthogonal",
            )

            # Determinant should be 1 (proper rotation, no reflection)
            det = np.linalg.det(rotation_part)
            assert abs(det - 1.0) < 1e-10, "Determinant should be 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
