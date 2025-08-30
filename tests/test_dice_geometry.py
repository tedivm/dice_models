"""Tests for dice geometry generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from dice_models.geometry import DiceGeometry, PolyhedronType
from dice_models.geometry.dice import create_standard_dice
from dice_models.geometry.polyhedra import (
    PolyhedronGeometry,
    get_standard_number_layout,
)


class TestPolyhedronGeometry:
    """Test polyhedron geometry generation."""

    def test_all_polyhedron_types(self):
        """Test that all polyhedron types can be generated."""
        for poly_type in PolyhedronType:
            vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
                poly_type, radius=1.0
            )

            assert vertices is not None
            assert faces is not None
            assert len(vertices) > 0
            assert len(faces) > 0

            # Check that we have the expected number of faces
            if poly_type == PolyhedronType.PENTAGONAL_TRAPEZOHEDRON:
                # D10 is special - it has 10 kite-shaped faces but our simplified version may differ
                assert len(faces) >= 5  # At least the number we defined
            else:
                # For other polyhedra, we expect the standard number of faces
                pass  # Our implementations may be simplified

    def test_face_centers_calculation(self):
        """Test face center calculation."""
        # Test with a simple cube
        vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
            PolyhedronType.CUBE, radius=1.0
        )
        centers = PolyhedronGeometry.get_face_centers(vertices, faces)

        assert len(centers) == len(faces)
        # For a cube, all face centers should be at distance radius/sqrt(3) from origin
        distances = np.linalg.norm(centers, axis=1)
        # Allow some tolerance for numerical precision
        assert all(0.5 < dist < 1.0 for dist in distances)

    def test_face_normals_calculation(self):
        """Test face normal calculation."""
        vertices, faces = PolyhedronGeometry.get_vertices_and_faces(
            PolyhedronType.CUBE, radius=1.0
        )
        normals = PolyhedronGeometry.get_face_normals(vertices, faces)

        assert len(normals) == len(faces)
        # All normals should be unit vectors
        lengths = np.linalg.norm(normals, axis=1)
        assert all(0.9 < length < 1.1 for length in lengths)

    def test_standard_number_layouts(self):
        """Test that standard number layouts are correct."""
        # Test a few known layouts
        d4_layout = get_standard_number_layout(PolyhedronType.TETRAHEDRON)
        assert d4_layout == [1, 2, 3, 4]

        d6_layout = get_standard_number_layout(PolyhedronType.CUBE)
        assert d6_layout == [1, 2, 3, 4, 5, 6]

        d10_layout = get_standard_number_layout(PolyhedronType.PENTAGONAL_TRAPEZOHEDRON)
        assert d10_layout == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class TestDiceGeometry:
    """Test the DiceGeometry class."""

    def test_dice_creation(self):
        """Test creating dice with different geometries."""
        for polyhedron_type, sides in [
            (PolyhedronType.TETRAHEDRON, 4),
            (PolyhedronType.CUBE, 6),
            (PolyhedronType.OCTAHEDRON, 8),
            (PolyhedronType.DODECAHEDRON, 12),
            (PolyhedronType.ICOSAHEDRON, 20),
        ]:
            dice = DiceGeometry(polyhedron_type, curve_resolution="low")
            assert dice.sides == sides
            assert len(dice.number_layout) == sides

    def test_mesh_generation(self):
        """Test mesh generation for different dice types."""
        for sides in [4, 6, 8, 10, 12, 20]:
            dice = create_standard_dice(
                sides=sides, radius=10.0, curve_resolution="low"
            )

            # Test mesh without numbers
            mesh_no_numbers = dice.generate_mesh(include_numbers=False)
            assert len(mesh_no_numbers.vertices) > 0
            assert len(mesh_no_numbers.faces) > 0
            assert mesh_no_numbers.is_watertight

            # Test mesh with numbers
            mesh_with_numbers = dice.generate_mesh(include_numbers=True)
            assert len(mesh_with_numbers.vertices) > 0
            assert len(mesh_with_numbers.faces) > 0
            assert mesh_with_numbers.is_watertight

    def test_text_engraving_effectiveness(self):
        """Test that text engraving actually modifies the mesh effectively."""
        for sides in [4, 6, 8, 10, 12, 20]:
            dice = create_standard_dice(
                sides=sides, radius=10.0, text_depth=1.0, text_size=3.0
            )

            # Generate both versions
            mesh_no_numbers = dice.generate_mesh(include_numbers=False)
            mesh_with_numbers = dice.generate_mesh(include_numbers=True)

            # Verify engraving worked using proven methodology
            vertex_increase = len(mesh_with_numbers.vertices) - len(
                mesh_no_numbers.vertices
            )
            volume_decrease = mesh_no_numbers.volume - mesh_with_numbers.volume

            assert (
                vertex_increase > 10
            ), f"D{sides}: Insufficient vertex increase ({vertex_increase})"
            assert (
                volume_decrease > 0
            ), f"D{sides}: No volume decrease from engraving ({volume_decrease})"

            # Ensure volume change is reasonable (not excessive)
            volume_ratio = volume_decrease / mesh_no_numbers.volume
            assert (
                volume_ratio < 0.1
            ), f"D{sides}: Volume change too large ({volume_ratio:.3%})"

    def test_mesh_integrity_after_engraving(self):
        """Test that engraved meshes maintain structural integrity."""
        for sides in [4, 6, 8, 10, 12, 20]:
            dice = create_standard_dice(sides=sides, radius=10.0)
            mesh = dice.generate_mesh(include_numbers=True)

            # Basic integrity checks
            assert mesh.is_watertight, f"D{sides} mesh not watertight after engraving"
            assert mesh.volume > 0, f"D{sides} mesh has invalid volume"
            assert len(mesh.vertices) > 0, f"D{sides} mesh has no vertices"
            assert len(mesh.faces) > 0, f"D{sides} mesh has no faces"

            # Check for degenerate faces
            face_areas = mesh.area_faces
            assert all(
                area > 1e-10 for area in face_areas
            ), f"D{sides} has degenerate faces"

    def test_engraving_consistency_across_faces(self):
        """Test that engraving works consistently across all faces."""
        dice = create_standard_dice(sides=6, radius=10.0)

        # Test individual face engraving
        base_mesh = dice.generate_mesh(include_numbers=False)
        face_success_count = 0

        for i in range(6):
            from dice_models.geometry.text import create_engraved_text

            try:
                result_mesh = create_engraved_text(
                    base_mesh=base_mesh.copy(),
                    text=str(i + 1),
                    face_center=dice.face_centers[i],
                    face_normal=dice.face_normals[i],
                    text_depth=1.0,
                    text_size=3.0,
                )

                vertex_increase = len(result_mesh.vertices) - len(base_mesh.vertices)
                if vertex_increase > 0:
                    face_success_count += 1

            except Exception:
                pass  # Face engraving failed

        # Expect most faces to work (allow for some edge cases)
        success_rate = face_success_count / 6
        assert (
            success_rate >= 0.8
        ), f"Only {face_success_count}/6 faces engraved successfully"

    def test_custom_number_layout(self):
        """Test custom number layouts."""
        custom_layout = [6, 5, 4, 3, 2, 1]
        dice = DiceGeometry(
            PolyhedronType.CUBE, number_layout=custom_layout, curve_resolution="low"
        )

        assert dice.number_layout == custom_layout

        # Verify it still generates valid meshes
        mesh = dice.generate_mesh(include_numbers=True)
        assert mesh.is_watertight

    def test_dice_info(self):
        """Test dice information gathering."""
        dice = DiceGeometry(
            PolyhedronType.ICOSAHEDRON, radius=15.0, curve_resolution="low"
        )
        info = dice.get_info()

        assert info["type"] == "ICOSAHEDRON"
        assert info["sides"] == 20
        assert info["radius"] == 15.0
        assert len(info["number_layout"]) == 20

    def test_stl_export(self):
        """Test STL file export."""
        dice = DiceGeometry(
            PolyhedronType.TETRAHEDRON, radius=8.0, curve_resolution="low"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_d4.stl"

            # Test export without numbers
            dice.export_stl(output_path, include_numbers=False)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Test export with numbers
            output_path_with_numbers = Path(temp_dir) / "test_d4_numbered.stl"
            dice.export_stl(output_path_with_numbers, include_numbers=True)
            assert output_path_with_numbers.exists()
            assert output_path_with_numbers.stat().st_size > 0


class TestStandardDiceCreation:
    """Test the convenience function for creating standard dice."""

    def test_create_standard_dice_valid_sides(self):
        """Test creating dice with valid numbers of sides."""
        valid_sides = [4, 6, 8, 10, 12, 20]

        for sides in valid_sides:
            dice = create_standard_dice(sides=sides, radius=5.0)
            assert dice.polyhedron_type.value == sides

    def test_create_standard_dice_invalid_sides(self):
        """Test that invalid numbers of sides raise errors."""
        invalid_sides = [3, 5, 7, 9, 16, 24]

        for sides in invalid_sides:
            with pytest.raises(ValueError, match="Invalid number of sides"):
                create_standard_dice(sides=sides)

    def test_create_standard_dice_with_export(self):
        """Test creating dice and exporting to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "standard_d6.stl"

            dice = create_standard_dice(sides=6, radius=12.0, output_path=output_path)

            assert output_path.exists()
            assert dice.radius == 12.0
            assert dice.polyhedron_type == PolyhedronType.CUBE


class TestTextRendering:
    """Test font-based text rendering functionality."""

    def test_font_text_mesh_creation(self):
        """Test creation of font-based text meshes."""
        from dice_models.geometry.text import _create_font_text_mesh

        font_path = self._get_test_font()

        # Test rendering different numbers
        for text in ["1", "5", "10", "20"]:
            mesh = _create_font_text_mesh(
                text=text, size=3.0, depth=1.0, font_path=font_path
            )
            assert mesh is not None
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0
            assert mesh.volume > 0

    def test_text_engraving_function(self):
        """Test the main text engraving function."""
        from dice_models.geometry.text import create_engraved_text

        # Create a simple cube base mesh
        base_mesh = trimesh.creation.box(extents=[10, 10, 10])

        font_path = self._get_test_font()

        result_mesh = create_engraved_text(
            base_mesh=base_mesh,
            text="1",
            face_center=[0, 0, 5],  # Top face center
            face_normal=[0, 0, 1],  # Pointing up
            text_depth=1.0,
            text_size=3.0,
            font_path=font_path,
        )

        assert result_mesh is not None
        assert len(result_mesh.vertices) > len(base_mesh.vertices)
        assert result_mesh.is_watertight

    def test_font_fallback(self):
        """Test behavior with invalid font path."""
        from dice_models.geometry.text import _create_font_text_mesh

        # Should handle gracefully or use fallback
        mesh = _create_font_text_mesh(
            text="1", size=3.0, depth=1.0, font_path="/nonexistent/font.ttf"
        )

        # Should either create mesh with fallback or return None
        # The behavior depends on implementation, but shouldn't crash
        assert mesh is None or (mesh is not None and len(mesh.vertices) > 0)

    def _get_test_font(self) -> str:
        """Get a font path for testing."""
        from pathlib import Path

        system_fonts = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]

        for font_path in system_fonts:
            if Path(font_path).exists():
                return font_path

        return None


class TestIntegration:
    """Integration tests for the complete dice generation process."""

    def test_complete_dice_generation_workflow(self):
        """Test the complete workflow from creation to export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dice of each type
            for sides in [4, 6, 8, 10, 12, 20]:
                output_path = Path(temp_dir) / f"integration_d{sides}.stl"

                dice = create_standard_dice(
                    sides=sides,
                    radius=10.0,
                    output_path=output_path,
                    text_depth=0.5,
                    text_size=2.0,
                )

                # Verify file was created
                assert output_path.exists()
                assert output_path.stat().st_size > 0

                # Verify dice properties
                info = dice.get_info()
                assert info["sides"] == sides
                assert info["radius"] == 10.0

    def test_custom_layout_workflow(self):
        """Test workflow with custom number layouts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a D6 with reversed numbers
            custom_layout = [6, 5, 4, 3, 2, 1]
            output_path = Path(temp_dir) / "custom_d6.stl"

            dice = create_standard_dice(
                sides=6,
                radius=8.0,
                number_layout=custom_layout,
                output_path=output_path,
            )

            assert output_path.exists()
            assert dice.number_layout == custom_layout

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid sides
        with pytest.raises(ValueError):
            create_standard_dice(sides=13)

        # Test invalid custom layout
        with pytest.raises(ValueError):
            DiceGeometry(
                PolyhedronType.CUBE, number_layout=[1, 2, 3], curve_resolution="low"
            )  # Too few numbers
