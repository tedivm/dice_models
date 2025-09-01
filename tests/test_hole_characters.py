"""Test cases for characters with holes (inner regions) in font-based engraving."""

import tempfile
from pathlib import Path

from dice_models.geometry.dice import DiceGeometry
from dice_models.geometry.polyhedra import PolyhedronType


class TestHoleCharacters:
    """Test font-based engraving with characters that contain holes."""

    def test_numeric_characters_with_holes(self):
        """Test that numeric characters with holes (4, 6, 8, 9, 0) engrave correctly."""
        hole_numbers = ["4", "6", "8", "9", "0"]

        for number in hole_numbers:
            dice = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=8.0,
                number_layout=[number] * 6,  # All faces show the same number
                text_size=3.0,
                text_depth=0.8,
                curve_resolution="high",  # High quality needed for hole detection
            )

            # Generate both blank and engraved versions
            blank_mesh = dice.generate_mesh(include_numbers=False)
            engraved_mesh = dice.generate_mesh(include_numbers=True)

            # Basic mesh validity checks
            assert engraved_mesh.is_watertight, f"Number '{number}' mesh is not watertight"
            assert engraved_mesh.volume > 0, f"Number '{number}' mesh has non-positive volume"
            assert len(engraved_mesh.vertices) > 0, f"Number '{number}' mesh has no vertices"
            assert len(engraved_mesh.faces) > 0, f"Number '{number}' mesh has no faces"

            # Check that engraving actually removed material
            volume_difference = blank_mesh.volume - engraved_mesh.volume
            assert volume_difference > 0, f"Number '{number}' engraving did not remove material"
            assert volume_difference < blank_mesh.volume * 0.1, f"Number '{number}' removed too much material"

    def test_letter_characters_with_holes(self):
        """Test that letter characters with holes (A, B, D, O, P, Q, R) engrave correctly."""
        hole_letters = ["A", "B", "D", "O", "P", "Q", "R"]

        for letter in hole_letters:
            dice = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=10.0,
                number_layout=[letter] * 6,  # All faces show the same letter
                text_size=4.0,
                text_depth=1.0,
                curve_resolution="high",  # High quality needed for hole detection
            )

            # Generate engraved mesh
            engraved_mesh = dice.generate_mesh(include_numbers=True)

            # Basic mesh validity checks
            assert engraved_mesh.is_watertight, f"Letter '{letter}' mesh is not watertight"
            assert engraved_mesh.volume > 0, f"Letter '{letter}' mesh has non-positive volume"
            assert len(engraved_mesh.vertices) >= 100, f"Letter '{letter}' mesh has too few vertices"

    def test_complex_hole_character_eight(self):
        """Test the number '8' specifically as it has two distinct holes."""
        dice = DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            radius=12.0,
            number_layout=["8"] * 6,
            text_size=5.0,
            text_depth=1.2,
            curve_resolution="high",  # High quality needed for complex hole detection
        )

        blank_mesh = dice.generate_mesh(include_numbers=False)
        engraved_mesh = dice.generate_mesh(include_numbers=True)

        # Specific checks for '8'
        assert engraved_mesh.is_watertight, "Number '8' mesh is not watertight"
        assert engraved_mesh.volume > 0, "Number '8' mesh has non-positive volume"

        # Check material removal
        volume_removed = blank_mesh.volume - engraved_mesh.volume
        assert volume_removed > 0, "Number '8' engraving did not remove material"

        # Check mesh complexity (should have more vertices due to holes)
        assert len(engraved_mesh.vertices) > 200, "Number '8' should have complex geometry"

    def test_mixed_hole_and_solid_characters(self):
        """Test a mix of characters with and without holes on the same die."""
        # Mix of hole-containing and solid characters
        mixed_layout = ["0", "1", "8", "7", "4", "3"]

        dice = DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            radius=10.0,
            number_layout=mixed_layout,
            text_size=3.5,
            text_depth=0.8,
            curve_resolution="high",  # High quality needed for hole detection
        )

        engraved_mesh = dice.generate_mesh(include_numbers=True)

        # Should handle the mix correctly
        assert engraved_mesh.is_watertight, "Mixed character mesh is not watertight"
        assert engraved_mesh.volume > 0, "Mixed character mesh has non-positive volume"
        assert len(engraved_mesh.vertices) > 200, "Mixed character mesh should be reasonably complex"

    def test_stl_export_with_hole_characters(self):
        """Test that hole characters can be exported to STL successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dice = DiceGeometry(
                polyhedron_type=PolyhedronType.OCTAHEDRON,  # D8
                radius=10.0,
                number_layout=["4", "6", "8", "9", "0", "1", "2", "3"],
                text_size=3.0,
                text_depth=0.8,
                curve_resolution="high",  # High quality needed for hole detection
            )

            output_path = Path(temp_dir) / "hole_characters_test.stl"
            dice.export_stl(output_path)

            # Check file was created and has reasonable size
            assert output_path.exists(), "STL file with hole characters was not created"
            assert output_path.stat().st_size > 1000, "STL file with hole characters is too small"

    def test_hole_character_regression_prevention(self):
        """Regression test to ensure hole characters don't get filled in."""
        # This test specifically checks that characters with holes maintain their holes
        # by comparing the vertex count of a hole character vs a solid character

        # Create dice with hole-containing character
        hole_dice = DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            radius=10.0,
            number_layout=["8"] * 6,  # '8' has two holes
            text_size=4.0,
            text_depth=1.0,
            curve_resolution="high",  # High quality needed for hole detection
        )

        # Create dice with solid character
        solid_dice = DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            radius=10.0,
            number_layout=["1"] * 6,  # '1' is a simple solid shape
            text_size=4.0,
            text_depth=1.0,
            curve_resolution="high",  # Use same resolution for fair comparison
        )

        hole_mesh = hole_dice.generate_mesh(include_numbers=True)
        solid_mesh = solid_dice.generate_mesh(include_numbers=True)

        # The hole character should have more vertices due to the holes
        # (internal boundaries create additional geometry)
        assert len(hole_mesh.vertices) > len(solid_mesh.vertices), (
            "Hole character '8' should have more vertices than solid character '1'"
        )

        # Both should be valid meshes
        assert hole_mesh.is_watertight and solid_mesh.is_watertight, (
            "Both hole and solid character meshes should be watertight"
        )
