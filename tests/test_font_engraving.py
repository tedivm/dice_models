"""Comprehensive tests for font-based text engraving system."""

import tempfile
from pathlib import Path

from dice_models.geometry.dice import create_standard_dice
from dice_models.geometry.text import _create_font_text_mesh, create_engraved_text


class TestFontBasedEngraving:
    """Test the font-based text engraving system."""

    def test_all_dice_types_engraving(self):
        """Test that all dice types can be engraved with font-based text."""
        dice_types = [4, 6, 8, 10, 12, 20]
        font_path = self._get_test_font()

        for sides in dice_types:
            dice = create_standard_dice(
                sides=sides,
                radius=10.0,
                text_depth=1.0,
                text_size=3.0,
                font_path=font_path,
                curve_resolution="medium",  # Basic functionality test
            )

            # Generate meshes to compare
            mesh_no_numbers = dice.generate_mesh(include_numbers=False)
            mesh_with_numbers = dice.generate_mesh(include_numbers=True)

            # Verify engraving worked
            vertex_increase = len(mesh_with_numbers.vertices) - len(mesh_no_numbers.vertices)
            volume_decrease = mesh_no_numbers.volume - mesh_with_numbers.volume

            assert vertex_increase > 10, f"D{sides}: No significant vertex increase ({vertex_increase})"
            assert volume_decrease > 0, f"D{sides}: No volume decrease ({volume_decrease})"
            assert mesh_with_numbers.is_watertight, f"D{sides}: Result mesh not watertight"

            # Verify mesh integrity
            assert len(mesh_with_numbers.vertices) > 0, f"D{sides}: No vertices in result"
            assert len(mesh_with_numbers.faces) > 0, f"D{sides}: No faces in result"

    def test_individual_character_engraving(self):
        """Test engraving individual characters on fresh base meshes."""
        dice_types = [(6, 6), (8, 8), (20, 20)]  # (sides, max_number)
        font_path = self._get_test_font()

        for sides, max_number in dice_types:
            dice = create_standard_dice(sides=sides, radius=10.0, curve_resolution="medium")

            success_count = 0
            failed_chars = []

            for i in range(max_number):
                number = i + 1

                # Test on fresh base mesh
                base_mesh = dice.generate_mesh(include_numbers=False)

                try:
                    result_mesh = create_engraved_text(
                        base_mesh=base_mesh,
                        text=str(number),
                        face_center=dice.face_centers[i],
                        face_normal=dice.face_normals[i],
                        text_depth=1.0,
                        text_size=3.0,
                        font_path=font_path,
                    )

                    vertex_increase = len(result_mesh.vertices) - len(base_mesh.vertices)
                    volume_decrease = base_mesh.volume - result_mesh.volume

                    if vertex_increase > 0 and volume_decrease >= 0:
                        success_count += 1
                    else:
                        failed_chars.append(number)

                except Exception:
                    failed_chars.append(f"{number}(error)")

            # Allow for some failures but expect most to work
            success_rate = success_count / max_number
            assert success_rate >= 0.8, (
                f"D{sides}: Only {success_count}/{max_number} characters worked. Failed: {failed_chars}"
            )

    def test_problematic_character_combinations(self):
        """Test specific character and face combinations that have been problematic."""
        font_path = self._get_test_font()

        # Test problematic combinations we discovered
        test_cases = [
            (8, "5", [6]),  # D8 face 7 (index 6) with character '5'
            (8, "7", [0, 6]),  # D8 faces 1 and 7 with character '7'
        ]

        for sides, char, face_indices in test_cases:
            dice = create_standard_dice(sides=sides, radius=10.0, curve_resolution="medium")

            for face_idx in face_indices:
                base_mesh = dice.generate_mesh(include_numbers=False)

                result_mesh = create_engraved_text(
                    base_mesh=base_mesh,
                    text=char,
                    face_center=dice.face_centers[face_idx],
                    face_normal=dice.face_normals[face_idx],
                    text_depth=1.0,
                    text_size=3.0,
                    font_path=font_path,
                )

                vertex_increase = len(result_mesh.vertices) - len(base_mesh.vertices)
                assert vertex_increase > 0, f"D{sides} char '{char}' on face {face_idx + 1} failed"

    def test_different_characters_on_same_face(self):
        """Test different characters on the same face to isolate character vs face issues."""
        font_path = self._get_test_font()
        dice = create_standard_dice(sides=8, radius=10.0, curve_resolution="medium")

        # Test various characters on D8 face 7 (previously problematic)
        test_chars = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "A", "B"]
        face_idx = 6  # Face 7 (index 6)

        failed_chars = []

        for char in test_chars:
            base_mesh = dice.generate_mesh(include_numbers=False)

            try:
                result_mesh = create_engraved_text(
                    base_mesh=base_mesh,
                    text=char,
                    face_center=dice.face_centers[face_idx],
                    face_normal=dice.face_normals[face_idx],
                    text_depth=1.0,
                    text_size=3.0,
                    font_path=font_path,
                )

                vertex_increase = len(result_mesh.vertices) - len(base_mesh.vertices)
                if vertex_increase <= 0:
                    failed_chars.append(char)

            except Exception:
                failed_chars.append(f"{char}(error)")

        # Most characters should work
        success_rate = (len(test_chars) - len(failed_chars)) / len(test_chars)
        assert success_rate >= 0.8, f"Too many characters failed on D8 face 7: {failed_chars}"

    def test_accumulating_engraving_complexity(self):
        """Test that multiple engravings on the same mesh work correctly."""
        font_path = self._get_test_font()
        dice = create_standard_dice(sides=6, radius=10.0, curve_resolution="medium")

        base_mesh = dice.generate_mesh(include_numbers=False)
        current_mesh = base_mesh.copy()

        successful_engravings = 0

        for i in range(6):
            number = i + 1

            try:
                result_mesh = create_engraved_text(
                    base_mesh=current_mesh,
                    text=str(number),
                    face_center=dice.face_centers[i],
                    face_normal=dice.face_normals[i],
                    text_depth=1.0,
                    text_size=3.0,
                    font_path=font_path,
                )

                vertex_increase = len(result_mesh.vertices) - len(current_mesh.vertices)
                if vertex_increase > 0:
                    successful_engravings += 1
                    current_mesh = result_mesh

            except Exception:
                pass  # Continue with next engraving

        # Should successfully engrave most numbers
        assert successful_engravings >= 4, f"Only {successful_engravings}/6 accumulated engravings succeeded"

        # Final mesh should be more complex than base
        final_vertex_increase = len(current_mesh.vertices) - len(base_mesh.vertices)
        assert final_vertex_increase > 50, f"Insufficient final complexity increase: {final_vertex_increase}"

    def test_text_mesh_creation(self):
        """Test font-based text mesh creation for various characters."""
        font_path = self._get_test_font()
        test_strings = ["1", "7", "10", "ABC", "!@#"]

        for text in test_strings:
            text_mesh = _create_font_text_mesh(text, 3.0, 1.0, font_path)

            assert text_mesh is not None, f"Failed to create mesh for '{text}'"
            assert len(text_mesh.vertices) > 0, f"No vertices in mesh for '{text}'"
            assert len(text_mesh.faces) > 0, f"No faces in mesh for '{text}'"
            assert text_mesh.volume > 0, f"Non-positive volume for '{text}'"
            assert text_mesh.is_watertight, f"Non-watertight mesh for '{text}'"

    def test_font_fallback_behavior(self):
        """Test behavior when font is not available."""
        # Test with non-existent font path
        fake_font_path = "/path/to/nonexistent/font.ttf"

        dice = create_standard_dice(
            sides=6,
            radius=10.0,
            text_depth=1.0,
            text_size=3.0,
            font_path=fake_font_path,
            curve_resolution="medium",
        )

        # Should still work with fallback font or simple geometry
        mesh_no_numbers = dice.generate_mesh(include_numbers=False)
        mesh_with_numbers = dice.generate_mesh(include_numbers=True)

        vertex_increase = len(mesh_with_numbers.vertices) - len(mesh_no_numbers.vertices)
        assert vertex_increase > 0, "Fallback engraving failed"

    def test_engraving_parameters(self):
        """Test various engraving parameters."""
        font_path = self._get_test_font()
        dice = create_standard_dice(sides=6, radius=10.0, curve_resolution="medium")
        base_mesh = dice.generate_mesh(include_numbers=False)

        # Test different text depths
        for depth in [0.5, 1.0, 2.0]:
            result = create_engraved_text(
                base_mesh=base_mesh.copy(),
                text="1",
                face_center=dice.face_centers[0],
                face_normal=dice.face_normals[0],
                text_depth=depth,
                text_size=3.0,
                font_path=font_path,
            )

            vertex_increase = len(result.vertices) - len(base_mesh.vertices)
            assert vertex_increase > 0, f"Engraving failed with depth {depth}"

        # Test different text sizes
        for size in [1.0, 3.0, 5.0]:
            result = create_engraved_text(
                base_mesh=base_mesh.copy(),
                text="1",
                face_center=dice.face_centers[0],
                face_normal=dice.face_normals[0],
                text_depth=1.0,
                text_size=size,
                font_path=font_path,
            )

            vertex_increase = len(result.vertices) - len(base_mesh.vertices)
            assert vertex_increase > 0, f"Engraving failed with size {size}"

    def test_stl_export_with_engraving(self):
        """Test STL export of engraved dice."""
        font_path = self._get_test_font()

        with tempfile.TemporaryDirectory() as temp_dir:
            for sides in [6, 12, 20]:
                output_path = Path(temp_dir) / f"test_d{sides}_engraved.stl"

                dice = create_standard_dice(
                    sides=sides,
                    radius=10.0,
                    text_depth=1.0,
                    text_size=3.0,
                    font_path=font_path,
                    curve_resolution="medium",
                )

                dice.export_stl(output_path, include_numbers=True)

                assert output_path.exists(), f"STL file not created for D{sides}"
                assert output_path.stat().st_size > 1000, f"STL file too small for D{sides}"

    def _get_test_font(self) -> str:
        """Get a font path for testing."""
        # Try to find a system font
        system_fonts = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]

        for font_path in system_fonts:
            if Path(font_path).exists():
                return font_path

        # If no system font found, return None and let the system use fallback
        return None


class TestEngravingRegressions:
    """Test for specific regressions that were discovered during development."""

    def test_d8_number_7_regression(self):
        """Test the specific D8 number 7 issue that was discovered."""
        font_path = self._get_test_font()
        dice = create_standard_dice(sides=8, radius=10.0, curve_resolution="medium")

        # This specific combination was failing
        base_mesh = dice.generate_mesh(include_numbers=False)
        result = create_engraved_text(
            base_mesh=base_mesh,
            text="7",
            face_center=dice.face_centers[6],  # Face 7 (index 6)
            face_normal=dice.face_normals[6],
            text_depth=1.0,
            text_size=3.0,
            font_path=font_path,
        )

        vertex_increase = len(result.vertices) - len(base_mesh.vertices)
        assert vertex_increase > 0, "D8 number 7 regression: engraving failed"

    def test_volume_change_consistency(self):
        """Test that volume changes are consistent and correct."""
        font_path = self._get_test_font()

        for sides in [4, 6, 8, 12, 20]:
            dice = create_standard_dice(
                sides=sides,
                radius=10.0,
                text_depth=1.0,
                text_size=3.0,
                font_path=font_path,
                curve_resolution="medium",
            )

            mesh_no_numbers = dice.generate_mesh(include_numbers=False)
            mesh_with_numbers = dice.generate_mesh(include_numbers=True)

            volume_change = mesh_no_numbers.volume - mesh_with_numbers.volume

            # Volume should decrease (material removed by engraving)
            assert volume_change > 0, f"D{sides}: Volume didn't decrease ({volume_change})"

            # Volume change should be reasonable (not too large)
            volume_ratio = volume_change / mesh_no_numbers.volume
            assert volume_ratio < 0.1, f"D{sides}: Volume change too large ({volume_ratio:.3%})"

    def test_backwards_text_regression(self):
        """Test for backwards text issues that were reported."""
        font_path = self._get_test_font()
        dice = create_standard_dice(sides=6, radius=10.0, curve_resolution="medium")

        # Create a simple test to verify text orientation
        # This is more of a visual test, but we can at least verify the mesh is created
        for i in range(6):
            base_mesh = dice.generate_mesh(include_numbers=False)
            result = create_engraved_text(
                base_mesh=base_mesh,
                text=str(i + 1),
                face_center=dice.face_centers[i],
                face_normal=dice.face_normals[i],
                text_depth=1.0,
                text_size=3.0,
                font_path=font_path,
            )

            # Basic validation that engraving worked
            vertex_increase = len(result.vertices) - len(base_mesh.vertices)
            assert vertex_increase > 0, f"D6 face {i + 1} engraving failed"
            assert result.is_watertight, f"D6 face {i + 1} result not watertight"

    def _get_test_font(self) -> str:
        """Get a font path for testing."""
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
