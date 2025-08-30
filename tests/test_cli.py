"""Tests for CLI functionality."""

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from dice_models.cli import app


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "dice_models" in result.stdout

    def test_list_types_command(self):
        """Test list-types command."""
        result = self.runner.invoke(app, ["list-types"])
        assert result.exit_code == 0
        assert "Supported dice types:" in result.stdout
        assert "4 sides" in result.stdout
        assert "20 sides" in result.stdout

    def test_generate_command_valid_dice(self):
        """Test generate command with valid parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_d6.stl"

            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",  # sides
                    str(output_path),  # output
                    "--radius",
                    "15.0",
                    "--text-depth",
                    "0.8",
                    "--text-size",
                    "4.0",
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()
            assert "Generated CUBE dice:" in result.stdout

    def test_generate_command_invalid_sides(self):
        """Test generate command with invalid number of sides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "invalid.stl"

            result = self.runner.invoke(
                app, ["generate", "13", str(output_path)]  # invalid sides
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_generate_command_no_numbers(self):
        """Test generate command without numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "no_numbers_d8.stl"

            result = self.runner.invoke(
                app, ["generate", "8", str(output_path), "--no-numbers"]
            )

            assert result.exit_code == 0
            assert output_path.exists()
            assert "Numbers: None" in result.stdout

    def test_generate_command_custom_layout(self):
        """Test generate command with custom number layout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_d4.stl"

            result = self.runner.invoke(
                app, ["generate", "4", str(output_path), "--layout", "4,3,2,1"]
            )

            assert result.exit_code == 0
            assert output_path.exists()
            assert "[4, 3, 2, 1]" in result.stdout

    def test_generate_command_invalid_layout(self):
        """Test generate command with invalid custom layout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "invalid_layout.stl"

            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--layout",
                    "1,2,3",  # Too few numbers for d6
                ],
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_batch_generate_command(self):
        """Test batch generate command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a configuration file
            config = {
                "dice": [
                    {
                        "sides": 4,
                        "filename": "batch_d4.stl",
                        "radius": 8.0,
                        "text_depth": 0.3,
                    },
                    {
                        "sides": 6,
                        "filename": "batch_d6.stl",
                        "radius": 10.0,
                        "no_numbers": True,
                    },
                ]
            }

            config_path = Path(temp_dir) / "batch_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            output_dir = Path(temp_dir) / "batch_output"

            result = self.runner.invoke(
                app,
                ["batch-generate", str(config_path), "--output-dir", str(output_dir)],
            )

            assert result.exit_code == 0
            assert (output_dir / "batch_d4.stl").exists()
            assert (output_dir / "batch_d6.stl").exists()
            assert "Batch generation complete" in result.stdout

    def test_batch_generate_missing_file(self):
        """Test batch generate with missing configuration file."""
        result = self.runner.invoke(app, ["batch-generate", "nonexistent_config.json"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_help_commands(self):
        """Test help output for all commands."""
        # Test main help
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout

        # Test command-specific help
        commands = ["generate", "list-types", "batch-generate", "version"]
        for command in commands:
            result = self.runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.stdout


class TestCLIEdgeCases:
    """Test edge cases and error conditions in CLI."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_generate_with_all_dice_types(self):
        """Test generating all supported dice types."""
        valid_sides = [4, 6, 8, 10, 12, 20]

        with tempfile.TemporaryDirectory() as temp_dir:
            for sides in valid_sides:
                output_path = Path(temp_dir) / f"test_d{sides}.stl"

                result = self.runner.invoke(
                    app, ["generate", str(sides), str(output_path), "--radius", "5.0"]
                )

                assert result.exit_code == 0, f"Failed for D{sides}: {result.stdout}"
                assert output_path.exists(), f"File not created for D{sides}"

    def test_generate_with_extreme_parameters(self):
        """Test generate command with extreme but valid parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "extreme_d6.stl"

            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--radius",
                    "100.0",  # Very large
                    "--text-depth",
                    "10.0",  # Very deep
                    "--text-size",
                    "20.0",  # Very large text
                ],
            )

            # Should still work, even if not practical
            assert result.exit_code == 0
            assert output_path.exists()

    def test_directory_creation_for_output(self):
        """Test that output directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dirs" / "test_d8.stl"

            result = self.runner.invoke(app, ["generate", "8", str(nested_path)])

            assert result.exit_code == 0
            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_output_file_overwrite(self):
        """Test overwriting existing output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "overwrite_test.stl"

            # Create first file
            result1 = self.runner.invoke(app, ["generate", "4", str(output_path)])
            assert result1.exit_code == 0
            first_size = output_path.stat().st_size

            # Overwrite with different dice
            result2 = self.runner.invoke(
                app, ["generate", "20", str(output_path), "--radius", "20.0"]
            )
            assert result2.exit_code == 0
            second_size = output_path.stat().st_size

            # Files should be different sizes
            assert first_size != second_size

    def test_generate_with_font_parameter(self):
        """Test generate command with font parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_d6_font.stl"

            # Test with a non-existent font (should still work with fallback)
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--font",
                    "/nonexistent/font.ttf",
                    "--text-depth",
                    "1.0",
                    "--text-size",
                    "4.0",
                ],
            )
            assert result.exit_code == 0
            assert output_path.exists()
            assert output_path.stat().st_size > 1000

    def test_generate_with_system_font(self):
        """Test generate command with system font if available."""
        # Common system font paths
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]

        available_font = None
        for font_path in font_paths:
            if Path(font_path).exists():
                available_font = font_path
                break

        if available_font:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "test_d6_system_font.stl"

                result = self.runner.invoke(
                    app,
                    [
                        "generate",
                        "6",
                        str(output_path),
                        "--font",
                        available_font,
                    ],
                )
                assert result.exit_code == 0
                assert output_path.exists()
                assert (
                    "font" in result.stdout.lower() or output_path.stat().st_size > 1000
                )

    def test_generate_parameter_validation(self):
        """Test parameter validation edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_validation.stl"

            # Test with very small radius
            result = self.runner.invoke(
                app,
                ["generate", "6", str(output_path), "--radius", "0.1"],
            )
            assert result.exit_code == 0  # Should work but generate small dice

            # Test with very large text size relative to radius
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--radius",
                    "5.0",
                    "--text-size",
                    "10.0",
                ],
            )
            assert result.exit_code == 0  # Should handle gracefully

            # Test with zero text depth (effectively blank)
            result = self.runner.invoke(
                app,
                ["generate", "6", str(output_path), "--text-depth", "0.0"],
            )
            assert result.exit_code == 0

    def test_batch_generate_error_handling(self):
        """Test batch generation error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid JSON
            invalid_config = Path(temp_dir) / "invalid.json"
            invalid_config.write_text("{ invalid json")

            result = self.runner.invoke(
                app, ["batch-generate", str(invalid_config), "--output-dir", temp_dir]
            )
            assert result.exit_code != 0

            # Test with valid JSON but invalid dice configuration
            invalid_dice_config = {
                "dice": [
                    {"sides": 7, "filename": "invalid.stl"},  # Invalid sides
                ]
            }
            config_file = Path(temp_dir) / "invalid_dice.json"
            with open(config_file, "w") as f:
                json.dump(invalid_dice_config, f)

            result = self.runner.invoke(
                app, ["batch-generate", str(config_file), "--output-dir", temp_dir]
            )
            assert result.exit_code != 0

    def test_generate_layout_edge_cases(self):
        """Test custom layout edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_layout_edge.stl"

            # Test layout with wrong number of faces
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--layout",
                    "1,2,3",  # Too few numbers for D6
                ],
            )
            assert result.exit_code != 0  # Should fail validation

            # Test layout with duplicate numbers
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--layout",
                    "1,1,1,1,1,1",  # All the same number
                ],
            )
            assert result.exit_code == 0  # Should work (might be desired for some uses)

            # Test layout with large numbers
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "6",
                    str(output_path),
                    "--layout",
                    "100,200,300,400,500,600",
                ],
            )
            assert result.exit_code == 0  # Should handle large numbers

    def test_generate_output_info_display(self):
        """Test that generate command displays comprehensive output info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_info.stl"

            result = self.runner.invoke(
                app,
                [
                    "generate",
                    "12",
                    str(output_path),
                    "--radius",
                    "15.0",
                    "--text-depth",
                    "1.2",
                    "--text-size",
                    "4.5",
                ],
            )
            assert result.exit_code == 0

            # Check that all important info is displayed
            output_text = result.stdout
            assert "DODECAHEDRON" in output_text or "12" in output_text
            assert "15.0" in output_text  # radius
            assert "1.2" in output_text  # text depth
            assert "4.5" in output_text  # text size
            assert str(output_path) in output_text

    def test_cli_helper_functions(self):
        """Test the CLI helper functions directly."""
        import sys
        from io import StringIO

        from dice_models.cli import _display_generation_info, _parse_custom_layout
        from dice_models.geometry.dice import create_standard_dice

        # Test layout parsing
        assert _parse_custom_layout("1,2,3,4,5,6") == [1, 2, 3, 4, 5, 6]
        assert _parse_custom_layout(None) is None
        assert _parse_custom_layout("") is None

        try:
            _parse_custom_layout("1,2,invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Test display function (capture output)
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            dice = create_standard_dice(6, radius=10.0)
            _display_generation_info(dice, "test.stl", False)
            output = captured_output.getvalue()
            assert "CUBE" in output or "6" in output
            assert "10.0" in output
            assert "test.stl" in output
        finally:
            sys.stdout = old_stdout
            sys.stdout = old_stdout
