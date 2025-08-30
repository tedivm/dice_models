"""Integration tests for the complete dice generation workflow."""

import tempfile
from pathlib import Path

from dice_models import create_standard_dice
from dice_models.geometry import DiceGeometry, PolyhedronType


class TestWorkflowIntegration:
    """Test complete workflows from start to finish."""

    def test_complete_library_workflow(self):
        """Test the complete library usage workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test creating dice with all standard types
            dice_types = [4, 6, 8, 10, 12, 20]

            created_files = []

            for sides in dice_types:
                output_path = Path(temp_dir) / f"workflow_d{sides}.stl"

                # Create dice using convenience function
                dice = create_standard_dice(
                    sides=sides,
                    radius=10.0,
                    text_depth=0.8,
                    text_size=3.5,
                )

                # Generate both numbered and blank versions
                dice.export_stl(output_path, include_numbers=True)
                blank_path = Path(temp_dir) / f"workflow_d{sides}_blank.stl"
                dice.export_stl(blank_path, include_numbers=False)

                # Verify files exist and have reasonable sizes
                assert output_path.exists(), f"D{sides} file not created"
                assert blank_path.exists(), f"D{sides} blank file not created"

                numbered_size = output_path.stat().st_size
                blank_size = blank_path.stat().st_size

                # File size expectations (based on actual output)
                assert numbered_size > 100, f"D{sides} file too small: {numbered_size} bytes"
                assert blank_size > 100, f"D{sides} blank file too small: {blank_size} bytes"

                # Files should be valid STL files (not empty)
                assert numbered_size > 0 and blank_size > 0

                created_files.extend([output_path, blank_path])

                # Test dice info
                info = dice.get_info()
                assert info["sides"] == sides
                assert info["radius"] == 10.0
                assert len(info["number_layout"]) == sides

            # Verify we created all expected files
            assert len(created_files) == len(dice_types) * 2

    def test_custom_configuration_workflow(self):
        """Test workflow with heavily customized configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a D6 with custom everything
            custom_layout = [6, 5, 4, 3, 2, 1]  # Reverse layout

            d6_custom = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=15.0,
                number_layout=custom_layout,
                text_depth=1.2,
                text_size=5.0,
            )

            # Generate and export
            output_path = Path(temp_dir) / "custom_d6.stl"
            d6_custom.export_stl(output_path, include_numbers=True)

            # Verify customization took effect
            info = d6_custom.get_info()
            assert info["number_layout"] == custom_layout
            assert info["radius"] == 15.0
            assert info["text_depth"] == 1.2
            assert info["text_size"] == 5.0

            # File should exist and be substantial
            assert output_path.exists()
            assert output_path.stat().st_size > 5000  # Should be larger due to custom size

    def test_font_workflow_integration(self):
        """Test complete workflow with font specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with system fonts if available
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

            # Test with available font
            if available_font:
                d6_font = create_standard_dice(
                    sides=6,
                    radius=12.0,
                    font_path=available_font,
                    text_depth=1.0,
                    text_size=4.0,
                )

                font_output = Path(temp_dir) / "font_d6.stl"
                d6_font.export_stl(font_output, include_numbers=True)

                assert font_output.exists()
                assert font_output.stat().st_size > 2000

                info = d6_font.get_info()
                assert info["font_path"] == available_font

            # Test with non-existent font (should fallback gracefully)
            d6_fallback = create_standard_dice(
                sides=6,
                radius=12.0,
                font_path="/nonexistent/font.ttf",
                text_depth=1.0,
                text_size=4.0,
            )

            fallback_output = Path(temp_dir) / "fallback_d6.stl"
            d6_fallback.export_stl(fallback_output, include_numbers=True)

            assert fallback_output.exists()
            assert fallback_output.stat().st_size > 2000

    def test_batch_processing_workflow(self):
        """Test batch processing multiple dice configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define batch configuration
            batch_configs = [
                {"sides": 4, "radius": 8.0, "text_depth": 0.4},
                {"sides": 6, "radius": 10.0, "text_depth": 0.6, "text_size": 3.5},
                {
                    "sides": 8,
                    "radius": 12.0,
                    "text_depth": 0.8,
                    "number_layout": [8, 7, 6, 5, 4, 3, 2, 1],
                },
                {"sides": 20, "radius": 20.0, "text_depth": 1.0, "text_size": 5.0},
            ]

            results = []

            for i, config in enumerate(batch_configs):
                # Extract config
                sides = config["sides"]
                radius = config["radius"]
                text_depth = config["text_depth"]
                text_size = config.get("text_size", 3.0)
                number_layout = config.get("number_layout")

                # Create dice
                dice = create_standard_dice(
                    sides=sides,
                    radius=radius,
                    number_layout=number_layout,
                    text_depth=text_depth,
                    text_size=text_size,
                )

                # Export
                output_path = Path(temp_dir) / f"batch_{i}_d{sides}.stl"
                dice.export_stl(output_path, include_numbers=True)

                # Validate
                assert output_path.exists()
                assert output_path.stat().st_size > 1000

                results.append({"dice": dice, "path": output_path, "config": config})

            # Verify all dice were created with correct properties
            assert len(results) == len(batch_configs)

            for result in results:
                info = result["dice"].get_info()
                config = result["config"]

                assert info["sides"] == config["sides"]
                assert info["radius"] == config["radius"]
                assert info["text_depth"] == config["text_depth"]

    def test_parameter_range_workflow(self):
        """Test workflow with various parameter ranges."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with different parameter combinations
            test_cases = [
                # Small dice
                {"sides": 6, "radius": 5.0, "text_depth": 0.2, "text_size": 1.5},
                # Large dice
                {"sides": 6, "radius": 25.0, "text_depth": 2.0, "text_size": 8.0},
                # Deep engraving
                {"sides": 6, "radius": 15.0, "text_depth": 3.0, "text_size": 4.0},
                # Shallow engraving
                {"sides": 6, "radius": 15.0, "text_depth": 0.1, "text_size": 4.0},
                # Large text
                {"sides": 6, "radius": 20.0, "text_depth": 1.0, "text_size": 10.0},
                # Small text
                {"sides": 6, "radius": 20.0, "text_depth": 1.0, "text_size": 1.0},
            ]

            for i, params in enumerate(test_cases):
                dice = create_standard_dice(**params)
                output_path = Path(temp_dir) / f"params_{i}_d{params['sides']}.stl"

                # Should handle all parameter ranges gracefully
                dice.export_stl(output_path, include_numbers=True)

                assert output_path.exists()
                assert output_path.stat().st_size > 500  # Should create some geometry

                # Verify parameters were applied
                info = dice.get_info()
                for key, expected_value in params.items():
                    if key in info:
                        assert info[key] == expected_value

    def test_error_recovery_workflow(self):
        """Test workflow resilience to various error conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid parameters that should be handled gracefully
            try:
                # Invalid sides
                dice = create_standard_dice(sides=7)  # Should raise ValueError
                assert False, "Should have raised ValueError for invalid sides"
            except ValueError:
                pass  # Expected

            # Test with extreme but valid parameters
            extreme_cases = [
                {"sides": 6, "radius": 0.1, "text_depth": 0.01, "text_size": 0.1},
                {"sides": 20, "radius": 100.0, "text_depth": 10.0, "text_size": 20.0},
            ]

            for params in extreme_cases:
                try:
                    dice = create_standard_dice(**params)
                    output_path = Path(temp_dir) / f"extreme_{params['sides']}.stl"
                    dice.export_stl(output_path, include_numbers=True)

                    # Should create some file even with extreme parameters
                    assert output_path.exists()

                except Exception as e:
                    # If it fails, it should fail gracefully with a reasonable error
                    assert isinstance(e, (ValueError, RuntimeError))

    def test_concurrent_workflow(self):
        """Test that workflows can handle concurrent operations."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            errors = []

            def create_dice_worker(sides, thread_id):
                try:
                    dice = create_standard_dice(
                        sides=sides,
                        radius=10.0 + thread_id,  # Slightly different sizes
                        text_depth=0.5 + thread_id * 0.1,
                    )

                    output_path = Path(temp_dir) / f"concurrent_{thread_id}_d{sides}.stl"
                    dice.export_stl(output_path, include_numbers=True)

                    results.append(
                        {
                            "thread_id": thread_id,
                            "sides": sides,
                            "path": output_path,
                            "size": output_path.stat().st_size,
                        }
                    )
                except Exception as e:
                    errors.append({"thread_id": thread_id, "error": str(e)})

            # Create multiple dice concurrently
            threads = []
            for i in range(4):
                sides = [6, 8, 12, 20][i]
                thread = threading.Thread(target=create_dice_worker, args=(sides, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            assert len(errors) == 0, f"Concurrent errors: {errors}"
            assert len(results) == 4, "Not all concurrent operations completed"

            # All files should exist and have reasonable sizes
            for result in results:
                assert result["path"].exists()
                assert result["size"] > 1000

    def test_memory_efficiency_workflow(self):
        """Test workflow with multiple operations to check for memory leaks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and destroy many dice objects
            for i in range(20):  # Reduced from 100 to keep test fast
                dice = create_standard_dice(
                    sides=6,
                    radius=10.0,
                    text_depth=0.5,
                    text_size=3.0,
                )

                # Generate mesh multiple times
                mesh1 = dice.generate_mesh(include_numbers=True)
                mesh2 = dice.generate_mesh(include_numbers=False)

                # Basic validation that meshes are different
                assert len(mesh1.vertices) != len(mesh2.vertices) or len(mesh1.faces) != len(mesh2.faces)

                # Export only occasionally to avoid too many files
                if i % 5 == 0:
                    output_path = Path(temp_dir) / f"memory_test_{i}.stl"
                    dice.export_stl(output_path, include_numbers=True)
                    assert output_path.exists()

                # Clear references to help garbage collection
                del dice, mesh1, mesh2

            # If we get here without memory errors, the test passes
