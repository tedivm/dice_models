#!/usr/bin/env python3
"""
Comprehensive demonstration of the dice_models library.

This refactored demo system demonstrates all key features of the dice_models library
with a clean CLI interface and organized output structure.

The demo uses the library's default settings whenever possible, allowing it to automatically
stay up-to-date with any changes to the library defaults. Custom settings are only specified
when they are the focus of a particular demonstration.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import typer

# Add the package to the path so we can import it
sys.path.insert(0, str(Path(__file__).parent.parent))

from dice_models import create_standard_dice

# Define the standard dice set as a constant for easy updates
STANDARD_DICE_SIDES: List[int] = [4, 6, 8, 10, 12, 20]

app = typer.Typer(
    help="Dice Models Demo System - Generate comprehensive demonstrations of all features using parallel processing for faster generation."
)

# No default settings defined here - we use the library's defaults
# This way if the library defaults change, the demo updates automatically


def create_single_dice_worker(args: Tuple[int, Path, dict]) -> Tuple[int, str, float]:
    """
    Worker function to create a single dice in a separate process.

    Args:
        args: Tuple of (sides, output_path, settings)

    Returns:
        Tuple of (sides, filename, size_kb)
    """
    sides, output_path, settings = args

    # Import here to ensure clean process state
    from dice_models import create_standard_dice

    # Create dice with specified settings
    dice = create_standard_dice(sides=sides, **settings)

    # Export to specified path
    dice.export_stl(output_path)

    # Get size info for reporting
    size_kb = output_path.stat().st_size / 1024 if output_path.exists() else 0.0

    return sides, output_path.name, size_kb


def create_dice_set(output_dir: Path, demo_name: str, custom_settings: dict = None) -> None:
    """
    Create a complete set of dice for a demo with specified custom settings.
    Uses multiprocessing to create all dice in parallel for faster generation.

    Args:
        output_dir: Base output directory (demo_output)
        demo_name: Name of the demo (used for subdirectory)
        custom_settings: Dictionary of settings that override library defaults
    """
    demo_dir = output_dir / demo_name
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Use custom settings if provided, otherwise let the library use its defaults
    settings = custom_settings if custom_settings else {}

    print(f"\nGenerating {demo_name} dice set...")
    start_time = time.time()

    # Prepare arguments for parallel processing
    worker_args = []
    for sides in STANDARD_DICE_SIDES:
        output_path = demo_dir / f"d{sides}.stl"
        worker_args.append((sides, output_path, settings))

    # Use ProcessPoolExecutor to create dice in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all dice creation tasks
        future_to_sides = {executor.submit(create_single_dice_worker, args): args[0] for args in worker_args}

        # Collect results as they complete
        results = []
        for future in as_completed(future_to_sides):
            try:
                sides, filename, size_kb = future.result()
                results.append((sides, filename, size_kb))
                print(f"  ✓ D{sides} completed")
            except Exception as e:
                sides = future_to_sides[future]
                print(f"  ✗ D{sides} failed: {e}")

        # Sort results by dice sides for consistent output
        results.sort(key=lambda x: x[0])

        # Calculate and display timing information
        elapsed_time = time.time() - start_time
        dice_count = len(results)

        # Display final results with timing
        print(f"  Generated {dice_count} dice in {elapsed_time:.1f}s (parallel processing)")
        for sides, filename, size_kb in results:
            print(f"    D{sides}: {filename} ({size_kb:.1f} KB)")


def demo_basic_dice_creation() -> None:
    """Demonstrate creating standard dice using the library's default settings."""
    print("=" * 60)
    print("DEMO: Basic Dice Creation")
    print("=" * 60)
    print("Creates standard dice using the library's default settings.")
    print("Each dice in the set is generated in parallel for faster completion.")

    output_dir = Path("demo_output")

    # Use highest curve resolution for the basic demo
    create_dice_set(output_dir, "basic_standard", {"curve_resolution": "highest"})


def demo_custom_number_layouts() -> None:
    """Demonstrate custom number layouts."""
    print("=" * 60)
    print("DEMO: Custom Number Layouts")
    print("=" * 60)
    print("Creates dice sets with reverse and custom number arrangements.")

    output_dir = Path("demo_output")

    # Reverse numbering demo
    print("\nReverse numbering (highest to lowest):")
    demo_dir = output_dir / "layout_custom"
    demo_dir.mkdir(parents=True, exist_ok=True)

    for sides in STANDARD_DICE_SIDES:
        if sides == 10:
            # D10 uses 0-9, so reverse is [9,8,7,6,5,4,3,2,1,0]
            reverse_layout = list(range(9, -1, -1))
        else:
            # Regular dice use 1-N, so reverse is [N, N-1, ..., 1]
            reverse_layout = list(range(sides, 0, -1))

        dice = create_standard_dice(sides=sides, number_layout=reverse_layout)
        dice.export_stl(demo_dir / f"d{sides}.stl")

        print(f"  D{sides} reverse: {reverse_layout[:5]}{'...' if len(reverse_layout) > 5 else ''}")


def demo_text_customization() -> None:
    """Demonstrate text size and depth customization."""
    print("=" * 60)
    print("DEMO: Text Customization")
    print("=" * 60)
    print("Creates dice sets with different text sizes and engraving depths.")

    output_dir = Path("demo_output")

    # Small, shallow text
    print("\nSmall, shallow text:")
    create_dice_set(output_dir, "text_small_shallow", {"text_depth": 0.3, "text_size": 2.0})

    # Large, deep text
    print("\nLarge, deep text:")
    create_dice_set(
        output_dir,
        "text_large_deep",
        {
            "text_depth": 1.2,
            "text_size": 5.0,
            "radius": 12.0,  # Larger radius to accommodate larger text
        },
    )

    # Different depth variations
    print("\nText depth variations:")
    for depth in [0.3, 0.8, 1.5]:
        create_dice_set(output_dir, f"text_depth_{depth:.1f}mm", {"text_depth": depth})

    # Different size variations
    print("\nText size variations:")
    for size in [2.0, 4.0, 6.0]:
        create_dice_set(
            output_dir,
            f"text_size_{size:.1f}mm",
            {
                "text_size": size,
                "radius": 12.0 if size > 4.0 else 10.0,  # Larger radius for larger text
            },
        )


def demo_blank_dice() -> None:
    """Demonstrate creating dice without numbers."""
    print("=" * 60)
    print("DEMO: Blank Dice (No Numbers)")
    print("=" * 60)
    print("Creates dice without any numbers for custom engraving.")

    output_dir = Path("demo_output")
    demo_dir = output_dir / "blank_dice"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating blank dice for custom engraving...")

    for sides in STANDARD_DICE_SIDES:
        dice = create_standard_dice(sides=sides)
        output_path = demo_dir / f"d{sides}.stl"
        dice.export_stl(output_path, include_numbers=False)

        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  ✓ D{sides} blank dice ({size_kb:.1f} KB)")


def find_available_fonts() -> List[str]:
    """Find available system fonts for demonstration."""
    potential_fonts = [
        # macOS fonts
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        # Linux fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows fonts
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]

    available_fonts = []
    for path in potential_fonts:
        if os.path.exists(path):
            available_fonts.append(path)

    return available_fonts


def demo_font_specification() -> None:
    """Demonstrate font path specification."""
    print("=" * 60)
    print("DEMO: Font Specification")
    print("=" * 60)
    print("Creates dice sets with different font files.")

    output_dir = Path("demo_output")
    available_fonts = find_available_fonts()

    if available_fonts:
        print(f"\nFound {len(available_fonts)} system fonts for demonstration:")

        # Limit to first 3 fonts to keep demo manageable
        fonts_to_demo = available_fonts[:3]

        for i, font_path in enumerate(fonts_to_demo):
            font_name = Path(font_path).stem.lower().replace(" ", "_")
            print(f"\nFont {i + 1}: {Path(font_path).stem}")
            print(f"  Path: {font_path}")

            create_dice_set(
                output_dir,
                f"font_{font_name}",
                {
                    "font_path": font_path,
                    "text_depth": 0.8,  # Deeper for better visibility
                    "text_size": 4.0,  # Larger for better comparison
                },
            )

    else:
        print("\nNo system fonts found at expected locations.")
        print("Creating dice set with default font rendering...")
        create_dice_set(
            output_dir,
            "font_default",
            {
                "font_path": None,
                "text_depth": 0.8,
                "text_size": 4.0,
            },
        )


def demo_curve_resolution() -> None:
    """Demonstrate different curve resolution settings for font quality."""
    print("=" * 60)
    print("DEMO: Curve Resolution Quality")
    print("=" * 60)
    print("Creates dice sets with different curve resolution settings.")
    print("Characters with curves (like 8, 0, 6, 9) will show quality differences.")

    output_dir = Path("demo_output")

    # Different curve resolutions
    resolutions = [
        ("low", "Low quality (fast)"),
        ("medium", "Medium quality (balanced)"),
        ("high", "High quality (standard)"),
        ("highest", "Highest quality (smooth curves)"),
    ]

    for resolution, description in resolutions:
        print(f"\nCreating dice set with {resolution} resolution...")
        print(f"  {description}")

        create_dice_set(
            output_dir,
            f"curve_{resolution}",
            {
                "curve_resolution": resolution,
                "text_size": 4.0,  # Larger text to show curve quality better
                "text_depth": 1.0,
            },
        )


def demo_layout_types() -> None:
    """Demonstrate different layout types (from demo_layout_types.py)."""
    print("=" * 60)
    print("DEMO: Layout Types")
    print("=" * 60)
    print("Creates dice sets with different number arrangement strategies.")

    output_dir = Path("demo_output")

    # Import LayoutType here to avoid import issues
    from dice_models.geometry.layouts import LayoutType

    layout_types = [
        (LayoutType.NAIVE, "naive", "Simple sequential placement"),
        (
            LayoutType.OPPOSING_BALANCED,
            "opposing_balanced",
            "Opposing faces sum consistently, balanced distribution",
        ),
        (
            LayoutType.OPPOSING_WEIGHTED,
            "opposing_weighted",
            "Opposing faces sum consistently, high numbers clustered",
        ),
    ]

    for layout_type, name, description in layout_types:
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  {description}")

        create_dice_set(output_dir, f"layout_{name}", {"layout_type": layout_type})


def demo_batch_configuration() -> None:
    """Demonstrate batch generation configuration."""
    print("=" * 60)
    print("DEMO: Batch Configuration")
    print("=" * 60)
    print("Creates a sample batch configuration file for reference.")

    # Create a batch configuration example
    batch_config = {"dice": []}

    # Add configuration for each dice type
    for sides in STANDARD_DICE_SIDES:
        dice_config = {
            "sides": sides,
            "filename": f"batch_d{sides}.stl",
            "radius": 16.0,  # Library default
            "text_depth": 0.5,  # Library default
            "text_size": 6.0,  # Library default
        }

        # Customize some dice for variety
        if sides == 4:
            dice_config.update({"radius": 8.0, "text_size": 2.5})
        elif sides == 12:
            dice_config.update({"text_depth": 0.8, "text_size": 4.0})
        elif sides == 20:
            dice_config.update({"radius": 15.0, "text_depth": 1.0, "text_size": 4.5})

        batch_config["dice"].append(dice_config)

    import json

    config_path = Path("demo_output") / "sample_batch_config.json"
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(batch_config, f, indent=2)

    print(f"\nCreated sample batch configuration: {config_path}")
    print("This can be used with the CLI command:")
    print("  dice_models batch-generate sample_batch_config.json --output-dir dice_set")


# CLI Commands using Typer
@app.command(help="Run all available demos.")
def all() -> None:
    """Run all demonstrations."""
    print("DICE MODELS LIBRARY DEMONSTRATION")
    print("=" * 60)
    print("Running all demos. STL files will be created in organized subdirectories.")
    print("within the 'demo_output' directory.")
    print("Using parallel processing for faster generation...")

    start_time = time.time()

    try:
        demo_basic_dice_creation()
        demo_custom_number_layouts()
        demo_text_customization()
        demo_blank_dice()
        demo_font_specification()
        demo_curve_resolution()
        demo_layout_types()
        demo_batch_configuration()

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("=" * 60)
        print("All STL files have been created in organized subdirectories")
        print("within the 'demo_output' directory.")
        print(f"Total generation time: {elapsed_time:.1f} seconds (with parallel processing)")

        # Show summary
        output_dir = Path("demo_output")
        if output_dir.exists():
            subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
            total_files = sum(len(list(subdir.glob("*.stl"))) for subdir in subdirs)

            print(f"\nGenerated {total_files} STL files across {len(subdirs)} demo categories:")
            for subdir in sorted(subdirs):
                files = list(subdir.glob("*.stl"))
                print(f"  {subdir.name}: {len(files)} files")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


@app.command(help="Generate basic standard dice.")
def basic() -> None:
    """Generate basic standard dice set."""
    demo_basic_dice_creation()


@app.command(help="Generate dice with custom number layouts.")
def layouts() -> None:
    """Generate dice with custom number layouts."""
    demo_custom_number_layouts()


@app.command(help="Generate dice with different text customizations.")
def text() -> None:
    """Generate dice with text customizations."""
    demo_text_customization()


@app.command(help="Generate blank dice without numbers.")
def blank() -> None:
    """Generate blank dice without numbers."""
    demo_blank_dice()


@app.command(help="Generate dice with different fonts.")
def fonts() -> None:
    """Generate dice with different font specifications."""
    demo_font_specification()


@app.command(help="Generate dice with different curve resolutions.")
def curves() -> None:
    """Generate dice with different curve resolution settings."""
    demo_curve_resolution()


@app.command(help="Generate dice with different layout types.")
def layout_types() -> None:
    """Generate dice with different layout types."""
    demo_layout_types()


@app.command(help="Create sample batch configuration file.")
def batch_config() -> None:
    """Create sample batch configuration file."""
    demo_batch_configuration()


@app.command(help="Clean up all demo output files.")
def clean() -> None:
    """Remove all generated demo files."""
    import shutil

    output_dir = Path("demo_output")
    if output_dir.exists():
        print(f"Removing {output_dir} and all contents...")
        shutil.rmtree(output_dir)
        print("✓ Demo output cleaned.")
    else:
        print("No demo output directory found.")


def main() -> None:
    """Main entry point for the demo system."""
    app()


if __name__ == "__main__":
    main()
