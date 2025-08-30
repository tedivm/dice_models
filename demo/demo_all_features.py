#!/usr/bin/env python3
"""
Comprehensive demonstration of the dice_models library.

This script demonstrates all the key features of the dice_models library:
- Creating dice with different numbers of sides (4, 6, 8, 10, 12, 20)
- Customizing number placement (which number is on each face)
- Configuring text properties (depth, size)
- Font specification (when available)
- Both CLI and library usage
"""

import os
import sys
from pathlib import Path

# Add the package to the path so we can import it
sys.path.insert(0, str(Path(__file__).parent.parent))

from dice_models import DiceGeometry, PolyhedronType, create_standard_dice


def demo_basic_dice_creation():
    """Demonstrate creating standard dice with different numbers of sides."""
    print("=" * 60)
    print("DEMO 1: Basic Dice Creation")
    print("=" * 60)

    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create all standard dice types
    dice_types = [4, 6, 8, 10, 12, 20]

    for sides in dice_types:
        print(f"\nCreating D{sides}...")
        dice = create_standard_dice(
            sides=sides, radius=10.0, output_path=output_dir / f"standard_d{sides}.stl"
        )

        info = dice.get_info()
        print(f"  ✓ {info['type']} with {info['sides']} sides")
        print(f"    Radius: {info['radius']}mm")
        print(f"    Numbers: {info['number_layout']}")


def demo_custom_number_layouts():
    """Demonstrate custom number layouts."""
    print("\n" + "=" * 60)
    print("DEMO 2: Custom Number Layouts")
    print("=" * 60)

    output_dir = Path("demo_output")

    # D6 with opposite faces summing to 7 (traditional die layout)
    print("\nCreating D6 with traditional opposite-face layout...")
    d6_traditional = DiceGeometry(
        polyhedron_type=PolyhedronType.CUBE,
        radius=12.0,
        number_layout=[1, 2, 3, 6, 5, 4],  # Opposite faces: 1-6, 2-5, 3-4
        text_depth=0.6,
        text_size=3.5,
    )
    d6_traditional.export_stl(output_dir / "d6_traditional_layout.stl")
    print(f"  ✓ Layout: {d6_traditional.number_layout}")

    # D8 with reverse numbering
    print("\nCreating D8 with reverse numbering...")
    d8_reverse = DiceGeometry(
        polyhedron_type=PolyhedronType.OCTAHEDRON,
        radius=10.0,
        number_layout=[8, 7, 6, 5, 4, 3, 2, 1],
        text_depth=0.5,
        text_size=3.0,
    )
    d8_reverse.export_stl(output_dir / "d8_reverse_layout.stl")
    print(f"  ✓ Layout: {d8_reverse.number_layout}")

    # D20 with custom layout
    print("\nCreating D20 with custom layout...")
    custom_d20_layout = [
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
    ]
    d20_custom = DiceGeometry(
        polyhedron_type=PolyhedronType.ICOSAHEDRON,
        radius=15.0,
        number_layout=custom_d20_layout,
        text_depth=0.8,
        text_size=4.0,
    )
    d20_custom.export_stl(output_dir / "d20_custom_layout.stl")
    print(f"  ✓ Layout: First few numbers: {d20_custom.number_layout[:5]}...")


def demo_text_customization():
    """Demonstrate text size and depth customization."""
    print("\n" + "=" * 60)
    print("DEMO 3: Text Customization")
    print("=" * 60)

    output_dir = Path("demo_output")

    # Small, shallow text
    print("\nCreating D6 with small, shallow text...")
    d6_small = DiceGeometry(
        polyhedron_type=PolyhedronType.CUBE,
        radius=10.0,
        text_depth=0.2,  # Very shallow
        text_size=2.0,  # Small text
    )
    d6_small.export_stl(output_dir / "d6_small_text.stl")
    print(f"  ✓ Text depth: {d6_small.text_depth}mm, size: {d6_small.text_size}mm")

    # Large, deep text
    print("\nCreating D6 with large, deep text...")
    d6_large = DiceGeometry(
        polyhedron_type=PolyhedronType.CUBE,
        radius=15.0,
        text_depth=1.2,  # Very deep
        text_size=5.0,  # Large text
    )
    d6_large.export_stl(output_dir / "d6_large_text.stl")
    print(f"  ✓ Text depth: {d6_large.text_depth}mm, size: {d6_large.text_size}mm")


def demo_blank_dice():
    """Demonstrate creating dice without numbers."""
    print("\n" + "=" * 60)
    print("DEMO 4: Blank Dice (No Numbers)")
    print("=" * 60)

    output_dir = Path("demo_output")

    print("\nCreating blank dice for custom engraving...")

    for sides in [4, 6, 8, 10, 12, 20]:
        dice = create_standard_dice(sides=sides, radius=12.0)
        output_path = output_dir / f"blank_d{sides}.stl"
        dice.export_stl(output_path, include_numbers=False)
        print(f"  ✓ Blank D{sides} saved to {output_path.name}")


def demo_font_specification():
    """Demonstrate font path specification."""
    print("\n" + "=" * 60)
    print("DEMO 5: Font Specification")
    print("=" * 60)

    output_dir = Path("demo_output")

    # Test with system fonts (broader set for better compatibility)
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

    # Find multiple fonts for comparison
    available_fonts = []
    for path in potential_fonts:
        if os.path.exists(path):
            available_fonts.append(path)
            if len(available_fonts) >= 3:  # Limit to 3 for demo
                break

    if available_fonts:
        print(f"\nFound {len(available_fonts)} system fonts for demonstration:")

        for i, font_path in enumerate(available_fonts):
            font_name = Path(font_path).stem
            print(f"\nUsing font {i+1}: {font_name}")
            print(f"  Path: {font_path}")

            d6_font = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=12.0,
                font_path=font_path,
                text_depth=0.8,  # Deeper for better visibility
                text_size=4.0,  # Larger for better comparison
            )
            output_file = (
                output_dir / f"d6_font_{font_name.lower().replace(' ', '_')}.stl"
            )
            d6_font.export_stl(output_file)
            print(f"  ✓ D6 created: {output_file.name}")

        print("\n  Compare the different font styles in the output files!")

    else:
        print("\nNo system fonts found at expected locations.")
        print("Creating D6 with default font rendering...")
        d6_default = DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            radius=12.0,
            font_path=None,  # Use default font system
            text_depth=0.6,
            text_size=3.5,
        )
        d6_default.export_stl(output_dir / "d6_default_font.stl")
        print("  ✓ D6 created with default font rendering")

    # Demonstrate font parameters effect
    print("\nDemonstrating font parameter effects...")
    if available_fonts:
        font_path = available_fonts[0]

        # Different text depths
        for depth in [0.3, 0.8, 1.5]:
            d6_depth = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=10.0,
                font_path=font_path,
                text_depth=depth,
                text_size=3.0,
            )
            d6_depth.export_stl(output_dir / f"d6_depth_{depth:.1f}mm.stl")
            print(f"  ✓ D6 with {depth}mm text depth")

        # Different text sizes
        for size in [2.0, 4.0, 6.0]:
            d6_size = DiceGeometry(
                polyhedron_type=PolyhedronType.CUBE,
                radius=10.0,
                font_path=font_path,
                text_depth=0.6,
                text_size=size,
            )
            d6_size.export_stl(output_dir / f"d6_size_{size:.1f}mm.stl")
            print(f"  ✓ D6 with {size}mm text size")


def demo_batch_configuration():
    """Demonstrate batch generation configuration."""
    print("\n" + "=" * 60)
    print("DEMO 6: Batch Configuration")
    print("=" * 60)

    # Create a batch configuration
    batch_config = {
        "dice": [
            {
                "sides": 4,
                "filename": "set_d4.stl",
                "radius": 8.0,
                "text_depth": 0.4,
                "text_size": 2.5,
            },
            {
                "sides": 6,
                "filename": "set_d6.stl",
                "radius": 10.0,
                "text_depth": 0.5,
                "text_size": 3.0,
            },
            {
                "sides": 8,
                "filename": "set_d8.stl",
                "radius": 10.0,
                "text_depth": 0.5,
                "text_size": 3.0,
            },
            {
                "sides": 10,
                "filename": "set_d10.stl",
                "radius": 10.0,
                "text_depth": 0.5,
                "text_size": 3.0,
            },
            {
                "sides": 12,
                "filename": "set_d12.stl",
                "radius": 12.0,
                "text_depth": 0.6,
                "text_size": 3.5,
            },
            {
                "sides": 20,
                "filename": "set_d20.stl",
                "radius": 15.0,
                "text_depth": 0.8,
                "text_size": 4.0,
            },
        ]
    }

    import json

    config_path = Path("demo_batch_config.json")
    with open(config_path, "w") as f:
        json.dump(batch_config, f, indent=2)

    print(f"\nCreated batch configuration: {config_path}")
    print("This can be used with the CLI command:")
    print(f"  dice_models batch-generate {config_path} --output-dir dice_set")


def demo_info_and_validation():
    """Demonstrate getting dice information and validation."""
    print("\n" + "=" * 60)
    print("DEMO 7: Information and Validation")
    print("=" * 60)

    # Create a dice and show all its information
    print("\nCreating D12 and displaying full information...")
    d12 = DiceGeometry(
        polyhedron_type=PolyhedronType.DODECAHEDRON,
        radius=14.0,
        text_depth=0.7,
        text_size=3.8,
    )

    info = d12.get_info()
    print(f"  Type: {info['type']}")
    print(f"  Sides: {info['sides']}")
    print(f"  Radius: {info['radius']}mm")
    print(f"  Number layout: {info['number_layout']}")
    print(f"  Vertex count: {info['vertex_count']}")
    print(f"  Face count: {info['face_count']}")
    print(f"  Text depth: {info['text_depth']}mm")
    print(f"  Text size: {info['text_size']}mm")

    # Test validation
    print("\nTesting validation...")
    try:
        # This should fail - wrong number of numbers for a D6
        DiceGeometry(
            polyhedron_type=PolyhedronType.CUBE,
            number_layout=[1, 2, 3],  # Only 3 numbers for 6-sided die
        )
    except ValueError as e:
        print(f"  ✓ Validation works: {e}")


def main():
    """Run all demonstrations."""
    print("DICE MODELS LIBRARY DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates all the key features of the dice_models library.")
    print("STL files will be created in the 'demo_output' directory.")

    try:
        demo_basic_dice_creation()
        demo_custom_number_layouts()
        demo_text_customization()
        demo_blank_dice()
        demo_font_specification()
        demo_batch_configuration()
        demo_info_and_validation()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All STL files have been created in the 'demo_output' directory.")
        print("You can now 3D print these files to create master dice for mold making!")

        # Show file listing
        output_dir = Path("demo_output")
        if output_dir.exists():
            files = list(output_dir.glob("*.stl"))
            print(f"\nGenerated {len(files)} STL files:")
            for file in sorted(files):
                size_kb = file.stat().st_size / 1024
                print(f"  {file.name:25} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
