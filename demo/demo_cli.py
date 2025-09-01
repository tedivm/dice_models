#!/usr/bin/env python3
"""
CLI demonstration script for dice_models.

This script demonstrates all the CLI functionality of the dice_models library.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display the results."""
    print(f"\n{description}")
    print("=" * len(description))
    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Demonstrate all CLI features."""
    print("DICE MODELS CLI DEMONSTRATION")
    print("=" * 60)

    # Create output directory
    cli_output = Path("cli_demo_output")
    cli_output.mkdir(exist_ok=True)

    # Base command
    base_cmd = [sys.executable, "-m", "dice_models.cli"]

    # Demo 1: Help and version
    run_command(base_cmd + ["--help"], "Display CLI help")
    run_command(base_cmd + ["version"], "Display version")
    run_command(base_cmd + ["list-types"], "List supported dice types")

    # Demo 2: Basic generation
    run_command(
        base_cmd + ["generate", "6", str(cli_output / "cli_d6.stl")],
        "Generate a standard D6",
    )

    # Demo 3: Advanced generation
    run_command(
        base_cmd
        + [
            "generate",
            "20",
            str(cli_output / "cli_d20_custom.stl"),
            "--radius",
            "15.0",
            "--text-depth",
            "0.8",
            "--text-size",
            "4.0",
        ],
        "Generate D20 with custom parameters",
    )

    # Demo 4: Custom layout
    run_command(
        base_cmd
        + [
            "generate",
            "6",
            str(cli_output / "cli_d6_custom_layout.stl"),
            "--layout",
            "6,5,4,3,2,1",
        ],
        "Generate D6 with custom number layout",
    )

    # Demo 5: Font demonstrations
    common_fonts = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Geneva.ttf",
    ]

    # Find an available font
    available_font = None
    for font_path in common_fonts:
        if Path(font_path).exists():
            available_font = font_path
            break

    if available_font:
        run_command(
            base_cmd
            + [
                "generate",
                "6",
                str(cli_output / "cli_d6_custom_font.stl"),
                "--font",
                available_font,
                "--text-depth",
                "1.0",
                "--text-size",
                "4.0",
            ],
            f"Generate D6 with custom font: {Path(available_font).name}",
        )
    else:
        print("\nSkipping font demo - no common system fonts found")

    # Demo 6: Curve resolution for quality
    run_command(
        base_cmd
        + [
            "generate",
            "6",
            str(cli_output / "cli_d6_low_quality.stl"),
            "--curve-resolution",
            "low",
            "--layout",
            "8,8,8,8,8,8",  # All 8s to show curve quality
        ],
        "Generate D6 with low curve resolution (faster)",
    )

    run_command(
        base_cmd
        + [
            "generate",
            "6",
            str(cli_output / "cli_d6_highest_quality.stl"),
            "--curve-resolution",
            "highest",
            "--layout",
            "8,8,8,8,8,8",  # All 8s to show curve quality
        ],
        "Generate D6 with highest curve resolution (smoothest)",
    )

    # Demo 8: Blank dice
    run_command(
        base_cmd + ["generate", "8", str(cli_output / "cli_d8_blank.stl"), "--no-numbers"],
        "Generate blank D8 (no numbers)",
    )

    # Demo 9: Batch generation
    batch_config = {
        "dice": [
            {"sides": 4, "filename": "cli_batch_d4.stl", "radius": 8.0},
            {"sides": 6, "filename": "cli_batch_d6.stl", "radius": 10.0},
            {
                "sides": 20,
                "filename": "cli_batch_d20.stl",
                "radius": 15.0,
                "no_numbers": True,
            },
        ]
    }

    import json

    config_file = Path("cli_batch_config.json")
    with open(config_file, "w") as f:
        json.dump(batch_config, f, indent=2)

    run_command(
        base_cmd + ["batch-generate", str(config_file), "--output-dir", str(cli_output)],
        "Batch generate multiple dice",
    )

    # Show results
    print("\n" + "=" * 60)
    print("CLI DEMONSTRATION COMPLETE")
    print("=" * 60)

    if cli_output.exists():
        files = list(cli_output.glob("*.stl"))
        print(f"Generated {len(files)} STL files via CLI:")
        for file in sorted(files):
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name:25} ({size_kb:.1f} KB)")

    # Cleanup
    config_file.unlink()

    print("\nCLI demonstration complete!")


if __name__ == "__main__":
    main()
