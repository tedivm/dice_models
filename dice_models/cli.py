import asyncio
import logging
from functools import wraps
from pathlib import Path
from typing import Optional

import typer

from .geometry import PolyhedronType
from .geometry.dice import create_standard_dice
from .settings import settings

logger = logging.getLogger(__name__)

app = typer.Typer()


def syncify(f):
    """This simple decorator converts an async function into a sync function,
    allowing it to work with Typer.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.command(help=f"Display the current installed version of {settings.project_name}.")
def version():
    from . import __version__

    typer.echo(f"{settings.project_name} - {__version__}")


@app.command(help="Generate a standard dice model and export to STL.")
def generate(
    sides: int = typer.Argument(..., help="Number of sides (4, 6, 8, 10, 12, or 20)"),
    output: str = typer.Argument(..., help="Output STL file path"),
    radius: float = typer.Option(
        10.0, "--radius", "-r", help="Radius of the dice in mm"
    ),
    font_path: Optional[str] = typer.Option(
        None, "--font", "-f", help="Path to TTF font file"
    ),
    text_depth: float = typer.Option(
        0.5, "--text-depth", help="Depth of number engraving in mm"
    ),
    text_size: float = typer.Option(3.0, "--text-size", help="Size of numbers in mm"),
    curve_resolution: str = typer.Option(
        "high",
        "--curve-resolution",
        "-q",
        help="Curve quality: 'low', 'medium', 'high', 'highest', or integer",
    ),
    no_numbers: bool = typer.Option(
        False, "--no-numbers", help="Generate dice without numbers"
    ),
    custom_layout: Optional[str] = typer.Option(
        None,
        "--layout",
        help="Custom number layout (comma-separated, e.g., '1,3,5,2,4,6')",
    ),
) -> None:
    """Generate a dice model with the specified parameters."""
    try:
        # Parse custom layout if provided
        number_layout = _parse_custom_layout(custom_layout)

        # Parse curve resolution - try as integer first, then as string
        parsed_curve_resolution: str | int
        try:
            parsed_curve_resolution = int(curve_resolution)
        except ValueError:
            parsed_curve_resolution = curve_resolution

        # Create the dice
        dice = create_standard_dice(
            sides=sides,
            radius=radius,
            number_layout=number_layout,
            font_path=font_path,
            text_depth=text_depth,
            text_size=text_size,
            curve_resolution=parsed_curve_resolution,
        )

        # Export to STL
        dice.export_stl(output_path=output, include_numbers=not no_numbers)

        # Display generation info
        _display_generation_info(dice, output, no_numbers)

    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}")
        logger.exception("Failed to generate dice")
        raise typer.Exit(1)


def _parse_custom_layout(custom_layout: Optional[str]) -> Optional[list[int]]:
    """
    Parse custom number layout string into list of integers.

    Args:
        custom_layout: Comma-separated string of numbers

    Returns:
        List of integers or None if no layout provided

    Raises:
        ValueError: If layout format is invalid
    """
    if not custom_layout:
        return None

    try:
        return [int(x.strip()) for x in custom_layout.split(",")]
    except ValueError:
        raise ValueError("Invalid number layout format. Use comma-separated integers.")


def _display_generation_info(dice, output: str, no_numbers: bool) -> None:
    """
    Display information about the generated dice.

    Args:
        dice: The DiceGeometry object that was created
        output: Output file path
        no_numbers: Whether numbers were included
    """
    info = dice.get_info()
    typer.echo(f"Generated {info['type']} dice:")
    typer.echo(f"  Sides: {info['sides']}")
    typer.echo(f"  Radius: {info['radius']}mm")
    if info.get("font_path"):
        font_name = Path(info["font_path"]).name
        typer.echo(f"  Font: {font_name}")
    typer.echo(f"  Text depth: {info['text_depth']}mm")
    typer.echo(f"  Text size: {info['text_size']}mm")
    typer.echo(f"  Numbers: {info['number_layout'] if not no_numbers else 'None'}")
    typer.echo(f"  Output: {output}")


@app.command(help="List supported dice types and their properties.")
def list_types() -> None:
    """List all supported dice types."""
    typer.echo("Supported dice types:")
    typer.echo()

    for poly_type in PolyhedronType:
        typer.echo(f"  {poly_type.value:2d} sides - {poly_type.name}")

    typer.echo()
    typer.echo("Standard number layouts:")

    for poly_type in PolyhedronType:
        from .geometry.polyhedra import get_standard_number_layout

        layout = get_standard_number_layout(poly_type)
        typer.echo(f"  D{poly_type.value:2d}: {layout}")


@app.command(help="Generate multiple dice models.")
def batch_generate(
    config_file: str = typer.Argument(
        ..., help="Path to configuration file (JSON/YAML)"
    ),
    output_dir: str = typer.Option(
        "./dice_output", "--output-dir", "-o", help="Output directory"
    ),
    radius: float = typer.Option(10.0, "--radius", "-r", help="Default radius in mm"),
) -> None:
    """Generate multiple dice from a configuration file."""
    try:
        import json
        from pathlib import Path

        config_path = Path(config_file)
        if not config_path.exists():
            typer.echo(f"Error: Configuration file '{config_file}' not found.")
            raise typer.Exit(1)

        # Load configuration
        with open(config_path) as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    config = yaml.safe_load(f)
                except ImportError:
                    typer.echo(
                        "Error: PyYAML not installed. Install with 'pip install pyyaml'"
                    )
                    raise typer.Exit(1)
            else:
                config = json.load(f)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate dice for each configuration
        for dice_config in config.get("dice", []):
            sides = dice_config["sides"]
            filename = dice_config.get("filename", f"d{sides}.stl")
            dice_radius = dice_config.get("radius", radius)

            output_file = output_path / filename

            typer.echo(f"Generating D{sides}...")

            dice = create_standard_dice(
                sides=sides,
                radius=dice_radius,
                number_layout=dice_config.get("number_layout"),
                font_path=dice_config.get("font_path"),
                text_depth=dice_config.get("text_depth", 0.5),
                text_size=dice_config.get("text_size", 3.0),
            )

            dice.export_stl(
                output_path=output_file,
                include_numbers=not dice_config.get("no_numbers", False),
            )

            typer.echo(f"  Saved to {output_file}")

        typer.echo(f"\nBatch generation complete. Files saved to {output_dir}")

    except Exception as e:
        typer.echo(f"Error: {e}")
        logger.exception("Batch generation failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
