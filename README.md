# Dice Models

A sophisticated Python library for generating high-quality 3D dice models with font-based number engraving. Designed for 3D printing, mold creation, and custom gaming dice production.

## Development Status

This project is still under development and there are a few active bugs that need to be resolved before it will be ready for use:

- The d4 acts like other dice, with one number centered per face, which does not work.
- The default number placement isn't "balanced" yet.

## ✨ Features

### Core Capabilities

- **Complete RPG Dice Set**: D4, D6, D8, D10, D12, and D20 polyhedra
- **Font-Based Text Engraving**: Uses actual TTF fonts for crisp, readable numbers and characters
- **Customizable Layouts**: Control which number appears on each face
- **Precision Geometry**: Mathematically accurate polyhedra with configurable dimensions
- **STL Export**: High-quality mesh output ready for 3D printing

### Advanced Options

- **Custom Fonts**: Support for any TTF font file with automatic system font detection
- **Configurable Text**: Adjustable depth, size, and positioning for optimal readability
- **Curve Resolution**: High-quality curve rendering with configurable smoothness levels
- **Blank Dice**: Generate dice without numbers for custom applications
- **Batch Processing**: Create multiple dice with different parameters from configuration files
- **Multiple Interfaces**: Command-line tool and Python API

## 🚀 Quick Start

### Installation

```bash
pip install dice_models
```

### Command Line Usage

Generate a standard D6 die:

```bash
python -m dice_models.cli generate 6 my_d6.stl
```

Create a large D20 with custom font and highest quality curves:

```bash
python -m dice_models.cli generate 20 d20.stl \
    --radius 15.0 \
    --font "/System/Library/Fonts/Arial.ttf" \
    --text-depth 1.0 \
    --text-size 4.0 \
    --curve-resolution highest
```

Generate a blank die (no numbers):

```bash
python -m dice_models.cli generate 12 blank_d12.stl --no-numbers
```

### Python API

```python
from dice_models import create_standard_dice, DiceGeometry, PolyhedronType

# Quick generation with high-quality curves
d6 = create_standard_dice(6, radius=10.0, curve_resolution="highest", output_path="d6.stl")

# Advanced customization
from dice_models.geometry.dice import DiceGeometry
from dice_models.geometry.polyhedra import PolyhedronType

dice = DiceGeometry(
    polyhedron_type=PolyhedronType.ICOSAHEDRON,  # D20
    radius=12.0,
    font_path="/path/to/custom/font.ttf",
    text_depth=0.8,
    text_size=3.5,
    curve_resolution="highest",  # Smoothest curves for curved characters
    number_layout=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
)

# Generate and export
dice.export_stl("custom_d20.stl")

# Access mesh data
mesh = dice.generate_mesh()
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
```

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `generate` | Create a single die with specified parameters |
| `list-types` | Show all supported dice types and their properties |
| `batch-generate` | Generate multiple dice from a JSON configuration file |
| `version` | Display the installed version |

### Generate Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--radius -r` | Dice radius in mm | 10.0 |
| `--font -f` | Path to TTF font file | System default |
| `--text-depth` | Number engraving depth in mm | 0.5 |
| `--text-size` | Number size in mm | 3.0 |
| `--curve-resolution -q` | Curve quality: "low", "medium", "high", "highest", or integer | high |
| `--no-numbers` | Generate blank die | False |
| `--layout` | Custom number layout (comma-separated) | Standard layout |

## 🎲 Supported Dice Types

| Sides | Polyhedron | Shape Description |
|-------|------------|-------------------|
| D4 | Tetrahedron | 4-sided pyramid |
| D6 | Cube | Standard 6-sided cube |
| D8 | Octahedron | 8-sided double pyramid |
| D10 | Pentagonal Trapezohedron | 10-sided with quadrilateral faces |
| D12 | Dodecahedron | 12-sided with pentagonal faces |
| D20 | Icosahedron | 20-sided with triangular faces |

## 🎨 Curve Resolution Quality

The library supports configurable curve resolution for optimal font rendering quality:

### Resolution Levels

| Level | Points per Curve | Use Case | Performance |
|-------|------------------|----------|-------------|
| `low` | 5 | Fast prototyping, testing | Fastest |
| `medium` | 10 | General use, good balance | Balanced |
| `high` | 20 | Production quality (default) | Good |
| `highest` | 50 | Maximum smoothness | Slower |

### When to Use Different Resolutions

- **Low/Medium**: Basic geometry tests, rapid prototyping
- **High**: Production dice with standard quality (recommended)
- **Highest**: Characters with curves (0, 6, 8, 9, O, D, B) for premium quality

### Examples

```bash
# Fast prototyping
python -m dice_models.cli generate 6 prototype.stl --curve-resolution low

# Production quality
python -m dice_models.cli generate 6 standard.stl --curve-resolution high

# Premium quality for curved characters
python -m dice_models.cli generate 6 premium.stl --curve-resolution highest --layout "8,6,9,0,8,6"
```

```python
# API usage
from dice_models import create_standard_dice

# Different quality levels
fast_dice = create_standard_dice(6, curve_resolution="low")
standard_dice = create_standard_dice(6, curve_resolution="high")
premium_dice = create_standard_dice(6, curve_resolution="highest")
```

## 🔧 Advanced Usage

### Custom Number Layouts

```python
# Traditional D6 layout (opposite faces sum to 7)
d6 = DiceGeometry(
    polyhedron_type=PolyhedronType.CUBE,
    number_layout=[1, 2, 3, 6, 5, 4]  # Specific face arrangement
)
```

### Batch Generation

Create a JSON configuration file:

```json
{
  "output_directory": "dice_collection",
  "dice": [
    {
      "sides": 6,
      "filename": "standard_d6.stl",
      "radius": 10.0,
      "text_depth": 0.5
    },
    {
      "sides": 20,
      "filename": "large_d20.stl",
      "radius": 15.0,
      "font_path": "/path/to/bold_font.ttf",
      "text_size": 4.0
    }
  ]
}
```

Then generate:

```bash
python -m dice_models.cli batch-generate config.json
```

### Font Selection

The library automatically detects system fonts:

**macOS**: Arial, Geneva, Times New Roman
**Linux**: DejaVu Sans, Liberation Sans
**Windows**: Arial, Calibri, Times New Roman

Or specify custom fonts:

```bash
python -m dice_models.cli generate 12 d12.stl --font "/custom/path/font.ttf"
```

## 🧪 Demo and Examples

The `demo/` directory contains comprehensive examples:

- **`demo_all_features.py`**: Showcases all library capabilities
- **`demo_cli.py`**: Command-line interface demonstrations
- **Font comparisons**: Multiple fonts on identical dice
- **Parameter effects**: Different sizes, depths, and layouts

Run demos:

```bash
cd demo
python demo_all_features.py
python demo_cli.py
```

## 🔬 Technical Details

### Geometry Engine

- **Precision Polyhedra**: Mathematically accurate vertex and face generation
- **Font Rendering**: TTF font parsing with vector-to-mesh conversion using proper Bézier curve mathematics
- **Curve Quality**: Configurable resolution from fast (5 points) to ultra-smooth (50+ points per curve)
- **Mesh Operations**: Subdivision, boolean operations, and mesh cleaning
- **Quality Assurance**: Watertight mesh validation and normal correction

### Dependencies

- **Core**: `numpy`, `trimesh`, `fonttools`
- **CLI**: `typer` for command-line interface
- **Export**: `numpy-stl` for STL file generation
- **Optional**: `fastapi` for web interface components

### File Formats

- **Input**: TTF/OTF font files, JSON configuration files
- **Output**: STL files compatible with all 3D printing software

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```text
dice_models/
├── geometry/          # Core geometric algorithms
│   ├── dice.py       # Main dice generation class with curve resolution support
│   ├── polyhedra.py  # Polyhedron definitions and utilities
│   └── text.py       # Font-based text engraving with Bézier curve mathematics
├── cli.py            # Command-line interface
└── settings.py       # Configuration management

demo/                 # Comprehensive examples and demonstrations
tests/                # Full test suite (79+ tests)
docs/                 # Documentation and development guides
```

## 📊 Quality Assurance

- **Comprehensive Testing**: 79+ tests covering all functionality including curve rendering
- **Geometric Validation**: Mesh integrity and mathematical accuracy
- **Cross-Platform**: Tested on macOS, Linux, and Windows
- **Font Compatibility**: Robust handling of various TTF fonts with proper curve support
- **Performance Optimized**: Configurable quality levels for development vs. production
- **3D Print Ready**: Generated meshes validated for manufacturability

## 🤝 Use Cases

- **3D Printing**: Direct printing of custom gaming dice
- **Mold Creation**: Master dice for silicone mold production
- **Game Development**: Digital dice assets with custom branding
- **Educational**: Teaching polyhedron geometry and 3D modeling
- **Prototyping**: Rapid iteration of dice designs

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

**Ready to create your perfect dice?** Start with the quick start guide above or explore the comprehensive demos to see all the possibilities!
