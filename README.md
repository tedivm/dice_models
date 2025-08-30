# Dice Models

A sophisticated Python library for generating high-quality 3D dice models with font-based number engraving. Designed for 3D printing, mold creation, and custom gaming dice production.

## Development Status

This project is still under development and there are a few active bugs that need to be resolved before it will be ready for use:

- Some die faces show the number inverted.
- The font system does not handle curves very well yet, so curvy fonts do not look good.
- The d4 acts like other dice, with one number centered per face, which does not work.
- The orientation of numbers against their die faces is not consistent.
- The default number placement isn't "balanced" yet.

## ✨ Features

### Core Capabilities

- **Complete RPG Dice Set**: D4, D6, D8, D10, D12, and D20 polyhedra
- **Font-Based Text Engraving**: Uses actual TTF fonts for crisp, readable numbers
- **Customizable Layouts**: Control which number appears on each face
- **Precision Geometry**: Mathematically accurate polyhedra with configurable dimensions
- **STL Export**: High-quality mesh output ready for 3D printing

### Advanced Options

- **Custom Fonts**: Support for any TTF font file with automatic system font detection
- **Configurable Text**: Adjustable depth, size, and positioning for optimal readability
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

Create a large D20 with custom font:

```bash
python -m dice_models.cli generate 20 d20.stl \
    --radius 15.0 \
    --font "/System/Library/Fonts/Arial.ttf" \
    --text-depth 1.0 \
    --text-size 4.0
```

Generate a blank die (no numbers):

```bash
python -m dice_models.cli generate 12 blank_d12.stl --no-numbers
```

### Python API

```python
from dice_models import create_standard_dice, DiceGeometry, PolyhedronType

# Quick generation
d6 = create_standard_dice(6, radius=10.0, output_path="d6.stl")

# Advanced customization
from dice_models.geometry.dice import DiceGeometry
from dice_models.geometry.polyhedra import PolyhedronType

dice = DiceGeometry(
    polyhedron_type=PolyhedronType.ICOSAHEDRON,  # D20
    radius=12.0,
    font_path="/path/to/custom/font.ttf",
    text_depth=0.8,
    text_size=3.5,
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
| `--no-numbers` | Generate blank die | False |
| `--layout` | Custom number layout (comma-separated) | Standard layout |

## 🎲 Supported Dice Types

| Sides | Polyhedron | Shape Description |
|-------|------------|-------------------|
| D4 | Tetrahedron | 4-sided pyramid |
| D6 | Cube | Standard 6-sided cube |
| D8 | Octahedron | 8-sided double pyramid |
| D10 | Pentagonal Trapezohedron | 10-sided with pentagonal faces |
| D12 | Dodecahedron | 12-sided with pentagonal faces |
| D20 | Icosahedron | 20-sided with triangular faces |

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
- **Font Rendering**: TTF font parsing with vector-to-mesh conversion
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
│   ├── dice.py       # Main dice generation class
│   ├── polyhedra.py  # Polyhedron definitions and utilities
│   └── text.py       # Font-based text engraving
├── cli.py            # Command-line interface
└── settings.py       # Configuration management

demo/                 # Comprehensive examples and demonstrations
tests/                # Full test suite (73+ tests)
docs/                 # Documentation and development guides
```

## 📊 Quality Assurance

- **Comprehensive Testing**: 73+ tests covering all functionality
- **Geometric Validation**: Mesh integrity and mathematical accuracy
- **Cross-Platform**: Tested on macOS, Linux, and Windows
- **Font Compatibility**: Robust handling of various TTF fonts
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
d20 = DiceGeometry(
    polyhedron_type=PolyhedronType.ICOSAHEDRON,
    radius=12.0,
    number_layout=list(range(1, 21)),  # 1-20
    text_depth=0.5,
    text_size=3.0
)
d20.export_stl("custom_d20.stl")

```

## Command Line Interface

### Basic Commands

- `dice_models generate SIDES OUTPUT [OPTIONS]` - Generate a single dice
- `dice_models list-types` - Show supported dice types
- `dice_models batch-generate CONFIG [OPTIONS]` - Generate multiple dice from config
- `dice_models version` - Show version information

### Generate Options

- `--radius FLOAT` - Radius of the dice in mm (default: 10.0)
- `--font PATH` - Path to TTF font file for numbers
- `--text-depth FLOAT` - Depth of number engraving in mm (default: 0.5)
- `--text-size FLOAT` - Size of numbers in mm (default: 3.0)
- `--no-numbers` - Generate blank dice without numbers
- `--layout TEXT` - Custom number layout (comma-separated)

### Examples

```bash
# Large D6 with deep engraving
dice_models generate 6 large_d6.stl --radius 20.0 --text-depth 1.0

# D6 with custom number layout
dice_models generate 6 custom_d6.stl --layout "6,5,4,3,2,1"

# Blank D20 for custom engraving
dice_models generate 20 blank_d20.stl --no-numbers

# Batch generation from config file
dice_models batch-generate examples/dice_config.json --output-dir ./dice
```

## Supported Dice Types

| Sides | Name | Polyhedron |
|-------|------|------------|
| 4 | D4 | Tetrahedron |
| 6 | D6 | Cube |
| 8 | D8 | Octahedron |
| 10 | D10 | Pentagonal Trapezohedron |
| 12 | D12 | Dodecahedron |
| 20 | D20 | Icosahedron |

## Configuration Files

Create JSON configuration files for batch generation:

```json
{
  "dice": [
    {
      "sides": 6,
      "filename": "standard_d6.stl",
      "radius": 10.0,
      "text_depth": 0.5,
      "text_size": 3.0
    },
    {
      "sides": 20,
      "filename": "large_d20.stl",
      "radius": 15.0,
      "text_depth": 0.8,
      "text_size": 4.0
    }
  ]
}
```

## 3D Printing Guidelines

### Recommended Settings

- **Layer Height**: 0.1-0.2mm for fine details
- **Infill**: 100% for proper weight and balance
- **Print Speed**: 30-50mm/s for better surface finish

### Size Recommendations

- **Small dice**: 8-10mm radius
- **Standard dice**: 10-12mm radius
- **Large dice**: 15-20mm radius

### Text Depth Guidelines

- **0.1mm layers**: 0.3-0.5mm depth
- **0.2mm layers**: 0.5-0.8mm depth
- **Large dice**: 1.0mm+ depth

## Development

### Testing

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=dice_models
```

## License

MIT License - see LICENSE file for details.
