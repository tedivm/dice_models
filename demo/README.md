# Dice Models Demo

This directory contains comprehensive demonstrations of the dice_models library functionality through a unified CLI-based demo system.

## Demo File

### `demo.py`

A comprehensive CLI-based demonstration system showcasing all library features with organized output and full dice sets.

**Available Commands:**

```bash
cd demo

# Show all available commands
python demo.py --help

# Run all demonstrations (creates 132 dice files)
python demo.py all

# Run individual demo categories
python demo.py basic          # Standard dice with default settings
python demo.py layouts        # Custom number arrangements
python demo.py text           # Text size and depth variations
python demo.py blank          # Dice without numbers
python demo.py fonts          # Different font demonstrations
python demo.py curves         # Curve resolution quality levels
python demo.py layout-types   # Layout algorithm variations
python demo.py batch-config   # Create sample batch configuration

# Clean up all generated files
python demo.py clean
```

## Demo Categories

The refactored system creates **22 organized demo categories**, each containing a complete set of 6 dice:

### Core Demonstrations

- **basic_standard** - Standard dice with highest curve resolution
- **blank_dice** - Dice without numbers for custom engraving

### Layout Demonstrations

- **layout_traditional** - Traditional D6 layouts (opposite faces sum to 7)
- **layout_reverse** - All dice with reverse numbering (highest to lowest)
- **layout_naive** - Simple sequential placement algorithm
- **layout_opposing_balanced** - Balanced opposing face sums
- **layout_opposing_weighted** - Weighted opposing face sums (clustering)

### Text Customization Demonstrations

- **text_small_shallow** - Small text (2.0mm) with shallow engraving (0.3mm)
- **text_large_deep** - Large text (5.0mm) with deep engraving (1.2mm)
- **text_depth_0.3mm** - Shallow engraving demonstration
- **text_depth_0.8mm** - Medium engraving demonstration
- **text_depth_1.5mm** - Deep engraving demonstration
- **text_size_2.0mm** - Small text demonstration
- **text_size_4.0mm** - Medium text demonstration
- **text_size_6.0mm** - Large text demonstration

### Font Demonstrations

- **font_arial** - Arial font rendering
- **font_arial_bold** - Arial Bold font rendering
- **font_times_new_roman_bold** - Times New Roman Bold font rendering

### Curve Quality Demonstrations

- **curve_low** - Low resolution (fast rendering, 6x less detail)
- **curve_medium** - Medium resolution (balanced quality/speed)
- **curve_high** - High resolution (standard quality)
- **curve_highest** - Highest resolution (maximum smoothness)

## Output Structure

### `demo_output/` - Organized Demo Results

The new system creates a well-organized directory structure:

```text
demo_output/
├── basic_standard/          # Standard dice (6 files)
├── blank_dice/             # Dice without numbers (6 files)
├── curve_low/              # Low resolution curves (6 files)
├── curve_medium/           # Medium resolution curves (6 files)
├── curve_high/             # High resolution curves (6 files)
├── curve_highest/          # Highest resolution curves (6 files)
├── font_arial/             # Arial font demos (6 files)
├── font_arial_bold/        # Arial Bold font demos (6 files)
├── font_times_new_roman_bold/ # Times New Roman Bold demos (6 files)
├── layout_naive/           # Naive layout algorithm (6 files)
├── layout_opposing_balanced/ # Balanced layout algorithm (6 files)
├── layout_opposing_weighted/ # Weighted layout algorithm (6 files)
├── layout_reverse/         # Reverse numbering (6 files)
├── layout_traditional/     # Traditional layouts (6 files)
├── text_depth_0.3mm/       # Shallow text (6 files)
├── text_depth_0.8mm/       # Medium depth text (6 files)
├── text_depth_1.5mm/       # Deep text (6 files)
├── text_large_deep/        # Large, deep text (6 files)
├── text_size_2.0mm/        # Small text size (6 files)
├── text_size_4.0mm/        # Medium text size (6 files)
├── text_size_6.0mm/        # Large text size (6 files)
├── text_small_shallow/     # Small, shallow text (6 files)
└── sample_batch_config.json # Example batch configuration
```

## Key Features Demonstrated

### 1. Comprehensive Font Support

The system automatically detects and demonstrates available system fonts:

**macOS Fonts:**

- Arial Bold (`/System/Library/Fonts/Supplemental/Arial Bold.ttf`)
- Arial (`/System/Library/Fonts/Supplemental/Arial.ttf`)
- Times New Roman Bold (`/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf`)
- Geneva (`/System/Library/Fonts/Geneva.ttf`)

**Linux Fonts:**

- DejaVu Sans Bold (`/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`)
- DejaVu Sans (`/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`)
- Liberation Sans Bold (`/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf`)

**Windows Fonts:**

- Arial (`C:/Windows/Fonts/arial.ttf`)
- Arial Bold (`C:/Windows/Fonts/arialbd.ttf`)

### 2. Multiple Layout Algorithms

Demonstrates three different number placement strategies:

```python
from dice_models.geometry.layouts import LayoutType

# Simple sequential placement
dice = create_standard_dice(6, layout_type=LayoutType.NAIVE)

# Balanced opposing face sums
dice = create_standard_dice(6, layout_type=LayoutType.OPPOSING_BALANCED)

# Weighted clustering for bias control
dice = create_standard_dice(6, layout_type=LayoutType.OPPOSING_WEIGHTED)
```

### 3. Precise Text Control

Shows fine-grained control over text appearance:

```python
dice = DiceGeometry(
    polyhedron_type=PolyhedronType.CUBE,
    font_path="/path/to/font.ttf",
    text_depth=0.8,    # Engraving depth in mm
    text_size=4.0,     # Text size in mm
    curve_resolution="highest"  # Quality level
)
```

### 4. Quality vs Performance Options

Demonstrates curve resolution impact:

- **Low**: 6x faster rendering, suitable for rapid prototyping
- **Medium**: Balanced quality and speed for most uses
- **High**: Standard quality for production use
- **Highest**: Maximum smoothness for detailed curved characters

### 5. Batch Processing Integration

Creates sample configurations for automated generation:

```bash
# Use the generated sample configuration
dice_models batch-generate demo_output/sample_batch_config.json --output-dir custom_dice_set
```

## Usage Examples

### Quick Start

```bash
# See all available demos
python demo.py --help

# Run all demos (creates 132 dice files)
python demo.py all

# Run specific demo category
python demo.py fonts
```

### Integration with Main CLI

The demo system works alongside the main dice_models CLI:

```bash
# Generate individual dice with main CLI
python -m dice_models.cli generate 6 my_d6.stl --text-depth 0.8

# Use demo-generated batch configuration
python -m dice_models.cli batch-generate demo_output/sample_batch_config.json --output-dir production_dice
```

## Performance Notes

### File Sizes by Quality

- **Low resolution**: ~120-920 KB per dice
- **Medium resolution**: ~250-1,850 KB per dice
- **High resolution**: ~450-3,700 KB per dice
- **Highest resolution**: ~1,170-8,960 KB per dice

### Generation Time

- **Individual demo**: 5-30 seconds depending on complexity
- **All demos**: 2-5 minutes total (132 files)
- **Font demos**: Slightly longer due to font processing

## Troubleshooting

### Virtual Environment

Always activate your virtual environment first:

```bash
source .venv/bin/activate  # or venv/bin/activate
cd demo
python demo.py --help
```

### Font Issues

If no system fonts are detected:

```bash
# Check available fonts on your system
python demo.py fonts

# On Linux, install font packages
sudo apt-get install fonts-dejavu-core fonts-liberation

# Use custom fonts by placing TTF files in accessible location
```

### Permission Errors

Ensure write permissions in demo directory:

```bash
chmod +w demo/
cd demo
python demo.py basic
```

### Clean Up

Remove all generated files:

```bash
python demo.py clean
```

## Development

### Adding New Demo Categories

To add new demonstrations:

1. **Add demo function** to `demo.py`:

   ```python
   def demo_new_feature() -> None:
       """Demonstrate new feature."""
       print("=" * 60)
       print("DEMO: New Feature")
       print("=" * 60)

       create_dice_set(output_dir, "new_feature", {
           "custom_parameter": custom_value
       })
   ```

2. **Add CLI command**:

   ```python
   @app.command(help="Generate dice with new feature.")
   def new_feature() -> None:
       """Generate dice with new feature."""
       demo_new_feature()
   ```

3. **Update constants** if needed:

   ```python
   # Add new dice types to the standard set
   STANDARD_DICE_SIDES: List[int] = [4, 6, 8, 10, 12, 20, 24]
   ```

4. **Update this README** with the new demo category

### Testing New Features

Use the demo system to validate new library features:

```bash
# Test individual components
python demo.py basic

# Test complete integration
python demo.py all

# Clean up between tests
python demo.py clean
```
