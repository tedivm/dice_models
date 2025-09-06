# Dice Models Demo

This directory contains comprehensive demonstrations of the dice_models library functionality through a unified CLI-based demo system.

## ✨ **New Refactored Demo System**

The demo system has been completely refactored to provide:

- **Full dice sets** for every demo (D4, D6, D8, D10, D12, D20)
- **Organized output** with separate folders for each demo category
- **Unified CLI interface** using Typer for easy access
- **132 total dice files** across 22 demo categories (vs. 30 in the old system)
- **Clean architecture** with consistent default settings

## Demo File

### `demo_all_features.py` - Unified Demo System

A comprehensive CLI-based demonstration system showcasing all library features with organized output and full dice sets.

**Available Commands:**

```bash
cd demo

# Show all available commands
python demo_all_features.py --help

# Run all demonstrations (creates 132 dice files)
python demo_all_features.py all

# Run individual demo categories
python demo_all_features.py basic          # Standard dice with default settings
python demo_all_features.py layouts        # Custom number arrangements
python demo_all_features.py text           # Text size and depth variations
python demo_all_features.py blank          # Dice without numbers
python demo_all_features.py fonts          # Different font demonstrations
python demo_all_features.py curves         # Curve resolution quality levels
python demo_all_features.py layout-types   # Layout algorithm variations
python demo_all_features.py batch-config   # Create sample batch configuration

# Clean up all generated files
python demo_all_features.py clean
```

### Deprecated Demo Files

**Note:** The following files have been integrated into the unified system:

- `demo_cli.py` - CLI features now available through main demo system
- `demo_layout_types.py` - Layout type demos integrated as `layout-types` command

These files remain but show deprecation messages directing users to the new system.

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

### Summary

- **Total**: 132 STL files across 22 demo categories
- **Each category**: Contains 6 dice files (`d4.stl`, `d6.stl`, `d8.stl`, `d10.stl`, `d12.stl`, `d20.stl`)

## Configuration Files

### `sample_batch_config.json`

Auto-generated example batch configuration for use with the main dice_models CLI:

```bash
dice_models batch-generate demo_output/sample_batch_config.json --output-dir custom_set
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
python demo_all_features.py --help

# Run all demos (creates 132 dice files)
python demo_all_features.py all

# Run specific demo category
python demo_all_features.py fonts
```

### Integration with Main CLI

The demo system works alongside the main dice_models CLI:

```bash
# Generate individual dice with main CLI
python -m dice_models.cli generate 6 my_d6.stl --text-depth 0.8

# Use demo-generated batch configuration
python -m dice_models.cli batch-generate demo_output/sample_batch_config.json --output-dir production_dice
```

## System Requirements

- **Python**: 3.10+ (minimum version as per project standards)
- **Dependencies**: dice_models library with all dependencies
- **Fonts**: System fonts automatically detected (optional)
- **Storage**: ~500MB for complete demo output (132 files)
- **Viewer**: STL viewing software for examining results

## Viewing Results

Generated STL files can be viewed with:

- **Free Software**: Blender, FreeCAD, MeshLab, OpenSCAD
- **Online Viewers**: Thingiverse Customizer, Online 3D Viewer
- **3D Printing Software**: PrusaSlicer, Cura, Bambu Studio
- **CAD Software**: Fusion 360, SolidWorks (import STL)

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
python demo_all_features.py --help
```

### Font Issues

If no system fonts are detected:

```bash
# Check available fonts on your system
python demo_all_features.py fonts

# On Linux, install font packages
sudo apt-get install fonts-dejavu-core fonts-liberation

# Use custom fonts by placing TTF files in accessible location
```

### Permission Errors

Ensure write permissions in demo directory:

```bash
chmod +w demo/
cd demo
python demo_all_features.py basic
```

### Clean Up

Remove all generated files:

```bash
python demo_all_features.py clean
```

## Development

### Adding New Demo Categories

To add new demonstrations:

1. **Add demo function** to `demo_all_features.py`:

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

### Demo Architecture

The system uses a clean, consistent architecture:

- **`DEFAULT_SETTINGS`**: Base configuration for all demos
- **`STANDARD_DICE_SIDES`**: Dice types to generate for each demo
- **`create_dice_set()`**: Core function that generates full dice sets
- **Typer CLI**: Clean command interface with help system
- **Organized output**: Each demo creates its own subdirectory

### Testing New Features

Use the demo system to validate new library features:

```bash
# Test individual components
python demo_all_features.py basic

# Test complete integration
python demo_all_features.py all

# Clean up between tests
python demo_all_features.py clean
```

## Integration & Uses

This demo system serves multiple purposes:

### For Users

- **Learning Examples**: See how to use every library feature
- **Parameter Reference**: Understand the effect of different settings
- **Quality Comparison**: Choose appropriate settings for your needs
- **Batch Templates**: Use generated configurations as starting points

### For Development

- **Feature Validation**: Verify new features work across all dice types
- **Regression Testing**: Ensure changes don't break existing functionality
- **Performance Benchmarking**: Compare rendering times and file sizes
- **Documentation**: Living examples that stay up-to-date with code

### For 3D Printing

- **Master Molds**: High-quality dice for creating silicon molds
- **Direct Printing**: STL files ready for 3D printer slicing
- **Quality Testing**: Compare different resolution settings for your printer
- **Font Testing**: See how different fonts render at your scale

### Integration Points

The demo system integrates with:

- **Main CLI**: Uses same configuration format for batch processing
- **Core Library**: Validates all major API functions
- **Documentation**: Provides concrete examples for all features
- **Testing Suite**: Serves as comprehensive integration tests

## Migration from Old System

If you were using the old demo files:

### Old Commands → New Commands

```bash
# Old way
python demo_all_features.py        → python demo_all_features.py all
python demo_cli.py                 → python demo_all_features.py --help
python demo_layout_types.py        → python demo_all_features.py layout-types

# New individual demos (not available in old system)
python demo_all_features.py basic
python demo_all_features.py fonts
python demo_all_features.py curves
```

### Output Changes

- **Old**: 30 files in flat `demo_output/` structure
- **New**: 132 files in organized subdirectories
- **Benefit**: Much easier to find and compare specific demonstrations

The refactored system preserves all original functionality while adding significant improvements in organization, completeness, and usability.
