"""Tests for layout type functionality."""

from dice_models import create_standard_dice
from dice_models.geometry.dice import DiceGeometry
from dice_models.geometry.layouts import (
    LayoutType,
    get_layout_description,
    get_standard_number_layout,
)
from dice_models.geometry.polyhedra import PolyhedronType


def test_layout_type_enum():
    """Test that all layout types are available."""
    expected_types = ["naive", "opposing-balanced", "opposing-weighted"]
    actual_types = [lt.value for lt in LayoutType]
    assert set(actual_types) == set(expected_types)


def test_layout_descriptions():
    """Test that all layout types have descriptions."""
    for layout_type in LayoutType:
        description = get_layout_description(layout_type)
        assert isinstance(description, str)
        assert len(description) > 0


def test_opposing_balanced_default():
    """Test that opposing-balanced is the default layout."""
    d6 = create_standard_dice(sides=6)
    expected_layout = get_standard_number_layout(
        polyhedron_type=PolyhedronType.CUBE, layout_type=LayoutType.OPPOSING_BALANCED
    )
    assert d6.number_layout == expected_layout


def test_all_dice_types_support_all_layouts():
    """Test that all dice types support all layout types."""
    dice_types = [4, 6, 8, 10, 12, 20]

    for sides in dice_types:
        for layout_type in LayoutType:
            dice = create_standard_dice(sides=sides, layout_type=layout_type)
            assert len(dice.number_layout) == sides
            assert dice.layout_type == layout_type


def test_naive_layout_is_sequential():
    """Test that naive layouts are sequential."""
    for sides in [4, 6, 8, 10, 12, 20]:
        dice = create_standard_dice(sides=sides, layout_type=LayoutType.NAIVE)
        expected = list(range(1, sides + 1)) if sides != 10 else list(range(0, 10))
        assert dice.number_layout == expected


def test_custom_layout_overrides_layout_type():
    """Test that custom layouts override layout_type parameter."""
    custom_layout = [6, 5, 4, 3, 2, 1]
    d6 = create_standard_dice(sides=6, layout_type=LayoutType.NAIVE, number_layout=custom_layout)
    assert d6.number_layout == custom_layout


def test_dice_geometry_constructor_with_layout_type():
    """Test DiceGeometry constructor with layout_type parameter."""
    dice = DiceGeometry(
        polyhedron_type=PolyhedronType.CUBE,
        layout_type=LayoutType.OPPOSING_WEIGHTED,
    )
    expected = get_standard_number_layout(polyhedron_type=PolyhedronType.CUBE, layout_type=LayoutType.OPPOSING_WEIGHTED)
    assert dice.number_layout == expected
    assert dice.layout_type == LayoutType.OPPOSING_WEIGHTED


def test_dice_info_includes_layout_type():
    """Test that dice info includes layout type information."""
    dice = create_standard_dice(sides=6, layout_type=LayoutType.OPPOSING_WEIGHTED)
    info = dice.get_info()
    assert "layout_type" in info
    assert info["layout_type"] == "opposing-weighted"


def test_layout_uniqueness():
    """Test that different layout types produce different arrangements."""
    sides = 20  # Use D20 for good variety

    naive = create_standard_dice(sides=sides, layout_type=LayoutType.NAIVE)
    balanced = create_standard_dice(sides=sides, layout_type=LayoutType.OPPOSING_BALANCED)
    weighted = create_standard_dice(sides=sides, layout_type=LayoutType.OPPOSING_WEIGHTED)

    # All should be different
    assert naive.number_layout != balanced.number_layout
    assert naive.number_layout != weighted.number_layout
    assert balanced.number_layout != weighted.number_layout

    # But all should contain the same numbers
    assert set(naive.number_layout) == set(balanced.number_layout)
    assert set(naive.number_layout) == set(weighted.number_layout)


def test_all_numbers_present():
    """Test that all layouts contain all expected numbers."""
    for sides in [4, 6, 8, 10, 12, 20]:
        for layout_type in LayoutType:
            dice = create_standard_dice(sides=sides, layout_type=layout_type)
            layout = dice.number_layout

            if sides == 10:
                expected_numbers = set(range(0, 10))
            else:
                expected_numbers = set(range(1, sides + 1))

            assert set(layout) == expected_numbers
