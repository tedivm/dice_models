try:
    from . import _version

    __version__ = _version.__version__
except:  # noqa: E722
    __version__ = "0.0.0-dev"

# Import main classes for easy access
from .geometry import DiceGeometry, PolyhedronType
from .geometry.dice import create_standard_dice

__all__ = ["__version__", "DiceGeometry", "PolyhedronType", "create_standard_dice"]
