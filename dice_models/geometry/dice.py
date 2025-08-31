"""Main dice geometry generation module."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import trimesh
from stl import mesh

from .polyhedra import PolyhedronGeometry, PolyhedronType, get_standard_number_layout
from .text import create_engraved_number

logger = logging.getLogger(__name__)


def resolve_curve_resolution(resolution: Union[str, int]) -> int:
    """
    Convert curve resolution string to integer value.

    Args:
        resolution: Either a string ("low", "medium", "high", "highest") or an integer

    Returns:
        Integer resolution value

    Raises:
        ValueError: If string is not a recognized resolution level
    """
    if isinstance(resolution, int):
        if resolution < 3:
            raise ValueError("Curve resolution must be at least 3 points")
        return resolution

    resolution_map = {"low": 5, "medium": 10, "high": 20, "highest": 50}

    if resolution.lower() not in resolution_map:
        valid_options = ", ".join(resolution_map.keys())
        raise ValueError(
            f"Invalid curve resolution '{resolution}'. Must be one of: {valid_options}"
        )

    return resolution_map[resolution.lower()]


class DiceGeometry:
    """Main class for generating 3D dice models."""

    def __init__(
        self,
        polyhedron_type: PolyhedronType,
        radius: float = 10.0,
        number_layout: Optional[List[int]] = None,
        font_path: Optional[str] = None,
        text_depth: float = 0.5,
        text_size: float = 6.0,
        curve_resolution: Union[str, int] = "high",
    ):
        """
        Initialize a dice geometry generator.

        Args:
            polyhedron_type: Type of polyhedron to generate
            radius: Radius of the circumscribed sphere in mm
            number_layout: Custom number layout for faces. If None, uses standard layout
            font_path: Path to TTF font file for numbers
            text_depth: Depth of number engraving in mm
            text_size: Size of numbers in mm
            curve_resolution: Number of points for curve approximation. Can be:
                - String: "low" (5), "medium" (10), "high" (20), "highest" (50)
                - Integer: Custom number of points (minimum 3)
        """
        self.polyhedron_type = polyhedron_type
        self.radius = radius
        self.font_path = font_path
        self.text_depth = text_depth
        self.text_size = text_size
        self.curve_resolution = resolve_curve_resolution(curve_resolution)

        # Set number layout
        if number_layout is None:
            self.number_layout = get_standard_number_layout(polyhedron_type)
        else:
            expected_faces = polyhedron_type.value
            if len(number_layout) != expected_faces:
                raise ValueError(
                    f"Number layout must have {expected_faces} numbers for {polyhedron_type.name}, "
                    f"got {len(number_layout)}"
                )
            self.number_layout = number_layout

        # Generate base geometry
        self.vertices, self.faces = PolyhedronGeometry.get_vertices_and_faces(
            polyhedron_type, radius
        )

        # Calculate face centers and normals
        if polyhedron_type == PolyhedronType.DODECAHEDRON:
            # Special handling for dodecahedron - use logical pentagonal faces
            self.face_centers, self.face_normals = (
                PolyhedronGeometry.get_dodecahedron_logical_face_centers_and_normals(
                    self.vertices, self.faces, radius
                )
            )
        else:
            # Standard calculation for all other polyhedra
            self.face_centers = PolyhedronGeometry.get_face_centers(
                self.vertices, self.faces
            )
            self.face_normals = PolyhedronGeometry.get_face_normals(
                self.vertices, self.faces
            )

    @property
    def sides(self) -> int:
        """Get the number of sides on this dice."""
        return self.polyhedron_type.value

    def generate_mesh(self, include_numbers: bool = True) -> trimesh.Trimesh:
        """
        Generate the complete dice mesh.

        Args:
            include_numbers: Whether to include engraved numbers

        Returns:
            Trimesh object representing the dice
        """
        # Create and prepare base mesh
        base_mesh = self._create_base_mesh(include_numbers)

        if not include_numbers:
            return base_mesh

        # Add number engravings
        return self._add_number_engravings(base_mesh)

    def _create_base_mesh(self, subdivide: bool = True) -> trimesh.Trimesh:
        """
        Create the base polyhedron mesh with optional subdivision.

        Args:
            subdivide: Whether to subdivide for better text engraving

        Returns:
            Base mesh ready for number engraving
        """
        # Create base polyhedron mesh
        base_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

        # Subdivide the mesh to have more vertices for text engraving
        # This gives us more geometry to work with for vertex displacement
        if subdivide:
            try:
                base_mesh = base_mesh.subdivide()  # One level of subdivision
                base_mesh = base_mesh.subdivide()  # Two levels for better detail
            except Exception as e:
                logger.warning(f"Failed to subdivide mesh: {e}")

        # Ensure the mesh is watertight and properly oriented
        if not base_mesh.is_watertight:
            logger.warning("Base mesh is not watertight, attempting to fix")
            base_mesh.fill_holes()

        # Fix face normals to ensure it's a proper volume
        try:
            base_mesh.fix_normals()
        except Exception as e:
            logger.warning(f"Failed to fix normals: {e}")

        return base_mesh

    def _add_number_engravings(self, base_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Add number engravings to the base mesh.

        Args:
            base_mesh: The base mesh to engrave numbers onto

        Returns:
            Mesh with engraved numbers
        """
        # Calculate face centers and normals from the ORIGINAL polyhedron
        # since these are more stable than subdivided face centers
        original_face_centers = self.face_centers
        original_face_normals = self.face_normals

        # Add numbers to each face
        result_mesh = base_mesh

        # Ensure we don't try to add more numbers than we have faces or numbers
        max_faces = min(
            len(self.number_layout),
            len(original_face_centers),
            len(original_face_normals),
        )

        for i in range(max_faces):
            number = self.number_layout[i]
            try:
                # Get face vertices for D20 edge alignment
                face_vertices = None
                if self.polyhedron_type == PolyhedronType.ICOSAHEDRON and i < len(
                    self.faces
                ):
                    face_vertices = self.vertices[self.faces[i]]

                result_mesh = create_engraved_number(
                    base_mesh=result_mesh,
                    number=number,
                    face_center=original_face_centers[i],
                    face_normal=original_face_normals[i],
                    text_depth=self.text_depth,
                    text_size=self.text_size,
                    font_path=self.font_path,
                    curve_resolution=self.curve_resolution,
                    sides=self.sides,
                    face_vertices=face_vertices,
                    face_index=i,
                )
                logger.debug(
                    f"Successfully engraved number {number} on face {i + 1}/{max_faces}"
                )
            except Exception as e:
                logger.exception(f"Failed to add number {number} to face {i}: {e}")

        return result_mesh

    def export_stl(self, output_path: str | Path, include_numbers: bool = True) -> None:
        """
        Export the dice model to an STL file.

        Args:
            output_path: Path where to save the STL file
            include_numbers: Whether to include engraved numbers
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate the mesh
        dice_mesh = self.generate_mesh(include_numbers=include_numbers)

        # Export using trimesh (it handles STL export well)
        dice_mesh.export(str(output_path))
        logger.info(f"Exported {self.polyhedron_type.name} dice to {output_path}")

    def export_stl_numpy(
        self, output_path: str | Path, include_numbers: bool = True
    ) -> None:
        """
        Export the dice model to an STL file using numpy-stl.

        Args:
            output_path: Path where to save the STL file
            include_numbers: Whether to include engraved numbers
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate the mesh
        dice_mesh = self.generate_mesh(include_numbers=include_numbers)

        # Convert to numpy-stl format
        stl_mesh = mesh.Mesh(np.zeros(len(dice_mesh.faces), dtype=mesh.Mesh.dtype))

        for i, face in enumerate(dice_mesh.faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = dice_mesh.vertices[face[j]]

        # Save to file
        stl_mesh.save(str(output_path))
        logger.info(f"Exported {self.polyhedron_type.name} dice to {output_path}")

    def get_info(self) -> dict:
        """
        Get information about the dice geometry.

        Returns:
            Dictionary with dice information
        """
        return {
            "type": self.polyhedron_type.name,
            "sides": self.polyhedron_type.value,
            "radius": self.radius,
            "number_layout": self.number_layout,
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "font_path": self.font_path,
            "text_depth": self.text_depth,
            "text_size": self.text_size,
            "curve_resolution": self.curve_resolution,
        }


def create_standard_dice(
    sides: int,
    radius: float = 10.0,
    output_path: Optional[str | Path] = None,
    **kwargs,
) -> DiceGeometry:
    """
    Create a standard dice with the specified number of sides.

    Args:
        sides: Number of sides (4, 6, 8, 10, 12, or 20)
        radius: Radius of the dice in mm
        output_path: If provided, export STL to this path
        **kwargs: Additional arguments passed to DiceGeometry, including:
            - curve_resolution: "low", "medium", "high", "highest", or integer

    Returns:
        DiceGeometry object

    Raises:
        ValueError: If sides is not a valid dice type
    """
    # Map sides to polyhedron types
    sides_to_type = {
        4: PolyhedronType.TETRAHEDRON,
        6: PolyhedronType.CUBE,
        8: PolyhedronType.OCTAHEDRON,
        10: PolyhedronType.PENTAGONAL_TRAPEZOHEDRON,
        12: PolyhedronType.DODECAHEDRON,
        20: PolyhedronType.ICOSAHEDRON,
    }

    if sides not in sides_to_type:
        valid_sides = list(sides_to_type.keys())
        raise ValueError(
            f"Invalid number of sides: {sides}. Valid options: {valid_sides}"
        )

    polyhedron_type = sides_to_type[sides]
    dice = DiceGeometry(polyhedron_type=polyhedron_type, radius=radius, **kwargs)

    if output_path:
        dice.export_stl(output_path)

    return dice
