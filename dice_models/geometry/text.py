"""Text rendering and geometry conversion for dice numbers using actual fonts."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

try:
    import triangle
    from fontTools.pens.basePen import BasePen
    from fontTools.ttLib import TTFont

    FONT_TOOLS_AVAILABLE = True
except ImportError:
    FONT_TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default system font paths to try for fallback
DEFAULT_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arialbd.ttf",  # Windows
    "C:/Windows/Fonts/arial.ttf",  # Windows
]


def create_engraved_text(
    base_mesh: trimesh.Trimesh,
    text: str,
    face_center: np.ndarray,
    face_normal: np.ndarray,
    text_depth: float = 0.05,
    text_size: float = 6.0,
    font_path: Optional[str] = None,
    curve_resolution: int = 20,
    sides: Optional[int] = None,
    face_vertices: Optional[np.ndarray] = None,
    face_index: Optional[int] = None,
    radius: Optional[float] = None,
) -> trimesh.Trimesh:
    """
    Create engraved text on a mesh face using actual font rendering with configurable curve quality.

    Args:
        base_mesh: The base mesh to engrave into
        text: Text to engrave (can be numbers, letters, symbols)
        face_center: Center point of the face to engrave on
        face_normal: Normal vector of the face
        text_depth: How deep to make the engraving
        text_size: Size of the text
        font_path: Path to TTF font file
        curve_resolution: Number of points to use for curve approximation (higher = smoother curves)
        sides: Number of sides of the dice (used for face-specific scaling)

    Returns:
        Mesh with the text engraved
    """
    try:
        # Calculate appropriate scaling based on mesh size
        mesh_bounds = base_mesh.bounds
        mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])

        # Calculate face size using a heuristic approach
        # For most dice, mesh_size / 3.0 is a good approximation
        base_face_size = mesh_size / 3.0

        # Apply specific scaling for different dice types
        if sides == 6:
            # D6 has square faces which have more usable area for text
            face_size = base_face_size * 2.2
            logger.debug("D6 detected, applying 2.2x face size scaling for square faces")
        else:
            face_size = base_face_size

        actual_text_size = min(text_size, face_size * 0.4)  # Max 40% of face size
        actual_depth = max(text_depth, mesh_size * 0.02)  # At least 2% of mesh size

        logger.debug(f"Engraving text '{text}': size={actual_text_size:.2f}, depth={actual_depth:.2f}")

        # Create 3D text geometry from font with specified curve resolution
        text_mesh = _create_font_text_mesh(text, actual_text_size, actual_depth, font_path, curve_resolution)

        if text_mesh is None:
            logger.warning(f"Failed to create text mesh for '{text}'")
            return base_mesh

        # Position and orient the text on the face
        text_mesh = _position_text_on_face(
            text_mesh,
            face_center,
            face_normal,
            actual_depth,
            sides,
            face_vertices,
            face_index,
            radius,
        )

        # Perform boolean difference to engrave
        try:
            # Ensure base mesh is watertight and has positive volume
            if not base_mesh.is_watertight:
                base_mesh.fill_holes()
            if base_mesh.volume < 0:
                base_mesh.invert()

            # Fix and validate text mesh more thoroughly
            text_mesh = _ensure_valid_volume_mesh(text_mesh)
            if text_mesh is None:
                logger.warning(f"Could not create valid volume mesh for text '{text}'")
                return base_mesh

            # Perform the engraving operation
            result = base_mesh.difference(text_mesh)

            if result.is_empty or len(result.vertices) == 0:
                logger.warning(f"Boolean difference failed for text '{text}', returning original mesh")
                return base_mesh

            # Ensure result is properly oriented and cleaned
            if result.volume < 0:
                result.invert()

            # Clean up the result mesh to prevent accumulation of complexity
            try:
                result.update_faces(result.nondegenerate_faces())
                result.update_faces(result.unique_faces())
                result.remove_unreferenced_vertices()
                result.fix_normals()

                # Ensure it remains watertight
                if not result.is_watertight:
                    result.fill_holes()

            except Exception as cleanup_error:
                logger.debug(f"Mesh cleanup warning: {cleanup_error}")

            logger.debug(f"Successfully engraved text '{text}': {len(result.vertices)} vertices")
            return result

        except Exception as bool_error:
            logger.warning(f"Boolean operation failed for text '{text}': {bool_error}")
            return base_mesh

    except Exception as e:
        logger.exception(f"Failed to engrave text '{text}': {e}")
        return base_mesh


def _create_font_text_mesh(
    text: str,
    size: float,
    depth: float,
    font_path: Optional[str] = None,
    curve_resolution: int = 20,
) -> Optional[trimesh.Trimesh]:
    """
    Create a 3D mesh from text using actual font rendering with configurable curve quality.

    Args:
        text: The text to render
        size: Size of the text
        depth: Depth of extrusion
        font_path: Path to font file
        curve_resolution: Number of points to use for curve approximation (higher = smoother)

    Returns:
        3D mesh of the text or None if failed
    """
    if not FONT_TOOLS_AVAILABLE:
        logger.warning("fontTools or triangle not available, falling back to simple geometry")
        return _create_fallback_text_mesh(text, size, depth)

    # Use provided font or try to find a system font
    if font_path and Path(font_path).exists():
        actual_font_path = font_path
    else:
        # Try to find a default font
        actual_font_path = _find_default_font()
        if not actual_font_path:
            logger.warning("No font available, falling back to simple geometry")
            return _create_fallback_text_mesh(text, size, depth)

    try:
        return _create_font_based_mesh(text, size, depth, actual_font_path, curve_resolution)
    except Exception as e:
        logger.warning(f"Font-based rendering failed: {e}")
        return _create_fallback_text_mesh(text, size, depth)


def _find_default_font() -> Optional[str]:
    """Find a default font to use."""
    # Try common system fonts
    system_fonts = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/System/Library/Fonts/HelveticaNeue.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]

    for font_path in system_fonts:
        if Path(font_path).exists():
            return font_path

    return None


class PathRecorderPen(BasePen):
    """A pen that records font outline paths as coordinates with high-quality curve approximation."""

    def __init__(self, glyphSet=None, curve_resolution=20):
        super().__init__(glyphSet)
        self.paths = []
        self.current_path = []
        self.curve_resolution = curve_resolution  # Number of points to use for curve approximation

    def moveTo(self, pt):
        if self.current_path:
            self.paths.append(self.current_path)
        self.current_path = [pt]

    def lineTo(self, pt):
        self.current_path.append(pt)

    def curveTo(self, *points):
        """Handle cubic Bézier curves with proper approximation."""
        if len(points) < 3:
            # Not enough control points, just add the last point
            if points:
                self.current_path.append(points[-1])
            return

        # Current point is the start of the curve
        if not self.current_path:
            return

        start_point = self.current_path[-1]

        # For cubic Bézier: start, control1, control2, end
        if len(points) == 3:
            control1, control2, end_point = points
        else:
            # Handle cases with more control points by using the last 3
            control1, control2, end_point = points[-3:]

        # Generate curve points using cubic Bézier formula
        curve_points = self._approximate_cubic_bezier(start_point, control1, control2, end_point, self.curve_resolution)

        # Add the curve points (skip the first one as it's already in the path)
        self.current_path.extend(curve_points[1:])

    def qCurveTo(self, *points):
        """Handle quadratic Bézier curves with proper approximation."""
        if not points:
            return

        if not self.current_path:
            return

        start_point = self.current_path[-1]

        # For quadratic curves, we might have multiple control points
        # Process them in pairs (control_point, end_point)
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                control_point = points[i]
                end_point = points[i + 1]
            else:
                # Odd number of points, treat the last one as both control and end
                control_point = points[i]
                end_point = points[i]

            # Generate curve points using quadratic Bézier formula
            curve_points = self._approximate_quadratic_bezier(
                start_point, control_point, end_point, self.curve_resolution
            )

            # Add the curve points (skip the first one as it's already in the path)
            self.current_path.extend(curve_points[1:])

            # Update start point for next curve segment
            start_point = end_point

    def _approximate_cubic_bezier(self, p0, p1, p2, p3, num_points):
        """
        Approximate a cubic Bézier curve with line segments.

        Args:
            p0: Start point (x, y)
            p1: First control point (x, y)
            p2: Second control point (x, y)
            p3: End point (x, y)
            num_points: Number of points to generate

        Returns:
            List of (x, y) points along the curve
        """
        points = []
        for i in range(num_points + 1):
            t = i / num_points

            # Cubic Bézier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
            x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]

            points.append((x, y))

        return points

    def _approximate_quadratic_bezier(self, p0, p1, p2, num_points):
        """
        Approximate a quadratic Bézier curve with line segments.

        Args:
            p0: Start point (x, y)
            p1: Control point (x, y)
            p2: End point (x, y)
            num_points: Number of points to generate

        Returns:
            List of (x, y) points along the curve
        """
        points = []
        for i in range(num_points + 1):
            t = i / num_points

            # Quadratic Bézier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]

            points.append((x, y))

        return points

    def closePath(self):
        if self.current_path and len(self.current_path) > 2:
            # Close the path by connecting back to start
            self.current_path.append(self.current_path[0])
            self.paths.append(self.current_path)
            self.current_path = []

    def endPath(self):
        if self.current_path:
            self.paths.append(self.current_path)
            self.current_path = []


def _create_font_based_mesh(
    text: str, size: float, depth: float, font_path: str, curve_resolution: int = 20
) -> trimesh.Trimesh:
    """
    Create 3D text mesh using actual font outlines with high-quality curve rendering.

    Args:
        text: The text to render
        size: Size of the text
        depth: Depth of extrusion
        font_path: Path to font file
        curve_resolution: Number of points to use for curve approximation (higher = smoother)
    """
    # Load font
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    # Get font metrics for scaling
    units_per_em = font["head"].unitsPerEm
    scale = size / units_per_em

    all_paths = []
    x_offset = 0

    for char in text:
        char_code = ord(char)
        if char_code not in cmap:
            # Skip characters not in font
            x_offset += size * 0.6  # Add space for missing character
            continue

        glyph_name = cmap[char_code]
        glyph = glyph_set[glyph_name]

        # Extract glyph outline using our custom pen with high-resolution curve handling
        pen = PathRecorderPen(curve_resolution=curve_resolution)
        glyph.draw(pen)

        # Convert paths to our coordinate system
        for path in pen.paths:
            if len(path) >= 3:  # Need at least 3 points for a valid path
                scaled_path = []
                for x, y in path:
                    scaled_x = (x + x_offset) * scale
                    scaled_y = y * scale
                    scaled_path.append((scaled_x, scaled_y))
                all_paths.append(scaled_path)

        # Advance to next character position
        if hasattr(glyph, "width") and glyph.width:
            x_offset += glyph.width
        else:
            x_offset += units_per_em * 0.6  # Default character width

    if not all_paths:
        # No valid paths found
        return _create_fallback_text_mesh(text, size, depth)

    # Convert 2D paths to 3D mesh
    try:
        return _paths_to_3d_mesh(all_paths, depth)
    except Exception as e:
        logger.warning(f"Failed to convert paths to 3D mesh: {e}")
        return _create_fallback_text_mesh(text, size, depth)


def _paths_to_3d_mesh(paths: List[List[Tuple[float, float]]], depth: float) -> trimesh.Trimesh:
    """
    Convert 2D font paths to a 3D mesh using proper triangulation with hole support.
    """
    if not paths:
        raise ValueError("No paths to convert")

    # Separate outer boundaries from holes based on winding order and area
    boundaries, holes = _separate_boundaries_and_holes(paths)

    if not boundaries:
        # No valid boundaries found, create simple box
        return _create_simple_text_box(paths, depth)

    # Collect all vertices and segments for triangulation
    all_vertices = []
    all_segments = []
    hole_points = []  # Points inside holes for triangulation
    vertex_map = {}  # To avoid duplicate vertices

    # Add boundary paths
    for boundary in boundaries:
        _add_path_to_triangulation(boundary, all_vertices, all_segments, vertex_map)

    # Add hole paths and mark hole points
    for hole in holes:
        _add_path_to_triangulation(hole, all_vertices, all_segments, vertex_map)

        # Add a point inside the hole for triangulation
        hole_center = _calculate_path_centroid(hole)
        if hole_center and _point_in_polygon(hole_center, hole):
            hole_points.append(hole_center)

    if len(all_vertices) < 3:
        # Not enough vertices for triangulation, create simple rectangle
        return _create_simple_text_box(paths, depth)

    # Prepare triangulation input
    vertices_array = np.array(all_vertices)
    segments_array = np.array(all_segments) if all_segments else None

    # Perform constrained Delaunay triangulation with holes
    try:
        triangulation_input = {
            "vertices": vertices_array,
        }

        if segments_array is not None and len(segments_array) > 0:
            triangulation_input["segments"] = segments_array

        if hole_points:
            triangulation_input["holes"] = np.array(hole_points)

        # Use triangle library for proper 2D triangulation with holes
        # 'p' for polygon, 'q' for quality mesh, 'a' for area constraint
        flags = "pq30"  # polygon, quality 30 degrees minimum angle
        if hole_points:
            flags += "a0.1"  # small area constraint for better hole handling

        triangulated = triangle.triangulate(triangulation_input, flags)

        if "triangles" not in triangulated or len(triangulated["triangles"]) == 0:
            # Triangulation failed, fall back to simple box
            return _create_simple_text_box(paths, depth)

        # Extract triangulated 2D mesh
        vertices_2d = triangulated["vertices"]
        faces_2d = triangulated["triangles"]

        # Extrude to 3D
        return _extrude_2d_to_3d(vertices_2d, faces_2d, depth)

    except Exception as e:
        logger.warning(f"Triangulation failed: {e}")
        return _create_simple_text_box(paths, depth)


def _separate_boundaries_and_holes(
    paths: List[List[Tuple[float, float]]],
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """
    Separate font paths into outer boundaries and inner holes based on winding order and containment.

    Returns:
        Tuple of (boundaries, holes)
    """
    if not paths:
        return [], []

    # Calculate area and winding for each path
    path_info = []
    for i, path in enumerate(paths):
        if len(path) < 3:
            continue

        area = _calculate_polygon_area(path)
        winding = 1 if area > 0 else -1  # Positive area = CCW, negative = CW
        path_info.append(
            {
                "index": i,
                "path": path,
                "area": abs(area),
                "winding": winding,
                "bbox": _calculate_bbox(path),
            }
        )

    if not path_info:
        return [], []

    # Sort by area (largest first) to process outer boundaries before holes
    path_info.sort(key=lambda x: x["area"], reverse=True)

    boundaries = []
    holes = []

    for info in path_info:
        path = info["path"]

        # Check if this path is contained within any existing boundary
        is_hole = False
        for boundary_info in [p for p in path_info if p["area"] > info["area"]]:
            if _path_contains_path(boundary_info["path"], path):
                is_hole = True
                break

        if is_hole:
            # Ensure holes have clockwise winding (negative area)
            if info["winding"] > 0:
                path = path[::-1]  # Reverse for clockwise winding
            holes.append(path)
        else:
            # Ensure boundaries have counter-clockwise winding (positive area)
            if info["winding"] < 0:
                path = path[::-1]  # Reverse for counter-clockwise winding
            boundaries.append(path)

    return boundaries, holes


def _calculate_polygon_area(path: List[Tuple[float, float]]) -> float:
    """Calculate signed area of a polygon using the shoelace formula."""
    if len(path) < 3:
        return 0.0

    area = 0.0
    n = len(path)
    for i in range(n):
        j = (i + 1) % n
        area += path[i][0] * path[j][1]
        area -= path[j][0] * path[i][1]

    return area / 2.0


def _calculate_bbox(
    path: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Calculate bounding box of a path."""
    if not path:
        return 0, 0, 0, 0

    xs, ys = zip(*path)
    return min(xs), min(ys), max(xs), max(ys)


def _path_contains_path(outer_path: List[Tuple[float, float]], inner_path: List[Tuple[float, float]]) -> bool:
    """Check if outer_path contains inner_path using bounding box and point-in-polygon tests."""
    if not outer_path or not inner_path:
        return False

    # Quick bounding box test first
    outer_bbox = _calculate_bbox(outer_path)
    inner_bbox = _calculate_bbox(inner_path)

    # Check if inner bbox is completely inside outer bbox
    if not (
        outer_bbox[0] <= inner_bbox[0]
        and inner_bbox[2] <= outer_bbox[2]
        and outer_bbox[1] <= inner_bbox[1]
        and inner_bbox[3] <= outer_bbox[3]
    ):
        return False

    # Check if center point of inner path is inside outer path
    center = _calculate_path_centroid(inner_path)
    if center:
        return _point_in_polygon(center, outer_path)

    return False


def _calculate_path_centroid(
    path: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Calculate the centroid of a path."""
    if len(path) < 3:
        return None

    area = _calculate_polygon_area(path)
    if abs(area) < 1e-10:
        return None

    cx = cy = 0.0
    n = len(path)

    for i in range(n):
        j = (i + 1) % n
        cross = path[i][0] * path[j][1] - path[j][0] * path[i][1]
        cx += (path[i][0] + path[j][0]) * cross
        cy += (path[i][1] + path[j][1]) * cross

    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Test if a point is inside a polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _add_path_to_triangulation(
    path: List[Tuple[float, float]],
    all_vertices: List,
    all_segments: List,
    vertex_map: dict,
):
    """Add a path's vertices and segments to the triangulation data structures."""
    if len(path) < 2:
        return

    path_vertex_indices = []

    # Add vertices from this path
    for x, y in path:
        vertex_key = (round(x, 6), round(y, 6))  # Round to avoid floating point issues
        if vertex_key not in vertex_map:
            vertex_map[vertex_key] = len(all_vertices)
            all_vertices.append([x, y])
        path_vertex_indices.append(vertex_map[vertex_key])

    # Add segments connecting consecutive vertices
    for i in range(len(path_vertex_indices)):
        v1_idx = path_vertex_indices[i]
        v2_idx = path_vertex_indices[(i + 1) % len(path_vertex_indices)]
        if v1_idx != v2_idx:  # Avoid degenerate segments
            all_segments.append([v1_idx, v2_idx])


def _create_simple_text_box(paths: List[List[Tuple[float, float]]], depth: float) -> trimesh.Trimesh:
    """Create a simple bounding box for text when triangulation fails."""
    # Find bounding box of all paths
    all_points = []
    for path in paths:
        all_points.extend(path)

    if not all_points:
        return trimesh.creation.box(extents=[1.0, 1.0, depth])

    xs, ys = zip(*all_points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x if max_x > min_x else 1.0
    height = max_y - min_y if max_y > min_y else 1.0

    # Create box centered on the text
    box = trimesh.creation.box(extents=[width, height, depth])
    box.apply_translation([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])

    return box


def _extrude_2d_to_3d(vertices_2d: np.ndarray, faces_2d: np.ndarray, depth: float) -> trimesh.Trimesh:
    """Extrude 2D triangulated mesh to create 3D text volume."""
    if len(vertices_2d) == 0:
        return trimesh.creation.box(extents=[1.0, 1.0, depth])

    # Create bottom vertices (z=0)
    bottom_vertices = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])

    # Create top vertices (z=depth)
    top_vertices = np.column_stack([vertices_2d, np.full(len(vertices_2d), depth)])

    # Combine all vertices
    all_vertices = np.vstack([bottom_vertices, top_vertices])

    num_2d_vertices = len(vertices_2d)
    all_faces = []

    # Bottom faces (reverse winding for outward normals)
    for face in faces_2d:
        all_faces.append([face[2], face[1], face[0]])

    # Top faces (normal winding)
    for face in faces_2d:
        all_faces.append(
            [
                face[0] + num_2d_vertices,
                face[1] + num_2d_vertices,
                face[2] + num_2d_vertices,
            ]
        )

    # Side faces (connect edges of bottom and top)
    # Find boundary edges
    edge_count = {}
    for face in faces_2d:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Create side faces for boundary edges
    for v1, v2 in boundary_edges:
        # Two triangles for each side edge
        all_faces.extend(
            [
                [v1, v2, v2 + num_2d_vertices],
                [v1, v2 + num_2d_vertices, v1 + num_2d_vertices],
            ]
        )

    # Create the final mesh
    text_mesh = trimesh.Trimesh(vertices=all_vertices, faces=np.array(all_faces))

    # Clean up and validate the mesh thoroughly
    try:
        text_mesh.update_faces(text_mesh.unique_faces())
        text_mesh.remove_unreferenced_vertices()
        text_mesh.update_faces(text_mesh.nondegenerate_faces())
        text_mesh.fix_normals()

        if not text_mesh.is_watertight:
            text_mesh.fill_holes()

        # If still not watertight, try more aggressive fixing
        if not text_mesh.is_watertight:
            # Try to merge nearby vertices
            text_mesh.merge_vertices()
            text_mesh.update_faces(text_mesh.nondegenerate_faces())
            text_mesh.fix_normals()

            if not text_mesh.is_watertight:
                text_mesh.fill_holes()

        # Ensure positive volume
        if text_mesh.volume < 0:
            text_mesh.invert()

    except Exception as e:
        logger.warning(f"Mesh cleanup failed: {e}")

    return text_mesh


def _ensure_valid_volume_mesh(mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
    """
    Ensure a mesh is a valid volume (watertight, positive volume, valid geometry).

    Args:
        mesh: Input mesh to validate and fix

    Returns:
        Fixed mesh or None if unfixable
    """
    if mesh is None:
        return None

    try:
        # Remove degenerate faces
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

        # Fix winding and normals
        mesh.fix_normals()

        # Try to make watertight
        if not mesh.is_watertight:
            mesh.fill_holes()

        # Check if it's still not watertight
        if not mesh.is_watertight:
            logger.debug("Mesh is not watertight after fill_holes, trying convex hull")
            # For simple text, try using convex hull as last resort
            try:
                mesh = mesh.convex_hull
            except Exception:
                return None

        # Ensure positive volume
        if mesh.volume <= 0:
            if abs(mesh.volume) > 1e-10:  # Only invert if volume is significant
                mesh.invert()
            else:
                # Volume is essentially zero, mesh is degenerate
                return None

        # Final validation
        if not mesh.is_watertight or mesh.volume <= 0:
            return None

        return mesh

    except Exception as e:
        logger.debug(f"Failed to fix mesh: {e}")
        return None


def _create_fallback_text_mesh(text: str, size: float, depth: float) -> trimesh.Trimesh:
    """
    Create simple fallback text mesh when font rendering fails.
    """
    # Create a simple rectangle based on text length
    width = size * len(text) * 0.7
    height = size * 0.8

    fallback = trimesh.creation.box(extents=[width, height, depth])
    fallback.fix_normals()

    return fallback


def _calculate_d20_edge_alignment(
    face_vertices: np.ndarray,
    face_center: np.ndarray,
    face_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calculate rotation matrix to align text with the closest edge on a D20 triangular face.

    This function finds the edge that requires the least rotation from the default
    text orientation and creates a rotation matrix to align with that edge.

    Args:
        face_vertices: 3 vertices of the triangular face (3x3 array)
        face_center: Center point of the face
        face_normal: Normal vector of the face (should be normalized)

    Returns:
        4x4 transformation matrix for edge alignment, or None if calculation fails
    """
    if face_vertices.shape != (3, 3):
        logger.warning("D20 edge alignment requires exactly 3 vertices")
        return None

    try:
        # Normalize the face normal
        face_normal = face_normal / np.linalg.norm(face_normal)

        # After text is aligned with face normal, the default text baseline direction
        # is what we get when we rotate the original X-axis [1,0,0] by the same rotation
        # that aligns Z-axis with face_normal

        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])

        # Calculate the rotation that aligns z_axis with face_normal
        if np.allclose(face_normal, z_axis):
            # Face is already aligned with Z, no rotation needed
            default_baseline = x_axis
        else:
            # Calculate rotation axis and angle
            rotation_axis = np.cross(z_axis, face_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, face_normal), -1, 1))

            # Apply this same rotation to the x_axis to get the default baseline direction
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            # Rodrigues' rotation formula
            default_baseline = (
                x_axis * cos_a
                + np.cross(rotation_axis, x_axis) * sin_a
                + rotation_axis * np.dot(rotation_axis, x_axis) * (1 - cos_a)
            )

        # Calculate the three edges of the triangle
        edges = np.array(
            [
                face_vertices[1] - face_vertices[0],  # Edge 0-1
                face_vertices[2] - face_vertices[1],  # Edge 1-2
                face_vertices[0] - face_vertices[2],  # Edge 2-0
            ]
        )

        # Normalize edges
        edge_lengths = np.linalg.norm(edges, axis=1)
        edges_normalized = edges / edge_lengths[:, np.newaxis]

        # Find the edge that requires the least rotation from the default baseline
        best_edge_idx = 0
        min_rotation_angle = np.pi  # Start with maximum possible angle
        best_edge_direction = None

        for i, edge in enumerate(edges_normalized):
            # Calculate angle between default baseline and this edge
            # We want the minimum angle, considering both directions of the edge
            angle1 = np.arccos(np.clip(np.dot(default_baseline, edge), -1, 1))
            angle2 = np.arccos(np.clip(np.dot(default_baseline, -edge), -1, 1))

            # For each direction, calculate what the "up" direction of text would be
            # Text "up" is perpendicular to the baseline in the face plane
            text_up1 = np.cross(face_normal, edge)
            text_up1 = text_up1 / np.linalg.norm(text_up1)

            text_up2 = np.cross(face_normal, -edge)
            text_up2 = text_up2 / np.linalg.norm(text_up2)

            # Critical fix: For dice, text should appear "upright" when viewed from outside
            # This means the text "up" direction should generally point away from dice center
            dice_center = np.array([0, 0, 0])  # Dice is centered at origin
            from_center_to_face = face_center - dice_center
            from_center_to_face = from_center_to_face / np.linalg.norm(from_center_to_face)

            # Prefer text "up" direction that has positive component away from dice center
            upright_preference1 = np.dot(text_up1, from_center_to_face)
            upright_preference2 = np.dot(text_up2, from_center_to_face)

            # Choose direction based on angle, but use upright preference as tiebreaker
            if abs(angle1 - angle2) < np.radians(60):  # If angles are reasonably close
                if upright_preference1 > upright_preference2:
                    chosen_angle = angle1
                    chosen_direction = edge
                    logger.debug(
                        f"Edge {i}: used upright preference (away1={upright_preference1:.3f} > away2={upright_preference2:.3f})"
                    )
                else:
                    chosen_angle = angle2
                    chosen_direction = -edge
                    logger.debug(
                        f"Edge {i}: used upright preference (away2={upright_preference2:.3f} > away1={upright_preference1:.3f})"
                    )
            else:
                # Large angle difference, just use minimum
                if angle1 < angle2:
                    chosen_angle = angle1
                    chosen_direction = edge
                else:
                    chosen_angle = angle2
                    chosen_direction = -edge

            logger.debug(
                f"Edge {i}: angle1={np.degrees(angle1):.1f}°, angle2={np.degrees(angle2):.1f}°, chosen={np.degrees(chosen_angle):.1f}°"
            )

            if chosen_angle < min_rotation_angle:
                min_rotation_angle = chosen_angle
                best_edge_idx = i
                best_edge_direction = chosen_direction

        logger.debug(f"Selected edge {best_edge_idx} with rotation {np.degrees(min_rotation_angle):.1f}°")

        # Calculate rotation to align default baseline with the best edge
        target_direction = best_edge_direction

        # Calculate rotation angle in the face plane
        cos_angle = np.dot(default_baseline, target_direction)
        # Use the face normal to determine the rotation direction
        cross_product = np.cross(default_baseline, target_direction)
        sin_angle = np.dot(cross_product, face_normal)
        rotation_angle = np.arctan2(sin_angle, cos_angle)

        # Create rotation matrix around the face normal
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, face_normal, point=face_center)

        logger.debug(f"Final rotation: {np.degrees(rotation_angle):.1f}° around face normal")

        return rotation_matrix

    except Exception as e:
        logger.warning(f"Failed to calculate D20 edge alignment: {e}")
        return None


def _calculate_d8_edge_alignment(
    face_vertices: np.ndarray,
    face_center: np.ndarray,
    face_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calculate rotation matrix to align text with a hemisphere-bridging edge on a D8 triangular face.

    For a D8 (octahedron), we want to align text with edges that meaningfully bridge the two hemispheres.
    The octahedron can be viewed as two square pyramids (top +Z, bottom -Z) joined at their equatorial base.

    The best hemisphere-bridging edges are those that:
    1. Lie in or close to the XY plane (minimal Z-component)
    2. Run in a direction that maximally separates the top/bottom hemispheres
    3. Provide the most readable text orientation

    Args:
        face_vertices: 3 vertices of the triangular face (3x3 array)
        face_center: Center point of the face
        face_normal: Normal vector of the face (should be normalized)

    Returns:
        4x4 transformation matrix for edge alignment, or None if calculation fails
    """
    if face_vertices.shape != (3, 3):
        logger.warning("D8 edge alignment requires exactly 3 vertices")
        return None

    try:
        # Normalize the face normal
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Calculate the default text baseline direction after face alignment
        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])

        if np.allclose(face_normal, z_axis):
            default_baseline = x_axis
        else:
            rotation_axis = np.cross(z_axis, face_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, face_normal), -1, 1))

            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            default_baseline = (
                x_axis * cos_a
                + np.cross(rotation_axis, x_axis) * sin_a
                + rotation_axis * np.dot(rotation_axis, x_axis) * (1 - cos_a)
            )

        # Calculate the three edges of the triangle
        edges = np.array(
            [
                face_vertices[1] - face_vertices[0],  # Edge 0-1
                face_vertices[2] - face_vertices[1],  # Edge 1-2
                face_vertices[0] - face_vertices[2],  # Edge 2-0
            ]
        )

        # Normalize edges
        edge_lengths = np.linalg.norm(edges, axis=1)
        edges_normalized = edges / edge_lengths[:, np.newaxis]

        # For D8, find the edge that best bridges hemispheres by:
        # 1. Having minimal Z-component (horizontal)
        # 2. Having maximal XY-plane component (separates top/bottom hemispheres)
        # 3. Providing good text readability

        best_edge_idx = 0
        best_score = float("inf")
        best_edge_direction = None

        dice_center = np.array([0, 0, 0])
        from_center_to_face = face_center - dice_center
        from_center_to_face = from_center_to_face / np.linalg.norm(from_center_to_face)

        for i, edge in enumerate(edges_normalized):
            # Check both directions of the edge
            for direction_multiplier, direction_name in [
                (1, "forward"),
                (-1, "reverse"),
            ]:
                edge_direction = edge * direction_multiplier

                # Key insight: For hemisphere-bridging, we want edges that:
                # 1. Are horizontal (low |Z|)
                z_component = abs(edge_direction[2])

                # 2. Have significant XY-plane projection (separate hemispheres)
                xy_magnitude = np.sqrt(edge_direction[0] ** 2 + edge_direction[1] ** 2)

                # 3. Create readable text "up" direction
                text_up = np.cross(face_normal, edge_direction)
                text_up = text_up / np.linalg.norm(text_up)
                upright_preference = np.dot(text_up, from_center_to_face)

                # 4. Align well with the face's hemisphere-bridging orientation
                # For octahedron faces, prefer edges that align with the global XY directions
                # rather than diagonal directions
                xy_direction = np.array([edge_direction[0], edge_direction[1], 0])
                if np.linalg.norm(xy_direction) > 0:
                    xy_direction = xy_direction / np.linalg.norm(xy_direction)

                    # Prefer alignment with cardinal directions (X or Y axes)
                    x_alignment = abs(np.dot(xy_direction, [1, 0, 0]))
                    y_alignment = abs(np.dot(xy_direction, [0, 1, 0]))
                    cardinal_preference = max(x_alignment, y_alignment)
                else:
                    cardinal_preference = 0

                # Combine factors into a score (lower is better)
                # Prioritize: horizontal edges > XY magnitude > cardinal alignment > readability
                hemisphere_bridging_score = (
                    z_component * 10.0  # Strongly prefer horizontal edges
                    + (1.0 - xy_magnitude) * 5.0  # Prefer strong XY-plane presence
                    + (1.0 - cardinal_preference) * 2.0  # Prefer cardinal directions
                    + (1.0 - max(0, upright_preference)) * 1.0  # Prefer upright text
                )

                logger.debug(
                    f"D8 Edge {i} {direction_name}: z={z_component:.3f}, xy_mag={xy_magnitude:.3f}, "
                    f"cardinal={cardinal_preference:.3f}, upright={upright_preference:.3f}, "
                    f"score={hemisphere_bridging_score:.3f}"
                )

                if hemisphere_bridging_score < best_score:
                    best_score = hemisphere_bridging_score
                    best_edge_idx = i
                    best_edge_direction = edge_direction

        logger.debug(f"D8 Selected hemisphere-bridging edge {best_edge_idx} with score {best_score:.3f}")

        # Calculate rotation to align default baseline with the best hemisphere-bridging edge
        target_direction = best_edge_direction

        # Calculate rotation angle in the face plane
        cos_angle = np.dot(default_baseline, target_direction)
        cross_product = np.cross(default_baseline, target_direction)
        sin_angle = np.dot(cross_product, face_normal)
        rotation_angle = np.arctan2(sin_angle, cos_angle)

        # Create rotation matrix around the face normal
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, face_normal, point=face_center)

        logger.debug(f"D8 Final rotation: {np.degrees(rotation_angle):.1f}° around face normal")

        return rotation_matrix

    except Exception as e:
        logger.warning(f"Failed to calculate D8 edge alignment: {e}")
        return None


def _calculate_d12_edge_alignment(
    face_vertices: np.ndarray,
    face_center: np.ndarray,
    face_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calculate rotation matrix to align text with the closest edge on a D12 pentagonal face.

    This function finds the edge that requires the least rotation from the default
    text orientation and creates a rotation matrix to align with that edge.

    Args:
        face_vertices: 5 vertices of the pentagonal face (5x3 array)
        face_center: Center point of the face
        face_normal: Normal vector of the face (should be normalized)

    Returns:
        4x4 transformation matrix for edge alignment, or None if calculation fails
    """
    if face_vertices.shape != (5, 3):
        logger.warning("D12 edge alignment requires exactly 5 vertices")
        return None

    try:
        # Normalize the face normal
        face_normal = face_normal / np.linalg.norm(face_normal)

        # After text is aligned with face normal, the default text baseline direction
        # is what we get when we rotate the original X-axis [1,0,0] by the same rotation
        # that aligns Z-axis with face_normal

        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])

        # Calculate the rotation that aligns z_axis with face_normal
        if np.allclose(face_normal, z_axis):
            # Face is already aligned with Z, no rotation needed
            default_baseline = x_axis
        else:
            # Calculate rotation axis and angle
            rotation_axis = np.cross(z_axis, face_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, face_normal), -1, 1))

            # Apply this same rotation to the x_axis to get the default baseline direction
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            # Rodrigues' rotation formula
            default_baseline = (
                x_axis * cos_a
                + np.cross(rotation_axis, x_axis) * sin_a
                + rotation_axis * np.dot(rotation_axis, x_axis) * (1 - cos_a)
            )

        # Calculate the five edges of the pentagon
        edges = np.array(
            [
                face_vertices[1] - face_vertices[0],  # Edge 0-1
                face_vertices[2] - face_vertices[1],  # Edge 1-2
                face_vertices[3] - face_vertices[2],  # Edge 2-3
                face_vertices[4] - face_vertices[3],  # Edge 3-4
                face_vertices[0] - face_vertices[4],  # Edge 4-0
            ]
        )

        # Normalize edges
        edge_lengths = np.linalg.norm(edges, axis=1)
        edges_normalized = edges / edge_lengths[:, np.newaxis]

        # Find the edge that requires the least rotation from the default baseline
        best_edge_idx = 0
        min_rotation_angle = np.pi  # Start with maximum possible angle
        best_edge_direction = None

        for i, edge in enumerate(edges_normalized):
            # Calculate angle between default baseline and this edge
            # We want the minimum angle, considering both directions of the edge
            angle1 = np.arccos(np.clip(np.dot(default_baseline, edge), -1, 1))
            angle2 = np.arccos(np.clip(np.dot(default_baseline, -edge), -1, 1))

            # For each direction, calculate what the "up" direction of text would be
            # Text "up" is perpendicular to the baseline in the face plane
            text_up1 = np.cross(face_normal, edge)
            text_up1 = text_up1 / np.linalg.norm(text_up1)

            text_up2 = np.cross(face_normal, -edge)
            text_up2 = text_up2 / np.linalg.norm(text_up2)

            # For dice, text should appear "upright" when viewed from outside
            # This means the text "up" direction should generally point away from dice center
            dice_center = np.array([0, 0, 0])  # Dice is centered at origin
            from_center_to_face = face_center - dice_center
            from_center_to_face = from_center_to_face / np.linalg.norm(from_center_to_face)

            # Prefer text "up" direction that has positive component away from dice center
            upright_preference1 = np.dot(text_up1, from_center_to_face)
            upright_preference2 = np.dot(text_up2, from_center_to_face)

            # Choose direction based on angle, but use upright preference as tiebreaker
            if abs(angle1 - angle2) < np.radians(60):  # If angles are reasonably close
                if upright_preference1 > upright_preference2:
                    chosen_angle = angle1
                    chosen_direction = edge
                    logger.debug(
                        f"D12 Edge {i}: used upright preference (away1={upright_preference1:.3f} > away2={upright_preference2:.3f})"
                    )
                else:
                    chosen_angle = angle2
                    chosen_direction = -edge
                    logger.debug(
                        f"D12 Edge {i}: used upright preference (away2={upright_preference2:.3f} > away1={upright_preference1:.3f})"
                    )
            else:
                # Large angle difference, just use minimum
                if angle1 < angle2:
                    chosen_angle = angle1
                    chosen_direction = edge
                else:
                    chosen_angle = angle2
                    chosen_direction = -edge

            logger.debug(
                f"D12 Edge {i}: angle1={np.degrees(angle1):.1f}°, angle2={np.degrees(angle2):.1f}°, chosen={np.degrees(chosen_angle):.1f}°"
            )

            if chosen_angle < min_rotation_angle:
                min_rotation_angle = chosen_angle
                best_edge_idx = i
                best_edge_direction = chosen_direction

        logger.debug(f"D12 Selected edge {best_edge_idx} with rotation {np.degrees(min_rotation_angle):.1f}°")

        # Calculate rotation to align default baseline with the best edge
        target_direction = best_edge_direction

        # Calculate rotation angle in the face plane
        cos_angle = np.dot(default_baseline, target_direction)
        # Use the face normal to determine the rotation direction
        cross_product = np.cross(default_baseline, target_direction)
        sin_angle = np.dot(cross_product, face_normal)
        rotation_angle = np.arctan2(sin_angle, cos_angle)

        # Create rotation matrix around the face normal
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, face_normal, point=face_center)

        logger.debug(f"D12 Final rotation: {np.degrees(rotation_angle):.1f}° around face normal")

        return rotation_matrix

    except Exception as e:
        logger.warning(f"Failed to calculate D12 edge alignment: {e}")
        return None


def _calculate_d10_pole_alignment(
    face_center: np.ndarray,
    face_normal: np.ndarray,
    radius: float,
) -> Optional[np.ndarray]:
    """
    Calculate rotation matrix to align text so that the top of the text points toward the closest pole on a D10.

    For a D10 (pentagonal trapezohedron), there are two poles:
    - Top pole at [0, 0, polar_height] where polar_height = radius * 1.2
    - Bottom pole at [0, 0, -polar_height]

    The text should be oriented so that the "top" of the text (text up direction)
    points toward whichever pole is closest to the face center.

    Args:
        face_center: Center point of the face
        face_normal: Normal vector of the face (should be normalized)
        radius: Radius of the dice (needed to calculate pole positions)

    Returns:
        4x4 transformation matrix for pole alignment, or None if calculation fails
    """
    try:
        # Normalize the face normal
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Calculate the pole positions for D10 (same as in polyhedra.py)
        polar_height = radius * 1.2
        top_pole = np.array([0, 0, polar_height])
        bottom_pole = np.array([0, 0, -polar_height])

        # Determine which pole is closest to this face
        distance_to_top = np.linalg.norm(face_center - top_pole)
        distance_to_bottom = np.linalg.norm(face_center - bottom_pole)

        if distance_to_top < distance_to_bottom:
            target_pole = top_pole
            logger.debug(f"Face closer to top pole (distance: {distance_to_top:.2f})")
        else:
            target_pole = bottom_pole
            logger.debug(f"Face closer to bottom pole (distance: {distance_to_bottom:.2f})")

        # Calculate the direction from face center to the closest pole
        pole_direction = target_pole - face_center
        pole_direction = pole_direction / np.linalg.norm(pole_direction)

        # After text is aligned with face normal, we need to determine what the
        # default text "up" direction would be in the face plane
        z_axis = np.array([0, 0, 1])
        y_axis = np.array([0, 1, 0])

        # Calculate the rotation that aligns z_axis with face_normal
        if np.allclose(face_normal, z_axis):
            # Face is already aligned with Z, default text up is Y
            default_text_up = y_axis
        elif np.allclose(face_normal, -z_axis):
            # Face is aligned with negative Z, default text up is negative Y
            default_text_up = -y_axis
        else:
            # Calculate rotation axis and angle
            rotation_axis = np.cross(z_axis, face_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, face_normal), -1, 1))

            # Apply this same rotation to the y_axis to get the default text up direction
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            # Rodrigues' rotation formula
            default_text_up = (
                y_axis * cos_a
                + np.cross(rotation_axis, y_axis) * sin_a
                + rotation_axis * np.dot(rotation_axis, y_axis) * (1 - cos_a)
            )

        # Project pole direction onto the face plane
        # The face plane is defined by the face normal
        pole_direction_in_face_plane = pole_direction - np.dot(pole_direction, face_normal) * face_normal

        # Normalize the projected direction (this will be our desired text up direction)
        if np.linalg.norm(pole_direction_in_face_plane) < 1e-6:
            # Pole direction is parallel to face normal, can't determine alignment
            logger.debug("Pole direction is parallel to face normal, no alignment needed")
            return None

        desired_text_up = pole_direction_in_face_plane / np.linalg.norm(pole_direction_in_face_plane)

        # Calculate the rotation angle needed to align default_text_up with desired_text_up
        # Both vectors are in the face plane, so we rotate around the face normal
        cos_angle = np.clip(np.dot(default_text_up, desired_text_up), -1, 1)
        rotation_angle = np.arccos(cos_angle)

        # Determine rotation direction using cross product
        cross_product = np.cross(default_text_up, desired_text_up)
        # Project cross product onto face normal to determine direction
        if np.dot(cross_product, face_normal) < 0:
            rotation_angle = -rotation_angle

        logger.debug(f"D10 pole alignment: rotating {np.degrees(rotation_angle):.1f}° around face normal")

        # Create rotation matrix around face normal
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, face_normal)

        return rotation_matrix

    except Exception as e:
        logger.warning(f"Failed to calculate D10 pole alignment: {e}")
        return None


def _position_text_on_face(
    text_mesh: trimesh.Trimesh,
    face_center: np.ndarray,
    face_normal: np.ndarray,
    depth: float,
    sides: Optional[int] = None,
    face_vertices: Optional[np.ndarray] = None,
    face_index: Optional[int] = None,
    radius: Optional[float] = None,
) -> trimesh.Trimesh:
    """
    Position and orient text mesh on the dice face.
    """
    # Center the text mesh at origin first
    text_bounds = text_mesh.bounds
    text_center = (text_bounds[0] + text_bounds[1]) / 2
    text_mesh.apply_translation(-text_center)

    # Align text with face normal
    face_normal = np.array(face_normal) / np.linalg.norm(face_normal)
    z_axis = np.array([0, 0, 1])

    if not np.allclose(face_normal, z_axis):
        # Calculate rotation to align Z-axis with face normal
        rotation_axis = np.cross(z_axis, face_normal)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, face_normal), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            text_mesh.apply_transform(rotation_matrix)

    # Special alignment for D20 (icosahedron) - align text with the closest edge
    if sides == 20 and face_vertices is not None and len(face_vertices) == 3:
        edge_alignment_matrix = _calculate_d20_edge_alignment(face_vertices, face_center, face_normal)
        if edge_alignment_matrix is not None:
            text_mesh.apply_transform(edge_alignment_matrix)

        # DIRECT FIX: Apply 180-degree rotation to specific problematic faces
        # These are the faces that appear upside down: 3, 5, 6, 8, 9, 15, 16, 17, 18
        # Convert numbers to face indices (0-based): 2, 4, 5, 7, 8, 14, 15, 16, 17
        problematic_face_indices = {2, 4, 5, 7, 8, 14, 15, 16, 17}

        if face_index is not None and face_index in problematic_face_indices:
            logger.debug(f"Applying 180° fix to problematic face index {face_index}")
            # Apply 180-degree rotation around the face normal
            fix_rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, face_normal, point=face_center)
            text_mesh.apply_transform(fix_rotation_matrix)

    # Special alignment for D8 (octahedron) - align text with hemisphere-bridging edge
    if sides == 8 and face_vertices is not None and len(face_vertices) == 3:
        edge_alignment_matrix = _calculate_d8_edge_alignment(face_vertices, face_center, face_normal)
        if edge_alignment_matrix is not None:
            text_mesh.apply_transform(edge_alignment_matrix)

    # Special alignment for D12 (dodecahedron) - align text with the closest edge
    if sides == 12 and face_vertices is not None and len(face_vertices) == 5:
        edge_alignment_matrix = _calculate_d12_edge_alignment(face_vertices, face_center, face_normal)
        if edge_alignment_matrix is not None:
            text_mesh.apply_transform(edge_alignment_matrix)

        # DIRECT FIX: Apply 180-degree rotation to specific problematic D12 faces
        # These are the faces that appear upside down: 1, 2, 5, 8, 9
        # Convert numbers to face indices (0-based): 0, 1, 4, 7, 8
        problematic_d12_face_indices = {0, 1, 4, 7, 8}

        if face_index is not None and face_index in problematic_d12_face_indices:
            logger.debug(f"Applying 180° fix to problematic D12 face index {face_index}")
            # Apply 180-degree rotation around the face normal
            fix_rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, face_normal, point=face_center)
            text_mesh.apply_transform(fix_rotation_matrix)

    # Special alignment for D10 (pentagonal trapezohedron) - align text toward closest pole
    if sides == 10:
        if radius is not None:
            pole_alignment_matrix = _calculate_d10_pole_alignment(face_center, face_normal, radius)
            if pole_alignment_matrix is not None:
                text_mesh.apply_transform(pole_alignment_matrix)
        else:
            logger.warning("D10 pole alignment requires radius parameter")

    # Position text to intersect with the face for proper boolean difference
    # Use a more conservative approach: position text to partially penetrate the dice
    # The face_center is ON the surface, face_normal points OUTWARD
    # Move text slightly INTO the dice (opposite to outward normal) so it intersects properly
    penetration_depth = min(depth * 0.3, 0.5)  # Conservative penetration
    text_position = face_center - face_normal * penetration_depth
    text_mesh.apply_translation(text_position)

    return text_mesh


# Legacy function for backward compatibility
def create_engraved_number(
    base_mesh: trimesh.Trimesh,
    number: int,
    face_center: np.ndarray,
    face_normal: np.ndarray,
    text_depth: float = 0.05,
    text_size: float = 0.3,
    font_path: Optional[str] = None,
    curve_resolution: int = 20,
    sides: Optional[int] = None,
    face_vertices: Optional[np.ndarray] = None,
    face_index: Optional[int] = None,
    radius: Optional[float] = None,
) -> trimesh.Trimesh:
    """
    Legacy function - now calls the proper font-based text engraving with curve resolution support.
    """
    return create_engraved_text(
        base_mesh=base_mesh,
        text=str(number),
        face_center=face_center,
        face_normal=face_normal,
        text_depth=text_depth,
        text_size=text_size,
        font_path=font_path,
        curve_resolution=curve_resolution,
        sides=sides,
        face_vertices=face_vertices,
        face_index=face_index,
        radius=radius,
    )
