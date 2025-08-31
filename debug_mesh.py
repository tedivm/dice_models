#!/usr/bin/env python3
"""
Debug script to check bounding box characteristics of different dice.
"""

from dice_models.geometry.dice import create_standard_dice


def debug_mesh_characteristics():
    """Debug mesh characteristics for different dice types."""

    radius = 10.0
    dice_types = [4, 6, 8, 20]

    for sides in dice_types:
        print(f"\nAnalyzing D{sides}:")

        dice = create_standard_dice(sides=sides, radius=radius)
        mesh = dice.generate_mesh(
            include_numbers=False
        )  # Without text to see pure geometry

        bounds = mesh.bounds
        bounds_diff = bounds[1] - bounds[0]

        print(f"  Bounds: {bounds}")
        print(f"  Bounds diff: {bounds_diff}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Vertices: {len(mesh.vertices)}")

        vertex_face_ratio = (
            len(mesh.vertices) / len(mesh.faces) if len(mesh.faces) > 0 else 0
        )
        print(f"  Vertex/Face ratio: {vertex_face_ratio:.3f}")

        aspect_ratios = [
            bounds_diff[0] / bounds_diff[1],
            bounds_diff[1] / bounds_diff[2],
            bounds_diff[0] / bounds_diff[2],
        ]
        print(f"  Aspect ratios: {aspect_ratios}")

        is_cube_like = all(0.8 < ratio < 1.25 for ratio in aspect_ratios)
        print(f"  Is cube-like (old method): {is_cube_like}")

        # Test new cube detection logic
        is_cube_like_new = (
            num_faces > 1000 and vertex_face_ratio > 0.45 and vertex_face_ratio < 0.75
        ) or (num_faces >= 6 and num_faces <= 50 and vertex_face_ratio > 0.8)
        print(f"  Is cube-like (new method): {is_cube_like_new}")


if __name__ == "__main__":
    debug_mesh_characteristics()
