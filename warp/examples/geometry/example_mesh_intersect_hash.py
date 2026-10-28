# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize hashed intersection candidates with headless Polyscope.

Run from the repository root::

    uv run --with polyscope==2.6.1 -m warp.examples.geometry.example_mesh_intersect_hash --device cuda:0

The screenshot shows the overlapping boxes on the left, and separated display
copies of A and B in the middle and right. Gray faces have no candidate pairs,
amber faces participate in candidates, and red faces participate in confirmed
intersections. Display offsets do not affect the intersection calculation.

Use ``--segments`` to replace model A with its unique geometric edge segments,
including triangulation diagonals. Colors then classify segments in A and faces
in B. Exactly coincident vertices and duplicate undirected edges are merged.

Hash primitive bounding-sphere centers and query with the source radius plus the
largest target radius, then filter by sphere overlap. Triangle bounds receive
simple spatial padding ``tol / max(abs(normal))`` for the unnormalized plane
tolerance. This does not guarantee all hits of the narrow phase: for positive
``tol``, nearly parallel planes can amplify snapping beyond that padding, so
some candidate pairs may be skipped. Use ``--tol 0`` to disable plane snapping.
The pure Warp predicates
``wp.geometry.intersect_tri_tri`` and ``wp.geometry.intersect_seg_tri`` provide
the narrow phase. Dense pair masks intentionally
limit this testbed to small models. Surface intersections do not detect containment.
Polyscope is an optional dependency, imported only when rendering.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry


@wp.kernel
def primitive_bounds(
    vertices: wp.array[wp.vec3],
    faces: wp.array2d[int],
    centers: wp.array[wp.vec3],
    radii: wp.array[float],
    max_radius: wp.array[float],
    tol: wp.float64,
):
    i = wp.tid()
    a = vertices[faces[i, 0]]
    b = vertices[faces[i, 1]]
    center = (a + b) * 0.5
    radius = wp.max(wp.length(a - center), wp.length(b - center))
    if faces.shape[1] == 3:
        c = vertices[faces[i, 2]]
        center = (a + b + c) / 3.0
        radius = wp.max(wp.length(a - center), wp.max(wp.length(b - center), wp.length(c - center)))
        # Convert the plane-value tolerance to a displacement along the
        # dominant projection axis. This is simple padding, not a guaranteed
        # enclosure of every near-parallel configuration accepted by snapping.
        normal = wp.abs(wp.cross(b - a, c - a))
        normal_scale = wp.max(normal[0], wp.max(normal[1], normal[2]))
        if normal_scale > 0.0:
            radius += float(tol / wp.float64(normal_scale))
    centers[i] = center
    radii[i] = radius
    wp.atomic_max(max_radius, 0, radius)


@wp.func
def primitive_pair_intersects(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array2d[int],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array2d[int],
    i: int,
    j: int,
    tol: wp.float64,
):
    if faces_a.shape[1] == 2:
        return wp.geometry.intersect_seg_tri(
            vertices_a[faces_a[i, 0]],
            vertices_a[faces_a[i, 1]],
            vertices_b[faces_b[j, 0]],
            vertices_b[faces_b[j, 1]],
            vertices_b[faces_b[j, 2]],
            tol=tol,
        )
    return wp.geometry.intersect_tri_tri(
        vertices_a[faces_a[i, 0]],
        vertices_a[faces_a[i, 1]],
        vertices_a[faces_a[i, 2]],
        vertices_b[faces_b[j, 0]],
        vertices_b[faces_b[j, 1]],
        vertices_b[faces_b[j, 2]],
        tol=tol,
    )


@wp.kernel
def intersect_hashed(
    grid: wp.uint64,
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array2d[int],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array2d[int],
    centers_a: wp.array[wp.vec3],
    radii_a: wp.array[float],
    centers_b: wp.array[wp.vec3],
    radii_b: wp.array[float],
    max_radius_b: wp.array[float],
    padding: float,
    tol: wp.float64,
    candidates: wp.array2d[int],
    hits: wp.array2d[int],
):
    i = wp.tid()
    radius = radii_a[i] + max_radius_b[0] + padding
    for j in wp.hash_grid_query(grid, centers_a[i], radius):
        # Hash collisions can return distant points; explicitly filter them.
        if wp.length(centers_a[i] - centers_b[j]) <= radii_a[i] + radii_b[j] + padding:
            candidates[i, j] += 1
            hits[i, j] = int(primitive_pair_intersects(vertices_a, faces_a, vertices_b, faces_b, i, j, tol))


def make_box(subdivisions=3, scale=1.0, translation=(0.0, 0.0, 0.0), angle=0.0):
    """Create a triangulated box, optionally rotated about Z and translated."""
    vertices = []
    faces = []
    coordinates = np.linspace(-scale, scale, subdivisions + 1)
    for axis in range(3):
        for side in (-scale, scale):
            for u in range(subdivisions):
                for v in range(subdivisions):
                    start = len(vertices)
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = [0.0, 0.0, 0.0]
                        point[axis] = side
                        point[(axis + 1) % 3] = coordinates[u + du]
                        point[(axis + 2) % 3] = coordinates[v + dv]
                        vertices.append(point)
                    quad_faces = ((start, start + 1, start + 2), (start, start + 2, start + 3))
                    faces.extend(tuple(reversed(face)) if side < 0.0 else face for face in quad_faces)

    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    vertices = np.asarray(vertices) @ rotation.T + np.asarray(translation)
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32)


def mesh_edges(model):
    """Extract unique undirected edges, merging exactly coincident vertices."""
    vertices, faces = model
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return vertices, edges.astype(np.int32)


def intersect_models(model_a, model_b, device, *, cell_size=0.5, grid_dim=16, tol=1.0e-6):
    """Intersect A's primitives with B's triangles using a padded spatial hash.

    ``tol`` is the unnormalized plane tolerance passed to the narrow phase.
    Triangle radii are padded by ``tol / max(abs(cross(b-a, c-a)))``; segments
    receive the padding of their target triangle. A small fixed pair padding
    covers roundoff in the unit-scale example fixtures.

    With positive tolerance, plane snapping for nearly parallel triangles can
    accept hits beyond these padded bounds, so this broad phase can skip such
    candidates. Zero disables snapping and tolerance padding; floating-point
    limitations still apply. Triangle inputs must be nondegenerate.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Plane tolerance must be finite and nonnegative.")
    if cell_size <= 0.0 or grid_dim < 1:
        raise ValueError("Cell size and grid dimensions must be positive.")
    if model_a[1].shape[1] not in (2, 3) or model_b[1].shape[1] != 3:
        raise ValueError("Model A must contain segments or triangles, and model B must contain triangles.")
    arrays = []
    bounds = []
    for vertices, faces in (model_a, model_b):
        vertex_array = wp.array(vertices, dtype=wp.vec3, device=device)
        face_array = wp.array(faces, dtype=int, device=device)
        centers = wp.empty(len(faces), dtype=wp.vec3, device=device)
        radii = wp.empty(len(faces), dtype=float, device=device)
        max_radius = wp.zeros(1, dtype=float, device=device)
        wp.launch(
            primitive_bounds,
            dim=len(faces),
            inputs=[vertex_array, face_array, centers, radii, max_radius, wp.float64(tol)],
            device=device,
        )
        arrays.extend((vertex_array, face_array))
        bounds.append((centers, radii, max_radius))

    grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device)
    grid.build(bounds[1][0], cell_size)
    shape = (len(model_a[1]), len(model_b[1]))
    candidates = wp.zeros(shape, dtype=int, device=device)
    hashed = wp.zeros(shape, dtype=int, device=device)
    # Small padding for roundoff in these unit-scale float32 fixtures.
    wp.launch(
        intersect_hashed,
        dim=shape[0],
        inputs=[grid.id, *arrays, *bounds[0][:2], *bounds[1], 1.0e-5, wp.float64(tol), candidates, hashed],
        device=device,
    )
    return candidates.numpy(), hashed.numpy()


def render_screenshot(model_a, model_b, candidates, hits, path):
    """Render the original configuration and separated display copies using EGL."""
    import polyscope as ps  # noqa: PLC0415

    ps.set_use_prefs_file(False)
    ps.set_window_size(1800, 800)
    # Explicit EGL initialization: no window, X server, or ps.show() loop.
    # https://polyscope.run/py/features/headless_rendering/
    ps.init("openGL3_egl")
    ps.set_background_color((0.96, 0.97, 0.99))
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_view_projection_mode("orthographic")

    try:
        for index, (vertices, faces) in enumerate((model_a, model_b)):
            axis = 1 if index == 0 else 0
            candidate_faces = np.any(candidates != 0, axis=axis)
            hit_faces = np.any(hits != 0, axis=axis)
            colors = np.tile((0.66, 0.72, 0.79), (len(faces), 1))
            colors[candidate_faces] = (1.0, 0.62, 0.12)
            colors[hit_faces] = (0.88, 0.12, 0.22)
            center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            offsets = (
                ("overlap", np.array((-4.15, -0.15, -0.1))),
                ("isolated", np.array((0.0 if index == 0 else 3.5, 0.0, 0.0)) - center),
            )
            for view, offset in offsets:
                if faces.shape[1] == 2:
                    network = ps.register_curve_network(f"{view}: segments A", vertices + offset, faces, material="wax")
                    network.set_radius(0.012, relative=False)
                    network.add_color_quantity("segment classification", colors, defined_on="edges", enabled=True)
                    continue
                mesh = ps.register_surface_mesh(
                    f"{view}: mesh {'A' if index == 0 else 'B'}",
                    vertices + offset,
                    faces,
                    edge_width=1.0,
                    edge_color=(0.13, 0.17, 0.24),
                    smooth_shade=False,
                    material="wax",
                )
                mesh.add_color_quantity("triangle classification", colors, defined_on="faces", enabled=True)

        # Oblique view avoids superimposing opposite wireframe faces.
        eye_x = -4.4 if model_a[1].shape[1] == 2 else -0.4
        ps.look_at((eye_x, -12.0, 7.0), (-0.4, 0.0, 0.0))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ps.screenshot(str(path), transparent_bg=False)
        return {"polyscope_version": ps.__version__, "headless": ps.is_headless(), "backend": "openGL3_egl"}
    finally:
        ps.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Warp device; defaults to GPU when available.")
    parser.add_argument(
        "--subdivisions", type=int, default=6, help="Subdivisions per box edge (12*n*n faces per mesh)."
    )
    parser.add_argument("--cell-size", type=float, default=0.5, help="Spatial hash cell width.")
    parser.add_argument("--grid-dim", type=int, default=16, help="Hash bucket count per axis.")
    parser.add_argument(
        "--tol", type=float, default=1.0e-6, help="Nonnegative plane-equation tolerance for the narrow phase."
    )
    parser.add_argument("--segments", action="store_true", help="Replace model A with its unique mesh-edge segments.")
    parser.add_argument("--screenshot", type=Path, default=Path("outputs/mesh_intersect_hash.png"))
    parser.add_argument("--stats", type=Path, default=Path("outputs/mesh_intersect_hash.json"))
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("--subdivisions must be positive")
    if args.cell_size <= 0.0 or args.grid_dim < 1:
        parser.error("--cell-size and --grid-dim must be positive")

    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and nonnegative")

    device = wp.get_device(args.device)
    model_a = make_box(subdivisions=args.subdivisions)
    if args.segments:
        model_a = mesh_edges(model_a)
    model_b = make_box(subdivisions=args.subdivisions, translation=(0.7, 0.3, 0.2), angle=0.31)
    candidates, hits = intersect_models(
        model_a, model_b, device, cell_size=args.cell_size, grid_dim=args.grid_dim, tol=args.tol
    )
    candidate_count = int(np.count_nonzero(candidates))
    hit_count = int(np.count_nonzero(hits))
    stats = {
        "warp_version": wp.__version__,
        "device": str(device),
        "device_name": device.name,
        "subdivisions": args.subdivisions,
        "triangles_a": len(model_a[1]),
        "triangles_b": len(model_b[1]),
        "cell_size": args.cell_size,
        "plane_tolerance": args.tol,
        "tolerance_padding": "tol / max(abs(triangle_normal))",
        "grid_dimensions": [args.grid_dim] * 3,
        "all_pairs": int(candidates.size),
        "candidate_pairs": candidate_count,
        "intersecting_pairs": hit_count,
        "candidate_pairs_rejected_by_narrow_phase": candidate_count - hit_count,
        "pair_reduction_percent": 100.0 * (1.0 - candidate_count / candidates.size),
        "candidate_faces_a": int(np.count_nonzero(np.any(candidates, axis=1))),
        "candidate_faces_b": int(np.count_nonzero(np.any(candidates, axis=0))),
        "intersecting_faces_a": int(np.count_nonzero(np.any(hits, axis=1))),
        "intersecting_faces_b": int(np.count_nonzero(np.any(hits, axis=0))),
        "screenshot": str(args.screenshot),
    }
    if args.segments:
        stats["mode"] = "segment_triangle"
        stats["segments_a"] = stats.pop("triangles_a")
        stats["candidate_segments_a"] = stats.pop("candidate_faces_a")
        stats["intersecting_segments_a"] = stats.pop("intersecting_faces_a")
    stats.update(render_screenshot(model_a, model_b, candidates, hits, args.screenshot))
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
