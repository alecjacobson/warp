# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Mesh Topology Statistics
#
# Shows how to use warp.geometry.triangle_mesh_topology_statistics() to check
# the combinatorial health of a triangle mesh: edge manifoldness, orientation,
# vertex manifoldness, boundaries, and degenerate triangles -- all from the
# triangle index array alone, without vertex positions.
#
# A shipped asset (the bunny) is a clean closed oriented manifold, so the
# example also derives a few deliberately broken variants from it to show what
# each defect looks like in the reported statistics.
#
###########################################################################

import os

import numpy as np

import warp as wp
import warp.examples
import warp.geometry

try:
    from pxr import Usd, UsdGeom

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


def _report(name, indices, num_points, device):
    """Print the topology statistics for a flat triangle-index array."""
    stats = warp.geometry.triangle_mesh_topology_statistics(
        wp.array(indices, dtype=wp.int32, device=device), num_points=num_points, device=device
    )
    print(f"\n{name}")
    print(f"  vertices={stats.num_vertices}  triangles={stats.num_triangles}  edges={stats.num_edges}")
    print(
        f"  boundary={stats.num_boundary_edges}  nonmanifold_edges={stats.num_nonmanifold_edges}  "
        f"misoriented={stats.num_misoriented_edges}"
    )
    print(
        f"  nonmanifold_vertices={stats.num_nonmanifold_vertices}  "
        f"unreferenced={stats.num_unreferenced_vertices}  degenerate={stats.num_degenerate_triangles}"
    )
    print(
        f"  is_manifold={stats.is_manifold}  is_closed_manifold={stats.is_closed_manifold}  "
        f"is_oriented={stats.is_oriented}"
    )
    return stats


def _load_bunny():
    """Return ``(indices, num_points)`` for the shipped bunny mesh, or ``None``."""
    if not USD_AVAILABLE:
        return None
    path = os.path.join(warp.examples.get_asset_directory(), "bunny.usd")
    stage = Usd.Stage.Open(path)
    geom = UsdGeom.Mesh(next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)))
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    num_points = len(geom.GetPointsAttr().Get())
    return indices, num_points


class Example:
    def __init__(self, device=None):
        self.device = wp.get_device(device)

    def step(self):
        with wp.ScopedDevice(self.device):
            bunny = _load_bunny()
            if bunny is not None:
                indices, num_points = bunny
                self._report_bunny(indices, num_points)
            else:
                print("usd-core not available; using a synthetic tetrahedron instead.")
                # Consistently outward-oriented tetrahedron.
                indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
                num_points = 4
                _report("closed tetrahedron", indices, num_points, self.device)

    def _report_bunny(self, indices, num_points):
        # Pristine asset: a closed, oriented, manifold surface.
        _report("bunny (as shipped)", indices, num_points, self.device)

        tris = indices.reshape(-1, 3)

        # Drop the first triangle to open a hole -> boundary edges appear.
        _report("bunny with one triangle removed", tris[1:].reshape(-1), num_points, self.device)

        # Flip the winding of one triangle -> its three edges become misoriented.
        flipped = tris.copy()
        flipped[0] = flipped[0, ::-1]
        _report("bunny with one triangle flipped", flipped.reshape(-1), num_points, self.device)

        # Duplicate a triangle onto an existing edge fan -> a non-manifold edge.
        i, j = tris[0, 0], tris[0, 1]
        extra = np.array([[i, j, num_points]], dtype=np.int32)  # new apex on edge (i, j)
        nonmanifold = np.concatenate([tris, extra], axis=0)
        _report("bunny with an extra fin on one edge", nonmanifold.reshape(-1), num_points + 1, self.device)

        # Collapse a triangle to a repeated index -> a combinatorially degenerate face.
        degenerate = tris.copy()
        degenerate[0] = np.array([tris[0, 0], tris[0, 0], tris[0, 1]], dtype=np.int32)
        _report("bunny with one degenerate triangle", degenerate.reshape(-1), num_points, self.device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Unused; accepted for consistency with other examples.",
    )
    args = parser.parse_known_args()[0]

    example = Example(device=args.device)
    example.step()
