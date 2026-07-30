# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Geometry Normals
#
# Computes the total surface area and the volume/first/second moments of a
# closed triangle mesh, then differentiates through those scalars with a
# warp.Tape to recover two classic per-vertex normal fields as gradients:
#
#   d(area)/dV    -> mean-curvature normals  (H * n, the gradient of surface area)
#   d(volume)/dV  -> area-weighted normals   (outward normals scaled by vertex area)
#
# The surface area is computed with warp.geometry.triangle_areas; the moment
# kernels live here in the example. Results are written to a USD stage (two
# copies of the mesh, each colored by one normal field encoded as RGB) and can
# optionally be shown in an interactive polyscope viewer.
#
#   uv run --with usd-core warp/examples/geometry/normals.py
#   uv run --with usd-core --with polyscope warp/examples/geometry/normals.py --polyscope
###########################################################################


import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.geometry
import warp.render


@wp.kernel
def sum_kernel(values: wp.array(dtype=float), total: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(total, 0, values[i])


def main(show_polyscope=False):

    usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
    points = wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3, requires_grad=True)
    indices = wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int)

    total_area = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        areas = warp.geometry.triangle_areas(points, indices)
        wp.launch(sum_kernel, dim=areas.shape[0], inputs=[areas], outputs=[total_area])
        m0, m1, m2 = wp.geometry.moments(points, indices)

    # d(area)/dV: mean-curvature normals.
    tape.backward(loss=total_area)
    mean_curvature_normals = points.grad.numpy().copy()

    # d(volume)/dV: area-weighted (outward) normals.
    tape.reset()
    with tape:
        m0, m1, m2 = wp.geometry.moments(points, indices)
    tape.backward(loss=m0)
    dm0dpoints = points.grad.numpy().copy()

    volume = float(m0.numpy()[0])
    centroid = m1.numpy()[0] / volume
    print(f"vertices = {len(points)}, triangles = {len(indices) / 3}")
    print(f"total area = {float(total_area.numpy()[0]):.6f}")
    print(f"volume     = {volume:.6f}")
    print(f"centroid   = {centroid}")
    print(f"inertia    =\n{m2.numpy()[0]}")

    # triangle normals
    tN = wp.geometry.triangle_normals(points, indices, normalized=True)
    Weighting = wp.geometry.VertexNormalWeighting
    area_weighted_vertex_normals = wp.geometry.vertex_normals(
        points, indices, weighting=Weighting.AREA, normalized=True
    )
    uniform_weighted_vertex_normals = wp.geometry.vertex_normals(
        points, indices, weighting=Weighting.UNIFORM, normalized=True
    )
    angle_weighted_vertex_normals = wp.geometry.vertex_normals(
        points, indices, weighting=Weighting.ANGLE, normalized=True
    )

    # discrete Gaussian curvature (angle defect) per vertex
    gaussian_curvature = wp.geometry.vertex_gaussian_curvature(points, indices)
    # Gauss-Bonnet check: the total should be 2*pi*chi (4*pi for the genus-0 bunny).
    print(f"total Gaussian curvature = {float(gaussian_curvature.numpy().sum()):.6f}  (expected {4 * np.pi:.6f})")

    if show_polyscope:
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as psim  # noqa: PLC0415

        ps.init()
        mesh = ps.register_surface_mesh("torus", points.numpy(), indices.numpy().reshape(-1, 3))
        mesh.add_scalar_quantity("triangle areas", areas.numpy(), enabled=True, defined_on="faces")
        mesh.add_vector_quantity("∂total-area/∂V (mean-curvature normals)", mean_curvature_normals, enabled=True)
        mesh.add_vector_quantity("∂volume/∂V (area-weighted normals)", dm0dpoints, enabled=True)
        mesh.add_vector_quantity("triangle normals", tN.numpy(), defined_on="faces", enabled=True)
        mesh.add_vector_quantity("area-weighted vertex normals", area_weighted_vertex_normals.numpy(), enabled=True)
        mesh.add_vector_quantity(
            "uniformly weighted vertex normals", uniform_weighted_vertex_normals.numpy(), enabled=True
        )
        mesh.add_vector_quantity("angle-weighted vertex normals", angle_weighted_vertex_normals.numpy(), enabled=True)

        # point cloud at the vertices, colored by Gaussian curvature
        cloud = ps.register_point_cloud("vertices", points.numpy())
        cloud.add_scalar_quantity("Gaussian curvature", gaussian_curvature.numpy(), enabled=True, cmap="coolwarm")

        def callback():
            if psim.IsKeyPressed(psim.ImGuiKey_Escape):
                ps.unshow()

        ps.set_user_callback(callback)
        ps.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")

    parser.add_argument("--polyscope", action="store_true", help="Launch an interactive polyscope viewer.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        main(show_polyscope=args.polyscope)
