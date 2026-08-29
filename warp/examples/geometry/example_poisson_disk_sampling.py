# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Parallel Poisson-Disk Sampling on Surfaces
#
# Draws a dense blue-noise (Poisson-disk) point set on the Stanford bunny with
# warp.geometry.PoissonDiskSampler, the parallel algorithm of Bowers et al.,
# "Parallel Poisson Disk Sampling with Spectrum Analysis on Surfaces"
# (SIGGRAPH Asia 2010).
#
# It renders the mesh and its samples with polyscope and saves a still image.
# Rendering runs headless (offscreen via EGL), so no display is needed. The
# printed pair-correlation statistics confirm the blue-noise spectrum: no two
# samples are closer than the radius.
#
#   uv run --with usd-core --with polyscope \
#       warp/examples/geometry/example_poisson_disk_sampling.py
###########################################################################

import os

import numpy as np

import warp as wp
import warp.examples
import warp.geometry


def load_bunny():
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            points = np.array(m.GetPointsAttr().Get(), dtype=np.float32)
            indices = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            return points, indices
    raise RuntimeError("no mesh found in bunny.usd")


class Example:
    def __init__(self, stage_path="example_poisson_disk_sampling.png", radius=0.012):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.stage_path = stage_path

        self.points, self.faces = load_bunny()

        # Dense blue-noise sampling: no two points closer than `radius`.
        self.sampler = warp.geometry.PoissonDiskSampler(
            wp.array(self.points, dtype=wp.vec3),
            wp.array(self.faces, dtype=wp.int32),
            radius=radius,
            seed=0,
        )
        self.samples = self.sampler.points.numpy()

        r, g = self.sampler.pair_correlation(num_bins=48)
        wp.synchronize_device()
        inside = float(g[r < 0.85 * radius].mean())
        print(
            f"bunny: radius={radius}  samples={self.sampler.num_samples}  "
            f"candidates={self.sampler.num_candidates}  g(r<0.85r)={inside:.3f}  peak_g={float(g.max()):.2f}"
        )

        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("y_up")
        ps.set_front_dir("z_front")
        ps.set_SSAA_factor(4)

        mesh = ps.register_surface_mesh("bunny", self.points, self.faces.reshape(-1, 3), smooth_shade=True)
        mesh.set_color((0.5, 0.55, 0.62))

        cloud = ps.register_point_cloud("poisson samples", self.samples)
        cloud.set_radius(0.45 * radius, relative=False)
        # Color the samples by height for a bit of depth in the still.
        h = self.samples[:, 1]
        cloud.add_scalar_quantity("height", h, enabled=True, cmap="viridis")

    def render(self):
        ps = self.ps
        ps.look_at((1.85, 1.2, 1.85), (-0.15, 0.82, 0.02))
        if self.stage_path:
            ps.screenshot(self.stage_path, transparent_bg=False)
            print(f"wrote {self.stage_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_poisson_disk_sampling.png",
        help="Path to the output image.",
    )
    parser.add_argument("--radius", type=float, default=0.012, help="Poisson-disk minimum distance.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, radius=args.radius)
        example.render()
