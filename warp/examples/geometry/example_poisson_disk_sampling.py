# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Parallel Poisson-Disk Sampling on Surfaces
#
# Draws a dense blue-noise (Poisson-disk) point set on a mesh surface with
# warp.geometry.PoissonDiskSampler, the parallel algorithm of Bowers et al.,
# "Parallel Poisson Disk Sampling with Spectrum Analysis on Surfaces"
# (SIGGRAPH Asia 2010).
#
# It renders the mesh (blue) and its samples (orange points) with polyscope and
# a soft ground shadow, and saves a still image. Rendering runs headless (offscreen via
# EGL), so no display is needed. The printed pair-correlation statistics confirm
# the blue-noise spectrum: no two samples are closer than the radius.
#
# By default it samples the bundled Stanford bunny. Pass --mesh to load any
# triangle .obj instead; the gallery image uses the xyzrgb dragon from
# https://github.com/alecjacobson/common-3d-test-models
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


def load_obj(path):
    """Minimal triangle-.obj reader (positions and faces only, fan-triangulated)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


class Example:
    def __init__(self, stage_path="example_poisson_disk_sampling.png", mesh=None, radius=None):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.stage_path = stage_path

        if mesh is None:
            self.points, self.faces = load_bunny()
            radius = radius if radius is not None else 0.012
        else:
            self.points, self.faces = load_obj(mesh)
            # Default to roughly 32k samples if no radius was given.
            if radius is None:
                lo, hi = self.points.min(0), self.points.max(0)
                area = float(np.prod(hi - lo))
                radius = 0.02 * float(np.cbrt(area))

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
            f"radius={radius:.4g}  samples={self.sampler.num_samples}  "
            f"candidates={self.sampler.num_candidates}  g(r<0.85r)={inside:.3f}  peak_g={float(g.max()):.2f}"
        )

        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_shadow_darkness(0.35)
        ps.set_up_dir("y_up")
        # Pin the shadow plane to the lowest point of the mesh so the dragon rests
        # on its shadow (no floating gap) rather than at the bounding-box bottom.
        ps.set_ground_plane_height_mode("manual")
        ps.set_ground_plane_height(float(self.points[:, 1].min()))
        ps.set_SSAA_factor(4)
        ps.set_window_size(1920, 1080)

        surf = ps.register_surface_mesh("mesh", self.points, self.faces.reshape(-1, 3), smooth_shade=True)
        surf.set_color((0.2, 0.3, 0.8))  # gptoolbox blue

        cloud = ps.register_point_cloud("poisson samples", self.samples)
        cloud.set_color((1.0, 0.7, 0.2))  # gptoolbox orange
        cloud.set_radius(0.2 * radius, relative=False)

    def render(self):
        ps = self.ps
        lo, hi = self.points.min(0), self.points.max(0)
        center = 0.5 * (lo + hi)
        extent = hi - lo
        # Aim a little below the mesh center so the ground shadow beneath the feet
        # stays comfortably inside the lower part of the frame (not clipped).
        center = center.copy()
        center[1] -= 0.18 * float(extent[1])
        # Side profile: look essentially along the -Z axis so the X-Y plane faces
        # the camera, with only a hint of downward tilt. A near-horizontal view keeps
        # the dragon resting on its shadow and keeps the shadow inside the frame.
        # The mesh's long axis is X; frame so it fills the width with head on the left.
        direction = np.array([0.0, 0.03, -1.0])
        direction /= np.linalg.norm(direction)
        # Distance tuned to the X (width) extent so the dragon fills the frame,
        # with extra margin so the full ground shadow stays in view.
        dist = 1.08 * float(extent[0])
        cam = center + direction * dist
        ps.look_at(tuple(float(x) for x in cam), tuple(float(x) for x in center))
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
    parser.add_argument("--mesh", type=str, default=None, help="Optional triangle .obj to sample instead of the bunny.")
    parser.add_argument("--radius", type=float, default=None, help="Poisson-disk minimum distance.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, mesh=args.mesh, radius=args.radius)
        example.render()
