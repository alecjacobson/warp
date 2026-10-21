# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Parallel Poisson-Disk Sampling on Surfaces
#
# Draws a dense blue-noise (Poisson-disk) point set on a mesh surface with
# warp.geometry.PoissonDiskSampler, the parallel algorithm of Bowers et al.,
# "Parallel Poisson Disk Sampling with Spectrum Analysis on Surfaces"
# (SIGGRAPH Asia 2010).
#
# It writes the mesh (blue) and its samples (orange points) to a USD file. The
# printed pair-correlation statistics confirm the blue-noise spectrum: no two
# samples are closer than the radius.
#
# By default it samples the bundled Stanford bunny. Pass --mesh to load any
# triangle .obj instead.
#
#   uv run --with usd-core warp/examples/geometry/example_poisson_disk_sampling.py
###########################################################################

import os

import numpy as np

import warp as wp
import warp.examples
import warp.geometry
import warp.render


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
    """Minimal triangle-.obj reader (positions and faces only, fan-triangulated).

    Drops vertices that no face references and remaps the indices accordingly.
    """
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # Positive OBJ indices are 1-based; negative ones count back from the
                # vertices read so far.
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.extend([idx[0], idx[k], idx[k + 1]])
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Keep only vertices referenced by a face and renumber the faces to match.
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return verts[used], remap[faces]


class Example:
    def __init__(self, stage_path="example_poisson_disk_sampling.usd", mesh=None, radius=None):
        if mesh is None:
            self.points, self.faces = load_bunny()
            radius = radius if radius is not None else 0.012
        else:
            self.points, self.faces = load_obj(mesh)
            # Default to roughly 32k samples if no radius was given.
            if radius is None:
                # Scale with the surface area, which stays nonzero for flat meshes.
                tri = self.points[self.faces.reshape(-1, 3)]
                area = 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
                radius = 0.02 * float(np.sqrt(area))
        self.radius = radius

        # Dense blue-noise sampling: no two points closer than `radius`.
        self.sampler = warp.geometry.PoissonDiskSampler(
            wp.array(self.points, dtype=wp.vec3),
            wp.array(self.faces, dtype=wp.int32),
            radius=radius,
            seed=0,
        )
        self.samples = self.sampler.points.numpy()

        r, g = self.sampler.pair_correlation(num_bins=48)
        inside = float(g[r < 0.85 * radius].mean())
        print(
            f"radius={radius:.4g}  samples={self.sampler.num_samples}  "
            f"candidates={self.sampler.num_candidates}  g(r<0.85r)={inside:.3f}  peak_g={float(g.max()):.2f}"
        )

        self.renderer = wp.render.UsdRenderer(stage_path) if stage_path else None

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(0.0)
            self.renderer.render_mesh(
                name="mesh",
                points=self.points,
                indices=self.faces.reshape(-1),
                colors=(0.2, 0.3, 0.8),
            )
            self.renderer.render_points(
                name="poisson_samples",
                points=self.samples,
                radius=0.2 * self.radius,
                colors=(1.0, 0.7, 0.2),
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_poisson_disk_sampling.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--mesh", type=str, default=None, help="Optional triangle .obj to sample instead of the bunny.")
    parser.add_argument("--radius", type=float, default=None, help="Poisson-disk minimum distance.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, mesh=args.mesh, radius=args.radius)
        example.render()

        if example.renderer:
            example.renderer.save()
