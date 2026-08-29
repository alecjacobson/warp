# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Parallel Poisson-Disk Sampling on Surfaces
#
# Draws blue-noise (Poisson-disk) point sets on a mesh surface with
# warp.geometry.PoissonDiskSampler, the parallel algorithm of Bowers et al.,
# "Parallel Poisson Disk Sampling with Spectrum Analysis on Surfaces"
# (SIGGRAPH Asia 2010).
#
# The example sweeps the Poisson radius from coarse to fine and renders each
# resulting point set on the mesh with polyscope, writing the frames to an
# animated GIF. It also prints the sample count and the pair-correlation
# statistics that confirm the blue-noise spectrum (no pairs closer than the
# radius). Rendering runs headless (offscreen via EGL), so no display is needed.
#
#   uv run --with polyscope --with imageio \
#       warp/examples/geometry/example_poisson_disk_sampling.py
###########################################################################

import numpy as np

import warp as wp
import warp.geometry


def build_torus(num_major=64, num_minor=32, major_radius=1.0, minor_radius=0.35):
    """A torus with mildly non-uniform tessellation (denser on the inner rim)."""
    u = np.linspace(0.0, 2.0 * np.pi, num_major, endpoint=False)
    s = np.linspace(0.0, 1.0, num_minor, endpoint=False)
    v = 2.0 * np.pi * (s + 0.3 * np.sin(2.0 * np.pi * s) / (2.0 * np.pi))
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    y = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    z = minor_radius * np.sin(vv)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(num_major):
        for j in range(num_minor):
            a = i * num_minor + j
            b = ((i + 1) % num_major) * num_minor + j
            c = ((i + 1) % num_major) * num_minor + (j + 1) % num_minor
            d = i * num_minor + (j + 1) % num_minor
            faces.extend([a, b, c, a, c, d])
    return points, np.array(faces, dtype=np.int32)


class Example:
    def __init__(self, output="example_poisson_disk_sampling.gif", num_frames=24):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.output = output
        self.num_frames = num_frames

        self.points, self.faces = build_torus()
        self.tri = self.faces.reshape(-1, 3)

        # A reusable sampler avoids rebuilding the mesh/CDF every frame.
        self.sampler_points = wp.array(self.points, dtype=wp.vec3)
        self.sampler_faces = wp.array(self.faces, dtype=wp.int32)

        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_SSAA_factor(2)
        self.mesh = ps.register_surface_mesh("torus", self.points, self.tri, smooth_shade=True)
        self.mesh.set_color((0.55, 0.6, 0.7))
        self.mesh.set_transparency(0.45)

        self.frames = []

    def step(self, frame):
        # Sweep the radius from coarse to fine, and back, so the GIF loops
        # smoothly through increasing then decreasing sample density.
        t = frame / max(self.num_frames - 1, 1)
        tri = 1.0 - abs(2.0 * t - 1.0)  # 0 -> 1 -> 0
        radius = float(0.28 * (1.0 - tri) + 0.05 * tri)

        sampler = warp.geometry.PoissonDiskSampler(self.sampler_points, self.sampler_faces, radius=radius, seed=0)
        pts = sampler.points.numpy()

        r, g = sampler.pair_correlation(num_bins=48)
        inside = float(g[r < 0.85 * radius].mean()) if np.any(r < 0.85 * radius) else 0.0
        print(
            f"frame {frame:2d}: radius={radius:.3f}  samples={sampler.num_samples:5d}  "
            f"g(r<0.85r)={inside:.3f}  peak_g={float(g.max()):.2f}"
        )
        return pts, radius

    def render(self, frame, pts, radius):
        ps = self.ps
        cloud = ps.register_point_cloud("poisson samples", pts)
        cloud.set_color((0.95, 0.55, 0.15))
        cloud.set_radius(0.4 * radius, relative=False)

        # Slowly orbit the camera so the GIF shows the distribution in the round.
        angle = 2.0 * np.pi * frame / self.num_frames
        cam = (3.4 * np.cos(angle), 3.4 * np.sin(angle), 2.0)
        ps.look_at(cam, (0.0, 0.0, 0.0))

        buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=False))
        if buf.dtype != np.uint8:
            buf = (np.clip(buf, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.frames.append(buf[:, :, :3])

    def save(self):
        if not self.output or not self.frames:
            return
        import imageio.v2 as imageio  # noqa: PLC0415

        imageio.mimsave(self.output, self.frames, fps=8, loop=0)
        print(f"wrote {len(self.frames)} frames to {self.output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--output",
        type=lambda x: None if x == "None" else str(x),
        default="example_poisson_disk_sampling.gif",
        help="Path to the output animated GIF.",
    )
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames in the sweep.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(output=args.output, num_frames=args.num_frames)
        for frame in range(args.num_frames):
            pts, radius = example.step(frame)
            example.render(frame, pts, radius)
        example.save()
