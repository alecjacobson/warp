# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Uniform Surface Sampling
#
# Draws points uniformly over the surface of a triangle mesh with
# warp.geometry.UniformSampler.
#
# A torus is built procedurally with deliberately non-uniform tessellation
# (triangles on the inner rim are far smaller than those on the outer rim). The
# sampler weights each triangle by its area, so the drawn points stay evenly
# spread across the surface regardless of that tessellation. Each frame draws a
# fresh set of points from within a kernel by calling the sampler's `draw`
# member function, evaluating world-space positions on the GPU.
#
#   uv run --with usd-core warp/examples/geometry/example_uniform_sampling.py
###########################################################################

import numpy as np

import warp as wp
import warp.geometry
import warp.render


def build_torus(num_major=48, num_minor=24, major_radius=1.0, minor_radius=0.35):
    """Build a torus whose triangles vary widely in area.

    The minor-circle vertices are spaced non-uniformly (clustered on the inner
    rim), so a naive per-vertex or per-triangle sampling would over-sample the
    dense inner region. Area weighting is what keeps the samples uniform.
    """
    u = np.linspace(0.0, 2.0 * np.pi, num_major, endpoint=False)
    # Bias the minor angle so vertices bunch up near the inner rim (theta = pi).
    s = np.linspace(0.0, 1.0, num_minor, endpoint=False)
    v = 2.0 * np.pi * (s + 0.35 * np.sin(2.0 * np.pi * s) / (2.0 * np.pi))

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
    faces = np.array(faces, dtype=np.int32)
    return points, faces


@wp.kernel(enable_backward=False)
def sample_points_kernel(
    sampler: warp.geometry.UniformSamplerState,
    seed: int,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)

    # Draw one uniform surface sample and evaluate its world-space position.
    s = warp.geometry.draw(sampler, rng)
    out_points[tid] = wp.mesh_eval_position(sampler.mesh, s.face, s.uv[0], s.uv[1])


class Example:
    def __init__(self, stage_path="example_uniform_sampling.usd", num_samples=4000):
        points, faces = build_torus()
        self.sampler = warp.geometry.UniformSampler(
            points=wp.array(points, dtype=wp.vec3),
            faces=wp.array(faces, dtype=wp.int32),
        )

        self.points = wp.empty(shape=(num_samples,), dtype=wp.vec3)

        self.fps = 4
        self.frame = 0

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path, fps=self.fps)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            wp.launch(
                sample_points_kernel,
                dim=self.points.shape,
                inputs=(self.sampler.state, self.frame),
                outputs=(self.points,),
            )
            self.frame += 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.frame / self.fps)
            self.renderer.render_mesh(
                name="mesh",
                points=self.sampler.mesh.points.numpy(),
                indices=self.sampler.mesh.indices.numpy(),
                colors=(0.35, 0.55, 0.9),
            )
            self.renderer.render_points(name="samples", points=self.points.numpy(), radius=0.01, colors=(0.9, 0.4, 0.2))
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_uniform_sampling.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=16, help="Total number of frames.")
    parser.add_argument("--num-samples", type=int, default=4000, help="Number of points to draw each frame.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_samples=args.num_samples)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
