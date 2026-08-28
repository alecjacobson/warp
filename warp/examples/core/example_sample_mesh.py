# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sample Mesh
#
# Shows how to sample points uniformly on a mesh's surface with
# warp.geometry.UniformSampler.
#
# The sampler weights each triangle by its area, so points are spread evenly
# across the surface even when the tessellation is non-uniform. It precomputes a
# cumulative area distribution once; each draw is then a binary search plus a
# within-triangle sample. Here the per-sample device function is called inside a
# kernel via the sampler's `draw` member so that positions are evaluated on the
# GPU without a round trip.
#
###########################################################################

import numpy as np

import warp as wp
import warp.geometry
import warp.render

# fmt: off
POINTS = np.array(
    (
        (-0.986598, -0.400638, -0.175759), (-0.81036 , -0.482105, -0.541125),
        (-1.079616,  0.022652, -0.023381), (-0.894468, -0.080795, -0.618379),
        (-0.607365, -0.702012, -0.556551), (-0.366107, -0.800096, -0.620734),
        (-0.801777, -0.690991, -0.239593), (-0.553576, -0.871746, -0.335518),
        (-0.309133, -0.370805, -0.965784), (-0.288299, -0.956987, -0.402091),
        (-0.051878, -0.894342, -0.597583), (-0.386774, -1.003107, -0.145116),
        (-0.19062 , -1.061165,  0.012418), (-0.176053, -1.044838, -0.217194),
        ( 0.001479, -1.020045, -0.356905), (-0.105375, -0.655117, -0.861365),
        (-0.542102, -0.517255, -0.795259), (-0.476599, -0.105709, -0.981171),
        (-1.047915, -0.121584,  0.322098), (-0.527852,  0.137252,  0.501813),
        (-0.721762, -0.803275,  0.117162), (-0.904992, -0.573281,  0.168408),
        (-0.796762, -0.473428,  0.569649), (-0.606446, -0.753374,  0.492938),
        (-0.466481, -0.576566,  0.802562), (-0.50476 , -0.908596,  0.300064),
        (-0.337425, -1.008902,  0.170911), (-0.048676, -1.055594,  0.246732),
        (-0.212871, -0.760442,  0.738447), (-0.281356, -0.9322  ,  0.474965),
        (-0.560476,  0.062512, -0.561019), (-0.003252,  0.083237, -1.049784),
        (-0.009392,  0.593703, -0.522479), (-0.530465,  0.577231,  0.007172),
        (-0.02106 ,  0.064189,  1.066722), (-0.003512,  0.59714 ,  0.516904),
        ( 0.000194,  1.093899,  0.001113), ( 0.256861, -0.955856, -0.445325),
        ( 0.251205, -1.038759, -0.174212), ( 0.170201, -0.800019, -0.712158),
        ( 0.364385, -0.560298, -0.866843), ( 0.092809, -0.269437, -1.058467),
        ( 0.628127, -0.12359 , -0.9012  ), ( 0.507433, -0.930658, -0.215908),
        ( 0.496448, -0.800205, -0.545904), ( 0.757415, -0.527449, -0.565395),
        ( 0.908704, -0.596257,  0.028995), ( 0.754069, -0.731365, -0.256687),
        ( 0.921362, -0.09028 , -0.546421), ( 1.017846, -0.335787, -0.263017),
        ( 0.016768, -1.080014, -0.058473), ( 0.204245, -1.056388,  0.078346),
        ( 0.260892, -1.001704,  0.322104), ( 0.16608 , -0.739172,  0.788097),
        ( 0.021091, -0.931327,  0.557789), (-0.046158, -0.408417,  1.011046),
        ( 0.429623, -0.987237,  0.088537), ( 0.704993, -0.739396,  0.386838),
        ( 0.37277 , -0.825639,  0.591102), ( 0.493947, -0.896091,  0.339163),
        ( 0.321112, -0.540547,  0.890161), ( 0.654753, -0.520495,  0.690104),
        ( 0.922472, -0.124429,  0.530498), ( 0.662544, -0.85601 ,  0.054375),
        ( 0.950976, -0.422783,  0.327726), ( 0.536849,  0.109943, -0.52279 ),
        ( 0.517242,  0.120634,  0.535708), ( 0.532707,  0.598943, -0.000767),
        ( 1.086691,  0.048722,  0.032517), ( 0.528734, -0.109809,  0.96863 ),
        (-0.581832, -0.916941, -0.027829), (-0.625071, -0.14445 ,  0.906538),
    ),
    dtype=np.float32,
)

FACE_VERTEX_INDICES = np.array(
    (
         6,  0,  1,  6, 21,  0,  2,  0, 18,  0,  3,  1,  2,  3,  0,  5,
         7,  4, 70,  7, 11,  4,  6,  1, 16,  1,  3,  7,  6,  4,  4,  1,
        16,  9,  7,  5,  3, 17, 16, 16, 17,  8, 41,  8, 17, 30, 17,  3,
        10, 14,  9,  5, 10,  9, 10, 37, 14, 15, 10,  5,  7,  9, 11, 11,
         9, 13, 11, 13, 12, 50, 12, 13,  9, 14, 13, 15, 16,  8, 15,  8,
        41, 16,  5,  4, 16, 15,  5, 17, 31, 41, 21, 22, 18, 20, 21,  6,
        18,  0, 21, 20, 25, 23, 20, 70, 25, 70, 11, 26, 26, 25, 70, 25,
        29, 23, 21, 20, 23, 21, 23, 22, 23, 24, 22, 24, 71, 22, 26, 29,
        25, 26, 11, 12, 12, 27, 26, 26, 27, 29, 27, 54, 29, 27, 12, 50,
        28, 29, 54, 54, 53, 28, 23, 28, 24, 29, 28, 23, 28, 55, 24, 28,
        53, 55, 53, 60, 55, 24, 55, 71, 55, 34, 71, 30,  3,  2,  2, 33,
        30, 17, 30, 31, 32, 31, 30, 33, 36, 32, 19, 33,  2, 19, 35, 33,
        19, 71, 34, 35, 19, 34, 34, 66, 35, 35, 36, 33, 35, 67, 36, 15,
        39, 10, 10, 39, 37, 44, 37, 39, 14, 50, 13, 14, 38, 50, 14, 37,
        38, 37, 43, 38, 40, 15, 41, 40, 39, 15, 41, 42, 40, 44, 39, 40,
        31, 42, 41, 38, 43, 56, 44, 43, 37, 44, 47, 43, 47, 63, 43, 44,
        40, 45, 42, 45, 40, 46, 63, 47, 45, 47, 44, 65, 48, 42, 46, 47,
        49, 49, 47, 45, 48, 45, 42, 45, 48, 49, 68, 49, 48, 27, 52, 54,
        50, 51, 27, 27, 51, 52, 50, 38, 51, 38, 56, 51, 51, 56, 52, 54,
        52, 58, 52, 59, 58, 53, 54, 58, 60, 69, 55, 55, 69, 34, 43, 63,
        56, 59, 52, 56, 63, 59, 56, 63, 57, 59, 58, 60, 53, 57, 58, 59,
        58, 57, 61, 60, 58, 61, 57, 64, 61, 62, 61, 64, 60, 61, 69, 62,
        69, 61, 46, 57, 63, 64, 57, 46, 46, 49, 64, 68, 64, 49, 62, 64,
        68, 32, 65, 31, 65, 32, 67, 32, 36, 67, 65, 42, 31, 67, 68, 65,
        48, 65, 68, 34, 69, 66, 67, 35, 66, 68, 66, 62, 66, 69, 62, 67,
        66, 68, 33, 32, 30, 19,  2, 18, 20,  6, 70,  7, 70,  6, 18, 71,
        19, 22, 71, 18,
    ),
    dtype=np.int32,
)
# fmt: on


@wp.kernel(enable_backward=False)
def sample_mesh(
    sampler: warp.geometry.UniformSamplerState,
    seed: int,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    rng = wp.rand_init(seed, tid)

    # Draw a point uniformly on the surface: a triangle chosen with probability
    # proportional to its area, plus a uniform location within that triangle.
    s = warp.geometry.draw(sampler, rng)

    # Evaluate the world-space position from the sampled face and barycentrics.
    out_points[tid] = wp.mesh_eval_position(sampler.mesh, s.face, s.uv[0], s.uv[1])


class Example:
    def __init__(self, stage_path="example_sample_mesh.usd"):
        # Build a uniform surface sampler. This constructs the mesh and its
        # cumulative area distribution once, up front.
        self.sampler = warp.geometry.UniformSampler(
            points=wp.array(POINTS, dtype=wp.vec3),
            faces=wp.array(FACE_VERTEX_INDICES, dtype=wp.int32),
        )

        # Array to store the sampled points.
        self.points = wp.empty(shape=(100,), dtype=wp.vec3)

        self.fps = 4
        self.frame = 0

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path, fps=self.fps)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            # Sample new points on the mesh, using the current frame number as
            # the seed so each frame draws a fresh set.
            wp.launch(
                sample_mesh,
                dim=self.points.shape,
                inputs=(
                    self.sampler.state,
                    self.frame,
                ),
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
            self.renderer.render_points(name="points", points=self.points.numpy(), radius=0.05, colors=(0.8, 0.3, 0.2))
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sample_mesh.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=16, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
