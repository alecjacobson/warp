# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid Registration (Iterative Closest Point)
#
# Aligns a misaligned source point cloud onto a target mesh with
# warp.geometry.register_rigid, the point-to-plane Gauss-Newton ICP. The source
# is a set of points sampled on the Stanford bunny, then displaced by a known
# rigid transform; ICP recovers the inverse and slides the points back onto the
# surface.
#
# Each frame runs one ICP iteration (continuing from the previous estimate) and
# renders the current source points (orange) against the fixed target mesh
# (blue) with polyscope, writing an animated GIF. Rendering runs headless
# (offscreen via EGL), so no display is needed. Because the motion is rigid, the
# target BVH is built once and never rebuilt across iterations.
#
#   uv run --with usd-core --with polyscope --with imageio \
#       warp/examples/geometry/example_icp_registration.py
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
            return (
                np.array(m.GetPointsAttr().Get(), dtype=np.float32),
                np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32),
            )
    raise RuntimeError("no mesh found in bunny.usd")


def _rigid(rot_deg, axis, trans):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = trans
    return T


class Example:
    def __init__(self, output="example_icp_registration.gif", num_iters=28, num_points=4000):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.output = output
        self.num_iters = num_iters

        self.points, self.faces = load_bunny()
        self.target = wp.Mesh(
            points=wp.array(self.points, dtype=wp.vec3),
            indices=wp.array(self.faces, dtype=wp.int32),
        )

        # Source: points sampled on the bunny, then displaced by a known transform.
        sampler = warp.geometry.UniformSampler(self.points, self.faces)
        on_surface = sampler.sample_points(num_points, seed=0).numpy()
        self.misalign = _rigid(45.0, (0.3, 1.0, 0.2), (0.35, 0.1, -0.25))
        self.source = (on_surface @ self.misalign[:3, :3].T + self.misalign[:3, 3]).astype(np.float32)

        self.estimate = np.eye(4)  # current source-to-target transform
        self.rmse = float("inf")

        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("y_up")
        ps.set_SSAA_factor(4)
        ps.set_window_size(1280, 720)

        mesh = ps.register_surface_mesh("target", self.points, self.faces.reshape(-1, 3), smooth_shade=True)
        mesh.set_color((0.2, 0.3, 0.8))  # gptoolbox blue
        mesh.set_transparency(0.55)

        self.frames = []

    def step(self):
        # One ICP iteration, continuing from the current estimate. The target
        # wp.Mesh is reused (no BVH rebuild).
        result = warp.geometry.register_rigid(self.source, self.target, init=self.estimate, max_iters=1, tol=0.0)
        self.estimate = result.transform
        self.rmse = result.rmse

    def render(self, frame):
        ps = self.ps
        moved = self.source @ self.estimate[:3, :3].T + self.estimate[:3, 3]
        cloud = ps.register_point_cloud("source", moved)
        cloud.set_color((1.0, 0.7, 0.2))  # gptoolbox orange
        cloud.set_radius(0.006, relative=False)

        lo, hi = self.points.min(0), self.points.max(0)
        center = 0.5 * (lo + hi)
        diag = float(np.linalg.norm(hi - lo))
        direction = np.array([0.5, 0.35, 1.0])
        direction /= np.linalg.norm(direction)
        ps.look_at(tuple(float(x) for x in center + direction * 1.4 * diag), tuple(float(x) for x in center))

        buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=False))
        if buf.dtype != np.uint8:
            buf = (np.clip(buf, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.frames.append(buf[:, :, :3])

    def save(self):
        if not self.output or not self.frames:
            return
        import imageio.v2 as imageio  # noqa: PLC0415

        # Hold the first and last frames a moment so the loop reads clearly.
        frames = [self.frames[0]] * 4 + self.frames + [self.frames[-1]] * 6
        imageio.mimsave(self.output, frames, fps=8, loop=0)
        print(f"wrote {self.output} ({len(frames)} frames), final rmse={self.rmse:.2e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--output",
        type=lambda x: None if x == "None" else str(x),
        default="example_icp_registration.gif",
        help="Path to the output animated GIF.",
    )
    parser.add_argument("--num-iters", type=int, default=28, help="Number of ICP iterations (frames).")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(output=args.output, num_iters=args.num_iters)
        example.render(0)  # initial misaligned state
        for frame in range(args.num_iters):
            example.step()
            example.render(frame + 1)
        example.save()
