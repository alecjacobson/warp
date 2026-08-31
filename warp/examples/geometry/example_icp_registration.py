# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid Registration (Iterative Closest Point)
#
# Fits a real range scan onto a clean template with
# warp.geometry.register_rigid, the point-to-plane Gauss-Newton ICP. The data is
# the Stanford Bunny: the template is the zippered reconstruction (a triangle
# mesh) and the source is the raw "bun045" range scan (a partial, noisy point
# cloud from a single 45-degree view), downloaded on first run. The scan starts
# from a displaced pose and ICP slides it onto the template surface.
#
# Because the scan only partially overlaps the template, correspondences beyond
# ``max_corr_dist`` are rejected and a robust weight discounts the rest -- the
# stochastic/robust insight of Bouaziz et al. 2013. Because the motion is rigid,
# the template BVH is built once and queried each iteration, never rebuilt.
#
# Each frame runs one ICP iteration and renders the current scan points (orange)
# against the fixed template mesh (blue) with polyscope, writing an animated GIF
# headless (offscreen via EGL), so no display is needed.
#
#   uv run --with polyscope --with imageio \
#       warp/examples/geometry/example_icp_registration.py
###########################################################################

import os
import tarfile
import tempfile
import urllib.request

import numpy as np

import warp as wp
import warp.geometry

_BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
_TEMPLATE_REL = os.path.join("bunny", "reconstruction", "bun_zipper_res2.ply")
_SCAN_REL = os.path.join("bunny", "data", "bun045.ply")


def read_ply(path):
    """Read an ASCII PLY file, returning ``(vertices, faces)``.

    ``faces`` is ``None`` when the file has no face element. Non-position vertex
    properties (confidence, intensity, ...) and non-triangle elements (e.g. the
    Stanford ``range_grid``) are skipped.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        elements = []  # (name, count)
        fmt = None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, count = line.split()
                elements.append((name, int(count)))
            elif line == "end_header":
                break
        if fmt != "ascii":
            raise ValueError(f"{path}: only ASCII PLY is supported (got {fmt})")

        vertices, faces = None, None
        for name, count in elements:
            if name == "vertex":
                vertices = np.empty((count, 3), np.float32)
                for i in range(count):
                    values = f.readline().split()
                    vertices[i] = (float(values[0]), float(values[1]), float(values[2]))
            elif name == "face":
                triangles = []
                for _ in range(count):
                    values = f.readline().split()
                    k = int(values[0])
                    idx = [int(x) for x in values[1 : 1 + k]]
                    for j in range(1, k - 1):  # fan-triangulate polygons
                        triangles.append((idx[0], idx[j], idx[j + 1]))
                faces = np.array(triangles, np.int32)
            else:
                for _ in range(count):
                    f.readline()
        return vertices, faces


def ensure_stanford_bunny(cache_dir):
    """Download and extract the Stanford Bunny scans, returning
    ``(template_path, scan_path)``. Cached under ``cache_dir`` after the first run."""
    template = os.path.join(cache_dir, _TEMPLATE_REL)
    scan = os.path.join(cache_dir, _SCAN_REL)
    if not (os.path.exists(template) and os.path.exists(scan)):
        os.makedirs(cache_dir, exist_ok=True)
        archive = os.path.join(cache_dir, "bunny.tar.gz")
        print(f"Downloading Stanford Bunny from {_BUNNY_URL} ...")
        urllib.request.urlretrieve(_BUNNY_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(cache_dir)
    return template, scan


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
    def __init__(self, output="example_icp_registration.gif", num_iters=32, num_points=8000, cache_dir=None):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.output = output
        self.num_iters = num_iters

        cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "warp_stanford_bunny")
        template_path, scan_path = ensure_stanford_bunny(cache_dir)
        self.points, self.faces = read_ply(template_path)
        scan, _ = read_ply(scan_path)

        self.target = wp.Mesh(
            points=wp.array(self.points, dtype=wp.vec3),
            indices=wp.array(self.faces.reshape(-1), dtype=wp.int32),
        )

        # Subsample the raw scan, then displace it from the template pose. The
        # rotation is about the template center so the offset reads as a pose, not
        # a drift.
        rng = np.random.default_rng(0)
        scan = scan[rng.choice(len(scan), min(num_points, len(scan)), replace=False)]
        self.center = 0.5 * (self.points.min(0) + self.points.max(0))
        self.misalign = _rigid(12.0, (0.2, 1.0, 0.1), (0.07, 0.01, -0.04))
        self.source = ((scan - self.center) @ self.misalign[:3, :3].T + self.center + self.misalign[:3, 3]).astype(
            np.float32
        )

        # Partial overlap: reject far correspondences and robustly weight the rest.
        self.max_corr_dist = 0.05

        self.estimate = np.eye(4)  # current source-to-template transform
        self.rmse = float("inf")

        ps.set_allow_headless_backends(True)
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("y_up")
        ps.set_SSAA_factor(4)
        ps.set_window_size(1280, 720)

        mesh = ps.register_surface_mesh("template", self.points, self.faces, smooth_shade=True)
        mesh.set_color((0.2, 0.3, 0.8))  # gptoolbox blue

        self.frames = []

    def step(self):
        # One ICP iteration, continuing from the current estimate. The template
        # wp.Mesh is reused (no BVH rebuild).
        result = warp.geometry.register_rigid(
            self.source,
            self.target,
            init=self.estimate,
            max_iters=1,
            tol=0.0,
            max_corr_dist=self.max_corr_dist,
            robust="welsch",
        )
        self.estimate = result.transform
        self.rmse = result.rmse

    def render(self):
        ps = self.ps
        moved = self.source @ self.estimate[:3, :3].T + self.estimate[:3, 3]
        cloud = ps.register_point_cloud("scan", moved)
        cloud.set_color((1.0, 0.55, 0.15))  # gptoolbox orange
        cloud.set_radius(0.0016, relative=False)

        lo, hi = self.points.min(0), self.points.max(0)
        center = 0.5 * (lo + hi)
        diag = float(np.linalg.norm(hi - lo))
        direction = np.array([0.4, 0.25, 1.0])
        direction /= np.linalg.norm(direction)
        ps.look_at(tuple(float(x) for x in center + direction * 1.6 * diag), tuple(float(x) for x in center))

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
    parser.add_argument("--num-iters", type=int, default=32, help="Number of ICP iterations (frames).")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory to cache the downloaded scans.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(output=args.output, num_iters=args.num_iters, cache_dir=args.data_dir)
        example.render()  # initial displaced pose
        for _ in range(args.num_iters):
            example.step()
            example.render()
        example.save()
