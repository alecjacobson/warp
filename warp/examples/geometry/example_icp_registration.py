# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid Registration (Iterative Closest Point)
#
# Assembles several real range scans onto a clean template with
# warp.geometry.register_rigid_batched, the batched point-to-plane Gauss-Newton
# ICP. The data is the Stanford Bunny: the template is the zippered
# reconstruction (a triangle mesh) and the sources are the raw single-view range
# scans (partial, noisy point clouds taken from different angles), downloaded on
# first run.
#
# Each scan's orientation is unknown, so every scan is registered from K
# different rotational initializations at once (one batched solve) and the best
# of the K -- the lowest-residual fit -- is kept. The animation then replays,
# simultaneously, the best trajectory for each of the N scans: they swing in from
# their winning initializations and settle onto the surface, each covering the
# part of the bunny its camera saw, together tiling the whole model.
#
# Because the scans only partially overlap the template, correspondences beyond
# ``max_corr_dist`` are rejected and a robust weight discounts the rest (Bouaziz
# et al. 2013); because the motion is rigid, the template BVH is built once and
# shared by every problem in the batch, never rebuilt. Rendering runs headless
# (offscreen via EGL) with polyscope, writing an animated GIF, so no display is
# needed.
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
# Single-view range scans that together cover the whole bunny.
_SCAN_VIEWS = ["bun000", "bun045", "bun090", "bun180", "bun270", "bun315"]
_SCAN_REL = os.path.join("bunny", "data", "bun045.ply")  # presence check for the download


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
    """Download and extract the Stanford Bunny, returning the template mesh path.
    The single-view scans live alongside it under ``bunny/data/``. Cached under
    ``cache_dir`` after the first run."""
    template = os.path.join(cache_dir, _TEMPLATE_REL)
    scan = os.path.join(cache_dir, _SCAN_REL)
    if not (os.path.exists(template) and os.path.exists(scan)):
        os.makedirs(cache_dir, exist_ok=True)
        archive = os.path.join(cache_dir, "bunny.tar.gz")
        print(f"Downloading Stanford Bunny from {_BUNNY_URL} ...")
        urllib.request.urlretrieve(_BUNNY_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(cache_dir)
    return template


def _yaw_about(center, deg):
    """A rigid transform rotating by ``deg`` about the vertical (y) axis through
    ``center``."""
    t = np.radians(deg)
    r = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = center - r @ center
    return T


def _hue_colors(n):
    """``n`` evenly spaced, saturated RGB colors around the hue wheel."""
    h = (np.arange(n) / n) * 6.0
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    z = np.zeros(n)
    o = np.ones(n)
    table = np.array([[o, x, z], [x, o, z], [z, o, x], [z, x, o], [x, z, o], [o, z, x]])
    rgb = table[np.clip(h.astype(int), 0, 5), :, np.arange(n)]
    return 0.25 + 0.72 * rgb  # lift off pure black/white so every cloud reads


class Example:
    def __init__(
        self, output="example_icp_registration.gif", num_iters=34, num_points=4000, num_inits=8, cache_dir=None
    ):
        import polyscope as ps  # noqa: PLC0415

        self.ps = ps
        self.output = output
        self.num_iters = num_iters

        cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "warp_stanford_bunny")
        template_path = ensure_stanford_bunny(cache_dir)
        self.points, self.faces = read_ply(template_path)
        center = 0.5 * (self.points.min(0) + self.points.max(0))

        # Build the template mesh once and share it across every scan and init.
        self.target = wp.Mesh(
            points=wp.array(self.points, dtype=wp.vec3),
            indices=wp.array(self.faces.reshape(-1), dtype=wp.int32),
        )
        # Partial overlap: reject far correspondences and robustly weight the rest.
        self.max_corr_dist = 0.1

        # Load the single-view scans and subsample each to a common point count.
        rng = np.random.default_rng(0)
        data_dir = os.path.join(cache_dir, "bunny", "data")
        scans = []
        for view in _SCAN_VIEWS:
            pts, _ = read_ply(os.path.join(data_dir, view + ".ply"))
            pts = pts[rng.choice(len(pts), min(num_points, len(pts)), replace=False)]
            scans.append(pts.astype(np.float32))
        count = min(len(s) for s in scans)
        self.sources = np.stack([s[:count] for s in scans])  # (num_scans, count, 3)
        self.num_scans = len(scans)

        # Each scan's orientation is unknown, so try K rotations about the vertical
        # axis and keep whichever converges best -- multi-initialization.
        inits = np.stack([_yaw_about(center, a) for a in np.linspace(0.0, 360.0, num_inits, endpoint=False)])
        best_inits = []
        for s in range(self.num_scans):
            result = warp.geometry.register_rigid_batched(
                self.sources[s], self.target, inits, max_iters=60, tol=1e-8,
                max_corr_dist=self.max_corr_dist, robust="welsch",
            )  # fmt: skip
            best_inits.append(inits[result.best_index])
        self.transforms = np.stack(best_inits)  # animate from each scan's winning init

        self.colors = _hue_colors(self.num_scans)

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
        # Advance every scan's best trajectory by one iteration together: a
        # multi-source batch, one problem per scan, sharing the fixed template
        # mesh (never rebuilt).
        result = warp.geometry.register_rigid_batched(
            self.sources,
            self.target,
            self.transforms,
            max_iters=1,
            tol=0.0,
            max_corr_dist=self.max_corr_dist,
            robust="welsch",
        )
        self.transforms = result.transforms
        self.rmse = result.rmse

    def render(self):
        ps = self.ps
        for s in range(self.num_scans):
            moved = self.sources[s] @ self.transforms[s, :3, :3].T + self.transforms[s, :3, 3]
            cloud = ps.register_point_cloud(f"scan_{s}", moved)
            cloud.set_color(tuple(float(c) for c in self.colors[s]))
            cloud.set_radius(0.0012, relative=False)

        lo, hi = self.points.min(0), self.points.max(0)
        center = 0.5 * (lo + hi)
        diag = float(np.linalg.norm(hi - lo))
        direction = np.array([0.4, 0.25, 1.0])
        direction /= np.linalg.norm(direction)
        ps.look_at(tuple(float(x) for x in center + direction * 1.7 * diag), tuple(float(x) for x in center))

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
        converged = int((self.rmse < 5e-4).sum())
        print(
            f"wrote {self.output} ({len(frames)} frames); "
            f"{converged}/{self.num_scans} scans registered, best rmse={self.rmse.min():.2e}"
        )


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
    parser.add_argument("--num-iters", type=int, default=34, help="Number of ICP iterations (frames).")
    parser.add_argument("--num-inits", type=int, default=8, help="Rotational initializations tried per scan (K).")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory to cache the downloaded scans.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            output=args.output, num_iters=args.num_iters, num_inits=args.num_inits, cache_dir=args.data_dir
        )
        example.render()  # initial poses (each scan at its winning rotation)
        for _ in range(args.num_iters):
            example.step()
            example.render()
        example.save()
