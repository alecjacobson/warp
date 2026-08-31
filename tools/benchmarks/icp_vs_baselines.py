# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare warp.geometry rigid registration against external ICP baselines.

Registers a source point cloud onto a target (the source is the target displaced
by a known rigid transform, plus optional noise), so accuracy is measured
against ground truth: geodesic rotation error and translation error. Runtime is
the full solve, including whatever spatial structure each library builds
internally -- Warp's hash grid, the baselines' KD-trees -- which is the fair
comparison for a one-shot registration.

Baselines are optional and import-guarded; the harness records which are
installed and skips the rest:

* Open3D tensor API (``open3d``): point-to-plane and point-to-point ICP.
* PyTorch3D (``pytorch3d``): point-to-point (Umeyama) ICP.
* fast_gicp (``pygicp``): CUDA (V)GICP.
* PCL (``pcl`` / ``python-pcl``): CPU ICP.

    uv run --with open3d tools/benchmarks/icp_vs_baselines.py
    uv run --with open3d --with torch --with pytorch3d --with pygicp \
        tools/benchmarks/icp_vs_baselines.py
"""

import time

import numpy as np

import warp as wp
import warp.geometry as geo

wp.init()
DEVICE = "cuda:0"


# --------------------------------------------------------------------------- #
# Benchmark problem
# --------------------------------------------------------------------------- #
def icosphere(subdiv, scale):
    """Subdivided icosphere scaled into an ellipsoid (rotation is recoverable)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    verts = [np.array(v, dtype=np.float64) for v in verts]
    for _ in range(subdiv):
        mid: dict = {}
        new_faces = []

        def midpoint(a, b, mid=mid):
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((verts[a] + verts[b]) * 0.5)
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    v = np.array(verts)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * np.array(scale)).astype(np.float64)


def known_transform(rot_deg, axis, trans):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    T[:3, 3] = trans
    return T


def rotation_error_deg(a, b):
    c = (np.trace(a[:3, :3] @ b[:3, :3].T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def estimate_normals_np(points, k=16):
    """Simple KD-tree-free PCA normals for baselines that need supplied normals
    (brute force; fine for a few thousand points)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.empty_like(points)
    for i in range(len(points)):
        nbr = points[idx[i]] - points[idx[i]].mean(0)
        _, vecs = np.linalg.eigh(nbr.T @ nbr)
        normals[i] = vecs[:, 0]
    return normals


# --------------------------------------------------------------------------- #
# Method adapters: (source, target, target_normals, init, gt) -> (T, ms)
# --------------------------------------------------------------------------- #
def _timed(fn, repeats=3):
    best = np.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        wp.synchronize_device()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return out, best


def warp_point_to_plane(source, target, normals, init, max_dist):
    def run():
        r = geo.register_rigid(
            source.astype(np.float32), target.astype(np.float32), init=init,
            max_iters=50, tol=1e-8, max_corr_dist=max_dist, device=DEVICE,
        )  # fmt: skip
        return r.transform

    return _timed(run)


def warp_symmetric(source, target, normals, init, max_dist):
    def run():
        r = geo.register_rigid(
            source.astype(np.float32), (target.astype(np.float32), normals.astype(np.float32)),
            init=init, variant="symmetric", max_iters=50, tol=1e-8, max_corr_dist=max_dist, device=DEVICE,
        )  # fmt: skip
        return r.transform

    return _timed(run)


def open3d_adapter(estimation_name):
    import open3d as o3d  # noqa: PLC0415

    def adapter(source, target, normals, init, max_dist):
        dev = o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available() else o3d.core.Device("CPU:0")
        f32 = o3d.core.float32
        src = o3d.t.geometry.PointCloud(dev)
        src.point.positions = o3d.core.Tensor(source.astype(np.float32), f32, dev)
        tgt = o3d.t.geometry.PointCloud(dev)
        tgt.point.positions = o3d.core.Tensor(target.astype(np.float32), f32, dev)
        tgt.point.normals = o3d.core.Tensor(normals.astype(np.float32), f32, dev)
        if estimation_name == "point_to_plane":
            est = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(1e-8, 1e-8, 50)
        init_t = o3d.core.Tensor(init.astype(np.float32), f32, dev)

        def run():
            res = o3d.t.pipelines.registration.icp(src, tgt, max_dist, init_t, est, criteria)
            return res.transformation.cpu().numpy().astype(np.float64)

        return _timed(run)

    return adapter


def pytorch3d_adapter():
    import torch  # noqa: PLC0415
    from pytorch3d.ops import iterative_closest_point  # noqa: PLC0415

    def adapter(source, target, normals, init, max_dist):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        src = torch.tensor(source, dtype=torch.float32, device=dev)[None]
        tgt = torch.tensor(target, dtype=torch.float32, device=dev)[None]

        def run():
            sol = iterative_closest_point(src, tgt, max_iterations=50)
            T = np.eye(4)
            T[:3, :3] = sol.RTs.R[0].cpu().numpy().T  # pytorch3d applies X @ R
            T[:3, 3] = sol.RTs.T[0].cpu().numpy()
            return T

        return _timed(run)

    return adapter


def pygicp_adapter():
    import pygicp  # noqa: PLC0415

    def adapter(source, target, normals, init, max_dist):
        def run():
            return pygicp.align_points(target.astype(np.float64), source.astype(np.float64))

        return _timed(run)

    return adapter


def collect_methods():
    methods = {
        "warp_point_to_plane": warp_point_to_plane,
        "warp_symmetric": warp_symmetric,
    }
    availability = {}
    for name, factory in [
        ("open3d_point_to_plane", lambda: open3d_adapter("point_to_plane")),
        ("open3d_point_to_point", lambda: open3d_adapter("point_to_point")),
        ("pytorch3d_point_to_point", pytorch3d_adapter),
        ("fast_gicp", pygicp_adapter),
    ]:
        try:
            methods[name] = factory()
            availability[name] = "installed"
        except Exception as e:
            availability[name] = f"unavailable ({type(e).__name__})"
    return methods, availability


# --------------------------------------------------------------------------- #
def main():
    target = icosphere(subdiv=5, scale=(1.5, 1.0, 0.7))  # ~10k points
    gt = known_transform(15.0, (0.2, 1.0, 0.3), (0.08, -0.05, 0.06))
    rng = np.random.default_rng(0)
    source = target @ gt[:3, :3].T + gt[:3, 3]
    source += rng.standard_normal(source.shape) * 0.002  # mild sensor noise
    gt_inv = np.linalg.inv(gt)  # source -> target

    normals = estimate_normals_np(target)
    max_dist = 0.3
    init = np.eye(4)

    print(f"Benchmark: {len(target)} points, rot=15deg, |t|={np.linalg.norm(gt[:3, 3]):.3f}, noise=2e-3")
    print(f"Warp device: {DEVICE}")
    try:
        import open3d as o3d  # noqa: PLC0415

        o3d_dev = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
        print(f"Open3D device: {o3d_dev} (pip wheels are often CPU-only; a CUDA build would be faster)")
    except ImportError:
        pass
    print()

    methods, availability = collect_methods()
    print(f"{'method':<28}{'rot err (deg)':>14}{'trans err':>12}{'time (ms)':>12}")
    print("-" * 66)
    for name, fn in methods.items():
        try:
            T, ms = fn(source, target, normals, init, max_dist)
            print(
                f"{name:<28}{rotation_error_deg(T, gt_inv):>14.4f}{np.linalg.norm(T[:3, 3] - gt_inv[:3, 3]):>12.5f}{ms:>12.2f}"
            )
        except Exception as e:
            print(f"{name:<28}{'FAILED':>14}  {type(e).__name__}: {e}")

    print("\nBaseline availability:")
    for name, status in availability.items():
        print(f"  {name:<28}{status}")
    print("  pcl                         adapter not implemented (install python-pcl to add)")


if __name__ == "__main__":
    main()
