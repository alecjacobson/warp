# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ablation study of warp.geometry.register_rigid options.

Answers, with numbers, three questions:

1. Are computed normals (surface / PCA) worth it versus the query-closest
   direction ``normalize(p - q)`` (``plane_normal="closest_point"``)?
2. Does each variant / option (symmetric, robust weighting, subsampling) actually
   help, and where?
3. What are the best settings for each target type and data condition?

Everything runs on a scaled icosphere (an ellipsoid, so rotation is
recoverable), as both a mesh target and a point-cloud target, under four data
conditions: clean, noisy, outliers, and partial overlap. Accuracy is measured
against the known ground-truth transform; a run "succeeds" when the rotation
error is under 1 degree.

    uv run tools/benchmarks/icp_ablation.py
"""

import time

import numpy as np

import warp as wp
import warp.geometry as geo

wp.init()
DEVICE = "cuda:0"
SUCCESS_DEG = 1.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def icosphere(subdiv, scale):
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
    return (v * np.array(scale)).astype(np.float64), np.array(faces, np.int32).reshape(-1)


def rot_matrix(rot_deg, axis):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)


def transform(rot_deg, axis, trans):
    T = np.eye(4)
    T[:3, :3] = rot_matrix(rot_deg, axis)
    T[:3, 3] = trans
    return T


def rot_err(a, b):
    c = (np.trace(a[:3, :3] @ b[:3, :3].T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Scenarios: return (source, gt) where source ~ gt applied to the target verts.
# --------------------------------------------------------------------------- #
VERTS, FACES = icosphere(subdiv=5, scale=(1.5, 1.0, 0.7))  # ~10k points
DIAG = float(np.linalg.norm(VERTS.max(0) - VERTS.min(0)))
GT = transform(15.0, (0.2, 1.0, 0.3), (0.08, -0.05, 0.06))
GT_INV = np.linalg.inv(GT)


def scenario(name, seed=0):
    rng = np.random.default_rng(seed)
    pts = VERTS.copy()
    if name == "clean":
        src = pts @ GT[:3, :3].T + GT[:3, 3] + rng.standard_normal(pts.shape) * (2e-4 * DIAG)
    elif name == "noisy":
        src = pts @ GT[:3, :3].T + GT[:3, 3] + rng.standard_normal(pts.shape) * (0.02 * DIAG)
    elif name == "outliers":
        src = pts @ GT[:3, :3].T + GT[:3, 3]
        m = int(0.2 * len(src))
        src[rng.choice(len(src), m, replace=False)] = rng.standard_normal((m, 3)) * (0.4 * DIAG)
    elif name == "partial":
        keep = pts[:, 0] > np.median(pts[:, 0])  # a ~half-view of the surface
        src = (pts[keep] @ GT[:3, :3].T + GT[:3, 3]) + rng.standard_normal((keep.sum(), 3)) * (5e-3 * DIAG)
    else:
        raise ValueError(name)
    return src.astype(np.float32)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def solve(source, target, *, init=None, repeats=3, **kw):
    init = np.eye(4) if init is None else init
    best_ms = np.inf
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = geo.register_rigid(source, target, init=init, device=DEVICE, **kw)
        wp.synchronize_device()
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1e3)
    T = result.transform
    return {
        "rot": rot_err(T, GT_INV),
        "trans": float(np.linalg.norm(T[:3, 3] - GT_INV[:3, 3])),
        "iters": result.iterations,
        "ms": best_ms,
        "rmse": result.rmse,
    }


def mesh_target():
    return (VERTS.astype(np.float32), FACES)


def cloud_target():
    return VERTS.astype(np.float32)


def row(label, r):
    ok = "ok " if r["rot"] < SUCCESS_DEG else "MISS"
    print(f"  {label:<34}{r['rot']:>9.4f}{r['trans']:>11.5f}{r['iters']:>7}{r['ms']:>9.2f}   {ok}")


def header(cols=("rot deg", "trans", "iters", "ms")):
    print(f"  {'config':<34}{cols[0]:>9}{cols[1]:>11}{cols[2]:>7}{cols[3]:>9}")
    print("  " + "-" * 74)


# --------------------------------------------------------------------------- #
def exp1_normals():
    print("\n=== Q1: computed normals vs closest-point direction ===")
    print("point-to-plane, max_corr_dist=0.3, 100 iters max\n")
    for cond in ("clean", "noisy", "partial"):
        src = scenario(cond)
        print(f"[{cond}]")
        header()
        for tname, tgt in (("mesh", mesh_target()), ("cloud", cloud_target())):
            for on in ("surface", "closest_point"):
                r = solve(src, tgt, max_iters=100, tol=1e-8, max_corr_dist=0.3, plane_normal=on)
                row(f"{tname} / {on}", r)
        print()


def exp2_basin():
    print("\n=== Q2: convergence basin (max recoverable rotation) ===")
    print("largest single-axis init rotation still reaching < 1 deg error\n")
    angles = list(range(0, 91, 5))
    configs = [
        ("mesh  point_to_plane/surface", mesh_target(), {"plane_normal": "surface"}),
        ("mesh  point_to_plane/closest", mesh_target(), {"plane_normal": "closest_point"}),
        ("mesh  symmetric", mesh_target(), {"variant": "symmetric"}),
        ("cloud point_to_plane/surface", cloud_target(), {"plane_normal": "surface"}),
        ("cloud point_to_plane/closest", cloud_target(), {"plane_normal": "closest_point"}),
        ("cloud symmetric", cloud_target(), {"variant": "symmetric"}),
    ]
    src = scenario("clean")
    print(f"  {'config':<34}{'max basin deg':>14}")
    print("  " + "-" * 50)
    for label, tgt, kw in configs:
        best = 0
        for a in angles:
            init = np.eye(4)
            init[:3, :3] = rot_matrix(a, (0.3, 0.7, 0.2))
            r = solve(src, tgt, init=init, repeats=1, max_iters=100, tol=1e-8, max_corr_dist=0.5, **kw)
            if r["rot"] < SUCCESS_DEG:
                best = a
            else:
                break
        print(f"  {label:<34}{best:>11} deg")


def exp3_robust_subsample():
    print("\n=== Q3a: robust weighting & subsampling (mesh target) ===\n")
    for cond in ("clean", "outliers"):
        src = scenario(cond)
        print(f"[{cond}]")
        header()
        row("plain", solve(src, mesh_target(), max_iters=80, tol=1e-8, max_corr_dist=0.3))
        row("robust=welsch", solve(src, mesh_target(), max_iters=80, tol=1e-8, max_corr_dist=0.3, robust="welsch"))
        row(
            "subsample 25%",
            solve(src, mesh_target(), max_iters=80, tol=1e-8, max_corr_dist=0.3, sample_count=len(src) // 4, seed=1),
        )
        combo = solve(
            src, mesh_target(), max_iters=80, tol=1e-8, max_corr_dist=0.3,
            robust="welsch", sample_count=len(src) // 4, seed=1,
        )  # fmt: skip
        row("robust + subsample 25%", combo)
        print()


def exp3b_robust_k():
    print("=== Q3b: robust_k sweep on 20% outliers (mesh) ===\n")
    src = scenario("outliers")
    header()
    for k in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        r = solve(src, mesh_target(), max_iters=80, tol=1e-8, max_corr_dist=0.3, robust="welsch", robust_k=k)
        row(f"robust_k={k}", r)
    print()


def exp4_maxdist():
    print("=== Q3c: max_corr_dist sweep on partial overlap (cloud) ===\n")
    src = scenario("partial")
    header()
    for mcd in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        r = solve(src, cloud_target(), max_iters=100, tol=1e-8, max_corr_dist=mcd)
        row(f"max_corr_dist={mcd}", r)
    print()


if __name__ == "__main__":
    print(
        f"Ablation on ellipsoid ({len(VERTS)} pts), gt rot=15deg |t|={np.linalg.norm(GT[:3, 3]):.3f}, device={DEVICE}"
    )
    exp1_normals()
    exp2_basin()
    exp3_robust_subsample()
    exp3b_robust_k()
    exp4_maxdist()
