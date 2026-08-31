# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the fixed ICP benchmark problem shared by every baseline.

An ellipsoid (a scaled icosphere, so rotation is recoverable) is displaced by a
known rigid transform plus mild sensor noise; the source registers back onto the
target. Writes ``problem.npz`` (source, target, faces, target normals, ground
truth) into ``common.BENCH_DIR`` and text copies of the clouds for the PCL
binary.

    uv run --with scipy tools/benchmarks/icp_baselines/problem.py
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import BENCH_DIR, PROBLEM_PATH


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
    return (v * np.array(scale)).astype(np.float64), np.array(faces, np.int32)


def transform(rot_deg, axis, trans):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    T[:3, 3] = trans
    return T


def pca_normals(points, k=16):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.empty_like(points)
    for i in range(len(points)):
        nbr = points[idx[i]] - points[idx[i]].mean(0)
        _, vecs = np.linalg.eigh(nbr.T @ nbr)
        normals[i] = vecs[:, 0]
    return normals


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    target, faces = icosphere(subdiv=5, scale=(1.5, 1.0, 0.7))  # ~10k points
    gt = transform(15.0, (0.2, 1.0, 0.3), (0.08, -0.05, 0.06))
    rng = np.random.default_rng(0)
    source = target @ gt[:3, :3].T + gt[:3, 3] + rng.standard_normal(target.shape) * 0.002
    normals = pca_normals(target)

    np.savez(
        PROBLEM_PATH, source=source, target=target, faces=faces,
        normals=normals, gt=gt, gt_inv=np.linalg.inv(gt), max_dist=0.3,
    )  # fmt: skip
    np.savetxt(os.path.join(BENCH_DIR, "source.xyz"), source, fmt="%.7f")
    np.savetxt(os.path.join(BENCH_DIR, "target.xyz"), target, fmt="%.7f")
    print(f"wrote {PROBLEM_PATH}: {len(target)} pts, gt rot=15deg |t|={np.linalg.norm(gt[:3, 3]):.3f}, noise=2e-3")


if __name__ == "__main__":
    main()
