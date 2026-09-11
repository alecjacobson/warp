# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Poisson-disk benchmark: mesh loading, radii, timing,
and a blue-noise quality check."""

import os
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))

# Radii as a fraction of the mesh's bounding-box diagonal (coarse -> dense).
RADIUS_FRACTIONS = (0.02, 0.01, 0.005)


def load_mesh():
    """Return ``(vertices, faces, diag)`` from ``mesh.npz`` (see export_mesh.py)."""
    path = os.path.join(HERE, "mesh.npz")
    if not os.path.exists(path):
        raise SystemExit(
            "mesh.npz not found. Run first (in the Warp env):\n"
            "  uv run --with usd-core tools/benchmarks/poisson_vs_physicsnemo/export_mesh.py"
        )
    d = np.load(path)
    return d["vertices"], d["faces"], float(d["diag"])


def min_dist_over_radius(points: np.ndarray, radius: float) -> float:
    """Smallest nearest-neighbor distance in ``points`` divided by ``radius``.

    A correct Poisson-disk set has ``min_dist / radius >= 1`` (no two samples
    closer than the radius); the closer to ``1``, the tighter the packing.
    """
    tree = cKDTree(points)
    dd, _ = tree.query(points, k=2)
    return float(dd[:, 1].min() / radius)


def best_of(fn, warmup=3, runs=7):
    """Return the best wall-clock time (seconds) of ``fn`` over ``runs`` timed
    calls after ``warmup`` untimed ones. ``fn`` must fully synchronize the device."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)
