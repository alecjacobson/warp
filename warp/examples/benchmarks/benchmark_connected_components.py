# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmark: connected components of a triangle mesh
#
# Non-gating benchmark (no assertions) for
# warp.geometry.connected_components(). Times a few representative meshes and,
# for comparison, a serial SciPy union-find reference when SciPy is available.
#
# The pointer-jumping union-find converges in O(log(diameter)) rounds, so the
# long strip (a single component with a very large graph diameter) is included
# to stress round count, alongside compact grids at ~1e4/1e5/1e6 triangles and a
# mesh of many small components.
#
# Run: uv run warp/examples/benchmarks/benchmark_connected_components.py [--device ...]
###########################################################################

import statistics
import time

import numpy as np

import warp as wp
import warp.geometry

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scipy_cc

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _grid_mesh(n):
    def vid(i, j):
        return i * (n + 1) + j

    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    bl, br = vid(i, j), vid(i + 1, j)
    tr, tl = vid(i + 1, j + 1), vid(i, j + 1)
    tris = np.stack([bl, br, tr, bl, tr, tl], axis=-1).reshape(-1)
    return tris.astype(np.int32), (n + 1) * (n + 1)


def _strip(num_triangles):
    i = np.arange(num_triangles)
    tris = np.stack([i, i + 1, i + 2], axis=-1).reshape(-1)
    return tris.astype(np.int32), num_triangles + 2


def _many_components(num_components):
    # num_components disjoint triangles.
    base = np.arange(num_components) * 3
    tris = np.stack([base, base + 1, base + 2], axis=-1).reshape(-1)
    return tris.astype(np.int32), num_components * 3


def _time(fn, device, warmup=2, iters=7):
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3  # milliseconds


def _scipy_time(tris, num_points, warmup=1, iters=3):
    row = tris.reshape(-1, 3)
    edges = np.concatenate([row[:, [0, 1]], row[:, [1, 2]], row[:, [2, 0]]], axis=0)
    data = np.ones(len(edges), dtype=np.int8)
    graph = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(num_points, num_points)).tocsr()

    def run():
        return scipy_cc(graph, directed=False)

    for _ in range(warmup):
        run()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _run(name, tris, num_points, device):
    num_triangles = len(tris) // 3
    indices = wp.array(tris, dtype=wp.int32, device=device)

    _, k = warp.geometry.connected_components(indices, num_points=num_points, device=device)
    t_warp = _time(lambda: warp.geometry.connected_components(indices, num_points=num_points, device=device), device)

    scipy_str = "n/a"
    if SCIPY_AVAILABLE:
        scipy_str = f"{_scipy_time(tris, num_points):9.3f} ms (CPU, serial)"

    print(
        f"{name:22s} F={num_triangles:>9d} V={num_points:>9d} comps={k:>8d} "
        f"| warp {t_warp:9.3f} ms | scipy {scipy_str} | allocation included"
    )


def main(device):
    device = wp.get_device(device)
    print(f"\nDevice: {device} (medians of repeated runs, allocation included)\n")

    meshes = []
    for n in (70, 224, 707):  # ~1e4, ~1e5, ~1e6 triangles
        tris, v = _grid_mesh(n)
        meshes.append((f"grid {n}x{n}", tris, v))
    for length in (100_000, 1_000_000):
        tris, v = _strip(length)
        meshes.append((f"strip ({length} tris)", tris, v))
    tris, v = _many_components(200_000)
    meshes.append(("200k components", tris, v))

    for name, tris, v in meshes:
        _run(name, tris, v, device)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    args = parser.parse_known_args()[0]
    main(args.device)
