# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmark: triangle-mesh topology statistics
#
# Non-gating benchmark (no assertions). Compares two ways of building the
# vertex -> incident-corner CSR that warp.geometry.triangle_mesh_topology_
# statistics() relies on:
#
#   A. No-sort path:  atomic count -> scan -> atomic scatter (the production
#      path, exercised here directly via the module's private kernels).
#   B. Sort path:     warp.fem.utils.compress_node_indices(), which prepares
#      index/value pairs, radix-sorts them, run-length encodes, and scans.
#
# It also times the full statistics routine. Representative regular grids of
# ~1e4/1e5/1e6 triangles are used, plus a pathological single high-valence fan
# to characterize the O(d_v^2) per-vertex analysis cost.
#
# Run: uv run warp/examples/benchmarks/benchmark_mesh_topology_statistics.py [--device ...]
###########################################################################

import statistics
import time

import numpy as np

import warp as wp
import warp.geometry
from warp._src.fem.utils import compress_node_indices
from warp._src.geometry import _TopologyStatistics
from warp._src.utils import array_scan


def _grid_mesh(n):
    """Triangulated regular ``n x n`` grid: ``2*n*n`` triangles, ``(n+1)^2`` vertices."""

    def vid(i, j):
        return i * (n + 1) + j

    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    bl, br = vid(i, j), vid(i + 1, j)
    tr, tl = vid(i + 1, j + 1), vid(i, j + 1)
    tris = np.stack([bl, br, tr, bl, tr, tl], axis=-1).reshape(-1)
    return tris.astype(np.int32), (n + 1) * (n + 1)


def _fan_mesh(num_triangles):
    """A single high-valence fan: center vertex 0 with ``num_triangles`` incident faces."""
    n = num_triangles
    i = np.arange(n)
    tris = np.stack([np.zeros(n, dtype=np.int64), 1 + i, 1 + (i + 1) % n], axis=-1).reshape(-1)
    return tris.astype(np.int32), n + 1


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


def _build_csr_nosort(indices, num_vertices, num_triangles, device):
    vertex_offsets = wp.zeros(num_vertices + 1, dtype=wp.int32, device=device)
    raw_stats = wp.zeros(7, dtype=wp.int32, device=device)
    incident_corners = wp.empty(3 * num_triangles, dtype=wp.int32, device=device)
    vertex_cursors = wp.zeros(num_vertices, dtype=wp.int32, device=device)
    wp.launch(
        _TopologyStatistics._count_incident_corners,
        dim=num_triangles,
        inputs=[indices, vertex_offsets, raw_stats],
        device=device,
    )
    array_scan(vertex_offsets, vertex_offsets, inclusive=True)
    wp.launch(
        _TopologyStatistics._scatter_incident_corners,
        dim=num_triangles,
        inputs=[indices, vertex_offsets, vertex_cursors, incident_corners],
        device=device,
    )


def _build_csr_sort(indices, num_vertices, device):
    # compress_node_indices() returns (node_offsets, sorted_array_indices), which
    # is exactly the vertex-offset / incident-corner CSR, built via radix sort.
    compress_node_indices(num_vertices, indices)


def _run(name, tris, num_points, device):
    num_triangles = len(tris) // 3
    indices = wp.array(tris.reshape(-1, 3), dtype=wp.int32, device=device)
    max_valence = int(np.max(np.bincount(tris)))

    t_nosort = _time(lambda: _build_csr_nosort(indices, num_points, num_triangles, device), device)
    t_sort = _time(lambda: _build_csr_sort(indices, num_points, device), device)
    t_full = _time(
        lambda: warp.geometry.triangle_mesh_topology_statistics(indices, num_points=num_points, device=device), device
    )

    print(
        f"{name:20s} F={num_triangles:>9d} V={num_points:>9d} maxdeg={max_valence:>7d} "
        f"| no-sort {t_nosort:9.3f} ms | sort {t_sort:9.3f} ms | full {t_full:9.3f} ms "
        f"| allocation included"
    )


def main(device):
    device = wp.get_device(device)
    print(f"\nDevice: {device} (timings are medians of repeated runs, allocation included)\n")
    meshes = []
    for n in (70, 224, 707):  # ~1e4, ~1e5, ~1e6 triangles
        tris, v = _grid_mesh(n)
        meshes.append((f"grid {n}x{n}", tris, v))
    # The full-statistics analysis is O(d_v^2) at the single high-valence center,
    # so valences are kept modest here to keep the pathological run tractable.
    for fan in (2_000, 8_000):
        tris, v = _fan_mesh(fan)
        meshes.append((f"fan (valence {fan})", tris, v))

    for name, tris, v in meshes:
        _run(name, tris, v, device)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    args = parser.parse_known_args()[0]
    main(args.device)
