# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic batch kNN and a fused scalar kernel using the same launch policy."""

import argparse

import numpy as np

import warp as wp


def fused_kernel(k, stride):
    @wp.kernel(module="unique")
    def neighbor_sum(bvh: wp.uint64, points: wp.array[wp.vec3], queries: wp.array[wp.vec3], sums: wp.array[float]):
        t = wp.tid()
        if t % wp.static(stride) != 0:
            return
        t = t // wp.static(stride)
        search = wp.point_query_knn(bvh, points, queries[t], wp.inf, wp.static(k))
        index = int(-1)
        total = float(0.0)
        while wp.point_query_next(search, index):
            # Replace this with the caller's own per-neighbor computation.
            total += wp.length_sq(queries[t] - points[index])
        sums[t] = total

    return neighbor_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    wp.init()
    with wp.ScopedDevice(args.device):
        rng = np.random.default_rng(42)
        points = wp.array(rng.normal(size=(100_000, 3)).astype(np.float32), dtype=wp.vec3)
        queries = wp.array(rng.normal(size=(1_024, 3)).astype(np.float32), dtype=wp.vec3)
        k = 8
        bvh, indices, distances_sq, _counts = wp.point_query_knn_batch(points, queries, k)
        print("First query neighbors:", indices.numpy()[0])
        config = wp.point_query_config(len(points), len(queries), points.device)
        sums = wp.empty(len(queries), dtype=float)
        wp.launch(
            fused_kernel(k, config.query_stride),
            len(queries) * config.query_stride,
            inputs=[bvh.id, points, queries, sums],
            block_dim=config.block_dim,
        )
        np.testing.assert_allclose(sums.numpy(), distances_sq.numpy().sum(axis=1), rtol=2e-6)
        print("Fused neighbor sums match batch output; configuration:", config)


if __name__ == "__main__":
    main()
