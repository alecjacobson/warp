# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query neighbors and differentiate gathered distances, without a point-cloud owner."""

import argparse

import warp as wp


@wp.kernel
def neighbors(
    bvh: wp.uint64,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    indices: wp.array2d[int],
    loss: wp.array[float],
):
    t = wp.tid()
    search = wp.point_query_knn(bvh, points, queries[t], wp.inf, 4)
    i = int(-1)
    count = int(0)
    while wp.point_query_next(search, i):
        indices[t, count] = i
        wp.atomic_add(loss, 0, wp.length_sq(queries[t] - points[i]))
        count += 1
    for r in range(count, 4):
        indices[t, r] = -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    wp.init()
    with wp.ScopedDevice(args.device):
        points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, requires_grad=True)
        queries = wp.array([[0.9, 0.1, 0]], dtype=wp.vec3, requires_grad=True)
        bvh = wp.Bvh(points, points, leaf_size=4)
        indices = wp.empty((1, 4), dtype=int)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(neighbors, 1, inputs=[bvh.id, points, queries, indices, loss])
        tape.backward(loss)
        print("Neighbor indices (padding=-1):", indices.numpy())
        print("Query gradient:", queries.grad.numpy())
        print("Point gradients:", points.grad.numpy())


if __name__ == "__main__":
    main()
