# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory exhaustive point queries, implemented entirely in Warp."""

import functools

import warp as wp


class BruteForcePointIndex:
    """Retain targets without constructing a spatial index."""

    def __init__(self, points):
        self.points = points


@functools.cache
def query_kernel(k, index_type, squared, has_exclusions, tile_size):
    distances_type = wp.types.vector(length=k, dtype=wp.float32)
    indices_type = wp.types.vector(length=k, dtype=wp.int32)

    # Match native Warp's noncontracted squared-distance arithmetic for ranking
    # and strict cutoff comparisons, including near ties.
    @wp.kernel(module="unique", module_options={"fuse_fp": False}, enable_backward=False)
    def brute_force_query(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        num_points: int,
        num_queries: int,
        max_dist: float,
        exclusions: wp.array[int],
        indices: wp.array[index_type],
        distances: wp.array[float],
        counts: wp.array[int],
    ):
        batch, block, lane = wp.tid()
        row = block * wp.block_dim() + lane
        query_id = batch * num_queries + row
        position = wp.vec3(0.0)
        excluded = int(-1)
        valid = bool(False)
        if row < num_queries:
            position = queries[query_id]
            valid = (
                max_dist > 0.0 and wp.isfinite(position[0]) and wp.isfinite(position[1]) and wp.isfinite(position[2])
            )
            if wp.static(has_exclusions):
                excluded = exclusions[query_id]
        cutoff = max_dist * max_dist
        if max_dist > 0.0 and cutoff == 0.0:
            cutoff = 1.401298464324817e-45
        best = distances_type(wp.inf)
        selected = indices_type(-1)
        count = int(0)
        # CPU uses a scalar target stream. CUDA cooperatively loads bounded
        # shared tiles; all lanes participate, including padded query rows.
        for start in range(0, num_points, wp.static(max(tile_size, 1))):
            if wp.static(tile_size > 0):
                targets = wp.tile_load(points, shape=tile_size, offset=batch * num_points + start, storage="shared")
            for j in range(wp.static(max(tile_size, 1))):
                target_id = start + j
                if wp.static(tile_size > 0):
                    target = targets[j]
                else:
                    target = points[batch * num_points + target_id]
                delta = position - target
                d = (delta[0] * delta[0] + delta[1] * delta[1]) + delta[2] * delta[2]
                # Targets are visited in original-index order. Strict insertion
                # preserves the smallest index on equal computed distances.
                if valid and target_id < num_points and target_id != excluded and d < cutoff and d < best[k - 1]:
                    if wp.static(k == 1):
                        best[0] = d
                        selected[0] = target_id
                    else:
                        slot = int(k - 1)
                        while slot > 0 and d < best[slot - 1]:
                            best[slot] = best[slot - 1]
                            selected[slot] = selected[slot - 1]
                            slot -= 1
                        best[slot] = d
                        selected[slot] = target_id
                    count = wp.min(count + 1, k)
        if row < num_queries:
            counts[query_id] = count
            for j in range(k):
                d = best[j]
                if wp.static(not squared):
                    d = wp.sqrt(d)
                indices[query_id * k + j] = index_type(selected[j])
                distances[query_id * k + j] = d

    return brute_force_query


def query(points, queries, k, max_dist, exclusions, indices, distances, counts, squared):
    """Launch one fused batch search without a pairwise distance allocation."""
    if not queries.size:
        return
    device = points.device
    block_dim = 256 if device.is_cuda else 1
    tile_size = 256 if device.is_cuda else 0
    nq = queries.shape[-1]
    wp.launch(
        query_kernel(k, indices.dtype, squared, exclusions is not None, tile_size),
        (points.shape[0] if points.ndim == 2 else 1, (nq + block_dim - 1) // block_dim, block_dim),
        inputs=[
            points.flatten(),
            queries.flatten(),
            points.shape[-1],
            nq,
            max_dist,
            exclusions.flatten() if exclusions is not None else None,
            indices.flatten(),
            distances.flatten(),
            counts.flatten(),
        ],
        block_dim=block_dim,
        device=device,
        record_tape=False,
    )
