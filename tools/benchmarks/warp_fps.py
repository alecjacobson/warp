# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Warp farthest-point sampling (the Kaolin algorithm, NumPy interface).

Two variants are provided for benchmarking against
:class:`warp.geometry.PoissonDiskSampler`:

* ``farthest_point_sampling_warp_sequential`` -- the straightforward
  ``O(N * k)`` greedy loop (find the farthest point, update every distance,
  repeat ``k`` times).
* ``farthest_point_sampling_warp_batchsort`` -- the block-aware version that
  radix-sorts by distance and accepts a whole head-chunk of farthest points per
  round using Warp's Tile API. This is Kaolin's ``farthest_point_sampling``
  algorithm (https://github.com/NVIDIAGameWorks/kaolin), here taking and
  returning NumPy arrays so no PyTorch is required.
"""

import time

import numpy as np

import warp as wp

INVALID_DIST = -1.0  # distance for inf/nan points, so they are sampled only after all others
TAKEN_DIST = -2.0  # distance for a point already sampled (not 0., to keep logic right for coincident points)
PADDED_DIST = -3.0  # distance for padding points added to reach the tile size
# ^^^ the relative values of these constants are important for algorithm logic

M_TOP_PROCESS = 512


def farthest_point_sampling_warp_sequential(points, k, B=1, return_time=False):
    N = points.shape[0]
    points = wp.array(points, dtype=wp.vec3)

    wp.synchronize()
    start_time = time.perf_counter()

    for _ in range(B):
        center_point = wp.zeros(1, dtype=wp.vec3)
        center_count = wp.zeros(1, dtype=wp.float32)
        distancesSq = wp.zeros(shape=N, dtype=wp.float32)
        farthest_point_inds = wp.full(shape=k, value=-1, dtype=wp.int32)
        i_round = wp.zeros(shape=1, dtype=wp.int32)

        wp.launch(compute_center, dim=N, inputs=[points], outputs=[center_point, center_count])
        wp.launch(divide_center, dim=1, inputs=[center_point, center_count])
        wp.launch(initialize_distances, dim=N, inputs=[points, distancesSq, center_point])

        for _i in range(k):
            wp.launch(find_farthest_point_ind, dim=N, inputs=[distancesSq, farthest_point_inds, i_round])
            wp.launch(update_distances, dim=N, inputs=[points, distancesSq, farthest_point_inds, i_round])
            wp.launch(increment_round, dim=1, inputs=[i_round])

    wp.synchronize()
    elapsed_time = time.perf_counter() - start_time
    if return_time:
        return farthest_point_inds.numpy(), elapsed_time
    return farthest_point_inds.numpy()


def farthest_point_sampling_warp_batchsort(points, k, B=1, return_time=False):
    assert k >= 0, f"k must be non-negative, got {k}"
    assert k <= points.shape[0], f"k must be <= N={points.shape[0]}, got {k}"

    N = points.shape[0]
    N_PADDED = max(N, M_TOP_PROCESS)
    PAD_LEN = N_PADDED - N
    if PAD_LEN > 0:
        points = np.concatenate((points, np.zeros((PAD_LEN, 3))), axis=0)
    points = wp.array(points, dtype=wp.vec3)

    wp.synchronize()
    start_time = time.perf_counter()

    for _ in range(B):
        center_point = wp.zeros(1, dtype=wp.vec3)
        center_count = wp.zeros(1, dtype=wp.float32)
        # NOTE: radix sort requires 2N space in array, so these are allocated bigger
        point_inds = wp.array(np.concatenate((np.arange(N_PADDED), np.full(N_PADDED, -1))), dtype=wp.int32)
        distancesSq = wp.full(shape=2 * N_PADDED, value=PADDED_DIST, dtype=wp.float32)
        farthest_point_inds = wp.full(shape=k, value=-1, dtype=wp.int32)
        i_round = wp.zeros(shape=1, dtype=wp.int32)
        i_prev_round = wp.zeros(shape=1, dtype=wp.int32)

        if k == 0:
            return farthest_point_inds.numpy()

        wp.launch(compute_center, dim=N, inputs=[points], outputs=[center_point, center_count])
        wp.launch(divide_center, dim=1, inputs=[center_point, center_count])
        wp.launch(initialize_distances, dim=N, inputs=[points, distancesSq, center_point])
        wp.launch(find_farthest_point_ind, dim=N_PADDED, inputs=[distancesSq, farthest_point_inds, i_round])
        wp.launch(initialize_distances_indexed, dim=N, inputs=[points, distancesSq, farthest_point_inds, N])
        wp.launch(increment_round, dim=1, inputs=[i_round])
        wp.launch(increment_round, dim=1, inputs=[i_prev_round])

        # Track a range of how many points we might have found, and read the exact
        # value back only when we might be done -- removing the per-round sync.
        found_estimate_min = i_round.numpy()[0]
        found_estimate_max = found_estimate_min
        while found_estimate_min < k:
            wp.utils.radix_sort_pairs(distancesSq, point_inds, count=N_PADDED)
            wp.launch(
                take_top_m_farthest,
                dim=M_TOP_PROCESS,
                inputs=[points, N_PADDED, distancesSq, point_inds, farthest_point_inds, i_round, i_prev_round],
                block_dim=M_TOP_PROCESS,
            )
            wp.launch(
                update_distances_from_round,
                dim=N_PADDED,
                inputs=[points, distancesSq, point_inds, farthest_point_inds, i_round, i_prev_round, N],
            )
            found_estimate_min += 1
            found_estimate_max += M_TOP_PROCESS
            if found_estimate_max >= k:
                curr_found = i_round.numpy()[0]
                found_estimate_min = curr_found
                found_estimate_max = curr_found

    wp.synchronize()
    elapsed_time = time.perf_counter() - start_time
    if return_time:
        return farthest_point_inds.numpy(), elapsed_time
    return farthest_point_inds.numpy()


@wp.kernel
def compute_center(
    points: wp.array(dtype=wp.vec3),
    center_point_sum: wp.array(dtype=wp.vec3),
    center_point_count: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p = points[i]
    if wp.isfinite(p.x) and wp.isfinite(p.y) and wp.isfinite(p.z):
        center_point_sum[0] += p
        center_point_count[0] += 1.0


@wp.kernel
def divide_center(center_point_sum: wp.array(dtype=wp.vec3), center_point_count: wp.array(dtype=wp.float32)):
    center = center_point_sum[0] / center_point_count[0]
    if not (wp.isfinite(center.x) and wp.isfinite(center.y) and wp.isfinite(center.z)):
        center = wp.vec3(0.0, 0.0, 0.0)
    center_point_sum[0] = center


@wp.kernel
def initialize_distances(
    points: wp.array(dtype=wp.vec3),
    distancesSq: wp.array(dtype=wp.float32),
    first_point: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dist = wp.length_sq(points[i] - first_point[0])
    if not wp.isfinite(dist):
        dist = INVALID_DIST
    distancesSq[i] = dist


@wp.kernel
def initialize_distances_indexed(
    points: wp.array(dtype=wp.vec3),
    distancesSq: wp.array(dtype=wp.float32),
    first_point_ind: wp.array(dtype=wp.int32),
    N: wp.int32,
):
    i = wp.tid()
    dist = wp.length_sq(points[i] - points[first_point_ind[0]])
    if not wp.isfinite(dist):
        dist = INVALID_DIST
    if i == first_point_ind[0]:
        dist = TAKEN_DIST
    if i >= N:  # padding point
        dist = PADDED_DIST
    distancesSq[i] = dist


@wp.kernel
def find_farthest_point_ind(
    distancesSq: wp.array(dtype=wp.float32),
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    i_round = i_round_arr[0]
    my_dist = distancesSq[i]
    curr_farthest_ind = farthest_point_inds[i_round]
    while curr_farthest_ind < 0 or my_dist > distancesSq[curr_farthest_ind]:
        # Swap in this distance if it is greater; if another thread changed it
        # first, re-read and check again.
        wp.atomic_cas(farthest_point_inds, i_round, curr_farthest_ind, i)
        curr_farthest_ind = farthest_point_inds[i_round]


@wp.kernel
def update_distances(
    points: wp.array(dtype=wp.vec3),
    distancesSq: wp.array(dtype=wp.float32),
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    i_round = i_round_arr[0]
    new_p = points[farthest_point_inds[i_round]]
    dist = wp.length_sq(points[i] - new_p)
    if not wp.isfinite(dist):
        dist = INVALID_DIST
    distancesSq[i] = wp.min(distancesSq[i], dist)


@wp.kernel
def increment_round(i_round_arr: wp.array(dtype=wp.int32)):
    i_round_arr[0] += 1


@wp.kernel
def take_top_m_farthest(
    points: wp.array(dtype=wp.vec3),
    N: wp.int32,
    distancesSq: wp.array(dtype=wp.float32),
    point_inds: wp.array(dtype=wp.int32),
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
    i_prev_round_arr: wp.array(dtype=wp.int32),
):
    # Load the sorted head-chunk into one block and accept as many farthest points
    # as possible before the array must be re-sorted. Valid because accepting
    # points only ever shrinks distances: while the head max exceeds the chunk's
    # smallest distance, it is the true global farthest point.
    block_i = wp.tid()
    i_prev_round_arr[0] = i_round_arr[0]
    i_round = i_round_arr[0]
    top_block_offset = N - M_TOP_PROCESS  # padded so this is always >= 0

    inds_tile = wp.tile_load(point_inds, M_TOP_PROCESS, offset=top_block_offset, storage="shared")
    distsSq_tile = wp.tile_load(distancesSq, M_TOP_PROCESS, offset=top_block_offset, storage="shared")

    head_chunk_threshold = distsSq_tile[0]

    while i_round < farthest_point_inds.shape[0]:
        top_block_max_i = wp.tile_argmax(distsSq_tile)[0]
        top_point_ind = inds_tile[top_block_max_i]
        top_point_dist = distsSq_tile[top_block_max_i]
        top_point_p = points[top_point_ind]

        if top_point_dist < head_chunk_threshold or top_point_dist == TAKEN_DIST:
            break

        old_dist = distsSq_tile[block_i]
        dist = old_dist
        if old_dist != PADDED_DIST:
            p = points[inds_tile[block_i]]
            new_dist = wp.length_sq(p - top_point_p)
            if wp.isfinite(new_dist):
                dist = wp.min(dist, new_dist)
            if block_i == top_block_max_i:
                dist = TAKEN_DIST
        distsSq_tile[block_i] = dist

        if block_i == 0:
            farthest_point_inds[i_round] = top_point_ind
        i_round += 1

    if block_i == 0:
        i_round_arr[0] = i_round


@wp.kernel
def update_distances_from_round(
    points: wp.array(dtype=wp.vec3),
    distancesSq: wp.array(dtype=wp.float32),
    point_inds: wp.array(dtype=wp.int32),
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
    i_prev_round_arr: wp.array(dtype=wp.int32),
    N: wp.int32,
):
    i = wp.tid()
    i_round = i_round_arr[0]
    i_prev_round = i_prev_round_arr[0]
    point_ind = point_inds[i]
    if point_ind >= N:  # padding point
        return
    p = points[point_ind]
    for i_new_round in range(i_prev_round, i_round):
        new_point_ind = farthest_point_inds[i_new_round]
        dist = wp.length_sq(p - points[new_point_ind])
        if not wp.isfinite(dist):
            dist = INVALID_DIST
        if point_ind == new_point_ind:
            dist = TAKEN_DIST
        distancesSq[i] = wp.min(distancesSq[i], dist)
