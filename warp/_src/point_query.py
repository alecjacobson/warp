# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-only batch point queries and shared scalar-kernel launch policy."""

import functools
from dataclasses import dataclass

import warp as wp
from warp._src.context import DeviceLike, get_device
from warp._src.utils import ScopedDevice, radix_sort_pairs


@dataclass(frozen=True)
class PointQueryConfig:
    """Explicit point-query tuning choices, also usable by fused scalar kernels."""

    leaf_size: int
    """Suggested BVH leaf size when building a point tree."""
    query_stride: int
    """Launch this many threads per query; only each stride's first thread works."""
    block_dim: int
    """Suggested CUDA block size."""
    ordered: bool
    """Whether the batch wrapper should spatially order queries before searching."""


def point_query_config(num_points: int, num_queries: int, device: DeviceLike = None) -> PointQueryConfig:
    """Choose a point-query configuration from array sizes and the device.

    GPU launches target 16 scheduled warps per SM, with at most 32 threads per
    query. Large batches use dense launches and spatial query ordering. CPU
    queries use dense launches. This heuristic can be overridden by the caller;
    it does not change the defaults of :class:`warp.Bvh` or :func:`warp.launch`.
    """
    if num_points < 0 or num_queries < 0:
        raise ValueError("Point and query counts must be nonnegative")
    device = get_device(device)
    leaf = 32 if num_queries <= 32 or num_points >= 1_000_000 else 4
    stride = 1
    if device.is_cuda:
        while stride < 32 and num_queries * stride < 512 * device.sm_count:
            stride *= 2
    return PointQueryConfig(
        leaf, stride, 32 if stride > 1 else 256, device.is_cuda and stride == 1 and num_points >= 1_000_000
    )


@wp.kernel(enable_backward=False)
def _init_bounds(bounds: wp.array2d[float]):
    a = wp.tid()
    bounds[0, a] = wp.inf
    bounds[1, a] = -wp.inf


@wp.kernel(enable_backward=False)
def _query_bounds(queries: wp.array[wp.vec3], bounds: wp.array2d[float], workers: int):
    lo = wp.vec3(wp.inf)
    hi = wp.vec3(-wp.inf)
    for i in range(wp.tid(), queries.shape[0], workers):
        p = queries[i]
        if wp.isfinite(p):
            lo = wp.min(lo, p)
            hi = wp.max(hi, p)
    for a in range(3):
        wp.atomic_min(bounds, 0, a, lo[a])
        wp.atomic_max(bounds, 1, a, hi[a])


@wp.func
def _spread_bits(n: wp.uint32):
    n = (n | (n << wp.uint32(16))) & wp.uint32(0x030000FF)
    n = (n | (n << wp.uint32(8))) & wp.uint32(0x0300F00F)
    n = (n | (n << wp.uint32(4))) & wp.uint32(0x030C30C3)
    n = (n | (n << wp.uint32(2))) & wp.uint32(0x09249249)
    return n


@wp.kernel(enable_backward=False)
def _query_codes(queries: wp.array[wp.vec3], bounds: wp.array2d[float], keys: wp.array[int], order: wp.array[int]):
    i = wp.tid()
    p = queries[i]
    code = wp.uint32(0)
    if wp.isfinite(p):
        for a in range(3):
            edge = bounds[1, a] - bounds[0, a]
            if not (edge > 0.0):
                edge = 1.0
            unit = wp.clamp((p[a] - bounds[0, a]) / edge, 0.0, 1.0)
            v = wp.uint32(wp.min(int(unit * 1024.0), 1023))
            code = code | (_spread_bits(v) << wp.uint32(a))
    keys[i] = int(code)
    order[i] = i


@functools.cache
def _query_kernel(k, stride, ordered, squared, index_type, has_excludes):
    @wp.kernel(module="unique", enable_backward=False)
    def query(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        max_dist: float,
        excludes: wp.array[int],
        order: wp.array[int],
        indices: wp.array2d[index_type],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ):
        t = wp.tid()
        if t % wp.static(stride) != 0:
            return
        t = t // wp.static(stride)
        if wp.static(ordered):
            t = order[t]
        exclude = int(-1)
        if wp.static(has_excludes):
            exclude = excludes[t]
        search = wp.point_query_knn(bvh, points, queries[t], max_dist, wp.static(k), exclude)
        index = int(-1)
        count = int(0)
        while wp.point_query_next(search, index):
            indices[t, count] = index_type(index)
            d = wp.length_sq(queries[t] - points[index])
            if wp.static(not squared):
                d = wp.sqrt(d)
            distances[t, count] = d
            count += 1
        counts[t] = count
        for j in range(count, wp.static(k)):
            indices[t, j] = index_type(-1)
            distances[t, j] = wp.inf

    return query


@functools.cache
def _distance_kernel(index_type, squared):
    @wp.kernel(module="unique")
    def distances_from_indices(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        indices: wp.array2d[index_type],
        distances: wp.array2d[float],
    ):
        q, j = wp.tid()
        index = int(indices[q, j])
        if index >= 0:
            d = wp.length_sq(queries[q] - points[index])
            if wp.static(not squared):
                d = wp.sqrt(d)
            distances[q, j] = d
        else:
            distances[q, j] = wp.inf

    return distances_from_indices


def _validate_array(array, name, dtype, shape, device):
    if array.dtype not in dtype or array.shape != shape or not array.is_contiguous or array.device != device:
        raise ValueError(f"{name} must be contiguous, have shape {shape}, a supported dtype, and live on {device}")


def point_query_knn_batch(
    points: wp.array,
    queries: wp.array,
    k: int,
    *,
    bvh: wp.Bvh | None = None,
    max_dist: float = float("inf"),
    exclude_indices: wp.array | None = None,
    indices: wp.array | None = None,
    distances: wp.array | None = None,
    counts: wp.array | None = None,
    squared: bool = True,
    leaf_size: int | None = None,
    query_stride: int | None = None,
    ordered: bool | None = None,
) -> tuple[wp.Bvh, wp.array, wp.array, wp.array]:
    """Find exact 3D nearest neighbors using only Warp arrays and kernels.

    The default configuration selects the BVH leaf size, query spacing and
    ordering using :func:`warp.point_query_config`. No Torch dependency, host data
    readback, or device synchronization is introduced. Output rows retain the
    original query order. Selection uses :func:`warp._src.lang.point_query_knn`, including its
    float32 arithmetic, strict cutoff, original-index tie breaks and partial
    results. Gradients flow through gathered distances with indices held fixed.

    Args:
        points: Contiguous one-dimensional ``vec3`` reference array.
        queries: Contiguous one-dimensional ``vec3`` query array on the same device.
        k: Positive Python integer, specialized when compiling the query kernel.
        bvh: Optional reusable BVH enclosing the same point indices. Refit after
            moving points. When omitted, build a point BVH on the input device.
        max_dist: Strict distance cutoff in coordinate units, even for squared outputs.
        exclude_indices: Optional contiguous int32 array of shape ``(Q,)``;
            each row excludes that original point index. -1 excludes nothing.
        indices: Optional output of shape ``(Q, k)`` and dtype int32 or int64.
            Newly allocated outputs use int32.
        distances: Optional float32 output of shape ``(Q, k)``.
        counts: Optional int32 output of shape ``(Q,)``.
        squared: Return squared distances; set False for Euclidean distances.
        leaf_size: Override leaf size when constructing a new BVH. Not accepted
            with an existing BVH, whose layout is retained.
        query_stride: Override threads per query with a power of two from 1 to 32.
        ordered: Override spatial query ordering. Ordering is performed entirely
            on device using query bounds, Morton keys and Warp's radix sort.

    Returns:
        A tuple ``(bvh, indices, distances, counts)``. Keep the BVH, input arrays
        and outputs alive through asynchronous execution and graph replay.
        Warm up compilation before CUDA capture. Capture of construction requires
        CUDA memory pools. Use the device's current stream (e.g. ScopedStream).
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive Python integer")
    if not isinstance(squared, bool) or (ordered is not None and not isinstance(ordered, bool)):
        raise ValueError("squared and ordered must be booleans (or None for automatic ordering)")
    device = points.device
    if points.ndim != 1 or queries.ndim != 1:
        raise ValueError("points and queries must be one-dimensional vec3 arrays")
    _validate_array(points, "points", (wp.vec3,), (len(points),), device)
    _validate_array(queries, "queries", (wp.vec3,), (len(queries),), device)
    nq = len(queries)
    config = point_query_config(len(points), nq, device)
    stride = config.query_stride if query_stride is None else query_stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1 or stride > 32 or stride & (stride - 1):
        raise ValueError("query_stride must be a power of two from 1 to 32")
    if bvh is not None and (bvh.device != device or len(bvh.lowers) != len(points) or leaf_size is not None):
        raise ValueError("An existing BVH must match the point count and device; leaf_size cannot override its layout")
    leaf = config.leaf_size if leaf_size is None else leaf_size
    use_order = config.ordered if ordered is None else ordered
    block_dim = 32 if stride > 1 else 256
    if exclude_indices is not None:
        _validate_array(exclude_indices, "exclude_indices", (wp.int32,), (nq,), device)
    for output, name, dtype, shape in (
        (indices, "indices", (wp.int32, wp.int64), (nq, k)),
        (distances, "distances", (wp.float32,), (nq, k)),
        (counts, "counts", (wp.int32,), (nq,)),
    ):
        if output is not None:
            _validate_array(output, name, dtype, shape, device)
    with ScopedDevice(device):
        if bvh is None:
            bvh = wp.Bvh(points, points, leaf_size=leaf)
        if indices is None:
            indices = wp.empty((nq, k), dtype=int, device=device)
        if distances is None:
            distances = wp.empty(
                (nq, k), dtype=float, device=device, requires_grad=points.requires_grad or queries.requires_grad
            )
        if counts is None:
            counts = wp.empty(nq, dtype=int, device=device)
        keys = order = bounds = None
        if nq:
            if use_order:
                keys = wp.empty(2 * nq, dtype=int, device=device)
                order = wp.empty_like(keys)
                bounds = wp.empty((2, 3), dtype=float, device=device)
                workers = min(nq, 256)
                wp.launch(_init_bounds, 3, inputs=[bounds], device=device, record_tape=False)
                wp.launch(_query_bounds, workers, inputs=[queries, bounds, workers], device=device, record_tape=False)
                wp.launch(_query_codes, nq, inputs=[queries, bounds, keys, order], device=device, record_tape=False)
                radix_sort_pairs(keys, order, nq)
            wp.launch(
                _query_kernel(k, stride, use_order, squared, indices.dtype, exclude_indices is not None),
                nq * stride,
                inputs=[bvh.id, points, queries, max_dist, exclude_indices, order, indices, distances, counts],
                block_dim=block_dim,
                device=device,
                record_tape=False,
            )
            if points.requires_grad or queries.requires_grad:
                # A stored-index gather gives backward stable neighbor choices
                # without differentiating selection or replaying its cursor.
                wp.launch(
                    _distance_kernel(indices.dtype, squared),
                    (nq, k),
                    inputs=[points, queries, indices, distances],
                    device=device,
                )
        # Ordinary Warp frees retire on the CUDA default stream. Graph-owned
        # scratch receives dependency-tracked graph free nodes instead.
        if device.is_cuda and not device.stream.is_blocking and not device.stream.is_capturing:
            device.null_stream.wait_stream(device.stream)
    return bvh, indices, distances, counts
