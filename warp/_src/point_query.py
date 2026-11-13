# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud queries with independent batches and optional native cuBQL traversal."""

import functools
from dataclasses import dataclass
from enum import IntEnum

import warp as wp
from warp._src import context
from warp._src.context import DeviceLike, get_device
from warp._src.types import type_size_in_bytes
from warp._src.utils import ScopedDevice, radix_sort_pairs


class PointQueryBackend(IntEnum):
    """Choose the implementation used for point-cloud queries."""

    WARP = 0
    """Use Warp traversal with deterministic original-index ties and CPU/CUDA support."""
    CUBQL = 1
    """Use native cuBQL traversal on CUDA with implementation-dependent ties."""


class PointQueryObjective(IntEnum):
    """Choose which cost to optimize when constructing a cuBQL point index."""

    BUILD_AND_QUERY = 0
    """Optimize construction followed by one query batch."""
    REUSE = 1
    """Optimize repeated queries against an existing target cloud."""


def _cubql_config(num_points, num_queries, k, objective):
    # Frozen settings from independent training on the 46-SM Blackwell slice.
    # Other shapes use the corresponding size regime; this is a heuristic.
    if objective == PointQueryObjective.REUSE:
        if num_queries <= 32 and num_points < 1_000_000:
            return 0, 8, 32, 128
        if num_points >= 1_000_000:
            return 0, 8, 8, 256
        return (0, 0, 32, 64) if k == 1 else (0, 8, 32, 32)
    if num_queries <= 32 and num_points < 1_000_000:
        return 1, 32, 32, 256
    if num_points >= 1_000_000:
        return 2, 32, 8, 256
    return (2, 64, 32, 32) if k == 1 else (1, 16, 32, 256)


class CubqlPointIndex:
    """Retain a native CUDA cuBQL tree for repeated point queries.

    Construction and rebuilding synchronize with the host and cannot be captured.
    Queries can be captured after construction. Keep this index and its point
    allocation alive through asynchronous work and graph replay. Order updates
    against readers on all streams. Rebuild after modifying target coordinates.

    Args:
        points: Contiguous one-dimensional ``wp.vec3`` CUDA target array.
        num_queries: Expected query count, used to choose the construction policy.
        k: Expected neighbor count from 1 through 64, used for tuning.
        objective: Optimize construction-inclusive or reused-index queries.
    """

    backend = PointQueryBackend.CUBQL

    def __init__(
        self,
        points: wp.array[wp.vec3],
        num_queries: int,
        k: int,
        objective: PointQueryObjective = PointQueryObjective.REUSE,
    ):
        self.id = None
        if not isinstance(points, wp.array) or points.dtype != wp.vec3 or points.ndim != 1:
            raise TypeError("points must be a one-dimensional Warp vec3 array")
        if not points.is_contiguous or not points.device.is_cuda:
            raise ValueError("Direct cuBQL requires contiguous CUDA points")
        if not isinstance(objective, PointQueryObjective):
            raise TypeError("objective must be a PointQueryObjective")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 64:
            raise ValueError("Direct cuBQL supports k from 1 through 64")
        if not isinstance(num_queries, int) or isinstance(num_queries, bool) or num_queries < 0:
            raise ValueError("num_queries must be a nonnegative integer")
        if not context.is_cubql_available():
            raise ValueError("Direct cuBQL is unavailable in this Warp build")
        if not points.device.is_mempool_supported:
            raise ValueError("Direct cuBQL requires CUDA memory-pool support")
        self.points = points
        self.device = points.device
        self.runtime = context.runtime
        self.objective = objective
        self._builder, self.leaf_size, _, _ = _cubql_config(len(points), num_queries, k, objective)
        self.rebuild()

    def rebuild(self):
        """Rebuild the index after changing target coordinates in place."""
        with ScopedDevice(self.device):
            if self.device.stream.is_capturing:
                raise ValueError("Direct cuBQL construction cannot be captured; build the index before capture")
            replacement = self.runtime.core.wp_point_query_cubql_create_device(
                self.device.context, self.points.ptr, len(self.points), self._builder, self.leaf_size
            )
            if not replacement:
                raise RuntimeError(f"Failed to construct cuBQL point index: {self.runtime.get_error_string()}")
            previous, self.id = self.id, replacement
            if previous:
                self.runtime.core.wp_point_query_cubql_destroy_device(previous)

    def __del__(self):
        if self.id:
            try:
                with self.device.context_guard:
                    self.runtime.core.wp_point_query_cubql_destroy_device(self.id)
            except (TypeError, AttributeError):
                pass  # Runtime callables can disappear during interpreter shutdown.


def _batch_view(array, batch):
    if array is None:
        return None
    if all(array.shape[1:]):
        return array[batch]
    # Warp's general slice validation rejects empty dimensions. Preserve empty
    # views and their gradient storage without invoking that indexing path.
    view = wp.array(
        ptr=array.ptr or 0,
        dtype=array.dtype,
        shape=array.shape[1:],
        device=array.device,
        grad=_batch_view(array.grad, batch),
    )
    view._ref = array
    return view


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
    if not isinstance(array, wp.array):
        raise TypeError(f"{name} must be a Warp array")
    if array.dtype not in dtype or array.shape != shape or not array.is_contiguous or array.device != device:
        raise ValueError(f"{name} must be contiguous, have shape {shape}, a supported dtype, and live on {device}")


def point_query_knn_batch(
    points: wp.array,
    queries: wp.array,
    k: int,
    *,
    bvh: wp.Bvh | CubqlPointIndex | tuple | None = None,
    max_dist: float = float("inf"),
    exclude_indices: wp.array | None = None,
    indices: wp.array | None = None,
    distances: wp.array | None = None,
    counts: wp.array | None = None,
    squared: bool = True,
    leaf_size: int | None = None,
    query_stride: int | None = None,
    ordered: bool | None = None,
    backend: PointQueryBackend | None = None,
    objective: PointQueryObjective = PointQueryObjective.REUSE,
) -> tuple[wp.Bvh | CubqlPointIndex | tuple, wp.array, wp.array, wp.array]:
    """Find exact 3D nearest neighbors in independent point clouds.

    The default configuration selects the BVH leaf size, query spacing and
    ordering using :func:`warp.point_query_config`. No Torch dependency, host data
    readback, or device synchronization is introduced by the Warp query path.
    Output rows retain the original query order. The Warp backend uses
    :func:`warp._src.lang.point_query_knn`, including its
    float32 arithmetic, strict cutoff, original-index tie breaks and partial
    results. Gradients flow through gathered distances with indices held fixed.

    Args:
        points: Contiguous ``vec3`` targets of shape ``(M,)`` or ``(B, M)``.
        queries: Contiguous ``vec3`` queries of shape ``(N,)`` or ``(B, N)`` on
            the same device. Each batch searches only its corresponding targets.
        k: Positive Python integer, specialized when compiling the query kernel.
        bvh: Optional reusable BVH or cuBQL point index, or a tuple of B indexes
            for batched inputs. Refit Warp BVHs or rebuild cuBQL indexes after
            moving targets. When omitted, construct the indexes.
        max_dist: Strict distance cutoff in coordinate units, even for squared outputs.
        exclude_indices: Optional int32 array matching the query shape; each row
            excludes a batch-local target index. -1 excludes nothing. Warp only.
        indices: Optional int32/int64 output with the query shape followed by k.
            Newly allocated outputs use int32. Returned indices are batch-local.
        distances: Optional float32 output with the same shape as indices.
        counts: Optional int32 output matching the query shape.
        squared: Return squared distances; set False for Euclidean distances.
        leaf_size: Override leaf size when constructing a new BVH. Not accepted
            with an existing BVH, whose layout is retained.
        query_stride: Override threads per query with a power of two from 1 to 32.
        ordered: Override spatial query ordering. Ordering is performed entirely
            on device using query bounds, Morton keys and Warp's radix sort.
        backend: Choose Warp or direct cuBQL. When omitted, infer from a supplied
            index, or use Warp. Direct cuBQL requires CUDA, k <= 64, an infinite
            cutoff, no exclusions, and no query ordering. It uses upstream
            distance arithmetic and does not promise original-index tie breaks.
        objective: Choose construction-inclusive or reused-index cuBQL tuning
            when creating an index. Existing indexes retain their objective.

    Returns:
        A tuple ``(bvh, indices, distances, counts)``. Batched inputs return a
        tuple of B indexes, outputs of shape ``(B, N, k)``, and counts ``(B, N)``.
        Keep the indexes, input arrays
        and outputs alive through asynchronous execution and graph replay.
        Warm up compilation before CUDA capture. Capture of construction requires
        CUDA memory pools and the Warp backend; construct cuBQL indexes before
        capture. Use the device's current stream (e.g. ScopedStream).
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive Python integer")
    if not isinstance(squared, bool) or (ordered is not None and not isinstance(ordered, bool)):
        raise ValueError("squared and ordered must be booleans (or None for automatic ordering)")
    if not isinstance(points, wp.array) or not isinstance(queries, wp.array):
        raise TypeError("points and queries must be Warp arrays")
    device = points.device
    if points.ndim not in (1, 2) or queries.ndim != points.ndim:
        raise ValueError("points and queries must have matching one- or two-dimensional vec3 layouts")
    if points.ndim == 2 and points.shape[0] != queries.shape[0]:
        raise ValueError("points and queries must have the same batch count")
    _validate_array(points, "points", (wp.vec3,), points.shape, device)
    _validate_array(queries, "queries", (wp.vec3,), queries.shape, device)
    nq = queries.shape[-1]
    num_points = points.shape[-1]
    if points.ndim == 2:
        if bvh is not None and (not isinstance(bvh, tuple) or len(bvh) != points.shape[0]):
            raise ValueError("Batched queries require a tuple containing one index per batch")
        existing = bvh or ()
    else:
        existing = () if bvh is None else (bvh,)
    if backend is not None and not isinstance(backend, PointQueryBackend):
        raise TypeError("backend must be a PointQueryBackend")
    if not isinstance(objective, PointQueryObjective):
        raise TypeError("objective must be a PointQueryObjective")
    if backend is None:
        backend = (
            PointQueryBackend.CUBQL if existing and isinstance(existing[0], CubqlPointIndex) else PointQueryBackend.WARP
        )
    for batch, index in enumerate(existing):
        expected_type = CubqlPointIndex if backend == PointQueryBackend.CUBQL else wp.Bvh
        if not isinstance(index, expected_type):
            raise TypeError("The supplied index must match the requested backend")
        references = index.points if isinstance(index, CubqlPointIndex) else index.lowers
        if index.device != device or len(references) != num_points or leaf_size is not None:
            raise ValueError(
                "An existing index must match the point count and device; leaf_size cannot override its layout"
            )
        if isinstance(index, CubqlPointIndex):
            offset = batch * points.strides[0] if points.ndim == 2 else 0
            if (references.ptr or 0) != ((points.ptr or 0) + offset):
                raise ValueError("A cuBQL index must reference the original target allocation")
    if backend == PointQueryBackend.CUBQL:
        if not device.is_cuda or not context.is_cubql_available():
            raise ValueError("Direct cuBQL requires CUDA and a Warp build with cuBQL enabled")
        if k > 64:
            raise ValueError("Direct cuBQL supports k from 1 through 64")
        if max_dist != float("inf") or exclude_indices is not None:
            raise ValueError(
                "Direct cuBQL requires an infinite cutoff and no exclusions; use the Warp backend for filtering"
            )
        if ordered or leaf_size is not None:
            raise ValueError(
                "Direct cuBQL uses its tuning policy; ordered=True and leaf_size overrides are unsupported"
            )
        if bvh is None and device.stream.is_capturing:
            raise ValueError("Direct cuBQL construction cannot be captured; build the index before capture")
    config = point_query_config(num_points, nq, device)
    stride = config.query_stride if query_stride is None else query_stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1 or stride > 32 or stride & (stride - 1):
        raise ValueError("query_stride must be a power of two from 1 to 32")
    leaf = config.leaf_size if leaf_size is None else leaf_size
    use_order = config.ordered if ordered is None else ordered
    block_dim = 32 if stride > 1 else 256
    if exclude_indices is not None:
        _validate_array(exclude_indices, "exclude_indices", (wp.int32,), queries.shape, device)
    output_shape = (*queries.shape, k)
    for output, name, dtype, shape in (
        (indices, "indices", (wp.int32, wp.int64), output_shape),
        (distances, "distances", (wp.float32,), output_shape),
        (counts, "counts", (wp.int32,), queries.shape),
    ):
        if output is not None:
            _validate_array(output, name, dtype, shape, device)
    buffers = [points, queries, exclude_indices]
    for output in (indices, distances, counts):
        if output is not None and output.size:
            end = output.ptr + output.size * type_size_in_bytes(output.dtype)
            for other in buffers:
                if other is not None and other.size:
                    other_end = other.ptr + other.size * type_size_in_bytes(other.dtype)
                    if output.ptr < other_end and other.ptr < end:
                        raise ValueError("Point-query outputs must not overlap inputs or one another")
            buffers.append(output)
    with ScopedDevice(device):
        if indices is None:
            indices = wp.empty(output_shape, dtype=int, device=device)
        if distances is None:
            distances = wp.empty(
                output_shape, dtype=float, device=device, requires_grad=points.requires_grad or queries.requires_grad
            )
        if counts is None:
            counts = wp.empty(queries.shape, dtype=int, device=device)
        if points.ndim == 2:
            indexes = []
            for batch in range(points.shape[0]):
                result = point_query_knn_batch(
                    _batch_view(points, batch),
                    _batch_view(queries, batch),
                    k,
                    bvh=bvh[batch] if bvh is not None else None,
                    max_dist=max_dist,
                    exclude_indices=_batch_view(exclude_indices, batch),
                    indices=_batch_view(indices, batch),
                    distances=_batch_view(distances, batch),
                    counts=_batch_view(counts, batch),
                    squared=squared,
                    leaf_size=leaf_size,
                    query_stride=query_stride,
                    ordered=ordered,
                    backend=backend,
                    objective=objective,
                )
                indexes.append(result[0])
            return bvh if bvh is not None else tuple(indexes), indices, distances, counts
        if bvh is None:
            bvh = (
                CubqlPointIndex(points, nq, k, objective)
                if backend == PointQueryBackend.CUBQL
                else wp.Bvh(points, points, leaf_size=leaf)
            )
        keys = order = bounds = None
        if nq:
            if backend == PointQueryBackend.CUBQL:
                _, _, cubql_stride, block_dim = _cubql_config(num_points, nq, k, bvh.objective)
                if query_stride is not None:
                    cubql_stride = query_stride
                if not context.runtime.core.wp_point_query_cubql_query_device(
                    bvh.id,
                    points.ptr,
                    queries.ptr,
                    nq,
                    k,
                    cubql_stride,
                    block_dim,
                    indices.ptr,
                    distances.ptr,
                    counts.ptr,
                    indices.dtype == wp.int64,
                    squared,
                ):
                    raise RuntimeError(f"Direct cuBQL query failed: {context.runtime.get_error_string()}")
            elif use_order:
                keys = wp.empty(2 * nq, dtype=int, device=device)
                order = wp.empty_like(keys)
                bounds = wp.empty((2, 3), dtype=float, device=device)
                workers = min(nq, 256)
                wp.launch(_init_bounds, 3, inputs=[bounds], device=device, record_tape=False)
                wp.launch(_query_bounds, workers, inputs=[queries, bounds, workers], device=device, record_tape=False)
                wp.launch(_query_codes, nq, inputs=[queries, bounds, keys, order], device=device, record_tape=False)
                radix_sort_pairs(keys, order, nq)
            if backend == PointQueryBackend.WARP:
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
