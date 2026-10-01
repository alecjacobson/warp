# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete exterior calculus operators for triangle and tetrahedral meshes.

The public entry point is :func:`dec_operators`, which returns the exterior
derivative ``d0`` (the signed edge-vertex incidence matrix) and the diagonal
Hodge star ``star1`` (the per-edge cotangent weights), so that

    ``laplacian(points, indices) == d0^T @ star1 @ d0``.

The per-edge weights are computed with the same device functions the cotangent
Laplacian uses (:func:`~warp._src.geometry.laplacian.triangle_cotangent_weights`
and :func:`~warp._src.geometry.laplacian.tet_edge_weights`), so the two operators
agree by construction and the weight code is not duplicated.

Unique edges are enumerated by emitting one ``(min, max)`` entry per element edge
and letting :func:`warp.sparse.bsr_from_triplets` sort and deduplicate them. The
same pass carries each edge's weight as the triplet value, so the deduplication
also accumulates ``star1`` -- in a deterministic, sorted order rather than with
atomics -- and the compressed pattern's stored order defines the edge ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from warp._src.geometry.laplacian import (
    _TET_EDGE_A,
    _TET_EDGE_B,
    _resolve_num_points,
    _resolve_simplex_indices,
    tet_edge_weights,
    triangle_cotangent_weights,
)
from warp._src.sparse import (
    BsrMatrix,
    bsr_from_triplets,
)

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


@wp.kernel
def triangle_edge_weight_triplets(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    out_rows: wp.array[int],
    out_columns: wp.array[int],
    out_values: wp.array[float],
):
    # One triplet per triangle edge: the edge as ``(min, max)`` endpoints and,
    # as its value, half the cotangent of the angle opposite it -- the same weight
    # ``laplacian`` places off the diagonal. Deduplicating sums these over the
    # triangles sharing each edge, giving ``star1``.
    tri = wp.tid()
    v = wp.vec3i(indices[3 * tri + 0], indices[3 * tri + 1], indices[3 * tri + 2])
    c = triangle_cotangent_weights(points[v[0]], points[v[1]], points[v[2]])
    for k in range(3):
        a = v[(k + 1) % 3]
        b = v[(k + 2) % 3]
        out = 3 * tri + k
        out_rows[out] = wp.min(a, b)
        out_columns[out] = wp.max(a, b)
        out_values[out] = c[k]


@wp.kernel
def tet_edge_weight_triplets(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    out_rows: wp.array[int],
    out_columns: wp.array[int],
    out_values: wp.array[float],
):
    tet = wp.tid()
    iv = wp.vec4i(indices[4 * tet + 0], indices[4 * tet + 1], indices[4 * tet + 2], indices[4 * tet + 3])
    w = tet_edge_weights(points[iv[0]], points[iv[1]], points[iv[2]], points[iv[3]])
    ea = _TET_EDGE_A
    eb = _TET_EDGE_B
    for k in range(6):
        a = iv[ea[k]]
        b = iv[eb[k]]
        out = 6 * tet + k
        out_rows[out] = wp.min(a, b)
        out_columns[out] = wp.max(a, b)
        out_values[out] = w[k]


@wp.kernel
def build_d0(
    edge_rows: wp.array[int],
    edge_columns: wp.array[int],
    out_rows: wp.array[int],
    out_columns: wp.array[int],
    out_values: wp.array[float],
):
    # Row ``e`` of ``d0`` is the edge oriented from its lower-indexed endpoint to
    # its higher-indexed one: ``-1`` at the tail, ``+1`` at the head.
    e = wp.tid()
    out_rows[2 * e + 0] = e
    out_columns[2 * e + 0] = edge_rows[e]
    out_values[2 * e + 0] = -1.0

    out_rows[2 * e + 1] = e
    out_columns[2 * e + 1] = edge_columns[e]
    out_values[2 * e + 1] = 1.0


def dec_operators(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    *,
    num_points: int | None = None,
    simplex_size: int | None = None,
    device: DeviceLike | None = None,
) -> tuple[BsrMatrix, wp.array]:
    """Assemble the DEC exterior derivative ``d0`` and diagonal Hodge star ``star1``.

    ``d0`` is the signed edge-vertex incidence matrix of shape
    ``(num_edges, num_points)``: each row is one undirected edge, with ``-1`` at
    its lower-indexed endpoint and ``+1`` at the higher. ``star1`` is the diagonal
    of the Hodge star, returned as the length-``num_edges`` array of per-edge
    cotangent weights (a diagonal operator is its diagonal, so storing it as a
    vector rather than a sparse matrix is both smaller and cheaper to use).
    Together they factor the cotangent Laplacian,

        ``d0^T @ diag(star1) @ d0 == laplacian(points, indices)``,

    which is the construction to reach for when the per-edge weights are the free
    variables -- for example optimizing ``star1`` as a metric while keeping the
    topological ``d0`` fixed. The factored operator need never be assembled: it
    acts on a vector matrix-free as ``d0^T @ (star1 * (d0 @ x))`` with two
    :func:`~warp.sparse.bsr_mv` calls and an elementwise product.

    The per-edge weights use the same device functions as
    :func:`~warp.geometry.laplacian`, so the two operators agree by construction.
    The weights are the sum, over the elements sharing an edge, of that element's
    cotangent contribution; they are positive on a well-shaped (intrinsically
    Delaunay) mesh but may be negative where a mesh has very obtuse angles, exactly
    as the off-diagonal Laplacian entries are. The weights are accumulated by the
    sorted deduplication of :func:`warp.sparse.bsr_from_triplets`, so they are
    deterministic (bit-for-bit reproducible across runs).

    With respect to ``points`` the operation is differentiable through ``star1``
    (``d0`` is purely topological): launch it inside a :class:`warp.Tape` with
    ``points.requires_grad`` set. The gradient of an element's contribution is
    undefined where it degenerates, so degenerate elements produce non-finite
    gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Mesh connectivity, either a flat :class:`warp.int32` array or a
            typed array of :class:`warp.vec3i` (triangles) or :class:`warp.vec4i`
            (tetrahedra); see :func:`~warp.geometry.laplacian`.
        num_points: Number of vertices. Inferred from ``points`` when omitted.
        simplex_size: Vertices per element (3 or 4) when ``indices`` is a flat
            array; see :func:`~warp.geometry.laplacian`.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(d0, star1)``. ``d0`` is a :class:`warp.sparse.BsrMatrix` with
        scalar (1x1) blocks and :class:`warp.float32` coefficients of shape
        ``(num_edges, num_points)``. ``star1`` is a :class:`warp.array` of
        :class:`warp.float32` and length ``num_edges`` holding the per-edge Hodge
        star weights (the diagonal of ``diag(star1)``).

    Note:
        Enumerating the unique edges reads the edge count back to the host, so
        this call is not CUDA-graph capturable.
    """
    device = wp.get_device(device) if device is not None else points.device
    indices, resolved_simplex_size = _resolve_simplex_indices(indices, simplex_size)
    num_points = _resolve_num_points(points, indices, num_points, None, device)
    num_simplices = indices.shape[0] // resolved_simplex_size

    # Emit one weighted triplet per element edge. Sorting and deduplicating them
    # yields the unique edges (whose stored order is the edge id) and, because the
    # values are summed, the per-edge weights of ``star1`` in one pass.
    edges_per_simplex = 3 if resolved_simplex_size == 3 else 6
    triplet_count = num_simplices * edges_per_simplex
    edge_rows = wp.empty(triplet_count, dtype=wp.int32, device=device)
    edge_columns = wp.empty(triplet_count, dtype=wp.int32, device=device)
    edge_values = wp.empty(triplet_count, dtype=wp.float32, device=device, requires_grad=points.requires_grad)
    edge_kernel = triangle_edge_weight_triplets if resolved_simplex_size == 3 else tet_edge_weight_triplets
    wp.launch(
        edge_kernel,
        dim=num_simplices,
        inputs=[points, indices],
        outputs=[edge_rows, edge_columns, edge_values],
        device=device,
    )

    # Keeping numerical zeros preserves every edge in the pattern even where a
    # weight vanishes (e.g. a right angle on a grid), so ``d0`` stays complete.
    weighted = bsr_from_triplets(
        num_points, num_points, edge_rows, edge_columns, edge_values, prune_numerical_zeros=False
    )
    num_edges = weighted.nnz_sync()

    # ``star1``: the accumulated per-edge weights. The compressed matrix stores one
    # value per unique edge, in edge-id order, so its scalar values are exactly the
    # Hodge star diagonal (grad-connected to ``points`` through the triplets).
    star1 = weighted.values[:num_edges].reshape(num_edges)

    # ``d0``: two signed entries per edge, placed by edge id.
    unique_rows = weighted.uncompress_rows()
    d0_rows = wp.empty(2 * num_edges, dtype=wp.int32, device=device)
    d0_columns = wp.empty(2 * num_edges, dtype=wp.int32, device=device)
    d0_values = wp.empty(2 * num_edges, dtype=wp.float32, device=device)
    wp.launch(
        build_d0,
        dim=num_edges,
        inputs=[unique_rows, weighted.columns],
        outputs=[d0_rows, d0_columns, d0_values],
        device=device,
    )
    d0 = bsr_from_triplets(num_edges, num_points, d0_rows, d0_columns, d0_values, prune_numerical_zeros=False)

    return d0, star1
