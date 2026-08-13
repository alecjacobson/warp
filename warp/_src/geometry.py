# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

This module currently provides in-place Delaunay edge flipping for 2D triangle
meshes, along with the triangle-triangle adjacency structure it relies on. All
routines run entirely on the Warp device (CPU or CUDA); no host round-trips are
performed apart from reading back the per-pass flip count to detect convergence.
"""

from __future__ import annotations

import warp as wp
from warp._src.utils import array_scan

__all__ = [
    "delaunay_edge_flip",
    "in_circle",
    "triangle_triangle_adjacency",
]


# ---------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------


@wp.func
def _signed_area(a: wp.vec2, b: wp.vec2, c: wp.vec2) -> float:
    """Twice the signed area of triangle ``abc`` (positive when counterclockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@wp.func
def _in_circle_det(a: wp.vec2, b: wp.vec2, c: wp.vec2, d: wp.vec2) -> float:
    """Return the in-circle determinant for the circumcircle of triangle ``abc`` and point ``d``.

    For a counterclockwise triangle ``abc``, the result is positive when ``d``
    lies strictly inside the circumcircle, zero when the four points are
    cocircular, and negative when ``d`` lies outside.
    """
    ad = a - d
    bd = b - d
    cd = c - d
    ad2 = wp.dot(ad, ad)
    bd2 = wp.dot(bd, bd)
    cd2 = wp.dot(cd, cd)
    return (
        ad[0] * (bd[1] * cd2 - cd[1] * bd2)
        - ad[1] * (bd[0] * cd2 - cd[0] * bd2)
        + ad2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


@wp.func
def in_circle(a: wp.vec2, b: wp.vec2, c: wp.vec2, d: wp.vec2) -> bool:
    """Return ``True`` if point ``d`` lies inside the circumcircle of triangle ``abc``.

    The triangle ``abc`` is assumed to be counterclockwise. This uses the
    standard floating-point in-circle determinant, which is accurate for
    well-conditioned configurations but is not an exact geometric predicate.
    """
    return _in_circle_det(a, b, c, d) > 0.0


# ---------------------------------------------------------------------------
# Triangle-triangle adjacency
# ---------------------------------------------------------------------------


@wp.kernel
def _count_vertex_edges(tri: wp.array2d[wp.int32], counts: wp.array[wp.int32]):
    # Bucket each half-edge under its lower-indexed endpoint.
    t = wp.tid()
    for j in range(3):
        v1 = tri[t, (j + 1) % 3]
        v2 = tri[t, (j + 2) % 3]
        wp.atomic_add(counts, wp.min(v1, v2), 1)


@wp.kernel
def _scatter_vertex_edges(
    tri: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    fill: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
):
    # Scatter each half-edge into its lower-endpoint bucket, recording the upper
    # endpoint and the packed half-edge id ``tri * 3 + local_edge``.
    t = wp.tid()
    for j in range(3):
        v1 = tri[t, (j + 1) % 3]
        v2 = tri[t, (j + 2) % 3]
        lo = wp.min(v1, v2)
        hi = wp.max(v1, v2)
        slot = offsets[lo] + wp.atomic_add(fill, lo, 1)
        bucket_hi[slot] = hi
        bucket_he[slot] = t * 3 + j


@wp.kernel
def _match_vertex_buckets(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
    TT: wp.array2d[wp.int32],
):
    # Each thread owns one vertex's bucket (a disjoint range), so it can pair
    # half-edges that share the same upper endpoint without synchronization. The
    # bucket holds only edges incident to this vertex (average degree ~6), so the
    # quadratic inner scan is over a handful of entries.
    v = wp.tid()
    beg = offsets[v]
    end = beg + counts[v]
    for a in range(beg, end):
        he_a = bucket_he[a]
        if he_a < 0:  # already paired
            continue
        hi_a = bucket_hi[a]
        for b in range(a + 1, end):
            if bucket_he[b] >= 0 and bucket_hi[b] == hi_a:
                he_b = bucket_he[b]
                TT[he_a // 3, he_a % 3] = he_b // 3
                TT[he_b // 3, he_b % 3] = he_a // 3
                bucket_he[a] = -1
                bucket_he[b] = -1
                break


@wp.kernel
def _match_vertex_buckets_full(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
    TT: wp.array2d[wp.int32],
    TTi: wp.array2d[wp.int32],
):
    # As _match_vertex_buckets, but also records the reciprocal local edge index.
    v = wp.tid()
    beg = offsets[v]
    end = beg + counts[v]
    for a in range(beg, end):
        he_a = bucket_he[a]
        if he_a < 0:
            continue
        hi_a = bucket_hi[a]
        for b in range(a + 1, end):
            if bucket_he[b] >= 0 and bucket_hi[b] == hi_a:
                he_b = bucket_he[b]
                ta = he_a // 3
                ja = he_a % 3
                tb = he_b // 3
                jb = he_b % 3
                TT[ta, ja] = tb
                TTi[ta, ja] = jb
                TT[tb, jb] = ta
                TTi[tb, jb] = ja
                bucket_he[a] = -1
                bucket_he[b] = -1
                break


def _bucket_half_edges(indices: wp.array, num_verts: int, device, num_tris: int):
    """Counting-sort the half-edges into per-vertex buckets keyed by their lower endpoint.

    Returns ``(offsets, counts, bucket_hi, bucket_he)``: for vertex ``v`` the slice
    ``[offsets[v] : offsets[v] + counts[v]]`` of ``bucket_hi`` / ``bucket_he`` lists
    the upper endpoint and packed half-edge id ``tri * 3 + local_edge`` of every
    half-edge whose lower endpoint is ``v``. Uses :func:`warp.utils.array_scan`
    rather than a global key sort, and needs no host synchronization.
    """
    num_half_edges = 3 * num_tris

    counts = wp.zeros(shape=num_verts, dtype=wp.int32, device=device)
    offsets = wp.empty(shape=num_verts, dtype=wp.int32, device=device)
    fill = wp.zeros(shape=num_verts, dtype=wp.int32, device=device)
    bucket_hi = wp.empty(shape=num_half_edges, dtype=wp.int32, device=device)
    bucket_he = wp.empty(shape=num_half_edges, dtype=wp.int32, device=device)

    wp.launch(_count_vertex_edges, dim=num_tris, inputs=[indices, counts], device=device)
    array_scan(counts, offsets, inclusive=False)
    wp.launch(
        _scatter_vertex_edges,
        dim=num_tris,
        inputs=[indices, offsets, fill, bucket_hi, bucket_he],
        device=device,
    )
    return offsets, counts, bucket_hi, bucket_he


def _build_triangle_adjacency(indices: wp.array, num_verts: int) -> wp.array:
    """Build the triangle-triangle neighbor array ``TT`` only (no reciprocal slots).

    Used internally by :func:`delaunay_edge_flip`, which recovers reciprocal edge
    indices on demand and therefore does not need ``TTi``.
    """
    device = indices.device
    num_tris = indices.shape[0]

    TT = wp.full(shape=(num_tris, 3), value=-1, dtype=wp.int32, device=device)
    if num_tris == 0:
        return TT

    offsets, counts, bucket_hi, bucket_he = _bucket_half_edges(indices, num_verts, device, num_tris)
    wp.launch(
        _match_vertex_buckets,
        dim=num_verts,
        inputs=[offsets, counts, bucket_hi, bucket_he, TT],
        device=device,
    )
    return TT


def triangle_triangle_adjacency(indices: wp.array, num_verts: int | None = None):
    """Build triangle-triangle adjacency for a triangle mesh.

    Args:
        indices: A ``(num_tris, 3)`` :class:`warp.array` of triangle vertex
            indices (``int32``).
        num_verts: Number of vertices in the mesh. If ``None``, inferred as one
            plus the maximum vertex index, which requires a host synchronization.

    Returns:
        A tuple ``(TT, TTi)`` of ``(num_tris, 3)`` ``int32`` arrays. ``TT[t, j]``
        is the triangle adjacent to triangle ``t`` across the edge opposite
        local vertex ``j`` (the edge joining local vertices ``(j + 1) % 3`` and
        ``(j + 2) % 3``), or ``-1`` on a boundary edge. ``TTi[t, j]`` is the
        local edge index of that shared edge within the neighboring triangle.

    Note:
        Assumes an orientable manifold mesh with consistent winding. Interior
        edges must be shared by exactly two triangles; for a non-manifold edge
        shared by more triangles only an arbitrary two are paired.

        Adjacency is built by counting-sorting the half-edges into per-vertex
        buckets (via :func:`warp.utils.array_scan`) and matching within each
        bucket -- no global key sort -- so the build is CUDA-graph capturable and
        needs no host synchronization when ``num_verts`` is given.
    """
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must be a (num_tris, 3) array of triangle vertex indices")

    device = indices.device
    num_tris = indices.shape[0]

    TT = wp.full(shape=(num_tris, 3), value=-1, dtype=wp.int32, device=device)
    TTi = wp.full(shape=(num_tris, 3), value=-1, dtype=wp.int32, device=device)

    if num_tris == 0:
        return TT, TTi

    if num_verts is None:
        num_verts = int(indices.numpy().max()) + 1

    offsets, counts, bucket_hi, bucket_he = _bucket_half_edges(indices, num_verts, device, num_tris)
    wp.launch(
        _match_vertex_buckets_full,
        dim=num_verts,
        inputs=[offsets, counts, bucket_hi, bucket_he, TT, TTi],
        device=device,
    )

    return TT, TTi


# ---------------------------------------------------------------------------
# Parallel Delaunay edge flipping
# ---------------------------------------------------------------------------


@wp.func
def _find_neighbor_edge(TT: wp.array2d[wp.int32], t: wp.int32, n: wp.int32) -> wp.int32:
    """Return the local edge of triangle ``t`` whose neighbor is ``n`` (or -1)."""
    if TT[t, 0] == n:
        return 0
    if TT[t, 1] == n:
        return 1
    if TT[t, 2] == n:
        return 2
    return -1


@wp.func
def _is_flippable(
    tri: wp.array2d[wp.int32],
    pos: wp.array[wp.vec2],
    ref: wp.array[wp.vec2],
    has_ref: wp.int32,
    area_eps: float,
    ref_eps: float,
    t: wp.int32,
    j: wp.int32,
    n: wp.int32,
    jn: wp.int32,
) -> bool:
    """Return whether the interior edge between triangles ``t`` and ``n`` should flip.

    ``j`` is the local edge of ``t`` opposite apex ``c`` and ``jn`` the local edge
    of ``n`` opposite apex ``d``. The shared edge joins vertices ``a`` and ``b``. A
    flip replaces edge ``ab`` with edge ``cd``, producing triangles ``(c, a, d)``
    and ``(c, d, b)``.
    """
    c = tri[t, j]
    a = tri[t, (j + 1) % 3]
    b = tri[t, (j + 2) % 3]
    d = tri[n, jn]

    pa = pos[a]
    pb = pos[b]
    pc = pos[c]
    pd = pos[d]

    # Reject non-convex quads: both resulting triangles must be counterclockwise.
    if _signed_area(pc, pa, pd) <= area_eps:
        return False
    if _signed_area(pc, pd, pb) <= area_eps:
        return False

    # Delaunay test: flip only if the opposite apex is inside the circumcircle.
    # Triangle (c, a, b) is counterclockwise, matching the input winding.
    if _in_circle_det(pc, pa, pb, pd) <= 0.0:
        return False

    # Reject flips that would create degenerate triangles in a reference config.
    if has_ref != 0:
        ra = ref[a]
        rb = ref[b]
        rc = ref[c]
        rd = ref[d]
        if wp.abs(_signed_area(rc, ra, rd)) <= ref_eps:
            return False
        if wp.abs(_signed_area(rc, rd, rb)) <= ref_eps:
            return False

    return True


@wp.func
def _claim(claim: wp.array[wp.int32], t: wp.int32, prio: wp.int32):
    if t >= 0:
        wp.atomic_max(claim, t, prio)


@wp.kernel
def _claim_flips(
    tri: wp.array2d[wp.int32],
    TT: wp.array2d[wp.int32],
    pos: wp.array[wp.vec2],
    ref: wp.array[wp.vec2],
    has_ref: wp.int32,
    area_eps: float,
    ref_eps: float,
    claim: wp.array[wp.int32],
):
    t = wp.tid()
    for j in range(3):
        n = TT[t, j]
        # Process each undirected edge once, from its lower-indexed triangle.
        if n < 0 or t >= n:
            continue
        jn = _find_neighbor_edge(TT, n, t)
        if jn < 0:
            continue
        if not _is_flippable(tri, pos, ref, has_ref, area_eps, ref_eps, t, j, n, jn):
            continue

        # Preserving each triangle's cyclic slot layout, a flip only reads and
        # writes the rows of {t, n, n_bc, n_ad}, so it claims exactly those four.
        n_bc = TT[t, (j + 1) % 3]
        n_ad = TT[n, (jn + 1) % 3]
        prio = t * 3 + j
        _claim(claim, t, prio)
        _claim(claim, n, prio)
        _claim(claim, n_bc, prio)
        _claim(claim, n_ad, prio)


@wp.kernel
def _apply_flips(
    tri: wp.array2d[wp.int32],
    TT: wp.array2d[wp.int32],
    pos: wp.array[wp.vec2],
    ref: wp.array[wp.vec2],
    has_ref: wp.int32,
    area_eps: float,
    ref_eps: float,
    claim: wp.array[wp.int32],
    num_flips: wp.array[wp.int32],
):
    t = wp.tid()
    for j in range(3):
        prio = t * 3 + j

        # Check ownership before reading any mutable topology: if this edge did
        # not win row t, another concurrent flip may be rewriting it. An edge that
        # placed a claim here passed the Delaunay predicate in _claim_flips, and no
        # winning conflicting flip can touch its four protected rows, so there is
        # no need to re-run the predicate.
        if claim[t] != prio:
            continue
        n = TT[t, j]
        if n < 0 or t >= n:
            continue
        if claim[n] != prio:
            continue

        jn = _find_neighbor_edge(TT, n, t)
        if jn < 0:
            continue

        j1 = (j + 1) % 3
        j2 = (j + 2) % 3
        jn1 = (jn + 1) % 3
        jn2 = (jn + 2) % 3

        n_bc = TT[t, j1]
        n_ad = TT[n, jn1]
        if n_bc >= 0 and claim[n_bc] != prio:
            continue
        if n_ad >= 0 and claim[n_ad] != prio:
            continue

        # This edge owns {t, n, n_bc, n_ad}; their rows are stable for this pass.
        #   t: (c, a, b) -> (c, a, d)   n: (d, b, a) -> (d, b, c)
        # so the new diagonal is c-d and only slots j2 and jn2 change vertex.
        c = tri[t, j]
        d = tri[n, jn]
        tri[t, j2] = d
        tri[n, jn2] = c

        # t keeps slot j2 (edge c-a); slot j (edge a-d) inherits n's a-d neighbor;
        # slot j1 (edge d-c) is the new diagonal to n.
        TT[t, j] = n_ad
        TT[t, j1] = n
        # n keeps slot jn2 (edge d-b); slot jn (edge b-c) inherits t's b-c
        # neighbor; slot jn1 (edge c-d) is the new diagonal to t.
        TT[n, jn] = n_bc
        TT[n, jn1] = t

        # Only these two outer neighbors change ownership; re-point them.
        if n_ad >= 0:
            TT[n_ad, _find_neighbor_edge(TT, n_ad, n)] = t
        if n_bc >= 0:
            TT[n_bc, _find_neighbor_edge(TT, n_bc, t)] = n

        wp.atomic_add(num_flips, 0, 1)


@wp.kernel
def _update_condition(
    num_flips: wp.array[wp.int32],
    max_passes: wp.int32,
    total_flips: wp.array[wp.int32],
    pass_count: wp.array[wp.int32],
    condition: wp.array[wp.int32],
):
    # Single-thread bookkeeping between passes: accumulate the running total and
    # decide whether another pass is warranted. Written to a device array so the
    # convergence loop needs no host synchronization under graph capture.
    total_flips[0] += num_flips[0]
    pass_count[0] += 1
    if num_flips[0] > 0 and pass_count[0] < max_passes:
        condition[0] = 1
    else:
        condition[0] = 0


def delaunay_edge_flip(
    positions: wp.array,
    indices: wp.array,
    ref_positions: wp.array | None = None,
    max_passes: int = 1000,
    area_epsilon: float = 0.0,
    ref_epsilon: float = 1.0e-10,
):
    """Flip interior edges in place until the 2D triangulation is Delaunay.

    The triangulation is modified in place: ``indices`` is updated with the
    flipped connectivity while ``positions`` is left unchanged. Flips run in
    parallel on the Warp device using a priority-based maximal independent set.
    The convergence loop is driven on-device with :func:`warp.capture_while`, so
    the whole routine can be recorded into a CUDA graph without host
    synchronization (a warm-up call is recommended so that internal allocations,
    including the radix-sort scratch buffer, are sized before capture).

    Args:
        positions: A ``(num_verts,)`` :class:`warp.array` of :class:`warp.vec2`
            vertex positions.
        indices: A ``(num_tris, 3)`` :class:`warp.array` of ``int32`` triangle
            vertex indices, assumed counterclockwise. Modified in place.
        ref_positions: Optional ``(num_verts,)`` :class:`warp.array` of
            :class:`warp.vec2` reference positions. When provided, flips that
            would create a degenerate triangle in the reference configuration
            are rejected. Useful when the working mesh is a deformation of a
            reference mesh that must stay non-degenerate.
        max_passes: Maximum number of parallel flip passes before stopping.
        area_epsilon: Minimum signed area (twice the triangle area) required for
            each triangle produced by a flip; guards against creating inverted
            or sliver triangles.
        ref_epsilon: Degeneracy threshold applied to ``ref_positions``.

    Returns:
        In eager mode, the total number of edges flipped as a Python ``int``.
        During CUDA graph capture the count is only known at replay time, so a
        single-element ``int32`` :class:`warp.array` accumulator is returned
        instead; read it after :func:`warp.capture_launch`.

    Note:
        Assumes an orientable manifold mesh with consistent counterclockwise
        winding. See :func:`triangle_triangle_adjacency`.
    """
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must be a (num_tris, 3) array of triangle vertex indices")

    device = indices.device
    num_tris = indices.shape[0]
    if num_tris == 0:
        return 0

    if positions.device != device:
        raise ValueError("positions and indices must be on the same device")
    if ref_positions is not None and ref_positions.device != device:
        raise ValueError("ref_positions and indices must be on the same device")

    num_verts = positions.shape[0]
    TT = _build_triangle_adjacency(indices, num_verts)

    has_ref = wp.int32(1 if ref_positions is not None else 0)
    ref = ref_positions if ref_positions is not None else positions
    max_passes_i = wp.int32(max_passes)

    claim = wp.empty(shape=num_tris, dtype=wp.int32, device=device)
    num_flips = wp.empty(shape=1, dtype=wp.int32, device=device)
    total_flips = wp.zeros(shape=1, dtype=wp.int32, device=device)
    pass_count = wp.zeros(shape=1, dtype=wp.int32, device=device)
    # Seed the condition so the loop body runs at least once; the body clears it
    # once a pass makes no progress (or the pass budget is exhausted).
    condition = wp.ones(shape=1, dtype=wp.int32, device=device)

    def _flip_pass():
        claim.fill_(-1)
        num_flips.zero_()
        wp.launch(
            _claim_flips,
            dim=num_tris,
            inputs=[indices, TT, positions, ref, has_ref, area_epsilon, ref_epsilon, claim],
            device=device,
        )
        wp.launch(
            _apply_flips,
            dim=num_tris,
            inputs=[indices, TT, positions, ref, has_ref, area_epsilon, ref_epsilon, claim, num_flips],
            device=device,
        )
        wp.launch(
            _update_condition,
            dim=1,
            inputs=[num_flips, max_passes_i, total_flips, pass_count, condition],
            device=device,
        )

    # Device-driven convergence loop. Under CUDA graph capture this records
    # conditional graph nodes; otherwise wp.capture_while reads the condition
    # back to the host to decide when to stop.
    wp.capture_while(condition, _flip_pass)

    if device.is_capturing:
        return total_flips

    return int(total_flips.numpy()[0])
