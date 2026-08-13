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
from warp._src.utils import radix_sort_pairs

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
def _pack_half_edges(
    tri: wp.array2d[wp.int32],
    num_verts: wp.int32,
    keys: wp.array[wp.int64],
    values: wp.array[wp.int32],
):
    t = wp.tid()
    for j in range(3):
        v1 = tri[t, (j + 1) % 3]
        v2 = tri[t, (j + 2) % 3]
        lo = wp.min(v1, v2)
        hi = wp.max(v1, v2)
        # Pack the undirected edge (lo, hi) into a single sortable key.
        keys[t * 3 + j] = wp.int64(lo) * wp.int64(num_verts) + wp.int64(hi)
        values[t * 3 + j] = t * 3 + j


@wp.kernel
def _match_half_edges(
    keys: wp.array[wp.int64],
    values: wp.array[wp.int32],
    num_half_edges: wp.int32,
    TT: wp.array2d[wp.int32],
    TTi: wp.array2d[wp.int32],
):
    k = wp.tid()

    # Only act at the start of a run of two identical keys (a manifold interior
    # edge). Boundary edges appear once and are left as -1; non-manifold edges
    # (runs of three or more) are skipped rather than mispaired.
    key = keys[k]
    if k > 0 and keys[k - 1] == key:
        return
    if k + 1 >= num_half_edges or keys[k + 1] != key:
        return
    if k + 2 < num_half_edges and keys[k + 2] == key:
        return

    e0 = values[k]
    e1 = values[k + 1]
    t0 = e0 // 3
    j0 = e0 % 3
    t1 = e1 // 3
    j1 = e1 % 3

    TT[t0, j0] = t1
    TTi[t0, j0] = j1
    TT[t1, j1] = t0
    TTi[t1, j1] = j0


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
        edges must be shared by exactly two triangles; non-manifold edges are
        ignored (left as boundaries).
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

    num_half_edges = 3 * num_tris

    # radix_sort_pairs requires 2*count storage for its scratch space.
    keys = wp.empty(shape=2 * num_half_edges, dtype=wp.int64, device=device)
    values = wp.empty(shape=2 * num_half_edges, dtype=wp.int32, device=device)

    wp.launch(
        _pack_half_edges,
        dim=num_tris,
        inputs=[indices, wp.int32(num_verts), keys, values],
        device=device,
    )

    radix_sort_pairs(keys, values, num_half_edges)

    wp.launch(
        _match_half_edges,
        dim=num_half_edges,
        inputs=[keys, values, wp.int32(num_half_edges), TT, TTi],
        device=device,
    )

    return TT, TTi


# ---------------------------------------------------------------------------
# Parallel Delaunay edge flipping
# ---------------------------------------------------------------------------


@wp.func
def _is_flippable(
    tri: wp.array2d[wp.int32],
    TT: wp.array2d[wp.int32],
    TTi: wp.array2d[wp.int32],
    pos: wp.array[wp.vec2],
    ref: wp.array[wp.vec2],
    has_ref: wp.int32,
    area_eps: float,
    ref_eps: float,
    t: wp.int32,
    j: wp.int32,
) -> bool:
    """Return whether the interior edge ``(t, j)`` should be flipped to restore Delaunay.

    The shared edge joins vertices ``a`` and ``b``; ``c`` is the apex in ``t``
    and ``d`` the apex in the neighboring triangle. A flip replaces edge ``ab``
    with edge ``cd``, producing triangles ``(c, a, d)`` and ``(c, d, b)``.
    """
    n = TT[t, j]
    jn = TTi[t, j]

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
    TTi: wp.array2d[wp.int32],
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
        if not _is_flippable(tri, TT, TTi, pos, ref, has_ref, area_eps, ref_eps, t, j):
            continue

        # A flip rewrites adjacency for its two triangles and the four
        # surrounding neighbors, so it must claim all six exclusively.
        prio = t * 3 + j
        jn = TTi[t, j]
        _claim(claim, t, prio)
        _claim(claim, n, prio)
        _claim(claim, TT[t, (j + 1) % 3], prio)
        _claim(claim, TT[t, (j + 2) % 3], prio)
        _claim(claim, TT[n, (jn + 1) % 3], prio)
        _claim(claim, TT[n, (jn + 2) % 3], prio)


@wp.kernel
def _apply_flips(
    tri: wp.array2d[wp.int32],
    TT: wp.array2d[wp.int32],
    TTi: wp.array2d[wp.int32],
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
        n = TT[t, j]
        if n < 0 or t >= n:
            continue
        if not _is_flippable(tri, TT, TTi, pos, ref, has_ref, area_eps, ref_eps, t, j):
            continue

        prio = t * 3 + j
        jn = TTi[t, j]

        j1 = (j + 1) % 3
        j2 = (j + 2) % 3
        jn1 = (jn + 1) % 3
        jn2 = (jn + 2) % 3

        # Outer neighbors around the quad and their local edge indices.
        n_bc = TT[t, j1]
        k_bc = TTi[t, j1]
        n_ca = TT[t, j2]
        k_ca = TTi[t, j2]
        n_ad = TT[n, jn1]
        k_ad = TTi[n, jn1]
        n_db = TT[n, jn2]
        k_db = TTi[n, jn2]

        # Proceed only if this edge won all six of its claimed triangles.
        if claim[t] != prio or claim[n] != prio:
            continue
        if n_bc >= 0 and claim[n_bc] != prio:
            continue
        if n_ca >= 0 and claim[n_ca] != prio:
            continue
        if n_ad >= 0 and claim[n_ad] != prio:
            continue
        if n_db >= 0 and claim[n_db] != prio:
            continue

        c = tri[t, j]
        a = tri[t, j1]
        b = tri[t, j2]
        d = tri[n, jn]

        # Rewrite the two triangles. The quad c, a, d, b (counterclockwise) is
        # split by the new diagonal c-d into (c, a, d) and (c, d, b).
        tri[t, 0] = c
        tri[t, 1] = a
        tri[t, 2] = d
        tri[n, 0] = c
        tri[n, 1] = d
        tri[n, 2] = b

        # Adjacency for the new triangle t = (c, a, d):
        #   edge 0 (a-d) -> former a-d neighbor, edge 1 (d-c) -> new diagonal,
        #   edge 2 (c-a) -> former c-a neighbor.
        TT[t, 0] = n_ad
        TTi[t, 0] = k_ad
        TT[t, 1] = n
        TTi[t, 1] = 2
        TT[t, 2] = n_ca
        TTi[t, 2] = k_ca

        # Adjacency for the new triangle n = (c, d, b):
        #   edge 0 (d-b) -> former d-b neighbor, edge 1 (b-c) -> former b-c
        #   neighbor, edge 2 (c-d) -> new diagonal.
        TT[n, 0] = n_db
        TTi[n, 0] = k_db
        TT[n, 1] = n_bc
        TTi[n, 1] = k_bc
        TT[n, 2] = t
        TTi[n, 2] = 1

        # Fix the back-pointers of the four outer neighbors.
        if n_ad >= 0:
            TT[n_ad, k_ad] = t
            TTi[n_ad, k_ad] = 0
        if n_ca >= 0:
            TT[n_ca, k_ca] = t
            TTi[n_ca, k_ca] = 2
        if n_db >= 0:
            TT[n_db, k_db] = n
            TTi[n_db, k_db] = 0
        if n_bc >= 0:
            TT[n_bc, k_bc] = n
            TTi[n_bc, k_bc] = 1

        wp.atomic_add(num_flips, 0, 1)


def delaunay_edge_flip(
    positions: wp.array,
    indices: wp.array,
    ref_positions: wp.array | None = None,
    max_passes: int = 1000,
    area_epsilon: float = 0.0,
    ref_epsilon: float = 1.0e-10,
) -> int:
    """Flip interior edges in place until the 2D triangulation is Delaunay.

    The triangulation is modified in place: ``indices`` is updated with the
    flipped connectivity while ``positions`` is left unchanged. Flips run in
    parallel on the Warp device using a priority-based maximal independent set,
    so no host round-trip is needed apart from the per-pass flip count used to
    detect convergence.

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
        The total number of edges flipped.

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
    TT, TTi = triangle_triangle_adjacency(indices, num_verts=num_verts)

    has_ref = wp.int32(1 if ref_positions is not None else 0)
    ref = ref_positions if ref_positions is not None else positions

    claim = wp.empty(shape=num_tris, dtype=wp.int32, device=device)
    num_flips = wp.zeros(shape=1, dtype=wp.int32, device=device)

    total_flips = 0
    for _ in range(max_passes):
        claim.fill_(-1)
        num_flips.zero_()

        wp.launch(
            _claim_flips,
            dim=num_tris,
            inputs=[indices, TT, TTi, positions, ref, has_ref, area_epsilon, ref_epsilon, claim],
            device=device,
        )
        wp.launch(
            _apply_flips,
            dim=num_tris,
            inputs=[indices, TT, TTi, positions, ref, has_ref, area_epsilon, ref_epsilon, claim, num_flips],
            device=device,
        )

        pass_flips = int(num_flips.numpy()[0])
        total_flips += pass_flips
        if pass_flips == 0:
            break

    return total_flips
