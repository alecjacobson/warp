# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

The public entry point is :func:`triangle_mesh_topology_statistics`, which
summarizes the combinatorial topology of a triangle mesh -- edge incidence and
orientation, vertex manifoldness, and degeneracies -- from an oriented flat
triangle-index array. It requires no vertex positions.

The statistics are gathered on the Warp device (CPU or CUDA) with a
sort-free pipeline: a per-triangle counting pass, a prefix scan, a scatter that
builds a vertex-to-incident-corner CSR, and a per-vertex analysis pass. Only a
single small counter array is read back to the host to assemble the returned
:class:`TriangleMeshTopologyStatistics`.

Kernels and helpers that are specific to this routine are grouped in the private
:class:`_TopologyStatistics` class so their names stay tied to the algorithm and
do not clutter the module namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp
from warp._src.utils import array_scan

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

__all__ = [
    "TriangleMeshTopologyStatistics",
    "triangle_mesh_topology_statistics",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriangleMeshTopologyStatistics:
    """Combinatorial topology statistics for a triangle mesh.

    All counts are computed from triangle connectivity alone (no vertex
    positions). See :func:`triangle_mesh_topology_statistics` for the exact
    definitions and the meaning of a combinatorially degenerate triangle.
    """

    num_vertices: int
    """Number of mesh vertices (referenced or not)."""
    num_triangles: int
    """Number of input triangles, including degenerate ones."""

    num_edges: int
    """Number of unique undirected edges, ignoring degenerate triangles."""
    num_boundary_edges: int
    """Edges incident to exactly one triangle half-edge."""
    num_nonmanifold_edges: int
    """Edges incident to more than two triangle half-edges."""
    num_misoriented_edges: int
    """Interior edges (incidence two) whose two half-edges traverse the edge in the same direction."""

    num_nonmanifold_vertices: int
    """Referenced vertices whose incident faces form more than one edge-connected fan."""
    num_unreferenced_vertices: int
    """Vertices not referenced by any nondegenerate triangle."""

    num_degenerate_triangles: int
    """Triangles with a repeated vertex index."""

    @property
    def is_edge_manifold(self) -> bool:
        """No edge is incident to more than two triangles, and no triangle is degenerate."""
        return self.num_nonmanifold_edges == 0 and self.num_degenerate_triangles == 0

    @property
    def is_closed_edge_manifold(self) -> bool:
        """Edge-manifold with no boundary edges."""
        return self.is_edge_manifold and self.num_boundary_edges == 0

    @property
    def is_vertex_manifold(self) -> bool:
        """Every vertex is referenced and its incident faces form a single fan."""
        return (
            self.num_nonmanifold_vertices == 0
            and self.num_unreferenced_vertices == 0
            and self.num_degenerate_triangles == 0
        )

    @property
    def is_manifold(self) -> bool:
        """Both edge- and vertex-manifold; boundaries are allowed."""
        return self.is_edge_manifold and self.is_vertex_manifold

    @property
    def is_closed_manifold(self) -> bool:
        """Manifold with no boundary edges."""
        return self.is_closed_edge_manifold and self.is_vertex_manifold

    @property
    def is_oriented(self) -> bool:
        """Every interior edge is traversed once in each direction."""
        return self.num_misoriented_edges == 0


# ---------------------------------------------------------------------------
# Raw counter layout (kept in a single device array so it reads back at once)
# ---------------------------------------------------------------------------

_STAT_DEGENERATE_TRIANGLES = wp.constant(0)
_STAT_NUM_EDGES = wp.constant(1)
_STAT_NUM_BOUNDARY_EDGES = wp.constant(2)
_STAT_NUM_NONMANIFOLD_EDGES = wp.constant(3)
_STAT_NUM_MISORIENTED_EDGES = wp.constant(4)
_STAT_NUM_NONMANIFOLD_VERTICES = wp.constant(5)
_STAT_NUM_UNREFERENCED_VERTICES = wp.constant(6)
_STAT_COUNT = 7


class _TopologyStatistics:
    """Kernels and device helpers for :func:`triangle_mesh_topology_statistics`.

    Grouped in a private class -- not part of the public API -- so their names
    stay tied to the algorithm and do not clutter the module namespace.
    """

    @wp.func
    def _corner_neighbors(indices: wp.array[wp.int32], corner: wp.int32):
        """Return ``(v, prev, next)`` for an incident corner id ``3 * tri + local``.

        ``v`` is the corner's vertex; in the triangle's orientation the boundary
        traverses ``prev -> v -> next``. All three are distinct because degenerate
        triangles are excluded from the CSR.
        """
        t = corner // 3
        c = corner % 3
        v = indices[3 * t + c]
        prv = indices[3 * t + (c + 2) % 3]
        nxt = indices[3 * t + (c + 1) % 3]
        return v, prv, nxt

    @wp.func
    def _faces_adjacent_at_v(indices: wp.array[wp.int32], corner_a: wp.int32, corner_b: wp.int32) -> bool:
        """Return whether two faces incident to a shared vertex also share a mesh edge.

        Two triangles around vertex ``v`` are adjacent iff they share an edge
        containing ``v``, i.e. one of ``v``'s two neighbors in face ``A`` matches
        one of its neighbors in face ``B``.
        """
        _, pa, na = _TopologyStatistics._corner_neighbors(indices, corner_a)
        _, pb, nb = _TopologyStatistics._corner_neighbors(indices, corner_b)
        return pa == pb or pa == nb or na == pb or na == nb

    @wp.kernel(enable_backward=False)
    def _check_bounds(indices: wp.array[wp.int32], num_points: wp.int32, bad: wp.array[wp.int32]):
        i = indices[wp.tid()]
        if i < 0 or i >= num_points:
            wp.atomic_max(bad, 0, 1)

    @wp.kernel(enable_backward=False)
    def _max_vertex_index(indices: wp.array[wp.int32], out_max: wp.array[wp.int32]):
        wp.atomic_max(out_max, 0, indices[wp.tid()])

    @wp.kernel(enable_backward=False)
    def _count_incident_corners(
        indices: wp.array[wp.int32],
        vertex_offsets: wp.array[wp.int32],
        raw_stats: wp.array[wp.int32],
    ):
        # One thread per triangle. Bucket each corner under its vertex (offset by
        # one so an inclusive scan turns the counts into CSR offsets), and set
        # aside combinatorially degenerate triangles.
        t = wp.tid()
        i = indices[3 * t + 0]
        j = indices[3 * t + 1]
        k = indices[3 * t + 2]
        if i == j or j == k or k == i:
            wp.atomic_add(raw_stats, _STAT_DEGENERATE_TRIANGLES, 1)
            return
        wp.atomic_add(vertex_offsets, i + 1, 1)
        wp.atomic_add(vertex_offsets, j + 1, 1)
        wp.atomic_add(vertex_offsets, k + 1, 1)

    @wp.kernel(enable_backward=False)
    def _scatter_incident_corners(
        indices: wp.array[wp.int32],
        vertex_offsets: wp.array[wp.int32],
        vertex_cursors: wp.array[wp.int32],
        incident_corners: wp.array[wp.int32],
    ):
        # One thread per triangle. Scatter each corner id into its vertex's CSR
        # range; ordering within a range is unspecified.
        t = wp.tid()
        i = indices[3 * t + 0]
        j = indices[3 * t + 1]
        k = indices[3 * t + 2]
        if i == j or j == k or k == i:
            return
        for c in range(3):
            v = indices[3 * t + c]
            slot = vertex_offsets[v] + wp.atomic_add(vertex_cursors, v, 1)
            incident_corners[slot] = 3 * t + c

    @wp.kernel(enable_backward=False)
    def _analyze_vertices(
        indices: wp.array[wp.int32],
        vertex_offsets: wp.array[wp.int32],
        incident_corners: wp.array[wp.int32],
        raw_stats: wp.array[wp.int32],
    ):
        # One thread per vertex. The thread exclusively owns its CSR slice
        # ``incident_corners[beg:end]`` and may read and later reorder it without
        # synchronizing with other threads.
        v = wp.tid()
        beg = vertex_offsets[v]
        end = vertex_offsets[v + 1]

        if beg == end:
            wp.atomic_add(raw_stats, _STAT_NUM_UNREFERENCED_VERTICES, 1)
            return

        # --- Edge statistics -------------------------------------------------
        # Own each undirected edge at its smaller endpoint, so only consider a
        # neighbor ``n`` with ``v < n``. Process each owned neighbor once, on its
        # first appearance in the CSR slice, then scan the whole slice to total
        # its incidence and signed orientation.
        for a in range(beg, end):
            _, prev_a, next_a = _TopologyStatistics._corner_neighbors(indices, incident_corners[a])
            for role in range(2):
                if role == 0:
                    n = next_a
                else:
                    n = prev_a
                if v >= n:
                    continue

                # First-occurrence check: skip if ``n`` already appeared in an
                # earlier corner of this slice (as either neighbor).
                seen = int(0)
                for b in range(beg, a):
                    _, prev_b, next_b = _TopologyStatistics._corner_neighbors(indices, incident_corners[b])
                    if prev_b == n or next_b == n:
                        seen = 1
                        break
                if seen != 0:
                    continue

                count = int(0)
                signed_count = int(0)
                for b in range(beg, end):
                    _, prev_b, next_b = _TopologyStatistics._corner_neighbors(indices, incident_corners[b])
                    if next_b == n:  # half-edge v -> n
                        count += 1
                        signed_count += 1
                    if prev_b == n:  # half-edge n -> v
                        count += 1
                        signed_count -= 1

                wp.atomic_add(raw_stats, _STAT_NUM_EDGES, 1)
                if count == 1:
                    wp.atomic_add(raw_stats, _STAT_NUM_BOUNDARY_EDGES, 1)
                if count > 2:
                    wp.atomic_add(raw_stats, _STAT_NUM_NONMANIFOLD_EDGES, 1)
                if count == 2 and signed_count != 0:
                    wp.atomic_add(raw_stats, _STAT_NUM_MISORIENTED_EDGES, 1)

        # --- Vertex manifoldness --------------------------------------------
        # Partition the CSR slice in place into ``[reached | unseen]`` faces via a
        # BFS through shared edges at ``v``. If the whole slice is reached, the
        # incident faces form a single fan.
        head = beg
        tail = beg + 1
        while head < tail:
            current = incident_corners[head]
            scan = tail
            while scan < end:
                candidate = incident_corners[scan]
                if _TopologyStatistics._faces_adjacent_at_v(indices, current, candidate):
                    incident_corners[scan] = incident_corners[tail]
                    incident_corners[tail] = candidate
                    tail += 1
                    scan = tail
                else:
                    scan += 1
            head += 1

        if tail != end:
            wp.atomic_add(raw_stats, _STAT_NUM_NONMANIFOLD_VERTICES, 1)


def _resolve_num_points(indices: wp.array, num_points: int | None, device: DeviceLike) -> int:
    """Determine the vertex count of a mesh.

    Uses the explicit ``num_points`` when given, otherwise falls back to one plus
    the largest index, which requires a single host readback.
    """
    if num_points is not None:
        if num_points < 0:
            raise ValueError(f"`num_points` must be non-negative, but got {num_points}.")
        return num_points

    if indices.shape[0] == 0:
        return 0

    largest = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_TopologyStatistics._max_vertex_index, dim=indices.shape[0], inputs=[indices, largest], device=device)
    return int(largest.numpy()[0]) + 1


def triangle_mesh_topology_statistics(
    indices: wp.array[wp.int32],
    num_points: int | None = None,
    *,
    device: DeviceLike | None = None,
) -> TriangleMeshTopologyStatistics:
    """Compute combinatorial topology statistics for an oriented triangle mesh.

    The mesh is given by a flat ``int32`` array of oriented triangle vertex
    indices (three per triangle); no vertex positions are needed. For every
    oriented triangle ``(i, j, k)`` the three half-edges ``i -> j``, ``j -> k``,
    ``k -> i`` are folded onto canonical undirected edges ``(min, max)``, and for
    each unique edge the incidence ``count`` and ``signed_count`` (``+1`` per
    ``u -> v`` half-edge, ``-1`` per ``v -> u``) classify it:

    - ``count == 1``: boundary edge.
    - ``count == 2``: ordinary interior edge.
    - ``count > 2``: non-manifold edge.
    - ``count == 2`` and ``signed_count != 0``: misoriented edge.

    A vertex is vertex-manifold when its incident faces form a single fan
    connected through edges containing that vertex; this is independent of edge
    manifoldness, so a vertex on a non-manifold edge can still be vertex-manifold.
    Unreferenced vertices are reported and treated as non-manifold.

    A triangle with a repeated vertex index is *combinatorially degenerate*: it is
    counted in ``num_degenerate_triangles`` and excluded from all edge and vertex
    statistics. Any degenerate triangle makes the manifold predicates ``False``.

    Args:
        indices: A 1-D :class:`warp.array` of ``int32`` triangle vertex indices,
            with length a multiple of three. Each consecutive triple is one
            oriented triangle.
        num_points: Number of vertices. If ``None``, inferred as one plus the
            largest index, which requires a host readback.
        device: Device on which to run. Defaults to the device of ``indices``.

    Returns:
        A :class:`TriangleMeshTopologyStatistics` with all scalar counts and
        derived manifoldness/orientation predicates.

    Raises:
        ValueError: If ``indices`` is not 1-D ``int32``, its length is not a
            multiple of three, or any index is outside ``[0, num_points)``.
    """
    if indices.ndim != 1:
        raise ValueError(f"`indices` must be a 1-D array, but got {indices.ndim} dimensions.")
    if indices.dtype != wp.int32:
        raise ValueError("`indices` must have dtype wp.int32.")
    if indices.shape[0] % 3 != 0:
        raise ValueError(f"`indices` length must be a multiple of 3, but got {indices.shape[0]}.")

    device = wp.get_device(device) if device is not None else indices.device
    if indices.device != device:
        indices = indices.to(device)

    num_triangles = indices.shape[0] // 3
    num_vertices = _resolve_num_points(indices, num_points, device)

    # Validate index bounds up front so out-of-range indices cannot corrupt the
    # counting/scatter passes (a single host readback).
    if indices.shape[0] > 0:
        bad = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            _TopologyStatistics._check_bounds,
            dim=indices.shape[0],
            inputs=[indices, num_vertices, bad],
            device=device,
        )
        if int(bad.numpy()[0]) != 0:
            raise ValueError(f"`indices` contains values outside the range [0, {num_vertices}).")

    raw_stats = wp.zeros(_STAT_COUNT, dtype=wp.int32, device=device)

    if num_vertices > 0:
        # Count -> scan -> scatter builds the vertex->incident-corner CSR without
        # a global key sort; the per-vertex analysis then reads (and reorders) it.
        vertex_offsets = wp.zeros(num_vertices + 1, dtype=wp.int32, device=device)
        incident_corners = wp.empty(3 * num_triangles, dtype=wp.int32, device=device)
        if num_triangles > 0:
            wp.launch(
                _TopologyStatistics._count_incident_corners,
                dim=num_triangles,
                inputs=[indices, vertex_offsets, raw_stats],
                device=device,
            )
            array_scan(vertex_offsets, vertex_offsets, inclusive=True)
            vertex_cursors = wp.zeros(num_vertices, dtype=wp.int32, device=device)
            wp.launch(
                _TopologyStatistics._scatter_incident_corners,
                dim=num_triangles,
                inputs=[indices, vertex_offsets, vertex_cursors, incident_corners],
                device=device,
            )
        # Runs even with no triangles, so every vertex is reported unreferenced.
        wp.launch(
            _TopologyStatistics._analyze_vertices,
            dim=num_vertices,
            inputs=[indices, vertex_offsets, incident_corners, raw_stats],
            device=device,
        )

    stats = raw_stats.numpy()
    num_degenerate = int(stats[_STAT_DEGENERATE_TRIANGLES])

    return TriangleMeshTopologyStatistics(
        num_vertices=num_vertices,
        num_triangles=num_triangles,
        num_edges=int(stats[_STAT_NUM_EDGES]),
        num_boundary_edges=int(stats[_STAT_NUM_BOUNDARY_EDGES]),
        num_nonmanifold_edges=int(stats[_STAT_NUM_NONMANIFOLD_EDGES]),
        num_misoriented_edges=int(stats[_STAT_NUM_MISORIENTED_EDGES]),
        num_nonmanifold_vertices=int(stats[_STAT_NUM_NONMANIFOLD_VERTICES]),
        num_unreferenced_vertices=int(stats[_STAT_NUM_UNREFERENCED_VERTICES]),
        num_degenerate_triangles=num_degenerate,
    )
