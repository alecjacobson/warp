# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

The public entry points operate on a 2-D ``int32`` index array (one simplex per
row) and require no vertex positions:

* :func:`triangle_mesh_topology_statistics` summarizes combinatorial topology of a
  triangle mesh -- edge incidence and orientation, vertex manifoldness, and
  degeneracies -- with a sort-free count/scan/scatter pipeline that builds a
  vertex-to-incident-corner CSR followed by a per-vertex analysis pass. Only a
  single small counter array is read back to assemble the returned
  :class:`TriangleMeshTopologyStatistics`.
* :func:`connected_components` labels the connected components of a simplicial mesh
  (segments, triangles, tetrahedra, ...) with a parallel union-find
  (simplex-parallel hooking plus vertex-parallel full path compression, iterated
  on-device to a fixpoint).

Kernels and helpers specific to each routine are grouped in the private
:class:`_TopologyStatistics` and :class:`_ConnectedComponents` classes so their
names stay tied to their algorithm and do not clutter the module namespace.
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
    "connected_components",
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
    def _corner_neighbors(indices: wp.array2d[wp.int32], corner: wp.int32):
        """Return ``(v, prev, next)`` for an incident corner id ``3 * tri + local``.

        ``v`` is the corner's vertex; in the triangle's orientation the boundary
        traverses ``prev -> v -> next``. All three are distinct because degenerate
        triangles are excluded from the CSR.
        """
        t = corner // 3
        c = corner % 3
        v = indices[t, c]
        prv = indices[t, (c + 2) % 3]
        nxt = indices[t, (c + 1) % 3]
        return v, prv, nxt

    @wp.func
    def _faces_adjacent_at_v(indices: wp.array2d[wp.int32], corner_a: wp.int32, corner_b: wp.int32) -> bool:
        """Return whether two faces incident to a shared vertex also share a mesh edge.

        Two triangles around vertex ``v`` are adjacent iff they share an edge
        containing ``v``, i.e. one of ``v``'s two neighbors in face ``A`` matches
        one of its neighbors in face ``B``.
        """
        _, pa, na = _TopologyStatistics._corner_neighbors(indices, corner_a)
        _, pb, nb = _TopologyStatistics._corner_neighbors(indices, corner_b)
        return pa == pb or pa == nb or na == pb or na == nb

    @wp.kernel(enable_backward=False)
    def _count_incident_corners(
        indices: wp.array2d[wp.int32],
        vertex_offsets: wp.array[wp.int32],
        raw_stats: wp.array[wp.int32],
    ):
        # One thread per triangle. Bucket each corner under its vertex (offset by
        # one so an inclusive scan turns the counts into CSR offsets), and set
        # aside combinatorially degenerate triangles.
        t = wp.tid()
        i = indices[t, 0]
        j = indices[t, 1]
        k = indices[t, 2]
        if i == j or j == k or k == i:
            wp.atomic_add(raw_stats, _STAT_DEGENERATE_TRIANGLES, 1)
            return
        wp.atomic_add(vertex_offsets, i + 1, 1)
        wp.atomic_add(vertex_offsets, j + 1, 1)
        wp.atomic_add(vertex_offsets, k + 1, 1)

    @wp.kernel(enable_backward=False)
    def _scatter_incident_corners(
        indices: wp.array2d[wp.int32],
        vertex_offsets: wp.array[wp.int32],
        vertex_cursors: wp.array[wp.int32],
        incident_corners: wp.array[wp.int32],
    ):
        # One thread per triangle. Scatter each corner id into its vertex's CSR
        # range; ordering within a range is unspecified.
        t = wp.tid()
        i = indices[t, 0]
        j = indices[t, 1]
        k = indices[t, 2]
        if i == j or j == k or k == i:
            return
        for c in range(3):
            v = indices[t, c]
            slot = vertex_offsets[v] + wp.atomic_add(vertex_cursors, v, 1)
            incident_corners[slot] = 3 * t + c

    @wp.kernel(enable_backward=False)
    def _analyze_vertices(
        indices: wp.array2d[wp.int32],
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


# ---------------------------------------------------------------------------
# Shared simplex-index validation
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _max_vertex_index(indices: wp.array2d[wp.int32], out_max: wp.array[wp.int32]):
    i, j = wp.tid()
    wp.atomic_max(out_max, 0, indices[i, j])


@wp.kernel(enable_backward=False)
def _check_index_bounds(indices: wp.array2d[wp.int32], num_points: wp.int32, bad: wp.array[wp.int32]):
    i, j = wp.tid()
    x = indices[i, j]
    if x < 0 or x >= num_points:
        wp.atomic_max(bad, 0, 1)


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
    wp.launch(_max_vertex_index, dim=indices.shape, inputs=[indices, largest], device=device)
    return int(largest.numpy()[0]) + 1


def _prepare_indices(indices: wp.array, num_points: int | None, device: DeviceLike | None):
    """Validate a ``(num_simplices, simplex_size)`` index array and resolve its vertex count.

    Each row is one simplex. Returns ``(indices, device, num_vertices)`` with
    ``indices`` moved to the resolved device. Structural checks (rank, dtype,
    non-empty simplex size) are always applied. Vertex-count inference and the
    index-bounds check each need a host readback, so they are skipped while the
    device is capturing a CUDA graph; there, ``num_points`` is required and the
    caller is responsible for valid indices.
    """
    if indices.ndim != 2:
        raise ValueError(f"`indices` must be a 2-D (num_simplices, simplex_size) array, but got {indices.ndim} dims.")
    if indices.dtype != wp.int32:
        raise ValueError("`indices` must have dtype wp.int32.")
    if indices.shape[1] < 1:
        raise ValueError(f"`indices` must have at least one column (simplex size), but got shape {indices.shape}.")

    device = wp.get_device(device) if device is not None else indices.device
    if indices.device != device:
        indices = indices.to(device)

    if device.is_capturing:
        if num_points is None:
            raise ValueError("`num_points` must be provided while capturing a CUDA graph (inference needs a readback).")
        if num_points < 0:
            raise ValueError(f"`num_points` must be non-negative, but got {num_points}.")
        return indices, device, num_points

    num_vertices = _resolve_num_points(indices, num_points, device)

    if indices.shape[0] > 0:
        bad = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_check_index_bounds, dim=indices.shape, inputs=[indices, num_vertices, bad], device=device)
        if int(bad.numpy()[0]) != 0:
            raise ValueError(f"`indices` contains values outside the range [0, {num_vertices}).")

    return indices, device, num_vertices


def triangle_mesh_topology_statistics(
    indices: wp.array2d[wp.int32],
    num_points: int | None = None,
    *,
    device: DeviceLike | None = None,
) -> TriangleMeshTopologyStatistics:
    """Compute combinatorial topology statistics for an oriented triangle mesh.

    The mesh is given by a ``(num_triangles, 3)`` ``int32`` array of oriented
    triangle vertex indices; no vertex positions are needed. For every oriented
    triangle ``(i, j, k)`` the three half-edges ``i -> j``, ``j -> k``,
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
        indices: A ``(num_triangles, 3)`` :class:`warp.array` of ``int32``
            oriented triangle vertex indices.
        num_points: Number of vertices. If ``None``, inferred as one plus the
            largest index, which requires a host readback.
        device: Device on which to run. Defaults to the device of ``indices``.

    Returns:
        A :class:`TriangleMeshTopologyStatistics` with all scalar counts and
        derived manifoldness/orientation predicates.

    Raises:
        ValueError: If ``indices`` is not a 2-D ``int32`` array with three
            columns, or any index is outside ``[0, num_points)``.
    """
    # Validate up front so out-of-range indices cannot corrupt the
    # counting/scatter passes (a single host readback).
    if indices.ndim == 2 and indices.shape[1] != 3:
        raise ValueError(f"`indices` must have shape (num_triangles, 3), but got {indices.shape}.")
    indices, device, num_vertices = _prepare_indices(indices, num_points, device)
    num_triangles = indices.shape[0]

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


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

# Backstop on the number of hook/compress rounds. Full path compression converges
# in ~2 rounds regardless of graph diameter, so this ceiling only guards against a
# would-be infinite loop (and bounds the device-driven capture loop); it is never
# reached in practice.
_CC_MAX_ROUNDS = 1000


class _ConnectedComponents:
    """Kernels for :func:`connected_components`.

    ``labels`` is a union-find parent array: ``labels[v]`` points to another
    vertex in ``v``'s component, always with a smaller-or-equal id, so the forest
    is acyclic and each component's root is its minimum vertex id.
    """

    @wp.kernel(enable_backward=False)
    def _init_labels(labels: wp.array[wp.int32]):
        v = wp.tid()
        labels[v] = v

    @wp.kernel(enable_backward=False)
    def _hook(indices: wp.array2d[wp.int32], labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
        # One thread per simplex (row). Connecting every vertex of the simplex to
        # its first vertex (a star) is enough to put them in one component -- the
        # exact intra-simplex edge set is irrelevant for connected components. Each
        # hook attaches the endpoint with the larger current label under the smaller
        # via atomic-min; the outer round loop propagates across shared vertices.
        t = wp.tid()
        first = indices[t, 0]
        for c in range(1, indices.shape[1]):
            la = labels[first]
            lb = labels[indices[t, c]]
            if la != lb:
                lo = wp.min(la, lb)
                hi = wp.max(la, lb)
                old = wp.atomic_min(labels, hi, lo)
                if lo < old:
                    wp.atomic_max(changed, 0, 1)

    @wp.kernel(enable_backward=False)
    def _compress(labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
        # One thread per vertex. Full path compression: point straight at the
        # root. Chasing the whole chain each round (rather than a single
        # grandparent jump) empirically converges in ~2 rounds regardless of graph
        # diameter, where single-jump pointer jumping needs O(log diameter) rounds.
        # Concurrent updates only ever shorten paths toward the same root, so the
        # chase is safe without locking.
        v = wp.tid()
        r = labels[v]
        while labels[r] != r:
            r = labels[r]
        if r != labels[v]:
            labels[v] = r
            wp.atomic_max(changed, 0, 1)

    @wp.kernel(enable_backward=False)
    def _record(
        changed: wp.array[wp.int32],
        round_count: wp.array[wp.int32],
        max_rounds: wp.int32,
        condition: wp.array[wp.int32],
    ):
        # Single-thread bookkeeping for the device-driven (graph-capturable) loop:
        # continue while the last round made progress and the cap is not reached.
        round_count[0] += 1
        if changed[0] != 0 and round_count[0] < max_rounds:
            condition[0] = 1
        else:
            condition[0] = 0

    @wp.kernel(enable_backward=False)
    def _read_count(root_ids: wp.array[wp.int32], n: wp.int32, count: wp.array[wp.int32]):
        # The inclusive scan's last element is the number of components.
        count[0] = root_ids[n - 1]

    @wp.kernel(enable_backward=False)
    def _flag_roots(labels: wp.array[wp.int32], is_root: wp.array[wp.int32]):
        v = wp.tid()
        if labels[v] == v:
            is_root[v] = 1
        else:
            is_root[v] = 0

    @wp.kernel(enable_backward=False)
    def _relabel(labels: wp.array[wp.int32], root_ids: wp.array[wp.int32], out: wp.array[wp.int32]):
        # ``labels[v]`` is a root; ``root_ids`` is the inclusive scan of the root
        # flags, so the 0-based component id of root ``r`` is ``root_ids[r] - 1``.
        v = wp.tid()
        out[v] = root_ids[labels[v]] - 1


def connected_components(
    indices: wp.array2d[wp.int32],
    num_points: int | None = None,
    *,
    device: DeviceLike | None = None,
) -> tuple[wp.array, int | wp.array]:
    """Label the connected components of a simplicial mesh.

    The mesh is a ``(num_simplices, simplex_size)`` ``int32`` array, one simplex
    per row; the simplex size is read from the second dimension (``3`` triangles,
    ``2`` segments, ``4`` tetrahedra, and so on). Two vertices are in the same
    component when they are joined by a path through simplices that share a vertex;
    equivalently, all vertices of a simplex are mutually connected. This matches
    the edge-based connectivity of gptoolbox's ``connected_components`` and may
    span non-manifold edges or vertices. Vertices not referenced by any simplex --
    including a repeated vertex that shares its simplex with no other distinct
    vertex -- each form their own singleton component.

    The labeling runs on the Warp device (CPU or CUDA) as a parallel union-find:
    simplex-parallel hooking attaches the larger of two component representatives
    under the smaller (linking each simplex's vertices to its first), and
    vertex-parallel full path compression flattens the forest. The two passes are
    iterated to a fixpoint, at which every component collapses to a single root,
    and the roots are then renumbered to a contiguous range.

    Args:
        indices: A ``(num_simplices, simplex_size)`` :class:`warp.array` of
            ``int32`` vertex indices, one simplex per row.
        num_points: Number of vertices. If ``None``, inferred as one plus the
            largest index, which requires a host readback.
        device: Device on which to run. Defaults to the device of ``indices``.

    Returns:
        A tuple ``(labels, num_components)`` where ``labels`` is a
        ``(num_points,)`` ``int32`` :class:`warp.array` on ``device`` giving each
        vertex's component id in ``[0, num_components)``. In eager mode
        ``num_components`` is a Python ``int``. While a CUDA graph is being
        captured the count is not known until replay, so a single-element
        ``int32`` :class:`warp.array` is returned instead; read it after
        :func:`warp.capture_launch`.

    Raises:
        ValueError: If ``indices`` is not a 2-D ``int32`` array with at least one
            column, or (eager mode only) any index is outside ``[0, num_points)``.

    Note:
        The whole routine is CUDA-graph capturable: the convergence loop is
        driven on-device with :func:`warp.capture_while`. Because validation and
        vertex-count inference need a host readback, during capture ``num_points``
        is required and indices are assumed in range. A warm-up call before
        capture is recommended so scratch allocations are sized beforehand.
    """
    indices, device, num_vertices = _prepare_indices(indices, num_points, device)
    num_simplices = indices.shape[0]

    if num_vertices == 0:
        return wp.zeros(0, dtype=wp.int32, device=device), 0

    labels = wp.empty(num_vertices, dtype=wp.int32, device=device)
    wp.launch(_ConnectedComponents._init_labels, dim=num_vertices, inputs=[labels], device=device)

    if num_simplices > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)

        def _round():
            changed.zero_()
            wp.launch(_ConnectedComponents._hook, dim=num_simplices, inputs=[indices, labels, changed], device=device)
            wp.launch(_ConnectedComponents._compress, dim=num_vertices, inputs=[labels, changed], device=device)

        if device.is_capturing:
            # Device-driven loop: a bookkeeping kernel updates the loop condition
            # so no per-round host readback is needed under graph capture.
            round_count = wp.zeros(1, dtype=wp.int32, device=device)
            condition = wp.ones(1, dtype=wp.int32, device=device)  # run the body at least once

            def _body():
                _round()
                wp.launch(
                    _ConnectedComponents._record,
                    dim=1,
                    inputs=[changed, round_count, wp.int32(_CC_MAX_ROUNDS), condition],
                    device=device,
                )

            wp.capture_while(condition, _body)
        else:
            rounds = 0
            while True:
                _round()
                rounds += 1
                if int(changed.numpy()[0]) == 0:
                    break
                if rounds >= _CC_MAX_ROUNDS:
                    raise RuntimeError(
                        f"connected_components did not converge within {_CC_MAX_ROUNDS} rounds; this is a bug."
                    )

    # Renumber the (now fully flattened) roots to a contiguous [0, num_components).
    is_root = wp.empty(num_vertices, dtype=wp.int32, device=device)
    wp.launch(_ConnectedComponents._flag_roots, dim=num_vertices, inputs=[labels, is_root], device=device)
    root_ids = wp.empty(num_vertices, dtype=wp.int32, device=device)
    array_scan(is_root, root_ids, inclusive=True)

    out = wp.empty(num_vertices, dtype=wp.int32, device=device)
    wp.launch(_ConnectedComponents._relabel, dim=num_vertices, inputs=[labels, root_ids, out], device=device)

    if device.is_capturing:
        num_components = wp.empty(1, dtype=wp.int32, device=device)
        wp.launch(
            _ConnectedComponents._read_count,
            dim=1,
            inputs=[root_ids, wp.int32(num_vertices), num_components],
            device=device,
        )
        return out, num_components

    return out, int(root_ids.numpy()[num_vertices - 1])
