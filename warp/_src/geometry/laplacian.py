# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laplacian and vertex-adjacency operators for triangle meshes.

The public entry points are :func:`laplacian`, which assembles the cotangent or
uniform mesh Laplacian as a sparse matrix, and :func:`vertex_adjacency_matrix`.
The device functions and kernels above them are the reusable building blocks and
are not part of the public API.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import warp as wp
from warp._src.sparse import (
    BsrMatrix,
    bsr_compress,
    bsr_from_triplets,
    bsr_set_from_triplets,
    bsr_set_zero,
    bsr_zeros,
)
from warp._src.types import type_repr, types_equal, vector

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


# Fixed-length vector types used to carry the six per-edge weights of a
# tetrahedron and the local endpoints of those edges.
vec6f = vector(length=6, dtype=wp.float32)
vec6i = vector(length=6, dtype=wp.int32)
vec12i = vector(length=12, dtype=wp.int32)

# The six edges of a tetrahedron in a canonical order, given as the local
# vertex indices (0..3) of each edge's two endpoints. Edge ``k`` joins local
# vertices ``_TET_EDGE_A[k]`` and ``_TET_EDGE_B[k]``.
_TET_EDGE_A = wp.constant(vec6i(0, 0, 0, 1, 1, 2))
_TET_EDGE_B = wp.constant(vec6i(1, 2, 3, 2, 3, 3))

# Row-grouped view of the same six edges, for the row-compressed path. For each
# of the four corners ``c`` (in blocks of three), ``_TET_CORNER_NBR`` lists the
# local indices of its three neighbors and ``_TET_CORNER_EDGE`` the index into
# the six edge weights of the edge joining ``c`` to that neighbor.
_TET_CORNER_NBR = wp.constant(vec12i(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2))
_TET_CORNER_EDGE = wp.constant(vec12i(0, 1, 2, 0, 3, 4, 1, 3, 5, 2, 4, 5))


##########################################################################
## Device functions that operate on a single triangle (reusable within
## kernels). These are the building blocks for the assembly kernels below.
##########################################################################
@wp.func
def triangle_edge_length_sq(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.length_sq(v2 - v1), wp.length_sq(v0 - v2), wp.length_sq(v1 - v0))


@wp.func
def triangle_cotangent_weights(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    # Half the cotangent of each corner angle, indexed so that entry ``k`` weights
    # the edge opposite vertex ``k``. Positive for acute angles; the Laplacian's
    # sign convention is applied where these are assembled, not here.
    l2 = triangle_edge_length_sq(v0, v1, v2)
    A8 = triangle_double_area(v0, v1, v2) * 4.0

    return wp.vec3(l2[1] + l2[2] - l2[0], l2[2] + l2[0] - l2[1], l2[0] + l2[1] - l2[2]) / A8


@wp.func
def triangle_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, normalized: bool = False) -> wp.vec3:
    n = wp.cross(v1 - v0, v2 - v0)
    if normalized:
        n = wp.normalize(n)
    return n


@wp.func
def triangle_double_area(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.float32:
    n = triangle_normal(v0, v1, v2, normalized=False)
    return wp.length(n)


@wp.func
def tet_edge_weights(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> vec6f:
    # The six cotangent edge weights of a linear tetrahedron, one per edge in
    # ``_TET_EDGE_A``/``_TET_EDGE_B`` order. Each is the negated off-diagonal of
    # the P1 stiffness matrix, ``-vol * grad(phi_i) . grad(phi_j)``, where the
    # barycentric basis gradients are the rows of the inverse edge Jacobian and
    # the fourth gradient closes the partition of unity. Positive for a
    # well-shaped element.
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    m = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    vol = wp.abs(wp.determinant(m)) / 6.0
    minv = wp.inverse(m)

    g1 = wp.vec3(minv[0, 0], minv[0, 1], minv[0, 2])
    g2 = wp.vec3(minv[1, 0], minv[1, 1], minv[1, 2])
    g3 = wp.vec3(minv[2, 0], minv[2, 1], minv[2, 2])
    g0 = -(g1 + g2 + g3)

    return vec6f(
        -vol * wp.dot(g0, g1),
        -vol * wp.dot(g0, g2),
        -vol * wp.dot(g0, g3),
        -vol * wp.dot(g1, g2),
        -vol * wp.dot(g1, g3),
        -vol * wp.dot(g2, g3),
    )


class LaplacianWeighting(enum.IntEnum):
    """Edge weighting used to assemble a mesh Laplacian.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel launch (Warp kernels cannot take a ``str`` parameter).
    """

    COTANGENT = 0
    """Weight each edge by half the sum of the cotangents of the angles opposite it.

    This is the P1 finite-element Laplacian, which depends on vertex positions.
    """
    UNIFORM = 1
    """Weight every edge equally.

    This is the graph Laplacian ``D - A`` of the mesh's edge graph, which depends
    only on connectivity. Each edge counts once however many triangles share it.
    """


@wp.kernel
def laplacian_triplets(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    rows: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    tri = wp.tid()
    base = 9 * tri

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    c = triangle_cotangent_weights(
        points[v[0]],
        points[v[1]],
        points[v[2]],
    )

    # For each vertex k:
    #   - c[k] weights the edge opposite k
    #   - emit both symmetric off-diagonal entries
    #   - emit the diagonal entry for vertex k
    # Off-diagonals are negated and the diagonal is positive, which is what makes
    # the assembled operator positive semi-definite.
    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        out = base + 3 * k

        rows[out + 0] = v[i]
        columns[out + 0] = v[j]
        values[out + 0] = -c[k]

        rows[out + 1] = v[j]
        columns[out + 1] = v[i]
        values[out + 1] = -c[k]

        rows[out + 2] = v[k]
        columns[out + 2] = v[k]
        values[out + 2] = c[i] + c[j]


@wp.kernel
def tet_laplacian_triplets(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    rows: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    # The P1 finite-element stiffness matrix of a linear tetrahedron. Its
    # element matrix is ``vol * grad(phi_i) . grad(phi_j)``, where the barycentric
    # basis gradients are the rows of the inverse of the edge Jacobian. The
    # off-diagonal weight of edge ``(i, j)`` is the tetrahedral cotangent weight,
    # so this is the exact analogue of ``laplacian_triplets`` for triangles.
    tet = wp.tid()
    base = 24 * tet

    i0 = indices[4 * tet + 0]
    i1 = indices[4 * tet + 1]
    i2 = indices[4 * tet + 2]
    i3 = indices[4 * tet + 3]
    ia = wp.vec4i(i0, i1, i2, i3)

    w = tet_edge_weights(points[i0], points[i1], points[i2], points[i3])

    ea = _TET_EDGE_A
    eb = _TET_EDGE_B
    for k in range(6):
        a = ea[k]
        b = eb[k]
        wt = w[k]
        ra = ia[a]
        rb = ia[b]
        out = base + 4 * k

        # Symmetric off-diagonal pair, then the diagonal contribution each
        # endpoint accrues from this edge. Duplicate diagonal entries from a
        # vertex's other incident edges are summed during assembly.
        rows[out + 0] = ra
        columns[out + 0] = rb
        values[out + 0] = -wt

        rows[out + 1] = rb
        columns[out + 1] = ra
        values[out + 1] = -wt

        rows[out + 2] = ra
        columns[out + 2] = ra
        values[out + 2] = wt

        rows[out + 3] = rb
        columns[out + 3] = rb
        values[out + 3] = wt


@wp.kernel
def vertex_row_counts(indices: wp.array[int], simplex_size: int, entries_per_incidence: int, counts: wp.array[int]):
    # Reserved storage for row ``v`` is a fixed number of entries per incident
    # element: one off-diagonal to each of the element's other vertices, plus a
    # diagonal entry when the caller wants one. Counting incidences needs no edge
    # enumeration and makes no manifoldness assumption.
    elem = wp.tid()
    for k in range(simplex_size):
        wp.atomic_add(counts, indices[simplex_size * elem + k], entries_per_incidence)


@wp.kernel(enable_backward=False)
def laplacian_row_entries(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    offsets: wp.array[int],
    cursors: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    tri = wp.tid()

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    c = triangle_cotangent_weights(
        points[v[0]],
        points[v[1]],
        points[v[2]],
    )

    # Same contributions as ``laplacian_triplets``, but grouped by the row they
    # land in so that they can be written straight into that row's reserved
    # span. ``cursors`` starts at zero and doubles as the per-row write cursor
    # and the final active count for each row.
    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        row = v[k]
        base = offsets[row] + wp.atomic_add(cursors, row, 3)

        columns[base + 0] = row
        values[base + 0] = c[i] + c[j]

        columns[base + 1] = v[j]
        values[base + 1] = -c[i]

        columns[base + 2] = v[i]
        values[base + 2] = -c[j]


@wp.kernel(enable_backward=False)
def laplacian_row_slots(
    indices: wp.array[int],
    offsets: wp.array[int],
    cursors: wp.array[int],
    columns: wp.array[int],
    entry_slots: wp.array[int],
):
    # The topology half of a differentiable row-compressed assembly. It reserves
    # each triangle's three-entry span per corner exactly like
    # ``laplacian_row_entries``, writing the columns and recording, for each
    # contribution, the slot it landed in. The racy write cursor lives here, in a
    # kernel that touches only integers, so the value pass that follows can be
    # replayed by the backward pass without depending on atomic ordering.
    tri = wp.tid()

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        row = v[k]
        base = offsets[row] + wp.atomic_add(cursors, row, 3)

        out = 9 * tri + 3 * k
        columns[base + 0] = row
        entry_slots[out + 0] = base + 0

        columns[base + 1] = v[j]
        entry_slots[out + 1] = base + 1

        columns[base + 2] = v[i]
        entry_slots[out + 2] = base + 2


@wp.kernel
def laplacian_row_values(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    entry_slots: wp.array[int],
    values: wp.array[float],
):
    # The value half of a differentiable row-compressed assembly. Each
    # contribution is written to the fixed slot chosen by ``laplacian_row_slots``,
    # so this pass carries the gradient with respect to ``points`` while the
    # padded layout it fills matches ``laplacian_row_entries`` entry for entry.
    tri = wp.tid()

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    c = triangle_cotangent_weights(
        points[v[0]],
        points[v[1]],
        points[v[2]],
    )

    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        out = 9 * tri + 3 * k

        values[entry_slots[out + 0]] = c[i] + c[j]
        values[entry_slots[out + 1]] = -c[i]
        values[entry_slots[out + 2]] = -c[j]


@wp.kernel(enable_backward=False)
def tet_laplacian_row_entries(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    offsets: wp.array[int],
    cursors: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    # The row-grouped tetrahedral assembly: for each of a tet's four vertices it
    # writes the diagonal entry (the sum of that vertex's three incident edge
    # weights) and the three off-diagonals into the row's reserved span, four
    # entries per corner. Groups the contributions by row the way
    # ``laplacian_row_entries`` does for triangles.
    tet = wp.tid()
    iv = wp.vec4i(
        indices[4 * tet + 0],
        indices[4 * tet + 1],
        indices[4 * tet + 2],
        indices[4 * tet + 3],
    )

    w = tet_edge_weights(points[iv[0]], points[iv[1]], points[iv[2]], points[iv[3]])

    nbr = _TET_CORNER_NBR
    edg = _TET_CORNER_EDGE
    for c in range(4):
        row = iv[c]
        base = offsets[row] + wp.atomic_add(cursors, row, 4)

        diagonal = float(0.0)
        for m in range(3):
            weight = w[edg[3 * c + m]]
            columns[base + 1 + m] = iv[nbr[3 * c + m]]
            values[base + 1 + m] = -weight
            diagonal += weight

        columns[base + 0] = row
        values[base + 0] = diagonal


@wp.kernel(enable_backward=False)
def tet_laplacian_row_slots(
    indices: wp.array[int],
    offsets: wp.array[int],
    cursors: wp.array[int],
    columns: wp.array[int],
    entry_slots: wp.array[int],
):
    # Topology half of the differentiable tetrahedral row-compressed assembly,
    # the analogue of ``laplacian_row_slots``: it fixes the slot layout and writes
    # the columns, touching only integers, so the value pass can be replayed.
    tet = wp.tid()
    iv = wp.vec4i(
        indices[4 * tet + 0],
        indices[4 * tet + 1],
        indices[4 * tet + 2],
        indices[4 * tet + 3],
    )

    nbr = _TET_CORNER_NBR
    for c in range(4):
        row = iv[c]
        base = offsets[row] + wp.atomic_add(cursors, row, 4)
        out = 16 * tet + 4 * c

        columns[base + 0] = row
        entry_slots[out + 0] = base + 0
        for m in range(3):
            columns[base + 1 + m] = iv[nbr[3 * c + m]]
            entry_slots[out + 1 + m] = base + 1 + m


@wp.kernel
def tet_laplacian_row_values(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    entry_slots: wp.array[int],
    values: wp.array[float],
):
    # Value half of the differentiable tetrahedral row-compressed assembly, the
    # analogue of ``laplacian_row_values``: it writes each contribution into the
    # fixed slot chosen by ``tet_laplacian_row_slots`` and carries the gradient.
    tet = wp.tid()
    iv = wp.vec4i(
        indices[4 * tet + 0],
        indices[4 * tet + 1],
        indices[4 * tet + 2],
        indices[4 * tet + 3],
    )

    w = tet_edge_weights(points[iv[0]], points[iv[1]], points[iv[2]], points[iv[3]])

    edg = _TET_CORNER_EDGE
    for c in range(4):
        out = 16 * tet + 4 * c
        diagonal = float(0.0)
        for m in range(3):
            weight = w[edg[3 * c + m]]
            values[entry_slots[out + 1 + m]] = -weight
            diagonal += weight
        values[entry_slots[out + 0]] = diagonal


##########################################################################
## Connectivity-only assembly. These kernels build the sparsity pattern
## coupling vertices that share a triangle edge, and fill it from the
## pattern itself. They read no positions and write only integers or
## constants, so their adjoints are empty and they stay usable inside a
## warp.Tape without disabling backward passes.
##########################################################################


@wp.kernel
def connectivity_triplets(
    indices: wp.array[int],
    rows: wp.array[int],
    columns: wp.array[int],
):
    # Six entries per triangle: both directions of each of its three edges. Only
    # the pattern is emitted, because a uniform weight cannot be accumulated per
    # triangle -- an interior edge would then be counted once per incident
    # triangle. The values are filled from the deduplicated pattern instead.
    tri = wp.tid()
    base = 6 * tri

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        out = base + 2 * k

        rows[out + 0] = v[i]
        columns[out + 0] = v[j]

        rows[out + 1] = v[j]
        columns[out + 1] = v[i]


@wp.kernel
def connectivity_diagonal_triplets(
    indices: wp.array[int],
    offset: int,
    rows: wp.array[int],
    columns: wp.array[int],
):
    # Three diagonal entries per triangle, written past the off-diagonal block
    # emitted by ``connectivity_triplets``. Emitting them per triangle rather
    # than per vertex keeps the pattern identical to the cotangent one, which
    # also leaves a vertex belonging to no triangle out of the matrix.
    tri = wp.tid()
    for k in range(3):
        out = offset + 3 * tri + k
        v = indices[3 * tri + k]
        rows[out] = v
        columns[out] = v


@wp.kernel
def connectivity_row_entries(
    indices: wp.array[int],
    include_diagonal: int,
    offsets: wp.array[int],
    cursors: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    # Same entries as ``connectivity_triplets``, grouped by the row they land in
    # so they can be written straight into that row's reserved span. The values
    # are placeholders that the caller's fill kernel overwrites.
    tri = wp.tid()

    v = wp.vec3i(
        indices[3 * tri],
        indices[3 * tri + 1],
        indices[3 * tri + 2],
    )

    stride = 2 + include_diagonal

    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        row = v[k]
        base = offsets[row] + wp.atomic_add(cursors, row, stride)

        columns[base + 0] = v[i]
        values[base + 0] = 1.0

        columns[base + 1] = v[j]
        values[base + 1] = 1.0

        if include_diagonal != 0:
            columns[base + 2] = row
            values[base + 2] = 1.0


@wp.kernel
def uniform_laplacian_values(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    # ``D - A`` read off the deduplicated pattern: every off-diagonal is -1 and
    # the diagonal is the number of distinct neighbors. Taking the degree from
    # the pattern rather than from triangle incidences is what makes an interior
    # edge count once rather than twice.
    row = wp.tid()

    diagonal = int(-1)
    degree = float(0.0)

    for k in range(offsets[row], offsets[row + 1]):
        if columns[k] == row:
            diagonal = k
        else:
            values[k] = -1.0
            degree += 1.0

    if diagonal != -1:
        values[diagonal] = degree


@wp.kernel
def adjacency_values(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
):
    # One for every vertex pair in the pattern. A diagonal entry can only be
    # present when the caller supplied a pattern that has one, and a vertex is
    # not adjacent to itself, so it is zeroed rather than set.
    row = wp.tid()
    for k in range(offsets[row], offsets[row + 1]):
        values[k] = wp.where(columns[k] == row, 0.0, 1.0)


@wp.kernel
def max_vertex_index(indices: wp.array[int], out_max: wp.array[int]):
    wp.atomic_max(out_max, 0, indices[wp.tid()])


def _resolve_simplex_indices(indices: wp.array, simplex_size: int | None) -> tuple[wp.array, int]:
    """Normalize a mesh's index array to a flat ``int32`` array and a simplex size.

    Indices may be given either as a flat :class:`warp.int32` array of consecutive
    per-simplex entries, or as a typed array of :class:`warp.vec3i` (triangles) or
    :class:`warp.vec4i` (tetrahedra). Typed arrays fix the simplex size from their
    element type and are reinterpreted as a flat ``int32`` array without copying.
    """
    dtype = indices.dtype

    if types_equal(dtype, wp.vec3i):
        inferred = 3
    elif types_equal(dtype, wp.vec4i):
        inferred = 4
    else:
        inferred = None

    if inferred is not None:
        if simplex_size is not None and simplex_size != inferred:
            raise ValueError(
                f"`simplex_size` is {simplex_size} but `indices` has element type {type_repr(dtype)}, which "
                f"fixes the simplex size at {inferred}. Pass a flat `int32` array to choose the size explicitly."
            )
        return indices.view(wp.int32).flatten(), inferred

    if not types_equal(dtype, wp.int32):
        raise ValueError(
            f"`indices` must be a flat array of {type_repr(wp.int32)}, or an array of "
            f"{type_repr(wp.vec3i)} or {type_repr(wp.vec4i)}, but got {type_repr(dtype)}."
        )

    if simplex_size is None:
        simplex_size = 3
    if simplex_size not in (3, 4):
        raise ValueError(f"`simplex_size` must be 3 (triangles) or 4 (tetrahedra), but got {simplex_size}.")
    return indices, simplex_size


def _validate_output_matrix(out: BsrMatrix, name: str, shape: tuple[int, int], device: DeviceLike) -> None:
    """Check that a caller-supplied output matrix matches the expected block type, device, and shape."""
    if not types_equal(out.scalar_type, wp.float32):
        raise ValueError(
            f"`{name}` must have scalar type {type_repr(wp.float32)}, but got {type_repr(out.scalar_type)}."
        )
    if out.block_shape != (1, 1):
        raise ValueError(f"`{name}` must have scalar (1x1) blocks, but got blocks of shape {out.block_shape}.")
    if out.device != device:
        raise ValueError(f"`{name}` must be on device '{device}', but got '{out.device}'.")
    if out.shape != shape:
        raise ValueError(f"`{name}` must have shape {shape}, but got {out.shape}.")


def _resolve_num_points(
    points: wp.array | None,
    indices: wp.array,
    num_points: int | None,
    out_matrix: BsrMatrix | None,
    device: DeviceLike,
) -> int:
    """Determine the vertex count of a mesh, which fixes the size of its matrices.

    Sources are tried in order of decreasing certainty: the positions array, an
    explicit count, the shape of a caller-supplied output matrix, and finally the
    largest index in ``indices``. Only the last requires a device readback.
    """
    if points is not None:
        if num_points is not None and num_points != points.shape[0]:
            raise ValueError(
                f"`num_points` is {num_points} but `points` holds {points.shape[0]} positions. Pass only one of them."
            )
        return points.shape[0]

    if num_points is not None:
        if num_points < 0:
            raise ValueError(f"`num_points` must be non-negative, but got {num_points}.")
        return num_points

    if out_matrix is not None:
        return out_matrix.shape[0]

    if indices.shape[0] == 0:
        return 0

    largest = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(max_vertex_index, dim=indices.shape[0], inputs=[indices], outputs=[largest], device=device)
    return int(largest.numpy()[0]) + 1


def _connectivity_matrix(
    indices: wp.array[int],
    num_points: int,
    out_matrix: BsrMatrix | None,
    construction: str,
    reuse_topology: bool,
    include_diagonal: bool,
    device: DeviceLike,
) -> BsrMatrix:
    """Build the sparsity pattern coupling every pair of vertices that share a triangle edge.

    The values of the returned matrix are meaningless; the caller fills them from
    the pattern. Mirrors the two construction policies of the cotangent path.
    """
    if reuse_topology:
        return out_matrix

    num_triangles = indices.shape[0] // 3
    entries_per_incidence = 3 if include_diagonal else 2

    if construction == "row_compress":
        nnz = num_triangles * 3 * entries_per_incidence
        counts = wp.zeros(num_points, dtype=wp.int32, device=device)
        wp.launch(
            vertex_row_counts, dim=num_triangles, inputs=[indices, 3, entries_per_incidence, counts], device=device
        )

        if out_matrix is None:
            target = bsr_zeros(num_points, num_points, wp.float32, device=device, row_capacity=counts, nnz_capacity=nnz)
        else:
            target = out_matrix
            bsr_set_zero(target, topology="padded", row_capacity=counts, nnz_capacity=nnz)

        wp.launch(
            connectivity_row_entries,
            dim=num_triangles,
            inputs=[
                indices,
                int(include_diagonal),
                target.offsets,
                target.row_counts,
                target.columns,
                target.values,
            ],
            device=device,
        )
        return bsr_compress(target, inplace=True, prune_numerical_zeros=False)

    nnz = num_triangles * (6 + 3 * int(include_diagonal))
    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    columns = wp.empty(nnz, dtype=wp.int32, device=device)

    wp.launch(connectivity_triplets, dim=num_triangles, inputs=[indices], outputs=[rows, columns], device=device)
    if include_diagonal:
        wp.launch(
            connectivity_diagonal_triplets,
            dim=num_triangles,
            inputs=[indices, 6 * num_triangles],
            outputs=[rows, columns],
            device=device,
        )

    target = bsr_zeros(num_points, num_points, wp.float32, device=device) if out_matrix is None else out_matrix
    # Passing no values builds the topology alone, leaving the value array
    # allocated but uninitialized for the caller's fill kernel.
    bsr_set_from_triplets(target, rows, columns, None, topology="compact")
    return target


def laplacian(
    points: wp.array[wp.vec3] | None,
    indices: wp.array[int],
    out_laplacian: BsrMatrix | None = None,
    *,
    weighting: LaplacianWeighting = LaplacianWeighting.COTANGENT,
    num_points: int | None = None,
    simplex_size: int | None = None,
    construction: str = "row_compress",
    reuse_topology: bool = False,
    device: DeviceLike | None = None,
) -> BsrMatrix:
    """Assemble the Laplacian of a triangle or tetrahedral mesh.

    Both weightings produce a symmetric positive semi-definite operator whose
    rows sum to zero, coupling exactly the vertex pairs that share an edge.
    They differ in what an edge is worth.

    With :attr:`LaplacianWeighting.COTANGENT`, each element contributes, for
    every edge, a symmetric off-diagonal pair weighted by the negated cotangent
    weight of that edge, and adds the same weight to the diagonal entries of the
    edge's two endpoints. This is the P1 finite-element stiffness matrix of the
    Laplacian bilinear form ``int(grad(u) . grad(v))``. For a triangle the weight
    is half the cotangent of the angle opposite the edge; for a tetrahedron it is
    the cotangent weight of the linear element (one sixth of the opposite edge
    length times the cotangent of the dihedral angle along it).

    With :attr:`LaplacianWeighting.UNIFORM`, every off-diagonal is ``-1`` and
    every diagonal is the vertex's number of neighbors, giving the graph
    Laplacian ``D - A`` of the mesh's edge graph. An edge counts once however
    many triangles share it, so a boundary edge weighs the same as an interior
    one. This depends only on ``indices``, so ``points`` may be ``None``.
    Uniform weighting is currently supported for triangle meshes only.

    Note:
        The sign convention is the opposite of libigl's ``igl::cotmatrix``,
        which returns a negative semi-definite operator. Code ported from
        libigl needs ``laplacian() == -igl::cotmatrix()``.

    With cotangent weighting the operation is differentiable with respect to
    ``points`` under every construction policy: launch it inside a
    :class:`warp.Tape` with ``points.requires_grad`` set to obtain gradients. The
    gradient of a triangle's contribution is undefined where its area vanishes, so
    degenerate triangles will produce non-finite gradients. When gradients are
    requested, ``construction="row_compress"`` assembles in two passes -- an
    integer pass that fixes the slot layout and a value pass that carries the
    gradient -- so it stays differentiable at a small cost over its
    gradient-free path. The uniform Laplacian is a function of connectivity
    alone, so its derivative with respect to ``points`` is zero rather than
    unavailable.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`. May be
            ``None`` only with :attr:`LaplacianWeighting.UNIFORM`, which reads
            no positions.
        indices: Mesh connectivity, given either as a flat :class:`warp.int32`
            array of consecutive per-simplex vertex indices (length
            ``simplex_size * num_simplices``), or as a typed array of
            :class:`warp.vec3i` (triangles) or :class:`warp.vec4i` (tetrahedra).
            A typed array fixes the simplex size from its element type; a flat
            array uses ``simplex_size``.
        weighting: Edge weighting, as a :class:`LaplacianWeighting` member.
        num_points: Number of vertices, which fixes the size of the matrix.
            Needed only when ``points`` is ``None``; passing it alongside
            ``points`` is an error. When it is omitted and ``points`` is
            ``None``, the count is taken from ``out_laplacian`` if one is given,
            and otherwise from the largest entry of ``indices``, which costs a
            device readback and prevents CUDA graph capture.
        simplex_size: Number of vertices per element, either 3 (triangles) or 4
            (tetrahedra). Used only when ``indices`` is a flat ``int32`` array;
            it defaults to 3 and must agree with the element type of a typed
            ``indices`` array. Tetrahedral meshes support only cotangent
            weighting.
        out_laplacian: Optional output matrix of shape
            ``(num_points, num_points)`` with scalar (1x1) blocks and
            :class:`warp.float32` coefficients. Any blocks it already holds are
            discarded. Its storage is reused when large enough, and grown
            otherwise, so repeated calls on a mesh of fixed topology stop
            reallocating the matrix after the first one. If ``None``, a new
            matrix is allocated.
        construction: How the sparsity pattern is built. Defaults to
            ``"row_compress"``.

            ``"row_compress"`` reserves each vertex row one diagonal plus one
            off-diagonal per other vertex of each incident element -- three
            entries per incident triangle, four per incident tetrahedron -- writes
            the contributions directly into those spans, and compresses each row
            independently. It avoids the global sort and is substantially faster,
            so it is the default. It is differentiable: when ``points`` requires
            gradients it splits assembly into an integer slot pass and a
            differentiable value pass, at a small cost over its gradient-free
            single pass. Because it places entries with an atomic write cursor,
            the order in which a row's coincident contributions are summed varies
            between runs, so its coefficients are reproducible only to rounding
            (a few ULP), not bit for bit.

            ``"triplets"`` emits coordinate-oriented entries per element and lets
            :mod:`warp.sparse` sort and deduplicate them globally. It is slower,
            but its deterministic global sort makes its coefficients bit-for-bit
            reproducible across runs. It makes no assumption about how many
            entries land in a row, so it is also the general path: it is the
            kernel ``reuse_topology`` refills through, and the reference the other
            policies are checked against.

            Both produce the same matrix up to that rounding. Ignored when
            ``reuse_topology`` is set, since no pattern is built in that case.
        reuse_topology: If ``True``, keep the sparsity pattern ``out_laplacian``
            already holds and overwrite only its coefficients. Requires
            ``out_laplacian``. This skips the sort and deduplication that
            dominate assembly, and is several times faster, but it is only
            correct when that pattern already covers every vertex pair the mesh
            couples: contributions landing outside it are silently dropped. Pass
            it a matrix returned by an earlier call on the same ``indices``,
            with either weighting -- the pattern depends only on connectivity,
            never on the values, so a coefficient that happens to vanish still
            keeps its entry.

            With :attr:`LaplacianWeighting.UNIFORM` the pattern must match the
            mesh exactly rather than merely cover it, because the degrees are
            counted from the pattern: a surplus entry inflates a diagonal
            instead of staying zero.
        device: Device on which to run. Defaults to the device of ``points``,
            or of ``indices`` when ``points`` is ``None``.

    Returns:
        A square sparse matrix of shape ``(num_points, num_points)`` with
        scalar (1x1) blocks, which is ``out_laplacian`` when it is provided.

    Raises:
        ValueError: If ``out_laplacian`` is provided but its scalar type, block
            shape, device, or shape does not match the expected output, if
            ``reuse_topology`` is set without a populated ``out_laplacian``, if
            ``construction`` is not a recognized policy, if ``weighting`` is not
            a recognized member, or if ``points`` is ``None`` with cotangent
            weighting. Also raised when ``num_points`` disagrees with ``points``,
            and when ``points`` requires gradients but ``out_laplacian`` is
            provided and does not itself require gradients.
    """
    if construction not in ("triplets", "row_compress"):
        raise ValueError(f"Unsupported `construction` policy: {construction!r}. Expected 'triplets' or 'row_compress'.")

    weighting = LaplacianWeighting(weighting)

    if points is None and weighting == LaplacianWeighting.COTANGENT:
        raise ValueError(
            "`points` is required for cotangent weighting, which is a function of vertex positions. Pass positions, "
            "or request `weighting=LaplacianWeighting.UNIFORM` for the connectivity-only Laplacian."
        )

    device = wp.get_device(device) if device is not None else (indices if points is None else points).device
    indices, simplex_size = _resolve_simplex_indices(indices, simplex_size)
    num_simplices = indices.shape[0] // simplex_size
    num_points = _resolve_num_points(points, indices, num_points, out_laplacian, device)

    if out_laplacian is not None:
        _validate_output_matrix(out_laplacian, "out_laplacian", (num_points, num_points), device)

    # The uniform path still enumerates edges with the triangle-specific
    # connectivity kernels; tetrahedra are supported only with cotangent weighting.
    if simplex_size != 3 and weighting == LaplacianWeighting.UNIFORM:
        raise NotImplementedError(
            "Uniform weighting is currently implemented for triangle meshes only. Use "
            "`weighting=LaplacianWeighting.COTANGENT` for tetrahedral meshes."
        )

    # Reject the combinations that would run happily and hand back zero
    # gradients, which is harder to notice than a failure. Uniform weighting is
    # exempt: it does not read positions, so a zero gradient is the true answer.
    if weighting == LaplacianWeighting.COTANGENT and points.requires_grad:
        if out_laplacian is not None and not out_laplacian.requires_grad:
            raise ValueError(
                "`points` requires gradients but `out_laplacian` does not, so no gradient would reach "
                "`points`. Set `out_laplacian.values.requires_grad = True`."
            )

    if reuse_topology:
        if out_laplacian is None:
            raise ValueError("`reuse_topology` requires `out_laplacian`, whose sparsity pattern it reuses.")
        # An empty matrix has no pattern to reuse, so every contribution would be
        # dropped and the result would be silently zero. Catching that here costs
        # nothing, unlike verifying that a non-empty pattern is the right one,
        # which would need a device readback.
        if out_laplacian.nnz == 0:
            raise ValueError(
                "`reuse_topology` requires `out_laplacian` to already hold the sparsity pattern of the "
                "Laplacian, but it is empty. Call `laplacian()` without `reuse_topology` first."
            )

    if weighting == LaplacianWeighting.UNIFORM:
        # The uniform weights cannot be accumulated per triangle, so the pattern
        # is built first and the values are then read off it.
        target = _connectivity_matrix(
            indices,
            num_points,
            out_laplacian,
            construction,
            reuse_topology,
            include_diagonal=True,
            device=device,
        )
        wp.launch(
            uniform_laplacian_values,
            dim=num_points,
            inputs=[target.offsets, target.columns],
            outputs=[target.values],
            device=device,
        )
        return target

    # Refilling an existing pattern writes no topology, so it always goes
    # through the triplet path regardless of the construction policy.
    if construction == "row_compress" and not reuse_topology:
        # A vertex reserves one diagonal plus one off-diagonal per other vertex of
        # each incident element: three entries per incident triangle, four per
        # incident tetrahedron.
        entries_per_incidence = simplex_size
        nnz = num_simplices * simplex_size * entries_per_incidence
        if simplex_size == 3:
            entries_kernel = laplacian_row_entries
            slots_kernel = laplacian_row_slots
            values_kernel = laplacian_row_values
        else:
            entries_kernel = tet_laplacian_row_entries
            slots_kernel = tet_laplacian_row_slots
            values_kernel = tet_laplacian_row_values

        counts = wp.zeros(num_points, dtype=wp.int32, device=device)
        wp.launch(
            vertex_row_counts,
            dim=num_simplices,
            inputs=[indices, simplex_size, entries_per_incidence, counts],
            device=device,
        )

        if out_laplacian is None:
            target = bsr_zeros(num_points, num_points, wp.float32, device=device, row_capacity=counts, nnz_capacity=nnz)
        else:
            target = out_laplacian
            bsr_set_zero(target, topology="padded", row_capacity=counts, nnz_capacity=nnz)

        if not points.requires_grad:
            # No gradients wanted: assign the columns and values together with the
            # racy write cursor in one pass, then coalesce in place. This is the
            # fastest path and stays byte-identical to before.
            wp.launch(
                entries_kernel,
                dim=num_simplices,
                inputs=[points, indices, target.offsets, target.row_counts, target.columns, target.values],
                device=device,
            )
            return bsr_compress(target, inplace=True, prune_numerical_zeros=False)

        # Gradients wanted: split the pass in two so the backward pass can replay
        # it. The first pass fixes the slot layout with the racy cursor, touching
        # only integers; the second writes the values into those fixed slots and
        # carries the gradient. Coalescing then runs through the differentiable
        # (out-of-place) compression.
        target.values.requires_grad = True
        entry_slots = wp.empty(nnz, dtype=wp.int32, device=device)
        wp.launch(
            slots_kernel,
            dim=num_simplices,
            inputs=[indices, target.offsets, target.row_counts, target.columns, entry_slots],
            device=device,
        )
        wp.launch(
            values_kernel,
            dim=num_simplices,
            inputs=[points, indices, entry_slots],
            outputs=[target.values],
            device=device,
        )
        return bsr_compress(target, inplace=False, prune_numerical_zeros=False)

    # Triangles emit nine coordinate entries per face; tetrahedra emit
    # twenty-four per cell (four per edge over six edges).
    triplet_kernel = laplacian_triplets if simplex_size == 3 else tet_laplacian_triplets
    entries_per_simplex = 9 if simplex_size == 3 else 24
    nnz = num_simplices * entries_per_simplex

    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    columns = wp.empty(nnz, dtype=wp.int32, device=device)
    # The triplet values carry the gradient: warp.sparse propagates
    # ``requires_grad`` from here onto the assembled matrix and accumulates them
    # with differentiable kernels.
    values = wp.empty(nnz, dtype=wp.float32, device=device, requires_grad=points.requires_grad)

    wp.launch(
        triplet_kernel,
        dim=num_simplices,
        inputs=[points, indices, rows, columns, values],
        device=device,
    )

    # The sparsity pattern must stay purely topological. Pruning entries that
    # happen to be numerically zero would drop an edge whose opposite angles are
    # both right angles -- ubiquitous on grid meshes -- and `reuse_topology`
    # would then silently discard that edge once the mesh deforms.
    if out_laplacian is None:
        return bsr_from_triplets(num_points, num_points, rows, columns, values, prune_numerical_zeros=False)

    bsr_set_from_triplets(
        out_laplacian,
        rows,
        columns,
        values,
        prune_numerical_zeros=False,
        topology="masked" if reuse_topology else "compact",
    )
    return out_laplacian


def vertex_adjacency_matrix(
    indices: wp.array[int],
    out_adjacency: BsrMatrix | None = None,
    *,
    num_points: int | None = None,
    construction: str = "row_compress",
    reuse_topology: bool = False,
    device: DeviceLike | None = None,
) -> BsrMatrix:
    """Assemble the vertex adjacency matrix of a triangle mesh.

    Entry ``(i, j)`` is ``1`` when vertices ``i`` and ``j`` are joined by a
    triangle edge and absent otherwise, so the result is symmetric, has a zero
    diagonal, and holds one entry per directed edge. An edge counts once however
    many triangles share it.

    This is the ``A`` of the uniform Laplacian ``D - A``, and is built from the
    same sparsity pattern, minus its diagonal. Reach for
    :func:`laplacian` with :attr:`LaplacianWeighting.UNIFORM` when the degrees
    are wanted too.

    Args:
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_adjacency: Optional output matrix of shape
            ``(num_points, num_points)`` with scalar (1x1) blocks and
            :class:`warp.float32` coefficients. Any blocks it already holds are
            discarded. Its storage is reused when large enough, and grown
            otherwise. If ``None``, a new matrix is allocated.
        num_points: Number of vertices, which fixes the size of the matrix. When
            omitted, the count is taken from ``out_adjacency`` if one is given,
            and otherwise from the largest entry of ``indices``, which costs a
            device readback and prevents CUDA graph capture.
        construction: How the sparsity pattern is built, either ``"triplets"``
            or ``"row_compress"``. Both produce the same matrix; see
            :func:`laplacian` for the trade-off. Ignored when
            ``reuse_topology`` is set.
        reuse_topology: If ``True``, keep the sparsity pattern ``out_adjacency``
            already holds and overwrite only its coefficients. Requires
            ``out_adjacency``, whose pattern must match the mesh: a surplus
            entry becomes a spurious ``1`` rather than staying zero. Pass it a
            matrix returned by an earlier call on the same ``indices``.
        device: Device on which to run. Defaults to the device of ``indices``.

    Returns:
        A square sparse matrix of shape ``(num_points, num_points)`` with
        scalar (1x1) blocks, which is ``out_adjacency`` when it is provided.

    Raises:
        ValueError: If ``out_adjacency`` is provided but its scalar type, block
            shape, device, or shape does not match the expected output, if
            ``reuse_topology`` is set without a populated ``out_adjacency``, or
            if ``construction`` is not a recognized policy.
    """
    if construction not in ("triplets", "row_compress"):
        raise ValueError(f"Unsupported `construction` policy: {construction!r}. Expected 'triplets' or 'row_compress'.")

    device = wp.get_device(device) if device is not None else indices.device
    num_points = _resolve_num_points(None, indices, num_points, out_adjacency, device)

    if out_adjacency is not None:
        _validate_output_matrix(out_adjacency, "out_adjacency", (num_points, num_points), device)

    if reuse_topology:
        if out_adjacency is None:
            raise ValueError("`reuse_topology` requires `out_adjacency`, whose sparsity pattern it reuses.")
        if out_adjacency.nnz == 0:
            raise ValueError(
                "`reuse_topology` requires `out_adjacency` to already hold the sparsity pattern of the "
                "adjacency matrix, but it is empty. Call `vertex_adjacency_matrix()` without "
                "`reuse_topology` first."
            )

    target = _connectivity_matrix(
        indices,
        num_points,
        out_adjacency,
        construction,
        reuse_topology,
        include_diagonal=False,
        device=device,
    )
    wp.launch(
        adjacency_values,
        dim=num_points,
        inputs=[target.offsets, target.columns],
        outputs=[target.values],
        device=device,
    )
    return target
