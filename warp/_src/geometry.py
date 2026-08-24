# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.marching_cubes import MarchingCubes
from warp._src.sparse import (
    BsrMatrix,
    bsr_compress,
    bsr_from_triplets,
    bsr_set_from_triplets,
    bsr_set_zero,
    bsr_zeros,
)
from warp._src.types import type_repr, types_equal

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

##########################################################################
## Functions that operate on local elements (reusable within kernels)
## These are **first-class** citizens and will be useful to programmers
## of warp as they write their own kernels.
##########################################################################


@wp.func
def line_segment_barycenter(v0: wp.vec3, v1: wp.vec3) -> wp.vec3:
    # Barycenter of a line segment, which is its midpoint.
    return (v0 + v1) * 0.5


@wp.func
def triangle_barycenter(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    # Barycenter of a triangle, which is the average of its three vertices.
    return (v0 + v1 + v2) / 3.0


@wp.func
def tetrahedron_barycenter(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> wp.vec3:
    # Barycenter of a tetrahedron, which is the average of its four vertices.
    return (v0 + v1 + v2 + v3) / 4.0


@wp.func
def line_segment_barycentric_coordinates(v0: wp.vec3, v1: wp.vec3, p: wp.vec3) -> wp.vec2:
    # Barycentric coordinates of a point ``p`` with respect to a line segment
    # ``(v0, v1)``. The coordinates are in ``[0, 1]`` if ``p`` is on the
    # segment.
    d = v1 - v0
    t = wp.dot(p - v0, d) / wp.dot(d, d)
    return wp.vec2(1.0 - t, t)


@wp.func
def triangle_barycentric_coordinates(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, p: wp.vec3) -> wp.vec3:
    # Barycentric coordinates of a point ``p`` with respect to a triangle
    # ``(v0, v1, v2)``. The coordinates are in ``[0, 1]`` if ``p`` is on the
    # triangle.
    d0 = v1 - v0
    d1 = v2 - v0
    d2 = p - v0

    d00 = wp.dot(d0, d0)
    d01 = wp.dot(d0, d1)
    d11 = wp.dot(d1, d1)
    d20 = wp.dot(d2, d0)
    d21 = wp.dot(d2, d1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return wp.vec3(u, v, w)


_SF_PHI = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
"""First Super-Fibonacci spiral constant, ``1 / sqrt(2)``."""

_SF_PSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))
"""Second Super-Fibonacci spiral constant, the reciprocal of the plastic-like root."""


@wp.func
def super_fibonacci(i: int, n: int) -> wp.quat:
    # Maps sample index ``i`` in ``[0, n)`` to a near-uniformly distributed unit
    # quaternion on SO(3) via the Super-Fibonacci spiral (Alexa, "Super-Fibonacci
    # Spirals: Fast, Low-Discrepancy Sampling of SO(3)", CVPR 2022).
    #
    # Both spiral angles grow linearly with ``i``, and only their fractional turn
    # matters. They are therefore wrapped into ``[0, 1)`` in double precision
    # before reaching sin/cos: in float32 the angle for ``i`` in the tens of
    # thousands is large enough that argument reduction consumes the fractional
    # bits the spiral depends on, and the samples stop being low-discrepancy.
    s = wp.float64(i) + wp.float64(0.5)
    a = s * _SF_PHI
    b = s * _SF_PSI
    alpha = wp.float32(a - wp.floor(a)) * 2.0 * wp.pi
    beta = wp.float32(b - wp.floor(b)) * 2.0 * wp.pi

    t = wp.float32(s / wp.float64(n))
    r = wp.sqrt(t)
    R = wp.sqrt(1.0 - t)
    return wp.quat(r * wp.sin(alpha), r * wp.cos(alpha), R * wp.sin(beta), R * wp.cos(beta))


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
def corner_half_angle(x: wp.vec3, y: wp.vec3, z: wp.vec3) -> wp.float32:
    # Interior angle at ``y`` between edges ``y->x`` and ``y->z``. Uses Kahan's
    # numerically stable half-angle formula, which stays accurate for angles near
    # 0 and pi where ``acos(dot(...))`` loses precision.
    a = wp.normalize(x - y)
    b = wp.normalize(z - y)
    return wp.atan2(wp.length(a - b), wp.length(a + b))


@wp.func
def triangle_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, normalized: bool = False) -> wp.vec3:
    n = wp.cross(v1 - v0, v2 - v0)
    if normalized:
        n = wp.normalize(n)
    return n


@wp.func
def triangle_corner_half_angles(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.vec3(
        corner_half_angle(v2, v0, v1),
        corner_half_angle(v0, v1, v2),
        corner_half_angle(v1, v2, v0),
    )


@wp.func
def triangle_double_area(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.float32:
    n = triangle_normal(v0, v1, v2, normalized=False)
    return wp.length(n)


##########################################################################
## Raw kernels. These are meant to be internal and are the
## "implementation details" for the public-facing functions below. They
## are not intended to be called directly by users.
##########################################################################


@wp.kernel
def simplex_barycenters_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    simplex_size: int,
    out_barycenters: wp.array(dtype=wp.vec3),
):
    # Doesn't actually call line_segment_barycenter, triangle_barycenter,
    # tetrahedron_barycenter
    simplex = wp.tid()
    base = simplex * simplex_size

    barycenter = wp.vec3(0.0)

    for local_vertex in range(simplex_size):
        vertex = points[indices[base + local_vertex]]
        barycenter += vertex

    out_barycenters[simplex] = barycenter / wp.float32(simplex_size)


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
def vertex_row_counts(indices: wp.array[int], entries_per_incidence: int, counts: wp.array[int]):
    # Reserved storage for row ``v`` is a fixed number of entries per incident
    # triangle: two off-diagonals to the triangle's other two vertices, plus a
    # diagonal entry when the caller wants one. Counting incidences needs no edge
    # enumeration and makes no manifoldness assumption.
    tri = wp.tid()
    for k in range(3):
        wp.atomic_add(counts, indices[3 * tri + k], entries_per_incidence)


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


class OBBMeasureType(enum.IntEnum):
    """Objective minimized when searching for an oriented bounding box.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel launch (Warp kernels cannot take a ``str`` parameter).
    """

    VOLUME = 0
    """Minimize the volume of the bounding box."""
    SURFACE_AREA = 1
    """Minimize the surface area of the bounding box."""


_OBB_POINT_CHUNKS = 256
"""Threads each candidate OBB orientation splits its point loop across."""


@wp.kernel(enable_backward=False)
def oriented_bounding_box_samples_kernel(num_samples: int, rotations: wp.array(dtype=wp.quat)):
    # Fill the spiral portion of the candidate list. Any extra candidates are
    # written into the slots past ``num_samples`` by the kernels below.
    i = wp.tid()
    rotations[i] = super_fibonacci(i, num_samples)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_identity_kernel(slot: int, rotations: wp.array(dtype=wp.quat)):
    # The spiral never contains the identity exactly, so the axis-aligned box is
    # not otherwise among the candidates.
    rotations[slot] = wp.quat_identity()


@wp.kernel(enable_backward=False)
def point_sum_kernel(points: wp.array(dtype=wp.vec3), out_sum: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wp.atomic_add(out_sum, 0, points[i])


@wp.kernel(enable_backward=False)
def point_covariance_kernel(
    points: wp.array(dtype=wp.vec3),
    point_sum: wp.array(dtype=wp.vec3),
    out_covariance: wp.array(dtype=wp.mat33),
):
    # Scatter matrix about the centroid. Left unnormalized: scaling by 1/n does
    # not change the eigenvectors, which is all the caller wants.
    i = wp.tid()
    centroid = point_sum[0] / float(points.shape[0])
    d = points[i] - centroid
    wp.atomic_add(out_covariance, 0, wp.outer(d, d))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_pca_kernel(
    covariance: wp.array(dtype=wp.mat33),
    slot: int,
    rotations: wp.array(dtype=wp.quat),
):
    # The principal axes of the point set are the eigenvectors of its covariance
    # matrix, which is a good starting guess for elongated shapes: the spiral
    # resolves orientation only to its sample spacing, and a few degrees of error
    # costs a lot of volume when one axis is much longer than the others.
    # Eigenvalues are unused: the box is the same whichever order the axes come in.
    Q, _eigenvalues = wp.eig3(covariance[0])

    # ``eig3`` returns orthonormal columns but does not promise a right-handed
    # frame, and ``quat_from_matrix`` is only defined for a pure rotation.
    if wp.determinant(Q) < 0.0:
        Q = wp.mat33(
            -Q[0, 0], Q[0, 1], Q[0, 2],
            -Q[1, 0], Q[1, 1], Q[1, 2],
            -Q[2, 0], Q[2, 1], Q[2, 2],
        )  # fmt: skip

    # Q's columns are the principal axes in world space, so Q maps box-local to
    # world; the search below wants the world-to-box-local direction.
    rotations[slot] = wp.quat_inverse(wp.quat_from_matrix(Q))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_bounds_kernel(
    points: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    num_chunks: int,
    min_bounds: wp.array(dtype=wp.vec3),
    max_bounds: wp.array(dtype=wp.vec3),
):
    # Parallel over (orientation, point chunk). Parallelizing over orientations
    # alone caps the launch at one thread per candidate no matter how many points
    # there are, which leaves most of the GPU idle; splitting the point loop as
    # well makes the available parallelism scale with the point count.
    candidate, chunk = wp.tid()

    rot = rotations[candidate]

    # Axis-aligned bounds of this thread's slice of points in the rotated frame.
    # A grid-stride slice keeps neighboring threads on neighboring points, so the
    # reads coalesce.
    lo = wp.vec3(wp.inf, wp.inf, wp.inf)
    hi = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

    num_points = points.shape[0]
    for j in range(chunk, num_points, num_chunks):
        rotated = wp.quat_rotate(rot, points[j])
        lo = wp.min(lo, rotated)
        hi = wp.max(hi, rotated)

    # Combine the per-chunk bounds. Unlike a sum, min/max are exact in floating
    # point, so this reduction is deterministic regardless of the order in which
    # the atomics land.
    wp.atomic_min(min_bounds, candidate, lo)
    wp.atomic_max(max_bounds, candidate, hi)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_measure_kernel(
    rotations: wp.array(dtype=wp.quat),
    min_bounds: wp.array(dtype=wp.vec3),
    max_bounds: wp.array(dtype=wp.vec3),
    measure_type: int,
    measures: wp.array(dtype=wp.float32),
    transforms: wp.array(dtype=wp.transform),
    extents: wp.array(dtype=wp.vec3),
):
    # One thread per candidate orientation, scoring the box found above.
    i = wp.tid()

    rot = rotations[i]
    lo = min_bounds[i]
    hi = max_bounds[i]

    # Full side lengths of the box and the measure being minimized. Default to the
    # volume; only the surface-area branch overrides it (keeps `measure` defined on
    # every path for codegen).
    dims = hi - lo
    measure = dims[0] * dims[1] * dims[2]
    if measure_type == wp.static(int(OBBMeasureType.SURFACE_AREA)):
        measure = 2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])

    # Box center in the rotated frame, mapped back into world space. The stored
    # transform takes the box's local axis-aligned frame to world coordinates.
    center = (hi + lo) * 0.5
    world_center = wp.quat_rotate_inv(rot, center)

    measures[i] = measure
    extents[i] = dims
    transforms[i] = wp.transform(world_center, wp.quat_inverse(rot))


@wp.kernel
def vertex_gaussian_curvature_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_curvature: wp.array(dtype=wp.float32),  # pre-initialized to 2*pi per vertex
):
    tri = wp.tid()

    i0 = indices[tri * 3 + 0]
    i1 = indices[tri * 3 + 1]
    i2 = indices[tri * 3 + 2]
    v0 = points[i0]
    v1 = points[i1]
    v2 = points[i2]

    # Subtract each incident interior angle from the vertex's running angle defect.
    neg_angles = -2.0 * triangle_corner_half_angles(v0, v1, v2)

    wp.atomic_add(out_curvature, i0, neg_angles[0])
    wp.atomic_add(out_curvature, i1, neg_angles[1])
    wp.atomic_add(out_curvature, i2, neg_angles[2])


@wp.kernel
def triangle_corner_angles_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_angles: wp.array(dtype=wp.vec3),
):
    tri = wp.tid()
    v0 = points[indices[tri * 3 + 0]]
    v1 = points[indices[tri * 3 + 1]]
    v2 = points[indices[tri * 3 + 2]]
    out_angles[tri] = 2.0 * triangle_corner_half_angles(v0, v1, v2)


@wp.kernel
def triangle_areas_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_areas: wp.array(dtype=wp.float32),
):
    tri = wp.tid()
    v0 = points[indices[tri * 3 + 0]]
    v1 = points[indices[tri * 3 + 1]]
    v2 = points[indices[tri * 3 + 2]]
    out_areas[tri] = 0.5 * triangle_double_area(v0, v1, v2)


@wp.kernel
def triangle_normals_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    normalized: bool,
    out_normals: wp.array(dtype=wp.vec3),
):
    tri = wp.tid()
    v0 = points[indices[tri * 3 + 0]]
    v1 = points[indices[tri * 3 + 1]]
    v2 = points[indices[tri * 3 + 2]]
    out_normals[tri] = triangle_normal(v0, v1, v2, normalized=normalized)


class VertexNormalWeighting(enum.IntEnum):
    """Weighting scheme for accumulating incident face normals into a vertex normal.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel launch (Warp kernels cannot take a ``str`` parameter).
    """

    AREA = 0
    """Weight each incident triangle by its area (unnormalized face normals)."""
    UNIFORM = 1
    """Weight every incident triangle equally (unit face normals)."""
    ANGLE = 2
    """Weight every incident triangle by incident angle."""


@wp.kernel
def vertex_normals_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    weighting: int,
    out_vertex_normals: wp.array(dtype=wp.vec3),
):
    tri = wp.tid()
    i0 = indices[tri * 3 + 0]
    i1 = indices[tri * 3 + 1]
    i2 = indices[tri * 3 + 2]
    v0 = points[i0]
    v1 = points[i1]
    v2 = points[i2]

    n = triangle_normal(v0, v1, v2)

    # Per-corner contribution weights. They are equal for the 'area' and 'uniform'
    # schemes; 'angle' weighting is the only one that differs per corner.
    weights = wp.vec3(1.0, 1.0, 1.0)
    if weighting == wp.static(int(VertexNormalWeighting.UNIFORM)):
        # Unit face normals -> every incident face contributes equally.
        n = wp.normalize(n)
    elif weighting == wp.static(int(VertexNormalWeighting.ANGLE)):
        # Unit face normals weighted by each triangle's interior angle at the vertex.
        n = wp.normalize(n)
        weights = triangle_corner_half_angles(v0, v1, v2)
    # VertexNormalWeighting.AREA: leave n unnormalized so its magnitude (2 * area) weights by area.

    wp.atomic_add(out_vertex_normals, i0, n * weights[0])
    wp.atomic_add(out_vertex_normals, i1, n * weights[1])
    wp.atomic_add(out_vertex_normals, i2, n * weights[2])


@wp.kernel
def normalize_kernel(
    in_vectors: wp.array(dtype=wp.vec3),
    out_vectors: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out_vectors[i] = wp.normalize(in_vectors[i])


@wp.kernel
def accumulate_moments(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out0: wp.array(dtype=float),  # volume
    out1: wp.array(dtype=wp.vec3),  # first moment (centroid * volume)
    raw2: wp.array(dtype=wp.mat33),  # raw second moment accumulators
):
    tri = wp.tid()

    p0 = points[indices[3 * tri + 0]]
    p1 = points[indices[3 * tri + 1]]
    p2 = points[indices[3 * tri + 2]]

    x0, y0, z0 = p0[0], p0[1], p0[2]
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]

    # Six times the signed volume of the tetrahedron (origin, p0, p1, p2).
    v = x0 * (y1 * z2 - y2 * z1) - x1 * (y0 * z2 - y2 * z0) + x2 * (y0 * z1 - y1 * z0)

    x3, y3, z3 = x0 + x1 + x2, y0 + y1 + y2, z0 + z1 + z2

    xx = v * (x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3)
    yy = v * (y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3)
    zz = v * (z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
    yx = v * (y0 * x0 + y1 * x1 + y2 * x2 + y3 * x3)
    zx = v * (z0 * x0 + z1 * x1 + z2 * x2 + z3 * x3)
    zy = v * (z0 * y0 + z1 * y1 + z2 * y2 + z3 * y3)

    wp.atomic_add(out0, 0, v / 6.0)
    wp.atomic_add(out1, 0, v * wp.vec3(x3, y3, z3) / 24.0)
    wp.atomic_add(raw2, 0, wp.mat33(xx, yx, zx, yx, yy, zy, zx, zy, zz))


@wp.kernel
def finalize_moments(
    m0: wp.array(dtype=float),
    m1: wp.array(dtype=wp.vec3),
    raw2: wp.array(dtype=wp.mat33),
    out2: wp.array(dtype=wp.mat33),  # inertia tensor about the centroid
):
    mass = m0[0]
    first = m1[0]
    R = raw2[0]
    r = 1.0 / 120.0

    xx = R[0, 0] * r - first[0] * first[0] / mass
    yy = R[1, 1] * r - first[1] * first[1] / mass
    zz = R[2, 2] * r - first[2] * first[2] / mass
    yx = first[1] * first[0] / mass - R[1, 0] * r
    zx = first[2] * first[0] / mass - R[2, 0] * r
    zy = first[2] * first[1] / mass - R[2, 1] * r

    out2[0] = wp.mat33(yy + zz, yx, zx, yx, xx + zz, zy, zx, zy, xx + yy)


##########################################################################
## Exposed functions
##########################################################################


def _validate_output(out: wp.array, name: str, length: int, dtype, device: DeviceLike) -> None:
    """Check that a caller-supplied output array matches the expected dtype, device, and length."""
    if not types_equal(out.dtype, dtype):
        raise ValueError(f"`{name}` must have dtype {type_repr(dtype)}, but got {type_repr(out.dtype)}.")
    if out.device != device:
        raise ValueError(f"`{name}` must be on device '{device}', but got '{out.device}'.")
    if out.shape[0] < length:
        raise ValueError(f"`{name}` must have length at least {length}, but got {out.shape[0]}.")


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
        wp.launch(vertex_row_counts, dim=num_triangles, inputs=[indices, entries_per_incidence, counts], device=device)

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


def simplex_barycenters(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    simplex_size: int,
    out_barycenters: wp.array(dtype=wp.vec3) | None = None,
    *,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.vec3):
    """Compute the barycenter of each simplex in a mesh.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of simplex vertex indices, with ``simplex_size ``
            consecutive entries per simplex (length ``(simplex_size ) * num_simplices``).
        simplex_size: Number of vertices per simplex (2 for line segments, 3 for triangles, 4 for tetrahedra).
        out_barycenters: Optional output array of length ``num_simplices`` to store the
            per-simplex barycenters. If ``None``, a new array is allocated with the same
            ``requires_grad`` setting as ``points``.
        device: Device on which to run. Defaults to the device of ``points``.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_simplices = indices.shape[0] // (simplex_size)

    if out_barycenters is None:
        out_barycenters = wp.empty(
            num_simplices,
            dtype=wp.vec3,
            device=device,
            requires_grad=points.requires_grad,
        )
    else:
        _validate_output(out_barycenters, "out_barycenters", num_simplices, wp.vec3, device)

    wp.launch(
        simplex_barycenters_kernel,
        dim=num_simplices,
        inputs=[points, indices, simplex_size],
        outputs=[out_barycenters],
        device=device,
    )

    return out_barycenters


def triangle_corner_angles(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_angles: wp.array(dtype=wp.vec3) | None = None,
    *,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.vec3):
    """Compute the three interior angles of each triangle in a triangle mesh.

    Each entry of the returned array holds the interior angles, in radians, at the
    triangle's three corners in vertex order ``(v0, v1, v2)``; the three angles of a
    triangle sum to ``pi``. The operation is differentiable with respect to
    ``points``: launch it inside a :class:`warp.Tape` with ``requires_grad`` arrays
    to obtain gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_angles: Optional output array of length ``num_triangles`` to store the
            per-triangle corner angles. If ``None``, a new array is allocated with
            the same ``requires_grad`` setting as ``points``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The ``out_angles`` array, containing the three corner angles of each triangle.

    Raises:
        ValueError: If ``out_angles`` is provided but its dtype, device, or length
            does not match the expected output.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3

    if out_angles is None:
        out_angles = wp.empty(
            num_triangles,
            dtype=wp.vec3,
            device=device,
            requires_grad=points.requires_grad,
        )
    else:
        _validate_output(out_angles, "out_angles", num_triangles, wp.vec3, device)

    wp.launch(
        triangle_corner_angles_kernel,
        dim=num_triangles,
        inputs=[points, indices],
        outputs=[out_angles],
        device=device,
    )

    return out_angles


def triangle_areas(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_areas: wp.array(dtype=wp.float32) | None = None,
    *,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.float32):
    """Compute the area of each triangle in a triangle mesh.

    Each triangle area is half the magnitude of the cross product of two of its
    edge vectors. The operation is differentiable with respect to ``points``:
    launch it inside a :class:`warp.Tape` with ``requires_grad`` arrays to obtain
    gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_areas: Optional output array of length ``num_triangles`` to store the
            per-triangle areas. If ``None``, a new array is allocated with the same
            ``requires_grad`` setting as ``points``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The ``out_areas`` array, containing the area of each triangle.

    Raises:
        ValueError: If ``out_areas`` is provided but its dtype, device, or length
            does not match the expected output.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3

    if out_areas is None:
        out_areas = wp.empty(
            num_triangles,
            dtype=wp.float32,
            device=device,
            requires_grad=points.requires_grad,
        )
    else:
        _validate_output(out_areas, "out_areas", num_triangles, wp.float32, device)

    wp.launch(
        triangle_areas_kernel,
        dim=num_triangles,
        inputs=[points, indices],
        outputs=[out_areas],
        device=device,
    )

    return out_areas


def triangle_normals(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_normals: wp.array(dtype=wp.vec3) | None = None,
    *,
    normalized: bool = False,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.vec3):
    """Compute the normal of each triangle in a triangle mesh.

    Each triangle normal is the cross product of two of its edge vectors. By
    default the result is unnormalized, so its magnitude equals twice the triangle
    area; pass ``normalized=True`` to obtain unit normals. The operation is
    differentiable with respect to ``points``: launch it inside a
    :class:`warp.Tape` with ``requires_grad`` arrays to obtain gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_normals: Optional output array of length ``num_triangles`` to store the
            per-triangle normals. If ``None``, a new array is allocated with the same
            ``requires_grad`` setting as ``points``.
        normalized: If ``True``, each normal is scaled to unit length.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The ``out_normals`` array, containing the normal of each triangle.

    Raises:
        ValueError: If ``out_normals`` is provided but its dtype, device, or length
            does not match the expected output.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3

    if out_normals is None:
        out_normals = wp.empty(
            num_triangles,
            dtype=wp.vec3,
            device=device,
            requires_grad=points.requires_grad,
        )
    else:
        _validate_output(out_normals, "out_normals", num_triangles, wp.vec3, device)

    wp.launch(
        triangle_normals_kernel,
        dim=num_triangles,
        inputs=[points, indices, normalized],
        outputs=[out_normals],
        device=device,
    )

    return out_normals


def vertex_normals(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_normals: wp.array(dtype=wp.vec3) | None = None,
    *,
    weighting: VertexNormalWeighting = VertexNormalWeighting.AREA,
    normalized: bool = False,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.vec3):
    """Compute a normal for each vertex of a triangle mesh.

    Each vertex normal is a weighted sum of the normals of its incident triangles.
    The ``weighting`` scheme sets how much each triangle contributes, and
    ``normalized`` optionally rescales the final vertex normals to unit length. The
    operation is differentiable with respect to ``points``: launch it inside a
    :class:`warp.Tape` with ``requires_grad`` arrays to obtain gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_normals: Optional output array of length ``len(points)`` to store the
            per-vertex normals. If ``None``, a new array is allocated with the same
            ``requires_grad`` setting as ``points``.
        weighting: How each incident triangle is weighted, as a
            :class:`VertexNormalWeighting` member. :attr:`VertexNormalWeighting.AREA`
            weights by triangle area (unnormalized face normals);
            :attr:`VertexNormalWeighting.UNIFORM` weights every incident triangle
            equally (unit face normals); :attr:`VertexNormalWeighting.ANGLE` weights
            each incident triangle by its interior angle at the vertex (unit face
            normals).
        normalized: If ``True``, each summed vertex normal is scaled to unit length.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The ``out_normals`` array, containing the normal of each vertex.

    Raises:
        ValueError: If ``out_normals`` is provided but its dtype, device, or length
            does not match the expected output.
    """
    weighting_code = int(VertexNormalWeighting(weighting))

    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3
    num_vertices = points.shape[0]

    if out_normals is None:
        out_normals = wp.empty(num_vertices, dtype=wp.vec3, device=device, requires_grad=points.requires_grad)
    else:
        _validate_output(out_normals, "out_normals", num_vertices, wp.vec3, device)

    # Face normals are scattered onto vertices with atomic_add, so the accumulation
    # target must start at zero. When normalizing, accumulate into a separate buffer
    # and normalize out-of-place into out_normals to keep the pass differentiable.
    accum = (
        wp.zeros(num_vertices, dtype=wp.vec3, device=device, requires_grad=points.requires_grad)
        if normalized
        else out_normals
    )
    if not normalized:
        accum.zero_()

    wp.launch(
        vertex_normals_kernel,
        dim=num_triangles,
        inputs=[points, indices, weighting_code],
        outputs=[accum],
        device=device,
    )

    if normalized:
        wp.launch(
            normalize_kernel,
            dim=num_vertices,
            inputs=[accum],
            outputs=[out_normals],
            device=device,
        )

    return out_normals


def moments(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_volume: wp.array(dtype=wp.float32) | None = None,
    out_first_moment: wp.array(dtype=wp.vec3) | None = None,
    out_inertia: wp.array(dtype=wp.mat33) | None = None,
    *,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Compute the volume, first moment, and inertia tensor of a closed triangle mesh.

    The mesh is assumed to bound a solid region of uniform unit density. Each
    quantity is accumulated over the tetrahedra spanned by the origin and each
    triangle, so the mesh must be closed and consistently oriented. The operation
    is differentiable with respect to ``points``: launch it inside a
    :class:`warp.Tape` with ``requires_grad`` arrays to obtain gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_volume: Optional length-1 output array for the enclosed volume. If
            ``None``, a new array is allocated.
        out_first_moment: Optional length-1 output array for the first moment
            (centroid scaled by volume). If ``None``, a new array is allocated.
        out_inertia: Optional length-1 output array for the inertia tensor about
            the centroid. If ``None``, a new array is allocated.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(volume, first_moment, inertia)`` of length-1 arrays.

    Raises:
        ValueError: If any provided output array's dtype, device, or length does
            not match the expected output.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3

    # Volume and first moment are accumulated with atomic_add, so they must start at zero.
    if out_volume is None:
        out_volume = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=points.requires_grad)
    else:
        _validate_output(out_volume, "out_volume", 1, wp.float32, device)
        out_volume.zero_()

    if out_first_moment is None:
        out_first_moment = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=points.requires_grad)
    else:
        _validate_output(out_first_moment, "out_first_moment", 1, wp.vec3, device)
        out_first_moment.zero_()

    # Inertia is written outright by finalize_moments, so it does not need to be zeroed.
    if out_inertia is None:
        out_inertia = wp.empty(1, dtype=wp.mat33, device=device, requires_grad=points.requires_grad)
    else:
        _validate_output(out_inertia, "out_inertia", 1, wp.mat33, device)

    # Raw second-moment accumulator is an internal temporary.
    raw2 = wp.zeros(1, dtype=wp.mat33, device=device, requires_grad=points.requires_grad)

    wp.launch(
        accumulate_moments,
        dim=num_triangles,
        inputs=[points, indices, out_volume, out_first_moment, raw2],
        device=device,
    )
    wp.launch(
        finalize_moments,
        dim=1,
        inputs=[out_volume, out_first_moment, raw2, out_inertia],
        device=device,
    )

    return out_volume, out_first_moment, out_inertia


def vertex_gaussian_curvature(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_curvature: wp.array(dtype=wp.float32) | None = None,
    *,
    device: DeviceLike | None = None,
) -> wp.array(dtype=wp.float32):
    """Compute the discrete Gaussian curvature at each vertex of a triangle mesh.

    The curvature is the angle defect ``2*pi - sum(theta)``, where ``theta`` ranges
    over the interior triangle angles incident to the vertex. This is the *integrated*
    Gaussian curvature over the vertex's dual cell (it is not divided by area), so by
    the Gauss-Bonnet theorem the values sum to ``2*pi*chi`` for a closed mesh (for
    example ``4*pi`` for any genus-0 surface). Boundary vertices are not treated
    specially. The operation is differentiable with respect to ``points``: launch it
    inside a :class:`warp.Tape` with ``requires_grad`` arrays to obtain gradients.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_curvature: Optional output array of length ``len(points)`` to store the
            per-vertex curvature. If ``None``, a new array is allocated with the same
            ``requires_grad`` setting as ``points``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The ``out_curvature`` array, containing the angle defect at each vertex.

    Raises:
        ValueError: If ``out_curvature`` is provided but its dtype, device, or length
            does not match the expected output.
    """
    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3
    num_vertices = points.shape[0]

    if out_curvature is None:
        out_curvature = wp.empty(num_vertices, dtype=wp.float32, device=device, requires_grad=points.requires_grad)
    else:
        _validate_output(out_curvature, "out_curvature", num_vertices, wp.float32, device)

    # Angle defect: start each vertex at 2*pi, then the kernel subtracts the incident
    # interior angles with atomic_add.
    out_curvature.fill_(2.0 * math.pi)

    wp.launch(
        vertex_gaussian_curvature_kernel,
        dim=num_triangles,
        inputs=[points, indices],
        outputs=[out_curvature],
        device=device,
    )

    return out_curvature


def laplacian(
    points: wp.array[wp.vec3] | None,
    indices: wp.array[int],
    out_laplacian: BsrMatrix | None = None,
    *,
    weighting: LaplacianWeighting = LaplacianWeighting.COTANGENT,
    num_points: int | None = None,
    construction: str = "triplets",
    reuse_topology: bool = False,
    device: DeviceLike | None = None,
) -> BsrMatrix:
    """Assemble the Laplacian of a triangle mesh.

    Both weightings produce a symmetric positive semi-definite operator whose
    rows sum to zero, coupling exactly the vertex pairs that share an edge.
    They differ in what an edge is worth.

    With :attr:`LaplacianWeighting.COTANGENT`, each triangle contributes, for
    every edge, a symmetric off-diagonal pair weighted by the negated cotangent
    of the angle opposite that edge, and adds the same weight to the diagonal
    entries of the edge's two endpoints. This is the P1 finite-element stiffness
    matrix of the Laplacian bilinear form ``int(grad(u) . grad(v))``.

    With :attr:`LaplacianWeighting.UNIFORM`, every off-diagonal is ``-1`` and
    every diagonal is the vertex's number of neighbors, giving the graph
    Laplacian ``D - A`` of the mesh's edge graph. An edge counts once however
    many triangles share it, so a boundary edge weighs the same as an interior
    one. This depends only on ``indices``, so ``points`` may be ``None``.

    Note:
        The sign convention is the opposite of libigl's ``igl::cotmatrix``,
        which returns a negative semi-definite operator. Code ported from
        libigl needs ``laplacian() == -igl::cotmatrix()``.

    With cotangent weighting and the default ``construction="triplets"``, the
    operation is differentiable with respect to ``points``: launch it inside a
    :class:`warp.Tape` with ``points.requires_grad`` set to obtain gradients.
    The gradient of a triangle's contribution is undefined where its area
    vanishes, so degenerate triangles will produce non-finite gradients.
    ``construction="row_compress"`` is not differentiable. The uniform Laplacian
    is a function of connectivity alone, so its derivative with respect to
    ``points`` is zero rather than unavailable, and every construction policy is
    accepted.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`. May be
            ``None`` only with :attr:`LaplacianWeighting.UNIFORM`, which reads
            no positions.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        weighting: Edge weighting, as a :class:`LaplacianWeighting` member.
        num_points: Number of vertices, which fixes the size of the matrix.
            Needed only when ``points`` is ``None``; passing it alongside
            ``points`` is an error. When it is omitted and ``points`` is
            ``None``, the count is taken from ``out_laplacian`` if one is given,
            and otherwise from the largest entry of ``indices``, which costs a
            device readback and prevents CUDA graph capture.
        out_laplacian: Optional output matrix of shape
            ``(num_points, num_points)`` with scalar (1x1) blocks and
            :class:`warp.float32` coefficients. Any blocks it already holds are
            discarded. Its storage is reused when large enough, and grown
            otherwise, so repeated calls on a mesh of fixed topology stop
            reallocating the matrix after the first one. If ``None``, a new
            matrix is allocated.
        construction: How the sparsity pattern is built.

            ``"triplets"`` emits nine coordinate-oriented entries per triangle
            and lets :mod:`warp.sparse` sort and deduplicate them globally. Its
            values are accumulated with differentiable kernels, so this is the
            path to keep if gradients through assembly are ever wanted.

            ``"row_compress"`` instead reserves each vertex row three entries
            per incident triangle, writes the contributions directly into those
            spans, and compresses each row independently. It avoids the global
            sort and is substantially faster, but is not differentiable, and is
            rejected when ``points`` requires gradients and the weighting is
            cotangent.

            Both produce the same matrix. Ignored when ``reuse_topology`` is
            set, since no pattern is built in that case.
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
            and when ``points`` requires gradients but the requested cotangent
            combination cannot deliver them: ``construction="row_compress"``, or
            an ``out_laplacian`` that does not itself require gradients.
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
    num_triangles = indices.shape[0] // 3
    num_points = _resolve_num_points(points, indices, num_points, out_laplacian, device)

    if out_laplacian is not None:
        _validate_output_matrix(out_laplacian, "out_laplacian", (num_points, num_points), device)

    # Reject the combinations that would run happily and hand back zero
    # gradients, which is harder to notice than a failure. Uniform weighting is
    # exempt: it does not read positions, so a zero gradient is the true answer.
    if weighting == LaplacianWeighting.COTANGENT and points.requires_grad:
        if construction == "row_compress":
            raise ValueError(
                "`construction='row_compress'` is not differentiable, because it places entries with an "
                "atomic write cursor that the backward pass cannot replay. Use the default "
                "`construction='triplets'` to differentiate with respect to `points`."
            )
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

    nnz = num_triangles * 9

    # Refilling an existing pattern writes no topology, so it always goes
    # through the triplet path regardless of the construction policy.
    if construction == "row_compress" and not reuse_topology:
        counts = wp.zeros(num_points, dtype=wp.int32, device=device)
        wp.launch(vertex_row_counts, dim=num_triangles, inputs=[indices, 3, counts], device=device)

        if out_laplacian is None:
            target = bsr_zeros(num_points, num_points, wp.float32, device=device, row_capacity=counts, nnz_capacity=nnz)
        else:
            target = out_laplacian
            bsr_set_zero(target, topology="padded", row_capacity=counts, nnz_capacity=nnz)

        wp.launch(
            laplacian_row_entries,
            dim=num_triangles,
            inputs=[points, indices, target.offsets, target.row_counts, target.columns, target.values],
            device=device,
        )
        return bsr_compress(target, inplace=True, prune_numerical_zeros=False)

    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    columns = wp.empty(nnz, dtype=wp.int32, device=device)
    # The triplet values carry the gradient: warp.sparse propagates
    # ``requires_grad`` from here onto the assembled matrix and accumulates them
    # with differentiable kernels.
    values = wp.empty(nnz, dtype=wp.float32, device=device, requires_grad=points.requires_grad)

    wp.launch(
        laplacian_triplets,
        dim=num_triangles,
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
    construction: str = "triplets",
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


def oriented_bounding_box(
    points: wp.array(dtype=wp.vec3),
    measure_type: OBBMeasureType = OBBMeasureType.VOLUME,
    num_samples: int = 4096,
    *,
    include_axis_aligned: bool = True,
    include_pca: bool = True,
    device: DeviceLike | None = None,
) -> tuple[wp.transform, wp.vec3, float]:
    """Approximate an oriented bounding box (OBB) of a point set by sampling orientations.

    Candidate orientations are drawn from a Super-Fibonacci spiral, which spreads
    ``num_samples`` rotations near-uniformly over SO(3) (Alexa, "Super-Fibonacci
    Spirals: Fast, Low-Discrepancy Sampling of SO(3)", CVPR 2022). For each
    orientation the axis-aligned bounding box of the rotated points is evaluated, and
    the orientation whose box minimizes ``measure_type`` is returned. The result is an
    approximation whose quality improves with ``num_samples``; it is not guaranteed to
    be the globally optimal OBB.

    Two further candidates are appended by default. ``include_axis_aligned`` adds the
    identity rotation, which the spiral never contains exactly and which guarantees the
    result is no worse than the axis-aligned bounding box. ``include_pca`` adds the
    principal axes of the point set, obtained from the eigenvectors of its covariance
    matrix. The spiral resolves orientation only to roughly its sample spacing, and on
    a strongly elongated shape a few degrees of error costs a large amount of volume,
    so the principal axes are often a much better answer there than any sampled
    rotation. For a near-isotropic point set the covariance eigenvectors are close to
    degenerate and that candidate carries little information, leaving accuracy governed
    by ``num_samples`` alone. Both extras are cheap relative to ``num_samples``
    candidates.

    Args:
        points: Array of point positions of type :class:`warp.vec3`.
        measure_type: Quantity to minimize, as an :class:`OBBMeasureType` member:
            :attr:`OBBMeasureType.VOLUME` minimizes the box volume;
            :attr:`OBBMeasureType.SURFACE_AREA` minimizes the box surface area.
        num_samples: Number of candidate orientations to sample from the spiral. Larger
            values give a tighter box at higher cost.
        include_axis_aligned: Whether to also evaluate the identity rotation.
        include_pca: Whether to also evaluate the principal axes of ``points``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(transform, extents, measure)`` where ``transform`` is a
        :class:`warp.transform` mapping the box's local frame (axis-aligned and
        centered at the origin) into world space, ``extents`` is a :class:`warp.vec3`
        of the box's full side lengths, and ``measure`` is the achieved value of
        ``measure_type``.

    Note:
        This function synchronizes with the device to select the winning
        orientation, so it cannot be captured in a CUDA graph.

        With ``include_pca`` enabled the covariance matrix is accumulated with
        floating-point atomics, so the principal axes -- and therefore the result,
        when they win -- can vary in the last bits between runs. Pass
        ``include_pca=False`` for a bitwise reproducible result.

    Raises:
        ValueError: If ``num_samples`` is not a positive integer, or if ``points``
            is empty.
    """
    if num_samples < 1:
        raise ValueError(f"`num_samples` must be a positive integer, but got {num_samples}.")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("`points` must contain at least one point, but got an empty array.")

    measure_code = int(OBBMeasureType(measure_type))
    device = wp.get_device(device) if device is not None else points.device

    # Candidate orientations: the spiral first, then any extras in the trailing slots.
    num_candidates = num_samples + int(include_axis_aligned) + int(include_pca)
    rotations = wp.empty(num_candidates, dtype=wp.quat, device=device)

    wp.launch(
        oriented_bounding_box_samples_kernel,
        dim=num_samples,
        inputs=[num_samples],
        outputs=[rotations],
        device=device,
    )

    slot = num_samples
    if include_axis_aligned:
        wp.launch(oriented_bounding_box_identity_kernel, dim=1, inputs=[slot], outputs=[rotations], device=device)
        slot += 1

    if include_pca:
        point_sum = wp.zeros(1, dtype=wp.vec3, device=device)
        covariance = wp.zeros(1, dtype=wp.mat33, device=device)
        wp.launch(point_sum_kernel, dim=num_points, inputs=[points], outputs=[point_sum], device=device)
        wp.launch(
            point_covariance_kernel, dim=num_points, inputs=[points, point_sum], outputs=[covariance], device=device
        )
        wp.launch(
            oriented_bounding_box_pca_kernel, dim=1, inputs=[covariance, slot], outputs=[rotations], device=device
        )
        slot += 1

    # Threads per orientation. The candidate count alone is far too little parallelism
    # to fill a GPU, so the point loop is split as well; the cap keeps the atomic
    # contention on each orientation's bounds low.
    num_chunks = max(1, min(num_points, _OBB_POINT_CHUNKS))

    # Seeded to an empty interval so the atomic min/max reduction can only shrink it.
    min_bounds = wp.full(num_candidates, wp.vec3(math.inf, math.inf, math.inf), dtype=wp.vec3, device=device)
    max_bounds = wp.full(num_candidates, wp.vec3(-math.inf, -math.inf, -math.inf), dtype=wp.vec3, device=device)

    wp.launch(
        oriented_bounding_box_bounds_kernel,
        dim=(num_candidates, num_chunks),
        inputs=[points, rotations, num_chunks],
        outputs=[min_bounds, max_bounds],
        device=device,
    )

    # Per-orientation results; the search is not differentiable, so no grads are needed.
    measures = wp.empty(num_candidates, dtype=wp.float32, device=device)
    transforms = wp.empty(num_candidates, dtype=wp.transform, device=device)
    extents = wp.empty(num_candidates, dtype=wp.vec3, device=device)

    wp.launch(
        oriented_bounding_box_measure_kernel,
        dim=num_candidates,
        inputs=[rotations, min_bounds, max_bounds, measure_code],
        outputs=[measures, transforms, extents],
        device=device,
    )

    # Select the sampled orientation whose box minimizes the measure. The candidate
    # count is small, so the argmin runs on the host.
    measures_np = measures.numpy()
    best = int(np.argmin(measures_np))

    row = transforms.numpy()[best]
    best_transform = wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )
    ext = extents.numpy()[best]
    best_extents = wp.vec3(float(ext[0]), float(ext[1]), float(ext[2]))
    best_measure = float(measures_np[best])

    return best_transform, best_extents, best_measure


##########################################################################
## Swept volume (motion envelope) of animated rigid meshes
##########################################################################


class SweptVolumeSign(enum.IntEnum):
    """Method used to classify inside/outside when evaluating the per-mesh SDF.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel and compared against.
    """

    NORMAL = 0
    """Sign from the closest face's normal, via
    :func:`warp.mesh_query_point_sign_normal`. Fast; assumes each input mesh is
    watertight and consistently oriented."""

    WINDING_NUMBER = 1
    """Sign from the solid angle (generalized winding number), via
    :func:`warp.mesh_query_point_sign_winding_number`. Robust to non-watertight
    or inconsistently oriented meshes, but slower and requires every input mesh
    to be built with ``support_winding_number=True``."""


@wp.func
def swept_volume_sdf(
    p: wp.vec3,
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    max_dist: wp.float32,
    sign_mode: wp.int32,
) -> wp.float32:
    # Signed distance from world-space point ``p`` to the union of every input
    # mesh over every sampled pose, i.e. ``min_m min_s sdf_m(X[m, s]^-1 p)``.
    #
    # The motion is rigid, so instead of transforming the geometry we push the
    # query point back into each mesh's rest frame. One closest-point query per
    # (mesh, sample) therefore evaluates one pose, and a single call here folds
    # the whole space-time union into one scalar. Returns ``+max_dist`` when no
    # face lies within ``max_dist`` of ``p`` at any pose. Callable from within
    # user kernels.
    num_meshes = mesh_ids.shape[0]
    num_samples = transforms.shape[1]

    best = max_dist
    for m in range(num_meshes):
        mesh_id = mesh_ids[m]
        for s in range(num_samples):
            # Map the query point into the mesh's rest frame for this pose. The
            # map is rigid, so distances are preserved between the two frames.
            p_local = wp.transform_point(wp.transform_inverse(transforms[m, s]), p)

            if sign_mode == wp.int32(SweptVolumeSign.WINDING_NUMBER.value):
                query = wp.mesh_query_point_sign_winding_number(mesh_id, p_local, max_dist)
            else:
                query = wp.mesh_query_point_sign_normal(mesh_id, p_local, max_dist)

            if query.result:
                closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                dist = query.sign * wp.length(p_local - closest)
                best = wp.min(best, dist)

    return best


@wp.kernel
def swept_volume_field_kernel(
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    origin: wp.vec3,
    spacing: wp.vec3,
    max_dist: wp.float32,
    sign_mode: wp.int32,
    out_field: wp.array3d[wp.float32],
):
    i, j, k = wp.tid()
    p = origin + wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    out_field[i, j, k] = swept_volume_sdf(p, mesh_ids, transforms, max_dist, sign_mode)


def _swept_volume_transforms(transforms, device):
    """Normalize the ``transforms`` argument into a ``(num_meshes, num_samples)``
    :class:`warp.array2d` of :class:`warp.transform` on ``device``."""
    if isinstance(transforms, wp.array):
        if transforms.ndim != 2:
            raise ValueError(
                f"'transforms' must be a 2D array of shape (num_meshes, num_samples), got ndim {transforms.ndim}."
            )
        if not types_equal(transforms.dtype, wp.transform):
            raise TypeError(f"'transforms' must have dtype wp.transform, got {type_repr(transforms.dtype)}.")
        return transforms.to(device)

    # Accept nested Python sequences / NumPy arrays of shape (M, S, 7).
    arr = np.asarray(transforms, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 7:
        raise ValueError(
            "'transforms' must be a wp.array2d(dtype=wp.transform) or an array of "
            f"shape (num_meshes, num_samples, 7), got shape {arr.shape}."
        )
    return wp.array2d(arr, dtype=wp.transform, device=device)


def _swept_volume_bounds(meshes, transforms_np, margin):
    """World-space axis-aligned bounds of every mesh over every sampled pose.

    Transforms each mesh's rest-pose corner cloud by all of its poses and unions
    the results, then pads by ``margin`` on every side.
    """
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)

    for m, mesh in enumerate(meshes):
        pts = mesh.points.numpy().astype(np.float64)
        if pts.size == 0:
            continue
        # Rest-pose AABB corners are enough: a rigid map sends the AABB into a
        # box whose extent is bounded by transforming the 8 corners.
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        corners = np.array(
            [[[lo[0], hi[0]][a], [lo[1], hi[1]][b], [lo[2], hi[2]][c]] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        )

        for pose in transforms_np[m]:
            t = pose[:3]
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
            # Rotate the corners by the pose quaternion (x, y, z, w) and translate.
            rotated = _quat_rotate_np(np.array([qx, qy, qz, qw]), corners) + t
            lower = np.minimum(lower, rotated.min(axis=0))
            upper = np.maximum(upper, rotated.max(axis=0))

    if not np.all(np.isfinite(lower)):
        raise ValueError("Could not compute swept-volume bounds: all input meshes are empty.")

    lower -= margin
    upper += margin
    return lower, upper


def _quat_rotate_np(q, v):
    """Rotate row vectors ``v`` (..., 3) by quaternion ``q`` = (x, y, z, w)."""
    x, y, z, w = q
    # Rotation matrix from a unit quaternion.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return v @ R.T


def swept_volume_field(
    meshes,
    transforms,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    margin: float | None = None,
    max_dist: float | None = None,
    sign_mode: SweptVolumeSign = SweptVolumeSign.NORMAL,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.vec3, wp.vec3]:
    """Sample the swept-volume signed-distance field on a dense regular grid.

    Computes, at each grid node ``p``, the signed distance to the union of the
    input meshes over all sampled poses::

        D(p) = min_m min_s  sdf_m( X[m, s]^-1 p )

    by brute force over every (mesh, sample) pair ("dense time stamping"). The
    field is negative inside the swept volume and positive outside, so extracting
    its zero isosurface (see :func:`swept_volume`) yields the motion envelope.

    Because the poses are the *provided* samples, the field only accounts for the
    geometry at those instants; motion between consecutive samples is not
    conservatively bounded. Supply a sufficiently fine time sampling for the
    desired tolerance.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses, either as a
            :class:`warp.array2d` of :class:`warp.transform` with shape
            ``(num_meshes, num_samples)`` or as an array of shape
            ``(num_meshes, num_samples, 7)`` (translation ``xyz`` followed by
            quaternion ``xyzw``).
        voxel_size: Edge length of a grid cell in world units. Required unless
            ``resolution`` is given.
        resolution: Optional explicit node counts ``(nx, ny, nz)``. Overrides
            ``voxel_size`` for choosing the grid dimensions.
        margin: Padding added on every side of the swept bounding box, in world
            units. Defaults to twice ``voxel_size`` so the surface is not
            clipped, or to ``0`` when only ``resolution`` is given (pass an
            explicit ``margin`` to avoid clipping the surface at the boundary).
        max_dist: Maximum search distance for the closest-point queries. Nodes
            farther than this from every posed mesh are left at ``+max_dist``.
            Defaults to the grid's diagonal length so the field is valid
            everywhere.
        sign_mode: Inside/outside classification method (see
            :class:`SweptVolumeSign`).
        device: Device on which to build the field. Defaults to the device of
            the first mesh.

    Returns:
        A tuple ``(field, lower, upper)`` where ``field`` is a
        ``wp.array3d(dtype=wp.float32)`` of signed distances, and ``lower`` and
        ``upper`` are the :class:`warp.vec3` world coordinates that grid nodes
        ``(0, 0, 0)`` and ``(nx-1, ny-1, nz-1)`` map to.
    """
    if len(meshes) == 0:
        raise ValueError("'meshes' must contain at least one mesh.")
    if voxel_size is None and resolution is None:
        raise ValueError("Provide either 'voxel_size' or 'resolution'.")

    device = wp.get_device(device) if device is not None else meshes[0].device

    mesh_ids = wp.array([mesh.id for mesh in meshes], dtype=wp.uint64, device=device)
    transforms_wp = _swept_volume_transforms(transforms, device)
    if transforms_wp.shape[0] != len(meshes):
        raise ValueError(f"'transforms' has {transforms_wp.shape[0]} rows but there are {len(meshes)} meshes.")
    transforms_np = transforms_wp.numpy()

    if margin is None:
        margin = 2.0 * voxel_size if voxel_size is not None else 0.0

    lower, upper = _swept_volume_bounds(meshes, transforms_np, margin)
    extent = upper - lower

    if resolution is not None:
        dims = tuple(int(n) for n in resolution)
        if any(n < 2 for n in dims):
            raise ValueError(f"'resolution' must be at least 2 along each axis, got {dims}.")
    else:
        # Number of nodes = number of cells + 1; guarantee at least 2 nodes.
        dims = tuple(max(2, int(math.ceil(e / voxel_size)) + 1) for e in extent)

    # Snap the upper corner so the node spacing is exactly (extent / cells).
    spacing = np.array([extent[a] / (dims[a] - 1) for a in range(3)], dtype=np.float64)
    upper = lower + spacing * (np.array(dims) - 1)

    if max_dist is None:
        max_dist = float(np.linalg.norm(extent))

    origin = wp.vec3(float(lower[0]), float(lower[1]), float(lower[2]))
    spacing_v = wp.vec3(float(spacing[0]), float(spacing[1]), float(spacing[2]))

    field = wp.empty(dims, dtype=wp.float32, device=device)
    wp.launch(
        swept_volume_field_kernel,
        dim=dims,
        inputs=[mesh_ids, transforms_wp, origin, spacing_v, float(max_dist), int(sign_mode)],
        outputs=[field],
        device=device,
    )

    lower_v = wp.vec3(float(lower[0]), float(lower[1]), float(lower[2]))
    upper_v = wp.vec3(float(upper[0]), float(upper[1]), float(upper[2]))
    return field, lower_v, upper_v


def swept_volume(
    meshes,
    transforms,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    margin: float | None = None,
    max_dist: float | None = None,
    iso: float = 0.0,
    sign_mode: SweptVolumeSign = SweptVolumeSign.NORMAL,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array]:
    """Extract the swept volume (motion envelope) of animated rigid meshes.

    Samples the swept-volume signed-distance field with
    :func:`swept_volume_field` and extracts its ``iso`` isosurface with marching
    cubes, returning a single triangle mesh in world coordinates that encloses
    the union of every input mesh over every sampled pose.

    This is the dense-stamping baseline: the field is evaluated by brute force at
    the provided pose samples, with no root finding or narrow-band acceleration.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses; see
            :func:`swept_volume_field`.
        voxel_size: Edge length of a grid cell in world units. Required unless
            ``resolution`` is given.
        resolution: Optional explicit node counts ``(nx, ny, nz)``.
        margin: Padding added on every side of the swept bounding box, in world
            units. Defaults to twice ``voxel_size`` (or ``0`` when only
            ``resolution`` is given); see :func:`swept_volume_field`.
        max_dist: Maximum closest-point search distance; see
            :func:`swept_volume_field`.
        iso: Field level to extract. ``0.0`` traces the envelope through the
            sampled poses; a positive value dilates it outward. Marching cubes
            reconstructs the 1-Lipschitz field by linear interpolation, which at
            sharp convex features overestimates the field and pulls the surface
            inside the true one, so a stamped pose can poke through the ``0.0``
            isosurface by up to the grid's covering radius. Extracting at
            ``iso = 0.5 * hypot(hx, hy, hz)`` (the covering radius, ``sqrt(3)/2 *
            voxel_size`` for a cubic cell of the actual spacings ``hx, hy, hz``)
            guarantees every stamped pose stays enclosed.
        sign_mode: Inside/outside classification method (see
            :class:`SweptVolumeSign`).
        device: Device on which to run. Defaults to the device of the first mesh.

    Returns:
        A tuple ``(vertices, indices)`` where ``vertices`` is a
        ``wp.array(dtype=wp.vec3)`` in world coordinates and ``indices`` is a
        flat ``wp.array(dtype=wp.int32)`` with three entries per triangle.
    """
    field, lower, upper = swept_volume_field(
        meshes,
        transforms,
        voxel_size,
        resolution=resolution,
        margin=margin,
        max_dist=max_dist,
        sign_mode=sign_mode,
        device=device,
    )

    return MarchingCubes.extract_surface_marching_cubes(
        field,
        threshold=iso,
        domain_bounds_lower_corner=lower,
        domain_bounds_upper_corner=upper,
    )
