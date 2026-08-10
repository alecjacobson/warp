# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
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
def triangle_cotmatrix_coefficients(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
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


@wp.kernel(enable_backward=False)
def cotmatrix_triplets(
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

    c = triangle_cotmatrix_coefficients(
        points[v[0]],
        points[v[1]],
        points[v[2]],
    )

    # For each vertex k:
    #   - c[k] weights the edge opposite k
    #   - emit both symmetric off-diagonal entries
    #   - emit the diagonal entry for vertex k
    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        out = base + 3 * k

        rows[out + 0] = v[i]
        columns[out + 0] = v[j]
        values[out + 0] = c[k]

        rows[out + 1] = v[j]
        columns[out + 1] = v[i]
        values[out + 1] = c[k]

        rows[out + 2] = v[k]
        columns[out + 2] = v[k]
        values[out + 2] = -(c[i] + c[j])


@wp.kernel(enable_backward=False)
def cotmatrix_row_counts(indices: wp.array[int], counts: wp.array[int]):
    # Reserved storage for row ``v`` is three entries per incident triangle: the
    # triangle's contribution to that vertex's diagonal, plus its two
    # off-diagonals to the other two vertices. Counting incidences needs no edge
    # enumeration and makes no manifoldness assumption.
    tri = wp.tid()
    for k in range(3):
        wp.atomic_add(counts, indices[3 * tri + k], 3)


@wp.kernel(enable_backward=False)
def cotmatrix_row_entries(
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

    c = triangle_cotmatrix_coefficients(
        points[v[0]],
        points[v[1]],
        points[v[2]],
    )

    # Same contributions as ``cotmatrix_triplets``, but grouped by the row they
    # land in so that they can be written straight into that row's reserved
    # span. ``cursors`` starts at zero and doubles as the per-row write cursor
    # and the final active count for each row.
    for k in range(3):
        i = (k + 1) % 3
        j = (k + 2) % 3
        row = v[k]
        base = offsets[row] + wp.atomic_add(cursors, row, 3)

        columns[base + 0] = row
        values[base + 0] = -(c[i] + c[j])

        columns[base + 1] = v[j]
        values[base + 1] = c[i]

        columns[base + 2] = v[i]
        values[base + 2] = c[j]


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


def cotmatrix(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    out_cotmatrix: BsrMatrix | None = None,
    *,
    construction: str = "triplets",
    reuse_topology: bool = False,
    device: DeviceLike | None = None,
) -> BsrMatrix:
    """Assemble the cotangent Laplacian of a triangle mesh.

    Each triangle contributes, for every edge, a symmetric off-diagonal pair
    weighted by the cotangent of the angle opposite that edge, and subtracts the
    same weight from the diagonal entries of the edge's two endpoints. This
    matches the convention of libigl's ``igl::cotmatrix``: the result is symmetric
    and negative semi-definite, with each row summing to zero.

    This operation is not differentiable with respect to ``points``.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Flat array of triangle vertex indices, with three consecutive
            entries per triangle (length ``3 * num_triangles``).
        out_cotmatrix: Optional output matrix of shape
            ``(len(points), len(points))`` with scalar (1x1) blocks and
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
            sort and is substantially faster, but compresses in place and is
            not differentiable.

            Both produce the same matrix. Ignored when ``reuse_topology`` is
            set, since no pattern is built in that case.
        reuse_topology: If ``True``, keep the sparsity pattern ``out_cotmatrix``
            already holds and overwrite only its coefficients. Requires
            ``out_cotmatrix``. This skips the sort and deduplication that
            dominate assembly, and is several times faster, but it is only
            correct when that pattern already covers every vertex pair the mesh
            couples: contributions landing outside it are silently dropped. Pass
            it a matrix returned by an earlier call on the same ``indices``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A square sparse matrix of shape ``(len(points), len(points))`` with
        scalar (1x1) blocks, which is ``out_cotmatrix`` when it is provided.

    Raises:
        ValueError: If ``out_cotmatrix`` is provided but its scalar type, block
            shape, device, or shape does not match the expected output, if
            ``reuse_topology`` is set without a populated ``out_cotmatrix``, or
            if ``construction`` is not a recognized policy.
    """
    if construction not in ("triplets", "row_compress"):
        raise ValueError(f"Unsupported `construction` policy: {construction!r}. Expected 'triplets' or 'row_compress'.")

    device = wp.get_device(device) if device is not None else points.device
    num_triangles = indices.shape[0] // 3
    num_points = points.shape[0]

    if out_cotmatrix is not None:
        _validate_output_matrix(out_cotmatrix, "out_cotmatrix", (num_points, num_points), device)

    if reuse_topology:
        if out_cotmatrix is None:
            raise ValueError("`reuse_topology` requires `out_cotmatrix`, whose sparsity pattern it reuses.")
        # An empty matrix has no pattern to reuse, so every contribution would be
        # dropped and the result would be silently zero. Catching that here costs
        # nothing, unlike verifying that a non-empty pattern is the right one,
        # which would need a device readback.
        if out_cotmatrix.nnz == 0:
            raise ValueError(
                "`reuse_topology` requires `out_cotmatrix` to already hold the sparsity pattern of the "
                "cotangent Laplacian, but it is empty. Call `cotmatrix()` without `reuse_topology` first."
            )

    nnz = num_triangles * 9

    # Refilling an existing pattern writes no topology, so it always goes
    # through the triplet path regardless of the construction policy.
    if construction == "row_compress" and not reuse_topology:
        counts = wp.zeros(num_points, dtype=wp.int32, device=device)
        wp.launch(cotmatrix_row_counts, dim=num_triangles, inputs=[indices, counts], device=device)

        if out_cotmatrix is None:
            target = bsr_zeros(num_points, num_points, wp.float32, device=device, row_capacity=counts, nnz_capacity=nnz)
        else:
            target = out_cotmatrix
            bsr_set_zero(target, topology="padded", row_capacity=counts, nnz_capacity=nnz)

        wp.launch(
            cotmatrix_row_entries,
            dim=num_triangles,
            inputs=[points, indices, target.offsets, target.row_counts, target.columns, target.values],
            device=device,
        )
        return bsr_compress(target, inplace=True)

    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    columns = wp.empty(nnz, dtype=wp.int32, device=device)
    values = wp.empty(nnz, dtype=wp.float32, device=device)

    wp.launch(
        cotmatrix_triplets,
        dim=num_triangles,
        inputs=[points, indices, rows, columns, values],
        device=device,
    )

    if out_cotmatrix is None:
        return bsr_from_triplets(num_points, num_points, rows, columns, values)

    bsr_set_from_triplets(
        out_cotmatrix,
        rows,
        columns,
        values,
        topology="masked" if reuse_topology else "compact",
    )
    return out_cotmatrix


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
