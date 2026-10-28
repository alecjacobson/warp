# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric intersection predicates implemented as Warp functions."""

# The triangle-triangle test is adapted from Tomas Möller's division-free test,
# as used by ``warp/native/intersect_tri.h``. That implementation has this license:
#
# Copyright 2020 Tomas Akenine-Möller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Any

import warp as wp
from warp._src.context import register_api_function

_DEFAULT_INTERSECTION_TOL = wp.float64(1.0e-6)


@wp.func(inline=True)
def _component(v: Any, index: int):
    # Dynamic vector subscripts force CUDA to materialize local arrays. Select
    # from fixed components instead so projection coordinates stay in registers.
    if index == 0:
        return v[0]
    if index == 1:
        return v[1]
    return v[2]


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float32, tol: wp.float32):
    if wp.abs(d) < tol:
        return wp.float32(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float32, tol: wp.float64):
    # Preserve the native double cutoff without float64 work for default calls.
    # float32(1e-6) is the largest float32 below the native double EPSILON.
    if tol == wp.float64(1.0e-6):
        if wp.abs(d) <= wp.float32(1.0e-6):
            return wp.float32(0.0)
    elif wp.float64(wp.abs(d)) < tol:
        return wp.float32(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float64, tol: wp.float64):
    if wp.abs(d) < tol:
        return wp.float64(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float64, tol: wp.float32):
    return _snap_plane_distance(d, wp.float64(tol))


@wp.func(inline=True)
def _plane_distance(n: Any, p: Any, offset: Any, tol: Any):
    d = (n[0] * p[0] + n[1] * p[1]) + n[2] * p[2] - offset
    return _snap_plane_distance(d, tol)


@wp.func(inline=True)
def _edge_edge(v0: Any, v1: Any, u0: Any, u1: Any, i: int, j: int):
    ax = _component(v1, i) - _component(v0, i)
    ay = _component(v1, j) - _component(v0, j)
    bx = _component(u0, i) - _component(u1, i)
    by = _component(u0, j) - _component(u1, j)
    cx = _component(v0, i) - _component(u0, i)
    cy = _component(v0, j) - _component(u0, j)
    f = ay * bx - ax * by
    d = by * cx - bx * cy
    zero = type(f)(0.0)
    if (f > zero and d >= zero and d <= f) or (f < zero and d <= zero and d >= f):
        e = ax * cy - ay * cx
        if f > zero:
            return e >= zero and e <= f
        return e <= zero and e >= f
    return False


@wp.func(inline=True)
def _point_in_triangle(p: Any, u0: Any, u1: Any, u2: Any, i: int, j: int):
    a = _component(u1, j) - _component(u0, j)
    b = -(_component(u1, i) - _component(u0, i))
    c = -a * _component(u0, i) - b * _component(u0, j)
    d0 = a * _component(p, i) + b * _component(p, j) + c
    a = _component(u2, j) - _component(u1, j)
    b = -(_component(u2, i) - _component(u1, i))
    c = -a * _component(u1, i) - b * _component(u1, j)
    d1 = a * _component(p, i) + b * _component(p, j) + c
    a = _component(u0, j) - _component(u2, j)
    b = -(_component(u0, i) - _component(u2, i))
    c = -a * _component(u2, i) - b * _component(u2, j)
    d2 = a * _component(p, i) + b * _component(p, j) + c
    zero = type(d0)(0.0)
    return d0 * d1 > zero and d0 * d2 > zero


@wp.func(inline=True)
def _coplanar_tri_tri(n: Any, v0: Any, v1: Any, v2: Any, u0: Any, u1: Any, u2: Any):
    # Project onto the coordinate plane with the largest triangle area.
    a = wp.abs(n)
    i = int(0)
    j = int(1)
    if a[0] > a[1]:
        if a[0] > a[2]:
            i = 1
            j = 2
    elif a[2] <= a[1]:
        j = 2

    if _edge_edge(v0, v1, u0, u1, i, j):
        return True
    if _edge_edge(v0, v1, u1, u2, i, j):
        return True
    if _edge_edge(v0, v1, u2, u0, i, j):
        return True
    if _edge_edge(v1, v2, u0, u1, i, j):
        return True
    if _edge_edge(v1, v2, u1, u2, i, j):
        return True
    if _edge_edge(v1, v2, u2, u0, i, j):
        return True
    if _edge_edge(v2, v0, u0, u1, i, j):
        return True
    if _edge_edge(v2, v0, u1, u2, i, j):
        return True
    if _edge_edge(v2, v0, u2, u0, i, j):
        return True
    if _point_in_triangle(v0, u0, u1, u2, i, j):
        return True
    if _point_in_triangle(u0, v0, v1, v2, i, j):
        return True
    return False


@wp.func(inline=True)
def _interval(v0: Any, v1: Any, v2: Any, d0: Any, d1: Any, d2: Any):
    # Choose the vertex opposite the two on the same side of the other plane.
    # The caller handles the all-zero (coplanar) distances before entering here.
    # Select the inputs first, then evaluate the coefficients once. Keeping
    # arithmetic outside divergent branches reduces double-precision work.
    zero = type(d0)(0.0)
    a = v0
    b = v1
    c = v2
    d = d0
    db = d1
    dc = d2
    if d0 * d1 > zero:
        a = v2
        b = v0
        c = v1
        d = d2
        db = d0
        dc = d1
    elif d0 * d2 > zero:
        a = v1
        b = v0
        c = v2
        d = d1
        db = d0
        dc = d2
    elif not (d1 * d2 > zero or d0 != zero):
        if d1 != zero:
            a = v1
            b = v0
            c = v2
            d = d1
            db = d0
            dc = d2
        else:
            a = v2
            b = v0
            c = v1
            d = d2
            db = d0
            dc = d1
    return a, (b - a) * d, (c - a) * d, d - db, d - dc


@wp.func(inline=True)
def _intersect_tri_tri(v0: Any, v1: Any, v2: Any, u0: Any, u1: Any, u2: Any, tol: Any):
    zero = type(v0[0])(0.0)
    n1 = wp.cross(v1 - v0, v2 - v0)
    offset1 = (n1[0] * v0[0] + n1[1] * v0[1]) + n1[2] * v0[2]
    du0 = _plane_distance(n1, u0, offset1, tol)
    du1 = _plane_distance(n1, u1, offset1, tol)
    du2 = _plane_distance(n1, u2, offset1, tol)
    if du0 * du1 > zero and du0 * du2 > zero:
        return False

    n2 = wp.cross(u1 - u0, u2 - u0)
    offset2 = (n2[0] * u0[0] + n2[1] * u0[1]) + n2[2] * u0[2]
    dv0 = _plane_distance(n2, v0, offset2, tol)
    dv1 = _plane_distance(n2, v1, offset2, tol)
    dv2 = _plane_distance(n2, v2, offset2, tol)
    if dv0 * dv1 > zero and dv0 * dv2 > zero:
        return False

    if dv0 == zero or du0 == zero:
        # Most noncoplanar pairs avoid the remaining four zero comparisons.
        if (dv0 == zero and dv1 == zero and dv2 == zero) or (du0 == zero and du1 == zero and du2 == zero):
            return _coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    direction = wp.abs(wp.cross(n1, n2))
    index = int(0)
    largest = direction[0]
    if direction[1] > largest:
        index = 1
        largest = direction[1]
    if direction[2] > largest:
        index = 2

    a, b, c, x0, x1 = _interval(_component(v0, index), _component(v1, index), _component(v2, index), dv0, dv1, dv2)
    d, e, f, y0, y1 = _interval(_component(u0, index), _component(u1, index), _component(u2, index), du0, du1, du2)
    xx = x0 * x1
    yy = y0 * y1
    xxyy = xx * yy
    p0 = a * xxyy + b * x1 * yy
    p1 = a * xxyy + c * x0 * yy
    q0 = d * xxyy + e * xx * y1
    q1 = d * xxyy + f * xx * y0
    if wp.max(p0, p1) < wp.min(q0, q1) or wp.max(q0, q1) < wp.min(p0, p1):
        return False
    return True


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Test whether two triangles intersect.

    Overloads accept ``vec3`` or ``vec3d`` vertices, each with a float32 or
    float64 ``tol``. All vertices must have the same type and be in the same
    coordinate space. Coplanar overlap counts as an intersection.

    The test follows Möller's division-free method. A vertex's plane value is ``d = dot(n, p) - dot(n, a)``, where
    ``n = cross(b - a, c - a)`` is the other triangle's unnormalized normal.
    Values satisfying ``abs(d) < tol`` are snapped to zero. Consequently,
    ``tol`` is not a world-space distance: for a nondegenerate triangle its
    corresponding distance threshold is ``tol / length(n)``. Uniformly scaling
    all coordinates by ``s`` scales plane values by ``abs(s)**3``; scale ``tol``
    by the same factor to preserve the classification, apart from roundoff.
    Interval and coplanar edge tests receive no additional tolerance.

    The omitted tolerance matches :func:`warp.intersect_tri_tri` and is stored
    in float64 even for ``vec3`` inputs. Explicit float32 tolerances use their
    rounded float32 values. To spell the exact default explicitly, use
    ``tol=wp.float64(1e-6)``; an ordinary kernel literal ``tol=1e-6`` is float32
    and differs immediately at the cutoff.

    This is a floating-point predicate, not an exact orientation test. Results
    near degeneracy may be inaccurate; use ``vec3d`` for greater precision.
    Zero-area triangles are not supported. The discrete result is not
    differentiable.

    Args:
        v0: First vertex of the first triangle.
        v1: Second vertex of the first triangle.
        v2: Third vertex of the first triangle.
        u0: First vertex of the second triangle.
        u1: Second vertex of the second triangle.
        u2: Third vertex of the second triangle.
        tol: Nonnegative finite plane-equation tolerance, in float32 or float64.
            Zero disables snapping but retains exact floating-point coplanarity
            and boundary checks. Increasing it does not dilate triangle edges.

    Returns:
        ``True`` if the triangles intersect, or ``False`` otherwise.

    Example:
        Given triangle vertices inside a Warp kernel or function::

            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2)
            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=1.0e-5)
            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=0.0)
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3d,
    v1: wp.vec3d,
    v2: wp.vec3d,
    u0: wp.vec3d,
    u1: wp.vec3d,
    u2: wp.vec3d,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_tri_tri`.

    Use the same coordinate space for all six vertices. Return ``True`` for an
    intersection, including coplanar overlap, or ``False`` otherwise. The plane
    tolerance and limitations are the same as for the single-precision overload.
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, u0: wp.vec3, u1: wp.vec3, u2: wp.vec3, tol: wp.float32
) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_tri_tri`."""
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, u0: wp.vec3d, u1: wp.vec3d, u2: wp.vec3d, tol: wp.float32
) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_tri_tri`."""
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


intersect_tri_tri.__doc__ = intersect_tri_tri.doc
register_api_function(intersect_tri_tri, module="warp.geometry", differentiable=False)


@wp.func(inline=True)
def _orient_projected(a: Any, b: Any, p: Any, i: int, j: int):
    return (_component(b, i) - _component(a, i)) * (_component(p, j) - _component(a, j)) - (
        _component(b, j) - _component(a, j)
    ) * (_component(p, i) - _component(a, i))


@wp.func(inline=True)
def _intersect_seg_tri_projected(p0: Any, p1: Any, v0: Any, v1: Any, v2: Any, n: Any, d0: Any, d1: Any):
    zero = type(p0[0])(0.0)
    one = type(p0[0])(1.0)
    a = wp.abs(n)
    # Project onto the coordinate plane with the largest triangle area.
    i = int(0)
    j = int(1)
    if a[0] > a[1]:
        if a[0] > a[2]:
            i = 1
            j = 2
    elif a[2] <= a[1]:
        j = 2
    area = _orient_projected(v0, v1, v2, i, j)
    if area == zero:
        return False

    # A single endpoint on the plane reduces the intersection to a point.
    if d0 != zero:
        p0 = p1
    elif d1 != zero:
        p1 = p0

    # Separating-axis tests need no intersection parameters or divisions. The
    # triangle edge normals and the segment normal form a complete set in 2D.
    orientation = one
    if area < zero:
        orientation = -one
    a0 = orientation * _orient_projected(v0, v1, p0, i, j)
    a1 = orientation * _orient_projected(v0, v1, p1, i, j)
    if a0 < zero and a1 < zero:
        return False
    b0 = orientation * _orient_projected(v1, v2, p0, i, j)
    b1 = orientation * _orient_projected(v1, v2, p1, i, j)
    if b0 < zero and b1 < zero:
        return False
    c0 = orientation * _orient_projected(v2, v0, p0, i, j)
    c1 = orientation * _orient_projected(v2, v0, p1, i, j)
    if c0 < zero and c1 < zero:
        return False

    # The segment's supporting line supplies the final separating axis.
    s0 = _orient_projected(p0, p1, v0, i, j)
    s1 = _orient_projected(p0, p1, v1, i, j)
    s2 = _orient_projected(p0, p1, v2, i, j)
    return not ((s0 > zero and s1 > zero and s2 > zero) or (s0 < zero and s1 < zero and s2 < zero))


@wp.func(inline=True)
def _intersect_seg_tri(p0: Any, p1: Any, v0: Any, v1: Any, v2: Any, tol: Any):
    zero = type(p0[0])(0.0)
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    offset = (n[0] * v0[0] + n[1] * v0[1]) + n[2] * v0[2]
    d0 = _plane_distance(n, p0, offset, tol)
    d1 = _plane_distance(n, p1, offset, tol)
    if d0 == zero or d1 == zero:
        # Preserve projection-based snapping for endpoint and coplanar contact.
        return _intersect_seg_tri_projected(p0, p1, v0, v1, v2, n, d0, d1)
    if (d0 > zero) == (d1 > zero):
        return False

    # Moller-Trumbore barycentric numerators. Opposite plane sides already
    # establish 0 < t < 1, so neither t nor a reciprocal is needed. Compare
    # against the determinant directly and normalize its sign for both windings.
    direction = p0 - p1
    det = wp.dot(direction, n)
    e = wp.cross(direction, p0 - v0)
    v = wp.dot(e2, e)
    w = -wp.dot(e1, e)
    if det < zero:
        det = -det
        v = -v
        w = -w
    return det > zero and v >= zero and w >= zero and v + w <= det


@wp.func(inline=True)
def intersect_seg_tri(
    p0: wp.vec3, p1: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, tol: wp.float64 = _DEFAULT_INTERSECTION_TOL
) -> bool:
    """Test whether a closed segment intersects a triangle.

    Overloads accept ``vec3`` or ``vec3d`` points, each with a float32 or float64
    ``tol``. All points must have the same type and be in the same coordinate
    space. Endpoint, edge, vertex, and coplanar contact count as
    intersections. A zero-length segment is treated as a point. A zero-area
    triangle returns ``False``.

    With ``n = cross(v1 - v0, v2 - v0)``, endpoint plane values
    ``d = dot(n, p) - dot(n, v0)`` satisfying ``abs(d) < tol`` are snapped to
    zero. Such endpoints are treated as lying in the triangle's plane; the
    dominant-axis projection determines their position within that plane.
    Barycentric and projected edge tests receive no additional tolerance.

    The normal is unnormalized: the corresponding world-space distance
    threshold is ``tol / length(n)``, so ``tol`` depends on triangle area.
    Under uniform coordinate scaling by ``s``, scale ``tol`` by ``abs(s)**3``
    to preserve the classification, apart from roundoff. See
    :func:`warp.geometry.intersect_tri_tri` for the default tolerance's float64
    representation and the rounding of explicit float32 tolerances.

    This floating-point predicate is not exact and may be inaccurate near
    degeneracy. Use ``vec3d`` for greater precision. The result is not
    differentiable.

    Args:
        p0: First endpoint of the segment.
        p1: Second endpoint of the segment.
        v0: First vertex of the triangle.
        v1: Second vertex of the triangle.
        v2: Third vertex of the triangle.
        tol: Nonnegative finite plane-equation tolerance, in float32 or float64.
            Zero disables snapping but retains exact floating-point coplanarity
            and boundary checks. Increasing it does not dilate triangle edges.

    Returns:
        ``True`` if the segment intersects the triangle, or ``False`` otherwise.

    Example:
        Given segment and triangle vertices inside a Warp kernel or function::

            hit = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol=0.0)
            hit = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol=user_tolerance)

        ``user_tolerance`` may be a runtime float32 or float64 scalar.
    """
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(
    p0: wp.vec3d, p1: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, tol: wp.float64 = _DEFAULT_INTERSECTION_TOL
) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_seg_tri`.

    Return ``True`` for an intersection, including boundary and coplanar contact.
    The tolerance and limitations are the same as for the single-precision overload.
    """
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(p0: wp.vec3, p1: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, tol: wp.float32) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_seg_tri`."""
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(p0: wp.vec3d, p1: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, tol: wp.float32) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_seg_tri`."""
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


intersect_seg_tri.__doc__ = intersect_seg_tri.doc
register_api_function(intersect_seg_tri, module="warp.geometry", differentiable=False)
