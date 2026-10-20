# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangle intersection predicates implemented as Warp functions.

Adapted from Tomas Möller's division-free triangle intersection test, as used by
``warp/native/intersect_tri.h``. The original implementation has this license:

Copyright 2020 Tomas Akenine-Möller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Any

import warp as wp
from warp._src.context import register_api_function


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
def _plane_distance(n: Any, p: Any, offset: Any):
    d = (n[0] * p[0] + n[1] * p[1]) + n[2] * p[2] - offset
    # The native EPSILON literal is double precision, including for float inputs.
    if wp.float64(wp.abs(d)) < wp.float64(1.0e-6):
        d = type(d)(0.0)
    return d


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
        return 1
    if _edge_edge(v0, v1, u1, u2, i, j):
        return 1
    if _edge_edge(v0, v1, u2, u0, i, j):
        return 1
    if _edge_edge(v1, v2, u0, u1, i, j):
        return 1
    if _edge_edge(v1, v2, u1, u2, i, j):
        return 1
    if _edge_edge(v1, v2, u2, u0, i, j):
        return 1
    if _edge_edge(v2, v0, u0, u1, i, j):
        return 1
    if _edge_edge(v2, v0, u1, u2, i, j):
        return 1
    if _edge_edge(v2, v0, u2, u0, i, j):
        return 1
    if _point_in_triangle(v0, u0, u1, u2, i, j):
        return 1
    if _point_in_triangle(u0, v0, v1, v2, i, j):
        return 1
    return 0


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
def _intersect_tri_tri(v0: Any, v1: Any, v2: Any, u0: Any, u1: Any, u2: Any):
    zero = type(v0[0])(0.0)
    n1 = wp.cross(v1 - v0, v2 - v0)
    offset1 = (n1[0] * v0[0] + n1[1] * v0[1]) + n1[2] * v0[2]
    du0 = _plane_distance(n1, u0, offset1)
    du1 = _plane_distance(n1, u1, offset1)
    du2 = _plane_distance(n1, u2, offset1)
    if du0 * du1 > zero and du0 * du2 > zero:
        return 0

    n2 = wp.cross(u1 - u0, u2 - u0)
    offset2 = (n2[0] * u0[0] + n2[1] * u0[1]) + n2[2] * u0[2]
    dv0 = _plane_distance(n2, v0, offset2)
    dv1 = _plane_distance(n2, v1, offset2)
    dv2 = _plane_distance(n2, v2, offset2)
    if dv0 * dv1 > zero and dv0 * dv2 > zero:
        return 0

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
        return 0
    return 1


@wp.func(inline=True)
def intersect_tri_tri(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, u0: wp.vec3, u1: wp.vec3, u2: wp.vec3) -> int:
    """Test whether two triangles intersect using a pure Warp implementation of Möller's method.

    All vertices must have the same type (``vec3`` or ``vec3d``) and be in the same
    coordinate space. Coplanar overlap counts as an intersection. Like
    :func:`warp.intersect_tri_tri`, plane equation values with absolute magnitude
    below ``1e-6`` are treated as zero. This absolute tolerance is applied to
    unnormalized plane equations and therefore depends on triangle scale.

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

    Returns:
        ``1`` if the triangles intersect, or ``0`` otherwise.
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2)


@wp.func(inline=True)
def intersect_tri_tri(v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, u0: wp.vec3d, u1: wp.vec3d, u2: wp.vec3d) -> int:
    """Double-precision overload of :func:`warp.geometry.intersect_tri_tri`.

    Use the same coordinate space for all six vertices. Return ``1`` for an
    intersection, including coplanar overlap, or ``0`` otherwise. The plane
    tolerance and limitations are the same as for the single-precision overload.
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2)


# Both overloads share the public documentation; the last @wp.func registration
# otherwise replaces __doc__ with the second Python function's shorter docstring.
intersect_tri_tri.__doc__ = intersect_tri_tri.doc
register_api_function(intersect_tri_tri, module="warp.geometry", differentiable=False)
