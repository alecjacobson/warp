# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import *


@wp.kernel
def intersect_tri(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    result_single_precision: wp.array[int],
    result_double_precision: wp.array[int],
):
    result_single_precision[0] = wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2)
    result_single_precision[1] = int(wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2))

    u0_d = wp.vec3d(u0)
    u1_d = wp.vec3d(u1)
    u2_d = wp.vec3d(u2)

    v0_d = wp.vec3d(v0)
    v1_d = wp.vec3d(v1)
    v2_d = wp.vec3d(v2)

    result_double_precision[0] = wp.intersect_tri_tri(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d)
    result_double_precision[1] = int(wp.geometry.intersect_tri_tri(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d))


def test_intersect_tri(test, device):
    points_intersect = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.5, -0.5, 0.0),
        wp.vec3(0.5, -0.5, 1.0),
        wp.vec3(0.5, 0.5, 0.0),
    ]

    points_separated = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(-0.5, -0.5, 0.0),
        wp.vec3(-0.5, -0.5, 1.0),
        wp.vec3(-0.5, 0.5, 0.0),
    ]

    result_single_precision = wp.zeros(2, dtype=int, device=device)
    result_double_precision = wp.zeros(2, dtype=int, device=device)

    wp.launch(
        intersect_tri,
        dim=1,
        inputs=[*points_intersect, result_single_precision, result_double_precision],
        device=device,
    )
    assert_np_equal(result_single_precision.numpy(), np.array([1, 1]))
    assert_np_equal(result_double_precision.numpy(), np.array([1, 1]))

    wp.launch(
        intersect_tri,
        dim=1,
        inputs=[*points_separated, result_single_precision, result_double_precision],
        device=device,
    )
    assert_np_equal(result_single_precision.numpy(), np.array([0, 0]))
    assert_np_equal(result_double_precision.numpy(), np.array([0, 0]))


@wp.kernel
def compare_triangles(points: wp.array2d[Any], native: wp.array[int], geometry: wp.array[bool]):
    i = wp.tid()
    native[i] = wp.intersect_tri_tri(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5])
    geometry[i] = wp.geometry.intersect_tri_tri(
        points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]
    )


def check_triangles(test, device, points, expected=None):
    for dtype in (wp.vec3, wp.vec3d):
        with test.subTest(dtype=dtype):
            inputs = wp.array(points, dtype=dtype, device=device)
            native = wp.empty(len(points), dtype=int, device=device)
            geometry = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(compare_triangles, dim=len(points), inputs=[inputs, native, geometry], device=device)
            np.testing.assert_array_equal(geometry.numpy(), native.numpy())
            if expected is not None:
                np.testing.assert_array_equal(geometry.numpy(), expected)


def test_geometry_tri_random(test, device):
    rng = np.random.default_rng(4829)
    points = rng.uniform(-1.0, 1.0, (10000, 6, 3))
    check_triangles(test, device, points)
    # Exercise all three dominant projection axes and both triangle windings.
    for axis in range(3):
        points[:, :, axis] = 0.25
        check_triangles(test, device, points)
        points = rng.uniform(-1.0, 1.0, (10000, 6, 3))


def test_geometry_tri_contacts(test, device):
    first = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    second = (
        ([[0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], 1),
        ([[0.1, 0.1, 0], [0.6, 0.1, 0], [0.1, 0.6, 0]], 1),
        ([[2, 0, 0], [3, 0, 0], [2, 1, 0]], 0),
        ([[1, 0, 0], [2, 0, 1], [2, 1, 0]], 1),
        ([[1, 0, 0], [0, 1, 0], [1, 1, 1]], 1),
        (first, 1),
        ([[0, 0, 1], [1, 0, 1], [0, 1, 1]], 0),
    )
    points = []
    expected = []
    # Check vertex permutations, swapped triangles, and all coordinate projections.
    for other, hit in second:
        for rotation in range(3):
            for reverse in (False, True):
                a = np.roll(first, rotation, axis=0)
                b = np.asarray(other)[::-1] if reverse else np.asarray(other)
                for axis in range(3):
                    pair = np.roll(np.concatenate((a, b)), axis, axis=1)
                    points.extend((pair, np.concatenate((pair[3:], pair[:3]))))
                    expected.extend((hit, hit))
    check_triangles(test, device, np.asarray(points), expected)


def test_geometry_tri_tolerance(test, device):
    first = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    points = []
    # Match the native tolerance, including the float32 value just below 1e-6.
    for height in (0.0, 0.5e-6, float(np.float32(1e-6)), 1e-6, 2e-6, -0.5e-6, -2e-6):
        other = first * 0.5 + (0.1, 0.1, height)
        points.append(np.concatenate((first, other)))
    # Almost parallel planes and thin triangles at several coordinate scales.
    for scale in (0.01, 1.0, 100.0):
        for height in (1e-7, 1e-5, 1e-3):
            other = np.array([[0.2, 0.2, -height], [0.2, 0.8, height], [0.8, 0.2, height]])
            points.append(np.concatenate((first, other)) * scale)
    check_triangles(test, device, np.asarray(points))


@wp.kernel
def intersect_segments(points: wp.array2d[Any], hits: wp.array[bool]):
    i = wp.tid()
    hits[i] = wp.geometry.intersect_seg_tri(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])


def check_segments(test, device, points, expected):
    for dtype in (wp.vec3, wp.vec3d):
        with test.subTest(dtype=dtype):
            data = wp.array(points, dtype=dtype, device=device)
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(intersect_segments, dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


def segment_triangle_reference(points):
    """Independent NumPy Möller-Trumbore reference for noncoplanar fixtures."""
    p0, p1, a, b, c = np.moveaxis(np.asarray(points, dtype=np.float64), 1, 0)
    direction = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    normal = np.cross(e1, e2)
    distances = np.einsum("ij,ij->i", normal, p0 - a)
    # This reference intentionally excludes the coplanar case, covered separately.
    if np.any((np.abs(det) < 1.0e-12) & (np.abs(distances) < 1.0e-6)):
        raise ValueError("The NumPy reference requires noncoplanar segments.")
    inverse = np.divide(1.0, det, out=np.zeros_like(det), where=det != 0.0)
    s = p0 - a
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inverse * np.einsum("ij,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", e2, q)
    return (det != 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)


def test_geometry_seg_contacts(test, device):
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cases = (
        # Interior crossing, miss, and an infinite-line hit beyond the segment.
        ((0.25, 0.25, -1), (0.25, 0.25, 1), True),
        ((0.75, 0.75, -1), (0.75, 0.75, 1), False),
        ((0.25, 0.25, 1), (0.25, 0.25, 2), False),
        # Endpoint, edge, and vertex contact from outside the plane.
        ((0.25, 0.25, 0), (0.25, 0.25, 1), True),
        ((0.5, 0.5, -1), (0.5, 0.5, 1), True),
        ((0, 0, -1), (0, 0, 1), True),
        # Coplanar containment, crossing with both endpoints outside, and miss.
        ((0.125, 0.125, 0), (0.5, 0.125, 0), True),
        ((-1, 0.25, 0), (2, 0.25, 0), True),
        ((1, 1, 0), (2, 2, 0), False),
        # Collinear overlap, disjoint collinear edges, and vertex-only contact.
        ((-1, 0, 0), (2, 0, 0), True),
        ((-2, 0, 0), (-1, 0, 0), False),
        ((-1, 0, 0), (0, 0, 0), True),
        ((-1, 1, 0), (1, -1, 0), True),
        ((0, 1, 0), (1, 0, 0), True),
        # Zero-length segments inside, outside, on a vertex, and on an edge.
        ((0.25, 0.25, 0), (0.25, 0.25, 0), True),
        ((1, 1, 0), (1, 1, 0), False),
        ((0, 0, 0), (0, 0, 0), True),
        ((0.5, 0.5, 0), (0.5, 0.5, 0), True),
        ((0.25, 0.25, 1), (0.25, 0.25, 1), False),
    )
    points, expected = [], []
    for p0, p1, hit in cases:
        for order in permutations(range(3)):
            for segment in ((p0, p1), (p1, p0)):
                pair = np.concatenate((segment, triangle[list(order)]))
                for axis in range(3):
                    for scale in (0.125, 1.0, 16.0):
                        points.append(np.roll(pair, axis, axis=1) * scale + (2.0, -3.0, 1.0))
                        expected.append(hit)
    # Degenerate triangles have a defined False result.
    points.extend(
        (
            [[0, 0, -1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
        )
    )
    expected.extend((False, False))
    check_segments(test, device, np.asarray(points), expected)


def test_geometry_seg_random(test, device):
    rng = np.random.default_rng(6927)
    # Quantized inputs are represented exactly in both precisions. Avoid the
    # plane tolerance and exact contact; those have explicit fixtures above.
    points = rng.integers(-16, 17, (10000, 5, 3)).astype(np.float64) / 8.0
    p0, p1, a, b, c = np.moveaxis(points, 1, 0)
    normal = np.cross(b - a, c - a)
    d0 = np.einsum("ij,ij->i", normal, p0 - a)
    d1 = np.einsum("ij,ij->i", normal, p1 - a)
    # Test general position here; exact rational contacts are tested separately.
    keep = (np.abs(d0) > 1.0e-4) & (np.abs(d1) > 1.0e-4) & (np.abs(d0 - d1) > 1.0e-4)
    points = points[keep]
    # Solve independently for intersection barycentrics and exclude contacts
    # susceptible to different floating-point rounding between algorithms.
    p0, p1, a, b, c = np.moveaxis(points, 1, 0)
    parameters = np.linalg.solve(np.stack((p1 - p0, a - b, a - c), axis=2), (a - p0)[..., None])[..., 0]
    t, u, v = parameters.T
    margin = np.min(np.abs(np.stack((t, 1 - t, u, v, 1 - u - v))), axis=0)
    points = points[margin > 1.0e-4]
    check_segments(test, device, points, segment_triangle_reference(points))


def test_geometry_seg_coplanar_random(test, device):
    rng = np.random.default_rng(137)
    points, expected = [], []
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # Exact rational oracle: a segment hits the unit right triangle iff an
    # endpoint is inside or it meets one of the three boundary lines inside it.
    for segment in rng.integers(-8, 17, (1000, 2, 2)):
        p, q = [[Fraction(int(x), 8) for x in endpoint] for endpoint in segment]
        direction = [q[i] - p[i] for i in range(2)]
        parameters = [Fraction(0), Fraction(1)]
        for numerator, denominator in (
            (-p[0], direction[0]),
            (-p[1], direction[1]),
            (1 - p[0] - p[1], direction[0] + direction[1]),
        ):
            if denominator:
                parameters.append(numerator / denominator)
        hit = False
        for t in parameters:
            x, y = [p[i] + t * direction[i] for i in range(2)]
            hit |= 0 <= t <= 1 and x >= 0 and y >= 0 and x + y <= 1
        pair = np.concatenate((np.column_stack((segment / 8.0, [0.0, 0.0])), triangle))
        for axis in range(3):
            points.append(np.roll(pair, axis, axis=1))
            expected.append(hit)
    check_segments(test, device, np.asarray(points), expected)


def test_geometry_seg_tolerance(test, device):
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    points = []
    heights = (0.0, 0.5e-6, float(np.float32(1.0e-6)), 1.0e-6, 2.0e-6, -0.5e-6, -2.0e-6)
    for height in heights:
        points.append(np.concatenate(([[0.25, 0.25, height], [0.5, 0.25, height]], triangle)))
    # Float32 rounds 1e-6 to just below the double-precision threshold.
    for dtype, scalar in ((wp.vec3, np.float32), (wp.vec3d, np.float64)):
        expected = np.abs(np.asarray(heights, dtype=scalar).astype(np.float64)) < 1.0e-6
        with test.subTest(dtype=dtype):
            data = wp.array(points, dtype=dtype, device=device)
            hits = wp.empty(len(points), dtype=bool, device=device)
            wp.launch(intersect_segments, dim=len(points), inputs=[data, hits], device=device)
            np.testing.assert_array_equal(hits.numpy(), expected)


devices = get_test_devices()


class TestIntersect(unittest.TestCase):
    pass


add_function_test(TestIntersect, "test_intersect_tri", test_intersect_tri, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_random", test_geometry_tri_random, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_contacts", test_geometry_tri_contacts, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_tolerance", test_geometry_tri_tolerance, devices=devices)
add_function_test(TestIntersect, "test_geometry_seg_contacts", test_geometry_seg_contacts, devices=devices)
add_function_test(TestIntersect, "test_geometry_seg_random", test_geometry_seg_random, devices=devices)
add_function_test(
    TestIntersect, "test_geometry_seg_coplanar_random", test_geometry_seg_coplanar_random, devices=devices
)
add_function_test(TestIntersect, "test_geometry_seg_tolerance", test_geometry_seg_tolerance, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
