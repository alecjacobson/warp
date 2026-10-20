# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
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
    result_single_precision[1] = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2)

    u0_d = wp.vec3d(u0)
    u1_d = wp.vec3d(u1)
    u2_d = wp.vec3d(u2)

    v0_d = wp.vec3d(v0)
    v1_d = wp.vec3d(v1)
    v2_d = wp.vec3d(v2)

    result_double_precision[0] = wp.intersect_tri_tri(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d)
    result_double_precision[1] = wp.geometry.intersect_tri_tri(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d)


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
def compare_triangles(points: wp.array2d[Any], native: wp.array[int], geometry: wp.array[int]):
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
            geometry = wp.empty_like(native)
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


devices = get_test_devices()


class TestIntersect(unittest.TestCase):
    pass


add_function_test(TestIntersect, "test_intersect_tri", test_intersect_tri, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_random", test_geometry_tri_random, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_contacts", test_geometry_tri_contacts, devices=devices)
add_function_test(TestIntersect, "test_geometry_tri_tolerance", test_geometry_tri_tolerance, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
