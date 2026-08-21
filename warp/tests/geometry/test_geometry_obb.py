# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oriented bounding box fitting."""

import math
import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *


def _rotation(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rod(rng, extents=(2.0, 0.4, 0.15), num_points=400):
    """A strongly elongated box, rotated away from the axes.

    The corners are included so the minimal box of the sample is exactly the
    generating box, making the optimal measure known in closed form.
    """
    extents = np.asarray(extents, dtype=np.float64)
    corners = np.array(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij")).reshape(3, -1).T * extents
    interior = rng.uniform(-0.5, 0.5, (num_points, 3)) * extents
    points = np.concatenate([corners, interior], axis=0)
    return np.ascontiguousarray(points @ _rotation((0.3, -0.7, 0.5), 0.9).T, dtype=np.float32), extents


def _to_local(points_np, transform, extents):
    """Map world points into the box frame and return their signed overhang."""
    t = np.array([transform[i] for i in range(7)], dtype=np.float64)
    position, quat = t[:3], t[3:]
    qv, qw = quat[:3], quat[3]

    # Inverse rotation of (p - position) by the unit quaternion.
    d = np.asarray(points_np, dtype=np.float64) - position
    tmp = 2.0 * np.cross(-qv, d)
    local = d + qw * tmp + np.cross(-qv, tmp)

    return np.abs(local) - 0.5 * np.asarray(extents, dtype=np.float64)


def _aabb_measure(points_np, measure_type):
    dims = points_np.max(axis=0) - points_np.min(axis=0)
    if measure_type == geo.OBBMeasureType.VOLUME:
        return float(np.prod(dims))
    return float(2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2]))


def _shapes(rng):
    return {
        "cube": U.unit_cube()[0],
        "icosphere": U.icosphere(2)[0],
        "rod": _rod(rng)[0],
        "cloud": rng.standard_normal((500, 3)).astype(np.float32),
    }


def test_obb_contains_all_points(test, device):
    # The defining property: the returned box must actually bound the input.
    rng = np.random.default_rng(59)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                transform, extents, _ = geo.oriented_bounding_box(points, measure_type=measure_type, num_samples=256)
                overhang = _to_local(p_np, transform, extents).max()
                scale = float(np.max(np.asarray(extents, dtype=np.float64)))
                test.assertLess(overhang, 1e-5 * max(scale, 1.0))


def test_obb_measure_matches_extents(test, device):
    rng = np.random.default_rng(61)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    _, extents, measure = geo.oriented_bounding_box(points, num_samples=128)
    e = np.asarray(extents, dtype=np.float64)
    test.assertAlmostEqual(measure / float(np.prod(e)), 1.0, places=4)

    _, extents, measure = geo.oriented_bounding_box(
        points, measure_type=geo.OBBMeasureType.SURFACE_AREA, num_samples=128
    )
    e = np.asarray(extents, dtype=np.float64)
    expected = 2.0 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2])
    test.assertAlmostEqual(measure / expected, 1.0, places=4)


def test_obb_never_worse_than_aabb(test, device):
    # include_axis_aligned adds the identity rotation as a candidate, so the
    # result is guaranteed to be at least as tight as the AABB.
    rng = np.random.default_rng(67)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                _, _, measure = geo.oriented_bounding_box(
                    points, measure_type=measure_type, num_samples=64, include_axis_aligned=True
                )
                test.assertLessEqual(measure, _aabb_measure(p_np, measure_type) * (1.0 + 1e-5))


def test_obb_recovers_rotated_box(test, device):
    # With enough samples the fit should be close to the true minimal box.
    rng = np.random.default_rng(71)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, got_extents, measure = geo.oriented_bounding_box(points, num_samples=4096)

    true_volume = float(np.prod(extents))
    test.assertGreaterEqual(measure, true_volume * (1.0 - 1e-4))
    test.assertLess(measure, true_volume * 2.0)

    # The longest side must be recovered accurately even if the two short sides
    # trade off against each other.
    test.assertAlmostEqual(max(np.asarray(got_extents, dtype=np.float64)), max(extents), delta=0.05)


def test_obb_pca_helps_elongated_shapes(test, device):
    # On a strongly elongated shape the spiral's angular resolution is the
    # limiting factor, and the covariance eigenvectors are a much better guess.
    rng = np.random.default_rng(73)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, _, without_pca = geo.oriented_bounding_box(points, num_samples=32, include_pca=False)
    _, _, with_pca = geo.oriented_bounding_box(points, num_samples=32, include_pca=True)

    test.assertLess(with_pca, without_pca)
    test.assertLess(with_pca, float(np.prod(extents)) * 2.0)


def test_obb_reproducible_without_pca(test, device):
    # The docstring promises a bitwise reproducible result with include_pca=False,
    # since the min/max reduction is exact regardless of atomic ordering.
    rng = np.random.default_rng(79)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    results = [geo.oriented_bounding_box(points, num_samples=128, include_pca=False) for _ in range(3)]
    for transform, extents, measure in results[1:]:
        test.assertEqual(measure, results[0][2])
        test.assertEqual(tuple(extents), tuple(results[0][1]))
        test.assertEqual(tuple(transform), tuple(results[0][0]))


def test_obb_single_point(test, device):
    points = wp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    transform, extents, measure = geo.oriented_bounding_box(points, num_samples=8)

    np.testing.assert_allclose(np.asarray(extents, dtype=np.float64), np.zeros(3), atol=1e-5)
    test.assertAlmostEqual(measure, 0.0, places=6)
    np.testing.assert_allclose(np.array([transform[0], transform[1], transform[2]]), [1.0, 2.0, 3.0], rtol=1e-5)


def test_obb_invalid_arguments(test, device):
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)

    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=0)
    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=-5)

    empty = wp.zeros(0, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "at least one point"):
        geo.oriented_bounding_box(empty, num_samples=8)


def test_obb_rejects_graph_capture(test, device):
    # Documented limitation: the argmin runs on the host, so the function
    # synchronizes and cannot be captured.
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)
    wp.load_module(geo, device=device)

    with test.assertRaises(RuntimeError):
        with wp.ScopedCapture(device=device):
            geo.oriented_bounding_box(points, num_samples=16)


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestGeometryOBB(unittest.TestCase):
    pass


add_function_test(TestGeometryOBB, "test_obb_contains_all_points", test_obb_contains_all_points, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_measure_matches_extents", test_obb_measure_matches_extents, devices=devices
)
add_function_test(TestGeometryOBB, "test_obb_never_worse_than_aabb", test_obb_never_worse_than_aabb, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_recovers_rotated_box", test_obb_recovers_rotated_box, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_pca_helps_elongated_shapes", test_obb_pca_helps_elongated_shapes, devices=devices
)
add_function_test(
    TestGeometryOBB, "test_obb_reproducible_without_pca", test_obb_reproducible_without_pca, devices=devices
)
add_function_test(TestGeometryOBB, "test_obb_single_point", test_obb_single_point, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_invalid_arguments", test_obb_invalid_arguments, devices=devices)
add_function_test(
    TestGeometryOBB, "test_obb_rejects_graph_capture", test_obb_rejects_graph_capture, devices=cuda_devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
