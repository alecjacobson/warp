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


def _host(result):
    """Read the length-1 device arrays returned by ``oriented_bounding_box`` to host."""
    transform, extents, measure = result
    return transform.numpy()[0], extents.numpy()[0], float(measure.numpy()[0])


def _rotation(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rod(rng, extents=(2.0, 0.4, 0.15), num_points=400):
    """Build a strongly elongated box, rotated away from the axes.

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
                transform, extents, _ = _host(
                    geo.oriented_bounding_box(points, measure_type=measure_type, num_samples=256)
                )
                overhang = _to_local(p_np, transform, extents).max()
                scale = float(np.max(np.asarray(extents, dtype=np.float64)))
                test.assertLess(overhang, 1e-5 * max(scale, 1.0))


def test_obb_measure_matches_extents(test, device):
    rng = np.random.default_rng(61)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    _, extents, measure = _host(geo.oriented_bounding_box(points, num_samples=128))
    e = np.asarray(extents, dtype=np.float64)
    test.assertAlmostEqual(measure / float(np.prod(e)), 1.0, places=4)

    _, extents, measure = _host(
        geo.oriented_bounding_box(points, measure_type=geo.OBBMeasureType.SURFACE_AREA, num_samples=128)
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
                _, _, measure = _host(
                    geo.oriented_bounding_box(
                        points, measure_type=measure_type, num_samples=64, include_axis_aligned=True
                    )
                )
                test.assertLessEqual(measure, _aabb_measure(p_np, measure_type) * (1.0 + 1e-5))


def test_obb_recovers_rotated_box(test, device):
    # With enough samples the fit should be close to the true minimal box.
    rng = np.random.default_rng(71)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, got_extents, measure = _host(geo.oriented_bounding_box(points, num_samples=4096))

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

    _, _, without_pca = _host(geo.oriented_bounding_box(points, num_samples=32, include_pca=False))
    _, _, with_pca = _host(geo.oriented_bounding_box(points, num_samples=32, include_pca=True))

    test.assertLess(with_pca, without_pca)
    test.assertLess(with_pca, float(np.prod(extents)) * 2.0)


def test_obb_reproducible_without_pca(test, device):
    # The docstring promises a bitwise reproducible result with include_pca=False,
    # since the min/max reduction is exact regardless of atomic ordering.
    rng = np.random.default_rng(79)
    points = wp.array(_shapes(rng)["cloud"], dtype=wp.vec3, device=device)

    results = [_host(geo.oriented_bounding_box(points, num_samples=128, include_pca=False)) for _ in range(3)]
    for transform, extents, measure in results[1:]:
        test.assertEqual(measure, results[0][2])
        test.assertEqual(tuple(extents), tuple(results[0][1]))
        test.assertEqual(tuple(transform), tuple(results[0][0]))


def test_obb_single_point(test, device):
    points = wp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    transform, extents, measure = _host(geo.oriented_bounding_box(points, num_samples=8))

    np.testing.assert_allclose(np.asarray(extents, dtype=np.float64), np.zeros(3), atol=1e-5)
    test.assertAlmostEqual(measure, 0.0, places=6)
    np.testing.assert_allclose(np.array([transform[0], transform[1], transform[2]]), [1.0, 2.0, 3.0], rtol=1e-5)


def test_obb_refinement_never_worse(test, device):
    # Refinement is a local search that only commits improvements, so the refined box is
    # never worse than the unrefined one on the same input, for either objective.
    rng = np.random.default_rng(83)
    for name, p_np in _shapes(rng).items():
        points = wp.array(p_np, dtype=wp.vec3, device=device)
        for measure_type in geo.OBBMeasureType:
            with test.subTest(shape=name, measure=measure_type.name):
                _, _, base = _host(
                    geo.oriented_bounding_box(points, measure_type=measure_type, num_samples=128, refine_iters=0)
                )
                _, _, refined = _host(
                    geo.oriented_bounding_box(points, measure_type=measure_type, num_samples=128, refine_iters=6)
                )
                test.assertLessEqual(refined, base * (1.0 + 1e-5))


def test_obb_refinement_reaches_higher_sample_quality(test, device):
    # Coarse-to-fine refinement lets a small sample count reach the quality of a much larger
    # one on an elongated shape, where the spiral's angular spacing is the limiting factor.
    rng = np.random.default_rng(89)
    p_np, extents = _rod(rng)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    _, _, coarse = _host(geo.oriented_bounding_box(points, num_samples=64, refine_iters=0))
    _, _, refined = _host(geo.oriented_bounding_box(points, num_samples=64, refine_iters=8))

    test.assertLess(refined, coarse)
    test.assertLess(refined, float(np.prod(extents)) * 1.2)


def test_obb_subsample_stays_exact(test, device):
    # The winner's box is re-measured over the full cloud, so scoring orientations on a
    # strided subsample still returns a box that encloses every point.
    rng = np.random.default_rng(97)
    p_np, extents = _rod(rng, num_points=20000)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    transform, got_extents, measure = _host(geo.oriented_bounding_box(points, num_samples=256, max_search_points=2000))
    overhang = _to_local(p_np, transform, got_extents).max()
    test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(got_extents, dtype=np.float64))), 1.0))
    test.assertLess(measure, float(np.prod(extents)) * 2.0)


def test_obb_subsample_boundary(test, device):
    # Point counts just above the cap (stride > 1, uneven division): must not read out of
    # bounds and must still enclose every point.
    rng = np.random.default_rng(101)
    for n, cap in ((201, 100), (1000, 999), (1000, 7)):
        with test.subTest(n=n, cap=cap):
            p_np = rng.standard_normal((n, 3)).astype(np.float32)
            points = wp.array(p_np, dtype=wp.vec3, device=device)
            transform, extents, _ = _host(geo.oriented_bounding_box(points, num_samples=32, max_search_points=cap))
            overhang = _to_local(p_np, transform, extents).max()
            test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(extents, dtype=np.float64))), 1.0))


def test_obb_refine_tiny_cloud(test, device):
    # Fewer points than a refine batch or a chunk count: must not crash and must enclose all.
    rng = np.random.default_rng(103)
    p_np = rng.standard_normal((3, 3)).astype(np.float32)
    points = wp.array(p_np, dtype=wp.vec3, device=device)

    transform, extents, _ = _host(geo.oriented_bounding_box(points, num_samples=16, refine_iters=4, refine_batch=8))
    overhang = _to_local(p_np, transform, extents).max()
    test.assertLess(overhang, 1e-4 * max(float(np.max(np.asarray(extents, dtype=np.float64))), 1.0))


def test_obb_invalid_arguments(test, device):
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)

    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=0)
    with test.assertRaisesRegex(ValueError, "num_samples"):
        geo.oriented_bounding_box(points, num_samples=-5)
    with test.assertRaisesRegex(ValueError, "refine_iters"):
        geo.oriented_bounding_box(points, refine_iters=-1)
    with test.assertRaisesRegex(ValueError, "refine_batch"):
        geo.oriented_bounding_box(points, refine_batch=0)
    with test.assertRaisesRegex(ValueError, "max_search_points"):
        geo.oriented_bounding_box(points, max_search_points=0)

    empty = wp.zeros(0, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "at least one point"):
        geo.oriented_bounding_box(empty, num_samples=8)


def test_obb_graph_capturable(test, device):
    # The whole search -- including the argmin -- runs on the device, so it can be
    # captured into a CUDA graph and replayed without a host synchronization.
    points = wp.array(U.unit_cube()[0], dtype=wp.vec3, device=device)
    wp.load_module(geo, device=device)

    with wp.ScopedCapture(device=device) as capture:
        result = geo.oriented_bounding_box(points, num_samples=16)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # The unit cube is already axis-aligned, so its minimal box is the cube itself.
    _, extents, measure = _host(result)
    np.testing.assert_allclose(np.asarray(extents, dtype=np.float64), np.ones(3), atol=1e-5)
    test.assertAlmostEqual(measure, 1.0, places=5)


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
add_function_test(TestGeometryOBB, "test_obb_refinement_never_worse", test_obb_refinement_never_worse, devices=devices)
add_function_test(
    TestGeometryOBB,
    "test_obb_refinement_reaches_higher_sample_quality",
    test_obb_refinement_reaches_higher_sample_quality,
    devices=devices,
)
add_function_test(TestGeometryOBB, "test_obb_subsample_stays_exact", test_obb_subsample_stays_exact, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_subsample_boundary", test_obb_subsample_boundary, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_refine_tiny_cloud", test_obb_refine_tiny_cloud, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_invalid_arguments", test_obb_invalid_arguments, devices=devices)
add_function_test(TestGeometryOBB, "test_obb_graph_capturable", test_obb_graph_capturable, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
