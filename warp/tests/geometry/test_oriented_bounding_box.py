# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.unittest_utils import add_function_test, get_test_devices


def _rotation_matrix(seed):
    rng = np.random.default_rng(seed)
    # A random rotation via QR of a Gaussian matrix (det fixed to +1).
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _obb_local(V, transform):
    # Map world points into the box-local frame using the returned transform
    # (box-local -> world), so an enclosing box means all |coord| <= extent/2.
    t = np.asarray(transform, dtype=np.float64)
    pos, q = t[:3], t[3:7]
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (V - pos) @ R  # world -> local


def test_obb_encloses_points(test, device):
    rng = np.random.default_rng(0)
    V = (rng.standard_normal((5000, 3)) @ np.diag([4.0, 2.0, 0.5])) @ _rotation_matrix(1).T
    points = wp.array(V, dtype=wp.vec3, device=device)
    transform, extents, _measure = geo.oriented_bounding_box(points, num_samples=1024)
    local = _obb_local(V, transform)
    half = np.abs(np.asarray(extents, dtype=np.float64)) / 2.0
    slack = 1e-4 * (half + 1.0)
    test.assertTrue(np.all(np.abs(local) <= half + slack), "OBB must enclose every point")


def test_obb_recovers_rotated_box(test, device):
    # An axis-aligned box of known size, randomly rotated: the fitted OBB volume should
    # be close to the true volume and far below the axis-aligned box of the rotated points.
    rng = np.random.default_rng(2)
    box = (rng.random((20000, 3)) - 0.5) * np.array([6.0, 3.0, 1.0])
    true_vol = 6.0 * 3.0 * 1.0
    V = box @ _rotation_matrix(3).T
    aabb_vol = float(np.prod(V.max(0) - V.min(0)))
    points = wp.array(V, dtype=wp.vec3, device=device)
    _t, _e, measure = geo.oriented_bounding_box(points, num_samples=2048)
    test.assertLess(measure, 1.05 * true_vol)
    test.assertLess(measure, 0.5 * aabb_vol)


def test_obb_subsample_matches_full(test, device):
    # Scoring on a subsample must still return a valid, near-identical box because the
    # winner's bounds are recomputed over all points.
    rng = np.random.default_rng(4)
    V = (rng.standard_normal((300000, 3)) @ np.diag([3.0, 1.5, 0.7])) @ _rotation_matrix(5).T
    points = wp.array(V, dtype=wp.vec3, device=device)
    _t0, _e0, m_full = geo.oriented_bounding_box(points, num_samples=512, refine_iters=0, max_search_points=None)
    t1, e1, m_sub = geo.oriented_bounding_box(points, num_samples=512, refine_iters=0, max_search_points=50_000)
    # Returned box still encloses all points.
    local = _obb_local(V, t1)
    half = np.abs(np.asarray(e1, dtype=np.float64)) / 2.0
    test.assertTrue(np.all(np.abs(local) <= half + 1e-4 * (half + 1.0)))
    # And is within a few percent of the full-search box.
    np.testing.assert_allclose(m_sub, m_full, rtol=0.05)


def test_obb_refinement_improves(test, device):
    # Refinement can only keep or improve on the sampled optimum at matched num_samples.
    rng = np.random.default_rng(6)
    V = (rng.standard_normal((40000, 3)) @ np.diag([5.0, 2.0, 0.4])) @ _rotation_matrix(7).T
    points = wp.array(V, dtype=wp.vec3, device=device)
    _t0, _e0, m_plain = geo.oriented_bounding_box(points, num_samples=128, refine_iters=0)
    _t1, _e1, m_refined = geo.oriented_bounding_box(points, num_samples=128, refine_iters=6)
    test.assertLessEqual(m_refined, m_plain * (1.0 + 1e-4))


def test_obb_invalid_args(test, device):
    points = wp.array(np.zeros((10, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with test.assertRaises(ValueError):
        geo.oriented_bounding_box(points, num_samples=0)
    empty = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with test.assertRaises(ValueError):
        geo.oriented_bounding_box(empty, num_samples=16)


devices = get_test_devices()


class TestOrientedBoundingBox(unittest.TestCase):
    pass


add_function_test(TestOrientedBoundingBox, "test_obb_encloses_points", test_obb_encloses_points, devices=devices)
add_function_test(TestOrientedBoundingBox, "test_obb_recovers_rotated_box", test_obb_recovers_rotated_box, devices=devices)
add_function_test(TestOrientedBoundingBox, "test_obb_subsample_matches_full", test_obb_subsample_matches_full, devices=devices)
add_function_test(TestOrientedBoundingBox, "test_obb_refinement_improves", test_obb_refinement_improves, devices=devices)
add_function_test(TestOrientedBoundingBox, "test_obb_invalid_args", test_obb_invalid_args, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
