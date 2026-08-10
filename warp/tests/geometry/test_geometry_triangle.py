# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-triangle operations: areas, corner angles, and face normals."""

import math
import unittest

import numpy as np

import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *


def _rotation(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


##########################################################################
## triangle_areas
##########################################################################


def test_areas_analytic(test, device):
    # A 3-4-5 right triangle has area 6.
    points, indices = U.to_warp(*U.single_triangle(), device)
    np.testing.assert_allclose(geo.triangle_areas(points, indices).numpy(), [6.0], rtol=1e-6)

    # An equilateral triangle of side s has area sqrt(3)/4 * s^2.
    points, indices = U.to_warp(*U.equilateral_triangle(2.0), device)
    np.testing.assert_allclose(geo.triangle_areas(points, indices).numpy(), [math.sqrt(3.0)], rtol=1e-5)


def test_areas_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(7), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)
    np.testing.assert_allclose(
        geo.triangle_areas(points, indices).numpy(), U.ref_triangle_areas(p_np, f_np), rtol=1e-5, atol=1e-7
    )


def test_areas_rigid_invariance(test, device):
    p_np, f_np = U.icosphere(2)
    moved = (p_np @ _rotation((0.2, 0.9, -0.4), 1.1).T + np.array([3.0, -1.0, 2.0])).astype(np.float32)

    a = geo.triangle_areas(*U.to_warp(p_np, f_np, device)).numpy()
    b = geo.triangle_areas(*U.to_warp(moved, f_np, device)).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_areas_scaling(test, device):
    # Area is quadratic in a uniform scale factor.
    p_np, f_np = U.icosphere(1)
    scale = 2.5
    a = geo.triangle_areas(*U.to_warp(p_np, f_np, device)).numpy()
    b = geo.triangle_areas(*U.to_warp((p_np * scale).astype(np.float32), f_np, device)).numpy()
    np.testing.assert_allclose(b, a * scale * scale, rtol=1e-5)


def test_areas_degenerate(test, device):
    # Collinear and coincident vertices both give zero area rather than NaN.
    p_np = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=np.float32)
    f_np = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    areas = geo.triangle_areas(*U.to_warp(p_np, f_np, device)).numpy()
    np.testing.assert_allclose(areas, [0.0, 0.0], atol=1e-12)


##########################################################################
## triangle_corner_angles
##########################################################################


def test_corner_angles_equilateral(test, device):
    points, indices = U.to_warp(*U.equilateral_triangle(), device)
    angles = geo.triangle_corner_angles(points, indices).numpy()
    np.testing.assert_allclose(angles, [[math.pi / 3.0] * 3], rtol=1e-5)


def test_corner_angles_sum_to_pi(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(11), subdivisions=2)
    angles = geo.triangle_corner_angles(*U.to_warp(p_np, f_np, device)).numpy()
    np.testing.assert_allclose(angles.sum(axis=1), np.full(angles.shape[0], math.pi), rtol=1e-5)


def test_corner_angles_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(13), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)
    np.testing.assert_allclose(
        geo.triangle_corner_angles(points, indices).numpy(),
        U.ref_triangle_corner_angles(p_np, f_np),
        rtol=1e-5,
        atol=1e-6,
    )


def test_corner_angles_slivers(test, device):
    # Needle triangles have angles approaching 0 and pi, where acos(dot(u, v))
    # collapses in float32: at an angle of 2e-4 the cosine is 1 - 2e-8, which
    # rounds to exactly 1.0 and yields an angle of 0. The half-angle formula used
    # by corner_half_angle stays accurate to a relative error of ~1e-5 instead,
    # so this asserts *relative* accuracy on the tiny angles.
    p_np, f_np, expected = U.sliver_triangles()
    angles = geo.triangle_corner_angles(*U.to_warp(p_np, f_np, device)).numpy()

    np.testing.assert_allclose(angles, expected, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(angles.sum(axis=1), np.full(angles.shape[0], math.pi), rtol=1e-6)

    # The smallest angles must not have been flushed to zero.
    test.assertGreater(angles[-1, 0], 0.0)


##########################################################################
## triangle_normals
##########################################################################


def test_normals_magnitude_is_double_area(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(17), subdivisions=1)
    points, indices = U.to_warp(p_np, f_np, device)

    normals = geo.triangle_normals(points, indices).numpy()
    areas = geo.triangle_areas(points, indices).numpy()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 2.0 * areas, rtol=1e-5, atol=1e-7)


def test_normals_orthogonal_to_edges(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(19), subdivisions=1)
    points, indices = U.to_warp(p_np, f_np, device)
    normals = geo.triangle_normals(points, indices, normalized=True).numpy()

    f = f_np.reshape(-1, 3)
    e0 = p_np[f[:, 1]] - p_np[f[:, 0]]
    e1 = p_np[f[:, 2]] - p_np[f[:, 0]]
    # Edges are O(1) in length here, so an absolute tolerance is meaningful.
    np.testing.assert_allclose(np.einsum("ij,ij->i", normals, e0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.einsum("ij,ij->i", normals, e1), 0.0, atol=1e-6)


def test_normals_normalized_unit_length(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(23), subdivisions=1)
    normals = geo.triangle_normals(*U.to_warp(p_np, f_np, device), normalized=True).numpy()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-6)


def test_normals_winding_flips_sign(test, device):
    p_np, f_np = U.icosphere(1)
    flipped = f_np.reshape(-1, 3)[:, ::-1].copy().flatten()

    a = geo.triangle_normals(*U.to_warp(p_np, f_np, device)).numpy()
    b = geo.triangle_normals(*U.to_warp(p_np, flipped, device)).numpy()
    np.testing.assert_allclose(b, -a, rtol=1e-5, atol=1e-7)


def test_normals_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(29), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)
    for normalized in (False, True):
        np.testing.assert_allclose(
            geo.triangle_normals(points, indices, normalized=normalized).numpy(),
            U.ref_triangle_normals(p_np, f_np, normalized=normalized),
            rtol=1e-5,
            atol=1e-6,
        )


devices = get_test_devices()


class TestGeometryTriangle(unittest.TestCase):
    pass


add_function_test(TestGeometryTriangle, "test_areas_analytic", test_areas_analytic, devices=devices)
add_function_test(TestGeometryTriangle, "test_areas_reference", test_areas_reference, devices=devices)
add_function_test(TestGeometryTriangle, "test_areas_rigid_invariance", test_areas_rigid_invariance, devices=devices)
add_function_test(TestGeometryTriangle, "test_areas_scaling", test_areas_scaling, devices=devices)
add_function_test(TestGeometryTriangle, "test_areas_degenerate", test_areas_degenerate, devices=devices)
add_function_test(
    TestGeometryTriangle, "test_corner_angles_equilateral", test_corner_angles_equilateral, devices=devices
)
add_function_test(TestGeometryTriangle, "test_corner_angles_sum_to_pi", test_corner_angles_sum_to_pi, devices=devices)
add_function_test(TestGeometryTriangle, "test_corner_angles_reference", test_corner_angles_reference, devices=devices)
add_function_test(TestGeometryTriangle, "test_corner_angles_slivers", test_corner_angles_slivers, devices=devices)
add_function_test(
    TestGeometryTriangle,
    "test_normals_magnitude_is_double_area",
    test_normals_magnitude_is_double_area,
    devices=devices,
)
add_function_test(
    TestGeometryTriangle, "test_normals_orthogonal_to_edges", test_normals_orthogonal_to_edges, devices=devices
)
add_function_test(
    TestGeometryTriangle, "test_normals_normalized_unit_length", test_normals_normalized_unit_length, devices=devices
)
add_function_test(
    TestGeometryTriangle, "test_normals_winding_flips_sign", test_normals_winding_flips_sign, devices=devices
)
add_function_test(TestGeometryTriangle, "test_normals_reference", test_normals_reference, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
