# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Volume, first moment, and inertia tensor of a closed triangle mesh."""

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


def _moments(mesh, device):
    volume, first, inertia = geo.moments(*U.to_warp(*mesh, device))
    return float(volume.numpy()[0]), first.numpy()[0], inertia.numpy()[0]


def test_moments_unit_cube(test, device):
    # A unit cube of unit density has volume 1 and, about its centroid,
    # I = M/12 * (b^2 + c^2) = 1/6 on each diagonal entry.
    volume, first, inertia = _moments(U.unit_cube(center=(0.0, 0.0, 0.0)), device)

    test.assertAlmostEqual(volume, 1.0, places=5)
    np.testing.assert_allclose(first, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(inertia, np.eye(3) / 6.0, rtol=1e-5, atol=1e-6)


def test_moments_first_moment_is_centroid_times_volume(test, device):
    center = np.array([0.5, -1.25, 2.0])
    side = 1.5
    volume, first, _ = _moments(U.unit_cube(center=tuple(center), side=side), device)

    test.assertAlmostEqual(volume, side**3, places=4)
    np.testing.assert_allclose(first, center * side**3, rtol=1e-4, atol=1e-5)


def test_moments_inertia_is_translation_invariant(test, device):
    # The inertia tensor is taken about the centroid, so translating the mesh
    # must not change it. Translations are kept modest: the implementation
    # forms the central second moment as int(x^2) - V*cx^2, and in float32 a
    # large offset would lose the difference to cancellation.
    p_np, f_np = U.unit_cube(side=1.0)
    _, _, base = _moments((p_np, f_np), device)

    for offset in ([1.0, 0.0, 0.0], [-2.0, 1.5, 0.75]):
        with test.subTest(offset=offset):
            moved = (p_np + np.asarray(offset, dtype=np.float64)).astype(np.float32)
            _, _, inertia = _moments((moved, f_np), device)
            np.testing.assert_allclose(inertia, base, rtol=1e-3, atol=1e-4)


def test_moments_inertia_rotates_covariantly(test, device):
    # Rotating the mesh by R takes the inertia tensor to R I R^T.
    p_np, f_np = U.unit_cube(side=1.0)
    # A box with distinct side lengths, so the tensor is not isotropic and the
    # covariance is actually being tested.
    p_np = (p_np * np.array([1.0, 2.0, 3.0])).astype(np.float32)
    _, _, base = _moments((p_np, f_np), device)

    rot = _rotation((0.3, -0.7, 0.5), 0.8)
    rotated = (p_np @ rot.T).astype(np.float32)
    _, _, inertia = _moments((rotated, f_np), device)

    np.testing.assert_allclose(inertia, rot @ base @ rot.T, rtol=1e-4, atol=1e-5)


def test_moments_scaling(test, device):
    # Volume scales as s^3 and inertia (unit density) as s^5.
    p_np, f_np = U.icosphere(2)
    volume, _, inertia = _moments((p_np, f_np), device)

    scale = 1.8
    scaled = (p_np * scale).astype(np.float32)
    volume_s, _, inertia_s = _moments((scaled, f_np), device)

    test.assertAlmostEqual(volume_s / volume, scale**3, places=3)
    np.testing.assert_allclose(inertia_s, inertia * scale**5, rtol=1e-3, atol=1e-5)


def test_moments_sphere_limit(test, device):
    # A refined icosphere approaches the analytic solid sphere: V = 4/3 pi r^3
    # and I = 2/5 M r^2. The polyhedron is inscribed, so it undershoots slightly.
    radius = 1.0
    volume, first, inertia = _moments(U.icosphere(4, radius=radius), device)

    test.assertAlmostEqual(volume, 4.0 / 3.0 * math.pi * radius**3, delta=0.02)
    np.testing.assert_allclose(first, np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(np.diag(inertia), np.full(3, 0.4 * volume * radius**2), rtol=5e-3)
    # Off-diagonal terms vanish for a sphere.
    np.testing.assert_allclose(inertia - np.diag(np.diag(inertia)), np.zeros((3, 3)), atol=1e-4)


def test_moments_reversed_winding_gives_negative_volume(test, device):
    # The tetrahedron sum is signed, so a consistently inward-facing mesh
    # reports a negative volume. This pins the documented orientation
    # requirement rather than silently returning an absolute value.
    p_np, f_np = U.unit_cube()
    flipped = f_np.reshape(-1, 3)[:, ::-1].copy().flatten()

    volume, _, _ = _moments((p_np, f_np), device)
    volume_flipped, _, _ = _moments((p_np, flipped), device)

    test.assertGreater(volume, 0.0)
    test.assertAlmostEqual(volume_flipped, -volume, places=5)


def test_moments_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(53), subdivisions=2)
    volume, first, inertia = _moments((p_np, f_np), device)
    ref_volume, ref_first, ref_inertia = U.ref_moments(p_np, f_np)

    test.assertAlmostEqual(volume, ref_volume, places=4)
    np.testing.assert_allclose(first, ref_first, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(inertia, ref_inertia, rtol=1e-3, atol=1e-4)


def test_moments_torus(test, device):
    # An independent closed shape with an analytic answer: V = 2 pi^2 R r^2.
    major, minor = 1.0, 0.35
    volume, first, inertia = _moments(U.torus(64, 32, major, minor), device)

    test.assertAlmostEqual(volume, 2.0 * math.pi**2 * major * minor**2, delta=0.02)
    np.testing.assert_allclose(first, np.zeros(3), atol=1e-5)
    # Symmetry of revolution about z: Ixx == Iyy, and Izz is the largest.
    test.assertAlmostEqual(inertia[0, 0], inertia[1, 1], places=4)
    test.assertGreater(inertia[2, 2], inertia[0, 0])


devices = get_test_devices()


class TestGeometryMoments(unittest.TestCase):
    pass


add_function_test(TestGeometryMoments, "test_moments_unit_cube", test_moments_unit_cube, devices=devices)
add_function_test(
    TestGeometryMoments,
    "test_moments_first_moment_is_centroid_times_volume",
    test_moments_first_moment_is_centroid_times_volume,
    devices=devices,
)
add_function_test(
    TestGeometryMoments,
    "test_moments_inertia_is_translation_invariant",
    test_moments_inertia_is_translation_invariant,
    devices=devices,
)
add_function_test(
    TestGeometryMoments,
    "test_moments_inertia_rotates_covariantly",
    test_moments_inertia_rotates_covariantly,
    devices=devices,
)
add_function_test(TestGeometryMoments, "test_moments_scaling", test_moments_scaling, devices=devices)
add_function_test(TestGeometryMoments, "test_moments_sphere_limit", test_moments_sphere_limit, devices=devices)
add_function_test(
    TestGeometryMoments,
    "test_moments_reversed_winding_gives_negative_volume",
    test_moments_reversed_winding_gives_negative_volume,
    devices=devices,
)
add_function_test(TestGeometryMoments, "test_moments_reference", test_moments_reference, devices=devices)
add_function_test(TestGeometryMoments, "test_moments_torus", test_moments_torus, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
