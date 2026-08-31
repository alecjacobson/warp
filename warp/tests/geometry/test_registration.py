# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid registration (ICP): warp.geometry.register_rigid."""

import unittest

import numpy as np

import warp as wp
import warp.geometry as reg
from warp.tests.unittest_utils import *


def _icosphere(subdiv=3, scale=(1.0, 1.0, 1.0)):
    """A subdivided icosphere, optionally scaled per-axis into an (asymmetric)
    ellipsoid so that rotation is recoverable by ICP."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    verts = [np.array(v, dtype=np.float64) for v in verts]
    for _ in range(subdiv):
        mid: dict = {}
        new_faces = []

        def midpoint(a, b, mid=mid):
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((verts[a] + verts[b]) * 0.5)
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    v = np.array(verts)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    v = (v * np.array(scale)).astype(np.float32)
    return v, np.array(faces, dtype=np.int32).reshape(-1)


def _rigid_transform(rot_deg, trans, seed):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = trans
    return T


def _rotation_error_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0))))


def test_recovers_known_transform(test, device):
    # Source = ellipsoid surface points moved by a known transform; ICP should
    # recover the inverse and land the source back on the target.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    for rot_deg in (5.0, 20.0):
        T = _rigid_transform(rot_deg, np.array([0.06, -0.04, 0.05]), seed=int(rot_deg))
        source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        result = reg.register_rigid(source, (verts, faces), max_iters=80, tol=1e-6, device=device)
        wp.synchronize_device()

        test.assertTrue(result.converged)
        test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)
        np.testing.assert_allclose(result.transform[:3, 3], np.linalg.inv(T)[:3, 3], atol=1e-3)
        test.assertLess(result.rmse, 1e-3)


def _ellipsoid_normals(verts, scale):
    # Outward unit normal of the ellipsoid (x/sx)^2+(y/sy)^2+(z/sz)^2=1 at each
    # surface point is proportional to (x/sx^2, y/sy^2, z/sz^2).
    n = verts / (np.asarray(scale, dtype=np.float64) ** 2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n.astype(np.float32)


def test_recovers_known_transform_point_cloud(test, device):
    # Point-cloud target given as bare points: normals are estimated by local PCA
    # and ICP recovers the known transform from the discrete nearest neighbors.
    scale = (1.5, 1.0, 0.7)
    verts, _ = _icosphere(subdiv=3, scale=scale)
    for rot_deg in (5.0, 10.0):
        T = _rigid_transform(rot_deg, np.array([0.05, -0.03, 0.04]), seed=int(rot_deg))
        source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        result = reg.register_rigid(source, verts, max_iters=80, tol=1e-6, device=device)
        wp.synchronize_device()

        test.assertTrue(result.converged)
        test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)
        np.testing.assert_allclose(result.transform[:3, 3], np.linalg.inv(T)[:3, 3], atol=1e-3)


def test_point_cloud_with_normals(test, device):
    # Point-cloud target given as (points, normals): supplied normals are used
    # directly instead of estimating them.
    scale = (1.5, 1.0, 0.7)
    verts, _ = _icosphere(subdiv=3, scale=scale)
    normals = _ellipsoid_normals(verts, scale)
    T = _rigid_transform(10.0, np.array([0.05, 0.04, -0.03]), seed=7)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    result = reg.register_rigid(source, (verts, normals), max_iters=80, tol=1e-6, device=device)
    wp.synchronize_device()

    test.assertTrue(result.converged)
    test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)
    np.testing.assert_allclose(result.transform[:3, 3], np.linalg.inv(T)[:3, 3], atol=1e-3)


def test_identity_when_aligned(test, device):
    # Already-aligned source: ICP returns ~identity in one step.
    verts, faces = _icosphere(subdiv=3, scale=(1.4, 1.0, 0.8))
    result = reg.register_rigid(verts, (verts, faces), max_iters=20, tol=1e-6, device=device)
    wp.synchronize_device()
    np.testing.assert_allclose(result.transform, np.eye(4), atol=1e-4)


def test_deterministic(test, device):
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(15.0, np.array([0.05, 0.05, -0.05]), seed=1)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    a = reg.register_rigid(source, (verts, faces), max_iters=50, device=device)
    b = reg.register_rigid(source, (verts, faces), max_iters=50, device=device)
    wp.synchronize_device()
    # Accumulation uses floating-point atomics, so runs match closely but not
    # necessarily bit-for-bit.
    np.testing.assert_allclose(a.transform, b.transform, atol=1e-5)


def test_reuses_target_mesh(test, device):
    # A prebuilt wp.Mesh can be passed as the target and reused across calls
    # without rebuild.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    T = _rigid_transform(12.0, np.array([0.04, -0.03, 0.02]), seed=2)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    result = reg.register_rigid(source, mesh, max_iters=60, tol=1e-9, device=device)
    wp.synchronize_device()
    test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.05)


devices = get_test_devices()


class TestRegistration(unittest.TestCase):
    pass


add_function_test(TestRegistration, "test_recovers_known_transform", test_recovers_known_transform, devices=devices)
add_function_test(
    TestRegistration,
    "test_recovers_known_transform_point_cloud",
    test_recovers_known_transform_point_cloud,
    devices=devices,
)
add_function_test(TestRegistration, "test_point_cloud_with_normals", test_point_cloud_with_normals, devices=devices)
add_function_test(TestRegistration, "test_identity_when_aligned", test_identity_when_aligned, devices=devices)
add_function_test(TestRegistration, "test_deterministic", test_deterministic, devices=devices)
add_function_test(TestRegistration, "test_reuses_target_mesh", test_reuses_target_mesh, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
