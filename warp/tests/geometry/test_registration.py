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


def test_robust_rejects_outliers(test, device):
    # Replace a fraction of source points with gross outliers. Plain least
    # squares is dragged off; the Welsch robust weight recovers a much better fit.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(10.0, np.array([0.05, -0.03, 0.04]), seed=3)
    Tinv = np.linalg.inv(T)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    rng = np.random.default_rng(0)
    num_outliers = int(0.2 * len(source))
    outlier_idx = rng.choice(len(source), num_outliers, replace=False)
    source[outlier_idx] = (rng.standard_normal((num_outliers, 3)) * 0.8).astype(np.float32)

    plain = reg.register_rigid(source, (verts, faces), max_iters=80, tol=1e-7, device=device)
    robust = reg.register_rigid(
        source, (verts, faces), max_iters=80, tol=1e-7, robust="welsch", robust_k=3.0, device=device
    )
    wp.synchronize_device()

    plain_rot = _rotation_error_deg(plain.transform, Tinv)
    robust_rot = _rotation_error_deg(robust.transform, Tinv)
    plain_trans = np.linalg.norm(plain.transform[:3, 3] - Tinv[:3, 3])
    robust_trans = np.linalg.norm(robust.transform[:3, 3] - Tinv[:3, 3])

    test.assertLess(robust_rot, plain_rot)
    test.assertLess(robust_trans, plain_trans)
    test.assertLess(robust_rot, 2.0)


def test_stochastic_subsampling_recovers(test, device):
    # Using a random subset of source points per iteration still recovers the
    # known transform (the source is a rigid copy, so subsets stay consistent).
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(10.0, np.array([0.05, -0.03, 0.04]), seed=3)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    result = reg.register_rigid(
        source, (verts, faces), max_iters=120, tol=1e-6, sample_count=200, seed=1, device=device
    )
    wp.synchronize_device()

    test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.5)
    np.testing.assert_allclose(result.transform[:3, 3], np.linalg.inv(T)[:3, 3], atol=2e-2)


def test_rejects_unknown_robust(test, device):
    verts, faces = _icosphere(subdiv=2, scale=(1.2, 1.0, 0.9))
    with test.assertRaises(ValueError):
        reg.register_rigid(verts, (verts, faces), robust="huber", device=device)


def test_plane_normal_closest_point(test, device):
    # For a mesh, the closest-point direction equals the face normal for
    # face-interior queries, so plane_normal="closest_point" recovers the
    # transform just like the surface normal (no precomputed normals needed).
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(15.0, np.array([0.05, -0.03, 0.04]), seed=4)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    result = reg.register_rigid(
        source, (verts, faces), max_iters=100, tol=1e-8, max_corr_dist=1.0,
        plane_normal="closest_point", device=device,
    )  # fmt: skip
    wp.synchronize_device()
    test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)


def test_rejects_unknown_plane_normal(test, device):
    verts, faces = _icosphere(subdiv=2, scale=(1.2, 1.0, 0.9))
    with test.assertRaises(ValueError):
        reg.register_rigid(verts, (verts, faces), plane_normal="tangent", device=device)


def _rot_about(rot_deg, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    th = np.radians(rot_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    return T


def test_batched_multi_init(test, device):
    # Several initializations against a shared target; the best one matches the
    # single-problem solve from that same init and recovers the transform.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(20.0, np.array([0.06, -0.04, 0.05]), seed=5)
    Tinv = np.linalg.inv(T)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    inits = np.stack([np.eye(4), _rot_about(30, (1, 0, 0)), _rot_about(-25, (0, 1, 0)), _rot_about(40, (0, 0, 1))])
    batched = reg.register_rigid_batched(source, (verts, faces), inits, max_iters=80, tol=1e-6, device=device)
    wp.synchronize_device()

    test.assertEqual(batched.transforms.shape, (4, 4, 4))
    test.assertLess(_rotation_error_deg(batched.transform, Tinv), 0.1)

    single = reg.register_rigid(
        source, (verts, faces), init=inits[batched.best_index], max_iters=80, tol=1e-6, device=device
    )
    wp.synchronize_device()
    np.testing.assert_allclose(batched.transform, single.transform, atol=1e-4)


def test_batched_multi_source(test, device):
    # A distinct source per batch entry, all aligned to the shared target.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    transforms = [_rigid_transform(d, np.array([0.05, 0.0, -0.03]), seed=d) for d in (8, 12, 16)]
    sources = np.stack([(verts @ Ti[:3, :3].T + Ti[:3, 3]).astype(np.float32) for Ti in transforms])
    inits = np.tile(np.eye(4), (3, 1, 1))

    batched = reg.register_rigid_batched(sources, (verts, faces), inits, max_iters=80, tol=1e-6, device=device)
    wp.synchronize_device()

    for b, Ti in enumerate(transforms):
        test.assertLess(_rotation_error_deg(batched.transforms[b], np.linalg.inv(Ti)), 0.1)


def test_batched_validates_inits(test, device):
    verts, faces = _icosphere(subdiv=2, scale=(1.2, 1.0, 0.9))
    with test.assertRaises(ValueError):
        reg.register_rigid_batched(verts, (verts, faces), np.eye(4), device=device)  # not (B, 4, 4)


def test_symmetric_recovers(test, device):
    # The symmetric variant (Rusinkiewicz 2019) recovers a known transform; source
    # normals are estimated by local PCA when not supplied.
    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    for rot_deg in (10.0, 25.0, 40.0):
        T = _rigid_transform(rot_deg, np.array([0.06, -0.04, 0.05]), seed=int(rot_deg))
        source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        result = reg.register_rigid(source, (verts, faces), max_iters=100, tol=1e-6, variant="symmetric", device=device)
        wp.synchronize_device()

        test.assertTrue(result.converged)
        test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)


def test_symmetric_with_source_normals(test, device):
    # Supplied source normals are used directly by the symmetric variant.
    scale = (1.5, 1.0, 0.7)
    verts, faces = _icosphere(subdiv=3, scale=scale)
    normals = _ellipsoid_normals(verts, scale)
    T = _rigid_transform(20.0, np.array([0.05, 0.04, -0.03]), seed=11)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    rotated_normals = (normals @ T[:3, :3].T).astype(np.float32)  # normals in the source frame
    result = reg.register_rigid(
        source,
        (verts, faces),
        max_iters=100,
        tol=1e-6,
        variant="symmetric",
        source_normals=rotated_normals,
        device=device,
    )
    wp.synchronize_device()

    test.assertTrue(result.converged)
    test.assertLess(_rotation_error_deg(result.transform, np.linalg.inv(T)), 0.1)


def test_rejects_unknown_variant(test, device):
    verts, faces = _icosphere(subdiv=2, scale=(1.2, 1.0, 0.9))
    with test.assertRaises(ValueError):
        reg.register_rigid(verts, (verts, faces), variant="two_plane", device=device)


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


def test_graph_capture_loop(test, device):
    # The ICP iteration (accumulate + on-device solve) touches no host memory, so
    # a fixed-iteration loop can be captured into a CUDA graph and replayed. This
    # verifies there is no hidden per-iteration host sync.
    from warp._src.geometry import _icp_accumulate_mesh_kernel, _icp_solve_kernel  # noqa: PLC0415

    verts, faces = _icosphere(subdiv=3, scale=(1.5, 1.0, 0.7))
    T = _rigid_transform(12.0, np.array([0.05, -0.03, 0.04]), seed=3)
    source = (verts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    src = wp.array(source, dtype=wp.vec3, device=device)
    n = src.shape[0]
    rots = wp.array(np.eye(3)[None].astype(np.float32), dtype=wp.mat33, device=device)
    transs = wp.zeros(1, dtype=wp.vec3, device=device)
    inv_scale = wp.zeros(1, dtype=wp.float32, device=device)
    update_sq = wp.zeros(1, dtype=wp.float32, device=device)
    src_normals = wp.zeros(1, dtype=wp.vec3, device=device)
    a_upper = wp.zeros(21, dtype=wp.float32, device=device)
    g = wp.zeros(6, dtype=wp.float32, device=device)
    stats = wp.zeros(2, dtype=wp.float32, device=device)

    def one_iter():
        a_upper.zero_()
        g.zero_()
        stats.zero_()
        wp.launch(
            _icp_accumulate_mesh_kernel,
            dim=n,
            inputs=[src, rots, transs, mesh.id, 1.0e30, n, 0, 0, inv_scale, src_normals, 0, 0],
            outputs=[a_upper, g, stats],
            device=device,
        )
        wp.launch(
            _icp_solve_kernel,
            dim=1,
            inputs=[a_upper, g, stats, 1.0e-9, 3.0, 0],
            outputs=[rots, transs, inv_scale, update_sq],
            device=device,
        )

    # Warm up (compile/load the kernels) outside the capture, then reset state.
    one_iter()
    wp.synchronize_device(device)
    rots.assign(np.eye(3)[None].astype(np.float32))
    transs.zero_()
    inv_scale.zero_()

    with wp.ScopedCapture(device) as capture:
        for _ in range(40):
            one_iter()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    transform = np.eye(4)
    transform[:3, :3] = rots.numpy()[0]
    transform[:3, 3] = transs.numpy()[0]
    test.assertLess(_rotation_error_deg(transform, np.linalg.inv(T)), 0.1)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


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
add_function_test(TestRegistration, "test_robust_rejects_outliers", test_robust_rejects_outliers, devices=devices)
add_function_test(
    TestRegistration, "test_stochastic_subsampling_recovers", test_stochastic_subsampling_recovers, devices=devices
)
add_function_test(TestRegistration, "test_rejects_unknown_robust", test_rejects_unknown_robust, devices=devices)
add_function_test(TestRegistration, "test_plane_normal_closest_point", test_plane_normal_closest_point, devices=devices)
add_function_test(
    TestRegistration, "test_rejects_unknown_plane_normal", test_rejects_unknown_plane_normal, devices=devices
)
add_function_test(TestRegistration, "test_batched_multi_init", test_batched_multi_init, devices=devices)
add_function_test(TestRegistration, "test_batched_multi_source", test_batched_multi_source, devices=devices)
add_function_test(TestRegistration, "test_batched_validates_inits", test_batched_validates_inits, devices=devices)
add_function_test(TestRegistration, "test_symmetric_recovers", test_symmetric_recovers, devices=devices)
add_function_test(
    TestRegistration, "test_symmetric_with_source_normals", test_symmetric_with_source_normals, devices=devices
)
add_function_test(TestRegistration, "test_rejects_unknown_variant", test_rejects_unknown_variant, devices=devices)
add_function_test(TestRegistration, "test_identity_when_aligned", test_identity_when_aligned, devices=devices)
add_function_test(TestRegistration, "test_deterministic", test_deterministic, devices=devices)
add_function_test(TestRegistration, "test_reuses_target_mesh", test_reuses_target_mesh, devices=devices)
add_function_test(TestRegistration, "test_graph_capture_loop", test_graph_capture_loop, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
