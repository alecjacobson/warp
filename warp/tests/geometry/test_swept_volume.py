# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense-stamping swept volume (motion envelope) of animated rigid meshes."""

import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *


def _sphere_mesh(device, radius=0.5, subdivisions=3, support_winding_number=False):
    points, indices = U.icosphere(subdivisions=subdivisions, radius=radius)
    return wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=wp.int32, device=device),
        support_winding_number=support_winding_number,
    )


def _translation_transforms(offsets):
    """Build a ``(num_meshes, num_samples, 7)`` transform array of pure translations."""
    offsets = np.asarray(offsets, dtype=np.float32)
    tr = np.zeros((*offsets.shape[:-1], 7), dtype=np.float32)
    tr[..., 0:3] = offsets
    tr[..., 6] = 1.0  # identity quaternion (x, y, z, w)
    return tr


def _capsule_sdf(P, a, b, radius):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - np.outer(h, ba), axis=1) - radius


def _grid_points(lower, upper, dims):
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    spacing = (upper - lower) / (np.array(dims) - 1)
    idx = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), axis=-1).reshape(-1, 3)
    return lower + idx * spacing, idx, spacing


def test_swept_sphere_is_a_capsule(test, device):
    # A sphere of radius r translated along the x axis from -1 to 1 sweeps a
    # capsule: a segment [-1, 1] on x, dilated by r. The dense field should match
    # the analytic capsule SDF to within the icosphere's polygonal error plus one
    # voxel.
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius, subdivisions=3)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=0.05, device=device)
    wp.synchronize_device()

    fnp = field.numpy()
    P, idx, _ = _grid_points(lower, upper, fnp.shape)
    d_ref = _capsule_sdf(P, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius)
    d_num = fnp[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Restrict the comparison to the near-surface band, where a signed distance
    # is meaningful (the max_dist plateau far away is expected to differ).
    band = np.abs(d_ref) < 0.3
    np.testing.assert_allclose(d_num[band], d_ref[band], atol=0.03)


def test_swept_sphere_mesh_bounds(test, device):
    # The extracted envelope of the swept sphere should span the capsule's AABB.
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    verts, indices = geo.swept_volume([mesh], tr, voxel_size=0.05, device=device)
    wp.synchronize_device()

    v = verts.numpy()
    test.assertGreater(len(v), 0)
    test.assertEqual(len(indices.numpy()) % 3, 0)
    np.testing.assert_allclose(v.min(axis=0), [-1.0 - radius, -radius, -radius], atol=0.06)
    np.testing.assert_allclose(v.max(axis=0), [1.0 + radius, radius, radius], atol=0.06)


def test_static_single_pose_matches_mesh(test, device):
    # With a single identity pose the field reduces to the mesh's own SDF, so the
    # zero isosurface should recover a sphere of the input radius.
    radius = 0.7
    mesh = _sphere_mesh(device, radius=radius, subdivisions=3)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    verts, _ = geo.swept_volume([mesh], tr, voxel_size=0.05, device=device)
    wp.synchronize_device()

    v = verts.numpy()
    r = np.linalg.norm(v - v.mean(axis=0), axis=1)
    np.testing.assert_allclose(r.mean(), radius, atol=0.05)
    test.assertLess(r.std(), 0.05)


def test_union_of_two_static_spheres(test, device):
    # Two separated spheres yield two connected components and a field that is the
    # per-sphere minimum. Sample the midpoint region to confirm the union.
    radius = 0.4
    left = _sphere_mesh(device, radius=radius)
    right = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[-1.5, 0.0, 0.0]], [[1.5, 0.0, 0.0]]])

    field, lower, upper = geo.swept_volume_field([left, right], tr, voxel_size=0.05, device=device)
    wp.synchronize_device()

    fnp = field.numpy()
    P, idx, _ = _grid_points(lower, upper, fnp.shape)
    d_num = fnp[idx[:, 0], idx[:, 1], idx[:, 2]]
    d_left = np.linalg.norm(P - np.array([-1.5, 0.0, 0.0]), axis=1) - radius
    d_right = np.linalg.norm(P - np.array([1.5, 0.0, 0.0]), axis=1) - radius
    d_ref = np.minimum(d_left, d_right)

    band = np.abs(d_ref) < 0.3
    np.testing.assert_allclose(d_num[band], d_ref[band], atol=0.03)


def test_conservative_encloses_all_poses(test, device):
    # Every input vertex, at every sampled pose, must lie inside the envelope:
    # the field there must be non-positive (within one voxel of the surface).
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    points = mesh.points.numpy().astype(np.float64)
    offsets = np.array([[x, 0.3 * x, 0.0] for x in np.linspace(-1.0, 1.0, 11)], dtype=np.float32)
    tr = _translation_transforms([offsets])

    voxel = 0.05
    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=voxel, device=device)
    wp.synchronize_device()
    fnp = field.numpy()
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    dims = np.array(fnp.shape)
    spacing = (upper - lower) / (dims - 1)

    # Trilinearly-cheap nearest-node lookup for each posed vertex.
    posed = (points[None, :, :] + offsets[:, None, :3]).reshape(-1, 3)
    node = np.rint((posed - lower) / spacing).astype(int)
    node = np.clip(node, 0, dims - 1)
    d = fnp[node[:, 0], node[:, 1], node[:, 2]]
    # Interior/boundary vertices should be at or inside the surface, allowing the
    # nearest-node rounding of up to half a voxel diagonal.
    test.assertLessEqual(float(d.max()), 0.5 * float(np.linalg.norm(spacing)) + 1e-4)


def test_rotation_pose(test, device):
    # An off-center sphere rotated 90 degrees about z sweeps a quarter-annulus;
    # the envelope's radial extent must reach the rotated positions. This checks
    # that the quaternion inverse-transform path is correct.
    radius = 0.3
    # Rest-pose sphere centered at +x, so a rotation about z sweeps an arc.
    points, indices = U.icosphere(subdivisions=2, radius=radius, center=(1.0, 0.0, 0.0))
    mesh = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=wp.int32, device=device),
    )

    angles = np.linspace(0.0, np.pi / 2.0, 13)
    tr = np.zeros((1, len(angles), 7), dtype=np.float32)
    for s, a in enumerate(angles):
        # Rotate the rest-pose sphere (centered at +x) about z by angle a.
        tr[0, s, 6] = np.cos(a / 2.0)
        tr[0, s, 5] = np.sin(a / 2.0)  # quaternion z component

    verts, _ = geo.swept_volume([mesh], tr, voxel_size=0.05, device=device)
    wp.synchronize_device()
    v = verts.numpy()
    # The sphere center traces an arc of radius 1 from +x to +y; the envelope must
    # span both extremes in x and y.
    test.assertGreater(v[:, 0].max(), 1.0 + radius - 0.06)
    test.assertGreater(v[:, 1].max(), 1.0 + radius - 0.06)


def test_winding_number_sign_mode(test, device):
    # The winding-number classifier should give the same capsule as the normal
    # classifier for a watertight sphere.
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius, support_winding_number=True)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    verts, _ = geo.swept_volume(
        [mesh], tr, voxel_size=0.05, sign_mode=geo.SweptVolumeSign.WINDING_NUMBER, device=device
    )
    wp.synchronize_device()
    v = verts.numpy()
    np.testing.assert_allclose(v.min(axis=0), [-1.0 - radius, -radius, -radius], atol=0.06)
    np.testing.assert_allclose(v.max(axis=0), [1.0 + radius, radius, radius], atol=0.06)


def test_resolution_argument(test, device):
    # Passing an explicit resolution should produce a field of that shape.
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])
    field, _, _ = geo.swept_volume_field([mesh], tr, resolution=(16, 20, 24), device=device)
    wp.synchronize_device()
    test.assertEqual(field.shape, (16, 20, 24))


def test_invalid_arguments(test, device):
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])
    with test.assertRaises(ValueError):
        geo.swept_volume_field([], tr, voxel_size=0.05, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, device=device)  # no voxel_size or resolution
    with test.assertRaises(ValueError):
        # Two rows of transforms but only one mesh.
        geo.swept_volume_field(
            [mesh], _translation_transforms([[[0.0] * 3], [[0.0] * 3]]), voxel_size=0.1, device=device
        )


devices = get_test_devices()


class TestSweptVolume(unittest.TestCase):
    pass


add_function_test(TestSweptVolume, "test_swept_sphere_is_a_capsule", test_swept_sphere_is_a_capsule, devices=devices)
add_function_test(TestSweptVolume, "test_swept_sphere_mesh_bounds", test_swept_sphere_mesh_bounds, devices=devices)
add_function_test(
    TestSweptVolume, "test_static_single_pose_matches_mesh", test_static_single_pose_matches_mesh, devices=devices
)
add_function_test(
    TestSweptVolume, "test_union_of_two_static_spheres", test_union_of_two_static_spheres, devices=devices
)
add_function_test(
    TestSweptVolume, "test_conservative_encloses_all_poses", test_conservative_encloses_all_poses, devices=devices
)
add_function_test(TestSweptVolume, "test_rotation_pose", test_rotation_pose, devices=devices)
add_function_test(TestSweptVolume, "test_winding_number_sign_mode", test_winding_number_sign_mode, devices=devices)
add_function_test(TestSweptVolume, "test_resolution_argument", test_resolution_argument, devices=devices)
add_function_test(TestSweptVolume, "test_invalid_arguments", test_invalid_arguments, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
