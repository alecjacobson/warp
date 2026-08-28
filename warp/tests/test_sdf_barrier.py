# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the point-mesh signed-distance derivatives in the SDF barrier example.

``signed_distance_derivs`` returns the signed distance to a mesh, its gradient,
and its feature-classified Hessian (face/edge/vertex). These tests validate:

* value and gradient of the mesh signed distance, against a finite-difference
  reference. These are exact and feature-independent.
* the Hessian is zero on a flat face, symmetric, and matches independent analytic
  ground truth at face / edge / vertex hits on a cube.

Two ``expectedFailure`` tests pin the known limitation that barycentric
classification recovers the triangulation simplex, not the geometric feature.
"""

import unittest

import numpy as np

import warp as wp
from warp.examples.optim import example_sdf_barrier as ex
from warp.tests.unittest_utils import *

MAX_DIST = 1.0e6


# ----------------------------------------------------------------------------
# Meshes built procedurally so the tests do not depend on USD assets.
# ----------------------------------------------------------------------------


def _plane_mesh(device):
    """A flat +z-facing quad in the z = 0 plane, spanning [-10, 10]^2."""
    points = wp.array(
        [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]],
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices)


def _flat_fan_mesh(device):
    """A flat +z-facing disk: a center vertex with four coplanar triangles around
    it. The center is a zero-defect (flat) interior vertex."""
    points = wp.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1], dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices)


def _cube_mesh(device):
    """The unit cube [0, 1]^3 as 12 triangles (from the wp.mesh_get docs)."""
    points = wp.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array(
        [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=wp.int32,
        device=device,
    )
    return wp.Mesh(points=points, indices=indices)


# ----------------------------------------------------------------------------
# Kernels used only by the tests.
# ----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _sdf_derivs_kernel(
    points: wp.array[wp.vec3],
    mesh: wp.uint64,
    max_dist: float,
    valid: wp.array[wp.int32],
    value: wp.array[float],
    grad: wp.array[wp.vec3],
    hess: wp.array[wp.mat33],
):
    i = wp.tid()
    hit, v, g, H = ex.signed_distance_derivs(mesh, points[i], max_dist)
    valid[i] = hit
    value[i] = v
    grad[i] = g
    hess[i] = H


@wp.kernel(enable_backward=False)
def _sdf_value_kernel(
    points: wp.array[wp.vec3],
    mesh: wp.uint64,
    max_dist: float,
    out: wp.array[float],
):
    # Signed distance only, for the finite-difference reference. Uses the same
    # mesh query as the example but no jets.
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, points[i], max_dist)
    if not q.result:
        out[i] = 1.0e30
        return
    c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    out[i] = q.sign * wp.length(points[i] - c)


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def _run_example(points_np, mesh, device):
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    valid = wp.zeros(n, dtype=wp.int32, device=device)
    value = wp.zeros(n, dtype=float, device=device)
    grad = wp.zeros(n, dtype=wp.vec3, device=device)
    hess = wp.zeros(n, dtype=wp.mat33, device=device)
    wp.launch(
        _sdf_derivs_kernel,
        dim=n,
        inputs=[points, mesh.id, MAX_DIST],
        outputs=[valid, value, grad, hess],
        device=device,
    )
    wp.synchronize_device(device)
    return valid.numpy(), value.numpy(), grad.numpy(), hess.numpy()


def _sdf_values(points_np, mesh, device):
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    out = wp.zeros(points_np.shape[0], dtype=float, device=device)
    wp.launch(
        _sdf_value_kernel, dim=points_np.shape[0], inputs=[points, mesh.id, MAX_DIST], outputs=[out], device=device
    )
    wp.synchronize_device(device)
    return out.numpy()


def _fd_gradient(points_np, mesh, device, h=1.0e-2):
    """Central-difference gradient of the mesh signed distance."""
    grad = np.empty_like(points_np)
    for k in range(3):
        step = np.zeros_like(points_np)
        step[:, k] = h
        grad[:, k] = (_sdf_values(points_np + step, mesh, device) - _sdf_values(points_np - step, mesh, device)) / (
            2.0 * h
        )
    return grad


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------


def test_plane_value_and_gradient(test, device):
    mesh = _plane_mesh(device)
    rng = np.random.default_rng(1)
    n = 256
    # Points safely over the face interior, away from edges and the surface.
    xy = rng.uniform(-3.0, 3.0, size=(n, 2))
    z = rng.uniform(0.5, 3.0, size=(n, 1)) * rng.choice([-1.0, 1.0], size=(n, 1))
    points = np.hstack([xy, z]).astype(np.float32)

    valid, value, grad, _ = _run_example(points, mesh, device)
    test.assertTrue(np.all(valid == 1))

    # Signed distance to the +z-facing plane is z, so its gradient is the
    # constant outward normal +z_hat on both sides (it increases toward +z).
    np.testing.assert_allclose(value, points[:, 2], rtol=1.0e-4, atol=1.0e-4)
    expected_grad = np.zeros_like(points)
    expected_grad[:, 2] = 1.0
    np.testing.assert_allclose(grad, expected_grad, rtol=1.0e-4, atol=1.0e-4)

    # Independent finite-difference check of the gradient.
    np.testing.assert_allclose(grad, _fd_gradient(points, mesh, device), rtol=1.0e-3, atol=1.0e-3)


def test_plane_hessian_zero(test, device):
    mesh = _plane_mesh(device)
    rng = np.random.default_rng(2)
    n = 256
    xy = rng.uniform(-3.0, 3.0, size=(n, 2))
    z = rng.uniform(0.5, 3.0, size=(n, 1)) * rng.choice([-1.0, 1.0], size=(n, 1))
    points = np.hstack([xy, z]).astype(np.float32)

    valid, _, _, hess = _run_example(points, mesh, device)
    test.assertTrue(np.all(valid == 1))
    # A flat face has a linear distance field: the Hessian is exactly zero.
    np.testing.assert_allclose(hess, 0.0, atol=1.0e-5)


def test_cube_hessian_symmetric(test, device):
    mesh = _cube_mesh(device)
    rng = np.random.default_rng(3)
    n = 512
    points = rng.uniform(-1.0, 2.0, size=(n, 3)).astype(np.float32)

    valid, _, _, hess = _run_example(points, mesh, device)
    hess = hess[valid == 1]
    test.assertGreater(hess.shape[0], 0)
    # The Hessian is (s/dist)(P - T) with symmetric P and T, so it stays symmetric.
    np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-5, atol=1.0e-5)


def test_cube_feature_hessians(test, device):
    """Feature-classified Hessian at face / edge / vertex hits vs. analytic ground truth.

    Points are placed in the Voronoi region of a single cube feature, where the
    mesh signed distance equals a distance to that feature with a known Hessian:

    * face   -> distance to a plane        -> Hessian 0
    * edge   -> distance to a line         -> (1/d)(I - n nᵀ - t tᵀ)
    * vertex -> distance to a point        -> (1/d)(I - n nᵀ)
    """
    mesh = _cube_mesh(device)

    # (query point, closest point, edge tangent or None for face/vertex)
    # The face point avoids x == y: that line is the top face's triangulation
    # diagonal, a coplanar internal edge the barycentric classifier would (wrongly)
    # read as a crease. See feature_tangent_projector's limitation note.
    face = (np.array([0.3, 0.6, 1.5]), np.array([0.3, 0.6, 1.0]), None)
    edge = (np.array([0.5, 1.4, 1.3]), np.array([0.5, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    vertex = (np.array([1.4, 1.3, 1.2]), np.array([1.0, 1.0, 1.0]), None)

    points = np.array([face[0], edge[0], vertex[0]], dtype=np.float32)
    valid, _, _, hess = _run_example(points, mesh, device)
    test.assertTrue(np.all(valid == 1))

    for i, (p, c, tangent) in enumerate((face, edge, vertex)):
        r = p - c
        d = np.linalg.norm(r)
        n = r / d
        expected = (np.eye(3) - np.outer(n, n)) / d  # vertex form
        if tangent is not None:
            expected -= np.outer(tangent, tangent) / d  # edge removes the tangent
        elif i == 0:
            expected = np.zeros((3, 3))  # face is flat
        np.testing.assert_allclose(hess[i], expected, rtol=1.0e-4, atol=1.0e-4)


devices = get_test_devices()


class TestSDFBarrier(unittest.TestCase):
    # Known limitations: barycentric classification recovers the triangulation
    # simplex, not the geometric feature, so it gives spurious curvature on a flat
    # feature (a collapsed normal cone). These assert the geometrically-correct
    # zero Hessian and are marked expectedFailure -- they pin the current spurious
    # behavior and would flip to a failure if the classifier is ever fixed. Both
    # only occur on a measure-zero set of query points (probability zero for
    # generic sampling).

    @unittest.expectedFailure
    def test_coplanar_diagonal_spurious_edge(self):
        # (0.5, 0.5, *) is directly above the cube top's triangulation diagonal
        # (0,0,1)->(1,1,1): a coplanar edge, so the true Hessian is zero, but the
        # classifier reads an edge and returns rank-1 curvature.
        mesh = _cube_mesh("cpu")
        _, _, _, hess = _run_example(np.array([[0.5, 0.5, 1.5]], dtype=np.float32), mesh, "cpu")
        np.testing.assert_allclose(hess[0], np.zeros((3, 3)), atol=1.0e-4)

    @unittest.expectedFailure
    def test_flat_vertex_spurious_rank2(self):
        # A point above the flat fan's center vertex projects onto that zero-defect
        # vertex, so the true Hessian is zero, but the classifier reads a vertex and
        # returns rank-2 curvature.
        mesh = _flat_fan_mesh("cpu")
        _, _, _, hess = _run_example(np.array([[0.0, 0.0, 0.5]], dtype=np.float32), mesh, "cpu")
        np.testing.assert_allclose(hess[0], np.zeros((3, 3)), atol=1.0e-4)


add_function_test(TestSDFBarrier, "test_plane_value_and_gradient", test_plane_value_and_gradient, devices=devices)
add_function_test(TestSDFBarrier, "test_plane_hessian_zero", test_plane_hessian_zero, devices=devices)
add_function_test(TestSDFBarrier, "test_cube_hessian_symmetric", test_cube_hessian_symmetric, devices=devices)
add_function_test(TestSDFBarrier, "test_cube_feature_hessians", test_cube_feature_hessians, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
