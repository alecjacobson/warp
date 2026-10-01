# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DEC operators: the d0/star1 factorization of the cotangent Laplacian."""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp.geometry
import warp.sparse
from warp.tests.unittest_utils import *

# A regular octahedron: a closed genus-0 triangle mesh (6 verts, 12 edges, 8 faces).
_OCTAHEDRON_POINTS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
_OCTAHEDRON_INDICES = np.array(
    [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int32
)

# A unit cube split into six tetrahedra.
_TET_CUBE_POINTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32
)
_TET_CUBE_INDICES = np.array(
    [[0, 5, 7, 4], [0, 1, 7, 5], [1, 6, 7, 5], [0, 7, 2, 3], [0, 7, 1, 2], [1, 7, 6, 2]], dtype=np.int32
)


def _bsr_to_dense(mat) -> np.ndarray:
    dense = np.zeros(mat.shape, dtype=np.float64)
    rows = mat.uncompress_rows().numpy()
    cols = mat.columns.numpy()
    vals = mat.values.numpy().reshape(-1)
    for k in range(mat.nnz_sync()):
        dense[rows[k], cols[k]] += vals[k]
    return dense


@wp.kernel
def _weighted_value_sum(
    values: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    count: int,
    out_loss: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k < count:
        wp.atomic_add(out_loss, 0, values[k] * weights[k])


def _reconstruct(d0, star1) -> np.ndarray:
    D0 = _bsr_to_dense(d0)
    return D0.T @ np.diag(star1.numpy()) @ D0


def test_dec_identity_triangle(test, device):
    """``d0^T star1 d0`` reproduces the cotangent Laplacian of a triangle mesh."""
    rng = np.random.default_rng(1)
    points_np = _OCTAHEDRON_POINTS + 0.05 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    laplacian = _bsr_to_dense(warp.geometry.laplacian(points, indices))
    assert_np_equal(_reconstruct(d0, star1).astype(np.float32), laplacian.astype(np.float32), tol=1e-4)


def test_dec_identity_tet(test, device):
    """``d0^T star1 d0`` reproduces the cotangent Laplacian of a tetrahedral mesh."""
    rng = np.random.default_rng(2)
    points_np = _TET_CUBE_POINTS + 0.05 * rng.standard_normal(_TET_CUBE_POINTS.shape).astype(np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_TET_CUBE_INDICES, dtype=wp.vec4i, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    laplacian = _bsr_to_dense(warp.geometry.laplacian(points, indices, simplex_size=4))
    assert_np_equal(_reconstruct(d0, star1).astype(np.float32), laplacian.astype(np.float32), tol=1e-4)


def test_dec_d0_structure_and_euler(test, device):
    """d0 is a signed incidence matrix and its edge count satisfies Euler's formula."""
    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    num_vertices = _OCTAHEDRON_POINTS.shape[0]
    num_faces = _OCTAHEDRON_INDICES.shape[0]
    num_edges = d0.shape[0]

    test.assertEqual(d0.shape, (num_edges, num_vertices))
    test.assertEqual(star1.shape, (num_edges,))
    # Closed genus-0 surface: V - E + F = 2.
    test.assertEqual(num_vertices - num_edges + num_faces, 2)

    D0 = _bsr_to_dense(d0)
    # Every row has exactly one +1 and one -1, so rows sum to zero.
    assert_np_equal(D0.sum(axis=1).astype(np.float32), np.zeros(num_edges, np.float32), tol=1e-8)
    test.assertTrue(np.all((D0 != 0).sum(axis=1) == 2))
    test.assertTrue(np.all(np.sort(D0[D0 != 0].reshape(num_edges, 2), axis=1)[:, 0] == -1.0))


def test_dec_dispatch(test, device):
    """Flat and typed index arrays produce the same operators."""
    points = wp.array(_TET_CUBE_POINTS, dtype=wp.vec3, device=device)
    flat = wp.array(_TET_CUBE_INDICES.flatten(), dtype=wp.int32, device=device)
    typed = wp.array(_TET_CUBE_INDICES, dtype=wp.vec4i, device=device)

    d0_flat, s1_flat = warp.geometry.dec_operators(points, flat, simplex_size=4)
    d0_typed, s1_typed = warp.geometry.dec_operators(points, typed)
    assert_np_equal(_bsr_to_dense(d0_flat), _bsr_to_dense(d0_typed), tol=1e-7)
    assert_np_equal(s1_flat.numpy(), s1_typed.numpy(), tol=1e-7)


def test_dec_boundary_and_obtuse_robustness(test, device):
    """Open meshes and obtuse triangles still factor the Laplacian exactly."""
    # Two triangles sharing edge (1, 2). The first is very obtuse (~155 degrees at
    # vertex 2), so the boundary edge (0, 1) opposite it gets a negative weight,
    # and several edges belong to a single triangle, giving the mesh a boundary.
    points_np = np.array([[0, 0, 0], [3, 0, 0], [1, 0.3, 0], [2, 1.5, 0]], dtype=np.float32)
    tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    star1_values = star1.numpy()

    test.assertTrue(np.isfinite(star1_values).all())
    # The obtuse triangle drives at least one edge weight negative, exactly as the
    # off-diagonal Laplacian entries go negative there.
    test.assertLess(star1_values.min(), 0.0)
    # The factorization is algebraic, so it holds regardless of sign.
    laplacian = _bsr_to_dense(warp.geometry.laplacian(points, indices))
    assert_np_equal(_reconstruct(d0, star1).astype(np.float32), laplacian.astype(np.float32), tol=1e-4)


def test_dec_star1_deterministic(test, device):
    """star1 is bit-for-bit reproducible, as its weights are summed by a sorted dedup."""
    rng = np.random.default_rng(5)
    points_np = _OCTAHEDRON_POINTS + 0.1 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    _, first = warp.geometry.dec_operators(points, indices)
    reference = first.numpy().copy()
    for _ in range(4):
        _, again = warp.geometry.dec_operators(points, indices)
        # Bit-for-bit, not merely close: the sorted accumulation is deterministic.
        test.assertTrue(np.array_equal(again.numpy(), reference))


def test_dec_star1_gradient(test, device):
    """Gradients flow through star1 to points and match finite differences."""
    rng = np.random.default_rng(3)
    points_np = _OCTAHEDRON_POINTS + 0.05 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    _, probe = warp.geometry.dec_operators(wp.array(points_np, dtype=wp.vec3, device=device), indices)
    count = probe.shape[0]
    weights = wp.array(rng.standard_normal(count).astype(np.float32), dtype=wp.float32, device=device)

    def loss_of(positions_np, tape=None):
        points = wp.array(positions_np, dtype=wp.vec3, device=device, requires_grad=tape is not None)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=tape is not None)
        with tape if tape is not None else contextlib.nullcontext():
            _, star1 = warp.geometry.dec_operators(points, indices)
            wp.launch(_weighted_value_sum, dim=count, inputs=[star1, weights, count], outputs=[loss], device=device)
        return points, loss

    tape = wp.Tape()
    points, loss = loss_of(points_np, tape)
    tape.backward(loss=loss)
    analytic = points.grad.numpy()

    test.assertTrue(np.isfinite(analytic).all())
    test.assertGreater(np.abs(analytic).max(), 0.0)

    eps = 1e-3
    numeric = np.zeros_like(points_np)
    for i in range(points_np.shape[0]):
        for c in range(3):
            forward = points_np.copy()
            backward = points_np.copy()
            forward[i, c] += eps
            backward[i, c] -= eps
            numeric[i, c] = (loss_of(forward)[1].numpy()[0] - loss_of(backward)[1].numpy()[0]) / (2.0 * eps)

    assert_np_equal(analytic, numeric, tol=2e-2)


devices = get_test_devices()


class TestGeometryDEC(unittest.TestCase):
    pass


add_function_test(TestGeometryDEC, "test_dec_identity_triangle", test_dec_identity_triangle, devices=devices)
add_function_test(TestGeometryDEC, "test_dec_identity_tet", test_dec_identity_tet, devices=devices)
add_function_test(TestGeometryDEC, "test_dec_d0_structure_and_euler", test_dec_d0_structure_and_euler, devices=devices)
add_function_test(TestGeometryDEC, "test_dec_dispatch", test_dec_dispatch, devices=devices)
add_function_test(
    TestGeometryDEC, "test_dec_boundary_and_obtuse_robustness", test_dec_boundary_and_obtuse_robustness, devices=devices
)
add_function_test(TestGeometryDEC, "test_dec_star1_deterministic", test_dec_star1_deterministic, devices=devices)
add_function_test(TestGeometryDEC, "test_dec_star1_gradient", test_dec_star1_gradient, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
