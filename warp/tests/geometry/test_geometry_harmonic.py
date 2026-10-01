# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harmonic and biharmonic solves: known solutions, dense references, and constraints."""

import unittest

import numpy as np

import warp as wp
import warp.geometry
import warp.sparse
from warp.tests.unittest_utils import *


def _bsr_to_dense(mat) -> np.ndarray:
    dense = np.zeros(mat.shape, dtype=np.float64)
    rows = mat.uncompress_rows().numpy()
    cols = mat.columns.numpy()
    vals = mat.values.numpy().reshape(-1)
    for k in range(mat.nnz_sync()):
        dense[rows[k], cols[k]] += vals[k]
    return dense


def _flat_grid(n: int):
    """Build a flat ``n`` by ``n`` grid of right triangles in the z=0 plane, with its boundary loop."""
    x = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    points = np.stack([xx, yy, np.zeros_like(xx)], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    lower, upper, right, above = idx[:-1, :-1], idx[1:, 1:], idx[1:, :-1], idx[:-1, 1:]
    tris = np.concatenate(
        [np.stack([lower, right, upper], -1).reshape(-1, 3), np.stack([lower, upper, above], -1).reshape(-1, 3)]
    ).astype(np.int32)
    boundary = sorted(set(idx[[0, -1], :].ravel().tolist()) | set(idx[1:-1, [0, -1]].ravel().tolist()))
    return points, tris, np.array(boundary, dtype=np.int32)


def _grid_tets(n: int):
    """Build an ``n`` by ``n`` by ``n`` grid of cubes split into tetrahedra, with its surface vertices."""
    coords = np.arange(n + 1)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    points = (np.stack([gx, gy, gz], -1).reshape(-1, 3) / n).astype(np.float32)

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    cube = np.array(
        [[0, 5, 7, 4], [0, 1, 7, 5], [1, 6, 7, 5], [0, 7, 2, 3], [0, 7, 1, 2], [1, 7, 6, 2]], dtype=np.int64
    )
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    base = np.stack([ii, jj, kk], -1).reshape(-1, 3)
    corner_ids = np.array([[vid(*(b + c)) for c in corner] for b in base])
    tets = np.concatenate([corner_ids[:, t] for t in cube], 0).astype(np.int32)

    g = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    on_surface = np.any((g == 0) | (g == n), axis=1)
    boundary = np.nonzero(on_surface)[0].astype(np.int32)
    return points, tets, boundary


def _dense_min_quad(Q: np.ndarray, boundary: np.ndarray, bc: np.ndarray, n: int) -> np.ndarray:
    interior = np.setdiff1d(np.arange(n), boundary)
    z = np.zeros(n)
    z[boundary] = bc
    z[interior] = np.linalg.solve(Q[np.ix_(interior, interior)], -Q[np.ix_(interior, boundary)] @ bc)
    return z


def test_harmonic_reproduces_linear(test, device):
    """A harmonic (k=1) solve reproduces linear boundary data everywhere."""
    points_np, tris, boundary_np = _flat_grid(10)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    linear = (0.3 * points_np[:, 0] + 0.7 * points_np[:, 1] + 0.1).astype(np.float32)
    bc = wp.array(linear[boundary_np], dtype=wp.float32, device=device)

    z = warp.geometry.harmonic(points, indices, boundary, bc, k=1).numpy()
    assert_np_equal(z, linear, tol=1e-4)
    # Boundary values are imposed exactly, not merely approximated.
    assert_np_equal(z[boundary_np], linear[boundary_np], tol=1e-8)


def test_harmonic_matches_dense_reference(test, device):
    """Harmonic and biharmonic solves match a dense min-quadratic-with-fixed reference."""
    points_np, tris, boundary_np = _flat_grid(10)
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    L = _bsr_to_dense(warp.geometry.laplacian(points, indices))
    # k=2 uses the Voronoi mass internally, matching igl::harmonic, so the dense
    # reference must use the same lumping.
    voronoi_diag = np.diag(
        _bsr_to_dense(warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI))
    )

    data = (np.sin(3.0 * points_np[:, 0]) * np.cos(2.0 * points_np[:, 1])).astype(np.float32)
    bc = wp.array(data[boundary_np], dtype=wp.float32, device=device)

    z1 = warp.geometry.harmonic(points, indices, boundary, bc, k=1).numpy()
    ref1 = _dense_min_quad(L, boundary_np, data[boundary_np], n)
    assert_np_equal(z1, ref1.astype(np.float32), tol=1e-4)

    # The biharmonic solve is double precision, so it tracks the double-precision
    # dense reference closely (the residual floor is CG convergence, not rounding).
    z2 = warp.geometry.harmonic(points, indices, boundary, bc, k=2).numpy()
    ref2 = _dense_min_quad(L @ np.diag(1.0 / voronoi_diag) @ L, boundary_np, data[boundary_np], n)
    assert_np_equal(z2, ref2.astype(np.float32), tol=1e-4)

    # The two powers give genuinely different operators, so their solutions differ.
    test.assertGreater(np.abs(z1 - z2).max(), 1e-3)


def test_harmonic_multiple_columns(test, device):
    """Solving several functions at once matches solving each on its own."""
    points_np, tris, boundary_np = _flat_grid(8)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    f = (0.3 * points_np[:, 0] + 0.7 * points_np[:, 1]).astype(np.float32)
    g = (np.sin(3.0 * points_np[:, 0])).astype(np.float32)

    z_f = warp.geometry.harmonic(points, indices, boundary, wp.array(f[boundary_np], dtype=wp.float32, device=device))
    z_g = warp.geometry.harmonic(points, indices, boundary, wp.array(g[boundary_np], dtype=wp.float32, device=device))

    stacked = np.stack([f[boundary_np], g[boundary_np]], axis=1).astype(np.float32)
    Z = warp.geometry.harmonic(points, indices, boundary, wp.array(stacked, dtype=wp.float32, device=device))

    test.assertEqual(Z.shape, (points_np.shape[0], 2))
    assert_np_equal(Z.numpy()[:, 0], z_f.numpy(), tol=1e-5)
    assert_np_equal(Z.numpy()[:, 1], z_g.numpy(), tol=1e-5)


def test_harmonic_tetrahedral(test, device):
    """A harmonic solve on a tetrahedral mesh reproduces linear boundary data."""
    points_np, tets, boundary_np = _grid_tets(3)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tets, dtype=wp.vec4i, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    linear = (0.5 * points_np[:, 0] - 0.2 * points_np[:, 1] + 0.9 * points_np[:, 2] + 0.3).astype(np.float32)
    bc = wp.array(linear[boundary_np], dtype=wp.float32, device=device)

    z = warp.geometry.harmonic(points, indices, boundary, bc, k=1).numpy()
    assert_np_equal(z, linear, tol=1e-4)


def test_harmonic_validation(test, device):
    """Bad arguments are rejected."""
    points_np, tris, boundary_np = _flat_grid(4)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)
    bc = wp.array(np.zeros(boundary_np.shape[0], dtype=np.float32), dtype=wp.float32, device=device)

    with test.assertRaisesRegex(ValueError, "`k` must be at least 1"):
        warp.geometry.harmonic(points, indices, boundary, bc, k=0)

    with test.assertRaisesRegex(ValueError, "disagree|has .* rows but"):
        wrong = wp.array(np.zeros(boundary_np.shape[0] + 1, dtype=np.float32), dtype=wp.float32, device=device)
        warp.geometry.harmonic(points, indices, boundary, wrong)


devices = get_test_devices()


class TestGeometryHarmonic(unittest.TestCase):
    pass


add_function_test(
    TestGeometryHarmonic, "test_harmonic_reproduces_linear", test_harmonic_reproduces_linear, devices=devices
)
add_function_test(
    TestGeometryHarmonic,
    "test_harmonic_matches_dense_reference",
    test_harmonic_matches_dense_reference,
    devices=devices,
)
add_function_test(
    TestGeometryHarmonic, "test_harmonic_multiple_columns", test_harmonic_multiple_columns, devices=devices
)
add_function_test(TestGeometryHarmonic, "test_harmonic_tetrahedral", test_harmonic_tetrahedral, devices=devices)
add_function_test(TestGeometryHarmonic, "test_harmonic_validation", test_harmonic_validation, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
