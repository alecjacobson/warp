# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric-harmonic solve and its adjoint gradient with respect to the edge metric."""

import unittest

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import *


def _grid(n: int):
    x = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    points = np.stack([xx, yy, 0.2 * np.sin(2.0 * xx)], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    lower, upper, right, above = idx[:-1, :-1], idx[1:, 1:], idx[1:, :-1], idx[:-1, 1:]
    tris = np.concatenate(
        [np.stack([lower, right, upper], -1).reshape(-1, 3), np.stack([lower, upper, above], -1).reshape(-1, 3)]
    ).astype(np.int32)
    boundary = sorted(set(idx[[0, -1], :].ravel().tolist()) | set(idx[1:-1, [0, -1]].ravel().tolist()))
    return points, tris, np.array(boundary, dtype=np.int32)


def test_metric_solve_matches_harmonic(test, device):
    """With the true cotangent metric, the solve equals the cotangent harmonic solve."""
    points_np, tris, boundary_np = _grid(6)
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    solver = warp.geometry.MetricHarmonicSolver(d0, n, boundary, tol=1e-10, max_iters=10 * n)

    bc_np = (np.sin(3.0 * points_np[boundary_np, 0])).astype(np.float32)
    bc = wp.array(bc_np, dtype=wp.float32, device=device)

    solver.prepare(star1)
    metric = solver.solve(bc).numpy()
    reference = warp.geometry.harmonic(points, indices, boundary, bc, k=1, tol=1e-10).numpy()
    assert_np_equal(metric, reference, tol=1e-4)


def test_metric_solve_multiple_columns(test, device):
    """Multi-column solve matches per-column solves and honors the boundary exactly."""
    points_np, tris, boundary_np = _grid(5)
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    positive = wp.array(np.abs(star1.numpy()) + 0.5, dtype=wp.float32, device=device)
    solver = warp.geometry.MetricHarmonicSolver(d0, n, boundary, tol=1e-10, max_iters=10 * n)
    solver.prepare(positive)

    rng = np.random.default_rng(0)
    bc = rng.standard_normal((boundary_np.shape[0], 3)).astype(np.float32)
    columns = [solver.solve(wp.array(bc[:, c].copy(), dtype=wp.float32, device=device)).numpy() for c in range(3)]
    stacked = solver.solve(wp.array(bc, dtype=wp.float32, device=device)).numpy()

    for c in range(3):
        assert_np_equal(stacked[:, c], columns[c], tol=1e-5)
        assert_np_equal(stacked[boundary_np, c], bc[:, c], tol=1e-6)


def test_metric_vjp_matches_finite_differences(test, device):
    """The adjoint gradient w.r.t. the edge metric matches central finite differences."""
    rng = np.random.default_rng(1)
    points_np, tris, boundary_np = _grid(3)
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    num_edges = star1.shape[0]
    s0 = np.abs(star1.numpy()) + 0.5  # strictly positive, well-posed metric

    solver = warp.geometry.MetricHarmonicSolver(d0, n, boundary, tol=1e-12, max_iters=20 * n)

    symmetric = rng.standard_normal((n, n))
    bilinear = ((symmetric + symmetric.T) * 0.5).astype(np.float64)
    bc = wp.array(rng.standard_normal((boundary_np.shape[0], 2)).astype(np.float32), dtype=wp.float32, device=device)

    def loss_only(s_np):
        s = wp.array(s_np.astype(np.float32), dtype=wp.float32, device=device)
        solver.prepare(s)
        solution = solver.solve(bc).numpy()
        return float(np.sum(solution * (bilinear @ solution)))

    s0_wp = wp.array(s0.astype(np.float32), dtype=wp.float32, device=device)
    solver.prepare(s0_wp)
    solution = solver.solve(bc).numpy()
    grad_solution = wp.array((2.0 * (bilinear @ solution)).astype(np.float32), dtype=wp.float32, device=device)
    analytic = solver.vjp(wp.array(solution, dtype=wp.float32, device=device), grad_solution).numpy()

    eps = 1e-3
    numeric = np.zeros(num_edges)
    for e in range(num_edges):
        forward = s0.copy()
        forward[e] += eps
        backward = s0.copy()
        backward[e] -= eps
        numeric[e] = (loss_only(forward) - loss_only(backward)) / (2.0 * eps)

    test.assertTrue(np.isfinite(analytic).all())
    assert_np_equal(analytic, numeric.astype(np.float32), tol=2e-2 * max(1.0, np.abs(numeric).max()))


class _DenseDirectSolver(warp.geometry.LinearSolver):
    """A factor-once / solve-many backend (dense Cholesky on the host).

    Stands in for a direct solver such as cuDSS: :meth:`prepare` factors the
    materialized operator a single time, and each :meth:`solve` back-substitutes.
    Used only to exercise the :class:`~warp.geometry.LinearSolver` seam; it round-trips
    through the host, so it is for small systems in tests, not production.
    """

    def __init__(self):
        self.factor = None
        self.num_prepares = 0
        self.num_solves = 0

    def prepare(self, matrix):
        offsets = matrix.offsets.numpy()
        columns = matrix.columns.numpy()
        values = matrix.values.numpy().reshape(matrix.nnz_sync())
        n = matrix.shape[0]
        dense = np.zeros((n, n), dtype=np.float64)
        for row in range(n):
            for k in range(offsets[row], offsets[row + 1]):
                dense[row, columns[k]] = values[k]
        self.factor = np.linalg.cholesky(dense)
        self._device = matrix.device
        self.num_prepares += 1

    def solve(self, rhs, x):
        b = rhs.numpy().astype(np.float64)
        y = np.linalg.solve(self.factor, b)
        solution = np.linalg.solve(self.factor.T, y)
        wp.copy(x, wp.array(solution.astype(np.float32), dtype=wp.float32, device=self._device))
        self.num_solves += 1


def test_metric_solve_custom_backend(test, device):
    """A caller-supplied LinearSolver backend plugs in and matches the default CG solve."""
    points_np, tris, boundary_np = _grid(5)
    n = points_np.shape[0]
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tris.flatten(), dtype=wp.int32, device=device)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)

    d0, star1 = warp.geometry.dec_operators(points, indices)
    positive = wp.array(np.abs(star1.numpy()) + 0.5, dtype=wp.float32, device=device)
    bc = wp.array(
        np.random.default_rng(2).standard_normal((boundary_np.shape[0], 2)).astype(np.float32),
        dtype=wp.float32,
        device=device,
    )

    reference = warp.geometry.MetricHarmonicSolver(d0, n, boundary, tol=1e-10, max_iters=10 * n)
    reference.prepare(positive)
    expected = reference.solve(bc).numpy()

    backend = _DenseDirectSolver()
    custom = warp.geometry.MetricHarmonicSolver(d0, n, boundary, solver=backend)
    custom.prepare(positive)
    got = custom.solve(bc).numpy()

    assert_np_equal(got, expected, tol=1e-4)
    test.assertEqual(backend.num_prepares, 1)  # factored once
    test.assertEqual(backend.num_solves, 2)  # reused for both columns


devices = get_test_devices()


class TestGeometryMetricHarmonic(unittest.TestCase):
    pass


add_function_test(
    TestGeometryMetricHarmonic,
    "test_metric_solve_matches_harmonic",
    test_metric_solve_matches_harmonic,
    devices=devices,
)
add_function_test(
    TestGeometryMetricHarmonic,
    "test_metric_solve_multiple_columns",
    test_metric_solve_multiple_columns,
    devices=devices,
)
add_function_test(
    TestGeometryMetricHarmonic,
    "test_metric_vjp_matches_finite_differences",
    test_metric_vjp_matches_finite_differences,
    devices=devices,
)
add_function_test(
    TestGeometryMetricHarmonic,
    "test_metric_solve_custom_backend",
    test_metric_solve_custom_backend,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
