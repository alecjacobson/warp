# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check symmetric eigen selection, metrics and prepared execution."""

import unittest
from unittest.mock import patch

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src import sparse_multi_rhs
from warp.optim import eigen, linear
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@fem.integrand
def _fem_stiffness(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@fem.integrand
def _fem_mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def _fem_vector_stiffness(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.grad(u, s), fem.grad(v, s)) + wp.dot(u(s), v(s))


@fem.integrand
def _fem_vector_mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


def _array(values, device):
    return wp.array(np.asarray(values), dtype=wp.float64, device=device)


def _check(test, result, a, expected, mass=None, tolerance=2e-7):
    result.check()
    values, vectors = result.eigenvalues.numpy(), result.eigenvectors.numpy()
    np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
    applied = vectors * a if a.ndim == 1 else vectors @ a.T
    metric = vectors if mass is None else vectors * mass if mass.ndim == 1 else vectors @ mass.T
    residual = np.linalg.norm(applied - values[:, None] * metric, axis=1)
    np.testing.assert_allclose(result.residual_norms.numpy(), residual, rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(vectors @ metric.T, np.eye(len(values)), rtol=1e-7, atol=1e-7)
    test.assertEqual(int(result.converged.numpy()[0]), len(values))
    test.assertTrue(np.all(result.valid.numpy()))


def test_low_modes(test, device):
    """Recover the lowest modes on both backends without zero-vector false positives."""
    diagonal = np.arange(1.0, 25.0)
    a = _array(diagonal, device)
    for cls, kwargs in ((eigen.LOBPCG, {}), (eigen.KrylovSchur, {"which": "SA", "ncv": 16})):
        with test.subTest(algorithm=cls.__name__):
            state = cls(a, 3, tol=1e-8, maxiter=100, **kwargs)
            test.assertEqual(int(state.result.status.numpy()[0]), eigen.EigenStatus.NOT_RUN)
            with test.assertRaisesRegex(RuntimeError, "prepare"):
                state()
            test.assertIs(state.prepare(), state)
            if device.is_cuda:
                test.assertIsNotNone(state._graph.graph_exec)
            result = state()
            _check(test, result, diagonal, np.arange(1.0, 4.0))
            test.assertIs(state(), result)
            _check(test, result, diagonal, np.arange(1.0, 4.0))


def test_spectrum_selection(test, device):
    """Select the requested spectrum independently of residual convergence."""
    diagonal = np.array([-30.0, -20.0, -10.0, -4.0, -1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 15.0, 25.0])
    a = _array(diagonal, device)
    requests = {
        "LM": [-30.0, -20.0, 25.0],
        "SM": [-1.0, 2.0, 3.0],
        "LA": [11.0, 15.0, 25.0],
        "SA": [-30.0, -20.0, -10.0],
        "BE": [-30.0, 15.0, 25.0],
    }
    for which, expected in requests.items():
        with test.subTest(which=which):
            result = eigen.krylov_schur(a, 3, which=which, ncv=len(diagonal), maxiter=5)
            _check(test, result, diagonal, expected)


def test_generalized_metric(test, device):
    """Recover a noncommuting SPD pencil in the mass inner product."""
    rng = np.random.default_rng(19)
    q, _ = np.linalg.qr(rng.normal(size=(18, 18)))
    mass = np.linspace(0.5, 2.5, 18)
    expected = np.arange(1.0, 19.0)
    a = np.sqrt(mass)[:, None] * ((q * expected) @ q.T) * np.sqrt(mass)[None, :]
    matrix, metric = _array(a, device), _array(mass, device)
    mass_inverse = linear.KrylovSolve(metric, method="cg", tol=1e-12, residual="fresh")
    for state in (
        eigen.LOBPCG(matrix, 3, mass=metric, maxiter=100),
        eigen.KrylovSchur(matrix, 3, mass=metric, mass_inverse=mass_inverse, which="SA", ncv=18),
    ):
        with test.subTest(algorithm=type(state).__name__):
            _check(test, state.prepare()(), a, expected[:3], mass)


def test_shift_invert(test, device):
    """Use the same prepared inverse contract for standard and generalized shifts."""
    diagonal = np.arange(1.0, 17.0)
    mass = np.linspace(1.0, 2.0, 16)
    for generalized in (False, True):
        metric = mass if generalized else np.ones(16)
        matrix = diagonal * metric
        system = _array(matrix + metric, device)
        inverse = linear.KrylovSolve(system, method="cg", tol=1e-12, residual="fresh")
        state = eigen.KrylovSchur(
            _array(matrix, device),
            3,
            mass=_array(metric, device) if generalized else None,
            sigma=-1.0,
            inverse=inverse,
            ncv=12,
            maxiter=30,
        )
        if generalized:
            result = state.prepare()()
        else:
            test.assertEqual(state._core.bx.ptr, state._core.eigenvectors.ptr)
            # A standard shifted problem must not fall back to mass products.
            with patch.object(state._core, "_metric_norm", side_effect=AssertionError("Unexpected metric path")):
                with patch.object(state._core.mass, "apply", side_effect=AssertionError("Unexpected identity copy")):
                    result = state.prepare()()
        _check(test, result, matrix, diagonal[:3], metric if generalized else None)
        # Reuse the fresh original-equation images from Krylov finalization.
        with patch.object(state._operator, "apply", side_effect=AssertionError("Repeated original product")):
            state._finalize()
        _check(test, result, matrix, diagonal[:3], metric if generalized else None)


def test_deflation(test, device):
    """Exclude a known nullspace and return mass-normalized physical modes."""
    diagonal = np.arange(24.0, dtype=float)
    basis = np.zeros((1, 24))
    basis[0, 0] = 3.0
    result = eigen.lobpcg(_array(diagonal, device), 3, deflation=_array(basis, device), maxiter=100)
    _check(test, result, diagonal, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.eigenvectors.numpy()[:, 0], 0.0, atol=1e-10)


def test_small_scale_and_absolute_tolerance(test, device):
    """Apply physical relative tolerances without an implicit unit absolute floor."""
    diagonal = np.arange(1.0, 25.0) * 1e-8
    for factory in (eigen.lobpcg, eigen.krylov_schur):
        kwargs = {} if factory is eigen.lobpcg else {"which": "SA", "ncv": 16}
        result = factory(_array(diagonal, device), 3, tol=1e-8, atol=1e-20, maxiter=100, **kwargs)
        result.check()
        np.testing.assert_allclose(result.eigenvalues.numpy() / 1e-8, [1.0, 2.0, 3.0], rtol=1e-7)
        test.assertLess(float(result.relative_residuals.numpy().max()), 2e-8)


def test_invalid_requests(test, device):
    """Reject unsupported metadata and Tape use before executing a solve."""
    a = _array(np.arange(1.0, 13.0), device)
    for kwargs in ({"k": 0}, {"k": 12}, {"k": True}, {"k": 2, "tol": -1.0}, {"k": 2, "maxiter": 0}):
        with test.assertRaises((ValueError, TypeError)):
            eigen.LOBPCG(a, **kwargs)
    with test.assertRaisesRegex(ValueError, "sigma and inverse"):
        eigen.KrylovSchur(a, 2, sigma=0.0)
    with test.assertRaises(TypeError):
        eigen.EigenSolverState()
    state = eigen.LOBPCG(a, 2).prepare()
    with wp.Tape(), test.assertRaisesRegex(RuntimeError, "reverse-mode"):
        state()


def test_graph_reuse(test, device):
    """Replay the prepared state in a caller graph after changing matrix values."""
    diagonal = np.arange(1.0, 25.0)
    matrix = _array(diagonal, device)
    for factory, kwargs in ((eigen.lobpcg, {}), (eigen.krylov_schur, {"which": "SA", "ncv": 16})):
        state = factory(matrix, 3, maxiter=100, run=False, **kwargs).prepare()
        with wp.ScopedCapture(device=device) as capture:
            result = state()
        wp.capture_launch(capture.graph)
        _check(test, result, diagonal, [1.0, 2.0, 3.0])
        matrix.assign(diagonal + 2.0)
        wp.capture_launch(capture.graph)
        _check(test, result, diagonal + 2.0, [3.0, 4.0, 5.0])
        matrix.assign(diagonal)


def test_inner_failure_and_reset(test, device):
    """Report an inaccurate inverse and clear its failure after a value update."""
    diagonal = np.arange(1.0, 17.0)
    matrix = _array(diagonal, device)
    system = _array(diagonal + 1.0, device)
    inverse = linear.KrylovSolve(system, method="cg", tol=1e-12, maxiter=1, residual="fresh")
    state = eigen.KrylovSchur(matrix, 3, sigma=-1.0, inverse=inverse, ncv=12, maxiter=3).prepare()
    result = state()
    test.assertEqual(int(result.status.numpy()[0]), eigen.EigenStatus.INNER_SOLVE_FAILED)
    test.assertFalse(np.any(result.valid.numpy()))
    # A scalar shifted system is exact in one CG iteration; no stale failure may survive.
    system.fill_(2.0)
    matrix.fill_(1.0)
    inverse.update()
    result = state()
    _check(test, result, np.ones(16), np.ones(3))


def test_fem_pencil(test, device):
    """Compare consistent-mass FEM modes with the discrete tensor-product spectrum."""
    resolution, modes = 6, 4
    with wp.ScopedDevice(device):
        geometry = fem.Grid2D(res=wp.vec2i(resolution), scalar_type=wp.float64)
        space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.float64)
        domain = fem.Cells(geometry)
        trial, test_field = fem.make_trial(space, domain=domain), fem.make_test(space, domain=domain)
        stiffness = fem.integrate(_fem_stiffness, fields={"u": trial, "v": test_field}, output_dtype=wp.float64)
        mass = fem.integrate(_fem_mass, fields={"u": trial, "v": test_field}, output_dtype=wp.float64)
        inverse = linear.KrylovSolve(
            stiffness,
            modes,
            method="cg",
            tol=0.01,
            maxiter=25,
            M=linear.FSAI(stiffness, max_row_size=8, factor_dtype=wp.float32),
            working_dtype=wp.float32,
        )
        with patch(
            "warp._src.optim.eigen._operators.MultiRHSOperator",
            side_effect=AssertionError("Unexpected generic sparse batch workspace"),
        ):
            state = eigen.LOBPCG(stiffness, modes, mass=mass, preconditioner=inverse, maxiter=100)
        test.assertEqual(state._ax.ptr, state._core.ax.ptr)
        test.assertEqual(state._mx.ptr, state._core.bx.ptr)
        result = state.prepare()().check()
    angles = np.arange(resolution + 1) * np.pi / resolution
    one_dimensional = 6 * resolution**2 * (1 - np.cos(angles)) / (2 + np.cos(angles))
    expected = np.sort((1 + one_dimensional[:, None] + one_dimensional[None, :]).ravel())[:modes]
    np.testing.assert_allclose(result.eigenvalues.numpy(), expected, rtol=2e-7, atol=1e-8)
    test.assertLess(float(result.relative_residuals.numpy().max()), 1e-8)


def test_vector_fem_product_strategy(test, device):
    """Retain coalesced CUDA block products and solve a vector FEM pencil."""
    resolution, modes, width = 4, 9, 12
    with wp.ScopedDevice(device):
        geometry = fem.Grid2D(res=wp.vec2i(resolution), scalar_type=wp.float64)
        space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.vec3d)
        domain = fem.Cells(geometry)
        trial, test_field = fem.make_trial(space, domain=domain), fem.make_test(space, domain=domain)
        stiffness = fem.integrate(_fem_vector_stiffness, fields={"u": trial, "v": test_field}, output_dtype=wp.float64)
        mass = fem.integrate(_fem_vector_mass, fields={"u": trial, "v": test_field}, output_dtype=wp.float64)
        test.assertEqual(stiffness.block_shape, (3, 3))
        state = eigen.LOBPCG(stiffness, modes, block_size=width, mass=mass, maxiter=150)
        for operator in (state._operator, state._mass):
            test.assertEqual(operator._direct_rows, not wp.get_device(device).is_cuda)
        with patch.object(sparse_multi_rhs, "product", wraps=sparse_multi_rhs.product) as products:
            state.prepare()
        for matrix in (stiffness, mass):
            strategies = {call.args[0] for call in products.call_args_list if call.args[1] is matrix}
            test.assertEqual(strategies, {"columns" if wp.get_device(device).is_cuda else "rows"})
        angles = np.arange(resolution + 1) * np.pi / resolution
        one_dimensional = 6 * resolution**2 * (1 - np.cos(angles)) / (2 + np.cos(angles))
        expected = np.repeat(np.sort((1 + one_dimensional[:, None] + one_dimensional[None, :]).ravel()), 3)[:modes]
        for _ in range(2):
            result = state().check()
            np.testing.assert_allclose(result.eigenvalues.numpy(), expected, rtol=2e-7, atol=1e-8)
            test.assertLess(float(result.relative_residuals.numpy().max()), 1e-8)


def test_nonfinite_and_exhaustion(test, device):
    """Distinguish an exhausted iteration budget from nonfinite input failure."""
    matrix = _array(np.arange(1.0, 33.0), device)
    state = eigen.LOBPCG(matrix, 3, maxiter=1, tol=1e-12).prepare()
    result = state()
    test.assertEqual(int(result.status.numpy()[0]), eigen.EigenStatus.ITERATION_LIMIT)
    test.assertLess(int(result.converged.numpy()[0]), 3)
    matrix.fill_(float("nan"))
    result = state()
    test.assertNotEqual(int(result.status.numpy()[0]), eigen.EigenStatus.CONVERGED)
    test.assertFalse(np.any(result.valid.numpy()))


def test_clustered_block(test, device):
    """Recover a repeated low eigenspace across projected block-width boundaries."""
    n, k = 180, 45
    diagonal = np.concatenate((np.ones(5), np.arange(2.0, n - 3.0)))
    state = eigen.LOBPCG(_array(diagonal, device), k, block_size=48, maxiter=150)
    _check(test, state.prepare()(), diagonal, diagonal[:k], tolerance=1e-6)
    if device.is_cuda and device.arch == 89:
        test.assertIsNotNone(state._core._second_whitening_usage)
        test.assertGreater(int(state._core._second_whitening_usage.numpy()[0]), 0)


def test_indefinite_shift(test, device):
    """Recover interior modes using GMRES on an indefinite shifted system."""
    diagonal = np.arange(1.0, 21.0)
    mass = np.linspace(0.5, 2.0, 20)
    sigma = 7.25
    matrix = _array(diagonal * mass, device)
    inverse = linear.KrylovSolve(
        _array((diagonal - sigma) * mass, device), method="gmres", restart=20, maxiter=40, tol=1e-11, residual="fresh"
    )
    result = eigen.krylov_schur(matrix, 3, mass=_array(mass, device), inverse=inverse, sigma=sigma, ncv=20, maxiter=3)
    _check(test, result, diagonal * mass, [6.0, 7.0, 8.0], mass)


def test_prepared_inverse_warmup(test, device):
    """Warm one prepared inverse application before the first captured solve."""
    if wp.get_cuda_driver_version() < (12, 3):
        test.skipTest("Kernel loading during capture requires CUDA driver 12.3")
    values = np.arange(1.0, 17.0)
    inverse = linear.KrylovSolve(_array(values + 1.0, device), method="cg", tol=1e-12, residual="fresh")
    state = eigen.KrylovSchur(_array(values, device), 3, inverse=inverse, sigma=-1.0, ncv=12)
    eager_calls = []
    solve = inverse.solve

    def record_solve(*args, **kwargs):
        if not wp.get_device(device).is_capturing:
            eager_calls.append(1)
        return solve(*args, **kwargs)

    with patch.object(inverse, "solve", side_effect=record_solve):
        with patch.object(state._core, "solve", side_effect=AssertionError("Unexpected full warmup solve")):
            state.prepare()
    test.assertEqual(len(eager_calls), 1)
    _check(test, state(), values, values[:3])


class TestEigen(unittest.TestCase):
    def test_cpu_capture_rejection(self):
        """Reject CPU capture before reading convergence or recording partial work."""
        device = wp.get_device("cpu")
        matrix = _array(np.arange(1.0, 17.0), device)
        state = eigen.LOBPCG(matrix, 3).prepare()
        for operation in (lambda: eigen.LOBPCG(matrix, 3), state.prepare, state):
            with wp.ScopedCapture(device=device, apic=True):
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    operation()


for test_function in (
    test_low_modes,
    test_spectrum_selection,
    test_generalized_metric,
    test_shift_invert,
    test_deflation,
    test_small_scale_and_absolute_tolerance,
    test_invalid_requests,
    test_inner_failure_and_reset,
    test_nonfinite_and_exhaustion,
    test_fem_pencil,
    test_vector_fem_product_strategy,
):
    add_function_test(TestEigen, test_function.__name__, test_function, devices=get_test_devices())
add_function_test(TestEigen, "test_clustered_block", test_clustered_block, devices=get_test_devices())
add_function_test(TestEigen, "test_indefinite_shift", test_indefinite_shift, devices=get_test_devices())
add_function_test(
    TestEigen, "test_prepared_inverse_warmup", test_prepared_inverse_warmup, devices=get_cuda_test_devices()
)
add_function_test(TestEigen, "test_graph_reuse", test_graph_reuse, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
