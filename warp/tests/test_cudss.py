# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check optional prepared direct solves and captured resource ownership."""

import gc
import importlib.util
import unittest
import weakref
from unittest.mock import patch

import numpy as np

import warp as wp
from warp.optim import eigen, linear
from warp.sparse import bsr_from_triplets
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _matrix(a, device, dtype=wp.float64):
    rows, columns = np.nonzero(a)
    return bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(rows, dtype=int, device=device),
        wp.array(columns, dtype=int, device=device),
        wp.array(a[rows, columns], dtype=dtype, device=device),
    )


def test_optional_contract(test, device):
    """Reject CPU and missing providers without importing optional native code."""
    a = _matrix(np.eye(4), device)
    if not device.is_cuda:
        with test.assertRaisesRegex(ValueError, "CUDA"):
            linear.CuDSS(a)
    else:
        with patch.dict("sys.modules", {"warp_cudss": None}), test.assertRaisesRegex(ImportError, "Install warp-cudss"):
            linear.CuDSS(a)
    with test.assertRaises(TypeError):
        linear.CuDSS(wp.ones(4, device=device))


def test_direct_categories(test, device):
    """Solve repeated and aliased right-hand sides with fresh physical diagnostics."""
    for dtype in (wp.float32, wp.float64):
        tolerance = 2e-5 if dtype == wp.float32 else 1e-10
        for category in ("spd", "symmetric", "general"):
            with test.subTest(dtype=dtype, category=category):
                a = np.diag(np.linspace(2.0, 6.0, 12)) + np.diag(np.full(11, -0.2), 1)
                a += np.diag(np.full(11, -0.2), -1)
                if category == "symmetric":
                    a[0, 0] = -3.0
                elif category == "general":
                    a[0, 1] = 0.3
                solver = linear.CuDSS(_matrix(a, device, dtype), 3, matrix_type=category)
                rhs = np.random.default_rng(24).normal(size=(3, 12))
                storage = wp.array(rhs, dtype=dtype, device=device)
                solver.solve(storage, storage)
                np.testing.assert_allclose(storage.numpy(), np.linalg.solve(a, rhs.T).T, rtol=tolerance, atol=tolerance)
                np.testing.assert_array_equal(solver.converged.numpy(), np.ones(3))
                residual = np.linalg.norm(storage.numpy() @ a.T - rhs, axis=1)
                np.testing.assert_allclose(solver.absolute_residual.numpy(), residual, atol=tolerance)
                solver.close()
                solver.close()
                with test.assertRaisesRegex(RuntimeError, "closed"):
                    solver.solve(storage, storage)


def test_capture_ownership(test, device):
    """Retain native factors through graph lifetime and replay changed right-hand sides."""
    a = np.diag(np.arange(1.0, 13.0))
    solver = linear.CuDSS(_matrix(a, device))
    rhs = wp.ones(12, dtype=wp.float64, device=device)
    out = wp.empty_like(rhs)
    with wp.ScopedCapture(device=device) as capture:
        solver.solve(rhs, out)
    graph = capture.graph
    del capture
    with test.assertRaisesRegex(RuntimeError, "Destroy captured graphs"):
        solver.close()
    reference = weakref.ref(solver)
    del solver
    gc.collect()
    test.assertIsNotNone(reference())
    rhs.fill_(2.0)
    wp.capture_launch(graph)
    np.testing.assert_allclose(out.numpy(), 2.0 / np.diag(a), rtol=1e-12)
    solver = reference()
    del graph
    gc.collect()
    solver.close()


def test_value_update(test, device):
    """Refactor values in place and reuse a graph without stale solutions."""
    a = _matrix(np.diag(np.arange(1.0, 13.0)), device)
    solver = linear.CuDSS(a)
    rhs = wp.ones(12, dtype=wp.float64, device=device)
    out = wp.empty_like(rhs)
    with wp.ScopedCapture(device=device) as capture:
        solver.solve(rhs, out)
    wp.copy(a.values, wp.array(np.arange(2.0, 14.0), dtype=wp.float64, device=device))
    solver.update()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(out.numpy(), 1.0 / np.arange(2.0, 14.0), rtol=1e-12)
    del capture
    gc.collect()
    solver.close()


def test_interchangeable_inverse(test, device):
    """Recover the same generalized modes with direct and pure-Warp inverses."""
    diagonal = np.arange(1.0, 17.0)
    mass = np.linspace(1.0, 2.0, 16)
    a = _matrix(np.diag(diagonal * mass), device)
    m = _matrix(np.diag(mass), device)
    shifted = _matrix(np.diag((diagonal + 1.0) * mass), device)
    direct = linear.CuDSS(shifted)
    iterative = linear.KrylovSolve(shifted, method="cg", tol=1e-12, residual="fresh")
    for inverse in (direct, iterative):
        state = eigen.KrylovSchur(a, 3, mass=m, inverse=inverse, sigma=-1.0, ncv=16, maxiter=3).prepare()
        result = state().check()
        np.testing.assert_allclose(result.eigenvalues.numpy(), diagonal[:3], rtol=1e-8)
        test.assertLess(float(result.relative_residuals.numpy().max()), 1e-8)
        del state, result
        gc.collect()
    direct.close()


def test_invalid_solve(test, device):
    """Reject unsupported initial guesses, streams, partial overlap and Tape use."""
    solver = linear.CuDSS(_matrix(np.eye(8), device))
    rhs = wp.ones(8, dtype=wp.float64, device=device)
    out = wp.empty_like(rhs)
    with test.assertRaisesRegex(ValueError, "initial guess"):
        solver.solve(rhs, out, rhs)
    with wp.ScopedStream(wp.Stream(device)), test.assertRaisesRegex(ValueError, "construction stream"):
        solver.solve(rhs, out)
    with wp.Tape(), test.assertRaisesRegex(RuntimeError, "reverse-mode"):
        solver.solve(rhs, out)
    buffer = wp.ones(9, dtype=wp.float64, device=device)
    with test.assertRaisesRegex(ValueError, "overlapping"):
        solver.solve(buffer[:8], buffer[1:])
    solver.close()


def test_factor_failure_and_stale_values(test, device):
    """Reject failed factors and flag original-system error after an omitted update."""
    a = _matrix(np.diag(np.arange(1.0, 9.0)), device)
    solver = linear.CuDSS(a)
    rhs = wp.ones(8, dtype=wp.float64, device=device)
    out = wp.empty_like(rhs)
    a.values.fill_(2.0)
    solver.solve(rhs, out)
    test.assertFalse(bool(solver.converged.numpy()[0]))
    solver.update()
    solver.solve(rhs, out)
    test.assertTrue(bool(solver.converged.numpy()[0]))
    rhs.fill_(wp.nan)
    solver.solve(rhs, out)
    test.assertFalse(bool(solver.converged.numpy()[0]))
    solver.close()
    with test.assertRaisesRegex(RuntimeError, "factorization|FACTORIZATION"):
        linear.CuDSS(_matrix(np.diag([-1.0, 1.0, 2.0, 3.0]), device), matrix_type="spd")


def test_block_storage_and_invalidation(test, device):
    """Solve block storage and reject reuse after a failed refactorization."""
    blocks = np.array([np.eye(3) * i + 0.1 for i in range(2, 6)])
    indices = wp.array(np.arange(4), dtype=int, device=device)
    a = bsr_from_triplets(4, 4, indices, indices, wp.array(blocks, dtype=wp.mat33d, device=device))
    solver = linear.CuDSS(a, 2)
    rhs = wp.ones((2, 12), dtype=wp.float64, device=device)
    out = wp.empty_like(rhs)
    solver.solve(rhs, out)
    expected = np.concatenate([np.linalg.solve(block, np.ones(3)) for block in blocks])
    np.testing.assert_allclose(out.numpy(), np.tile(expected, (2, 1)), atol=1e-12)
    with patch.object(solver._provider, "refactor", side_effect=RuntimeError("refactor failed")):
        with test.assertRaisesRegex(RuntimeError, "refactor failed"):
            solver.update()
    with test.assertRaisesRegex(RuntimeError, "rebuild"):
        solver.solve(rhs, out)
    solver.close()


class TestCuDSS(unittest.TestCase):
    pass


add_function_test(TestCuDSS, "test_optional_contract", test_optional_contract, devices=get_test_devices())
provider_available = importlib.util.find_spec("warp_cudss") is not None
for function in (
    test_direct_categories,
    test_capture_ownership,
    test_value_update,
    test_interchangeable_inverse,
    test_invalid_solve,
    test_factor_failure_and_stale_values,
    test_block_storage_and_invalidation,
):
    add_function_test(
        TestCuDSS,
        function.__name__,
        unittest.skipUnless(provider_available, "Requires optional warp-cudss and NVIDIA cuDSS")(function),
        devices=get_cuda_test_devices(),
    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
