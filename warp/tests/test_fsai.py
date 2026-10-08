# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np

import warp as wp
import warp.sparse as sp
from warp._src.optim import fsai as fsai_module
from warp.optim.linear import FSAI, cg, cr, preconditioner
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _matrix(a, device, dtype=wp.float64, block=1):
    rows, cols = np.nonzero(a)
    result = sp.bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(a[rows, cols], dtype=dtype, device=device),
    )
    return sp.bsr_copy(result, block_shape=(block, block)) if block != 1 else result


def _dense(a):
    offsets, columns, values = a.offsets.numpy(), a.columns.numpy(), a.values.numpy()
    result = np.zeros(a.shape, dtype=values.dtype)
    for row in range(a.nrow):
        start, end = offsets[row : row + 2]
        result[row, columns[start:end]] = values[start:end]
    return result


def _path(n):
    return np.diag(np.full(n, 3.0)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)


def _check_local_solves(test, pre, a, rtol=1e-10, atol=1e-12):
    g = _dense(pre.G)
    offsets, columns = pre.G.offsets.numpy(), pre.G.columns.numpy()
    np.testing.assert_array_equal(_dense(pre.GT), g.T)
    np.testing.assert_array_equal(g, np.tril(g))
    test.assertTrue(np.all(np.isfinite(g)))
    test.assertTrue(np.all(np.diag(g) > 0))
    for row in range(len(a)):
        support = columns[offsets[row] : offsets[row + 1]]
        test.assertEqual(support[-1], row)
        test.assertTrue(np.all(np.diff(support) > 0))
        rhs = np.zeros(len(support))
        rhs[-1] = 1
        z = np.linalg.solve(a[np.ix_(support, support)], rhs)
        np.testing.assert_allclose(g[row, support], z / np.sqrt(z[-1]), rtol=rtol, atol=atol)
    return g


def test_full_pattern(test, device, dtype):
    """Check that an unrestricted pattern reproduces the exact inverse and local solves."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 8))
    a = a @ a.T + np.eye(8)
    for step_size in (1, 3):
        pre = FSAI(_matrix(a, device, dtype), max_row_size=8, kap_tolerance=0, max_step_size=step_size)
        g = _check_local_solves(test, pre, a, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=2e-5, atol=1e-6)


def test_adaptive_support(test, device, dtype):
    """Check adaptive support selection against an independent dense greedy search."""
    rng = np.random.default_rng(193)
    a = rng.normal(size=(18, 18))
    a[rng.random(a.shape) < 0.65] = 0
    a = a + a.T
    np.fill_diagonal(a, np.sum(np.abs(a), axis=1) + 2)
    scaling = np.geomspace(0.01, 100, len(a))
    a *= scaling[:, None] * scaling[None, :]
    raw = _matrix(a, device, dtype, block=3)
    equilibrated = a / np.sqrt(np.diag(a)[:, None] * np.diag(a)[None, :])
    for step_size in (1, 3):
        for tolerance in (0.0, 0.003, 0.1):
            pre = FSAI(raw, max_row_size=8, kap_tolerance=tolerance, max_step_size=step_size)
            g = _dense(pre.G)
            # Independent dense greedy search; neither factor values nor supports
            # are obtained from the implementation under test.
            for i in range(len(a)):
                support = [i]
                z = np.ones(1)
                while len(support) < 8:
                    frontier = sorted(set(np.flatnonzero(np.any(a[support] != 0, axis=0))) - set(support))
                    frontier = [c for c in frontier if c < i]
                    scores = [abs(equilibrated[c, support] @ z) for c in frontier]
                    ranked = sorted(zip(scores, frontier, strict=True), key=lambda item: (-item[0], item[1]))
                    selected = [c for score, c in ranked if score > 0][: min(step_size, 8 - len(support))]
                    if not selected:
                        break
                    support.extend(selected)
                    old = z[0]
                    rhs = np.zeros(len(support))
                    rhs[0] = 1
                    z = np.linalg.solve(equilibrated[np.ix_(support, support)], rhs)
                    if tolerance > 0 and 1 - old / z[0] <= tolerance:
                        break
                expected = np.zeros(len(a))
                expected[support] = z / np.sqrt(z[0] * np.diag(a)[support])
                np.testing.assert_allclose(g[i], expected, rtol=3e-5, atol=1e-6)


def test_narrow_frontier_and_zero_tolerance(test, device):
    """Check that short frontiers and a disabled tolerance do not stop growth early."""
    # The frontier has one candidate. A short batch must not use a full batch's
    # budget; improvement rounding to zero must not stop a disabled tolerance.
    a = _path(32)
    for step_size in (1, 3):
        pre = FSAI(_matrix(a, device), max_row_size=32, max_step_size=step_size, kap_tolerance=0)
        np.testing.assert_array_equal(np.diff(pre.G.offsets.numpy()), np.arange(1, 33))
        _check_local_solves(test, pre, a)


def test_pivot_guard(test, device):
    """Check that a rejected pivot truncates the row and is counted."""
    a = np.array([[1.0, 1 - 1e-14, 0], [1 - 1e-14, 1.0, 0], [0, 0, 2.0]])
    for width in (1, 2):
        pre = FSAI(_matrix(a, device), max_row_size=width, max_step_size=6)
        test.assertEqual(pre.truncated_rows, 0 if width == 1 else 1)
        np.testing.assert_allclose(_dense(pre.G), np.diag(1 / np.sqrt(np.diag(a))))


def test_application(test, device, dtype):
    """Check ``G.T @ G`` application across lanes, storage types and aliasing."""
    rng = np.random.default_rng(18)
    a = _path(35)
    raw = _matrix(a, device, dtype)
    tol = 5e-3 if dtype == wp.float16 else 3e-6 if dtype == wp.float32 else 2e-12
    # Include partial CUDA thread blocks and both ordinary and compressed factors.
    for storage in (dtype,) if dtype == wp.float16 else dict.fromkeys((dtype, wp.float32)):
        for lanes in (1, 2, 4, 8, 16, 32):
            pre = FSAI(raw, max_row_size=5, factor_dtype=storage, apply_lanes=lanes)
            g = _dense(pre.G).astype(np.float64)
            np.testing.assert_array_equal(_dense(pre.GT), g.T)
            for alias in ("none", "xz", "yz", "xy", "xyz"):
                x = wp.array(rng.normal(size=len(a)), dtype=dtype, device=device)
                y = x if alias in ("xy", "xyz") else wp.array(rng.normal(size=len(a)), dtype=dtype, device=device)
                z = x if alias in ("xz", "xyz") else y if alias == "yz" else wp.empty_like(x)
                expected = 0.7 * g.T @ g @ x.numpy() - 1.2 * y.numpy()
                pre.matvec(x, y, z, 0.7, -1.2)
                np.testing.assert_allclose(z.numpy(), expected, rtol=tol, atol=tol)
            x = wp.ones(len(a), dtype=dtype, device=device)
            y = wp.empty_like(x)
            z = wp.empty_like(x)
            y.fill_(float("nan"))
            pre.matvec(x, y, z, 1, 0)
            np.testing.assert_allclose(z.numpy(), g.T @ g @ np.ones(len(a)), rtol=tol, atol=tol)
            x.fill_(float("nan"))
            y.fill_(2)
            pre.matvec(x, y, z, 0, 0.5)
            np.testing.assert_array_equal(z.numpy(), 1)


def test_solver_integration(test, device, dtype):
    """Check CG and CR convergence with FSAI against the true residual."""
    a = _path(24)
    tolerance = 1e-5 if dtype == wp.float32 else 1e-11
    for block in (1, 3):
        raw = _matrix(a, device, dtype, block)
        jacobi = FSAI(raw, max_row_size=1)
        np.testing.assert_allclose(_dense(jacobi.G), np.eye(len(a)) / np.sqrt(3), rtol=1e-6)
        pre = preconditioner(raw, "fsai")
        test.assertIsInstance(pre, FSAI)
        vector = dtype if block == 1 else wp.types.vector(block, dtype)
        b = wp.ones(len(a) // block, dtype=vector, device=device)
        for solver in (cg, cr):
            x = wp.zeros_like(b)
            with wp.ScopedDevice(device):
                solver(raw, b, x, M=pre, tol=tolerance, maxiter=100)
            # Check the true residual, independently of the solver's report.
            np.testing.assert_allclose(a @ x.numpy().ravel(), 1.0, atol=4 * tolerance, rtol=0)


def test_refit_and_rollback(test, device, dtype):
    """Check that refits reuse storage and failed updates leave the factor unchanged."""
    rng = np.random.default_rng(48)
    r = rng.normal(size=(12, 12))
    a = r @ r.T + np.eye(12) * 3
    b = a + np.diag(np.arange(12))
    for block in (1, 3):
        for storage in (dtype,) if dtype == wp.float16 else dict.fromkeys((dtype, wp.float32)):
            raw, updated = _matrix(a, device, dtype, block), _matrix(b, device, dtype, block)
            pre = FSAI(raw, max_row_size=5, reuse_pattern=True, factor_dtype=storage, max_step_size=3)
            pointers = (pre.G.values.ptr, pre.GT.values.ptr)
            topology = (pre.G.offsets.numpy().copy(), pre.G.columns.numpy().copy())
            test.assertIs(pre.update(updated), pre)
            rtol, atol = (8e-3, 5e-4) if dtype == wp.float16 else (2e-4, 2e-7)
            g = _check_local_solves(test, pre, b, rtol=rtol, atol=atol)
            test.assertEqual((pre.G.values.ptr, pre.GT.values.ptr), pointers)
            np.testing.assert_array_equal(pre.G.offsets.numpy(), topology[0])
            np.testing.assert_array_equal(pre.G.columns.numpy(), topology[1])
            for failure in ("diagonal", "pivot", "topology"):
                invalid = sp.bsr_copy(updated)
                if failure == "diagonal":
                    invalid.values.zero_()
                elif failure == "pivot":
                    # Positive diagonals are not enough: every selected pair is indefinite.
                    indefinite = np.full(a.shape, 10.0)
                    np.fill_diagonal(indefinite, 1.0)
                    invalid = _matrix(indefinite, device, dtype, block)
                else:
                    columns = invalid.columns.numpy().copy()
                    columns[0] += 1
                    invalid.columns.assign(columns)
                with test.assertRaises(ValueError):
                    pre.update(invalid)
                test.assertIs(pre.source, updated)
                np.testing.assert_array_equal(_dense(pre.G), g)
                np.testing.assert_array_equal(_dense(pre.GT), g.T)
            # Same object, changing values, then no-argument update.
            updated.values.assign(raw.values.numpy())
            pre.update()
            _check_local_solves(test, pre, a, rtol=rtol, atol=atol)


def test_stored_zeros_and_capacity(test, device):
    """Check that stored zeros are retained and unused capacity is ignored."""
    a = _path(3)
    a[0, 2] = a[2, 0] = 0
    raw = _matrix(a, device, block=3)
    count = raw.nnz_sync()
    columns, values = raw.columns, raw.values
    raw.columns = wp.full(16, -123, dtype=int, device=device)
    raw.values = wp.empty(16, dtype=raw.dtype, device=device)
    raw.values.fill_(float("nan"))
    wp.copy(raw.columns, columns, count=count)
    wp.copy(raw.values, values, count=count)
    raw.notify_nnz_changed(nnz=16)
    # Unused capacity must not enter topology checks, scalarization or refits.
    pre = FSAI(raw, max_row_size=3, kap_tolerance=0, reuse_pattern=True)
    updated = _matrix(a + 0.2 * np.ones((3, 3)), device, block=3)
    pre.update(updated)
    g = _check_local_solves(test, pre, a + 0.2)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(a + 0.2), atol=1e-12)
    # 1x1 matrix-valued BSR is distinct from scalar-valued CSR.
    scalar = _matrix(a, device)
    single = sp.bsr_zeros(3, 3, wp.types.matrix((1, 1), wp.float64), device=device)
    single.offsets = wp.clone(scalar.offsets)
    single.columns = wp.clone(scalar.columns)
    single.values = scalar.values.view(single.dtype)
    single.notify_nnz_changed(nnz=scalar.nnz_sync())
    _check_local_solves(test, FSAI(single, max_row_size=3, kap_tolerance=0, reuse_pattern=True).update(), a)


def test_padded_input(test, device):
    """Check that padded input matches compact input and is rejected for reuse."""
    a = _path(12)
    for block in (1, 3):
        raw = _matrix(a, device, block=block)
        padded = sp.bsr_zeros(raw.nrow, raw.ncol, raw.dtype, device=device, row_capacity=raw.ncol + 2)
        sp.bsr_assign(padded, raw, topology="padded")
        pre = FSAI(padded, max_row_size=5)
        expected = FSAI(raw, max_row_size=5)
        np.testing.assert_allclose(_dense(pre.G), _dense(expected.G), atol=1e-12)
        with test.assertRaisesRegex(ValueError, "compact"):
            FSAI(padded, reuse_pattern=True)


def test_empty(test, device):
    """Check construction, application and update on an empty matrix."""
    for dtype in (wp.float16, wp.float32, wp.float64):
        raw = sp.bsr_zeros(0, 0, dtype, device=device)
        for reuse in (False, True):
            pre = FSAI(raw, reuse_pattern=reuse)
            test.assertEqual(pre.G.nnz_sync(), 0)
            test.assertEqual(pre.shape, (0, 0))
            x = wp.empty(0, dtype=dtype, device=device)
            pre.matvec(x, x, x, 1, 0)
            if reuse:
                test.assertIs(pre.update(), pre)
            wp.synchronize_device(device)


def test_apply_dispatch(test, device):
    """Check that each apply configuration reaches its intended kernel path."""
    a = _path(12)
    for dtype, storage in ((wp.float64, wp.float64), (wp.float64, wp.float32)):
        raw = _matrix(a, device, dtype)
        x = wp.ones(12, dtype=dtype, device=device)
        y = wp.empty_like(x)
        for lanes in (1, 4):
            pre = FSAI(raw, max_row_size=5, factor_dtype=storage, apply_lanes=lanes)
            with (
                mock.patch.object(fsai_module, "_grouped_mv", wraps=fsai_module._grouped_mv) as grouped,
                mock.patch.object(fsai_module, "_mixed_kernel", wraps=fsai_module._mixed_kernel) as mixed,
            ):
                pre.matvec(x, y, y, 1, 0)
            # The grouped kernel is CUDA only; CPU always uses the ordinary product.
            expect_grouped = device.is_cuda and lanes > 1
            test.assertEqual(grouped.call_count, 2 if expect_grouped else 0)
            test.assertEqual(mixed.call_count, 2 if storage != dtype and not expect_grouped else 0)
            g = _dense(pre.G)
            np.testing.assert_allclose(y.numpy(), g.T @ g @ np.ones(12), rtol=1e-6, atol=1e-6)


def test_concurrent_streams(test, device):
    """Check that one instance applied on concurrent streams never corrupts an output."""
    num_blocks, block_size = 50_000, 12
    count = num_blocks * block_size
    A = sp.bsr_identity(num_blocks, block_type=wp.types.matrix((block_size, block_size), wp.float32), device=device)
    pre = FSAI(A)
    stream0 = wp.Stream(device, priority=0)
    stream1 = wp.Stream(device, priority=-1)
    x0 = wp.full(count, 1.0, dtype=wp.float32, device=device)
    x1 = wp.full(count, 2.0, dtype=wp.float32, device=device)
    z0, z1 = wp.zeros_like(x0), wp.zeros_like(x1)
    pre.matvec(x0, z0, z0, 1.0, 0.0)
    wp.synchronize_device(device)
    for _ in range(100):
        # Zero each output on the stream that writes it so the memset stays ordered.
        with wp.ScopedStream(stream0, sync_enter=False, sync_exit=False):
            z0.zero_()
            pre.matvec(x0, z0, z0, 1.0, 0.0)
        with wp.ScopedStream(stream1, sync_enter=False, sync_exit=False):
            z1.zero_()
            pre.matvec(x1, z1, z1, 1.0, 0.0)
        wp.synchronize_device(device)
        np.testing.assert_array_equal(z0.numpy(), x0.numpy())
        np.testing.assert_array_equal(z1.numpy(), x1.numpy())


def test_scaling_and_conversion(test, device):
    """Check badly scaled inputs and factor conversion overflow handling."""
    a = _path(12)
    scale = np.geomspace(1e-8, 1e8, len(a))
    a *= scale[:, None] * scale[None, :]
    _check_local_solves(test, FSAI(_matrix(a, device), max_row_size=5), a, atol=1e-9)
    for magnitude in (1e-100, 1e100):
        with test.assertRaisesRegex(ValueError, "positive diagonal"):
            FSAI(_matrix(np.eye(3) * magnitude, device), factor_dtype=wp.float32)
    raw = _matrix(np.eye(3), device)
    pre = FSAI(raw, reuse_pattern=True, factor_dtype=wp.float32)
    before = _dense(pre.G)
    for magnitude in (1e-100, 1e100):
        with test.assertRaises(ValueError):
            pre.update(_matrix(np.eye(3) * magnitude, device))
        np.testing.assert_array_equal(_dense(pre.G), before)
        np.testing.assert_array_equal(_dense(pre.GT), before.T)


def test_capture_update(test, device):
    """Check that CUDA graph application sees refit factor values."""
    a = _path(12)
    for block in (1, 3):
        for lanes in (1, 4):
            for storage in (wp.float32, wp.float64):
                raw = _matrix(a, device, block=block)
                pre = FSAI(raw, max_row_size=5, reuse_pattern=True, factor_dtype=storage, apply_lanes=lanes)
                vector = wp.float64 if block == 1 else wp.vec3d
                x = wp.ones(12 // block, dtype=vector, device=device)
                y = wp.empty_like(x)
                y.fill_(float("nan"))
                # Load the kernel modules before capture; capture uses force_module_load=False.
                pre.matvec(x, y, y, 1, 0)
                with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                    pre.matvec(x, y, y, 1, 0)
                for multiplier in (2.0, 0.5):
                    pre.update(_matrix(multiplier * a, device, block=block))
                    wp.capture_launch(cap.graph)
                    g = _dense(pre.G).astype(np.float64)
                    np.testing.assert_allclose(y.numpy().ravel(), g.T @ g @ np.ones(12), rtol=1e-12, atol=1e-12)
                wp.synchronize_device(device)


def test_float16_factors(test, device):
    """Check float16 factors against local solves and the exact inverse."""
    # Reference the matrix after input quantization, not the original float64 data.
    rng = np.random.default_rng(72)
    r = rng.normal(size=(12, 12))
    a = r @ r.T + 4 * np.eye(12)
    a = a.astype(np.float16).astype(np.float64)
    for block in (1, 3):
        raw = _matrix(a, device, wp.float16, block)
        for width, step in ((1, 1), (5, 3), (12, 1), (12, 3)):
            pre = FSAI(raw, max_row_size=width, max_step_size=step, kap_tolerance=0)
            test.assertEqual(pre.scalar_type, wp.float16)
            test.assertEqual(pre.G.scalar_type, wp.float16)
            test.assertEqual(pre.GT.scalar_type, wp.float16)
            g = _check_local_solves(test, pre, a, rtol=1e-2, atol=1e-3).astype(np.float64)
            if width == 12:
                reference = np.linalg.inv(a)
                test.assertLess(np.linalg.norm(g.T @ g - reference) / np.linalg.norm(reference), 1e-2)
            if width == 1:
                np.testing.assert_allclose(g.T @ g, np.diag(1 / np.diag(a)), rtol=2e-3, atol=1e-6)
        with test.assertRaisesRegex(ValueError, "factor_dtype"):
            FSAI(raw, factor_dtype=wp.float32)
        # Ordinary construction accepts padded blocks for half precision, too.
        padded = sp.bsr_zeros(raw.nrow, raw.ncol, raw.dtype, device=device, row_capacity=raw.ncol + 1)
        sp.bsr_assign(padded, raw, topology="padded")
        np.testing.assert_array_equal(_dense(FSAI(padded).G), _dense(FSAI(raw).G))


def test_float16_pivot_guard(test, device):
    """Check float16 pivot truncation and invalid diagonal rejection."""
    # Same rank-deficient inputs as block Jacobi. FSAI truncates a factor row
    # at a rejected pivot, retaining a positive Gram factor rather than
    # replacing an entire diagonal block with the identity.
    for a in (np.array([[1, 2], [2, 4]]), np.array([[9, 21], [21, 49]])):
        pre = FSAI(_matrix(a, device, wp.float16, 2), max_row_size=2)
        test.assertEqual(pre.truncated_rows, 1)
        g = _dense(pre.G).astype(np.float64)
        np.testing.assert_allclose(g, np.diag(1 / np.sqrt(np.diag(a))), rtol=2e-3)
        test.assertGreater(np.linalg.eigvalsh(g.T @ g).min(), 0)
    for diagonal in (0, -1, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "positive diagonals"):
            FSAI(_matrix(np.diag([1.0, diagonal]), device, wp.float16))


def test_float16_capture_refit(test, device):
    """Check that float16 CUDA graph application sees refit factor values."""
    a = _path(12)
    for lanes in (1, 4):
        raw = _matrix(a, device, wp.float16, 3)
        pre = FSAI(raw, reuse_pattern=True, apply_lanes=lanes, max_row_size=8, kap_tolerance=0)
        pointers = (pre.G.values.ptr, pre.GT.values.ptr)
        x = wp.ones(4, dtype=wp.vec3h, device=device)
        y = wp.empty_like(x)
        # Load the kernel modules before capture; capture uses force_module_load=False.
        pre.matvec(x, y, y, 1, 0)
        with wp.ScopedCapture(device=device, force_module_load=False) as cap:
            pre.matvec(x, y, y, 1, 0)
        for multiplier in (2.0, 0.5):
            pre.update(_matrix(multiplier * a, device, wp.float16, 3))
            test.assertEqual((pre.G.values.ptr, pre.GT.values.ptr), pointers)
            wp.capture_launch(cap.graph)
            g = _dense(pre.G).astype(np.float64)
            np.testing.assert_allclose(y.numpy().ravel(), g.T @ g @ np.ones(12), rtol=3e-3, atol=1e-3)
        before = y.numpy().copy()
        invalid = sp.bsr_copy(raw)
        invalid.values.zero_()
        with test.assertRaises(ValueError):
            pre.update(invalid)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(y.numpy(), before)


class TestFSAI(unittest.TestCase):
    def test_invalid_arguments(self):
        """Check that invalid arguments and matrices raise specific errors."""
        raw = _matrix(np.eye(3), "cpu")
        for option in ("max_row_size", "max_step_size"):
            for value in (0, -1, 65, 1.5):
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    FSAI(raw, **{option: value})
        for option, values in (
            ("kap_tolerance", (-1, 1, float("nan"), float("inf"))),
            ("pivot_floor", (0, 1, float("nan"))),
            ("apply_lanes", (0, 3, 64)),
            ("factor_dtype", (wp.float16, wp.int32)),
        ):
            for value in values:
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    FSAI(raw, **{option: value})
        for invalid in (
            None,
            sp.bsr_zeros(2, 3, wp.float64, device="cpu"),
            sp.bsr_zeros(3, 2, wp.types.matrix((2, 3), wp.float64), device="cpu"),
        ):
            with self.assertRaisesRegex(ValueError, "square"):
                FSAI(invalid)
        with self.assertRaises(TypeError):
            FSAI(sp.bsr_zeros(3, 3, wp.bfloat16, device="cpu"))
        with self.assertRaisesRegex(ValueError, "reuse_pattern"):
            FSAI(raw).update()
        for diagonal in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "positive diagonals"):
                FSAI(_matrix(np.diag([1.0, diagonal]), "cpu"))
        pre = FSAI(raw, reuse_pattern=True)
        with self.assertRaisesRegex(ValueError, "BSR"):
            pre.update(np.eye(3))
        for changed in (_matrix(np.eye(2), "cpu"), _matrix(np.eye(3), "cpu", wp.float32)):
            with self.assertRaisesRegex(ValueError, "shape, dtype, device"):
                pre.update(changed)


devices = get_test_devices()
for dtype in (wp.float32, wp.float64):
    for func in (
        test_full_pattern,
        test_adaptive_support,
        test_application,
        test_solver_integration,
        test_refit_and_rollback,
    ):
        add_function_test(TestFSAI, f"{func.__name__}_{dtype.__name__}", func, devices=devices, dtype=dtype)
for func in (
    test_narrow_frontier_and_zero_tolerance,
    test_pivot_guard,
    test_stored_zeros_and_capacity,
    test_padded_input,
    test_empty,
    test_scaling_and_conversion,
    test_apply_dispatch,
):
    add_function_test(TestFSAI, func.__name__, func, devices=devices)
add_function_test(TestFSAI, "test_capture_update", test_capture_update, devices=get_cuda_test_devices())
add_function_test(TestFSAI, "test_concurrent_streams", test_concurrent_streams, devices=get_cuda_test_devices())

for func in (test_application, test_refit_and_rollback):
    add_function_test(TestFSAI, f"{func.__name__}_float16", func, devices=devices, dtype=wp.float16)
for func in (test_float16_factors, test_float16_pivot_guard):
    add_function_test(TestFSAI, func.__name__, func, devices=devices)
add_function_test(TestFSAI, "test_float16_capture_refit", test_float16_capture_refit, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
