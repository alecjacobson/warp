# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Extended jet builtins (inverse-trig, pow variants, branching) for first- and
# second-order jets, checked against finite differences.
#
# CPU-only, by standard unittest.TestCase rather than add_function_test(). These
# kernels chain many of the new overloads, and building one module for both CPU
# and CUDA in the same process currently trips a Warp module-hasher instability:
# the second device's build perturbs shared hash state so a kernel's symbol is
# looked up under a hash that differs from the one it was compiled with. Single
# device keeps the hashes consistent. CUDA correctness of these ops is covered
# by the jet benchmarks' finite-difference gates.

import unittest

import numpy as np

import warp as wp

J2 = wp.JetSpace(2)
J2_2ND = wp.JetSpace2(2)

DEVICES = ["cpu"]

Z_NP = np.array([[0.7, -0.4], [-0.2, 0.9], [0.7, 0.9]], dtype=np.float32)
ZB_NP = np.array([[0.3, 0.7], [-0.2, 0.9], [0.4, -0.1]], dtype=np.float32)  # away from branch kinks


def _grad_fd(fn, z, h=1.0e-5):
    out = np.empty_like(z)
    for p in range(z.shape[1]):
        e = np.zeros_like(z)
        e[:, p] = h
        out[:, p] = (fn(z + e) - fn(z - e)) / (2.0 * h)
    return out


def _hess_fd(fn, z, h=1.0e-4):
    n, k = z.shape
    out = np.empty((n, k, k))
    for p in range(k):
        for q in range(k):
            ep = np.zeros_like(z)
            ep[:, p] = h
            eq = np.zeros_like(z)
            eq[:, q] = h
            out[:, p, q] = (fn(z + ep + eq) - fn(z + ep - eq) - fn(z - ep + eq) + fn(z - ep - eq)) / (4.0 * h * h)
    return out


# --------------------------------------------------------------------------
# First-order: inverse-trig, pow variants, and value-branching builtins.
# --------------------------------------------------------------------------


def smooth_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return (
        np.arcsin(0.4 * a)
        + np.arccos(0.3 * b)
        + np.arctan(1.5 * a)
        + np.arctan2(a, b + 2.0)
        + np.power(a + 2.0, b)
        + np.power(3.0, b)
    )


@wp.kernel(enable_backward=False)
def jet_smooth(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff]):
    i = wp.tid()
    a = J2.seed(z[i, 0], 0)
    b = J2.seed(z[i, 1], 1)
    e = (
        wp.asin(0.4 * a)
        + wp.acos(0.3 * b)
        + wp.atan(1.5 * a)
        + wp.atan2(a, b + 2.0)
        + wp.pow(a + 2.0, b)  # pow(jet, jet)
        + wp.pow(3.0, b)  # pow(dtype, jet)
    )
    val[i] = e.value
    grad[i] = e.coeff


def branch_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.minimum(a, b) + np.maximum(a, b) + np.clip(a, -0.5, 0.5) + np.where(a > 0.0, a, b) + np.abs(a)


@wp.kernel(enable_backward=False)
def jet_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff]):
    i = wp.tid()
    a = J2.seed(z[i, 0], 0)
    b = J2.seed(z[i, 1], 1)
    e = wp.min(a, b) + wp.max(a, b) + wp.clamp(a, -0.5, 0.5) + wp.where(a.value > 0.0, a, b) + wp.abs(a)
    val[i] = e.value
    grad[i] = e.coeff


# --------------------------------------------------------------------------
# Second-order: same op families, checked for value, gradient, and Hessian.
# --------------------------------------------------------------------------


def g3_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return (
        np.arctan2(a, b + 2.0)
        + np.power(a + 2.0, b)
        + np.power(3.0, b)
        + np.tan(0.3 * a)
        + np.arcsin(0.4 * b)
        + 1.0 / (a * a + 1.0)
    )


@wp.kernel(enable_backward=False)
def jet2_g3(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.seed(z[i, 0], 0)
    b = J2_2ND.seed(z[i, 1], 1)
    r = (
        wp.atan2(a, b + 2.0)
        + wp.pow(a + 2.0, b)  # pow(jet, jet)
        + wp.pow(3.0, b)  # pow(dtype, jet)
        + wp.tan(0.3 * a)
        + wp.asin(0.4 * b)
        + 1.0 / (a * a + 1.0)  # div(dtype, jet)
    )
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def branch2_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.minimum(a * a, b * b + 5.0) + np.maximum(a * a * a, b) + np.where(a > 0.0, a * a, b * b)


@wp.kernel(enable_backward=False)
def jet2_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.seed(z[i, 0], 0)
    b = J2_2ND.seed(z[i, 1], 1)
    r = wp.min(a * a, b * b + 5.0) + wp.max(a * a * a, b) + wp.where(a.value > 0.0, a * a, b * b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def _run_grad(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros(m, dtype=J2.coeff, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad], device=device)
    return val.numpy(), grad.numpy().reshape(m, 2)


def _run_hess(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros((m, 2), dtype=float, device=device)
    hess = wp.zeros((m, 2, 2), dtype=float, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad, hess], device=device)
    return val.numpy(), grad.numpy(), hess.numpy()


class TestJetOps(unittest.TestCase):
    def test_first_order_smooth(self):
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_smooth, Z_NP, device)
            np.testing.assert_allclose(val, smooth_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(smooth_np, z64), rtol=1.0e-3, atol=1.0e-4)

    def test_first_order_branch(self):
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch_np, z64), rtol=1.0e-4, atol=1.0e-5)

    def test_second_order_smooth(self):
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_g3, Z_NP, device)
            np.testing.assert_allclose(val, g3_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(g3_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(g3_np, z64), rtol=1.0e-2, atol=1.0e-3)
            np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)

    def test_second_order_branch(self):
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch2_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch2_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(branch2_np, z64), rtol=1.0e-2, atol=1.0e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
