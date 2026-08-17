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


# --------------------------------------------------------------------------
# Square matrix jets: determinant, trace, transpose, matmul via a real energy.
# --------------------------------------------------------------------------

J4 = wp.JetSpace(4)
J9 = wp.JetSpace(9)
J12 = wp.JetSpace(12)


@wp.kernel(enable_backward=False)
def jet_mat2_detrace(a: wp.array2d[float], det_g: wp.array[J4.coeff], tr_g: wp.array[J4.coeff]):
    # Seed the four entries (row-major m00,m01,m10,m11) to directions 0..3.
    i = wp.tid()
    c0 = J4.seed_vec2(wp.vec2(a[i, 0], a[i, 2]), 0, 2)  # column 0 = (m00, m10)
    c1 = J4.seed_vec2(wp.vec2(a[i, 1], a[i, 3]), 1, 3)  # column 1 = (m01, m11)
    m = J4.make_mat2(c0, c1)
    det_g[i] = wp.determinant(m).coeff
    tr_g[i] = wp.trace(m).coeff


def tet_np(z):
    p0, p1, p2, p3 = z[:, 0:3], z[:, 3:6], z[:, 6:9], z[:, 9:12]
    f = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)  # columns
    i1 = (f * f).sum((1, 2))
    j = np.linalg.det(f)
    return 0.5 * (i1 - 3.0) - np.log(j) + 0.5 * np.log(j) ** 2


@wp.kernel(enable_backward=False)
def jet_tet(z: wp.array2d[float], grad: wp.array[J12.coeff]):
    i = wp.tid()
    p0 = J12.seed_vec3(wp.vec3(z[i, 0], z[i, 1], z[i, 2]), 0, 1, 2)
    p1 = J12.seed_vec3(wp.vec3(z[i, 3], z[i, 4], z[i, 5]), 3, 4, 5)
    p2 = J12.seed_vec3(wp.vec3(z[i, 6], z[i, 7], z[i, 8]), 6, 7, 8)
    p3 = J12.seed_vec3(wp.vec3(z[i, 9], z[i, 10], z[i, 11]), 9, 10, 11)
    f = J12.make_mat3(p1 - p0, p2 - p0, p3 - p0)  # deformation gradient
    i1 = wp.trace(wp.transpose(f) * f)  # matmul -> trace = sum of squares
    logj = wp.log(wp.determinant(f))
    e = 0.5 * (i1 - 3.0) - logj + 0.5 * logj * logj
    grad[i] = e.coeff


@wp.kernel(enable_backward=False)
def jet_mat3_inverse(a: wp.array2d[float], out: wp.array3d[float]):
    # Seed the 3x3 entries (row-major dirs 0..8); write d(inv[r,c])/d(entry k).
    i = wp.tid()
    c0 = J9.seed_vec3(wp.vec3(a[i, 0], a[i, 3], a[i, 6]), 0, 3, 6)  # column 0
    c1 = J9.seed_vec3(wp.vec3(a[i, 1], a[i, 4], a[i, 7]), 1, 4, 7)
    c2 = J9.seed_vec3(wp.vec3(a[i, 2], a[i, 5], a[i, 8]), 2, 5, 8)
    inv = wp.inverse(J9.make_mat3(c0, c1, c2))
    for r in range(3):
        for c in range(3):
            for q in range(9):
                out[i, 3 * r + c, q] = inv.coeff[3 * r + c, q]


class TestJetMatrix(unittest.TestCase):
    def test_mat2_det_trace(self):
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((5, 4)) + np.array([2.0, 0.1, 0.1, 2.0])).astype(np.float32)  # well-conditioned
        det_g = wp.zeros(5, dtype=J4.coeff, device="cpu")
        tr_g = wp.zeros(5, dtype=J4.coeff, device="cpu")
        wp.launch(
            jet_mat2_detrace,
            dim=5,
            inputs=[wp.array(a, dtype=float, device="cpu")],
            outputs=[det_g, tr_g],
            device="cpu",
        )
        m = a.astype(np.float64)
        # d det / d(m00,m01,m10,m11) = (m11, -m10, -m01, m00); d trace = (1,0,0,1)
        det_expected = np.stack([m[:, 3], -m[:, 2], -m[:, 1], m[:, 0]], axis=1)
        tr_expected = np.tile([1.0, 0.0, 0.0, 1.0], (5, 1))
        np.testing.assert_allclose(det_g.numpy().reshape(5, 4), det_expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(tr_g.numpy().reshape(5, 4), tr_expected, rtol=1e-5, atol=1e-6)

    def test_mat3_tet_energy(self):
        rng = np.random.default_rng(1)
        rest = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], np.float32)  # unit reference tet
        z = (rest + 0.05 * rng.standard_normal((6, 12))).astype(np.float32)
        grad = wp.zeros(6, dtype=J12.coeff, device="cpu")
        wp.launch(jet_tet, dim=6, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[grad], device="cpu")
        np.testing.assert_allclose(
            grad.numpy().reshape(6, 12), _grad_fd(tet_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )

    def test_mat3_inverse(self):
        rng = np.random.default_rng(2)
        a = (np.eye(3) + 0.3 * rng.standard_normal((4, 3, 3))).astype(np.float32).reshape(4, 9)
        out = wp.zeros((4, 9, 9), dtype=float, device="cpu")
        wp.launch(jet_mat3_inverse, dim=4, inputs=[wp.array(a, dtype=float, device="cpu")], outputs=[out], device="cpu")
        got = out.numpy()
        # FD of inv(A) w.r.t. each entry, compared to d(inv[r,c])/d(entry k).
        h = 1e-4
        for n in range(4):
            A = a[n].reshape(3, 3).astype(np.float64)
            for k in range(9):
                r, c = k // 3, k % 3
                ap = A.copy()
                ap[r, c] += h
                am = A.copy()
                am[r, c] -= h
                dinv = (np.linalg.inv(ap) - np.linalg.inv(am)) / (2 * h)  # 3x3
                np.testing.assert_allclose(got[n, :, k], dinv.reshape(9), rtol=1e-3, atol=1e-4)


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
