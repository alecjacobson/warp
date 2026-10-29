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
from typing import Any

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
def jet_smooth(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
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
def jet_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
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
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
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
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
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
    grad = wp.zeros(m, dtype=J2.coeff_type_scalar, device=device)
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
# pow(jet, jet) with a jet exponent, at and around a base of zero.
#
# a**b has finite partials at a = 0 for b > 1, but the textbook factorization
# a**b * (b/a) evaluates as 0 * inf there. Both orders are checked at a = 0
# exactly, where finite differences cannot be used as the oracle.
# --------------------------------------------------------------------------


@wp.kernel
def jet_pow_jet_exponent(z: wp.array2d[float], grad: wp.array[J2.coeff_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
    grad[i] = wp.pow(a, b).coeff


@wp.kernel
def jet_pow_const_exponent(z: wp.array[float], grad: wp.array2d[float]):
    # Degenerate constant exponents at a zero base: p * a**(p-1) and
    # p (p-1) a**(p-2) are 0 * inf there for p == 0 and p == 1.
    i = wp.tid()
    a = J2.variable(z[i], 0)
    grad[i, 0] = wp.pow(a, 0.0).coeff[0]
    grad[i, 1] = wp.pow(a, 1.0).coeff[0]
    grad[i, 2] = wp.pow(a, 2.0).coeff[0]
    grad[i, 3] = wp.pow(a, 0).coeff[0]
    grad[i, 4] = wp.pow(a, 1).coeff[0]


@wp.kernel
def jet2_pow_const_exponent(z: wp.array[float], grad: wp.array2d[float], hess: wp.array2d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i], 0)
    e0 = wp.pow(a, 0)
    grad[i, 0] = e0.grad[0]
    hess[i, 0] = e0.hess[0, 0]

    e1 = wp.pow(a, 1)
    grad[i, 1] = e1.grad[0]
    hess[i, 1] = e1.hess[0, 0]

    e1f = wp.pow(a, 1.0)
    grad[i, 2] = e1f.grad[0]
    hess[i, 2] = e1f.hess[0, 0]

    e2 = wp.pow(a, 2)
    grad[i, 3] = e2.grad[0]
    hess[i, 3] = e2.hess[0, 0]


@wp.kernel
def jet2_pow_jet_exponent(z: wp.array2d[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    e = wp.pow(a, b)
    for p in range(2):
        grad[i, p] = e.grad[p]
        for q in range(2):
            hess[i, p, q] = e.hess[p, q]


# --------------------------------------------------------------------------
# Mixed jet/constant overloads that exist at first order must exist at second
# order too, or an energy written once cannot feed both strategies.
# --------------------------------------------------------------------------


@wp.func
def mixed_scalar_ops(a: J2.scalar, b: J2.scalar) -> J2.scalar:
    return wp.pow(a, 2.5) + wp.atan2(a, 0.75) + wp.atan2(1.25, b)


@wp.func
def mixed_scalar_ops2(a: J2_2ND.scalar, b: J2_2ND.scalar) -> J2_2ND.scalar:
    return wp.pow(a, 2.5) + wp.atan2(a, 0.75) + wp.atan2(1.25, b)


def mixed_scalar_np(z):
    return np.power(z[:, 0], 2.5) + np.arctan2(z[:, 0], 0.75) + np.arctan2(1.25, z[:, 1])


@wp.kernel
def jet_mixed(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff_type_scalar]):
    i = wp.tid()
    e = mixed_scalar_ops(J2.variable(z[i, 0], 0), J2.variable(z[i, 1], 1))
    val[i] = e.value
    grad[i] = e.coeff


@wp.kernel
def jet2_mixed(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    e = mixed_scalar_ops2(J2_2ND.variable(z[i, 0], 0), J2_2ND.variable(z[i, 1], 1))
    val[i] = e.value
    for p in range(2):
        grad[i, p] = e.grad[p]
        for q in range(2):
            hess[i, p, q] = e.hess[p, q]


# --------------------------------------------------------------------------
# Square matrix jets: determinant, trace, transpose, matmul via a real energy.
# --------------------------------------------------------------------------

J4 = wp.JetSpace(4)
J9 = wp.JetSpace(9)
J9_nodes = wp.JetSpace((wp.vec3,) * 3)  # a triangle's three nodes; shares J9's types
J12 = wp.JetSpace((wp.vec3,) * 4)  # a tetrahedron's four nodes


# --------------------------------------------------------------------------
# mat22 multiplication: mat22 * mat22 and mat22 * vec2.
# --------------------------------------------------------------------------


@wp.kernel
def jet_mat22_matmul(z: wp.array2d[float], out: wp.array[J4.coeff_type_scalar]):
    i = wp.tid()
    # A = [[z0, z2], [z1, z3]], built from its two seeded columns. Both
    # mat22 * mat22 and mat22 * vec2 appear in (A A) v.
    c0 = J4.make_vec2(J4.variable(z[i, 0], 0), J4.variable(z[i, 1], 1))
    c1 = J4.make_vec2(J4.variable(z[i, 2], 2), J4.variable(z[i, 3], 3))
    a = J4.make_mat22(c0, c1)
    v = J4.make_vec2(J4.constant(1.0), J4.constant(-2.0))
    w = (a * a) * v
    out[i] = (w[0] + w[1]).coeff


def mat22_matmul_np(z):
    n = z.shape[0]
    a = np.empty((n, 2, 2))
    a[:, 0, 0] = z[:, 0]
    a[:, 1, 0] = z[:, 1]
    a[:, 0, 1] = z[:, 2]
    a[:, 1, 1] = z[:, 3]
    w = np.einsum("nij,njk,k->ni", a, a, np.array([1.0, -2.0]))
    return w[:, 0] + w[:, 1]


@wp.kernel(enable_backward=False)
def jet_mat22_detrace(
    a: wp.array2d[float], det_g: wp.array[J4.coeff_type_scalar], tr_g: wp.array[J4.coeff_type_scalar]
):
    # Seed the four entries (row-major m00,m01,m10,m11) to directions 0..3.
    i = wp.tid()
    c0 = J4.make_vec2(J4.variable(a[i, 0], 0), J4.variable(a[i, 2], 2))  # column 0 = (m00, m10)
    c1 = J4.make_vec2(J4.variable(a[i, 1], 1), J4.variable(a[i, 3], 3))  # column 1 = (m01, m11)
    m = J4.make_mat22(c0, c1)
    det_g[i] = wp.determinant(m).coeff
    tr_g[i] = wp.trace(m).coeff


def tet_np(z):
    p0, p1, p2, p3 = z[:, 0:3], z[:, 3:6], z[:, 6:9], z[:, 9:12]
    f = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)  # columns
    i1 = (f * f).sum((1, 2))
    j = np.linalg.det(f)
    return 0.5 * (i1 - 3.0) - np.log(j) + 0.5 * np.log(j) ** 2


@wp.kernel(enable_backward=False)
def jet_tet(z: wp.array2d[float], grad: wp.array[J12.coeff_type_scalar]):
    i = wp.tid()
    p0, p1, p2, p3 = J12.seed(
        wp.vec3(z[i, 0], z[i, 1], z[i, 2]),
        wp.vec3(z[i, 3], z[i, 4], z[i, 5]),
        wp.vec3(z[i, 6], z[i, 7], z[i, 8]),
        wp.vec3(z[i, 9], z[i, 10], z[i, 11]),
    )
    f = J12.make_mat33(p1 - p0, p2 - p0, p3 - p0)  # deformation gradient
    i1 = wp.trace(wp.transpose(f) * f)  # matmul -> trace = sum of squares
    logj = wp.log(wp.determinant(f))
    e = 0.5 * (i1 - 3.0) - logj + 0.5 * logj * logj
    grad[i] = e.coeff


@wp.kernel(enable_backward=False)
def jet_mat33_inverse(a: wp.array2d[float], out: wp.array3d[float]):
    # Seed the 3x3 entries (row-major dirs 0..8); write d(inv[r,c])/d(entry k).
    i = wp.tid()
    c0 = J9.make_vec3(J9.variable(a[i, 0], 0), J9.variable(a[i, 3], 3), J9.variable(a[i, 6], 6))  # column 0
    c1 = J9.make_vec3(J9.variable(a[i, 1], 1), J9.variable(a[i, 4], 4), J9.variable(a[i, 7], 7))
    c2 = J9.make_vec3(J9.variable(a[i, 2], 2), J9.variable(a[i, 5], 5), J9.variable(a[i, 8], 8))
    inv = wp.inverse(J9.make_mat33(c0, c1, c2))
    for r in range(3):
        for c in range(3):
            for q in range(9):
                out[i, 3 * r + c, q] = inv.coeff[3 * r + c, q]


def triangle_rect_np(z):
    p0, p1, p2 = z[:, 0:3], z[:, 3:6], z[:, 6:9]
    a, b = p1 - p0, p2 - p0
    s00 = (a * a).sum(1)
    s01 = (a * b).sum(1)
    s11 = (b * b).sum(1)
    tr = s00 + s11
    return tr + tr / (s00 * s11 - s01 * s01)


@wp.kernel(enable_backward=False)
def jet_triangle_rect(z: wp.array2d[float], grad: wp.array[J9.coeff_type_scalar]):
    # 3x2 deformation gradient F, then S = F^T F (2x3 * 3x2 = 2x2), symmetric Dirichlet.
    i = wp.tid()
    p0, p1, p2 = J9_nodes.seed(
        wp.vec3(z[i, 0], z[i, 1], z[i, 2]), wp.vec3(z[i, 3], z[i, 4], z[i, 5]), wp.vec3(z[i, 6], z[i, 7], z[i, 8])
    )
    f = J9.make_mat32(p1 - p0, p2 - p0)
    s = wp.transpose(f) * f
    tr = wp.trace(s)
    e = tr + tr / wp.determinant(s)
    grad[i] = e.coeff


class TestJetMatrix(unittest.TestCase):
    def test_mat22_det_trace(self):
        """Check mat22 determinant and trace derivatives against finite differences."""
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((5, 4)) + np.array([2.0, 0.1, 0.1, 2.0])).astype(np.float32)  # well-conditioned
        det_g = wp.zeros(5, dtype=J4.coeff_type_scalar, device="cpu")
        tr_g = wp.zeros(5, dtype=J4.coeff_type_scalar, device="cpu")
        wp.launch(
            jet_mat22_detrace,
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

    def test_mat33_tet_energy(self):
        """Check a mat33 tetrahedron energy gradient against finite differences."""
        rng = np.random.default_rng(1)
        rest = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], np.float32)  # unit reference tet
        z = (rest + 0.05 * rng.standard_normal((6, 12))).astype(np.float32)
        grad = wp.zeros(6, dtype=J12.coeff_type_scalar, device="cpu")
        wp.launch(jet_tet, dim=6, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[grad], device="cpu")
        np.testing.assert_allclose(
            grad.numpy().reshape(6, 12), _grad_fd(tet_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )

    def test_mat33_inverse(self):
        """Check the mat33 inverse derivative against finite differences."""
        rng = np.random.default_rng(2)
        a = (np.eye(3) + 0.3 * rng.standard_normal((4, 3, 3))).astype(np.float32).reshape(4, 9)
        out = wp.zeros((4, 9, 9), dtype=float, device="cpu")
        wp.launch(
            jet_mat33_inverse, dim=4, inputs=[wp.array(a, dtype=float, device="cpu")], outputs=[out], device="cpu"
        )
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

    def test_mat32_triangle(self):
        """Check a rectangular mat32 triangle energy gradient against finite differences."""
        rng = np.random.default_rng(3)
        rest = np.array([0, 0, 0, 1, 0, 0, 0.5, 0.8660254, 0], np.float32)
        z = (rest + 0.1 * rng.standard_normal((5, 9))).astype(np.float32)
        grad = wp.zeros(5, dtype=J9.coeff_type_scalar, device="cpu")
        wp.launch(
            jet_triangle_rect, dim=5, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[grad], device="cpu"
        )
        np.testing.assert_allclose(
            grad.numpy().reshape(5, 9), _grad_fd(triangle_rect_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )

    def test_mat22_matmul(self):
        """Check mat22 * mat22 and mat22 * vec2 derivatives against finite differences.

        ``mat22`` is part of the public namespace and is what a rectangular
        ``mat23 * mat32`` product returns, so ordinary 2D matrix algebra has to
        work on the result.
        """
        rng = np.random.default_rng(0)
        z = (rng.standard_normal((5, 4)) + np.array([2.0, 0.1, 0.1, 2.0])).astype(np.float32)
        out = wp.zeros(5, dtype=J4.coeff_type_scalar, device="cpu")
        wp.launch(jet_mat22_matmul, dim=5, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[out], device="cpu")
        np.testing.assert_allclose(
            out.numpy().reshape(5, 4), _grad_fd(mat22_matmul_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )


class TestJetOps(unittest.TestCase):
    def test_first_order_smooth(self):
        """Check first-order gradients of the smooth op surface against finite differences."""
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_smooth, Z_NP, device)
            np.testing.assert_allclose(val, smooth_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(smooth_np, z64), rtol=1.0e-3, atol=1.0e-4)

    def test_first_order_branch(self):
        """Check first-order gradients of the branching builtins against finite differences."""
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch_np, z64), rtol=1.0e-4, atol=1.0e-5)

    def test_second_order_smooth(self):
        """Check second-order gradients and Hessians of the smooth ops against finite differences."""
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_g3, Z_NP, device)
            np.testing.assert_allclose(val, g3_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(g3_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(g3_np, z64), rtol=1.0e-2, atol=1.0e-3)
            np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)

    def test_second_order_branch(self):
        """Check second-order gradients and Hessians of the branching builtins against finite differences."""
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch2_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch2_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(branch2_np, z64), rtol=1.0e-2, atol=1.0e-3)

    def test_mixed_jet_and_constant_operands(self):
        """Check that mixed jet/constant pow and atan2 agree at first and second order.

        An energy written once against numeric literals has to compile and give
        the same derivatives under both spaces, so the second-order overload set
        must match the first-order one.
        """
        z = np.array([[0.7, 0.4], [1.3, -0.9], [0.2, 1.1]], dtype=np.float32)
        z64 = z.astype(np.float64)

        for device in DEVICES:
            val1, grad1 = _run_grad(jet_mixed, z, device)
            val2, grad2, hess2 = _run_hess(jet2_mixed, z, device)

            np.testing.assert_allclose(val1, mixed_scalar_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(val2, mixed_scalar_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad1, _grad_fd(mixed_scalar_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(grad2, grad1, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(hess2, _hess_fd(mixed_scalar_np, z64), rtol=1.0e-2, atol=1.0e-3)

    def test_pow_with_constant_exponent_at_zero_base(self):
        """Check that the degenerate constant exponents give finite derivatives at a zero base.

        ``x**0`` is constant and ``x**1`` is linear, so their missing
        derivatives are zero, but the general ``p a**(p-1)`` and
        ``p (p-1) a**(p-2)`` terms reach them through ``0 * inf``.
        """
        for device in DEVICES:
            z = wp.array(np.zeros(1, np.float32), dtype=float, device=device)

            g1 = wp.zeros((1, 5), dtype=float, device=device)
            wp.launch(jet_pow_const_exponent, dim=1, inputs=[z], outputs=[g1], device=device)
            # p = 0, 1, 2 as floats, then 0, 1 as ints.
            np.testing.assert_allclose(g1.numpy()[0], [0.0, 1.0, 0.0, 0.0, 1.0])

            g2 = wp.zeros((1, 4), dtype=float, device=device)
            h2 = wp.zeros((1, 4), dtype=float, device=device)
            wp.launch(jet2_pow_const_exponent, dim=1, inputs=[z], outputs=[g2, h2], device=device)
            # p = 0, 1 (int), 1.0 (float), 2.
            np.testing.assert_allclose(g2.numpy()[0], [0.0, 1.0, 1.0, 0.0])
            np.testing.assert_allclose(h2.numpy()[0], [0.0, 0.0, 0.0, 2.0])

    def test_pow_with_jet_exponent_at_zero_base(self):
        """Check that a jet exponent gives finite derivatives at a zero base.

        d(a**b)/da = b a**(b-1) is 0 at a = 0 for b > 1, and d/db is 0 in the
        limit, but the a**b * (b/a) factorization evaluates as 0 * inf there.
        Finite differences cannot be the oracle at the endpoint, so the closed
        form is used directly.
        """
        # Rows 2 and 3 straddle the zero base; b > 1 keeps both partials finite.
        z = np.array([[0.0, 2.0], [0.0, 3.0], [0.5, 2.0]], dtype=np.float32)

        for device in DEVICES:
            m = z.shape[0]
            zd = wp.array(z, dtype=float, device=device)

            grad = wp.zeros(m, dtype=J2.coeff_type_scalar, device=device)
            wp.launch(jet_pow_jet_exponent, dim=m, inputs=[zd], outputs=[grad], device=device)
            g1 = grad.numpy().reshape(m, 2)

            g2 = wp.zeros((m, 2), dtype=float, device=device)
            h2 = wp.zeros((m, 2, 2), dtype=float, device=device)
            wp.launch(jet2_pow_jet_exponent, dim=m, inputs=[zd], outputs=[g2, h2], device=device)

            self.assertTrue(np.isfinite(g1).all(), f"first-order gradient not finite: {g1}")
            self.assertTrue(np.isfinite(g2.numpy()).all(), f"second-order gradient not finite: {g2.numpy()}")
            self.assertTrue(np.isfinite(h2.numpy()).all(), f"second-order Hessian not finite: {h2.numpy()}")

            a, b = z[:, 0].astype(np.float64), z[:, 1].astype(np.float64)
            da = b * np.power(a, b - 1.0)
            db = np.where(a > 0.0, np.power(a, b) * np.log(np.where(a > 0.0, a, 1.0)), 0.0)

            np.testing.assert_allclose(g1, np.stack([da, db], axis=1), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(g2.numpy(), np.stack([da, db], axis=1), rtol=1.0e-5, atol=1.0e-6)


# --------------------------------------------------------------------------
# Mixed jet/plain operands. Each overload is checked against the jet/jet
# overload with the plain operand lifted to a constant jet, which carries the
# same value and a zero derivative, so the two must agree to rounding. The
# overloads do not depend on the width, so width-2 directional jets keep the
# kernels small; the directions are arbitrary functions of the inputs.
# --------------------------------------------------------------------------

mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
mat32 = wp.types.matrix(shape=(3, 2), dtype=float)

JV = wp.JetSpace(1, width=2)
JM2 = wp.JetSpace((wp.mat22, wp.vec2), width=2)
JM3 = wp.JetSpace((wp.mat33, wp.vec3), width=2)
JR = wp.JetSpace((mat23, mat32), width=2)


@wp.func
def scalar_err(a: Any, b: Any) -> float:
    dc = a.coeff - b.coeff
    return (a.value - b.value) * (a.value - b.value) + wp.dot(dc, dc)


@wp.func
def vec_err(a: Any, b: Any) -> float:
    dv = a.value - b.value
    return wp.dot(dv, dv) + wp.ddot(a.coeff - b.coeff, a.coeff - b.coeff)


@wp.func
def mat_err(a: Any, b: Any) -> float:
    return wp.ddot(a.value - b.value, a.value - b.value) + wp.ddot(a.coeff - b.coeff, a.coeff - b.coeff)


@wp.kernel(enable_backward=False)
def mixed_vec_kernel(
    s_in: wp.array[float],
    p2: wp.array[wp.vec2],
    p3: wp.array[wp.vec3],
    c: wp.array[float],
    err: wp.array2d[float],
):
    i = wp.tid()
    s = JV.directional(s_in[i], 1.0, 0.5 * s_in[i] - 0.25)
    l2 = JV.constant_vec2(p2[i])
    l3 = JV.constant_vec3(p3[i])
    lc = JV.constant(c[i])

    err[i, 0] = vec_err(s * p2[i], s * l2)
    err[i, 1] = vec_err(p2[i] * s, l2 * s)
    err[i, 2] = vec_err(s * p3[i], s * l3)
    err[i, 3] = vec_err(p3[i] * s, l3 * s)
    err[i, 4] = vec_err(p2[i] / s, l2 / s)
    err[i, 5] = vec_err(p3[i] / s, l3 / s)
    err[i, 6] = scalar_err(wp.min(s, c[i]), wp.min(s, lc))
    err[i, 7] = scalar_err(wp.min(c[i], s), wp.min(lc, s))
    err[i, 8] = scalar_err(wp.max(s, c[i]), wp.max(s, lc))
    err[i, 9] = scalar_err(wp.max(c[i], s), wp.max(lc, s))


@wp.kernel(enable_backward=False)
def mixed_mat22_kernel(
    a_in: wp.array[wp.mat22],
    v_in: wp.array[wp.vec2],
    pm: wp.array[wp.mat22],
    pv: wp.array[wp.vec2],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, v = JM2.directional(a_in[i], v_in[i], pm[i], pv[i], wp.transpose(a_in[i]), -v_in[i])
    lm = JM2.mat22()
    lm.value = pm[i]
    lv = JM2.constant_vec2(pv[i])

    err[i, 0] = mat_err(a * pm[i], a * lm)
    err[i, 1] = mat_err(pm[i] * a, lm * a)
    err[i, 2] = vec_err(a * pv[i], a * lv)
    err[i, 3] = vec_err(pm[i] * v, lm * v)
    err[i, 4] = mat_err(a + pm[i], a + lm)
    err[i, 5] = mat_err(pm[i] + a, lm + a)
    err[i, 6] = mat_err(a - pm[i], a - lm)
    err[i, 7] = mat_err(pm[i] - a, lm - a)


@wp.kernel(enable_backward=False)
def mixed_mat33_kernel(
    a_in: wp.array[wp.mat33],
    v_in: wp.array[wp.vec3],
    pm: wp.array[wp.mat33],
    pv: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, v = JM3.directional(a_in[i], v_in[i], pm[i], pv[i], wp.transpose(a_in[i]), -v_in[i])
    lm = JM3.mat33()
    lm.value = pm[i]
    lv = JM3.constant_vec3(pv[i])

    err[i, 0] = mat_err(a * pm[i], a * lm)
    err[i, 1] = mat_err(pm[i] * a, lm * a)
    err[i, 2] = vec_err(a * pv[i], a * lv)
    err[i, 3] = vec_err(pm[i] * v, lm * v)
    err[i, 4] = mat_err(a + pm[i], a + lm)
    err[i, 5] = mat_err(pm[i] + a, lm + a)
    err[i, 6] = mat_err(a - pm[i], a - lm)
    err[i, 7] = mat_err(pm[i] - a, lm - a)


@wp.kernel(enable_backward=False)
def mixed_rect_kernel(
    a_in: wp.array[mat23],
    b_in: wp.array[mat32],
    p23: wp.array[mat23],
    p32: wp.array[mat32],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, b = JR.directional(a_in[i], b_in[i], p23[i], p32[i], wp.transpose(b_in[i]), wp.transpose(a_in[i]))
    l23 = JR.mat23()
    l23.value = p23[i]
    l32 = JR.mat32()
    l32.value = p32[i]

    err[i, 0] = mat_err(a * p32[i], a * l32)
    err[i, 1] = mat_err(p23[i] * b, l23 * b)


@wp.kernel
def mixed_mat33_adjoint_kernel(
    a_in: wp.array[wp.mat33],
    v_in: wp.array[wp.vec3],
    pm: wp.array[wp.mat33],
    pv: wp.array[wp.vec3],
    da: wp.array2d[wp.mat33],
    dv: wp.array2d[wp.vec3],
    w: wp.array[wp.vec3],
    weights: wp.array[JM3.coeff_type_vec3],
    out: wp.array[float],
):
    i = wp.tid()
    a, v = JM3.directional(a_in[i], v_in[i], da[i, 0], dv[i, 0], da[i, 1], dv[i, 1])
    y = a * pv[i] + pm[i] * v
    out[i] = wp.dot(y.value, w[i]) + wp.ddot(y.coeff, weights[i])


def _random(rng, shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


class TestJetMixedOperands(unittest.TestCase):
    def _assert_agree(self, err):
        # Squared differences of values and coefficients against the lifted form.
        np.testing.assert_allclose(err, 0.0, atol=1.0e-10)

    def test_scalar_vector_and_selection(self):
        """Check jet-scalar times, and plain vector over, jet-scalar against lifted constants."""
        rng = np.random.default_rng(1)
        m = 6
        s = rng.uniform(0.5, 1.5, size=m).astype(np.float32) * np.where(np.arange(m) % 2, 1.0, -1.0).astype(np.float32)
        # Thresholds on both sides of s so min and max take each branch.
        c = (s + np.where(np.arange(m) % 3 == 0, 0.25, -0.25)).astype(np.float32)
        for device in DEVICES:
            err = wp.zeros((m, 10), dtype=float, device=device)
            wp.launch(
                mixed_vec_kernel,
                dim=m,
                inputs=[
                    wp.array(s, dtype=float, device=device),
                    wp.array(_random(rng, (m, 2)), dtype=wp.vec2, device=device),
                    wp.array(_random(rng, (m, 3)), dtype=wp.vec3, device=device),
                    wp.array(c, dtype=float, device=device),
                ],
                outputs=[err],
                device=device,
            )
            self._assert_agree(err.numpy())

    def test_square_matrix(self):
        """Check mat22 and mat33 products and sums with one plain operand against lifted constants."""
        rng = np.random.default_rng(2)
        m = 4
        for device in DEVICES:
            for kernel, mat, vec, n in (
                (mixed_mat22_kernel, wp.mat22, wp.vec2, 2),
                (mixed_mat33_kernel, wp.mat33, wp.vec3, 3),
            ):
                err = wp.zeros((m, 8), dtype=float, device=device)
                wp.launch(
                    kernel,
                    dim=m,
                    inputs=[
                        wp.array(_random(rng, (m, n, n)), dtype=mat, device=device),
                        wp.array(_random(rng, (m, n)), dtype=vec, device=device),
                        wp.array(_random(rng, (m, n, n)), dtype=mat, device=device),
                        wp.array(_random(rng, (m, n)), dtype=vec, device=device),
                    ],
                    outputs=[err],
                    device=device,
                )
                self._assert_agree(err.numpy())

    def test_rectangular_matrix(self):
        """Check mat23 times mat32 with either operand plain against lifted constants."""
        rng = np.random.default_rng(3)
        m = 4
        for device in DEVICES:
            err = wp.zeros((m, 2), dtype=float, device=device)
            wp.launch(
                mixed_rect_kernel,
                dim=m,
                inputs=[
                    wp.array(_random(rng, (m, 2, 3)), dtype=mat23, device=device),
                    wp.array(_random(rng, (m, 3, 2)), dtype=mat32, device=device),
                    wp.array(_random(rng, (m, 2, 3)), dtype=mat23, device=device),
                    wp.array(_random(rng, (m, 3, 2)), dtype=mat32, device=device),
                ],
                outputs=[err],
                device=device,
            )
            self._assert_agree(err.numpy())

    def test_plain_operand_adjoints(self):
        """Check that reverse mode through mixed products reaches the plain operands."""
        rng = np.random.default_rng(4)
        m = 3
        a_np = _random(rng, (m, 3, 3))
        v_np = _random(rng, (m, 3))
        pm_np = _random(rng, (m, 3, 3))
        pv_np = _random(rng, (m, 3))
        da_np = _random(rng, (m, 2, 3, 3))
        dv_np = _random(rng, (m, 2, 3))
        w_np = _random(rng, (m, 3))
        weights_np = _random(rng, (m, 3, 2))

        # y = A pv + Pm v carries y.coeff[:, q] = dA_q pv + Pm dv_q along direction q.
        pv_grad = np.einsum("mrk,mr->mk", a_np, w_np) + np.einsum("mqrk,mrq->mk", da_np, weights_np)
        pm_grad = np.einsum("mr,mk->mrk", w_np, v_np) + np.einsum("mrq,mqk->mrk", weights_np, dv_np)

        for device in DEVICES:
            pm = wp.array(pm_np, dtype=wp.mat33, requires_grad=True, device=device)
            pv = wp.array(pv_np, dtype=wp.vec3, requires_grad=True, device=device)
            out = wp.zeros(m, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    mixed_mat33_adjoint_kernel,
                    dim=m,
                    inputs=[
                        wp.array(a_np, dtype=wp.mat33, device=device),
                        wp.array(v_np, dtype=wp.vec3, device=device),
                        pm,
                        pv,
                        wp.array(da_np, dtype=wp.mat33, device=device),
                        wp.array(dv_np, dtype=wp.vec3, device=device),
                        wp.array(w_np, dtype=wp.vec3, device=device),
                        wp.array(weights_np, dtype=JM3.coeff_type_vec3, device=device),
                    ],
                    outputs=[out],
                    device=device,
                )
            tape.backward(grads={out: wp.ones(m, dtype=float, device=device)})

            np.testing.assert_allclose(pv.grad.numpy(), pv_grad, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(pm.grad.numpy(), pm_grad, rtol=1.0e-5, atol=1.0e-6)


# --------------------------------------------------------------------------
# Component-wise products, checked against the same product assembled from
# per-component scalar operations, for first- and second-order vector jets
# with either operand plain.
# --------------------------------------------------------------------------

JC2 = wp.JetSpace((wp.vec2, wp.vec2), width=2)
JC3 = wp.JetSpace((wp.vec3, wp.vec3), width=2)
J2C2 = wp.JetSpace2((wp.vec2, wp.vec2))
J2C3 = wp.JetSpace2((wp.vec3, wp.vec3))


@wp.func
def cw_mul_by_components(a: Any, b: Any, e0: Any, e1: Any):
    return a[0] * b[0] * e0 + a[1] * b[1] * e1


@wp.func
def cw_div_by_components(a: Any, b: Any, e0: Any, e1: Any):
    return a[0] / b[0] * e0 + a[1] / b[1] * e1


@wp.func
def cw_mul_by_components(a: Any, b: Any, e0: Any, e1: Any, e2: Any):
    return a[0] * b[0] * e0 + a[1] * b[1] * e1 + a[2] * b[2] * e2


@wp.func
def cw_div_by_components(a: Any, b: Any, e0: Any, e1: Any, e2: Any):
    return a[0] / b[0] * e0 + a[1] / b[1] * e1 + a[2] / b[2] * e2


@wp.func
def jet2_scalar_err(a: Any, b: Any) -> float:
    dg = a.grad - b.grad
    dh = a.hess - b.hess
    return (a.value - b.value) * (a.value - b.value) + wp.dot(dg, dg) + wp.ddot(dh, dh)


@wp.func
def jet2_vec2_err(a: Any, b: Any) -> float:
    return jet2_scalar_err(a.x, b.x) + jet2_scalar_err(a.y, b.y)


@wp.func
def jet2_vec3_err(a: Any, b: Any) -> float:
    return jet2_scalar_err(a.x, b.x) + jet2_scalar_err(a.y, b.y) + jet2_scalar_err(a.z, b.z)


@wp.kernel(enable_backward=False)
def cw_first_order_kernel(
    a2_in: wp.array[wp.vec2],
    b2_in: wp.array[wp.vec2],
    a3_in: wp.array[wp.vec3],
    b3_in: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    pa2 = a2_in[i]
    pb2 = b2_in[i]
    pa3 = a3_in[i]
    pb3 = b3_in[i]
    a2, b2 = JC2.directional(pa2, pb2, pb2, -pa2, wp.vec2(1.0, -0.5), pa2)
    a3, b3 = JC3.directional(pa3, pb3, pb3, -pa3, wp.vec3(1.0, -0.5, 0.25), pa3)
    x2 = wp.vec2(1.0, 0.0)
    y2 = wp.vec2(0.0, 1.0)
    x3 = wp.vec3(1.0, 0.0, 0.0)
    y3 = wp.vec3(0.0, 1.0, 0.0)
    z3 = wp.vec3(0.0, 0.0, 1.0)

    err[i, 0] = vec_err(wp.cw_mul(a2, b2), cw_mul_by_components(a2, b2, x2, y2))
    err[i, 1] = vec_err(wp.cw_mul(a2, pb2), cw_mul_by_components(a2, pb2, x2, y2))
    err[i, 2] = vec_err(wp.cw_mul(pa2, b2), cw_mul_by_components(pa2, b2, x2, y2))
    err[i, 3] = vec_err(wp.cw_div(a2, b2), cw_div_by_components(a2, b2, x2, y2))
    err[i, 4] = vec_err(wp.cw_div(a2, pb2), cw_div_by_components(a2, pb2, x2, y2))
    err[i, 5] = vec_err(wp.cw_div(pa2, b2), cw_div_by_components(pa2, b2, x2, y2))
    err[i, 6] = vec_err(wp.cw_mul(a3, b3), cw_mul_by_components(a3, b3, x3, y3, z3))
    err[i, 7] = vec_err(wp.cw_mul(a3, pb3), cw_mul_by_components(a3, pb3, x3, y3, z3))
    err[i, 8] = vec_err(wp.cw_mul(pa3, b3), cw_mul_by_components(pa3, b3, x3, y3, z3))
    err[i, 9] = vec_err(wp.cw_div(a3, b3), cw_div_by_components(a3, b3, x3, y3, z3))
    err[i, 10] = vec_err(wp.cw_div(a3, pb3), cw_div_by_components(a3, pb3, x3, y3, z3))
    err[i, 11] = vec_err(wp.cw_div(pa3, b3), cw_div_by_components(pa3, b3, x3, y3, z3))


@wp.kernel(enable_backward=False)
def cw_second_order_kernel(
    a2_in: wp.array[wp.vec2],
    b2_in: wp.array[wp.vec2],
    a3_in: wp.array[wp.vec3],
    b3_in: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    pa2 = a2_in[i]
    pb2 = b2_in[i]
    pa3 = a3_in[i]
    pb3 = b3_in[i]
    # Square the seeded jets so the operands carry nonzero Hessians.
    s2, t2 = J2C2.seed(pa2, pb2)
    s3, t3 = J2C3.seed(pa3, pb3)
    a2 = wp.cw_mul(s2, s2) + t2
    b2 = wp.cw_mul(t2, t2) - s2 + wp.vec2(2.0)
    a3 = wp.cw_mul(s3, s3) + t3
    b3 = wp.cw_mul(t3, t3) - s3 + wp.vec3(2.0)
    x2 = wp.vec2(1.0, 0.0)
    y2 = wp.vec2(0.0, 1.0)
    x3 = wp.vec3(1.0, 0.0, 0.0)
    y3 = wp.vec3(0.0, 1.0, 0.0)
    z3 = wp.vec3(0.0, 0.0, 1.0)

    err[i, 0] = jet2_vec2_err(wp.cw_mul(a2, b2), cw_mul_by_components(a2, b2, x2, y2))
    err[i, 1] = jet2_vec2_err(wp.cw_mul(a2, pb2), cw_mul_by_components(a2, pb2, x2, y2))
    err[i, 2] = jet2_vec2_err(wp.cw_mul(pa2, b2), cw_mul_by_components(pa2, b2, x2, y2))
    err[i, 3] = jet2_vec2_err(wp.cw_div(a2, b2), cw_div_by_components(a2, b2, x2, y2))
    err[i, 4] = jet2_vec2_err(wp.cw_div(a2, pb2), cw_div_by_components(a2, pb2, x2, y2))
    err[i, 5] = jet2_vec2_err(wp.cw_div(pa2, b2), cw_div_by_components(pa2, b2, x2, y2))
    err[i, 6] = jet2_vec3_err(wp.cw_mul(a3, b3), cw_mul_by_components(a3, b3, x3, y3, z3))
    err[i, 7] = jet2_vec3_err(wp.cw_mul(a3, pb3), cw_mul_by_components(a3, pb3, x3, y3, z3))
    err[i, 8] = jet2_vec3_err(wp.cw_mul(pa3, b3), cw_mul_by_components(pa3, b3, x3, y3, z3))
    err[i, 9] = jet2_vec3_err(wp.cw_div(a3, b3), cw_div_by_components(a3, b3, x3, y3, z3))
    err[i, 10] = jet2_vec3_err(wp.cw_div(a3, pb3), cw_div_by_components(a3, pb3, x3, y3, z3))
    err[i, 11] = jet2_vec3_err(wp.cw_div(pa3, b3), cw_div_by_components(pa3, b3, x3, y3, z3))


class TestJetComponentWise(unittest.TestCase):
    def _launch(self, kernel):
        rng = np.random.default_rng(5)
        m = 5

        def operand(n):
            # Magnitudes in [0.5, 1.5] with random signs keep the divisors away from zero.
            return (rng.uniform(0.5, 1.5, size=(m, n)) * rng.choice([-1.0, 1.0], size=(m, n))).astype(np.float32)

        for device in DEVICES:
            err = wp.zeros((m, 12), dtype=float, device=device)
            wp.launch(
                kernel,
                dim=m,
                inputs=[
                    wp.array(operand(2), dtype=wp.vec2, device=device),
                    wp.array(operand(2), dtype=wp.vec2, device=device),
                    wp.array(operand(3), dtype=wp.vec3, device=device),
                    wp.array(operand(3), dtype=wp.vec3, device=device),
                ],
                outputs=[err],
                device=device,
            )
            np.testing.assert_allclose(err.numpy(), 0.0, atol=1.0e-9)

    def test_first_order(self):
        """Check first-order cw_mul and cw_div against per-component products, with either operand plain."""
        self._launch(cw_first_order_kernel)

    def test_second_order(self):
        """Check second-order cw_mul and cw_div against per-component products, with either operand plain."""
        self._launch(cw_second_order_kernel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
