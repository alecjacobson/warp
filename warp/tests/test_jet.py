# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Two local variables per term, so width 2 seeds the identity and one forward
# pass yields the whole local gradient.
J2 = wp.JetSpace(2)

# Six directions: two 3D endpoints.
J6 = wp.JetSpace(6)

# A second width at the same dtype, to check that spaces coexist.
J3 = wp.JetSpace(3)

J2D = wp.JetSpace(2, dtype=wp.float64)


# ----------------------------------------------------------------------------
# g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)
#
# Written as a plain @wp.func over jets: no manual differentiation, and no jet
# names bound into this module. The operators resolve through Warp's builtin
# overload table by argument type.
# ----------------------------------------------------------------------------


@wp.func
def local_energy(a: J2.scalar, b: J2.scalar) -> J2.scalar:
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)


@wp.kernel
def local_gradient(z: wp.array2d[float], grad_g: wp.array[J2.coeff]):
    i = wp.tid()

    # Identity seed: dz0 = [1,0], dz1 = [0,1]
    z0 = J2.seed(z[i, 0], 0)
    z1 = J2.seed(z[i, 1], 1)

    grad_g[i] = local_energy(z0, z1).coeff


@wp.kernel
def local_value(z: wp.array2d[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = local_energy(J2.seed(z[i, 0], 0), J2.seed(z[i, 1], 1)).value


Z_NP = np.array(
    [
        [0.7, -0.4],
        [-0.2, 0.9],
        [0.7, 0.9],
    ],
    dtype=np.float32,
)


def g_np(z):
    """g evaluated with NumPy, independent of any Warp code."""
    a = z[:, 0]
    b = z[:, 1]
    return np.sin(a * b) + 0.1 * a**3 + np.exp(b)


def grad_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.stack(
        (
            b * np.cos(a * b) + 0.3 * a**2,
            a * np.cos(a * b) + np.exp(b),
        ),
        axis=1,
    )


def hessian_np(z):
    a = z[:, 0]
    b = z[:, 1]
    s = np.sin(a * b)

    h = np.empty((z.shape[0], 2, 2))
    h[:, 0, 0] = -(b**2) * s + 0.6 * a
    h[:, 0, 1] = np.cos(a * b) - a * b * s
    h[:, 1, 0] = h[:, 0, 1]
    h[:, 1, 1] = -(a**2) * s + np.exp(b)
    return h


def hessian_fd(z, h=1.0e-4):
    """Second differences of g in float64, deriving nothing by hand."""
    out = np.empty((z.shape[0], 2, 2))

    for p in range(2):
        for q in range(2):
            e_p = np.zeros_like(z)
            e_p[:, p] = h

            e_q = np.zeros_like(z)
            e_q[:, q] = h

            out[:, p, q] = (g_np(z + e_p + e_q) - g_np(z + e_p - e_q) - g_np(z - e_p + e_q) + g_np(z - e_p - e_q)) / (
                4.0 * h * h
            )

    return out


def hessian_from_tape(z_np, device):
    """Hessian of every local term by reverse-over-forward."""
    m = z_np.shape[0]

    z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
    grad_g = wp.zeros(m, dtype=J2.coeff, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(local_gradient, dim=m, inputs=[z], outputs=[grad_g], device=device)

    hessian = np.empty((m, 2, 2), dtype=np.float32)

    for row in range(2):
        seed_np = np.zeros((m, 2), dtype=np.float32)

        # Select the same gradient component for every local term.
        seed_np[:, row] = 1.0
        seed = wp.array(seed_np, dtype=J2.coeff, device=device)

        tape.backward(grads={grad_g: seed})

        # z.grad[i,b] = d grad_g[i,row] / d z[i,b] = H_i[row,b]
        hessian[:, row, :] = z.grad.numpy()

        tape.zero()

    return grad_g.numpy(), hessian


def test_jet_value(test, device):
    m = Z_NP.shape[0]
    out = wp.zeros(m, dtype=float, device=device)

    wp.launch(local_value, dim=m, inputs=[wp.array(Z_NP, dtype=float, device=device)], outputs=[out], device=device)

    np.testing.assert_allclose(out.numpy(), g_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


def test_jet_gradient(test, device):
    grad, _ = hessian_from_tape(Z_NP, device)

    np.testing.assert_allclose(grad, grad_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


def test_jet_hessian(test, device):
    _, hessian = hessian_from_tape(Z_NP, device)

    z64 = Z_NP.astype(np.float64)

    np.testing.assert_allclose(hessian, hessian_np(z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hessian, hessian_fd(z64), rtol=1.0e-3, atol=1.0e-4)


def test_jet_hessian_symmetric(test, device):
    _, hessian = hessian_from_tape(Z_NP, device)

    # The off-diagonals come from separate backward passes, so this is not
    # automatic.
    np.testing.assert_allclose(hessian, np.transpose(hessian, (0, 2, 1)), rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# 3D geometry through the ordinary Warp builtins.
# ----------------------------------------------------------------------------


@wp.func
def spring_energy(x0: J6.vec3, x1: J6.vec3) -> J6.scalar:
    d = x1 - x0
    r = wp.length(d) - 1.0
    return 0.5 * r * r


@wp.kernel
def spring_gradient(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    grad: wp.array[J6.coeff],
):
    i = wp.tid()

    a = J6.seed_vec3(x0[i], 0, 1, 2)
    b = J6.seed_vec3(x1[i], 3, 4, 5)

    grad[i] = spring_energy(a, b).coeff


def test_jet_spring_gradient(test, device):
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)

    x0 = wp.array(x0_np, dtype=wp.vec3, device=device)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device)
    grad = wp.zeros(2, dtype=J6.coeff, device=device)

    wp.launch(spring_gradient, dim=2, inputs=[x0, x1], outputs=[grad], device=device)

    # E = 0.5*(|d|-1)^2, dE/dx1 = (|d|-1) * dhat, dE/dx0 = -dE/dx1
    d = x1_np - x0_np
    lengths = np.linalg.norm(d, axis=1, keepdims=True)
    dedx1 = (lengths - 1.0) * (d / lengths)

    expected = np.concatenate((-dedx1, dedx1), axis=1)

    np.testing.assert_allclose(grad.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# Component access, dot, cross and normalize.
# ----------------------------------------------------------------------------


@wp.kernel
def extract_and_geometry(out: wp.array[float]):
    v = J6.seed_vec3(wp.vec3(3.0, 5.0, 0.0), 0, 1, 2)
    w = J6.seed_vec3(wp.vec3(0.0, 2.0, 4.0), 3, 4, 5)

    # v[i] resolves through the extract builtin
    p = v[0] * v[1]
    out[0] = p.value
    out[1] = p.coeff[0]
    out[2] = p.coeff[1]

    d = wp.dot(v, w)
    out[3] = d.value
    out[4] = d.coeff[0]

    c = wp.cross(v, w)
    out[5] = c.value[0]

    n = wp.normalize(w)
    out[6] = n.value[2]

    out[7] = wp.length_sq(v).value


def test_jet_component_and_geometry(test, device):
    out = wp.zeros(8, dtype=float, device=device)
    wp.launch(extract_and_geometry, dim=1, outputs=[out], device=device)

    got = out.numpy()
    v = np.array([3.0, 5.0, 0.0])
    w = np.array([0.0, 2.0, 4.0])

    # v[0]*v[1] = 15, d/dv0 = v1 = 5, d/dv1 = v0 = 3
    np.testing.assert_allclose(got[0:3], [15.0, 5.0, 3.0], rtol=1.0e-5, atol=1.0e-6)

    # dot = 10, d(dot)/dv0 = w0 = 0
    np.testing.assert_allclose(got[3:5], [float(v @ w), w[0]], rtol=1.0e-5, atol=1.0e-6)

    np.testing.assert_allclose(got[5], np.cross(v, w)[0], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(got[6], (w / np.linalg.norm(w))[2], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(got[7], float(v @ v), rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# Several spaces coexisting, and a non-default dtype.
# ----------------------------------------------------------------------------


@wp.func
def width3_energy(a: J3.scalar, b: J3.scalar) -> J3.scalar:
    return a * b + a


@wp.kernel
def width3_kernel(out: wp.array[float]):
    g = width3_energy(J3.seed(2.0, 0), J3.seed(3.0, 1))
    out[0] = g.value
    out[1] = g.coeff[0]
    out[2] = g.coeff[1]
    out[3] = g.coeff[2]


def test_jet_multiple_widths(test, device):
    """A second width must not disturb the first."""
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(width3_kernel, dim=1, outputs=[out], device=device)

    # g = a*b + a = 8 ; dg/da = b+1 = 4 ; dg/db = a = 2 ; dg/dc = 0
    np.testing.assert_allclose(out.numpy(), [8.0, 4.0, 2.0, 0.0], rtol=1.0e-6, atol=1.0e-6)

    # The width-2 space still resolves after width 3 registered its overloads.
    grad, _ = hessian_from_tape(Z_NP, device)
    np.testing.assert_allclose(grad, grad_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


@wp.func
def energy_f64(a: J2D.scalar, b: J2D.scalar) -> J2D.scalar:
    return wp.sin(a * b) + wp.exp(b)


@wp.kernel
def kernel_f64(out: wp.array[wp.float64]):
    g = energy_f64(J2D.seed(wp.float64(0.7), 0), J2D.seed(wp.float64(-0.4), 1))
    out[0] = g.value
    out[1] = g.coeff[0]
    out[2] = g.coeff[1]


def test_jet_float64(test, device):
    out = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(kernel_f64, dim=1, outputs=[out], device=device)

    a, b = 0.7, -0.4
    expected = [
        np.sin(a * b) + np.exp(b),
        b * np.cos(a * b),
        a * np.cos(a * b) + np.exp(b),
    ]

    # float64 throughout, so this should hold far tighter than float32 would.
    np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-12, atol=1.0e-14)


class TestJetSpace(unittest.TestCase):
    def test_jet_space_is_cached(self):
        self.assertIs(wp.JetSpace(2), J2)
        self.assertIs(wp.JetSpace(2, dtype=wp.float32), J2)
        self.assertIsNot(wp.JetSpace(2, dtype=wp.float64), J2)
        self.assertIs(wp.JetSpace(2, dtype=wp.float64), J2D)

    def test_jet_space_rejects_bad_width(self):
        with self.assertRaises(ValueError):
            wp.JetSpace(0)

        with self.assertRaises(ValueError):
            wp.JetSpace(-1)

    def test_jet_space_reports_width(self):
        self.assertEqual(J2.width, 2)
        self.assertEqual(J6.width, 6)
        self.assertEqual(wp.types.type_size(J2.coeff), 2)
        self.assertEqual(wp.types.type_size(J6.coeff), 6)


# ===========================================================================
# Second-order (forward-over-forward) jets: value, gradient, and Hessian from a
# single forward pass. Cross-checked against the analytic references above, the
# reverse-over-forward Hessian, and finite differences.
# ===========================================================================

J2_2ND = wp.JetSpace2(2)


@wp.kernel
def jet2_value_grad_hess(
    z: wp.array2d[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    # Same energy as local_energy: sin(a*b) + 0.1*a^3 + exp(b).
    i = wp.tid()
    a = J2_2ND.seed(z[i, 0], 0)
    b = J2_2ND.seed(z[i, 1], 1)
    r = wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def g2_np(z):
    # Exercises the ops local_energy does not: div(jet,jet), log, sqrt, cos.
    a = z[:, 0]
    b = z[:, 1]
    return np.log(a * a + b * b + 1.5) + np.sqrt(a * a + 1.0) + a / (b * b + 2.0) + np.cos(a * b)


@wp.kernel
def jet2_g2(
    z: wp.array2d[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    i = wp.tid()
    a = J2_2ND.seed(z[i, 0], 0)
    b = J2_2ND.seed(z[i, 1], 1)
    r = wp.log(a * a + b * b + 1.5) + wp.sqrt(a * a + 1.0) + a / (b * b + 2.0) + wp.cos(a * b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


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


def _run_jet2(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros((m, 2), dtype=float, device=device)
    hess = wp.zeros((m, 2, 2), dtype=float, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad, hess], device=device)
    return val.numpy(), grad.numpy(), hess.numpy()


def test_jet2_value_grad_hessian(test, device):
    val, grad, hess = _run_jet2(jet2_value_grad_hess, Z_NP, device)
    z64 = Z_NP.astype(np.float64)

    # One forward pass reproduces the analytic value, gradient, and Hessian.
    np.testing.assert_allclose(val, g_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad, grad_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(hess, hessian_np(z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hess, hessian_fd(z64), rtol=1.0e-3, atol=1.0e-4)

    # It also agrees with the reverse-over-forward Hessian of the same energy.
    _, hess_tape = hessian_from_tape(Z_NP, device)
    np.testing.assert_allclose(hess, hess_tape, rtol=1.0e-4, atol=1.0e-5)


def test_jet2_hessian_symmetric(test, device):
    # The forward-over-forward Hessian is symmetric by construction.
    _, _, hess = _run_jet2(jet2_value_grad_hess, Z_NP, device)
    np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)


def test_jet2_div_log_sqrt(test, device):
    # Covers div(jet, jet), log, sqrt, and cos, which local_energy does not.
    val, grad, hess = _run_jet2(jet2_g2, Z_NP, device)
    z64 = Z_NP.astype(np.float64)

    np.testing.assert_allclose(val, g2_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad, _grad_fd(g2_np, z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hess, _hess_fd(g2_np, z64), rtol=1.0e-3, atol=1.0e-4)
    np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)


# ===========================================================================
# Extended scalar builtins: inverse-trig, pow variants, and branching ops.
# ===========================================================================


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


@wp.kernel
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


@wp.kernel
def jet_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.coeff]):
    i = wp.tid()
    a = J2.seed(z[i, 0], 0)
    b = J2.seed(z[i, 1], 1)
    e = wp.min(a, b) + wp.max(a, b) + wp.clamp(a, -0.5, 0.5) + wp.where(a.value > 0.0, a, b) + wp.abs(a)
    val[i] = e.value
    grad[i] = e.coeff


def test_jet_smooth_ops(test, device):
    m = Z_NP.shape[0]
    z = wp.array(Z_NP, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros(m, dtype=J2.coeff, device=device)
    wp.launch(jet_smooth, dim=m, inputs=[z], outputs=[val, grad], device=device)
    z64 = Z_NP.astype(np.float64)
    np.testing.assert_allclose(val.numpy(), smooth_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad.numpy().reshape(m, 2), _grad_fd(smooth_np, z64), rtol=1.0e-3, atol=1.0e-4)


def test_jet_branch_ops(test, device):
    # Points chosen away from kinks (a != b, a != 0, a inside the clamp range).
    zb = np.array([[0.3, 0.7], [-0.2, 0.9], [0.4, -0.1]], dtype=np.float32)
    m = zb.shape[0]
    z = wp.array(zb, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros(m, dtype=J2.coeff, device=device)
    wp.launch(jet_branch, dim=m, inputs=[z], outputs=[val, grad], device=device)
    z64 = zb.astype(np.float64)
    np.testing.assert_allclose(val.numpy(), branch_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad.numpy().reshape(m, 2), _grad_fd(branch_np, z64), rtol=1.0e-4, atol=1.0e-5)


devices = get_test_devices()


class TestJet(unittest.TestCase):
    pass


add_function_test(TestJet, "test_jet_value", test_jet_value, devices=devices)
add_function_test(TestJet, "test_jet_gradient", test_jet_gradient, devices=devices)
add_function_test(TestJet, "test_jet_hessian", test_jet_hessian, devices=devices)
add_function_test(TestJet, "test_jet_hessian_symmetric", test_jet_hessian_symmetric, devices=devices)
add_function_test(TestJet, "test_jet_spring_gradient", test_jet_spring_gradient, devices=devices)
add_function_test(TestJet, "test_jet_component_and_geometry", test_jet_component_and_geometry, devices=devices)
add_function_test(TestJet, "test_jet_multiple_widths", test_jet_multiple_widths, devices=devices)
add_function_test(TestJet, "test_jet_float64", test_jet_float64, devices=devices)
add_function_test(TestJet, "test_jet2_value_grad_hessian", test_jet2_value_grad_hessian, devices=devices)
add_function_test(TestJet, "test_jet2_hessian_symmetric", test_jet2_hessian_symmetric, devices=devices)
add_function_test(TestJet, "test_jet2_div_log_sqrt", test_jet2_div_log_sqrt, devices=devices)
add_function_test(TestJet, "test_jet_smooth_ops", test_jet_smooth_ops, devices=devices)
add_function_test(TestJet, "test_jet_branch_ops", test_jet_branch_ops, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
