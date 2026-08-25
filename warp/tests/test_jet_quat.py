# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Second-order quaternion jets: the rotation-vector exp map, quaternion
# products, and quat_rotate, checked by driving a rigid-alignment energy and
# comparing its gradient and 3x3 (tangent) or 4x4 (full-space) Hessian against
# finite differences.
#
# CPU-only, by standard unittest.TestCase rather than add_function_test(), for
# the same reason as test_jet_ops.py: these kernels chain many jet overloads,
# and building one module for both CPU and CUDA in the same process trips a
# Warp module-hasher instability where a kernel is then looked up under a hash
# that differs from the one it compiled with. The generated jet code is
# device-independent, so the CPU checks cover the math on both; the runnable
# example_quat_shape_align.py additionally self-verifies on the active device.

import unittest

import numpy as np

import warp as wp

DEVICE = "cpu"

# Width 3: the SO(3) tangent chart. dtheta -> exp_map -> unit quaternion.
J3 = wp.JetSpace2(3, wp.float64)

# Width 4: the full-space quaternion route, seeding all four components.
J4 = wp.JetSpace2(4, wp.float64)

# First-order (width 3): the same chart, read back as a gradient in .coeff.
JF3 = wp.JetSpace(3, wp.float64)

# Cauchy robust-loss scale, for the nonconvex-objective check (log o quat_rotate).
DELTA_NP = 0.3
DELTA = wp.float64(DELTA_NP)


# --------------------------------------------------------------------------
# NumPy reference (quaternion storage [x, y, z, w], matching wp.quat).
# --------------------------------------------------------------------------


def rotvec_to_quat(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0 - 0.125 * a * a])
    s = np.sin(a / 2.0) / a
    return np.array([v[0] * s, v[1] * s, v[2] * s, np.cos(a / 2.0)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + bw * ax + ay * bz - az * by,
            aw * by + bw * ay + az * bx - ax * bz,
            aw * bz + bw * az + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def qrot(q, X):
    # Warp's quat_rotate expansion; agrees with the rotation matrix for unit q
    # and, unlike it, is the exact function Warp evaluates for non-unit q, so
    # the full-space (width-4) finite differences match the jet off the sphere.
    x, y, z, w = q
    c = 2.0 * w * w - 1.0
    out = np.empty_like(X)
    for k in range(X.shape[0]):
        xx, xy, xz = X[k]
        d = 2.0 * (x * xx + y * xy + z * xz)
        out[k, 0] = xx * c + x * d + (y * xz - z * xy) * w * 2.0
        out[k, 1] = xy * c + y * d + (z * xx - x * xz) * w * 2.0
        out[k, 2] = xz * c + z * d + (x * xy - y * xx) * w * 2.0
    return out


def _grad_hess_fd(fn, x0, h=1e-4):
    n = x0.shape[0]
    f0 = fn(x0)
    g = np.zeros(n)
    H = np.zeros((n, n))
    E = np.eye(n)
    for i in range(n):
        fp = fn(x0 + h * E[i])
        fm = fn(x0 - h * E[i])
        g[i] = (fp - fm) / (2.0 * h)
        H[i, i] = (fp - 2.0 * f0 + fm) / h**2
    for i in range(n):
        for j in range(i + 1, n):
            fpp = fn(x0 + h * E[i] + h * E[j])
            fpm = fn(x0 + h * E[i] - h * E[j])
            fmp = fn(x0 - h * E[i] + h * E[j])
            fmm = fn(x0 - h * E[i] - h * E[j])
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h**2)
            H[j, i] = H[i, j]
    return g, H


def _make_problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((9, 3))
    X = X - X.mean(0)

    axis = np.array([0.3, 0.6, -0.2])
    axis = axis / np.linalg.norm(axis)
    q_true = np.array([*(np.sin(np.deg2rad(35) / 2) * axis), np.cos(np.deg2rad(35) / 2)])
    Y = qrot(q_true, X)
    return X, Y, q_true


# --------------------------------------------------------------------------
# Kernels.
# --------------------------------------------------------------------------


@wp.func
def _tangent_energy(
    dtheta: J3.vec3,
    q0: wp.quatd,
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
) -> J3.scalar:
    q = J3.exp_map(dtheta) * q0
    e = J3.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, X[a]) - Y[a]
        e = e + wp.dot(r, r)
    return wp.float64(0.5) * e


@wp.kernel
def _tangent_grad_hess(
    q0: wp.array[wp.quatd],
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
    grad: wp.array[wp.vec3d],
    hess: wp.array[wp.mat33d],
):
    i = wp.tid()
    dtheta = J3.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
    e = _tangent_energy(dtheta, q0[i], X, Y, n)
    grad[i] = e.grad
    hess[i] = e.hess


@wp.func
def _fullspace_energy(
    q: J4.quat,
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
) -> J4.scalar:
    e = J4.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, X[a]) - Y[a]
        e = e + wp.dot(r, r)
    return wp.float64(0.5) * e


@wp.kernel
def _fullspace_grad_hess(
    q0: wp.array[wp.quatd],
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
    grad: wp.array[wp.vec4d],
    hess: wp.array[wp.mat44d],
):
    i = wp.tid()
    q = J4.seed_quat(q0[i], 0, 1, 2, 3)
    e = _fullspace_energy(q, X, Y, n)
    grad[i] = e.grad
    hess[i] = e.hess


@wp.func
def _tangent_energy_1st(
    dtheta: JF3.vec3,
    q0: wp.quatd,
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
) -> JF3.scalar:
    q = JF3.exp_map(dtheta) * q0
    e = JF3.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, X[a]) - Y[a]
        e = e + wp.dot(r, r)
    return wp.float64(0.5) * e


@wp.kernel
def _tangent_grad_1st(
    q0: wp.array[wp.quatd],
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
    grad: wp.array[wp.vec3d],
):
    i = wp.tid()
    dtheta = JF3.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
    e = _tangent_energy_1st(dtheta, q0[i], X, Y, n)
    grad[i] = e.coeff


@wp.func
def _robust_tangent_energy(
    dtheta: J3.vec3,
    q0: wp.quatd,
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
) -> J3.scalar:
    # Cauchy robust loss: sum_a 0.5*delta^2*log(1 + |r_a|^2/delta^2). Nonconvex,
    # and it composes wp.log with quat_rotate on the jet chain.
    q = J3.exp_map(dtheta) * q0
    d2 = DELTA * DELTA
    e = J3.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, X[a]) - Y[a]
        s = wp.dot(r, r)
        e = e + wp.float64(0.5) * d2 * wp.log(wp.float64(1.0) + s / d2)
    return e


@wp.kernel
def _robust_tangent_grad_hess(
    q0: wp.array[wp.quatd],
    X: wp.array[wp.vec3d],
    Y: wp.array[wp.vec3d],
    n: int,
    grad: wp.array[wp.vec3d],
    hess: wp.array[wp.mat33d],
):
    i = wp.tid()
    dtheta = J3.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
    e = _robust_tangent_energy(dtheta, q0[i], X, Y, n)
    grad[i] = e.grad
    hess[i] = e.hess


# --------------------------------------------------------------------------
# Tests (CPU-only; see the module note above).
# --------------------------------------------------------------------------


class TestJetQuat(unittest.TestCase):
    def _launch_tangent(self, q0, X, Y):
        q0_w = wp.array([wp.quatd(*q0)], dtype=wp.quatd, device=DEVICE)
        X_w = wp.array([wp.vec3d(*row) for row in X], dtype=wp.vec3d, device=DEVICE)
        Y_w = wp.array([wp.vec3d(*row) for row in Y], dtype=wp.vec3d, device=DEVICE)
        g_w = wp.zeros(1, dtype=wp.vec3d, device=DEVICE)
        h_w = wp.zeros(1, dtype=wp.mat33d, device=DEVICE)
        wp.launch(_tangent_grad_hess, dim=1, inputs=[q0_w, X_w, Y_w, len(X)], outputs=[g_w, h_w], device=DEVICE)
        wp.synchronize_device(DEVICE)
        return np.array(g_w.numpy()[0]), np.array(h_w.numpy()[0])

    def test_tangent_grad_hess_generic(self):
        # Chart origin around a generic (non-identity) base orientation.
        X, Y, _ = _make_problem(seed=1)
        axis = np.array([-0.5, 0.2, 0.8])
        axis = axis / np.linalg.norm(axis)
        q0 = np.array([*(np.sin(np.deg2rad(20) / 2) * axis), np.cos(np.deg2rad(20) / 2)])

        def energy(dtheta):
            q = qmul(rotvec_to_quat(dtheta), q0)
            r = qrot(q, X) - Y
            return 0.5 * np.sum(r * r)

        g_fd, H_fd = _grad_hess_fd(energy, np.zeros(3))
        g_jet, H_jet = self._launch_tangent(q0, X, Y)

        np.testing.assert_allclose(g_jet, g_fd, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(H_jet, H_fd, atol=1e-4, rtol=1e-4)

    def test_tangent_hessian_symmetric(self):
        X, Y, q_true = _make_problem(seed=2)
        _, H_jet = self._launch_tangent(q_true, X, Y)
        np.testing.assert_allclose(H_jet, H_jet.T, atol=1e-12)

    def test_tangent_at_optimum(self):
        # At the generating orientation the residual is exactly zero, so the
        # gradient vanishes and the Hessian is the Gauss-Newton term.
        X, Y, q_true = _make_problem(seed=3)
        g_jet, H_jet = self._launch_tangent(q_true, X, Y)
        np.testing.assert_allclose(g_jet, np.zeros(3), atol=1e-10)
        # Hessian is positive semidefinite at the minimum.
        w = np.linalg.eigvalsh(H_jet)
        self.assertGreater(w.min(), -1e-9)

    def test_fullspace_grad_hess(self):
        # Seeding all four quaternion components gives honest 4-DOF derivatives
        # of the (non-unit-aware) quat_rotate, matched by finite differences.
        X, Y, q_true = _make_problem(seed=4)
        q0 = q_true

        def energy(q):
            r = qrot(q, X) - Y
            return 0.5 * np.sum(r * r)

        g_fd, H_fd = _grad_hess_fd(energy, q0)

        q0_w = wp.array([wp.quatd(*q0)], dtype=wp.quatd, device=DEVICE)
        X_w = wp.array([wp.vec3d(*row) for row in X], dtype=wp.vec3d, device=DEVICE)
        Y_w = wp.array([wp.vec3d(*row) for row in Y], dtype=wp.vec3d, device=DEVICE)
        g_w = wp.zeros(1, dtype=wp.vec4d, device=DEVICE)
        h_w = wp.zeros(1, dtype=wp.mat44d, device=DEVICE)
        wp.launch(_fullspace_grad_hess, dim=1, inputs=[q0_w, X_w, Y_w, len(X)], outputs=[g_w, h_w], device=DEVICE)
        wp.synchronize_device(DEVICE)

        g_jet = np.array(g_w.numpy()[0])
        H_jet = np.array(h_w.numpy()[0])
        np.testing.assert_allclose(g_jet, g_fd, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(H_jet, H_fd, atol=1e-4, rtol=1e-4)

    def test_first_order_tangent_gradient(self):
        # The first-order jet reads the same tangent gradient off .coeff; its
        # reverse-over-jet Hessian route reuses the vec3 machinery covered in
        # test_jet.py, so here we gate the gradient against finite differences.
        X, Y, _ = _make_problem(seed=6)
        axis = np.array([0.1, -0.7, 0.4])
        axis = axis / np.linalg.norm(axis)
        q0 = np.array([*(np.sin(np.deg2rad(28) / 2) * axis), np.cos(np.deg2rad(28) / 2)])

        def energy(dtheta):
            q = qmul(rotvec_to_quat(dtheta), q0)
            r = qrot(q, X) - Y
            return 0.5 * np.sum(r * r)

        g_fd, _ = _grad_hess_fd(energy, np.zeros(3))

        q0_w = wp.array([wp.quatd(*q0)], dtype=wp.quatd, device=DEVICE)
        X_w = wp.array([wp.vec3d(*row) for row in X], dtype=wp.vec3d, device=DEVICE)
        Y_w = wp.array([wp.vec3d(*row) for row in Y], dtype=wp.vec3d, device=DEVICE)
        g_w = wp.zeros(1, dtype=wp.vec3d, device=DEVICE)
        wp.launch(_tangent_grad_1st, dim=1, inputs=[q0_w, X_w, Y_w, len(X)], outputs=[g_w], device=DEVICE)
        wp.synchronize_device(DEVICE)

        np.testing.assert_allclose(np.array(g_w.numpy()[0]), g_fd, atol=1e-5, rtol=1e-5)

    def test_robust_tangent_grad_hess(self):
        # The nonconvex Cauchy objective the example optimizes. Evaluated at the
        # identity, where the residuals are O(1) >> delta, so the loss is well
        # into its nonlinear regime -- a genuine test of log composed with
        # quat_rotate on the jet chain, not just the near-quadratic basin.
        X, Y, _ = _make_problem(seed=7)
        q0 = np.array([0.0, 0.0, 0.0, 1.0])

        def energy(dtheta):
            q = qmul(rotvec_to_quat(dtheta), q0)
            r = qrot(q, X) - Y
            s = np.sum(r * r, axis=1)
            return np.sum(0.5 * DELTA_NP**2 * np.log(1.0 + s / DELTA_NP**2))

        g_fd, H_fd = _grad_hess_fd(energy, np.zeros(3))

        q0_w = wp.array([wp.quatd(*q0)], dtype=wp.quatd, device=DEVICE)
        X_w = wp.array([wp.vec3d(*row) for row in X], dtype=wp.vec3d, device=DEVICE)
        Y_w = wp.array([wp.vec3d(*row) for row in Y], dtype=wp.vec3d, device=DEVICE)
        g_w = wp.zeros(1, dtype=wp.vec3d, device=DEVICE)
        h_w = wp.zeros(1, dtype=wp.mat33d, device=DEVICE)
        wp.launch(_robust_tangent_grad_hess, dim=1, inputs=[q0_w, X_w, Y_w, len(X)], outputs=[g_w, h_w], device=DEVICE)
        wp.synchronize_device(DEVICE)

        g_jet = np.array(g_w.numpy()[0])
        H_jet = np.array(h_w.numpy()[0])
        np.testing.assert_allclose(g_jet, g_fd, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(H_jet, H_fd, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(H_jet, H_jet.T, atol=1e-12)

    def test_newton_recovers_rotation(self):
        # A few intrinsic Newton steps on the tangent chart recover the
        # generating orientation from the identity, using the pure-forward jet
        # gradient and Hessian each step.
        X, Y, q_true = _make_problem(seed=5)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        for _ in range(20):
            g, H = self._launch_tangent(q, X, Y)
            if np.linalg.norm(g) < 1e-12:
                break
            # Hessian is PSD near the optimum; add a tiny ridge for the solve.
            dtheta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
            q = qmul(rotvec_to_quat(dtheta), q)
            q = q / np.linalg.norm(q)

        ang = 2.0 * np.arccos(min(1.0, abs(np.dot(q, q_true))))
        self.assertLess(np.rad2deg(ang), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
