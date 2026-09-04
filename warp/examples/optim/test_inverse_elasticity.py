# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the inverse-elasticity shape-optimization example.

These validate, in layers:
  1. the NumPy/SciPy host oracle (``_inverse_elasticity_oracle``) against dense
     finite differences (reproducing the C++ reference's ``test_main.cpp``);
  2. (added as the example is built) the Warp implementation against that oracle.

Run directly, they need SciPy (host oracle only) in addition to Warp::

    uv run --with scipy warp/examples/optim/test_inverse_elasticity.py

They are example-specific physics checks, deliberately kept out of the core
``warp/tests/**`` suite.
"""

import os
import sys
import unittest

import numpy as np

import warp as wp
import warp.sparse as wps

sys.path.insert(0, os.path.dirname(__file__))
import _inverse_elasticity_oracle as oracle
import example_inverse_elasticity as ex


def _maxabs(a):
    return float(np.abs(a).max())


def _test_devices():
    wp.init()
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    return devices


@wp.kernel
def _eval_local_ops(
    v0: wp.array(dtype=ex.vec2d),
    v1: wp.array(dtype=ex.vec2d),
    v2: wp.array(dtype=ex.vec2d),
    young: wp.float64,
    poisson: wp.float64,
    k_out: wp.array(dtype=ex.mat66d),
    m_out: wp.array(dtype=ex.mat66d),
):
    t = wp.tid()
    k_out[t] = ex.local_stiffness(v0[t], v1[t], v2[t], young, poisson)
    m_out[t] = ex.local_mass(v0[t], v1[t], v2[t])


class TestHostOracle(unittest.TestCase):
    """The numpy/scipy oracle matches dense finite differences and is self-consistent."""

    @classmethod
    def setUpClass(cls):
        cls.p = oracle.Problem(count=2)

    def _args(self):
        p = self.p
        return (p.V, p.F, p.young, p.poisson, p.f_ext, p.free_dofs, p.V_target)

    def test_forward_sags_under_gravity(self):
        p = self.p
        U, _, K, _ = oracle.forward_sim(p.V, p.F, p.young, p.poisson, p.f_ext, p.free_dofs)
        # Bridge sags: interior vertices move downward on average.
        sag = (U - p.V)[p.free_vertices, 1]
        self.assertLess(sag.mean(), 0.0)
        # Assembled stiffness is symmetric and the loss is positive.
        self.assertLess(_maxabs((K - K.T).toarray()), 1e-9)
        self.assertGreater(oracle.loss(*self._args()), 0.0)

    def test_gradient_matches_finite_difference(self):
        g = oracle.gradient_step(*self._args())
        g_fd = oracle.fd_gradient_step(*self._args())
        err = _maxabs(g - g_fd)
        scale = max(1e-12, _maxabs(g_fd))
        self.assertLess(err, 1e-5, f"gradient vs FD max abs err={err:.3e} (rel={err / scale:.3e})")

    def test_gauss_newton_matches_finite_difference(self):
        gn = oracle.gauss_newton_step(*self._args())
        gn_fd = oracle.fd_gauss_newton_step(*self._args())
        err = _maxabs(gn - gn_fd)
        self.assertLess(err, 1e-4, f"Gauss-Newton vs FD max abs err={err:.3e}")

    def test_kkt_matches_square_route(self):
        gn = oracle.gauss_newton_step(*self._args())
        kkt = oracle.gauss_newton_step_kkt(*self._args())
        self.assertLess(_maxabs(gn - kkt), 1e-8)


class TestWarpElementOperators(unittest.TestCase):
    """Warp local_stiffness / local_mass match the numpy oracle (per device)."""

    def test_local_ops_match_oracle(self):
        tris = np.array(
            [
                [[0.1, 0.0], [1.3, -0.2], [0.4, 0.9]],
                [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
                [[-0.5, 0.3], [0.7, 0.1], [0.2, 1.1]],
            ],
            dtype=np.float64,
        )
        young, poisson = 2e3, 0.49
        k_ref = np.stack([oracle.local_stiffness(t, young, poisson) for t in tris])
        m_ref = np.stack([oracle.local_mass(t) for t in tris])

        for device in _test_devices():
            v0 = wp.array(tris[:, 0, :], dtype=ex.vec2d, device=device)
            v1 = wp.array(tris[:, 1, :], dtype=ex.vec2d, device=device)
            v2 = wp.array(tris[:, 2, :], dtype=ex.vec2d, device=device)
            k_out = wp.zeros(len(tris), dtype=ex.mat66d, device=device)
            m_out = wp.zeros(len(tris), dtype=ex.mat66d, device=device)
            wp.launch(
                _eval_local_ops,
                dim=len(tris),
                inputs=[v0, v1, v2, wp.float64(young), wp.float64(poisson), k_out, m_out],
                device=device,
            )
            err_k = _maxabs(k_out.numpy() - k_ref)
            err_m = _maxabs(m_out.numpy() - m_ref)
            self.assertLess(err_k, 1e-8, f"{device}: local_stiffness err {err_k:.3e}")
            self.assertLess(err_m, 1e-10, f"{device}: local_mass err {err_m:.3e}")


class TestWarpForward(unittest.TestCase):
    """Warp assembly and forward solve match the scipy oracle (per device)."""

    def test_assembly_and_forward(self):
        p = oracle.Problem(count=3)
        Uh, _, K, _ = oracle.forward_sim(p.V, p.F, p.young, p.poisson, p.f_ext, p.free_dofs)
        Aff = K[np.ix_(p.free_dofs, p.free_dofs)]
        rng = np.random.default_rng(0)
        v = rng.standard_normal(p.free_dofs.size)
        y_host = Aff @ v

        for device in _test_devices():
            bp = ex.BridgeProblem(p.V, p.F, p.free_vertices, p.young, p.poisson, gravity=-9.8, device=device)
            U, _, _, A = bp.forward(tol=1e-12)

            # Assembled matrix reproduces the host A_ff exactly (matvec check).
            vv = wp.array(v.reshape(-1, 2), dtype=ex.vec2d, device=device)
            yv = wp.zeros(bp.num_free, dtype=ex.vec2d, device=device)
            wps.bsr_mv(A, vv, yv)
            err_mv = _maxabs(yv.numpy().reshape(-1) - y_host) / max(1e-12, _maxabs(y_host))
            self.assertLess(err_mv, 1e-10, f"{device}: A matvec rel err {err_mv:.3e}")

            # Forward displacement matches the direct solver.
            err_u = _maxabs(U.numpy() - Uh) / max(1e-12, _maxabs(Uh - p.V))
            self.assertLess(err_u, 1e-6, f"{device}: forward U rel err {err_u:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
