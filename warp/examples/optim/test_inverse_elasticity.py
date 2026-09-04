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

sys.path.insert(0, os.path.dirname(__file__))
import _inverse_elasticity_oracle as oracle


def _maxabs(a):
    return float(np.abs(a).max())


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
