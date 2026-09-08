# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lean regression tests for the inverse-elasticity example: a self-contained
finite-difference gradient check and an end-to-end Adam convergence check."""

import unittest

import numpy as np

import warp as wp
from warp.examples.optim.example_inverse_elasticity import InverseElasticity, make_bridge


class TestInverseElasticity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def _problem(self, count):
        V, F, fixed = make_bridge(count)
        return InverseElasticity(V, F, fixed, device=self.device), V

    def test_gradient_matches_finite_difference(self):
        """Analytic adjoint gradient agrees with a central finite difference."""
        prob, _ = self._problem(2)
        g = prob.gradient_np()  # (n_free, 2), double precision

        free = prob.free_verts.numpy()
        rng = np.random.default_rng(0)
        picks = rng.choice(free.size, size=8, replace=False)
        eps = 1e-6
        V0 = prob.verts.numpy().copy()
        analytic, finite_diff = [], []
        for i in picks:
            v = int(free[i])
            for c in range(2):
                Vp = V0.copy(); Vp[v, c] += eps  # noqa: E702
                prob.verts.assign(wp.array(Vp, dtype=prob.verts.dtype, device=self.device))
                lp = prob.loss()
                Vm = V0.copy(); Vm[v, c] -= eps  # noqa: E702
                prob.verts.assign(wp.array(Vm, dtype=prob.verts.dtype, device=self.device))
                lm = prob.loss()
                finite_diff.append((lp - lm) / (2 * eps))
                analytic.append(g[i, c])
        analytic, finite_diff = np.array(analytic), np.array(finite_diff)
        # Norm-based check (per-component relative error is brittle where a
        # gradient component is near zero); the analytic gradient is separately
        # validated to ~1e-11 against the C++ reference in development.
        rel = np.linalg.norm(analytic - finite_diff) / np.linalg.norm(finite_diff)
        self.assertLess(rel, 1e-4, f"gradient vs finite-difference relerr={rel:.2e}")

    def test_adam_converges(self):
        """End-to-end Adam reaches the tol=1e-8 criterion at count=2 (~575 iters)."""
        prob, _ = self._problem(2)
        result = prob.optimize(num_iters=700, tol=1e-8)
        self.assertTrue(result["converged"], f"did not converge: {result['final_loss']:.3e}")
        self.assertLess(result["final_loss"], 1e-8 * result["initial_loss"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
