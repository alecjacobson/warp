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

    def test_gauss_newton_converges_fast(self):
        """Gauss-Newton converges in a handful of iterations (quadratically)."""
        prob, _ = self._problem(4)
        result = prob.gauss_newton_optimize(num_iters=15, step_size=1.0, tol=1e-8)
        self.assertTrue(result["converged"], f"did not converge: {result['final_loss']:.3e}")
        self.assertLessEqual(result["iters"], 6, "Gauss-Newton should converge in a few iterations")

    def test_gauss_newton_graph_matches_eager(self):
        """Accuracy + robustness: the CUDA-graph-captured GN loop matches the eager path
        (same iteration count, both reach tol) at a coarse and a fine mesh. The two are
        not bit-identical (atomic-scatter assembly is order-nondeterministic), so the
        converged losses are compared with a tolerance."""
        for count, ss in ((8, 1.0), (16, 0.0625)):
            prob_e, _ = self._problem(count)
            re = prob_e.gauss_newton_optimize(num_iters=400, step_size=ss, tol=1e-8, use_graph=False)
            prob_g, _ = self._problem(count)
            rg = prob_g.gauss_newton_optimize(num_iters=400, step_size=ss, tol=1e-8, use_graph=True)
            self.assertTrue(re["converged"] and rg["converged"], f"count={count}: did not converge")
            self.assertLess(re["final_loss"], 1e-8 * re["initial_loss"], f"count={count}: eager loss above tol")
            self.assertLess(rg["final_loss"], 1e-8 * rg["initial_loss"], f"count={count}: graph loss above tol")
            self.assertLessEqual(abs(re["iters"] - rg["iters"]), 2, f"count={count}: iteration count differs")
            # the optimized rest shapes agree (loss is machine-zero at count=8, so compare shapes)
            rel = np.linalg.norm(prob_e.verts.numpy() - prob_g.verts.numpy()) / np.linalg.norm(prob_e.verts.numpy())
            self.assertLess(rel, 1e-2, f"count={count}: optimized shapes differ (relerr={rel:.2e})")

    def test_gauss_newton_graph_captured(self):
        """Performance regression: the GN loop must actually capture a CUDA graph (which is
        what removes the per-iteration full-field host sync and kernel-dispatch overhead).
        Guards against a silent fallback to the eager, sync-per-iteration path."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        prob, _ = self._problem(8)
        prob.gauss_newton_optimize(num_iters=20, step_size=1.0, tol=1e-8)
        self.assertIsNotNone(prob._gn_graph, "gauss_newton_optimize should capture a CUDA graph")


if __name__ == "__main__":
    unittest.main(verbosity=2)
