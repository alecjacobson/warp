# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the 3D inverse-elasticity example. There is no external
reference, so tests are implementation-independent physics checks (rigid-body
modes, eigenvalue count, mesh tiling), a finite-difference gradient check, and
quadratic Gauss-Newton convergence."""

import unittest

import numpy as np

import warp as wp
from warp.examples.optim.example_inverse_elasticity_3d import (
    element_stiffness,
    element_volume,
    make_box,
    mat12,
    scalar,
    vec3,
)


@wp.kernel
def _element_K(verts: wp.array(dtype=vec3), young: scalar, poisson: scalar, out: wp.array(dtype=mat12)):
    out[0] = element_stiffness(verts[0], verts[1], verts[2], verts[3], young, poisson)


@wp.kernel
def _sum_volume(verts: wp.array(dtype=vec3), tets: wp.array2d(dtype=wp.int32), total: wp.array(dtype=scalar)):
    t = wp.tid()
    v = element_volume(verts[tets[t, 0]], verts[tets[t, 1]], verts[tets[t, 2]], verts[tets[t, 3]])
    wp.atomic_add(total, 0, v)


def _numpy_tet_stiffness(P, young, poisson):
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    d = lam + 2 * mu
    C = np.array([[d, lam, lam, 0, 0, 0], [lam, d, lam, 0, 0, 0], [lam, lam, d, 0, 0, 0],
                  [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]])  # fmt: skip
    Dm = np.array([P[0] - P[3], P[1] - P[3], P[2] - P[3]])
    G = np.linalg.inv(Dm) @ np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], dtype=float)
    B = np.zeros((6, 12))
    for j in range(4):
        B[0, 3 * j] = G[0, j]; B[1, 3 * j + 1] = G[1, j]; B[2, 3 * j + 2] = G[2, j]  # noqa: E702
        B[3, 3 * j] = G[1, j]; B[3, 3 * j + 1] = G[0, j]  # noqa: E702
        B[4, 3 * j + 1] = G[2, j]; B[4, 3 * j + 2] = G[1, j]  # noqa: E702
        B[5, 3 * j] = G[2, j]; B[5, 3 * j + 2] = G[0, j]  # noqa: E702
    return (abs(np.linalg.det(Dm)) / 6) * B.T @ C @ B


def _numpy_forward(V, T, fixed, young, poisson, gravity=-9.8):
    """Independent dense numpy assembly + solve of the same physics."""
    nV = V.shape[0]
    K = np.zeros((nV * 3, nV * 3))
    mass = np.zeros(nV)
    for tet in T:
        Ke = _numpy_tet_stiffness(V[tet], young, poisson)
        dofs = np.array([3 * v + c for v in tet for c in range(3)])
        K[np.ix_(dofs, dofs)] += Ke
        Dm = np.array([V[tet[0]] - V[tet[3]], V[tet[1]] - V[tet[3]], V[tet[2]] - V[tet[3]]])
        for v in tet:
            mass[v] += abs(np.linalg.det(Dm)) / 6 / 4
    f_ext = np.zeros((nV, 3)); f_ext[:, 2] = gravity  # noqa: E702
    load = np.repeat(mass, 3) * f_ext.reshape(-1)
    free = [i for i in range(nV) if i not in set(fixed.tolist())]
    fdofs = np.array([3 * v + c for v in free for c in range(3)])
    u = np.zeros(nV * 3)
    u[fdofs] = np.linalg.solve(K[np.ix_(fdofs, fdofs)], load[fdofs])
    return V + u.reshape(-1, 3)


class TestElement3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def _K(self, P, young=2e3, poisson=0.3):
        out = wp.zeros(1, dtype=mat12, device=self.device)
        wp.launch(_element_K, dim=1, inputs=[wp.array(P, dtype=vec3, device=self.device),
                                             scalar(young), scalar(poisson), out], device=self.device)  # fmt: skip
        return out.numpy()[0]

    def test_element_stiffness(self):
        for P in (np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[0.1, 0.2, 0.0], [1.3, 0.1, 0.2], [0.2, 1.1, 0.1], [0.3, 0.2, 1.2]])):  # fmt: skip
            K = self._K(P)
            # matches an independent numpy assembly
            Kn = _numpy_tet_stiffness(P, 2e3, 0.3)
            self.assertLess(np.abs(K - Kn).max() / np.abs(Kn).max(), 1e-10)
            # symmetric, and annihilates the 3 rigid translations
            self.assertLess(np.abs(K - K.T).max() / np.abs(K).max(), 1e-12)
            for e in np.eye(3):
                self.assertLess(np.abs(K @ np.tile(e, 4)).max() / np.abs(K).max(), 1e-12)
            # exactly 6 zero eigenvalues (3 translations + 3 rotations), rest positive
            ev = np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))
            self.assertEqual(int(np.sum(np.abs(ev) < 1e-6 * ev[-1])), 6)
            self.assertGreater(ev[6], 0.0)

    def test_forward_matches_numpy(self):
        """The GPU forward solve matches an independent numpy assembly + solve."""
        from warp.examples.optim.example_inverse_elasticity_3d import InverseElasticity3D  # noqa: PLC0415
        V, T, fixed = make_box(1)
        prob = InverseElasticity3D(V, T, fixed, young=2e3, poisson=0.3, device=self.device)
        U_w = prob.forward().numpy()
        U_n = _numpy_forward(V, T, fixed, 2e3, 0.3)
        rel = np.linalg.norm(U_w - U_n) / np.linalg.norm(U_n - V)
        self.assertLess(rel, 1e-8, f"forward vs numpy relerr={rel:.2e}")

    def test_mesh_tiles_box(self):
        V, T, _ = make_box(2)
        total = wp.zeros(1, dtype=scalar, device=self.device)
        wp.launch(_sum_volume, dim=T.shape[0],
                  inputs=[wp.array(V, dtype=vec3, device=self.device),
                          wp.array(T, dtype=wp.int32, device=self.device), total], device=self.device)  # fmt: skip
        box_vol = np.ptp(V[:, 0]) * np.ptp(V[:, 1]) * np.ptp(V[:, 2])
        self.assertAlmostEqual(total.numpy()[0], box_vol, places=6)
        self.assertAlmostEqual(box_vol, 45.0, places=6)  # 15 : 3 : 1


class TestGaussNewton3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def _problem(self, count):
        from warp.examples.optim.example_inverse_elasticity_3d import InverseElasticity3D  # noqa: PLC0415
        V, T, fixed = make_box(count)
        prob = InverseElasticity3D(V, T, fixed, device=self.device)
        return prob, V, np.array([i for i in range(V.shape[0]) if i not in set(fixed.tolist())])

    def test_gauss_newton_step_matches_finite_difference(self):
        """Analytic GN step vs a Newton step from a finite-difference Jacobian of
        the forward map (independent of the sensitivity assembly). Square system,
        so GN = Newton: p = -J^{-1} r_free."""
        prob, V0, free = self._problem(1)

        def r_free(V):
            prob.verts.assign(wp.array(V, dtype=vec3, device=self.device))
            U = prob.forward().numpy()
            return (V0 - U)[free].reshape(-1)

        prob.verts.assign(wp.array(V0, dtype=vec3, device=self.device))
        p_analytic = prob.gauss_newton_step().numpy().reshape(-1)
        r0 = prob.r_free.numpy().reshape(-1)
        n, eps = free.size * 3, 1e-6
        J = np.zeros((n, n))
        for j in range(n):
            v, c = free[j // 3], j % 3
            Vp = V0.copy(); Vp[v, c] += eps  # noqa: E702
            Vm = V0.copy(); Vm[v, c] -= eps  # noqa: E702
            J[:, j] = (r_free(Vp) - r_free(Vm)) / (2 * eps)
        p_fd = -np.linalg.solve(J, r0)
        rel = np.linalg.norm(p_analytic - p_fd) / np.linalg.norm(p_fd)
        self.assertLess(rel, 1e-5, f"GN step vs FD-Newton relerr={rel:.2e}")

    def test_gauss_newton_converges(self):
        """Gauss-Newton converges quadratically to ~zero in a handful of steps."""
        prob, _, _ = self._problem(2)
        r = prob.gauss_newton_optimize(num_iters=12, step_size=1.0, tol=1e-8)
        self.assertTrue(r["converged"], f"did not converge: {r['final_loss']:.3e}")
        self.assertLessEqual(r["iters"], 7)

    def test_gauss_newton_graph_matches_eager(self):
        """The CUDA-graph-captured GN loop matches the eager path (both converge, same
        optimized shape). Not bit-identical (atomic-scatter assembly is nondeterministic)."""
        prob_e, _, _ = self._problem(2)
        re = prob_e.gauss_newton_optimize(num_iters=15, step_size=1.0, tol=1e-8, use_graph=False)
        prob_g, _, _ = self._problem(2)
        rg = prob_g.gauss_newton_optimize(num_iters=15, step_size=1.0, tol=1e-8, use_graph=True)
        self.assertTrue(re["converged"] and rg["converged"], "did not converge")
        self.assertLessEqual(abs(re["iters"] - rg["iters"]), 2, "eager/graph iteration count differs")
        rel = np.linalg.norm(prob_e.verts.numpy() - prob_g.verts.numpy()) / np.linalg.norm(prob_e.verts.numpy())
        self.assertLess(rel, 1e-2, f"optimized shapes differ (relerr={rel:.2e})")

    def test_gauss_newton_graph_captured(self):
        """Performance regression: the GN loop must capture a CUDA graph (removing the
        per-iteration full-field host sync). Guards against a silent eager fallback."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        prob, _, _ = self._problem(2)
        prob.gauss_newton_optimize(num_iters=10, step_size=1.0, tol=1e-8)
        self.assertIsNotNone(prob._gn_graph, "gauss_newton_optimize should capture a CUDA graph")


if __name__ == "__main__":
    unittest.main(verbosity=2)
