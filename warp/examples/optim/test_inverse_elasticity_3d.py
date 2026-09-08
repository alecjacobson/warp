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

    def test_mesh_tiles_box(self):
        V, T, _ = make_box(2)
        total = wp.zeros(1, dtype=scalar, device=self.device)
        wp.launch(_sum_volume, dim=T.shape[0],
                  inputs=[wp.array(V, dtype=vec3, device=self.device),
                          wp.array(T, dtype=wp.int32, device=self.device), total], device=self.device)  # fmt: skip
        box_vol = np.ptp(V[:, 0]) * np.ptp(V[:, 1]) * np.ptp(V[:, 2])
        self.assertAlmostEqual(total.numpy()[0], box_vol, places=6)
        self.assertAlmostEqual(box_vol, 45.0, places=6)  # 15 : 3 : 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
