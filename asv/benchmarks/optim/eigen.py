# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure prepared eigen solves on procedural FEM pencils."""

import warp as wp
import warp.fem as fem
from warp.optim import eigen, linear

from ..benchmarks_utils import setup_once


@fem.integrand
def _stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@fem.integrand
def _mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


class SymmetricEigen:
    params = [(8, 24), (4, 12), ("lobpcg", "shift-invert"), ("cpu", "cuda:0")]
    param_names = ["resolution", "modes", "algorithm", "device"]
    number = 1
    repeat = 5
    rounds = 2
    timeout = 600

    @setup_once
    def setup(self, resolution, modes, algorithm, device):
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            geometry = fem.Grid2D(res=wp.vec2i(resolution), scalar_type=wp.float64)
            space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.float64)
            domain = fem.Cells(geometry)
            test = fem.make_test(space, domain=domain)
            trial = fem.make_trial(space, domain=domain)
            self.stiffness = fem.integrate(_stiffness_form, fields={"u": trial, "v": test}, output_dtype=wp.float64)
            self.mass = fem.integrate(_mass_form, fields={"u": trial, "v": test}, output_dtype=wp.float64)
            self.preconditioner = linear.preconditioner(self.stiffness, "diag")
            self.modes, self.algorithm = modes, algorithm
            self.solver = self._make_solver().prepare()
            self.solver().check()
            wp.synchronize_device(self.device)

    def _make_solver(self):
        correction = linear.KrylovSolve(
            self.stiffness,
            self.modes if self.algorithm == "lobpcg" else 1,
            method="cg",
            M=self.preconditioner,
            tol=0.01 if self.algorithm == "lobpcg" else 1e-12,
            maxiter=200,
            residual="fresh",
        )
        if self.algorithm == "lobpcg":
            return eigen.LOBPCG(self.stiffness, self.modes, mass=self.mass, preconditioner=correction, maxiter=100)
        return eigen.KrylovSchur(self.stiffness, self.modes, mass=self.mass, inverse=correction, sigma=0.0, maxiter=100)

    def time_prepared_solve(self, resolution, modes, algorithm, device):
        self.solver()
        wp.synchronize_device(self.device)

    def time_setup_and_prepare(self, resolution, modes, algorithm, device):
        # Assembly is excluded; include all solver workspace and first-upload preparation.
        state = self._make_solver().prepare()
        wp.synchronize_device(self.device)
        return state

    def track_iterations(self, resolution, modes, algorithm, device):
        return int(self.solver.result.iterations.numpy()[0])

    def track_relative_residual(self, resolution, modes, algorithm, device):
        return float(self.solver.result.relative_residuals.numpy().max())
