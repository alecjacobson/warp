# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure optional direct setup and replay on procedural FEM systems."""

import importlib.util

import warp as wp
import warp.fem as fem
from warp.optim import linear

from ..benchmarks_utils import setup_once


@fem.integrand
def _stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


class PreparedCuDSS:
    params = [(24, 64), (1, 8, 32)]
    param_names = ["resolution", "nrhs"]
    number = 1
    repeat = 5
    rounds = 2
    timeout = 300

    @setup_once
    def setup(self, resolution, nrhs):
        if importlib.util.find_spec("warp_cudss") is None or not wp.is_cuda_available():
            raise NotImplementedError("Requires optional warp-cudss, NVIDIA cuDSS and CUDA")
        with wp.ScopedDevice("cuda:0"):
            geometry = fem.Grid2D(res=wp.vec2i(resolution), scalar_type=wp.float64)
            space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.float64)
            domain = fem.Cells(geometry)
            test = fem.make_test(space, domain=domain)
            trial = fem.make_trial(space, domain=domain)
            self.matrix = fem.integrate(_stiffness_form, fields={"u": trial, "v": test}, output_dtype=wp.float64)
            self.solver = linear.CuDSS(self.matrix, nrhs)
            self.rhs = wp.ones((nrhs, self.matrix.shape[0]), dtype=wp.float64)
            self.out = wp.empty_like(self.rhs)
            self.solver.solve(self.rhs, self.out)
            if not self.solver.converged.numpy().all():
                raise RuntimeError("Direct solve failed fresh residual acceptance")
            with wp.ScopedCapture() as capture:
                self.solver.solve(self.rhs, self.out)
            self.graph = capture.graph

    def time_setup(self, resolution, nrhs):
        state = linear.CuDSS(self.matrix, nrhs)
        state.close()

    def time_replay(self, resolution, nrhs):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def track_relative_residual(self, resolution, nrhs):
        return float(self.solver.relative_residual.numpy().max())
