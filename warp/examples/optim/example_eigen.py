# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute low modes of a reaction-diffusion operator assembled with warp.fem.

Run with ``python -m warp.examples.optim.example_eigen --device cpu``.
The unit-square problem has natural Neumann boundaries and a unit reaction
term, so its first eigenvalue is one and its mass matrix is positive definite.
"""

import argparse

import warp as wp
import warp.fem as fem
from warp.optim import eigen, linear


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


class Example:
    def __init__(self, resolution=8, modes=4, device=None):
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            geometry = fem.Grid2D(res=wp.vec2i(resolution), scalar_type=wp.float64)
            space = fem.make_polynomial_space(geometry, degree=1, dtype=wp.float64)
            domain = fem.Cells(geometry)
            test = fem.make_test(space, domain=domain)
            trial = fem.make_trial(space, domain=domain)
            self.stiffness = fem.integrate(stiffness_form, fields={"u": trial, "v": test}, output_dtype=wp.float64)
            self.mass = fem.integrate(mass_form, fields={"u": trial, "v": test}, output_dtype=wp.float64)
            preconditioner = linear.KrylovSolve(
                self.stiffness,
                modes,
                method="cg",
                tol=0.1,
                maxiter=20,
                M=linear.preconditioner(self.stiffness, "diag"),
            )
            self.solver = eigen.LOBPCG(
                self.stiffness, modes, mass=self.mass, preconditioner=preconditioner, maxiter=100, tol=1e-8
            ).prepare()

    def step(self):
        """Solve using the current matrices and return device eigenpairs."""
        return self.solver()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--modes", type=int, default=4)
    args = parser.parse_args()
    example = Example(args.resolution, args.modes, args.device)
    result = example.step().check()
    print("Eigenvalues:", result.eigenvalues.numpy())
    print("Maximum relative residual:", result.relative_residuals.numpy().max())
