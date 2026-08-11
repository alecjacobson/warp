# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Catenary
#
# Finds the equilibrium of an elastic hanging chain (a catenary) with Newton's
# method, assembling the sparse gradient and Hessian with wp.indexed_sum.
#
# The energy is the sum of two terms over different stencils:
#
#   springs (per edge):    0.5 * k * ( || p_i - p_j || - r )^2
#   gravity (per vertex):  m * g * y_i
#
# The spring rest length r is a *non-differentiable per-edge parameter* passed
# through a second index array (edge_ids -> rest_lengths). Gravity is a 1-node
# summand; being linear it has no Hessian. The total energy is the composition
# `spring + gravity` of two wp.indexed_sum terms, and value.gradient / value.hessian
# are the sums of the terms (the gravity term contributes only to the gradient).
#
# The natural length is shorter than the endpoint span, so the springs stay in
# tension (stretched) and the assembled Hessian stays positive definite; Newton
#
#   H dx = -g,   pinned endpoints projected out (warp.fem.project_linear_system),
#
# solved matrix-free with warp.optim.linear.cg, converges to the sagging chain.
#
# Demonstrates the wp.indexed_sum MVP with a non-differentiable per-element
# parameter and composition of two terms. See design/sparse-hessians.md.
#
###########################################################################

import argparse

import numpy as np

import warp as wp
import warp.fem as fem
from warp.optim.linear import cg
from warp.sparse import bsr_from_triplets

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

NUM_POINTS = 24
SPAN = 2.0  # horizontal distance between the pinned endpoints
STIFFNESS = 1.0
MASS_G = 0.004  # gravitational weight per node
# Natural spacing is half the taut spacing, so every spring is pre-stretched.
REST_SPACING = 0.5 * SPAN / (NUM_POINTS - 1)


@wp.summand
def spring_energy(p0: wp.vec3, p1: wp.vec3, r: float) -> float:
    return 0.5 * STIFFNESS * (wp.length(p0 - p1) - r) ** 2.0


@wp.summand_grad(spring_energy)
def spring_gradient(p0: wp.vec3, p1: wp.vec3, r: float):
    d = p0 - p1
    n = d / wp.length(d)
    g = STIFFNESS * (wp.length(d) - r) * n
    return {0: g, 1: -g}


@wp.summand_hessian(spring_energy)
def spring_hessian(p0: wp.vec3, p1: wp.vec3, r: float):
    d = p0 - p1
    l = wp.length(d)
    n = d / l
    ident = wp.identity(n=3, dtype=float)
    h = STIFFNESS * ((1.0 - r / l) * ident + (r / l) * wp.outer(n, n))
    return {(0, 0): h, (0, 1): -h, (1, 1): h}  # upper triangle only


@wp.summand
def gravity_energy(p0: wp.vec3) -> float:
    return MASS_G * p0[1]


@wp.summand_grad(gravity_energy)
def gravity_gradient(p0: wp.vec3):
    return {0: wp.vec3(0.0, MASS_G, 0.0)}


# Gravity is linear, so no @wp.summand_hessian: its Hessian is zero and the
# composite skips it.


@wp.kernel
def _apply_step(dx: wp.array(dtype=wp.vec3), positions: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    positions[i] = positions[i] + dx[i]


def _endpoint_projector(num_points, device):
    pinned = np.array([0, num_points - 1], dtype=np.int32)
    blocks = np.broadcast_to(np.eye(3), (pinned.size, 3, 3)).astype(np.float32).copy()
    return bsr_from_triplets(
        num_points,
        num_points,
        wp.array(pinned, dtype=int, device=device),
        wp.array(pinned, dtype=int, device=device),
        wp.array(blocks, dtype=float, device=device),
    )


class Example:
    def __init__(self, device=None):
        self.device = wp.get_device(device)
        self.num_points = NUM_POINTS

        with wp.ScopedDevice(self.device):
            # Start taut and level between the pinned endpoints; gravity sags it.
            x = np.linspace(-0.5 * SPAN, 0.5 * SPAN, NUM_POINTS, dtype=np.float32)
            pts = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

            ne = NUM_POINTS - 1
            edges = np.stack([np.arange(ne), np.arange(1, NUM_POINTS)], axis=1).astype(np.int32)

            self.positions = wp.array(pts, dtype=wp.vec3)
            self.edges = wp.array(edges, dtype=wp.vec2i)
            self.edge_ids = wp.array(np.arange(ne, dtype=np.int32), dtype=int)
            self.rest_lengths = wp.array(np.full(ne, REST_SPACING, dtype=np.float32), dtype=float)
            # The rest length is a constant input, not a differentiation variable.
            self.rest_lengths.hessian_variable = False
            self.vertex_ids = wp.array(np.arange(NUM_POINTS, dtype=np.int32), dtype=int)

            self.springs = wp.indexed_sum(spring_energy, (self.edges, self.edge_ids))
            self.gravity = wp.indexed_sum(gravity_energy, self.vertex_ids)
            self.projector = _endpoint_projector(NUM_POINTS, self.device)

    def _total(self):
        return self.springs(self.positions, self.rest_lengths) + self.gravity(self.positions)

    def energy(self):
        return self._total().value

    def step(self):
        with wp.ScopedDevice(self.device):
            total = self._total()
            rhs = total.vjp(self.positions, seed=-1.0)  # -g
            h = total.hessian[self.positions, self.positions]

            fem.project_linear_system(h, rhs, self.projector, normalize_projector=False)
            dx = wp.zeros_like(rhs)
            cg(h, rhs, dx, tol=1e-10)

            wp.launch(_apply_step, dim=self.num_points, inputs=[dx, self.positions])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-steps", type=int, default=16, help="Number of Newton iterations.")
    parser.add_argument("--headless", action="store_true", help="Do not display a plot.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for harness compatibility.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(device=args.device)

        initial = example.positions.numpy().copy()
        print(f"iter 0: energy {example.energy():.6e}")
        for it in range(args.num_steps):
            example.step()
            print(f"iter {it + 1}: energy {example.energy():.6e}")
        final = example.positions.numpy().copy()
        print(f"sag depth: {-float(final[:, 1].min()):.4f}")

        if not args.headless:
            if not MATPLOTLIB_AVAILABLE:
                print("matplotlib not available; skipping plot.")
                return
            plt.plot(initial[:, 0], initial[:, 1], "o-", label="initial", alpha=0.4)
            plt.plot(final[:, 0], final[:, 1], "o-", label="catenary")
            plt.axis("equal")
            plt.legend()
            plt.title("Catenary hanging chain (Newton)")
            plt.show()


if __name__ == "__main__":
    main()
