# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Catenary (finite-difference backend)
#
# Same elastic hanging chain as example_catenary.py, but the gradient and
# Hessian are obtained by finite-differencing the energy (backend="fd")
# instead of hand-written @wp.summand_grad / @wp.summand_hessian rules.
#
# The only summand code is the two energies; wp.indexed_sum(..., backend="fd")
# does the rest. This trades accuracy/speed for not having to derive and write
# the derivatives. See design/sparse-hessians.md.
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
SPAN = 2.0
STIFFNESS = 1.0
MASS_G = 0.004
REST_SPACING = 0.5 * SPAN / (NUM_POINTS - 1)


# Only the energies are needed -- no derivative rules.
@wp.summand
def spring_energy(p0: wp.vec3, p1: wp.vec3, r: float) -> float:
    return 0.5 * STIFFNESS * (wp.length(p0 - p1) - r) ** 2.0


@wp.summand
def gravity_energy(p0: wp.vec3) -> float:
    return MASS_G * p0[1]


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
            x = np.linspace(-0.5 * SPAN, 0.5 * SPAN, NUM_POINTS, dtype=np.float32)
            pts = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

            ne = NUM_POINTS - 1
            edges = np.stack([np.arange(ne), np.arange(1, NUM_POINTS)], axis=1).astype(np.int32)

            self.positions = wp.array(pts, dtype=wp.vec3)
            self.edges = wp.array(edges, dtype=wp.vec2i)
            self.edge_ids = wp.array(np.arange(ne, dtype=np.int32), dtype=int)
            self.rest_lengths = wp.array(np.full(ne, REST_SPACING, dtype=np.float32), dtype=float)
            self.rest_lengths.hessian_variable = False
            self.vertex_ids = wp.array(np.arange(NUM_POINTS, dtype=np.int32), dtype=int)

            # Finite-difference backend: no @wp.summand_grad / @wp.summand_hessian.
            self.springs = wp.indexed_sum(spring_energy, (self.edges, self.edge_ids), backend="fd")
            self.gravity = wp.indexed_sum(gravity_energy, self.vertex_ids, backend="fd")
            self.projector = _endpoint_projector(NUM_POINTS, self.device)

    def _total(self):
        return self.springs(self.positions, self.rest_lengths) + self.gravity(self.positions)

    def energy(self):
        return self._total().value

    def residual_norm(self):
        g = self._total().gradient[self.positions].numpy()
        g[0] = 0.0
        g[-1] = 0.0
        return float(np.linalg.norm(g))

    def step(self):
        with wp.ScopedDevice(self.device):
            total = self._total()
            rhs = total.vjp(self.positions, seed=-1.0)
            h = total.hessian[self.positions, self.positions]

            fem.project_linear_system(h, rhs, self.projector, normalize_projector=False)
            dx = wp.zeros_like(rhs)
            cg(h, rhs, dx, tol=1e-10)

            wp.launch(_apply_step, dim=self.num_points, inputs=[dx, self.positions])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-steps", type=int, default=8, help="Number of Newton iterations.")
    parser.add_argument("--headless", action="store_true", help="Do not display a plot.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for harness compatibility.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(device=args.device)

        initial = example.positions.numpy().copy()
        for it in range(args.num_steps):
            example.step()
            print(f"iter {it + 1}: energy = {example.energy():.3e}, |grad| = {example.residual_norm():.3e}")
        final = example.positions.numpy().copy()

        if not args.headless:
            if not MATPLOTLIB_AVAILABLE:
                print("matplotlib not available; skipping plot.")
                return
            plt.plot(initial[:, 0], initial[:, 1], "o-", label="initial", alpha=0.4)
            plt.plot(final[:, 0], final[:, 1], "o-", label="catenary (fd)")
            plt.axis("equal")
            plt.legend()
            plt.title("Catenary hanging chain (Newton, finite-difference backend)")
            plt.show()


if __name__ == "__main__":
    main()
