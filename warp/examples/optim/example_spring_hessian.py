# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Spring Hessian
#
# Straightens a pinned, wavy chain of zero-rest-length springs with Newton's
# method, using wp.indexed_sum to assemble the sparse gradient and Hessian.
#
#     E(x) = 0.5 * sum_{(i, j) in edges} || x_i - x_j ||^2
#
# Each Newton iteration re-assembles the right-hand side -g = value.vjp(x, -1)
# and the Hessian H = value.hessian[x, x] via wp.indexed_sum and solves
#
#     H dx = -g,   with the pinned endpoints held fixed,
#
# then updates x += dx. The endpoints are constrained by projecting the linear
# system (warp.fem.project_linear_system with an identity projector on the
# endpoint nodes), and the reduced system is solved matrix-free with
# warp.optim.linear.cg -- so the whole step stays sparse and on-device.
#
# Nothing here assumes the energy is quadratic -- it is the generic Newton loop
# -- but because this energy happens to be quadratic, the loop reaches the exact
# minimizer (the straight, uniformly spaced chain) in a single step; later
# iterations leave it unchanged.
#
# Demonstrates the wp.indexed_sum MVP: @wp.summand / @wp.summand_grad /
# @wp.summand_hessian, and value.value / value.gradient[...] / value.hessian[...]
# for a single vec3 variable. See design/sparse-hessians.md.
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

SEGMENT_LENGTH = 1.0
NUM_POINTS = 32


@wp.summand
def spring_energy(p0: wp.vec3, p1: wp.vec3) -> float:
    return 0.5 * wp.length_sq(p0 - p1)


@wp.summand_grad(spring_energy)
def spring_gradient(p0: wp.vec3, p1: wp.vec3):
    d = p0 - p1
    return {0: d, 1: -d}  # {arg_index: dE/d(arg)}


@wp.summand_hessian(spring_energy)
def spring_hessian(p0: wp.vec3, p1: wp.vec3):
    # A zero-rest-length spring's local Hessian is the constant [[I, -I], [-I, I]].
    ident = wp.identity(n=3, dtype=float)
    return {(0, 0): ident, (0, 1): -ident, (1, 1): ident}  # upper triangle only


@wp.kernel
def _apply_step(dx: wp.array(dtype=wp.vec3), positions: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    positions[i] = positions[i] + dx[i]


def _endpoint_projector(num_points, device):
    # Diagonal mat33 projector: identity on the two pinned endpoint nodes, zero
    # elsewhere. project_linear_system uses it to hold those DOFs fixed.
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
            # Uniformly spaced along x, with a transverse sine bump on the
            # interior; the pinned endpoints stay put.
            t = np.linspace(0.0, 1.0, NUM_POINTS, dtype=np.float32)
            x = t * SEGMENT_LENGTH
            y = 0.25 * np.sin(3.0 * np.pi * t)
            y[0] = 0.0
            y[-1] = 0.0
            pts = np.stack([x, y, np.zeros_like(t)], axis=1)

            edges = np.stack([np.arange(NUM_POINTS - 1), np.arange(1, NUM_POINTS)], axis=1).astype(np.int32)

            self.positions = wp.array(pts, dtype=wp.vec3)
            self.edges = wp.array(edges, dtype=wp.vec2i)
            self.total = wp.indexed_sum(spring_energy, self.edges)
            self.projector = _endpoint_projector(NUM_POINTS, self.device)

    def energy(self):
        return self.total(self.positions).value

    def step(self):
        with wp.ScopedDevice(self.device):
            # Re-assemble the sparse system at the current positions. The Newton
            # right-hand side -g is the VJP of the energy seeded with -1.
            value = self.total(self.positions)
            rhs = value.vjp(self.positions, seed=-1.0)
            H = value.hessian[self.positions, self.positions]

            # Hard-pin the endpoints, then solve H dx = -g with CG (matrix-free).
            fem.project_linear_system(H, rhs, self.projector, normalize_projector=False)
            dx = wp.zeros_like(rhs)
            cg(H, rhs, dx, tol=1e-10)

            wp.launch(_apply_step, dim=self.num_points, inputs=[dx, self.positions])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of Newton iterations.")
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

        if not args.headless:
            if not MATPLOTLIB_AVAILABLE:
                print("matplotlib not available; skipping plot.")
                return
            plt.plot(initial[:, 0], initial[:, 1], "o-", label="initial", alpha=0.5)
            plt.plot(final[:, 0], final[:, 1], "o-", label="straightened")
            plt.axis("equal")
            plt.legend()
            plt.title("Spring-Hessian Newton straightening")
            plt.show()


if __name__ == "__main__":
    main()
