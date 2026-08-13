# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Gradient of a summed element loss: forward jet vs reverse-mode autodiff
###########################################################################

"""Minimal example: a first-order forward jet beats reverse mode on a summed loss.

The loss sums a local energy over many overlapping elements -- the shape of every
mesh objective (spring, distortion, elasticity)::

    loss(x) = sum_e  0.5 (||x_i - x_j|| - r)^2        # e = spring (i, j)

We want the vertex gradient dloss/dx (one vec3 per vertex), assembled over a chain
where every interior vertex is shared by two springs. Two ways:

    reverse : one scalar-energy launch under wp.Tape, then tape.backward with a
              unit seed on every spring accumulates dloss/dx into x.grad. This is
              the idiomatic wp.grad path -- O(C) work, but a separate backward
              launch plus tape state.

    jet     : one width-6 forward-jet launch. Each spring seeds its two endpoints
              (2 nodes x 3 coords = 6 variables) as a jet; the value's 6 coeffs ARE
              the spring's local gradient, scattered (atomic_add) into the vertex
              gradient. One launch, no tape, no adjoint storage.

Both assemble the SAME global gradient -- the atomics accumulate shared-vertex
contributions exactly as reverse mode does. The jet launch does forward AND
gradient together; reverse is timed on the backward pass ALONE (its forward is not
even counted), so the comparison is conservative and the jet still wins on GPU.

    uv run warp/examples/benchmarks/benchmark_jet_gradient_mesh.py --device cuda:0
"""

import argparse
import time
from typing import Any

import numpy as np

import warp as wp

J = wp.JetSpace(6)  # 2 nodes x 3 coords = 6 local variables per spring


@wp.func
def spring_energy(a: Any, b: Any, r: float):
    # One generic definition: specializes on plain vec3 (reverse) and jet vec3 (forward).
    d = a - b
    s = wp.length(d) - r
    return 0.5 * s * s


@wp.kernel
def spring_energy_scalar(x: wp.array[wp.vec3], edge: wp.array[wp.vec2i], r: float, e: wp.array[wp.float32]):
    i = wp.tid()
    ij = edge[i]
    e[i] = spring_energy(x[ij[0]], x[ij[1]], r)


@wp.kernel
def spring_grad_jet(x: wp.array[wp.vec3], edge: wp.array[wp.vec2i], r: float, g: wp.array[wp.vec3]):
    i = wp.tid()
    ij = edge[i]
    a = J.seed_vec3(x[ij[0]], 0, 1, 2)
    b = J.seed_vec3(x[ij[1]], 3, 4, 5)
    c = spring_energy(a, b, r).coeff  # 6 coefficients = local gradient
    wp.atomic_add(g, ij[0], wp.vec3(c[0], c[1], c[2]))
    wp.atomic_add(g, ij[1], wp.vec3(c[3], c[4], c[5]))


def chain(m, seed=0):
    """A chain of m springs over m+1 vertices, laid on a line and perturbed."""
    n = m + 1
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = np.arange(n)  # unit spacing -> rest length 1
    pos += 0.2 * rng.standard_normal((n, 3)).astype(np.float32)
    edges = np.stack([np.arange(m), np.arange(1, n)], axis=1).astype(np.int32)
    return pos, edges, 1.0


class Reverse:
    def __init__(self, pos, edges, r, device):
        self.n, self.m, self.r, self.device = pos.shape[0], edges.shape[0], r, device
        self.x = wp.array(pos, dtype=wp.vec3, device=device, requires_grad=True)
        self.edge = wp.array(edges, dtype=wp.vec2i, device=device)
        self.e = wp.zeros(self.m, dtype=wp.float32, device=device, requires_grad=True)
        self.ones = wp.ones(self.m, dtype=wp.float32, device=device)
        self.tape = wp.Tape()
        with self.tape:
            wp.launch(spring_energy_scalar, dim=self.m, inputs=[self.x, self.edge, self.r], outputs=[self.e])

    def run(self):
        self.x.grad.zero_()
        self.tape.backward(grads={self.e: self.ones})

    def grad(self):
        return self.x.grad.numpy()


class Jet:
    def __init__(self, pos, edges, r, device):
        self.n, self.m, self.r, self.device = pos.shape[0], edges.shape[0], r, device
        self.x = wp.array(pos, dtype=wp.vec3, device=device)
        self.edge = wp.array(edges, dtype=wp.vec2i, device=device)
        self.g = wp.zeros(self.n, dtype=wp.vec3, device=device)

    def run(self):
        self.g.zero_()
        wp.launch(spring_grad_jet, dim=self.m, inputs=[self.x, self.edge, self.r], outputs=[self.g])

    def grad(self):
        return self.g.numpy()


def timeit(fn, device, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best


def time_strategy(s, device, reps, use_graph):
    s.run()
    wp.synchronize_device(device)
    if not use_graph:
        return timeit(s.run, device, reps)
    with wp.ScopedCapture(device) as cap:
        s.run()
    return timeit(lambda: wp.capture_launch(cap.graph), device, reps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--m", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay.")
    args = parser.parse_args()

    wp.init()
    device = args.device
    mode = "graph replay (warm)" if args.graph else "per-launch"
    print(f"device: {wp.get_device(device)}   reps: {args.reps}   timing: {mode}\n")

    header = f"{'springs':>10} {'jet':>10} {'reverse':>10} {'jet/rev':>9} {'|jet-rev|':>11}"
    print(header)
    print("-" * len(header))
    print("(jet does forward+grad in one launch; reverse is timed on backward alone)")

    for m in args.m:
        pos, edges, r = chain(m)
        jet = Jet(pos, edges, r, device)
        rev = Reverse(pos, edges, r, device)

        jet.run()
        rev.run()
        wp.synchronize_device(device)
        err = np.abs(jet.grad() - rev.grad()).max()

        tj = time_strategy(jet, device, args.reps, args.graph)
        tr = time_strategy(rev, device, args.reps, args.graph)
        print(f"{m:>10} {tj * 1e3:>10.3f} {tr * 1e3:>10.3f} {tj / tr:>8.2f}x {err:>11.1e}")


if __name__ == "__main__":
    main()
