# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmarks for local-gradient assembly: forward jet vs reverse-mode autodiff
###########################################################################

"""Gradient of a scalar energy over k variables, two ways, materializing m x k.

    jet-fwd   one width-k forward-mode jet pass: the value's k coefficients
              are grad g. One launch, no tape.  Work O(kC).

    reverse   Warp's built-in reverse-mode autodiff: one scalar-energy forward
              launch, then tape.backward seeds the output and sweeps once to
              z.grad = grad g.  Work O(C), but a second launch plus tape state.

Reverse mode is the asymptotically cheap way to get a scalar's gradient (O(C)
vs O(kC)), so it should win as k grows. The question is the small-k / GPU
regime: does the forward jet's single tape-free launch beat reverse's
forward+reverse+tape? --graph captures each strategy warm to separate intrinsic
cost from launch overhead.

    uv run warp/examples/benchmarks/benchmark_jet_gradient.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_jet_gradient.py --device cuda:0 --graph
"""

import argparse
import time

import numpy as np

import warp as wp


def strip_mine(k):
    inner = 1
    while inner * inner < k:
        inner += 1
    return (k + inner - 1) // inner, inner


def build(k, dtype=wp.float32):
    Jk = wp.JetSpace(k, dtype=dtype)
    outer, inner = strip_mine(k)

    @wp.kernel
    def jet_grad(z: wp.array2d[dtype], out: wp.array[Jk.coeff]):
        # Forward jet: the value's k coefficients ARE grad g. One launch, no tape.
        i = wp.tid()
        acc = Jk.constant(dtype(0.0))
        prev = Jk.seed(z[i, 0], 0)
        for b in range(wp.static(outer)):
            for t in range(wp.static(inner)):
                j = b * wp.static(inner) + t
                if j >= 1 and j < wp.static(k):
                    cur = Jk.seed(z[i, j], j)
                    acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
                    prev = cur
        out[i] = (acc + wp.exp(prev)).coeff

    @wp.kernel
    def energy(z: wp.array2d[dtype], e: wp.array[dtype]):
        # Plain scalar energy; Warp reverse-mode differentiates it.
        i = wp.tid()
        acc = dtype(0.0)
        prev = z[i, 0]
        for b in range(wp.static(outer)):
            for t in range(wp.static(inner)):
                j = b * wp.static(inner) + t
                if j >= 1 and j < wp.static(k):
                    cur = z[i, j]
                    acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
                    prev = cur
        e[i] = acc + wp.exp(prev)

    return Jk, jet_grad, energy


def energy_np(z, k):
    acc = np.zeros(z.shape[0])
    prev = z[:, 0]
    for j in range(1, k):
        acc = acc + np.sin(prev * z[:, j]) + 0.1 * prev**3
        prev = z[:, j]
    return acc + np.exp(prev)


def grad_fd(z, k, h=1.0e-4):
    out = np.empty((z.shape[0], k))
    for p in range(k):
        ep = np.zeros_like(z)
        ep[:, p] = h
        out[:, p] = (energy_np(z + ep, k) - energy_np(z - ep, k)) / (2.0 * h)
    return out


class JetFwd:
    def __init__(self, k, Jk, kernel, z_np, device):
        self.k, self.kernel, self.device, self.n = k, kernel, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=float, device=device)
        self.out = wp.zeros(self.n, dtype=Jk.coeff, device=device)

    def run(self):
        wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.out], device=self.device)

    def grad(self):
        return self.out.numpy().reshape(self.n, self.k)


class Reverse:
    def __init__(self, k, kernel, z_np, device):
        self.k, self.kernel, self.device, self.n = k, kernel, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
        self.e = wp.zeros(self.n, dtype=float, device=device, requires_grad=True)
        self.ones = wp.ones(self.n, dtype=float, device=device)
        self.tape = wp.Tape()
        with self.tape:
            wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.e], device=self.device)

    def run(self):
        self.z.grad.zero_()
        self.tape.backward(grads={self.e: self.ones})

    def grad(self):
        return self.z.grad.numpy().reshape(self.n, self.k)


def timeit(fn, device, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best


def time_strategy(strategy, device, reps, use_graph):
    strategy.run()
    wp.synchronize_device(device)
    if not use_graph:
        return timeit(strategy.run, device, reps)
    with wp.ScopedCapture(device) as cap:
        strategy.run()
    graph = cap.graph
    return timeit(lambda: wp.capture_launch(graph), device, reps)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__.split("\n")[0]
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 4, 6, 8, 12, 24, 48])
    parser.add_argument("--m", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay.")
    args = parser.parse_args()

    wp.init()
    device = args.device
    mode = "graph replay (warm)" if args.graph else "per-launch"
    print(f"device: {wp.get_device(device)}   m: {args.m}   reps: {args.reps}   timing: {mode}\n")

    header = f"{'k':>5} {'jet-fwd':>11} {'reverse':>11} {'jet/rev':>8} {'|max-fd|':>10}"
    print(header)
    print("-" * len(header))
    print("(jet/rev < 1 means the forward jet is faster)")

    for k in args.k:
        Jk, kj, ke = build(k)
        rng = np.random.default_rng(0)
        z = rng.uniform(-1.0, 1.0, size=(args.m, k)).astype(np.float32)

        jet = JetFwd(k, Jk, kj, z, device)
        rev = Reverse(k, ke, z, device)

        # Correctness gate on a small sample.
        zc = rng.uniform(-1.0, 1.0, size=(16, k)).astype(np.float32)
        jc = JetFwd(k, Jk, kj, zc, device)
        jc.run()
        rc = Reverse(k, ke, zc, device)
        rc.run()
        wp.synchronize_device(device)
        ref = grad_fd(zc.astype(np.float64), k)
        err = max(np.abs(jc.grad() - ref).max(), np.abs(rc.grad() - ref).max())

        tj = time_strategy(jet, device, args.reps, args.graph)
        tr = time_strategy(rev, device, args.reps, args.graph)
        print(f"{k:>5} {tj * 1e3:>11.3f} {tr * 1e3:>11.3f} {tj / tr:>7.2f}x {err:>10.1e}")


if __name__ == "__main__":
    main()
