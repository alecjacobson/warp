# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmarks for local-Hessian assembly with wp.JetSpace
###########################################################################

"""Benchmark local-Hessian assembly: width-k reverse-over-forward vs width-1 HVPs.

Both strategies materialize the same m x k x k result.

    width-k   one k-wide forward pass yields all of grad g, then k reverse
              sweeps through that widened program pull out its k x k Jacobian
              one row at a time.

                  forward ~ O(kC),  k reverses ~ O(kC) each  ->  O(k^2 C)

    width-1   for each basis direction e_j, a constant-width forward pass
              yields the scalar Dg[e_j] = grad g . e_j, and one reverse sweep
              over that scalar yields grad_z (grad g . e_j) = H e_j, i.e.
              column j of H.

                  forward ~ O(C),  reverse ~ O(C),  k of them  ->  O(kC)

Only the width-1 path keeps derivative state per intermediate at O(1), so its
register pressure does not grow with k.

That operation count predicts width-1, and on a CPU it wins by 5.3x at k=8 and
10.3x at k=16 (m=50000). On an L40 the ranking reverses: width-k is 3.0x faster
at k=8 and 2.7x at k=16 (m=500000). A GPU running many local terms is not
compute-bound here -- width-1 issues 2k launches to width-k's k+1, and re-reads
z and recomputes the primal once per direction, while width-k folds that into
one wide forward pass and has registers to spare for the k tangents. Which is
why this is a benchmark rather than a recommendation.

Usage:

    uv run warp/examples/benchmarks/benchmark_jet_hessian.py                      # both sweeps, auto device
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --sweep m --m 1000000
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --k 2 4 8 16 32 64 100

Correctness is gated before every timing run: a config whose two strategies
disagree, or which disagrees with a float64 finite-difference reference, is
reported as FAIL and not timed. See the strip-mining note below -- getting the
loop structure wrong yields silently incorrect derivatives, not an error.

On a GPU, pick m large enough to fill the device. At m=20000 an L40 is idle
between launches and the numbers measure launch and allocation overhead rather
than the kernels; prefer the m sweep, or --m 1000000.
"""

import argparse
import time

import numpy as np

import warp as wp

# ---------------------------------------------------------------------------
# A local energy over k variables, touching all of them:
#
#     g(z) = sum_j [ sin(z_j * z_{j+1}) + 0.1 * z_j^3 ] + exp(z_{k-1})
#
# The chain is strip-mined into two loops of about sqrt(k) each rather than one
# loop of k. Warp applies max_unroll (default 16) per loop, not to the total
# trip count, so a 10 x 10 pair unrolls at k=100 where a flat loop of 100 does
# not -- 11405 generated lines versus 869 for the rolled form.
#
# Unrolling is not a nicety here. The chain carries a seeded jet (`prev`)
# across iterations, and reverse mode through a *dynamic* loop with
# struct-valued loop-carried variables returns WRONG gradients with no
# diagnostic. Strip-mining keeps the whole chain straight-line for every k up
# to max_unroll^2 = 256 without touching module options. The correctness gate
# below catches it if that ever stops holding.
#
# This is a property of this harness, not of wp.JetSpace: the jet types are
# correct in both passes well past this threshold.
# ---------------------------------------------------------------------------


def strip_mine(k):
    """Split k into (outer, inner) factors, each small enough to unroll."""
    inner = 1
    while inner * inner < k:
        inner += 1

    outer = (k + inner - 1) // inner
    return outer, inner


def build(k, dtype=wp.float32):
    """Return (Jk, width-k gradient kernel, width-1 directional kernel)."""
    Jk = wp.JetSpace(k, dtype=dtype)
    J1 = wp.JetSpace(1, dtype=dtype)

    outer, inner = strip_mine(k)

    @wp.kernel
    def grad_wide(z: wp.array2d[dtype], out: wp.array[Jk.coeff]):
        i = wp.tid()

        acc = Jk.constant(dtype(0.0))
        prev = Jk.seed(z[i, 0], 0)

        for b in range(wp.static(outer)):
            for t in range(wp.static(inner)):
                j = b * wp.static(inner) + t

                # j == 0 seeds `prev` above; j >= k is padding when k is not a
                # product of the two factors.
                if j >= 1 and j < wp.static(k):
                    cur = Jk.seed(z[i, j], j)
                    acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
                    prev = cur

        out[i] = (acc + wp.exp(prev)).coeff

    @wp.kernel
    def directional(z: wp.array2d[dtype], v: wp.array[dtype], dv: wp.array[dtype]):
        i = wp.tid()

        acc = J1.constant(dtype(0.0))
        prev = J1.with_coeff(z[i, 0], J1.coeff(v[0]))

        for b in range(wp.static(outer)):
            for t in range(wp.static(inner)):
                j = b * wp.static(inner) + t

                if j >= 1 and j < wp.static(k):
                    cur = J1.with_coeff(z[i, j], J1.coeff(v[j]))
                    acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
                    prev = cur

        dv[i] = (acc + wp.exp(prev)).coeff[0]

    return Jk, grad_wide, directional


def energy_np(z, k):
    """The same energy in float64 NumPy, for the reference Hessian."""
    acc = np.zeros(z.shape[0])
    prev = z[:, 0]

    for j in range(1, k):
        cur = z[:, j]
        acc = acc + np.sin(prev * cur) + 0.1 * prev**3
        prev = cur

    return acc + np.exp(prev)


def hessian_fd(z, k, h=1.0e-4):
    """float64 second differences: independent of every line of Warp code."""
    out = np.empty((z.shape[0], k, k))

    for p in range(k):
        for q in range(k):
            ep = np.zeros_like(z)
            ep[:, p] = h

            eq = np.zeros_like(z)
            eq[:, q] = h

            out[:, p, q] = (
                energy_np(z + ep + eq, k)
                - energy_np(z + ep - eq, k)
                - energy_np(z - ep + eq, k)
                + energy_np(z - ep - eq, k)
            ) / (4.0 * h * h)

    return out


# ---------------------------------------------------------------------------
# The two strategies
#
# Everything stays device-resident. Host allocations and device-to-host copies
# are hoisted into the constructors, so run() times the assembly rather than
# the Python driver loop. On a GPU the difference dominates: at m=20000 a
# .numpy() per direction costs far more than the kernels it is reading back.
# ---------------------------------------------------------------------------


@wp.kernel
def _scatter_row(src: wp.array2d[float], row: int, dst: wp.array3d[float]):
    i, b = wp.tid()
    dst[i, row, b] = src[i, b]


@wp.kernel
def _scatter_col(src: wp.array2d[float], col: int, dst: wp.array3d[float]):
    i, b = wp.tid()
    dst[i, b, col] = src[i, b]


class WidthK:
    """One k-wide forward pass, then k reverse sweeps through it."""

    label = "width-k"

    def __init__(self, k, Jk, kernel, z_np, device):
        self.k = k
        self.kernel = kernel
        self.device = device
        self.n = z_np.shape[0]

        self.z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
        self.g = wp.zeros(self.n, dtype=Jk.coeff, device=device, requires_grad=True)
        self.hessian = wp.zeros((self.n, k, k), dtype=float, device=device)

        # One seed per gradient component, uploaded once.
        self.seeds = []
        for row in range(k):
            s = np.zeros((self.n, k), dtype=np.float32)
            s[:, row] = 1.0
            self.seeds.append(wp.array(s, dtype=Jk.coeff, device=device))

    def run(self):
        tape = wp.Tape()
        with tape:
            wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.g], device=self.device)

        for row in range(self.k):
            tape.backward(grads={self.g: self.seeds[row]})

            # z.grad[i,b] = H_i[row,b]
            wp.launch(
                _scatter_row,
                dim=(self.n, self.k),
                inputs=[self.z.grad, row, self.hessian],
                device=self.device,
            )
            tape.zero()


class Width1:
    """k forward+reverse pairs, each of constant width: one HVP per direction."""

    label = "width-1"

    def __init__(self, k, Jk, kernel, z_np, device):
        self.k = k
        self.kernel = kernel
        self.device = device
        self.n = z_np.shape[0]

        self.z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
        self.dv = wp.zeros(self.n, dtype=float, device=device, requires_grad=True)
        self.ones = wp.ones(self.n, dtype=float, device=device)
        self.hessian = wp.zeros((self.n, k, k), dtype=float, device=device)

        # One basis direction per column, uploaded once.
        self.dirs = []
        for j in range(k):
            e = np.zeros(k, dtype=np.float32)
            e[j] = 1.0
            self.dirs.append(wp.array(e, dtype=float, device=device))

    def run(self):
        for j in range(self.k):
            self.dv.zero_()

            tape = wp.Tape()
            with tape:
                wp.launch(
                    self.kernel,
                    dim=self.n,
                    inputs=[self.z, self.dirs[j]],
                    outputs=[self.dv],
                    device=self.device,
                )

            # grad_z (grad g . e_j) = H e_j = column j
            tape.backward(grads={self.dv: self.ones})

            wp.launch(
                _scatter_col,
                dim=(self.n, self.k),
                inputs=[self.z.grad, j, self.hessian],
                device=self.device,
            )
            tape.zero()


def assemble(strategy):
    """Run once and read the result back. The copy is outside any timed region."""
    strategy.run()
    return strategy.hessian.numpy()


# ---------------------------------------------------------------------------


def check(k, Jk, kw, kd, device, n=16, seed=0):
    """Gate: both strategies must agree with each other and with float64 FD."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)

    hw = assemble(WidthK(k, Jk, kw, z, device))
    hh = assemble(Width1(k, Jk, kd, z, device))
    ref = hessian_fd(z.astype(np.float64), k)

    return (
        np.abs(hw - hh).max(),
        np.abs(hw - ref).max(),
        np.abs(hh - ref).max(),
    )


def timeit(fn, device, reps):
    best = float("inf")

    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)

    return best


def estimate_bytes(m, k):
    """Rough peak footprint of the width-k path: z, z.grad, g, g.grad, seed."""
    return 5 * m * k * 4


def run(k_values, m_values, device, reps, tol, max_bytes, skip_check):
    print(f"device: {wp.get_device(device)}")
    print(f"reps: {reps}   correctness tol: {tol}   memory cap: {max_bytes / 2**30:.1f} GiB\n")

    header = (
        f"{'k':>5} {'m':>9} {'width-k (ms)':>14} {'width-1 (ms)':>14} {'speedup':>9} {'|wk-w1|':>10} {'|w1-fd|':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for k in k_values:
        Jk, kw, kd = build(k)

        if skip_check:
            d_ww = d_wf = d_hf = float("nan")
        else:
            d_ww, d_wf, d_hf = check(k, Jk, kw, kd, device)

            if not (d_ww < tol and d_hf < tol):
                print(f"{k:>5} {'-':>9}   FAIL  |wk-w1|={d_ww:.2e} |wk-fd|={d_wf:.2e} |w1-fd|={d_hf:.2e}")
                continue

        for m in m_values:
            need = estimate_bytes(m, k)
            if need > max_bytes:
                print(f"{k:>5} {m:>9}   skipped, needs ~{need / 2**30:.1f} GiB")
                continue

            rng = np.random.default_rng(0)
            z = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)

            # Buffers and uploads are hoisted out of the timed region.
            wide = WidthK(k, Jk, kw, z, device)
            hvp = Width1(k, Jk, kd, z, device)

            # Warm up: compile and touch every path.
            wide.run()
            hvp.run()
            wp.synchronize_device(device)

            tw = timeit(wide.run, device, reps)
            th = timeit(hvp.run, device, reps)

            print(f"{k:>5} {m:>9} {tw * 1e3:>14.2f} {th * 1e3:>14.2f} {tw / th:>8.2f}x {d_ww:>10.1e} {d_hf:>10.1e}")
            results.append((k, m, tw, th))

    return results


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--sweep", choices=("k", "m", "both"), default="both", help="Which sweep to run.")
    parser.add_argument("--k", type=int, nargs="+", default=None, help="Override the k values.")
    parser.add_argument("--m", type=int, nargs="+", default=None, help="Override the m values.")
    parser.add_argument("--reps", type=int, default=5, help="Timed repetitions; the best is reported.")
    parser.add_argument("--tol", type=float, default=1.0e-3, help="Correctness tolerance.")
    parser.add_argument("--max-gib", type=float, default=8.0, help="Skip configs above this estimate.")
    parser.add_argument("--skip-check", action="store_true", help="Skip the correctness gate (not advised).")
    args = parser.parse_args()

    wp.init()

    device = args.device
    max_bytes = args.max_gib * 2**30

    if args.sweep in ("k", "both"):
        print("=" * 80)
        print("k sweep -- cost of widening the forward program")
        print("=" * 80)
        run(
            args.k or [2, 4, 8, 16, 32, 64, 100],
            args.m or [20_000],
            device,
            args.reps,
            args.tol,
            max_bytes,
            args.skip_check,
        )
        print()

    if args.sweep in ("m", "both"):
        print("=" * 80)
        print("m sweep -- throughput over many local terms")
        print("=" * 80)
        run(
            args.k or [6],
            args.m or [1_000, 10_000, 100_000, 1_000_000],
            device,
            args.reps,
            args.tol,
            max_bytes,
            args.skip_check,
        )


if __name__ == "__main__":
    main()
