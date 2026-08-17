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

    forward-2 a single forward pass of a *second-order* jet carries value,
              gradient, and the full k x k Hessian, so it reads the Hessian off
              directly -- no reverse, no tape.

                  one forward, each op O(k^2)  ->  O(k^2 C), state O(k^2)/node

Only the width-1 path keeps derivative state per intermediate at O(1), so its
register pressure does not grow with k. forward-2 trades all reverse/tape
overhead for O(k^2) forward state, so it is a small-k play: strong where launch
overhead dominates, infeasible once k^2 register pressure spills.

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

# forward-2 carries an O(k^2) Hessian per thread. Its kernel unrolls the whole
# chain over mat_k-by-k arithmetic, and nvcc register allocation on that is
# super-linear: ~1 s at k=2, ~2 s at k=4, ~60 s at k=8, impractical beyond. So
# forward-2 is run only for small k (shown as "-" above the cap); this is a
# compile limit, not a runtime one. Override with --forward2-max-k.
FORWARD2_MAX_K = 8

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
    """Return (Jk, width-k gradient kernel, width-1 directional kernel, forward-2 kernel)."""
    Jk = wp.JetSpace(k, dtype=dtype)
    J1 = wp.JetSpace(1, dtype=dtype)
    J2 = wp.JetSpace2(k, dtype=dtype)

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

    @wp.kernel
    def hess_forward(z: wp.array2d[dtype], out: wp.array3d[dtype]):
        # One forward pass of a second-order jet writes the whole k x k Hessian;
        # no reverse, no tape. State per intermediate is O(k^2).
        i = wp.tid()

        acc = J2.constant(dtype(0.0))
        prev = J2.seed(z[i, 0], 0)

        for b in range(wp.static(outer)):
            for t in range(wp.static(inner)):
                j = b * wp.static(inner) + t

                if j >= 1 and j < wp.static(k):
                    cur = J2.seed(z[i, j], j)
                    acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
                    prev = cur

        h = (acc + wp.exp(prev)).hess
        for p in range(wp.static(k)):
            for q in range(wp.static(k)):
                out[i, p, q] = h[p, q]

    return Jk, grad_wide, directional, hess_forward


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


class Forward2:
    """One forward pass of a second-order jet: the Hessian falls out directly."""

    label = "forward-2"

    def __init__(self, k, Jk, kernel, z_np, device):
        self.k = k
        self.kernel = kernel
        self.device = device
        self.n = z_np.shape[0]

        # No requires_grad, no tape: the Hessian is a forward output.
        self.z = wp.array(z_np, dtype=float, device=device)
        self.hessian = wp.zeros((self.n, k, k), dtype=float, device=device)

    def run(self):
        wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.hessian], device=self.device)


def assemble(strategy):
    """Run once and read the result back. The copy is outside any timed region."""
    strategy.run()
    return strategy.hessian.numpy()


# ---------------------------------------------------------------------------


def check(k, Jk, kw, kd, kf, device, n=16, seed=0):
    """Gate: strategies must agree with each other and with float64 FD.

    Returns (max cross-strategy disagreement, max error vs FD, forward-2 usable).
    forward-2 is optional: if its wide second-order jet fails to build at this k,
    the other two are still gated and timed.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)

    hw = assemble(WidthK(k, Jk, kw, z, device))
    hh = assemble(Width1(k, Jk, kd, z, device))
    ref = hessian_fd(z.astype(np.float64), k)

    cross = np.abs(hw - hh).max()
    fd = max(np.abs(hw - ref).max(), np.abs(hh - ref).max())

    if k > FORWARD2_MAX_K:
        f2_ok = False
    else:
        try:
            hf = assemble(Forward2(k, Jk, kf, z, device))
            cross = max(cross, np.abs(hw - hf).max())
            fd = max(fd, np.abs(hf - ref).max())
            f2_ok = True
        except Exception:
            f2_ok = False

    return cross, fd, f2_ok


def timeit(fn, device, reps):
    best = float("inf")

    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)

    return best


def time_strategy(strategy, device, reps, use_graph):
    """Warm up, then time strategy.run() -- as-is, or replayed from a captured graph.

    Graph capture records the strategy's launches once (warm) and replays them
    with a single graph launch, so per-launch and Python-driver overhead is
    amortized. This isolates the strategies' intrinsic device cost from the
    launch bookkeeping that dominates the small-m regime.
    """
    strategy.run()  # warm up: compile and allocate before capture/timing
    wp.synchronize_device(device)

    if not use_graph:
        return timeit(strategy.run, device, reps)

    with wp.ScopedCapture(device) as cap:
        strategy.run()
    graph = cap.graph

    return timeit(lambda: wp.capture_launch(graph), device, reps)


def estimate_bytes(m, k):
    """Rough peak footprint: z, z.grad, g, g.grad, seed, plus the m x k x k Hessian."""
    return (5 * m * k + m * k * k) * 4


def run(k_values, m_values, device, reps, tol, max_bytes, skip_check, use_graph):
    print(f"device: {wp.get_device(device)}")
    mode = "graph replay (warm)" if use_graph else "per-launch"
    print(f"reps: {reps}   correctness tol: {tol}   memory cap: {max_bytes / 2**30:.1f} GiB   timing: {mode}\n")

    header = (
        f"{'k':>5} {'m':>9} {'width-k':>11} {'width-1':>11} {'fwd-2':>11} {'wk/w1':>7} {'f2/w1':>7} {'|max-fd|':>10}"
    )
    print(header)
    print("-" * len(header))
    print("(speedups > 1 mean width-1 is faster than that strategy)")

    results = []

    for k in k_values:
        Jk, kw, kd, kf = build(k)

        if skip_check:
            cross = fd = float("nan")
            f2_ok = True
        else:
            cross, fd, f2_ok = check(k, Jk, kw, kd, kf, device)

            if not (cross < tol and fd < tol):
                print(f"{k:>5} {'-':>9}   FAIL  cross={cross:.2e} fd={fd:.2e}")
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

            tw = time_strategy(wide, device, reps, use_graph)
            th = time_strategy(hvp, device, reps, use_graph)

            # forward-2 is optional: skip it if the wide second-order jet is
            # infeasible (register blow-up) or too large to allocate at this k.
            tf = float("nan")
            if f2_ok:
                try:
                    fwd2 = Forward2(k, Jk, kf, z, device)
                    tf = time_strategy(fwd2, device, reps, use_graph)
                except Exception:
                    tf = float("nan")

            if tf == tf:  # not NaN
                f2_time = f"{tf * 1e3:>11.2f}"
                f2_speed = f"{tf / th:>6.2f}x"
            else:
                f2_time = f"{'-':>11}"
                f2_speed = f"{'-':>7}"

            print(
                f"{k:>5} {m:>9} {tw * 1e3:>11.2f} {th * 1e3:>11.2f} {f2_time} {tw / th:>6.2f}x {f2_speed} {fd:>10.1e}"
            )
            results.append((k, m, tw, th, tf))

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
    parser.add_argument("--forward2-max-k", type=int, default=None, help="Cap k for the forward-2 strategy.")
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay instead of per-launch.")
    args = parser.parse_args()

    if args.forward2_max_k is not None:
        global FORWARD2_MAX_K
        FORWARD2_MAX_K = args.forward2_max_k

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
            args.graph,
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
            args.graph,
        )


if __name__ == "__main__":
    main()
