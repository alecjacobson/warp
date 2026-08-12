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

Usage:

    uv run warp/examples/benchmarks/benchmark_jet_hessian.py                      # both sweeps, auto device
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --sweep m --m 1000000
    uv run warp/examples/benchmarks/benchmark_jet_hessian.py --k 2 4 8 16 32 64 100

Correctness is gated before every timing run: a config whose two strategies
disagree, or which disagrees with a float64 finite-difference reference, is
reported as FAIL and not timed. See the max_unroll note below -- getting this
wrong yields silently incorrect derivatives, not an error.
"""

import argparse
import time
from functools import partial

import numpy as np

import warp as wp

# The benchmark energy below carries jet-valued accumulators across a k-term
# Python loop. Warp only unrolls loops up to max_unroll (default 16); past that
# the loop stays dynamic, and reverse mode through a dynamic loop with
# struct-valued loop-carried variables produces WRONG gradients with no
# diagnostic. Raise it above the largest k benchmarked.
#
# This is a property of this harness, not of wp.JetSpace: the jet types
# themselves are correct in both passes well past this threshold.
wp.set_module_options({"max_unroll": 256})


# ---------------------------------------------------------------------------
# A local energy over k variables, touching all of them:
#
#     g(z) = sum_j [ sin(z_j * z_{j+1}) + 0.1 * z_j^3 ] + exp(z_{k-1})
# ---------------------------------------------------------------------------


def build(k, dtype=wp.float32):
    """Return (Jk, width-k gradient kernel, width-1 directional kernel)."""
    Jk = wp.JetSpace(k, dtype=dtype)
    J1 = wp.JetSpace(1, dtype=dtype)

    @wp.kernel
    def grad_wide(z: wp.array2d[dtype], out: wp.array[Jk.coeff]):
        i = wp.tid()

        acc = Jk.constant(dtype(0.0))
        prev = Jk.seed(z[i, 0], 0)

        for j in range(1, wp.static(k)):
            cur = Jk.seed(z[i, j], j)
            acc = acc + wp.sin(prev * cur) + dtype(0.1) * (prev * prev * prev)
            prev = cur

        out[i] = (acc + wp.exp(prev)).coeff

    @wp.kernel
    def directional(z: wp.array2d[dtype], v: wp.array[dtype], dv: wp.array[dtype]):
        i = wp.tid()

        acc = J1.constant(dtype(0.0))
        prev = J1.with_coeff(z[i, 0], J1.coeff(v[0]))

        for j in range(1, wp.static(k)):
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
# ---------------------------------------------------------------------------


def hessian_width_k(k, Jk, kernel, z_np, device):
    n = z_np.shape[0]

    z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
    g = wp.zeros(n, dtype=Jk.coeff, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[z], outputs=[g], device=device)

    out = np.empty((n, k, k), dtype=np.float32)

    for row in range(k):
        s = np.zeros((n, k), dtype=np.float32)
        s[:, row] = 1.0

        tape.backward(grads={g: wp.array(s, dtype=Jk.coeff, device=device)})

        # z.grad[i,b] = H_i[row,b]
        out[:, row, :] = z.grad.numpy()
        tape.zero()

    return out


def hessian_width_1(k, kernel, z_np, device):
    n = z_np.shape[0]

    z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
    ones = wp.ones(n, dtype=float, device=device)

    out = np.empty((n, k, k), dtype=np.float32)

    for j in range(k):
        e = np.zeros(k, dtype=np.float32)
        e[j] = 1.0

        dv = wp.zeros(n, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=n,
                inputs=[z, wp.array(e, dtype=float, device=device)],
                outputs=[dv],
                device=device,
            )

        # grad_z (grad g . e_j) = H e_j = column j
        tape.backward(grads={dv: ones})

        out[:, :, j] = z.grad.numpy()
        tape.zero()

    return out


# ---------------------------------------------------------------------------


def check(k, Jk, kw, kd, device, n=16, seed=0):
    """Gate: both strategies must agree with each other and with float64 FD."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)

    hw = hessian_width_k(k, Jk, kw, z, device)
    hh = hessian_width_1(k, kd, z, device)
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

            # Warm up: compile, allocate, touch every path.
            hessian_width_k(k, Jk, kw, z, device)
            hessian_width_1(k, kd, z, device)
            wp.synchronize_device(device)

            tw = timeit(partial(hessian_width_k, k, Jk, kw, z, device), device, reps)
            th = timeit(partial(hessian_width_1, k, kd, z, device), device, reps)

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
