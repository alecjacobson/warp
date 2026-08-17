# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Local gradient/Hessian of real element energies: forward jets vs reverse
###########################################################################

"""Per-element gradient and Hessian of geometry-processing energies, several ways.

Elements (k = scalars per element):

    spring    k=6   two vec3 nodes,   0.5 (||p0-p1|| - r)^2
    triangle  k=9   three vec3 nodes, symmetric Dirichlet  tr(S) + tr(S)/det(S)
    tet       k=12  four vec3 nodes,  neo-Hookean  0.5 mu (I1-3) - mu ln J + 0.5 lam (ln J)^2

Each energy is one generic wp.func in scalar form, so the same definition feeds
every strategy (jet arithmetic is registered as builtin overloads):

  gradient      jet-grad : one width-k forward jet pass -> grad (one launch, no tape)
                rev-grad : Warp reverse-mode (forward + tape.backward)

  hessian       forward-2 : one second-order jet pass -> full k x k Hessian
                width-k   : one width-k forward jet -> grad, then k reverse sweeps

forward-2 holds an O(k^2) Hessian in registers; nvcc compile is super-linear in
k (tet k=12 takes ~2 min the first time, then caches). --forward2-max-k skips it
above a chosen k for quick runs; --graph times warm graph-captured replay.

    uv run warp/examples/benchmarks/benchmark_element_hessian.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_element_hessian.py --device cuda:0 --graph
    uv run warp/examples/benchmarks/benchmark_element_hessian.py --element spring
"""

import argparse
import time
from typing import Any

import numpy as np

import warp as wp

# Reference (rest) shapes, baked as constants.
_REST_LEN = wp.constant(1.0)
# Equilateral triangle reference edge inverse M = [u v]^-1, u=(1,0), v=(0.5, sqrt(3)/2).
_M00 = wp.constant(1.0)
_M01 = wp.constant(-0.5773502692)
_M10 = wp.constant(0.0)
_M11 = wp.constant(1.1547005384)
_MU = wp.constant(1.0)
_LAMBDA = wp.constant(1.0)


# ---------------------------------------------------------------------------
# Energies -- generic wp.funcs, scalar form (no vec types, so forward-2 works).
# ---------------------------------------------------------------------------


@wp.func
def spring_energy(a0: Any, a1: Any, a2: Any, b0: Any, b1: Any, b2: Any):
    d0 = a0 - b0
    d1 = a1 - b1
    d2 = a2 - b2
    length = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    s = length - _REST_LEN
    return 0.5 * s * s


@wp.func
def triangle_energy(a0: Any, a1: Any, a2: Any, b0: Any, b1: Any, b2: Any, c0: Any, c1: Any, c2: Any):
    e10 = b0 - a0
    e11 = b1 - a1
    e12 = b2 - a2
    e20 = c0 - a0
    e21 = c1 - a1
    e22 = c2 - a2
    # J = deformation gradient (3x2) = [e1 e2] M, columns j0,j1 (vec3).
    j00 = _M00 * e10 + _M10 * e20
    j01 = _M00 * e11 + _M10 * e21
    j02 = _M00 * e12 + _M10 * e22
    j10 = _M01 * e10 + _M11 * e20
    j11 = _M01 * e11 + _M11 * e21
    j12 = _M01 * e12 + _M11 * e22
    s00 = j00 * j00 + j01 * j01 + j02 * j02
    s01 = j00 * j10 + j01 * j11 + j02 * j12
    s11 = j10 * j10 + j11 * j11 + j12 * j12
    tr = s00 + s11
    det = s00 * s11 - s01 * s01
    return tr + tr / det


@wp.func
def tet_energy(
    a0: Any,
    a1: Any,
    a2: Any,
    b0: Any,
    b1: Any,
    b2: Any,
    c0: Any,
    c1: Any,
    c2: Any,
    d0: Any,
    d1: Any,
    d2: Any,
):
    # F = [p1-p0 | p2-p0 | p3-p0] (unit-tet reference so Dm^-1 = I).
    f00 = b0 - a0
    f10 = b1 - a1
    f20 = b2 - a2
    f01 = c0 - a0
    f11 = c1 - a1
    f21 = c2 - a2
    f02 = d0 - a0
    f12 = d1 - a1
    f22 = d2 - a2
    i1 = f00 * f00 + f10 * f10 + f20 * f20 + f01 * f01 + f11 * f11 + f21 * f21 + f02 * f02 + f12 * f12 + f22 * f22
    detf = f00 * (f11 * f22 - f12 * f21) - f01 * (f10 * f22 - f12 * f20) + f02 * (f10 * f21 - f11 * f20)
    logj = wp.log(detf)
    return 0.5 * _MU * (i1 - 3.0) - _MU * logj + 0.5 * _LAMBDA * logj * logj


# NumPy references (float64) for the finite-difference gate.


def spring_np(z):
    d = z[:, 0:3] - z[:, 3:6]
    return 0.5 * (np.sqrt((d * d).sum(1)) - 1.0) ** 2


def triangle_np(z):
    p0, p1, p2 = z[:, 0:3], z[:, 3:6], z[:, 6:9]
    e1, e2 = p1 - p0, p2 - p0
    m = np.array([[1.0, -0.5773502692], [0.0, 1.1547005384]])
    jc0 = m[0, 0] * e1 + m[1, 0] * e2
    jc1 = m[0, 1] * e1 + m[1, 1] * e2
    s00 = (jc0 * jc0).sum(1)
    s01 = (jc0 * jc1).sum(1)
    s11 = (jc1 * jc1).sum(1)
    tr = s00 + s11
    return tr + tr / (s00 * s11 - s01 * s01)


def tet_np(z):
    p0, p1, p2, p3 = z[:, 0:3], z[:, 3:6], z[:, 6:9], z[:, 9:12]
    f = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)
    i1 = (f * f).sum((1, 2))
    logj = np.log(np.linalg.det(f))
    return 0.5 * (i1 - 3.0) - logj + 0.5 * logj * logj


def sample(k, m, seed=0):
    """Realistic states: reference element + small perturbation (non-degenerate)."""
    rng = np.random.default_rng(seed)
    rest = {
        6: [0, 0, 0, 1, 0, 0],
        9: [0, 0, 0, 1, 0, 0, 0.5, 0.8660254, 0],
        12: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    }[k]
    return np.array(rest, np.float32) + 0.15 * rng.standard_normal((m, k)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-element kernels. Explicit seeding calls the shared generic energy func.
# ---------------------------------------------------------------------------


def build_spring():
    Jk = wp.JetSpace(6)
    J2 = wp.JetSpace2(6)

    @wp.kernel
    def spring_grad_wide(z: wp.array2d[wp.float32], out: wp.array[Jk.coeff]):
        i = wp.tid()
        out[i] = spring_energy(
            Jk.seed(z[i, 0], 0),
            Jk.seed(z[i, 1], 1),
            Jk.seed(z[i, 2], 2),
            Jk.seed(z[i, 3], 3),
            Jk.seed(z[i, 4], 4),
            Jk.seed(z[i, 5], 5),
        ).coeff

    @wp.kernel
    def spring_energy_scalar(z: wp.array2d[wp.float32], e: wp.array[wp.float32]):
        i = wp.tid()
        e[i] = spring_energy(z[i, 0], z[i, 1], z[i, 2], z[i, 3], z[i, 4], z[i, 5])

    @wp.kernel
    def spring_hess_forward(z: wp.array2d[wp.float32], out: wp.array3d[wp.float32]):
        i = wp.tid()
        h = spring_energy(
            J2.seed(z[i, 0], 0),
            J2.seed(z[i, 1], 1),
            J2.seed(z[i, 2], 2),
            J2.seed(z[i, 3], 3),
            J2.seed(z[i, 4], 4),
            J2.seed(z[i, 5], 5),
        ).hess
        for p in range(6):
            for q in range(6):
                out[i, p, q] = h[p, q]

    return Jk, spring_grad_wide, spring_energy_scalar, spring_hess_forward


def build_triangle():
    Jk = wp.JetSpace(9)
    J2 = wp.JetSpace2(9)

    @wp.kernel
    def triangle_grad_wide(z: wp.array2d[wp.float32], out: wp.array[Jk.coeff]):
        i = wp.tid()
        out[i] = triangle_energy(
            Jk.seed(z[i, 0], 0),
            Jk.seed(z[i, 1], 1),
            Jk.seed(z[i, 2], 2),
            Jk.seed(z[i, 3], 3),
            Jk.seed(z[i, 4], 4),
            Jk.seed(z[i, 5], 5),
            Jk.seed(z[i, 6], 6),
            Jk.seed(z[i, 7], 7),
            Jk.seed(z[i, 8], 8),
        ).coeff

    @wp.kernel
    def triangle_energy_scalar(z: wp.array2d[wp.float32], e: wp.array[wp.float32]):
        i = wp.tid()
        e[i] = triangle_energy(z[i, 0], z[i, 1], z[i, 2], z[i, 3], z[i, 4], z[i, 5], z[i, 6], z[i, 7], z[i, 8])

    @wp.kernel
    def triangle_hess_forward(z: wp.array2d[wp.float32], out: wp.array3d[wp.float32]):
        i = wp.tid()
        h = triangle_energy(
            J2.seed(z[i, 0], 0),
            J2.seed(z[i, 1], 1),
            J2.seed(z[i, 2], 2),
            J2.seed(z[i, 3], 3),
            J2.seed(z[i, 4], 4),
            J2.seed(z[i, 5], 5),
            J2.seed(z[i, 6], 6),
            J2.seed(z[i, 7], 7),
            J2.seed(z[i, 8], 8),
        ).hess
        for p in range(9):
            for q in range(9):
                out[i, p, q] = h[p, q]

    return Jk, triangle_grad_wide, triangle_energy_scalar, triangle_hess_forward


def build_tet():
    Jk = wp.JetSpace(12)
    J2 = wp.JetSpace2(12)

    @wp.kernel
    def tet_grad_wide(z: wp.array2d[wp.float32], out: wp.array[Jk.coeff]):
        i = wp.tid()
        out[i] = tet_energy(
            Jk.seed(z[i, 0], 0),
            Jk.seed(z[i, 1], 1),
            Jk.seed(z[i, 2], 2),
            Jk.seed(z[i, 3], 3),
            Jk.seed(z[i, 4], 4),
            Jk.seed(z[i, 5], 5),
            Jk.seed(z[i, 6], 6),
            Jk.seed(z[i, 7], 7),
            Jk.seed(z[i, 8], 8),
            Jk.seed(z[i, 9], 9),
            Jk.seed(z[i, 10], 10),
            Jk.seed(z[i, 11], 11),
        ).coeff

    @wp.kernel
    def tet_energy_scalar(z: wp.array2d[wp.float32], e: wp.array[wp.float32]):
        i = wp.tid()
        e[i] = tet_energy(
            z[i, 0],
            z[i, 1],
            z[i, 2],
            z[i, 3],
            z[i, 4],
            z[i, 5],
            z[i, 6],
            z[i, 7],
            z[i, 8],
            z[i, 9],
            z[i, 10],
            z[i, 11],
        )

    @wp.kernel
    def tet_hess_forward(z: wp.array2d[wp.float32], out: wp.array3d[wp.float32]):
        i = wp.tid()
        h = tet_energy(
            J2.seed(z[i, 0], 0),
            J2.seed(z[i, 1], 1),
            J2.seed(z[i, 2], 2),
            J2.seed(z[i, 3], 3),
            J2.seed(z[i, 4], 4),
            J2.seed(z[i, 5], 5),
            J2.seed(z[i, 6], 6),
            J2.seed(z[i, 7], 7),
            J2.seed(z[i, 8], 8),
            J2.seed(z[i, 9], 9),
            J2.seed(z[i, 10], 10),
            J2.seed(z[i, 11], 11),
        ).hess
        for p in range(12):
            for q in range(12):
                out[i, p, q] = h[p, q]

    return Jk, tet_grad_wide, tet_energy_scalar, tet_hess_forward


ELEMENTS = {
    "spring": (6, build_spring, spring_np),
    "triangle": (9, build_triangle, triangle_np),
    "tet": (12, build_tet, tet_np),
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@wp.kernel
def _scatter_row(src: wp.array2d[wp.float32], row: int, dst: wp.array3d[wp.float32]):
    i, b = wp.tid()
    dst[i, row, b] = src[i, b]


class JetGrad:
    def __init__(self, k, Jk, kernel, z_np, device):
        self.k, self.kernel, self.device, self.n = k, kernel, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=wp.float32, device=device)
        self.out = wp.zeros(self.n, dtype=Jk.coeff, device=device)

    def run(self):
        wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.out], device=self.device)

    def grad(self):
        return self.out.numpy().reshape(self.n, self.k)


class ReverseGrad:
    def __init__(self, k, kernel, z_np, device):
        self.k, self.device, self.n = k, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=wp.float32, device=device, requires_grad=True)
        self.e = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self.ones = wp.ones(self.n, dtype=wp.float32, device=device)
        self.tape = wp.Tape()
        with self.tape:
            wp.launch(kernel, dim=self.n, inputs=[self.z], outputs=[self.e], device=device)

    def run(self):
        self.z.grad.zero_()
        self.tape.backward(grads={self.e: self.ones})

    def grad(self):
        return self.z.grad.numpy().reshape(self.n, self.k)


class WidthK:
    """Reverse-over-forward Hessian: one width-k forward jet, then k reverse sweeps."""

    def __init__(self, k, Jk, kernel, z_np, device):
        self.k, self.kernel, self.device, self.n = k, kernel, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=wp.float32, device=device, requires_grad=True)
        self.g = wp.zeros(self.n, dtype=Jk.coeff, device=device, requires_grad=True)
        self.hessian = wp.zeros((self.n, k, k), dtype=wp.float32, device=device)
        self.seeds = []
        for row in range(k):
            s = np.zeros((self.n, k), np.float32)
            s[:, row] = 1.0
            self.seeds.append(wp.array(s, dtype=Jk.coeff, device=device))

    def run(self):
        tape = wp.Tape()
        with tape:
            wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.g], device=self.device)
        for row in range(self.k):
            tape.backward(grads={self.g: self.seeds[row]})
            wp.launch(_scatter_row, dim=(self.n, self.k), inputs=[self.z.grad, row, self.hessian], device=self.device)
            tape.zero()

    def hess(self):
        return self.hessian.numpy()


class Forward2:
    """One second-order jet pass; the Hessian is a forward output."""

    def __init__(self, k, kernel, z_np, device):
        self.k, self.kernel, self.device, self.n = k, kernel, device, z_np.shape[0]
        self.z = wp.array(z_np, dtype=wp.float32, device=device)
        self.hessian = wp.zeros((self.n, k, k), dtype=wp.float32, device=device)

    def run(self):
        wp.launch(self.kernel, dim=self.n, inputs=[self.z], outputs=[self.hessian], device=self.device)

    def hess(self):
        return self.hessian.numpy()


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


def hess_fd(energy_np, z, k, h=1.0e-4):
    out = np.empty((z.shape[0], k, k))
    for p in range(k):
        for q in range(k):
            ep = np.zeros_like(z)
            ep[:, p] = h
            eq = np.zeros_like(z)
            eq[:, q] = h
            out[:, p, q] = (
                energy_np(z + ep + eq) - energy_np(z + ep - eq) - energy_np(z - ep + eq) + energy_np(z - ep - eq)
            ) / (4 * h * h)
    return out


def grad_fd(energy_np, z, k, h=1.0e-4):
    out = np.empty((z.shape[0], k))
    for p in range(k):
        ep = np.zeros_like(z)
        ep[:, p] = h
        out[:, p] = (energy_np(z + ep) - energy_np(z - ep)) / (2 * h)
    return out


# forward-2's O(k^2) register Hessian compiles super-linearly in k: spring k=6
# ~4 s, triangle k=9 ~45 s, tet k=12 ~2 min (one-time, then cached). Raise this to
# skip the slow tet compile during quick runs.
FORWARD2_MAX_K = 12


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__.split("\n")[0]
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--element", type=str, default=None, choices=list(ELEMENTS), help="Default: all.")
    parser.add_argument("--m", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay.")
    parser.add_argument(
        "--forward2-max-k", type=int, default=FORWARD2_MAX_K, help="Skip forward-2 above this k (slow compile)."
    )
    args = parser.parse_args()

    wp.init()
    device = args.device
    mode = "graph replay (warm)" if args.graph else "per-launch"
    print(f"device: {wp.get_device(device)}   m: {args.m}   reps: {args.reps}   timing: {mode}\n")

    header = (
        f"{'element':>9} {'k':>3} {'jet-grad':>9} {'rev-grad':>9} {'g:jet/rev':>10}"
        f" {'fwd-2':>9} {'width-k':>9} {'h:f2/wk':>9} {'|fd|':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in [args.element] if args.element else list(ELEMENTS):
        k, builder, energy_np = ELEMENTS[name]
        Jk, grad_wide, energy_k, hess_k = builder()
        if k > args.forward2_max_k:
            hess_k = None

        z_np = sample(k, args.m)

        # Correctness gate on a small sample.
        zc = sample(k, 16, seed=1)
        jg = JetGrad(k, Jk, grad_wide, zc, device)
        jg.run()
        wk = WidthK(k, Jk, grad_wide, zc, device)
        wk.run()
        wp.synchronize_device(device)
        zc64 = zc.astype(np.float64)
        fd_err = max(
            np.abs(jg.grad() - grad_fd(energy_np, zc64, k)).max(), np.abs(wk.hess() - hess_fd(energy_np, zc64, k)).max()
        )

        tjg = time_strategy(JetGrad(k, Jk, grad_wide, z_np, device), device, args.reps, args.graph)
        trg = time_strategy(ReverseGrad(k, energy_k, z_np, device), device, args.reps, args.graph)
        twk = time_strategy(WidthK(k, Jk, grad_wide, z_np, device), device, args.reps, args.graph)

        if hess_k is not None:
            f2 = Forward2(k, hess_k, zc, device)
            f2.run()
            wp.synchronize_device(device)
            fd_err = max(fd_err, np.abs(f2.hess() - hess_fd(energy_np, zc64, k)).max())
            tf2 = time_strategy(Forward2(k, hess_k, z_np, device), device, args.reps, args.graph)
            f2s, hr = f"{tf2 * 1e3:>9.3f}", f"{tf2 / twk:>8.2f}x"
        else:
            f2s, hr = f"{'-':>9}", f"{'-':>9}"

        print(
            f"{name:>9} {k:>3} {tjg * 1e3:>9.3f} {trg * 1e3:>9.3f} {tjg / trg:>9.2f}x"
            f" {f2s} {twk * 1e3:>9.3f} {hr} {fd_err:>9.1e}"
        )


if __name__ == "__main__":
    main()
