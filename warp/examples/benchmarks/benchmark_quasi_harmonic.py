# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tight GPU metric-optimization loop for quasi-harmonic (BBW-style) weights.

This benchmarks the optimization at the heart of Wang & Solomon's *Fast
Quasi-Harmonic Weights* and the DEC per-edge-metric variant studied in the
``gauss-newton-bbw`` repository. The model is identical to those references:

    minimize over the per-edge metric  theta
        f(U) = 1/2 sum_h U_h^T B U_h,   B = L M^-1 L   (fixed biharmonic energy)
    subject to  K(theta) U_h = 0,   U_h[boundary] = bc_h,
        K(theta) = d0^T diag(exp theta) d0     (the DEC-factored Laplacian)

solved with Adam on ``theta`` using the adjoint gradient from
:class:`warp.geometry.MetricHarmonicSolver`. Each Adam step rebuilds the reduced
interior operator ``K_uu(theta)`` and solves two linear systems per handle
(forward + adjoint).

The point of the benchmark is twofold:

1. The inner solve dominates, and the reduced operator is symmetric positive
   definite with a *fixed sparsity pattern* across the whole optimization -- only
   its values change. That is exactly the regime a sparse direct solver wins: factor
   the pattern once, refactor-and-solve cheaply thereafter. Plugging cuDSS into the
   :class:`~warp.geometry.LinearSolver` seam makes each iteration roughly an order of
   magnitude cheaper than Jacobi-preconditioned conjugate gradient, which struggles
   on the increasingly anisotropic metric.

2. With a direct solve the whole iteration -- metric exponentiation, operator
   assembly, refactorization, forward/adjoint solves, the biharmonic energy, the
   adjoint VJP, and the Adam update -- is free of host synchronization, so it can be
   captured into a single CUDA graph and replayed. This removes per-launch overhead
   and is what makes the loop *tight*.

It can run directly on the Wang & Solomon ``qhw-code`` example datasets (a
directory holding ``V.mtx``, ``F.mtx`` for triangles or tets, ``B.mtx`` boundary
indices, ``BC.mtx`` handle values, and optional ``mv.mtx`` lumped mass).

All handle columns are solved together: :class:`~warp.geometry.MetricHarmonicSolver`
assembles every column's right-hand side and calls the backend's ``solve_many``, and
the cuDSS backend below back-substitutes them in one multi-RHS solve (the solves,
which dominate, are then batched). ``--max-rhs`` caps how many columns share a solve;
a very large 3D factor plus a wide solve workspace can exhaust device memory, in which
case the columns are split into balanced chunks.

For reference, their own ``qhw`` ``adamd`` solver (Intel MKL + CHOLMOD supernodal
direct solve) on the ``tibiman-H`` tet mesh (22,263 vertices, 84,125 tets, 16
handles) costs ~180 ms/iteration in the paper (1.80 s for the k=10 run on an
i9-7900X); the exact binary measured on the shared CPU here costs ~300-350
ms/iteration (it runs ~1.9x slower than the paper's machine). The graph-captured
cuDSS loop here runs that mesh at ~2.8 ms/iteration on an L40, and the larger
``dragon-H`` mesh (330,206 vertices, 1,187,670 tets, 17 handles) at ~31 ms/iteration
(``--max-rhs 16``) versus ~11 s/iteration for the official binary. This benchmark
measures optimization-loop cost per iteration; it is not a weight-quality comparison
(for that, score each method's output weights through the same ``B = L M^-1 L``).

The cuDSS backend needs the optional ``warp-cudss`` package and a cuDSS shared
library (set ``CUDSS_LIBRARY_PATH`` or install ``nvidia-cudss-cu12``); the
conjugate-gradient backend always runs. Examples::

    python benchmark_quasi_harmonic.py --mesh grid --res 128
    python benchmark_quasi_harmonic.py --mesh /path/to/qhw-code/data/tibiman-H
"""

import argparse
import os
import struct
import time

import numpy as np

import warp as wp
import warp.geometry
import warp.sparse as ws

try:
    import warp_cudss

    _HAVE_CUDSS = True
except ImportError:
    _HAVE_CUDSS = False

# Published timings of the reference solvers on shared meshes, keyed by a substring
# of the mesh argument, for a side-by-side line in the output.
_REFERENCES = {
    # Paper numbers are on the authors' i9-7900X; the measured numbers are the actual
    # qhw binary on whatever CPU this benchmark shares, which runs it ~1.9x slower --
    # so the apples-to-apples claim is same-machine (GPU vs CPU), not vs the paper.
    "tibiman": "qhw adamd: paper 'Ours k=10' 1.80 s (i9-7900X); measured here "
    "346 ms/iter (3.47 s / 10 iters), 298 ms/iter (7.46 s / 25 iters)",
    "grid2d-40": "qhw adamd measured here: ~5.4 ms/iter (2.75 s / 510 iters to f=52.79)",
    "case02": "gauss-newton-bbw C++/CHOLMOD (2x Xeon): ~24.6 ms/iter (49.3 s / 2000 iters to f~1.9e-4)",
}


# ---------------------------------------------------------------------------
# cuDSS as a geometry.LinearSolver backend.
#
# prepare() factors the reduced operator the first time (analysis + numeric
# factorization, done once outside any graph capture) and refactors in place on
# later calls, since the sparsity pattern never changes. solve() copies the
# right-hand side into a persistent buffer, back-substitutes, and copies out --
# all on-device, so it is safe inside a captured graph.
# ---------------------------------------------------------------------------
class CudssLinearSolver(warp.geometry.LinearSolver):
    """cuDSS direct solver with batched multi-RHS support.

    ``solve_many`` solves all handle columns through cuDSS multi-RHS solves instead of
    one at a time. ``max_rhs`` caps the number of columns solved together (the solve
    workspace grows with it, and very large 3D factors can exhaust device memory); the
    columns are split into balanced chunks no larger than that cap.
    """

    def __init__(self, mtype="spd", max_rhs=None, device=None):
        self.mtype = mtype
        self.max_rhs = max_rhs
        self.device = wp.get_device(device)
        self._cud = None
        self._matrix = None
        self._b = None
        self._x = None
        self._nrhs = None

    def prepare(self, matrix):
        self._matrix = matrix
        if self._cud is not None:
            self._cud.refactor(matrix)  # same pattern, new values

    def _ensure(self, n, k):
        if self._cud is None:
            cap = k if self.max_rhs is None else min(k, self.max_rhs)
            num_chunks = (k + cap - 1) // cap
            self._nrhs = (k + num_chunks - 1) // num_chunks  # balanced chunk width
            self._b = wp.zeros(n * self._nrhs, dtype=self._matrix.scalar_type, device=self.device)
            self._x = wp.zeros(n * self._nrhs, dtype=self._matrix.scalar_type, device=self.device)
            self._cud = warp_cudss.CudssSolver(mtype=self.mtype, device=self.device)
            self._cud.setup(self._matrix, self._x, self._b, nrhs=self._nrhs)

    def solve(self, rhs, x):
        self._ensure(rhs.shape[0], 1)
        n = rhs.shape[0]
        wp.copy(self._b[:n], rhs)
        self._cud.solve()
        wp.copy(x, self._x[:n])

    def solve_many(self, rhs, x):
        n, k = rhs.shape
        self._ensure(n, k)
        nrhs = self._nrhs
        for start in range(0, k, nrhs):
            cols = range(start, min(start + nrhs, k))
            for j, c in enumerate(cols):
                wp.launch(
                    _get_column, dim=n, inputs=[rhs, c], outputs=[self._b[j * n : (j + 1) * n]], device=self.device
                )
            self._cud.solve()
            for j, c in enumerate(cols):
                wp.launch(_set_column, dim=n, inputs=[x, c, self._x[j * n : (j + 1) * n]], device=self.device)


# ---------------------------------------------------------------------------
# Restarting Adam / Nadam, matching Wang & Solomon's `adamd` (see the MATLAB
# `quasiharmonic` reference). Every `restart_period` steps the moments reset,
# the step size halves, and bias correction restarts from the local iteration
# counter -- the annealing schedule that lets it converge in ~25 steps. Everything
# (iteration counter, step size, restart) is computed on the device from a single
# iteration counter, so the loop stays graph-capturable.
# ---------------------------------------------------------------------------
@wp.kernel
def _adam_increment(t: wp.array(dtype=float)):
    t[0] = t[0] + 1.0


@wp.kernel
def _restart_adam_step(
    grad: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    it_arr: wp.array(dtype=float),
    lr0: float,
    period: float,
    beta1: float,
    beta2: float,
    eps: float,
    nadam: int,
    param: wp.array(dtype=float),
):
    i = wp.tid()
    it = it_arr[0]  # global iteration, 1-based
    r = wp.floor((it - 1.0) / period)  # number of restarts so far
    local = (it - 1.0) - r * period + 1.0  # iterations since last restart
    lr = lr0 * wp.pow(0.5, r)  # step size halves on each restart

    g = grad[i]
    m_prev = m[i]
    v_prev = v[i]
    if it > 1.0:
        if local == 1.0:  # restart: forget the moments
            m_prev = 0.0
            v_prev = 0.0
    mi = beta1 * m_prev + (1.0 - beta1) * g
    vi = beta2 * v_prev + (1.0 - beta2) * g * g
    m[i] = mi
    v[i] = vi

    m_hat = mi / (1.0 - wp.pow(beta1, local))
    v_hat = vi / (1.0 - wp.pow(beta2, local))
    if nadam == 1:  # Nesterov look-ahead on the first moment
        m_hat = beta1 * m_hat + (1.0 - beta1) * g / (1.0 - wp.pow(beta1, local))
    param[i] = param[i] - lr * m_hat / (wp.sqrt(v_hat) + eps)


class Adam:
    """Restarting Adam/Nadam. ``restart_period`` <= 0 disables restarts (plain Adam)."""

    def __init__(self, n, lr, device, betas=(0.9, 0.999), eps=1e-8, restart_period=4, nadam=False):
        self.m = wp.zeros(n, dtype=wp.float32, device=device)
        self.v = wp.zeros(n, dtype=wp.float32, device=device)
        self.t = wp.zeros(1, dtype=wp.float32, device=device)
        self.n, self.lr, self.beta1, self.beta2, self.eps, self.device = n, lr, betas[0], betas[1], eps, device
        self.period = float(restart_period) if restart_period and restart_period > 0 else 1.0e18
        self.nadam = 1 if nadam else 0

    def step(self, grad, param):
        wp.launch(_adam_increment, dim=1, inputs=[self.t], device=self.device)
        wp.launch(
            _restart_adam_step,
            dim=self.n,
            inputs=[
                grad,
                self.m,
                self.v,
                self.t,
                self.lr,
                self.period,
                self.beta1,
                self.beta2,
                self.eps,
                self.nadam,
                param,
            ],
            device=self.device,
        )


@wp.kernel
def _exp(src: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = wp.exp(src[wp.tid()])


@wp.kernel
def _multiply(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = a[wp.tid()] * b[wp.tid()]


@wp.kernel
def _reciprocal(src: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = 1.0 / src[wp.tid()]


@wp.kernel
def _get_column(src: wp.array2d(dtype=float), c: int, out: wp.array(dtype=float)):
    out[wp.tid()] = src[wp.tid(), c]


@wp.kernel
def _set_column(dst: wp.array2d(dtype=float), c: int, src: wp.array(dtype=float)):
    dst[wp.tid(), c] = src[wp.tid()]


def read_mtx(path):
    """Read a dense MatrixMarket array file (column-major) into a 2D NumPy array."""
    with open(path) as f:
        header = f.readline()
        if not header.startswith("%%MatrixMarket matrix array"):
            raise ValueError(f"not a dense MatrixMarket array: {path}")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, cols = (int(x) for x in line.split())
        values = np.fromstring(f.read(), sep="\n")
    return values.reshape(cols, rows).T  # MatrixMarket array is column-major


def load_qhw_dataset(directory):
    """Load a Wang & Solomon qhw-code example (V/F/B/BC, optional mv). Indices are 1-based."""
    vertices = read_mtx(os.path.join(directory, "V.mtx")).astype(np.float32)
    if vertices.shape[1] == 2:  # planar datasets store 2D coordinates
        vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1), np.float32)], axis=1)
    faces = read_mtx(os.path.join(directory, "F.mtx")).astype(np.int32) - 1
    boundary = read_mtx(os.path.join(directory, "B.mtx")).astype(np.int32).ravel() - 1
    bc = read_mtx(os.path.join(directory, "BC.mtx")).astype(np.float32)
    mv_path = os.path.join(directory, "mv.mtx")
    mass = read_mtx(mv_path).astype(np.float32).ravel() if os.path.exists(mv_path) else None
    return vertices, faces, boundary, bc, mass


def read_ply(path):
    """Read a little-endian binary PLY with double vertices and a triangle face list."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        if b"binary_little_endian" not in f.readline():
            raise ValueError("only binary_little_endian PLY is supported")
        num_vertices = num_faces = 0
        line = f.readline()
        while line.strip() != b"end_header":
            tokens = line.split()
            if tokens[:2] == [b"element", b"vertex"]:
                num_vertices = int(tokens[2])
            elif tokens[:2] == [b"element", b"face"]:
                num_faces = int(tokens[2])
            line = f.readline()
        vertices = np.frombuffer(f.read(num_vertices * 24), dtype="<f8").reshape(num_vertices, 3).astype(np.float32)
        face_buffer = f.read()
    faces = np.empty((num_faces, 3), np.int32)
    offset = 0
    for i in range(num_faces):
        (count,) = struct.unpack_from("<i", face_buffer, offset)
        offset += 4
        faces[i] = struct.unpack_from(f"<{count}i", face_buffer, offset)[:3]
        offset += 4 * count
    return vertices, faces


def build_grid(resolution):
    axis = np.linspace(0.0, 1.0, resolution + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([xx, yy, np.zeros_like(xx)], -1).reshape(-1, 3).astype(np.float32)
    grid = np.arange((resolution + 1) ** 2).reshape(resolution + 1, resolution + 1)
    lower, upper, right, above = grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1], grid[:-1, 1:]
    triangles = np.concatenate(
        [np.stack([lower, right, upper], -1).reshape(-1, 3), np.stack([lower, upper, above], -1).reshape(-1, 3)]
    ).astype(np.int32)
    return points, triangles


def two_region_bc(vertices, lo_frac=0.15, hi_frac=0.85, axis=0):
    """Dirichlet -1 on the low-``axis`` slab, +1 on the high slab (matches gauss-newton-bbw)."""
    lo = vertices[:, axis].min()
    span = vertices[:, axis].max() - lo
    below = np.nonzero(vertices[:, axis] < lo + lo_frac * span)[0]
    above = np.nonzero(vertices[:, axis] > lo + hi_frac * span)[0]
    boundary = np.concatenate([below, above]).astype(np.int32)
    values = np.concatenate([-np.ones(len(below)), np.ones(len(above))]).astype(np.float32)
    return boundary, values[:, None]  # (num_boundary, 1)


def load_problem(args):
    """Return vertices, faces, boundary indices, boundary values (2D), and optional lumped mass."""
    if args.mesh == "grid":
        vertices, faces = build_grid(args.res)
        boundary, bc = two_region_bc(vertices)
        return vertices, faces, boundary, bc, None
    if os.path.isdir(args.mesh):
        return load_qhw_dataset(args.mesh)
    vertices, faces = read_ply(args.mesh)
    boundary, bc = two_region_bc(vertices)
    return vertices, faces, boundary, bc, None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--mesh", type=str, default="grid", help="'grid', a binary PLY path, or a qhw-code dataset directory."
    )
    parser.add_argument("--res", type=int, default=128, help="Grid resolution per side (if --mesh grid).")
    parser.add_argument("--iters", type=int, default=300, help="Adam iterations to time.")
    parser.add_argument("--lr", type=float, default=0.2, help="Adam initial learning rate on the log-metric.")
    parser.add_argument(
        "--restart-period",
        type=int,
        default=8,
        help="Restart Adam every N steps (reset moments, halve the step), as in Wang & Solomon's adamd. "
        "Set 0 for no restarts (often as good or better, especially with Nadam).",
    )
    parser.add_argument(
        "--nadam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Nadam (Nesterov look-ahead). --no-nadam for plain Adam.",
    )
    parser.add_argument("--beta1", type=float, default=0.8, help="Adam first-moment decay.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam second-moment decay.")
    parser.add_argument(
        "--max-rhs",
        type=int,
        default=None,
        help="Cap on handle columns solved together by cuDSS (default: all at once). "
        "Lower it if a large 3D mesh exhausts device memory.",
    )
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        run(args)


def run(args):
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError("This benchmark requires a CUDA device.")

    vertices, faces, boundary_np, bc_np, mass_np = load_problem(args)
    num_points = vertices.shape[0]
    simplex_size = faces.shape[1]
    num_handles = bc_np.shape[1]

    points = wp.array(vertices, dtype=wp.vec3, device=device)
    # Pass a typed index array so the simplex size (triangle vs tet) is unambiguous.
    indices = wp.array(faces, dtype=wp.vec3i if simplex_size == 3 else wp.vec4i, device=device)

    # Fixed operators: the DEC factors, the biharmonic energy B = L M^-1 L.
    d0, cotangent = warp.geometry.dec_operators(points, indices)
    num_edges = cotangent.shape[0]
    laplacian = warp.geometry.laplacian(points, indices)
    if mass_np is not None:
        mass_inverse = wp.array(1.0 / mass_np, dtype=wp.float32, device=device)
    else:
        mass_diag = ws.bsr_get_diag(
            warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI)
        )
        mass_inverse = wp.empty(num_points, dtype=wp.float32, device=device)
        wp.launch(_reciprocal, dim=num_points, inputs=[mass_diag], outputs=[mass_inverse], device=device)

    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)
    bc = wp.array(bc_np, dtype=wp.float32, device=device)

    kind = "tets" if simplex_size == 4 else "triangles"
    print(
        f"mesh={args.mesh}  {num_points} vertices, {faces.shape[0]} {kind}, {num_edges} edges, "
        f"{num_handles} handles, |boundary|={boundary_np.shape[0]}, {args.iters} iterations\n"
    )

    # Persistent per-iteration scratch (so the captured graph reuses it).
    biharmonic_U = wp.empty((num_points, num_handles), dtype=wp.float32, device=device)
    column = wp.empty(num_points, dtype=wp.float32, device=device)
    scratch = wp.empty(num_points, dtype=wp.float32, device=device)
    grad_log = wp.empty(num_edges, dtype=wp.float32, device=device)
    log_metric0 = np.log(np.maximum(cotangent.numpy(), 1e-4)).astype(np.float32)

    def apply_biharmonic(U, out):
        # out[:, h] = L (Minv (L U[:, h])) for each handle column h.
        for h in range(num_handles):
            wp.launch(_get_column, dim=num_points, inputs=[U, h], outputs=[column], device=device)
            ws.bsr_mv(laplacian, column, y=scratch, beta=0.0)
            wp.launch(_multiply, dim=num_points, inputs=[scratch, mass_inverse], outputs=[scratch], device=device)
            ws.bsr_mv(laplacian, scratch, y=column, beta=0.0)
            wp.launch(_set_column, dim=num_points, inputs=[out, h], outputs=[column], device=device)

    def objective(U):
        # f = 1/2 sum_h U_h^T B U_h. Note qhw reports sum_h U_h^T B U_h (no 1/2), i.e.
        # twice this value, so compare qhw's printed energy against 2 * f here.
        apply_biharmonic(U, biharmonic_U)
        return 0.5 * float(np.sum(U.numpy() * biharmonic_U.numpy()))

    def make_solver(backend_name):
        backend = CudssLinearSolver(max_rhs=args.max_rhs, device=device) if backend_name == "cudss" else None
        return warp.geometry.MetricHarmonicSolver(
            d0, num_points, boundary, solver=backend, tol=1e-6, max_iters=4 * num_points
        )

    def time_backend(backend_name, capture):
        log_metric = wp.array(log_metric0, dtype=wp.float32, device=device)
        metric = wp.empty_like(log_metric)
        solver = make_solver(backend_name)
        optimizer = Adam(
            num_edges,
            args.lr,
            device,
            betas=(args.beta1, args.beta2),
            restart_period=args.restart_period,
            nadam=args.nadam,
        )

        def iteration():
            wp.launch(_exp, dim=num_edges, inputs=[log_metric], outputs=[metric], device=device)
            solver.prepare(metric)  # assemble + (re)factor K_uu(theta)
            U = solver.solve(bc)  # forward harmonic solve, all handles
            apply_biharmonic(U, biharmonic_U)  # dF/dU = B U
            grad_metric = solver.vjp(U, biharmonic_U)  # adjoint solves + edge gradient
            wp.launch(_multiply, dim=num_edges, inputs=[grad_metric, metric], outputs=[grad_log], device=device)
            optimizer.step(grad_log, log_metric)  # chain rule d/d(log) = metric * d/d(metric)
            return U

        U = iteration()  # warmup; also the one-time cuDSS analysis/factorization
        f0 = objective(U)

        if capture:
            wp.synchronize_device(device)
            with wp.ScopedCapture(device) as capture_ctx:
                U = iteration()  # U is written in place on every replay
            wp.synchronize_device(device)

        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            if capture:
                wp.capture_launch(capture_ctx.graph)
            else:
                U = iteration()
        wp.synchronize_device(device)
        elapsed = time.perf_counter() - start
        return elapsed, f0, objective(U)

    backends = [("cg", "GPU CG + Jacobi")]
    if _HAVE_CUDSS:
        backends.append(("cudss", "GPU cuDSS direct"))
    else:
        print("warp-cudss not available -- running the conjugate-gradient backend only.\n")

    print(f"{'backend':<22}{'mode':<10}{'ms/iter':>10}{'total s':>10}{'f_initial':>14}{'f_final':>14}")
    print("-" * 80)
    for name, label in backends:
        for capture in (False, True):
            elapsed, f0, ff = time_backend(name, capture)
            mode = "captured" if capture else "eager"
            print(f"{label:<22}{mode:<10}{1000 * elapsed / args.iters:>10.3f}{elapsed:>10.3f}{f0:>14.4e}{ff:>14.4e}")

    for key, line in _REFERENCES.items():
        if key in args.mesh:
            print(f"\nReference ({key}): {line}")
            break


if __name__ == "__main__":
    main()
