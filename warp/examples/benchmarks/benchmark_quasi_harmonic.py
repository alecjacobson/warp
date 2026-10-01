# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tight GPU metric-optimization loop for quasi-harmonic (BBW-style) weights.

This benchmarks the optimization at the heart of Wang & Solomon-style *Fast
Quasi-Harmonic Weights* and the DEC per-edge-metric variant studied in the
``gauss-newton-bbw`` repository. The model is identical to those references:

    minimize over the per-edge metric  theta
        f(u) = 1/2 u^T B u,     B = L M^-1 L   (fixed biharmonic energy)
    subject to  K(theta) u = 0,   u[boundary] = bc,
        K(theta) = d0^T diag(exp theta) d0     (the DEC-factored Laplacian)

solved with Adam on ``theta`` using the adjoint gradient from
:class:`warp.geometry.MetricHarmonicSolver`. Each Adam step rebuilds the reduced
interior operator ``K_uu(theta)`` and solves two linear systems (forward + adjoint).

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

For reference, the ``gauss-newton-bbw`` C++/CHOLMOD port (supernodal direct solve,
2x Xeon) runs this same optimization on the ``case02ctv`` mesh (23,140 vertices,
69,414 edges) at about 24.6 ms per Adam iteration. The graph-captured cuDSS loop
here runs the same mesh at well under 1 ms per iteration on an L40.

Running the cuDSS backend needs the optional ``warp-cudss`` package and a cuDSS
shared library (set ``CUDSS_LIBRARY_PATH`` or install ``nvidia-cudss-cu12``). The
conjugate-gradient backend always runs. Example::

    python benchmark_quasi_harmonic.py --mesh grid --res 128
    python benchmark_quasi_harmonic.py --mesh /path/to/mesh.ply --iters 500
"""

import argparse
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
    def __init__(self, mtype="spd", device=None):
        self.mtype = mtype
        self.device = wp.get_device(device)
        self._cud = None
        self._b = None
        self._x = None

    def prepare(self, matrix):
        if self._cud is None:
            self._b = wp.zeros(matrix.shape[0], dtype=matrix.scalar_type, device=self.device)
            self._x = wp.zeros(matrix.shape[0], dtype=matrix.scalar_type, device=self.device)
            self._cud = warp_cudss.CudssSolver(mtype=self.mtype, device=self.device)
            self._cud.setup(matrix, self._x, self._b)
        else:
            self._cud.refactor(matrix)

    def solve(self, rhs, x):
        wp.copy(self._b, rhs)
        self._cud.solve()
        wp.copy(x, self._x)


# ---------------------------------------------------------------------------
# Adam with its step counter on the device, so a captured graph replays the
# correct bias correction every iteration (a host-side Python counter would be
# frozen at its capture-time value).
# ---------------------------------------------------------------------------
@wp.kernel
def _adam_increment(t: wp.array(dtype=float)):
    t[0] = t[0] + 1.0


@wp.kernel
def _adam_step(
    grad: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    t: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    param: wp.array(dtype=float),
):
    i = wp.tid()
    g = grad[i]
    mi = beta1 * m[i] + (1.0 - beta1) * g
    vi = beta2 * v[i] + (1.0 - beta2) * g * g
    m[i] = mi
    v[i] = vi
    step = t[0]
    m_hat = mi / (1.0 - wp.pow(beta1, step))
    v_hat = vi / (1.0 - wp.pow(beta2, step))
    param[i] = param[i] - lr * m_hat / (wp.sqrt(v_hat) + eps)


class Adam:
    def __init__(self, n, lr, device, betas=(0.9, 0.999), eps=1e-8):
        self.m = wp.zeros(n, dtype=wp.float32, device=device)
        self.v = wp.zeros(n, dtype=wp.float32, device=device)
        self.t = wp.zeros(1, dtype=wp.float32, device=device)
        self.n, self.lr, self.beta1, self.beta2, self.eps, self.device = n, lr, betas[0], betas[1], eps, device

    def step(self, grad, param):
        wp.launch(_adam_increment, dim=1, inputs=[self.t], device=self.device)
        wp.launch(
            _adam_step,
            dim=self.n,
            inputs=[grad, self.m, self.v, self.t, self.lr, self.beta1, self.beta2, self.eps, param],
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
    return boundary, values


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mesh", type=str, default="grid", help="'grid' or a path to a binary PLY file.")
    parser.add_argument("--res", type=int, default=128, help="Grid resolution per side (if --mesh grid).")
    parser.add_argument("--iters", type=int, default=300, help="Adam iterations to time.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate on the log-metric.")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        run(args)


def run(args):
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError("This benchmark requires a CUDA device.")

    vertices, faces = build_grid(args.res) if args.mesh == "grid" else read_ply(args.mesh)
    num_points = vertices.shape[0]
    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices = wp.array(faces.flatten(), dtype=wp.int32, device=device)

    # Fixed operators: the DEC factors, the biharmonic energy B = L M^-1 L.
    d0, cotangent = warp.geometry.dec_operators(points, indices)
    num_edges = cotangent.shape[0]
    laplacian = warp.geometry.laplacian(points, indices)
    mass_diag = ws.bsr_get_diag(warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI))
    mass_inverse = wp.empty(num_points, dtype=wp.float32, device=device)
    wp.launch(_reciprocal, dim=num_points, inputs=[mass_diag], outputs=[mass_inverse], device=device)

    boundary_np, bc_np = two_region_bc(vertices)
    boundary = wp.array(boundary_np, dtype=wp.int32, device=device)
    bc = wp.array(bc_np, dtype=wp.float32, device=device)

    print(
        f"mesh={args.mesh}  {num_points} vertices, {faces.shape[0]} triangles, {num_edges} edges, "
        f"|boundary|={boundary_np.shape[0]}, {args.iters} iterations\n"
    )

    # Shared per-iteration scratch (persistent, so the captured graph reuses it).
    scratch = wp.empty(num_points, dtype=wp.float32, device=device)
    biharmonic_u = wp.empty(num_points, dtype=wp.float32, device=device)
    grad_log = wp.empty(num_edges, dtype=wp.float32, device=device)
    log_metric0 = np.log(np.maximum(cotangent.numpy(), 1e-4)).astype(np.float32)

    def apply_biharmonic(u, out):
        ws.bsr_mv(laplacian, u, y=scratch, beta=0.0)
        wp.launch(_multiply, dim=num_points, inputs=[scratch, mass_inverse], outputs=[scratch], device=device)
        ws.bsr_mv(laplacian, scratch, y=out, beta=0.0)

    def objective(u):
        apply_biharmonic(u, biharmonic_u)
        return 0.5 * float(np.dot(u.numpy(), biharmonic_u.numpy()))

    def make_solver(backend_name):
        backend = CudssLinearSolver(device=device) if backend_name == "cudss" else None
        return warp.geometry.MetricHarmonicSolver(
            d0, num_points, boundary, solver=backend, tol=1e-6, max_iters=4 * num_points
        )

    def time_backend(backend_name, capture):
        log_metric = wp.array(log_metric0, dtype=wp.float32, device=device)
        metric = wp.empty_like(log_metric)
        solver = make_solver(backend_name)
        optimizer = Adam(num_edges, args.lr, device)

        def iteration():
            wp.launch(_exp, dim=num_edges, inputs=[log_metric], outputs=[metric], device=device)
            solver.prepare(metric)  # assemble + (re)factor K_uu(theta)
            u = solver.solve(bc)  # forward harmonic solve
            apply_biharmonic(u, biharmonic_u)  # dF/du = B u
            grad_metric = solver.vjp(u, biharmonic_u)  # adjoint solve + edge gradient
            wp.launch(_multiply, dim=num_edges, inputs=[grad_metric, metric], outputs=[grad_log], device=device)
            optimizer.step(grad_log, log_metric)  # chain rule d/d(log) = metric * d/d(metric)
            return u

        u = iteration()  # warmup; also the one-time cuDSS analysis/factorization
        f0 = objective(u)

        if capture:
            wp.synchronize_device(device)
            with wp.ScopedCapture(device) as capture_ctx:
                u = iteration()  # u is written in place on every replay
            wp.synchronize_device(device)

        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            if capture:
                wp.capture_launch(capture_ctx.graph)
            else:
                u = iteration()
        wp.synchronize_device(device)
        elapsed = time.perf_counter() - start
        return elapsed, f0, objective(u)

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
    print(
        "\nReference (gauss-newton-bbw C++/CHOLMOD, 2x Xeon) on case02ctv: ~24.6 ms/iter "
        "(49.3 s / 2000 iters) to f ~ 1.9e-4."
    )


if __name__ == "__main__":
    main()
