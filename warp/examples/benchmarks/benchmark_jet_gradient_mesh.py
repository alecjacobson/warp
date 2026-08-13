# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Gradient of a summed element loss: forward jet vs reverse-mode autodiff
###########################################################################

"""First-order forward jet vs reverse mode for the gradient of a summed mesh loss.

The loss sums a local energy over overlapping elements -- the shape of every mesh
objective. We assemble the vertex gradient dloss/dx (one vec3 per vertex) two ways
on real shared-vertex meshes:

    spring    k=6    chain,           0.5 (||pi - pj|| - r)^2
    triangle  k=9    grid mesh,       symmetric Dirichlet  tr(S) + tr(S)/det(S)
    tet       k=12   cube-Kuhn mesh,  neo-Hookean  0.5 mu (I1-3) - mu lnJ + 0.5 lam lnJ^2

    reverse : one scalar-energy launch under wp.Tape, then tape.backward with a
              unit seed on every element accumulates dloss/dx into x.grad (the
              idiomatic wp.grad path). O(C) work, plus a backward launch and tape.

    jet     : one width-k forward-jet launch. Each element seeds its nodes as a jet
              (nodes x 3 = k variables); the value's k coeffs ARE the element's
              local gradient, scattered (atomic_add) into the vertex gradient. One
              launch, forward and gradient fused, no tape.

One generic wp.func per energy feeds both paths (jet arithmetic is registered as
builtin overloads, so the same vec3 code specializes on jets). The jet launch is
timed on forward+gradient together; reverse is timed on its backward pass ALONE,
so the comparison is conservative. As k grows the jet's O(kC) arithmetic grows,
so its edge over reverse shrinks -- watch the ratio across spring/triangle/tet.

    uv run warp/examples/benchmarks/benchmark_jet_gradient_mesh.py --device cuda:0
    uv run warp/examples/benchmarks/benchmark_jet_gradient_mesh.py --device cuda:0 --graph
"""

import argparse
import time
from typing import Any

import numpy as np

import warp as wp

_REST = wp.constant(1.0)  # spring rest length
_MU = wp.constant(1.0)
_LAM = wp.constant(1.0)

J6 = wp.JetSpace(6)
J9 = wp.JetSpace(9)
J12 = wp.JetSpace(12)


# ---------------------------------------------------------------------------
# Energies -- generic vec3 wp.funcs (specialize on plain vec3 and jet vec3).
# ---------------------------------------------------------------------------


@wp.func
def spring_energy(a: Any, b: Any):
    s = wp.length(a - b) - _REST
    return 0.5 * s * s


@wp.func
def triangle_energy(p0: Any, p1: Any, p2: Any):
    # Reference is a unit right triangle (rest edge matrix = I), so the deformation
    # gradient columns are just the current edges; S = J^T J is 2x2.
    j0 = p1 - p0
    j1 = p2 - p0
    s00 = wp.dot(j0, j0)
    s01 = wp.dot(j0, j1)
    s11 = wp.dot(j1, j1)
    tr = s00 + s11
    det = s00 * s11 - s01 * s01  # squared deformed area, > 0
    return tr + tr / det


@wp.func
def tet_energy(p0: Any, p1: Any, p2: Any, p3: Any):
    f0 = p1 - p0
    f1 = p2 - p0
    f2 = p3 - p0
    i1 = wp.dot(f0, f0) + wp.dot(f1, f1) + wp.dot(f2, f2)
    detf = wp.dot(f0, wp.cross(f1, f2))  # 6 * signed volume, > 0 by construction
    logj = wp.log(detf)
    return 0.5 * _MU * (i1 - 3.0) - _MU * logj + 0.5 * _LAM * logj * logj


# ---------------------------------------------------------------------------
# Per-element kernels: jet (scatter local gradient) + scalar (for reverse tape).
# ---------------------------------------------------------------------------


@wp.kernel
def spring_jet(x: wp.array[wp.vec3], elem: wp.array[wp.vec2i], g: wp.array[wp.vec3]):
    t = wp.tid()
    e = elem[t]
    c = spring_energy(J6.seed_vec3(x[e[0]], 0, 1, 2), J6.seed_vec3(x[e[1]], 3, 4, 5)).coeff
    wp.atomic_add(g, e[0], wp.vec3(c[0], c[1], c[2]))
    wp.atomic_add(g, e[1], wp.vec3(c[3], c[4], c[5]))


@wp.kernel
def spring_scalar(x: wp.array[wp.vec3], elem: wp.array[wp.vec2i], en: wp.array[wp.float32]):
    t = wp.tid()
    e = elem[t]
    en[t] = spring_energy(x[e[0]], x[e[1]])


@wp.kernel
def triangle_jet(x: wp.array[wp.vec3], elem: wp.array[wp.vec3i], g: wp.array[wp.vec3]):
    t = wp.tid()
    e = elem[t]
    c = triangle_energy(
        J9.seed_vec3(x[e[0]], 0, 1, 2),
        J9.seed_vec3(x[e[1]], 3, 4, 5),
        J9.seed_vec3(x[e[2]], 6, 7, 8),
    ).coeff
    wp.atomic_add(g, e[0], wp.vec3(c[0], c[1], c[2]))
    wp.atomic_add(g, e[1], wp.vec3(c[3], c[4], c[5]))
    wp.atomic_add(g, e[2], wp.vec3(c[6], c[7], c[8]))


@wp.kernel
def triangle_scalar(x: wp.array[wp.vec3], elem: wp.array[wp.vec3i], en: wp.array[wp.float32]):
    t = wp.tid()
    e = elem[t]
    en[t] = triangle_energy(x[e[0]], x[e[1]], x[e[2]])


@wp.kernel
def tet_jet(x: wp.array[wp.vec3], elem: wp.array[wp.vec4i], g: wp.array[wp.vec3]):
    t = wp.tid()
    e = elem[t]
    c = tet_energy(
        J12.seed_vec3(x[e[0]], 0, 1, 2),
        J12.seed_vec3(x[e[1]], 3, 4, 5),
        J12.seed_vec3(x[e[2]], 6, 7, 8),
        J12.seed_vec3(x[e[3]], 9, 10, 11),
    ).coeff
    wp.atomic_add(g, e[0], wp.vec3(c[0], c[1], c[2]))
    wp.atomic_add(g, e[1], wp.vec3(c[3], c[4], c[5]))
    wp.atomic_add(g, e[2], wp.vec3(c[6], c[7], c[8]))
    wp.atomic_add(g, e[3], wp.vec3(c[9], c[10], c[11]))


@wp.kernel
def tet_scalar(x: wp.array[wp.vec3], elem: wp.array[wp.vec4i], en: wp.array[wp.float32]):
    t = wp.tid()
    e = elem[t]
    en[t] = tet_energy(x[e[0]], x[e[1]], x[e[2]], x[e[3]])


# ---------------------------------------------------------------------------
# Shared-vertex meshes: positions (n, 3) float32 and elements (m, nodes) int32.
# ---------------------------------------------------------------------------


def chain_mesh(m, seed=0):
    n = m + 1
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = np.arange(n)  # unit spacing -> rest length 1
    pos += 0.2 * rng.standard_normal((n, 3)).astype(np.float32)
    elem = np.stack([np.arange(m), np.arange(1, n)], axis=1).astype(np.int32)
    return pos, elem


def triangle_mesh(m, seed=0):
    cells = max(1, m // 2)
    side = max(1, int(np.ceil(np.sqrt(cells))))
    gx, gy = np.meshgrid(np.arange(side + 1), np.arange(side + 1), indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    pos += 0.15 * rng.standard_normal(pos.shape).astype(np.float32)
    w = side + 1

    def vid(i, j):
        return i * w + j

    tris = []
    for i in range(side):
        for j in range(side):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            tris.append((a, b, c))
            tris.append((b, d, c))
    return pos, np.array(tris, np.int32)


# Kuhn (Freudenthal) decomposition of a cube into 6 tets sharing the main diagonal.
_CUBE_TETS = [
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
    (0, 2, 6, 7),
    (0, 2, 3, 7),
]


def tet_mesh(m, seed=0):
    side = max(1, int(np.ceil((m / 6.0) ** (1.0 / 3.0))))
    w = side + 1
    gx, gy, gz = np.meshgrid(np.arange(w), np.arange(w), np.arange(w), indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    pos += 0.1 * rng.standard_normal(pos.shape).astype(np.float32)

    def vid(i, j, k):
        return (i * w + j) * w + k

    corners = lambda i, j, k: [  # noqa: E731  cube corners in bit order (x,y,z)
        vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)
    ]
    tets = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                cs = corners(i, j, k)
                for a, b, c, d in _CUBE_TETS:
                    tets.append((cs[a], cs[b], cs[c], cs[d]))
    tets = np.array(tets, np.int32)
    # Ensure positive rest volume so det(F) > 0 (neo-Hookean log is defined).
    p = pos.astype(np.float64)
    f0 = p[tets[:, 1]] - p[tets[:, 0]]
    f1 = p[tets[:, 2]] - p[tets[:, 0]]
    f2 = p[tets[:, 3]] - p[tets[:, 0]]
    neg = np.einsum("ij,ij->i", f0, np.cross(f1, f2)) < 0.0
    tets[neg, 2], tets[neg, 3] = tets[neg, 3], tets[neg, 2].copy()
    return pos, tets


# element -> (k, nodes, jet_kernel, scalar_kernel, elem_dtype, mesh_fn)
ELEMENTS = {
    "spring": (6, 2, spring_jet, spring_scalar, wp.vec2i, chain_mesh),
    "triangle": (9, 3, triangle_jet, triangle_scalar, wp.vec3i, triangle_mesh),
    "tet": (12, 4, tet_jet, tet_scalar, wp.vec4i, tet_mesh),
}


class Reverse:
    def __init__(self, spec, pos, elem_np, device):
        _, _, _, scalar_kernel, elem_dtype, _ = spec
        self.m, self.device = elem_np.shape[0], device
        self.x = wp.array(pos, dtype=wp.vec3, device=device, requires_grad=True)
        self.elem = wp.array(elem_np, dtype=elem_dtype, device=device)
        self.en = wp.zeros(self.m, dtype=wp.float32, device=device, requires_grad=True)
        self.ones = wp.ones(self.m, dtype=wp.float32, device=device)
        self.tape = wp.Tape()
        with self.tape:
            wp.launch(scalar_kernel, dim=self.m, inputs=[self.x, self.elem], outputs=[self.en])

    def run(self):
        self.x.grad.zero_()
        self.tape.backward(grads={self.en: self.ones})

    def grad(self):
        return self.x.grad.numpy()


class Jet:
    def __init__(self, spec, pos, elem_np, device):
        _, _, jet_kernel, _, elem_dtype, _ = spec
        self.m, self.device, self.kernel = elem_np.shape[0], device, jet_kernel
        self.x = wp.array(pos, dtype=wp.vec3, device=device)
        self.elem = wp.array(elem_np, dtype=elem_dtype, device=device)
        self.g = wp.zeros(pos.shape[0], dtype=wp.vec3, device=device)

    def run(self):
        self.g.zero_()
        wp.launch(self.kernel, dim=self.m, inputs=[self.x, self.elem], outputs=[self.g])

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
    parser.add_argument("--element", type=str, default=None, choices=list(ELEMENTS), help="Default: all.")
    parser.add_argument("--m", type=int, nargs="+", default=[1_000_000], help="Approx element counts.")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--graph", action="store_true", help="Time warm graph-captured replay.")
    args = parser.parse_args()

    wp.init()
    device = args.device
    mode = "graph replay (warm)" if args.graph else "per-launch"
    print(f"device: {wp.get_device(device)}   reps: {args.reps}   timing: {mode}\n")

    header = f"{'element':>9} {'k':>3} {'elements':>10} {'jet':>9} {'reverse':>9} {'jet/rev':>9} {'rel-err':>9}"
    print(header)
    print("-" * len(header))
    print("(jet = forward+grad in one launch; reverse timed on backward alone)")

    for name in [args.element] if args.element else list(ELEMENTS):
        spec = ELEMENTS[name]
        k, _, _, _, _, mesh_fn = spec
        for m in args.m:
            pos, elem_np = mesh_fn(m)
            jet = Jet(spec, pos, elem_np, device)
            rev = Reverse(spec, pos, elem_np, device)

            jet.run()
            rev.run()
            wp.synchronize_device(device)
            gj, gr = jet.grad(), rev.grad()
            # Relative error: absolute is dominated by rare near-degenerate elements
            # (tiny det -> gradient ~1e6), where float32 forward vs reverse legitimately
            # differ in the last bits though both match double-precision FD.
            err = np.abs(gj - gr).max() / max(np.abs(gr).max(), 1.0)

            tj = time_strategy(jet, device, args.reps, args.graph)
            tr = time_strategy(rev, device, args.reps, args.graph)
            print(
                f"{name:>9} {k:>3} {elem_np.shape[0]:>10} {tj * 1e3:>9.3f} {tr * 1e3:>9.3f}"
                f" {tj / tr:>8.2f}x {err:>9.1e}"
            )


if __name__ == "__main__":
    main()
