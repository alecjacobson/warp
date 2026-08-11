# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NumPy reference oracles for the sparse-Hessian (``indexed_sum``) feature.

Pure NumPy, no Warp dependency. These are the ground-truth definitions used to
validate the Warp implementation:

* Tiny summand energies with hand-derived analytic local Hessians.
* A value-only finite-difference Hessian oracle (:func:`fd_hessian`).
* A dense global assembler (:func:`assemble_global_hessian`) realizing
  ``H = Σᵢ Sᵢᵀ Hᵢ Sᵢ`` from per-stencil local Hessians.

Everything runs in float64: second-order finite differences are numerically
hopeless in float32, so the oracle deliberately stays in double precision.
"""

import numpy as np

_I3 = np.eye(3)


def fd_hessian(f, x, eps=1.0e-4):
    """Central-difference Hessian of a scalar function ``f`` of a flat vector.

    Uses the value-only mixed second difference

    ``H[i, j] ≈ [f(x + εeᵢ + εeⱼ) - f(x + εeᵢ - εeⱼ)
                 - f(x - εeᵢ + εeⱼ) + f(x - εeᵢ - εeⱼ)] / (4 ε²)``

    which requires nothing but the forward evaluation, so it doubles as a
    backend-agnostic oracle for every later phase.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    h = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm = x.copy()
            xpm[i] += eps
            xpm[j] -= eps
            xmp = x.copy()
            xmp[i] -= eps
            xmp[j] += eps
            xmm = x.copy()
            xmm[i] -= eps
            xmm[j] -= eps
            h[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps)
            h[j, i] = h[i, j]
    return h


# ---------------------------------------------------------------------------
# Reference summands: value(x_local) -> float and hessian(x_local) -> (n, n)
# Local variables are the concatenation of the stencil's vec3 nodes, so a
# 2-node stencil has x_local of length 6, a 3-node stencil length 9, etc.
# ---------------------------------------------------------------------------


def zero_rest_spring_value(x):
    """``0.5 ‖p0 - p1‖²`` for a zero-rest-length spring."""
    d = x[0:3] - x[3:6]
    return 0.5 * float(d @ d)


def zero_rest_spring_hessian(x):
    """Constant ``[[I, -I], [-I, I]]`` (independent of ``x``)."""
    h = np.zeros((6, 6))
    h[0:3, 0:3] = _I3
    h[3:6, 3:6] = _I3
    h[0:3, 3:6] = -_I3
    h[3:6, 0:3] = -_I3
    return h


def rest_spring_value(x, r):
    """``0.5 (‖p0 - p1‖ - r)²`` for a spring with rest length ``r``."""
    d = x[0:3] - x[3:6]
    l = np.linalg.norm(d)
    return 0.5 * float((l - r) ** 2)


def rest_spring_hessian(x, r):
    """Analytic Hessian of the rest-length spring.

    ``H00 = u uᵀ + (l - r)/l (I - u uᵀ)`` with ``u = (p0 - p1)/l``. Note this is
    indefinite when the spring is compressed (``l < r``): the ``(l - r)/l`` term
    goes negative on the tangential subspace — the regime that motivates PSD
    projection.
    """
    d = x[0:3] - x[3:6]
    l = np.linalg.norm(d)
    u = d / l
    uut = np.outer(u, u)
    h00 = uut + (l - r) / l * (_I3 - uut)
    h = np.zeros((6, 6))
    h[0:3, 0:3] = h00
    h[3:6, 3:6] = h00
    h[0:3, 3:6] = -h00
    h[3:6, 0:3] = -h00
    return h


def inertia_value(x, phat):
    """``0.5 ‖p0 - p̂0‖²``; ``phat`` is a (non-differentiable) constant."""
    d = x[0:3] - phat
    return 0.5 * float(d @ d)


def inertia_hessian(x, phat):
    """Constant identity block ``I`` (per vertex)."""
    return _I3.copy()


def vertex_midpoint_value(x):
    """``0.5 ‖p0 - (p1 + p2)/2‖²`` — a genuinely 3-node-coupled stencil."""
    d = x[0:3] - 0.5 * (x[3:6] + x[6:9])
    return 0.5 * float(d @ d)


def vertex_midpoint_hessian(x):
    """Constant 9x9 Hessian of the vertex-to-edge-midpoint spring."""
    coeffs = {
        (0, 0): 1.0,
        (0, 1): -0.5,
        (0, 2): -0.5,
        (1, 0): -0.5,
        (1, 1): 0.25,
        (1, 2): 0.25,
        (2, 0): -0.5,
        (2, 1): 0.25,
        (2, 2): 0.25,
    }
    h = np.zeros((9, 9))
    for (a, b), c in coeffs.items():
        h[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = c * _I3
    return h


# ---------------------------------------------------------------------------
# Dense global assembly: H = Σᵢ Sᵢᵀ Hᵢ Sᵢ
# ---------------------------------------------------------------------------


def fd_gradient(f, x, eps=1.0e-4):
    """Central-difference gradient of a scalar function ``f`` of a flat vector."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def total_energy(stencils, value_fn, positions):
    """Sum of a summand's ``value_fn`` over all stencils (for FD gradient checks)."""
    positions = np.asarray(positions, dtype=np.float64)
    e = 0.0
    for stencil in stencils:
        x_local = np.concatenate([positions[v] for v in stencil])
        e += value_fn(x_local)
    return e


def assemble_global_hessian(num_verts, stencils, local_hessian_fn, positions):
    """Scatter per-stencil local Hessians into a dense ``(3V, 3V)`` matrix.

    Args:
        num_verts: Number of vec3 vertices ``V`` in the global system.
        stencils: Iterable of index tuples; each tuple lists the global vertex
            indices touched by one summand term, in local order.
        local_hessian_fn: ``x_local -> (3k, 3k)`` analytic Hessian for a
            ``k``-node stencil.
        positions: ``(V, 3)`` array of vertex positions.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = 3 * num_verts
    h = np.zeros((n, n))
    for stencil in stencils:
        x_local = np.concatenate([positions[v] for v in stencil])
        h_local = local_hessian_fn(x_local)
        for a, va in enumerate(stencil):
            for b, vb in enumerate(stencil):
                h[3 * va : 3 * va + 3, 3 * vb : 3 * vb + 3] += h_local[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
    return h
