# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy/SciPy host reference for the inverse-elasticity shape-optimization example.

This is a small, dependency-light (numpy + scipy) reimplementation of the 2D
constant-strain-triangle linear-elasticity forward model and its sparse
sensitivities, used **only** as a correctness oracle for the Warp example in
``example_inverse_elasticity.py``. It mirrors the C++ reference
https://github.com/alecjacobson/gauss-newton-sensitivity-analysis
(``elasticity.h``) closely enough to cross-check formulas, and validates its own
gradient / Gauss-Newton steps against dense finite differences.

Everything is double precision. Degrees of freedom use the interleaved
per-vertex ordering ``[v0x, v0y, v1x, v1y, ...]`` throughout, matching the Warp
example. Not an importable public API and not an example itself (underscore
prefix keeps it out of the example browser); imported by the test module.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- Local ordering ---------------------------------------------------------
# Displacement (and geometry) dofs of a triangle are interleaved:
#   [u0x, u0y, u1x, u1y, u2x, u2y]
# matching local_stiffness/local_mass in the C++ reference.

_M = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])  # 2x3, D_m = _M @ Vf


def constitutive(young: float, poisson: float) -> np.ndarray:
    """Plane-strain 3x3 constitutive matrix C (engineering strain)."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return np.array(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ]
    )


def _B_and_area(Vf: np.ndarray):
    """Strain-displacement matrix B (3x6) and triangle area for rest coords Vf (3x2)."""
    D_m = _M @ Vf  # 2x2
    G = np.linalg.inv(D_m) @ _M  # 2x3
    B = np.array(
        [
            [G[0, 0], 0.0, G[0, 1], 0.0, G[0, 2], 0.0],
            [0.0, G[1, 0], 0.0, G[1, 1], 0.0, G[1, 2]],
            [G[1, 0], G[0, 0], G[1, 1], G[0, 1], G[1, 2], G[0, 2]],
        ]
    )
    area = abs(np.linalg.det(D_m)) / 2.0
    return B, area


def local_stiffness(Vf: np.ndarray, young: float, poisson: float) -> np.ndarray:
    """Element stiffness K_e (6x6), plane strain, K_e = area * B^T C B."""
    B, area = _B_and_area(Vf)
    C = constitutive(young, poisson)
    return area * (B.T @ C @ B)


def local_mass(Vf: np.ndarray) -> np.ndarray:
    """Element lumped mass M_e (6x6): area/3 on each of the 6 diagonal entries."""
    _, area = _B_and_area(Vf)
    return np.diag(np.full(6, area / 3.0))


def dof_map(tri: np.ndarray) -> np.ndarray:
    """Global interleaved dofs for a triangle's three vertices."""
    return np.array([tri[0] * 2, tri[0] * 2 + 1, tri[1] * 2, tri[1] * 2 + 1, tri[2] * 2, tri[2] * 2 + 1])


# --- Mesh -------------------------------------------------------------------


def triangulated_grid(nx: int, ny: int):
    """Regular triangulated grid on [0,1]^2, matching igl::triangulated_grid ordering.

    Returns ``(V, F)``: ``V`` is ``(nx*ny, 2)`` with vertex ``j*nx + i`` at
    ``(i/(nx-1), j/(ny-1))``; ``F`` is ``(2*(nx-1)*(ny-1), 3)`` counterclockwise.
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    gx, gy = np.meshgrid(xs, ys)  # (ny, nx)
    V = np.stack([gx.ravel(), gy.ravel()], axis=1)  # row j*nx+i
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            tris.append([idx, idx + 1, idx + nx + 1])
            tris.append([idx, idx + nx + 1, idx + nx])
    return V, np.array(tris, dtype=np.int32)


class Problem:
    """Bridge problem: 15:1 grid pinned left/right, sagging under gravity."""

    def __init__(self, count: int, young: float = 2e3, poisson: float = 0.49):
        ny = 1 + count
        nx = 15 * (ny - 1) + 1
        V, F = triangulated_grid(nx, ny)
        V = V.copy()
        V[:, 0] *= (nx - 1) / (ny - 1)  # stretch to 15:1
        self.V = V
        self.F = F
        self.young = young
        self.poisson = poisson
        self.V_target = V.copy()
        self.f_ext = np.zeros_like(V)
        self.f_ext[:, 1] = -9.8

        x_min, x_max = V[:, 0].min(), V[:, 0].max()
        fixed, free = [], []
        for vi in range(V.shape[0]):
            if abs(V[vi, 0] - x_min) < 1e-8 or abs(V[vi, 0] - x_max) < 1e-8:
                fixed.append(vi)
            else:
                free.append(vi)
        self.fixed_vertices = np.array(fixed, dtype=np.int32)
        self.free_vertices = np.array(free, dtype=np.int32)
        self.free_dofs = free_dof_indices(self.free_vertices)


def free_dof_indices(free_vertices: np.ndarray) -> np.ndarray:
    fv = np.asarray(free_vertices)
    out = np.empty(fv.size * 2, dtype=np.int64)
    out[0::2] = fv * 2
    out[1::2] = fv * 2 + 1
    return out


# --- Assembly & forward solve -----------------------------------------------


def assemble_KM(V, F, young, poisson):
    """Assemble global sparse stiffness K and lumped mass M (2#V x 2#V)."""
    n = V.shape[0] * 2
    rows, cols, kvals, mvals = [], [], [], []
    for tri in F:
        Vf = V[tri]
        Ke = local_stiffness(Vf, young, poisson)
        Me = local_mass(Vf)
        dofs = dof_map(tri)
        for i in range(6):
            for j in range(6):
                rows.append(dofs[i])
                cols.append(dofs[j])
                kvals.append(Ke[i, j])
                mvals.append(Me[i, j])
    K = sp.csr_matrix((kvals, (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((mvals, (rows, cols)), shape=(n, n))
    return K, M


def forward_sim(V, F, young, poisson, f_ext, free_dofs):
    """Solve K u = M f_ext with zero Dirichlet on fixed dofs. Returns (U, u, K, M)."""
    K, M = assemble_KM(V, F, young, poisson)
    ell = M @ f_ext.reshape(-1)  # load
    A = K[np.ix_(free_dofs, free_dofs)].tocsc()
    q = spla.spsolve(A, ell[free_dofs])
    u = np.zeros(V.shape[0] * 2)
    u[free_dofs] = q
    U = V + u.reshape(-1, 2)
    return U, u, K, M


def loss(V, F, young, poisson, f_ext, free_dofs, V_target):
    U, _, _, _ = forward_sim(V, F, young, poisson, f_ext, free_dofs)
    return float(np.mean((V_target - U) ** 2))


# --- Sensitivity matrix G ---------------------------------------------------


def assemble_G(V, F, young, poisson, u, f_ext, eps=1e-6):
    """Assemble sparse G (2#V x 2#V), G_e(:,a) = dM_e/dx_a f_e - dK_e/dx_a u_e.

    Element derivatives via central finite differences over the 6 local geometry
    coordinates (the reference's "six broadcast finite-difference directions"),
    with u and f_ext held fixed.
    """
    n = V.shape[0] * 2
    f_flat = f_ext.reshape(-1)
    rows, cols, vals = [], [], []
    for tri in F:
        Vf = V[tri]
        dofs = dof_map(tri)
        u_e = u[dofs]
        f_e = f_flat[dofs]

        def h(Vf_local, u_e=u_e, f_e=f_e):
            return local_mass(Vf_local) @ f_e - local_stiffness(Vf_local, young, poisson) @ u_e

        for a in range(6):
            Vf_p = Vf.copy()
            Vf_m = Vf.copy()
            Vf_p[a // 2, a % 2] += eps
            Vf_m[a // 2, a % 2] -= eps
            col = (h(Vf_p) - h(Vf_m)) / (2.0 * eps)
            for i in range(6):
                rows.append(dofs[i])
                cols.append(dofs[a])
                vals.append(col[i])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# --- Sparse gradient / Gauss-Newton steps (the methods being validated) -----


def gradient_step(V, F, young, poisson, f_ext, free_dofs, V_target):
    """Ascent gradient of loss=mean((V_target-U)^2) via one adjoint solve. Returns dV (#V,2)."""
    U, u, K, _ = forward_sim(V, F, young, poisson, f_ext, free_dofs)
    r = (V_target - U).reshape(-1)
    G = assemble_G(V, F, young, poisson, u, f_ext)
    A = K[np.ix_(free_dofs, free_dofs)].tocsc()
    Gff = G[np.ix_(free_dofs, free_dofs)]
    r_free = r[free_dofs]
    lam = spla.spsolve(A, r_free)
    raw = -(r_free + Gff.T @ lam)
    N = V.shape[0] * 2
    grad_free = (2.0 / N) * raw
    dV = np.zeros_like(V)
    dV.reshape(-1)[free_dofs] = grad_free
    return dV


def gauss_newton_step(V, F, young, poisson, f_ext, free_dofs, V_target):
    """Sparse Gauss-Newton step via the square route: T = A + Gff, T w = Gff r, p = r - w."""
    U, u, K, _ = forward_sim(V, F, young, poisson, f_ext, free_dofs)
    r = (V_target - U).reshape(-1)
    G = assemble_G(V, F, young, poisson, u, f_ext)
    A = K[np.ix_(free_dofs, free_dofs)].tocsc()
    Gff = G[np.ix_(free_dofs, free_dofs)].tocsc()
    r_free = r[free_dofs]
    T = (A + Gff).tocsc()
    w = spla.spsolve(T, Gff @ r_free)
    p = r_free - w
    dV = np.zeros_like(V)
    dV.reshape(-1)[free_dofs] = p
    return dV


def gauss_newton_step_kkt(V, F, young, poisson, f_ext, free_dofs, V_target):
    """Same GN step via the full sparse KKT saddle system in (p, w, eta)."""
    U, u, K, _ = forward_sim(V, F, young, poisson, f_ext, free_dofs)
    r = (V_target - U).reshape(-1)
    G = assemble_G(V, F, young, poisson, u, f_ext)
    A = K[np.ix_(free_dofs, free_dofs)].tocsc()
    Gff = G[np.ix_(free_dofs, free_dofs)].tocsc()
    r_free = r[free_dofs]
    m = free_dofs.size
    I = sp.identity(m, format="csc")
    KKT = sp.bmat(
        [
            [I, I, -Gff.T],
            [I, I, A.T],
            [-Gff, A, None],
        ],
        format="csc",
    )
    rhs = np.concatenate([r_free, r_free, np.zeros(m)])
    sol = spla.spsolve(KKT, rhs)
    p = sol[:m]
    dV = np.zeros_like(V)
    dV.reshape(-1)[free_dofs] = p
    return dV


# --- Dense finite-difference oracles ----------------------------------------


def fd_gradient_step(V, F, young, poisson, f_ext, free_dofs, V_target, eps=1e-6):
    """Dense central-difference ascent gradient of the scalar loss."""
    dV = np.zeros_like(V)
    for d in free_dofs:
        vi, vc = d // 2, d % 2
        Vp = V.copy()
        Vm = V.copy()
        Vp[vi, vc] += eps
        Vm[vi, vc] -= eps
        lp = loss(Vp, F, young, poisson, f_ext, free_dofs, V_target)
        lm = loss(Vm, F, young, poisson, f_ext, free_dofs, V_target)
        dV[vi, vc] = (lp - lm) / (2.0 * eps)
    return dV


def fd_gauss_newton_step(V, F, young, poisson, f_ext, free_dofs, V_target, eps=1e-6):
    """Dense-FD Gauss-Newton step: build J = d(V_target-U)/dx over free dofs, solve normal eqs."""
    n = V.shape[0] * 2
    m = free_dofs.size
    U0, _, _, _ = forward_sim(V, F, young, poisson, f_ext, free_dofs)
    r0 = (V_target - U0).reshape(-1)
    J = np.zeros((n, m))
    for k, d in enumerate(free_dofs):
        vi, vc = d // 2, d % 2
        Vp = V.copy()
        Vm = V.copy()
        Vp[vi, vc] += eps
        Vm[vi, vc] -= eps
        Up, _, _, _ = forward_sim(Vp, F, young, poisson, f_ext, free_dofs)
        Um, _, _, _ = forward_sim(Vm, F, young, poisson, f_ext, free_dofs)
        J[:, k] = ((V_target - Up).reshape(-1) - (V_target - Um).reshape(-1)) / (2.0 * eps)
    Hff = J.T @ J
    p = -np.linalg.solve(Hff, J.T @ r0)
    dV = np.zeros_like(V)
    dV.reshape(-1)[free_dofs] = p
    return dV
