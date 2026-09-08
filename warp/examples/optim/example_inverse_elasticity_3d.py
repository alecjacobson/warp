# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inverse elasticity shape optimization (3D tetrahedral mesh)
#
# The 3D analogue of example_inverse_elasticity.py: optimizes the rest shape of a
# pinned 3D elastic bar (15:3:1 box, tetrahedralized) so that, once it sags under
# gravity, its deformed shape matches a flat target. Linear tetrahedra, isotropic
# 3D linear elasticity; the Gauss-Newton "square route" (T = A + G_ff) is solved
# directly on the GPU with cuDSS. There is no C++ reference for this variant, so
# correctness rests on implementation-independent checks (rigid-body modes, patch
# test, eigenvalue count), finite-difference gradient/step checks, and quadratic
# Gauss-Newton convergence.
#
# Requires the cuDSS sparse direct solver via warp-cuDSS (see the 2D example).

import numpy as np

import warp as wp

scalar = wp.float64
vec3 = wp.types.vector(3, scalar)
vec12 = wp.types.vector(12, scalar)
mat33 = wp.types.matrix((3, 3), scalar)
mat34 = wp.types.matrix((3, 4), scalar)
mat66 = wp.types.matrix((6, 6), scalar)
mat6_12 = wp.types.matrix((6, 12), scalar)
mat12 = wp.types.matrix((12, 12), scalar)


# ---------------------------------------------------------------------------
# Element operators (linear tetrahedron, isotropic 3D elasticity)
# ---------------------------------------------------------------------------


@wp.func
def shape_gradients(p0: vec3, p1: vec3, p2: vec3, p3: vec3):
    """Return (G, vol): G[i,j] = dN_j/dx_i (3x4) and the tet volume."""
    Dm = mat33(p0[0] - p3[0], p0[1] - p3[1], p0[2] - p3[2],
               p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2],
               p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2])  # fmt: skip
    Mg = mat34(scalar(1.0), scalar(0.0), scalar(0.0), scalar(-1.0),
               scalar(0.0), scalar(1.0), scalar(0.0), scalar(-1.0),
               scalar(0.0), scalar(0.0), scalar(1.0), scalar(-1.0))  # fmt: skip
    G = wp.inverse(Dm) * Mg
    vol = wp.abs(wp.determinant(Dm)) / scalar(6.0)
    return G, vol


@wp.func
def strain_matrix(G: mat34) -> mat6_12:
    """Engineering strain-displacement matrix B (6x12), dofs [u0x,u0y,u0z,...,u3z]."""
    B = mat6_12(scalar(0.0))
    for j in range(4):
        B[0, 3 * j + 0] = G[0, j]
        B[1, 3 * j + 1] = G[1, j]
        B[2, 3 * j + 2] = G[2, j]
        B[3, 3 * j + 0] = G[1, j]
        B[3, 3 * j + 1] = G[0, j]
        B[4, 3 * j + 1] = G[2, j]
        B[4, 3 * j + 2] = G[1, j]
        B[5, 3 * j + 0] = G[2, j]
        B[5, 3 * j + 2] = G[0, j]
    return B


@wp.func
def constitutive(young: scalar, poisson: scalar) -> mat66:
    """Isotropic 3D constitutive matrix C (6x6, engineering strain)."""
    lam = young * poisson / ((scalar(1.0) + poisson) * (scalar(1.0) - scalar(2.0) * poisson))
    mu = young / (scalar(2.0) * (scalar(1.0) + poisson))
    d = lam + scalar(2.0) * mu
    return mat66(
        d, lam, lam, scalar(0.0), scalar(0.0), scalar(0.0),
        lam, d, lam, scalar(0.0), scalar(0.0), scalar(0.0),
        lam, lam, d, scalar(0.0), scalar(0.0), scalar(0.0),
        scalar(0.0), scalar(0.0), scalar(0.0), mu, scalar(0.0), scalar(0.0),
        scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), mu, scalar(0.0),
        scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), scalar(0.0), mu,
    )  # fmt: skip


@wp.func
def element_stiffness(p0: vec3, p1: vec3, p2: vec3, p3: vec3, young: scalar, poisson: scalar) -> mat12:
    """12x12 tet stiffness ``K_e = vol * B^T C B``."""
    G, vol = shape_gradients(p0, p1, p2, p3)
    B = strain_matrix(G)
    C = constitutive(young, poisson)
    return vol * (wp.transpose(B) * C * B)


@wp.func
def element_volume(p0: vec3, p1: vec3, p2: vec3, p3: vec3) -> scalar:
    Dm = mat33(p0[0] - p3[0], p0[1] - p3[1], p0[2] - p3[2],
               p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2],
               p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2])  # fmt: skip
    return wp.abs(wp.determinant(Dm)) / scalar(6.0)


# ---------------------------------------------------------------------------
# Mesh: a 15:3:1 box, structured grid split into 6 tets per hex cell.
# ---------------------------------------------------------------------------


def make_box(count):
    """15:3:1 tetrahedralized box, pinned at its left/right (x=min/max) faces."""
    nz = 1 + count
    nx = 15 * (nz - 1) + 1
    ny = 3 * (nz - 1) + 1
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    V = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    V[:, 0] *= (nx - 1) / (nz - 1)  # stretch to 15 : 3 : 1
    V[:, 1] *= (ny - 1) / (nz - 1)

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    # 6-tet (Kuhn) split of each hex cell, all sharing the c0-c7 body diagonal.
    local = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [vid(i + dx, j + dy, k + dz) for dx, dy, dz in corners]
                for a, b, cc, d in local:
                    tets.append([c[a], c[b], c[cc], c[d]])
    T = np.array(tets, dtype=np.int32)
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    fixed = np.array([i for i in range(V.shape[0]) if abs(V[i, 0] - xmin) < 1e-8 or abs(V[i, 0] - xmax) < 1e-8],
                     dtype=np.int32)  # fmt: skip
    return V.astype(np.float64), T, fixed
