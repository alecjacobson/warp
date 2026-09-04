# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Inverse Elasticity Shape Optimization (sparse Gauss-Newton)
#
# A wide 2D "bridge" is pinned at its left and right edges and sags under
# gravity. We optimize the *rest* shape so that once it sags, the deformed
# shape matches a flat target -- the rest shape arches upward so gravity pulls
# it flat. This is a pure-Warp port of
#   https://github.com/alecjacobson/gauss-newton-sensitivity-analysis
#
# The forward model is 2D linear elasticity on constant-strain triangles. The
# design variables are the rest vertex positions x = vec(V); the state is the
# gravity displacement u(x) from K(x) u = M(x) f_ext with pinned (Dirichlet)
# vertices. The loss is mean((V_target - (V + u))^2).
#
# Plain gradient descent fails on this problem (it collapses triangles), so we
# feature a sparse Gauss-Newton step computed without ever forming a dense
# Jacobian: T = A + G_ff (A = K_ff, G the geometry sensitivity of the load
# minus stiffness*displacement), solved with Warp's GMRES because T is sparse
# but nonsymmetric. See design/inverse-elasticity-shape-optimization.md.
###########################################################################

import warp as wp

# Double precision throughout, matching the reference and keeping the sparse
# solves well conditioned. Degrees of freedom are interleaved per vertex:
# [v0x, v0y, v1x, v1y, ...].
vec2d = wp.vec2d
vec3d = wp.vec3d
vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat22d = wp.mat22d
mat33d = wp.mat33d
mat36d = wp.types.matrix(shape=(3, 6), dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


@wp.func
def constitutive(young: wp.float64, poisson: wp.float64) -> mat33d:
    """Plane-strain constitutive matrix C for engineering strain [exx, eyy, 2exy]."""
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    lam = young * poisson / ((one + poisson) * (one - two * poisson))
    mu = young / (two * (one + poisson))
    z = wp.float64(0.0)
    return mat33d(
        lam + two * mu, lam, z,
        lam, lam + two * mu, z,
        z, z, mu,
    )  # fmt: skip


@wp.func
def _strain_matrix(v0: vec2d, v1: vec2d, v2: vec2d):
    """Return (B, area): strain-displacement matrix (3x6) and triangle area.

    ``D_m`` has rows ``v0 - v2`` and ``v1 - v2``; ``G = D_m^{-1} M`` with
    ``M = [[1,0,-1],[0,1,-1]]`` gives the shape-function gradients, and B lays
    them out for the interleaved displacement ordering.
    """
    d0 = v0 - v2
    d1 = v1 - v2
    Dm = mat22d(d0[0], d0[1], d1[0], d1[1])
    Di = wp.inverse(Dm)

    # G = Di @ M: columns 0,1 are the columns of Di; column 2 is -(col0 + col1).
    g00 = Di[0, 0]
    g01 = Di[0, 1]
    g02 = -(Di[0, 0] + Di[0, 1])
    g10 = Di[1, 0]
    g11 = Di[1, 1]
    g12 = -(Di[1, 0] + Di[1, 1])

    z = wp.float64(0.0)
    B = mat36d(
        g00, z, g01, z, g02, z,
        z, g10, z, g11, z, g12,
        g10, g00, g11, g01, g12, g02,
    )  # fmt: skip
    area = wp.abs(wp.determinant(Dm)) / wp.float64(2.0)
    return B, area


@wp.func
def local_stiffness(v0: vec2d, v1: vec2d, v2: vec2d, young: wp.float64, poisson: wp.float64) -> mat66d:
    """Element stiffness K_e (6x6), plane strain: K_e = area * B^T C B."""
    B, area = _strain_matrix(v0, v1, v2)
    C = constitutive(young, poisson)
    return area * (wp.transpose(B) * C * B)


@wp.func
def local_mass(v0: vec2d, v1: vec2d, v2: vec2d) -> mat66d:
    """Element lumped mass M_e (6x6): area/3 on each of the 6 diagonal entries."""
    _, area = _strain_matrix(v0, v1, v2)
    a3 = area / wp.float64(3.0)
    M = mat66d(wp.float64(0.0))
    for i in range(6):
        M[i, i] = a3
    return M
