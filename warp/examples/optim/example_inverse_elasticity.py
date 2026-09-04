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

import numpy as np

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

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


# ---------------------------------------------------------------------------
# Global assembly and forward elastostatic solve
# ---------------------------------------------------------------------------
#
# Pinning is whole-vertex (both x and y are fixed), so the free degrees of
# freedom group cleanly into per-free-vertex 2x2 blocks. The stiffness matrix
# restricted to free vertices, A = K_ff, is assembled as a block-sparse (BSR)
# matrix of 2x2 blocks; displacement/load vectors are arrays of vec2d indexed
# by free-vertex index.


@wp.func
def _sub_block(K: mat66d, i: wp.int32, j: wp.int32) -> mat22d:
    """Extract the 2x2 block coupling local vertices i and j from a 6x6 element matrix."""
    return mat22d(K[2 * i, 2 * j], K[2 * i, 2 * j + 1], K[2 * i + 1, 2 * j], K[2 * i + 1, 2 * j + 1])


@wp.kernel
def _assemble_stiffness_triplets(
    tris: wp.array(dtype=wp.vec3i),
    verts: wp.array(dtype=vec2d),
    young: wp.float64,
    poisson: wp.float64,
    vert_to_free: wp.array(dtype=wp.int32),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=mat22d),
):
    t = wp.tid()
    f = tris[t]
    Ke = local_stiffness(verts[f[0]], verts[f[1]], verts[f[2]], young, poisson)
    for i in range(3):
        fi = vert_to_free[f[i]]
        for j in range(3):
            fj = vert_to_free[f[j]]
            slot = t * 9 + i * 3 + j
            # Only free-free couplings enter A; fixed vertices (u = 0) drop out.
            # Off-target slots are written as an explicit zero block at (0,0),
            # which bsr_from_triplets sums harmlessly.
            if fi >= 0 and fj >= 0:
                rows[slot] = fi
                cols[slot] = fj
                blocks[slot] = _sub_block(Ke, i, j)
            else:
                rows[slot] = 0
                cols[slot] = 0
                blocks[slot] = mat22d(wp.float64(0.0))


@wp.kernel
def _scatter_vertex_mass(
    tris: wp.array(dtype=wp.vec3i),
    verts: wp.array(dtype=vec2d),
    mass: wp.array(dtype=wp.float64),
):
    t = wp.tid()
    f = tris[t]
    Me = local_mass(verts[f[0]], verts[f[1]], verts[f[2]])
    for i in range(3):
        wp.atomic_add(mass, f[i], Me[2 * i, 2 * i])  # area/3 lumped mass


@wp.kernel
def _build_free_load(
    mass: wp.array(dtype=wp.float64),
    f_ext: wp.array(dtype=vec2d),
    vert_to_free: wp.array(dtype=wp.int32),
    load_free: wp.array(dtype=vec2d),
):
    v = wp.tid()
    fi = vert_to_free[v]
    if fi >= 0:
        load_free[fi] = mass[v] * f_ext[v]


@wp.kernel
def _scatter_free_to_full(
    q_free: wp.array(dtype=vec2d),
    vert_to_free: wp.array(dtype=wp.int32),
    u_full: wp.array(dtype=vec2d),
):
    v = wp.tid()
    fi = vert_to_free[v]
    if fi >= 0:
        u_full[v] = q_free[fi]
    else:
        u_full[v] = vec2d(wp.float64(0.0), wp.float64(0.0))


@wp.kernel
def _add(a: wp.array(dtype=vec2d), b: wp.array(dtype=vec2d), out: wp.array(dtype=vec2d)):
    v = wp.tid()
    out[v] = a[v] + b[v]


class BridgeProblem:
    """Device-side state and operators for the inverse-elasticity bridge problem.

    Holds the (fixed) topology and material on ``device`` and assembles the
    free-vertex stiffness ``A = K_ff`` and gravity load as the rest positions
    ``V`` evolve. All arrays are double precision.
    """

    def __init__(self, verts_np, tris_np, free_vertices_np, young, poisson, gravity=-9.8, device=None):
        self.device = wp.get_device(device)
        self.num_verts = int(verts_np.shape[0])
        self.num_tris = int(tris_np.shape[0])
        self.young = wp.float64(young)
        self.poisson = wp.float64(poisson)

        # Map each vertex to its free index, or -1 if pinned.
        vert_to_free = np.full(self.num_verts, -1, dtype=np.int32)
        vert_to_free[free_vertices_np] = np.arange(len(free_vertices_np), dtype=np.int32)
        self.num_free = int(len(free_vertices_np))

        self.verts = wp.array(verts_np, dtype=vec2d, device=self.device)
        self.tris = wp.array(tris_np, dtype=wp.vec3i, device=self.device)
        self.vert_to_free = wp.array(vert_to_free, dtype=wp.int32, device=self.device)

        f_ext_np = np.zeros((self.num_verts, 2), dtype=np.float64)
        f_ext_np[:, 1] = gravity
        self.f_ext = wp.array(f_ext_np, dtype=vec2d, device=self.device)

        # Triplet workspace for stiffness assembly (9 blocks per triangle).
        nt = self.num_tris * 9
        self._rows = wp.empty(nt, dtype=wp.int32, device=self.device)
        self._cols = wp.empty(nt, dtype=wp.int32, device=self.device)
        self._blocks = wp.empty(nt, dtype=mat22d, device=self.device)
        self._mass = wp.zeros(self.num_verts, dtype=wp.float64, device=self.device)

    def assemble_stiffness(self):
        """Assemble and return the free-vertex stiffness A = K_ff as a BSR matrix."""
        wp.launch(
            _assemble_stiffness_triplets,
            dim=self.num_tris,
            inputs=[self.tris, self.verts, self.young, self.poisson, self.vert_to_free,
                    self._rows, self._cols, self._blocks],
            device=self.device,
        )  # fmt: skip
        return wps.bsr_from_triplets(self.num_free, self.num_free, self._rows, self._cols, self._blocks)

    def assemble_load(self):
        """Assemble the gravity load M f_ext restricted to free vertices (array of vec2d)."""
        self._mass.zero_()
        wp.launch(
            _scatter_vertex_mass, dim=self.num_tris, inputs=[self.tris, self.verts, self._mass], device=self.device
        )
        load_free = wp.empty(self.num_free, dtype=vec2d, device=self.device)
        wp.launch(
            _build_free_load,
            dim=self.num_verts,
            inputs=[self._mass, self.f_ext, self.vert_to_free, load_free],
            device=self.device,
        )
        return load_free

    def forward(self, tol=1e-10, maxiter=None):
        """Solve A q = load for the free displacements; return (U, u, q, A).

        ``U`` and ``u`` are per-vertex (vec2d, length #V); ``q`` is per-free-vertex.
        """
        A = self.assemble_stiffness()
        load_free = self.assemble_load()
        q = wp.zeros(self.num_free, dtype=vec2d, device=self.device)
        wpl.cr(A, load_free, q, tol=tol, maxiter=maxiter or self.num_free * 2, M=wpl.preconditioner(A, "diag"))

        u = wp.empty(self.num_verts, dtype=vec2d, device=self.device)
        wp.launch(_scatter_free_to_full, dim=self.num_verts, inputs=[q, self.vert_to_free, u], device=self.device)
        U = wp.empty(self.num_verts, dtype=vec2d, device=self.device)
        wp.launch(_add, dim=self.num_verts, inputs=[self.verts, u, U], device=self.device)
        return U, u, q, A
