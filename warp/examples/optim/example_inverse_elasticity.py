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


# ---------------------------------------------------------------------------
# Geometry sensitivity G and the adjoint gradient
# ---------------------------------------------------------------------------
#
# The sensitivity of the free equilibrium to a rest-shape change is captured by
# G, defined column-wise by G(:,a) = dM/dx_a f_ext - dK/dx_a u (u, f_ext held
# fixed). Element-locally, G_e is a 6x6 matrix whose column a is the derivative
# of h(V_e) = M_e f_e - K_e u_e with respect to local geometry coordinate a. We
# evaluate those derivatives by central differences over the six local
# coordinates (the reference's "six broadcast finite-difference directions"),
# fully in parallel over elements, and scatter the free-free 2x2 blocks into a
# BSR matrix G_ff.


@wp.func
def _elem_h(v0: vec2d, v1: vec2d, v2: vec2d, young: wp.float64, poisson: wp.float64, u_e: vec6d, f_e: vec6d) -> vec6d:
    """Element load-minus-restoring-force vector h(V_e) = M_e f_e - K_e u_e."""
    return local_mass(v0, v1, v2) * f_e - local_stiffness(v0, v1, v2, young, poisson) * u_e


@wp.kernel
def _assemble_G_triplets(
    tris: wp.array(dtype=wp.vec3i),
    verts: wp.array(dtype=vec2d),
    young: wp.float64,
    poisson: wp.float64,
    u: wp.array(dtype=vec2d),
    f_ext: wp.array(dtype=vec2d),
    vert_to_free: wp.array(dtype=wp.int32),
    eps: wp.float64,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=mat22d),
):
    t = wp.tid()
    f = tris[t]
    p0 = verts[f[0]]
    p1 = verts[f[1]]
    p2 = verts[f[2]]
    u_e = vec6d(u[f[0]][0], u[f[0]][1], u[f[1]][0], u[f[1]][1], u[f[2]][0], u[f[2]][1])
    f_e = vec6d(f_ext[f[0]][0], f_ext[f[0]][1], f_ext[f[1]][0], f_ext[f[1]][1], f_ext[f[2]][0], f_ext[f[2]][1])

    Ge = mat66d(wp.float64(0.0))
    for g in range(6):
        vtx = g // 2
        comp = g % 2
        d = vec2d(wp.where(comp == 0, eps, wp.float64(0.0)), wp.where(comp == 1, eps, wp.float64(0.0)))
        # Perturb only the g-th local coordinate (vtx is constant per unrolled iter).
        pp0 = p0 + wp.where(vtx == 0, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        pp1 = p1 + wp.where(vtx == 1, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        pp2 = p2 + wp.where(vtx == 2, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        pm0 = p0 - wp.where(vtx == 0, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        pm1 = p1 - wp.where(vtx == 1, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        pm2 = p2 - wp.where(vtx == 2, d, vec2d(wp.float64(0.0), wp.float64(0.0)))
        col = (_elem_h(pp0, pp1, pp2, young, poisson, u_e, f_e) - _elem_h(pm0, pm1, pm2, young, poisson, u_e, f_e)) / (
            wp.float64(2.0) * eps
        )
        for i in range(6):
            Ge[i, g] = col[i]

    for bi in range(3):
        fi = vert_to_free[f[bi]]
        for bj in range(3):
            fj = vert_to_free[f[bj]]
            slot = t * 9 + bi * 3 + bj
            if fi >= 0 and fj >= 0:
                rows[slot] = fi
                cols[slot] = fj
                blocks[slot] = _sub_block(Ge, bi, bj)
            else:
                rows[slot] = 0
                cols[slot] = 0
                blocks[slot] = mat22d(wp.float64(0.0))


@wp.kernel
def _gather_residual_free(
    v_target: wp.array(dtype=vec2d),
    U: wp.array(dtype=vec2d),
    vert_to_free: wp.array(dtype=wp.int32),
    r_free: wp.array(dtype=vec2d),
):
    v = wp.tid()
    fi = vert_to_free[v]
    if fi >= 0:
        r_free[fi] = v_target[v] - U[v]


@wp.kernel
def _grad_combine(
    r_free: wp.array(dtype=vec2d),
    gt_lambda: wp.array(dtype=vec2d),
    scale: wp.float64,
    grad_free: wp.array(dtype=vec2d),
):
    i = wp.tid()
    grad_free[i] = scale * (-(r_free[i] + gt_lambda[i]))


@wp.kernel
def _scatter_free_step(
    step_free: wp.array(dtype=vec2d),
    vert_to_free: wp.array(dtype=wp.int32),
    dV: wp.array(dtype=vec2d),
):
    v = wp.tid()
    fi = vert_to_free[v]
    if fi >= 0:
        dV[v] = step_free[fi]
    else:
        dV[v] = vec2d(wp.float64(0.0), wp.float64(0.0))


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

        # verts is the (mutable) rest shape being optimized; v_target is the flat
        # target the sagged shape should match (the original rest shape).
        self.verts = wp.array(verts_np, dtype=vec2d, device=self.device)
        self.v_target = wp.array(verts_np, dtype=vec2d, device=self.device)
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

    def assemble_Gff(self, u, eps=1e-6):
        """Assemble the free-free geometry-sensitivity matrix G_ff as a BSR matrix."""
        wp.launch(
            _assemble_G_triplets,
            dim=self.num_tris,
            inputs=[self.tris, self.verts, self.young, self.poisson, u, self.f_ext, self.vert_to_free,
                    wp.float64(eps), self._rows, self._cols, self._blocks],
            device=self.device,
        )  # fmt: skip
        return wps.bsr_from_triplets(self.num_free, self.num_free, self._rows, self._cols, self._blocks)

    def residual_free(self, U):
        """Gather the free-vertex residual r = v_target - U (array of vec2d)."""
        r_free = wp.empty(self.num_free, dtype=vec2d, device=self.device)
        wp.launch(
            _gather_residual_free,
            dim=self.num_verts,
            inputs=[self.v_target, U, self.vert_to_free, r_free],
            device=self.device,
        )
        return r_free

    def loss(self, tol=1e-10):
        """Mean squared error mean((v_target - U)^2) at the current rest shape."""
        U, _, _, _ = self.forward(tol=tol)
        diff = (self.v_target.numpy() - U.numpy()).reshape(-1)
        return float(np.mean(diff**2))

    def gradient_step(self, tol=1e-10):
        """Ascent gradient of the loss w.r.t. free rest positions, via one adjoint solve.

        Returns a per-vertex ``dV`` (vec2d, zero at pinned vertices).
        """
        U, u, _, A = self.forward(tol=tol)
        r_free = self.residual_free(U)
        Gff = self.assemble_Gff(u)

        # Adjoint solve A lambda = r_free (A is SPD).
        lam = wp.zeros(self.num_free, dtype=vec2d, device=self.device)
        wpl.cr(A, r_free, lam, tol=tol, maxiter=self.num_free * 2, M=wpl.preconditioner(A, "diag"))

        gt_lambda = wp.zeros(self.num_free, dtype=vec2d, device=self.device)
        wps.bsr_mv(wps.bsr_transposed(Gff), lam, gt_lambda)

        grad_free = wp.empty(self.num_free, dtype=vec2d, device=self.device)
        scale = 2.0 / (self.num_verts * 2)
        wp.launch(
            _grad_combine,
            dim=self.num_free,
            inputs=[r_free, gt_lambda, wp.float64(scale), grad_free],
            device=self.device,
        )

        dV = wp.empty(self.num_verts, dtype=vec2d, device=self.device)
        wp.launch(_scatter_free_step, dim=self.num_verts, inputs=[grad_free, self.vert_to_free, dV], device=self.device)
        return dV
