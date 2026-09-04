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
# minus stiffness*displacement), solved with a Warp Krylov solver (BiCGSTAB)
# because T is sparse but nonsymmetric. Adam (a warp-parallel first-order
# optimizer) also converges, just far more slowly, giving a nice CPU/GPU
# contrast. See design/inverse-elasticity-shape-optimization.md.
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
def _scatter_stiffness_inplace(
    tris: wp.array(dtype=wp.vec3i),
    verts: wp.array(dtype=vec2d),
    young: wp.float64,
    poisson: wp.float64,
    dst: wp.array(dtype=wp.int32),  # per-triplet compact block index, -1 if it touches a pin
    values: wp.array3d(dtype=wp.float64),  # BSR A.scalar_values, shape (nnz, 2, 2)
):
    """Scatter-add each element's free-free 2x2 blocks into the fixed-pattern A.values.

    Assumes A.values was zeroed first. Because the mesh topology is fixed, the
    sparsity pattern never changes during optimization; only the values do, so
    this avoids re-running bsr_from_triplets (which host-syncs) each step and is
    graph-capturable.
    """
    t = wp.tid()
    f = tris[t]
    Ke = local_stiffness(verts[f[0]], verts[f[1]], verts[f[2]], young, poisson)
    for i in range(3):
        for j in range(3):
            d = dst[t * 9 + i * 3 + j]
            if d >= 0:
                blk = _sub_block(Ke, i, j)
                wp.atomic_add(values, d, 0, 0, blk[0, 0])
                wp.atomic_add(values, d, 0, 1, blk[0, 1])
                wp.atomic_add(values, d, 1, 0, blk[1, 0])
                wp.atomic_add(values, d, 1, 1, blk[1, 1])


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


@wp.kernel
def _sub(a: wp.array(dtype=vec2d), b: wp.array(dtype=vec2d), out: wp.array(dtype=vec2d)):
    v = wp.tid()
    out[v] = a[v] - b[v]


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


# ---------------------------------------------------------------------------
# Optimizers: gradient descent, Adam, Gauss-Newton
# ---------------------------------------------------------------------------


@wp.kernel
def _apply_step(verts: wp.array(dtype=vec2d), alpha: wp.float64, dV: wp.array(dtype=vec2d)):
    """In-place update V <- V + alpha dV (dV is zero at pins, so pins stay fixed)."""
    v = wp.tid()
    verts[v] = verts[v] + alpha * dV[v]


@wp.kernel
def _gather_free_positions(
    verts: wp.array(dtype=vec2d), free_verts: wp.array(dtype=wp.int32), out: wp.array(dtype=vec2d)
):
    i = wp.tid()
    out[i] = verts[free_verts[i]]


@wp.kernel
def _scatter_free_positions(
    params: wp.array(dtype=vec2d), free_verts: wp.array(dtype=wp.int32), verts: wp.array(dtype=vec2d)
):
    i = wp.tid()
    verts[free_verts[i]] = params[i]


@wp.kernel
def _adam_update(
    g: wp.array(dtype=vec2d),
    m: wp.array(dtype=vec2d),
    v: wp.array(dtype=vec2d),
    lr: wp.float64,
    b1: wp.float64,
    b2: wp.float64,
    bc1: wp.float64,  # 1 - b1**t
    bc2: wp.float64,  # 1 - b2**t
    eps: wp.float64,
    params: wp.array(dtype=vec2d),
):
    """Per-free-vertex Adam update (double precision; warp.optim.Adam is fp32/vec3 only)."""
    i = wp.tid()
    gi = g[i]
    one = wp.float64(1.0)
    mi = b1 * m[i] + (one - b1) * gi
    vi = b2 * v[i] + (one - b2) * wp.cw_mul(gi, gi)
    m[i] = mi
    v[i] = vi
    mhat = mi / bc1
    vhat = vi / bc2
    denom = vec2d(wp.sqrt(vhat[0]) + eps, wp.sqrt(vhat[1]) + eps)
    params[i] = params[i] - lr * wp.cw_div(mhat, denom)


@wp.kernel
def _incr_counter(t: wp.array(dtype=wp.int32)):
    t[0] = t[0] + 1


@wp.kernel
def _adam_step_dev(
    g: wp.array(dtype=vec2d),
    m: wp.array(dtype=vec2d),
    v: wp.array(dtype=vec2d),
    lr: wp.float64,
    b1: wp.float64,
    b2: wp.float64,
    eps: wp.float64,
    t_counter: wp.array(dtype=wp.int32),  # device iteration counter, for bias correction
    params: wp.array(dtype=vec2d),
):
    """Adam update with device-side bias correction (fully graph-capturable)."""
    i = wp.tid()
    t = wp.float64(t_counter[0])
    gi = g[i]
    one = wp.float64(1.0)
    mi = b1 * m[i] + (one - b1) * gi
    vi = b2 * v[i] + (one - b2) * wp.cw_mul(gi, gi)
    m[i] = mi
    v[i] = vi
    mhat = mi / (one - wp.pow(b1, t))
    vhat = vi / (one - wp.pow(b2, t))
    denom = vec2d(wp.sqrt(vhat[0]) + eps, wp.sqrt(vhat[1]) + eps)
    params[i] = params[i] - lr * wp.cw_div(mhat, denom)


@wp.kernel
def _scatter_G_inplace(
    tris: wp.array(dtype=wp.vec3i),
    verts: wp.array(dtype=vec2d),
    young: wp.float64,
    poisson: wp.float64,
    u: wp.array(dtype=vec2d),
    f_ext: wp.array(dtype=vec2d),
    eps: wp.float64,
    dst: wp.array(dtype=wp.int32),
    values: wp.array3d(dtype=wp.float64),  # G_ff.scalar_values (nnz, 2, 2), zeroed first
):
    """Scatter-add the element sensitivity blocks into the fixed-pattern G_ff.values."""
    t = wp.tid()
    f = tris[t]
    p0 = verts[f[0]]
    p1 = verts[f[1]]
    p2 = verts[f[2]]
    u_e = vec6d(u[f[0]][0], u[f[0]][1], u[f[1]][0], u[f[1]][1], u[f[2]][0], u[f[2]][1])
    f_e = vec6d(f_ext[f[0]][0], f_ext[f[0]][1], f_ext[f[1]][0], f_ext[f[1]][1], f_ext[f[2]][0], f_ext[f[2]][1])
    zero2 = vec2d(wp.float64(0.0), wp.float64(0.0))
    Ge = mat66d(wp.float64(0.0))
    for g in range(6):
        vtx = g // 2
        comp = g % 2
        d = vec2d(wp.where(comp == 0, eps, wp.float64(0.0)), wp.where(comp == 1, eps, wp.float64(0.0)))
        pp0 = p0 + wp.where(vtx == 0, d, zero2)
        pp1 = p1 + wp.where(vtx == 1, d, zero2)
        pp2 = p2 + wp.where(vtx == 2, d, zero2)
        pm0 = p0 - wp.where(vtx == 0, d, zero2)
        pm1 = p1 - wp.where(vtx == 1, d, zero2)
        pm2 = p2 - wp.where(vtx == 2, d, zero2)
        col = (_elem_h(pp0, pp1, pp2, young, poisson, u_e, f_e) - _elem_h(pm0, pm1, pm2, young, poisson, u_e, f_e)) / (
            wp.float64(2.0) * eps
        )
        for i in range(6):
            Ge[i, g] = col[i]

    for bi in range(3):
        for bj in range(3):
            dd = dst[t * 9 + bi * 3 + bj]
            if dd >= 0:
                blk = _sub_block(Ge, bi, bj)
                wp.atomic_add(values, dd, 0, 0, blk[0, 0])
                wp.atomic_add(values, dd, 0, 1, blk[0, 1])
                wp.atomic_add(values, dd, 1, 0, blk[1, 0])
                wp.atomic_add(values, dd, 1, 1, blk[1, 1])


@wp.kernel
def _add_blocks(a: wp.array(dtype=mat22d), b: wp.array(dtype=mat22d), out: wp.array(dtype=mat22d)):
    """out[k] = a[k] + b[k] over the shared block pattern (T.values = A.values + G_ff.values)."""
    k = wp.tid()
    out[k] = a[k] + b[k]


@wp.kernel
def _add_damping_diag(diag_block_idx: wp.array(dtype=wp.int32), damping: wp.float64, values: wp.array(dtype=mat22d)):
    """Add damping to the (v,v) diagonal blocks (indices precomputed)."""
    v = wp.tid()
    k = diag_block_idx[v]
    values[k] = values[k] + mat22d(damping, wp.float64(0.0), wp.float64(0.0), damping)


@wp.kernel
def _accumulate_sq(v_target: wp.array(dtype=vec2d), U: wp.array(dtype=vec2d), out: wp.array(dtype=wp.float64)):
    """Accumulate sum of squared residual components into out[0] (device-side loss)."""
    v = wp.tid()
    dv = v_target[v] - U[v]
    wp.atomic_add(out, 0, dv[0] * dv[0] + dv[1] * dv[1])


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
        self.free_verts = wp.array(np.asarray(free_vertices_np, dtype=np.int32), dtype=wp.int32, device=self.device)

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

    def build_sparsity(self):
        """Build the fixed sparsity pattern once and precompute the triplet->block map.

        Returns a BSR matrix ``self.A_pattern`` whose values are refilled in place by
        :meth:`assemble_stiffness_inplace` (and, sharing the identical pattern, by the
        sensitivity/T assembly). ``self.scatter_dst[k]`` is the compact block index for
        triplet ``k = t*9 + i*3 + j`` (``-1`` if it touches a pinned vertex).
        """
        A = self.assemble_stiffness()
        nnz = A.nnz_sync()
        offsets = A.offsets.numpy()
        columns = A.columns.numpy()[:nnz]
        tris = self.tris.numpy()
        v2f = self.vert_to_free.numpy()
        dst = np.full(self.num_tris * 9, -1, dtype=np.int32)
        for t in range(self.num_tris):
            tf = tris[t]
            for i in range(3):
                fi = int(v2f[tf[i]])
                if fi < 0:
                    continue
                beg, end = int(offsets[fi]), int(offsets[fi + 1])
                rowcols = columns[beg:end]
                for j in range(3):
                    fj = int(v2f[tf[j]])
                    if fj < 0:
                        continue
                    k = int(np.searchsorted(rowcols, fj))
                    if k < len(rowcols) and int(rowcols[k]) == fj:
                        dst[t * 9 + i * 3 + j] = beg + k
        # Diagonal (v,v) block index per free vertex, for Levenberg-Marquardt damping.
        diag = np.empty(self.num_free, dtype=np.int32)
        for v in range(self.num_free):
            beg, end = int(offsets[v]), int(offsets[v + 1])
            rowcols = columns[beg:end]
            k = int(np.searchsorted(rowcols, v))
            diag[v] = beg + k

        self.scatter_dst = wp.array(dst, dtype=wp.int32, device=self.device)
        self.diag_block_idx = wp.array(diag, dtype=wp.int32, device=self.device)
        self.nnz = nnz
        self.A_pattern = A
        # G_ff and T share A's exact sparsity pattern (same element free-free
        # couplings), so their values arrays are block-index-aligned with A's.
        self.Gff_pattern = wps.bsr_copy(A)
        self.T_pattern = wps.bsr_copy(A)
        return A

    def assemble_Gff_inplace(self, u, eps=1e-6):
        """Refill ``self.Gff_pattern`` values from the current rest shape (fixed pattern)."""
        if not hasattr(self, "Gff_pattern"):
            self.build_sparsity()
        self.Gff_pattern.values.zero_()
        wp.launch(
            _scatter_G_inplace,
            dim=self.num_tris,
            inputs=[self.tris, self.verts, self.young, self.poisson, u, self.f_ext,
                    wp.float64(eps), self.scatter_dst, self.Gff_pattern.scalar_values],
            device=self.device,
        )  # fmt: skip
        return self.Gff_pattern

    def assemble_T_inplace(self, damping=0.0):
        """Form ``T = A + G_ff (+ damping*I)`` in place, reusing the shared pattern."""
        wp.launch(
            _add_blocks,
            dim=self.nnz,
            inputs=[self.A_pattern.values, self.Gff_pattern.values, self.T_pattern.values],
            device=self.device,
        )
        if damping != 0.0:
            wp.launch(
                _add_damping_diag,
                dim=self.num_free,
                inputs=[self.diag_block_idx, wp.float64(damping), self.T_pattern.values],
                device=self.device,
            )
        return self.T_pattern

    def assemble_stiffness_inplace(self):
        """Refill ``self.A_pattern`` values from the current rest shape (fixed pattern)."""
        if not hasattr(self, "A_pattern"):
            self.build_sparsity()
        self.A_pattern.values.zero_()
        wp.launch(
            _scatter_stiffness_inplace,
            dim=self.num_tris,
            inputs=[self.tris, self.verts, self.young, self.poisson, self.scatter_dst, self.A_pattern.scalar_values],
            device=self.device,
        )
        return self.A_pattern

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

    def gradient_free(self, tol=1e-10):
        """Ascent gradient of the loss at the free vertices (array of vec2d, length #free)."""
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
        return grad_free

    def gradient_step(self, tol=1e-10):
        """Ascent gradient of the loss as a per-vertex ``dV`` (vec2d, zero at pins)."""
        grad_free = self.gradient_free(tol=tol)
        dV = wp.empty(self.num_verts, dtype=vec2d, device=self.device)
        wp.launch(_scatter_free_step, dim=self.num_verts, inputs=[grad_free, self.vert_to_free, dV], device=self.device)
        return dV

    def gauss_newton_step(self, tol=1e-10, solve_tol=1e-12, solver="bicgstab"):
        """Sparse Gauss-Newton step via the square route T = A + G_ff.

        Solves ``T w = G_ff r_free``, then ``p = r_free - w``. ``T`` is sparse but
        nonsymmetric and, on this problem, moderately ill-conditioned, so it is
        solved with BiCGSTAB by default (GMRES stalls here: restarted GMRES loses
        orthogonality at the conditioning of ``T``). Returns a per-vertex ``dV``
        (vec2d, zero at pins).
        """
        U, u, _, A = self.forward(tol=tol)
        r_free = self.residual_free(U)
        Gff = self.assemble_Gff(u)

        T = A + Gff  # sparse, generally nonsymmetric

        rhs = wp.zeros(self.num_free, dtype=vec2d, device=self.device)
        wps.bsr_mv(Gff, r_free, rhs)

        w = wp.zeros(self.num_free, dtype=vec2d, device=self.device)
        precond = wpl.preconditioner(T, "diag")
        n = self.num_free * 2
        if solver == "gmres":
            wpl.gmres(T, rhs, w, tol=solve_tol, maxiter=n, restart=n, M=precond)
        else:
            wpl.bicgstab(T, rhs, w, tol=solve_tol, maxiter=n * 4, M=precond)

        p = wp.empty(self.num_free, dtype=vec2d, device=self.device)
        wp.launch(_sub, dim=self.num_free, inputs=[r_free, w, p], device=self.device)

        dV = wp.empty(self.num_verts, dtype=vec2d, device=self.device)
        wp.launch(_scatter_free_step, dim=self.num_verts, inputs=[p, self.vert_to_free, dV], device=self.device)
        return dV

    # -- optimization driver --

    def get_free_positions(self):
        out = wp.empty(self.num_free, dtype=vec2d, device=self.device)
        wp.launch(
            _gather_free_positions, dim=self.num_free, inputs=[self.verts, self.free_verts, out], device=self.device
        )
        return out

    def set_free_positions(self, params):
        wp.launch(
            _scatter_free_positions, dim=self.num_free, inputs=[params, self.free_verts, self.verts], device=self.device
        )

    def apply_step(self, alpha, dV):
        wp.launch(_apply_step, dim=self.num_verts, inputs=[self.verts, wp.float64(alpha), dV], device=self.device)

    def optimize(
        self,
        method="gn",
        num_iters=100,
        step_size=None,
        tol=1e-8,
        betas=(0.9, 0.999),
        adam_eps=1e-8,
        forward_tol=1e-10,
        record_every=0,
        quiet=True,
    ):
        """Run a fixed-step-size optimization and return a result dict.

        ``method`` is one of ``"gn"`` (Gauss-Newton), ``"gd"`` (gradient descent),
        or ``"adam"``. Tracks the loss trajectory, the best loss, the worst spike
        ratio, and convergence/divergence flags (mirroring the reference). When
        ``record_every > 0``, snapshots ``(V, U)`` every that many iterations for
        visualization.
        """
        if step_size is None:
            step_size = {"gn": 1.0, "gd": 0.5, "adam": 0.02}[method]

        b1, b2 = betas
        if method == "adam":
            params = self.get_free_positions()
            m = wp.zeros_like(params)
            v = wp.zeros_like(params)

        losses = []
        frames = []
        init_loss = self.loss(tol=forward_tol)
        best = init_loss
        max_spike = 1.0
        diverged = False
        converged = False
        t = 0

        for it in range(num_iters):
            loss_val = self.loss(tol=forward_tol)
            losses.append(loss_val)
            if not np.isfinite(loss_val) or loss_val > 1e6 * init_loss:
                diverged = True
                break
            best = min(best, loss_val)
            max_spike = max(max_spike, loss_val / max(best, 1e-300))
            if record_every and it % record_every == 0:
                U, _, _, _ = self.forward(tol=forward_tol)
                frames.append((self.verts.numpy().copy(), U.numpy().copy()))
            if not quiet:
                print(f"  iter {it:4d}  loss {loss_val:.6e}")
            if loss_val < tol * init_loss:
                converged = True
                break

            if method == "gn":
                self.apply_step(step_size, self.gauss_newton_step(tol=forward_tol))
            elif method == "gd":
                self.apply_step(-step_size, self.gradient_step(tol=forward_tol))
            elif method == "adam":
                g = self.gradient_free(tol=forward_tol)
                t += 1
                bc1 = 1.0 - b1**t
                bc2 = 1.0 - b2**t
                wp.launch(
                    _adam_update,
                    dim=self.num_free,
                    inputs=[g, m, v, wp.float64(step_size), wp.float64(b1), wp.float64(b2),
                            wp.float64(bc1), wp.float64(bc2), wp.float64(adam_eps), params],
                    device=self.device,
                )  # fmt: skip
                self.set_free_positions(params)
            else:
                raise ValueError(f"unknown method {method!r}")

        final_loss = self.loss(tol=forward_tol)
        return {
            "method": method,
            "iters": len(losses),
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "best_loss": min(best, final_loss),
            "max_spike_ratio": max_spike,
            "converged": converged,
            "diverged": diverged,
            "losses": losses,
            "frames": frames,
        }


# ---------------------------------------------------------------------------
# Visualization (host-side; optional polyscope headless renderer)
# ---------------------------------------------------------------------------


def per_face_von_mises(V, U, F, young, poisson):
    """Per-triangle von Mises stress from the linear strain B (U - V), evaluated at rest V."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    C = np.array([[lam + 2 * mu, lam, 0.0], [lam, lam + 2 * mu, 0.0], [0.0, 0.0, mu]])
    Mg = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    vm = np.zeros(len(F))
    for f, tri in enumerate(F):
        Vf = V[tri]
        Uf = U[tri]
        Dm = Mg @ Vf
        G = np.linalg.inv(Dm) @ Mg
        B = np.array(
            [
                [G[0, 0], 0, G[0, 1], 0, G[0, 2], 0],
                [0, G[1, 0], 0, G[1, 1], 0, G[1, 2]],
                [G[1, 0], G[0, 0], G[1, 1], G[0, 1], G[1, 2], G[0, 2]],
            ]
        )
        u_local = (Uf - Vf).reshape(-1)
        sxx, syy, sxy = C @ (B @ u_local)
        vm[f] = np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
    return vm


def render_convergence_gif(frames, F, young, poisson, out_path, fps=4, hold=8, frame_stride=1, method_label=""):
    """Render optimization frames to a gif (headless matplotlib, Agg backend).

    Each frame stacks the rest shape being optimized (top, blue) over the
    gravity-deformed shape (bottom, colored by per-face von Mises stress). Uses
    matplotlib for a labeled 2D figure with a colorbar; the triangles are drawn
    with equal aspect so the isotropic grid reads correctly. ``frame_stride`` is
    the number of optimizer iterations between recorded frames (used only to
    label frames with the true iteration number). Requires ``matplotlib`` and
    ``imageio``.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import imageio.v2 as imageio  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.tri as mtri  # noqa: PLC0415

    F = np.asarray(F)
    all_V = np.concatenate([f[0] for f in frames])
    all_U = np.concatenate([f[1] for f in frames])
    vmax = max((per_face_von_mises(V, U, F, young, poisson).max() for V, U in frames), default=1.0)

    # Stack the rest shape (shifted up) above the deformed shape in one equal-
    # aspect axes so the isotropic grid renders with square cells (matching the
    # reference). The figure is sized to the data aspect so equal aspect fills it.
    xlo, xhi = all_V[:, 0].min(), all_V[:, 0].max()
    xspan = xhi - xlo
    # Shift the rest shape to sit just above the deformed shape's highest point.
    span = max(float(np.ptp(all_V[:, 1])), float(np.ptp(all_U[:, 1])), 1e-6)
    gap = (all_U[:, 1].max() - all_V[:, 1].min()) + 0.12 * span
    ylo = all_U[:, 1].min()
    yhi = all_V[:, 1].max() + gap
    yspan = yhi - ylo
    pad = 0.03 * xspan
    fig_w = 12.0
    fig_h = min(max(fig_w * yspan / xspan, 3.0), 9.0) * 1.15  # aspect-matched, clamped + title headroom

    images = []
    for i, (V, U) in enumerate(frames):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120, layout="constrained")
        vm = per_face_von_mises(V, U, F, young, poisson)

        tri_rest = mtri.Triangulation(V[:, 0], V[:, 1] + gap, F)
        ax.tripcolor(tri_rest, facecolors=np.zeros(len(F)), cmap="Blues", vmin=-1, vmax=1)
        ax.triplot(tri_rest, color="#2c6fbb", lw=0.5)

        tri_def = mtri.Triangulation(U[:, 0], U[:, 1], F)
        tpc = ax.tripcolor(tri_def, facecolors=vm, cmap="viridis", vmin=0.0, vmax=vmax, edgecolors="face")
        ax.triplot(tri_def, color="k", lw=0.12, alpha=0.3)

        ax.text(xlo, yhi, "rest shape being optimized", fontsize=9, va="bottom")
        ax.text(xlo, ylo, "gravity-deformed shape (von Mises stress) -> flat target", fontsize=9, va="top")
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(tpc, ax=ax, fraction=0.015, pad=0.01, label="von Mises stress")
        prefix = f"Inverse elasticity ({method_label})" if method_label else "Inverse elasticity shape optimization"
        fig.suptitle(f"{prefix}  -  iteration {i * frame_stride}", fontsize=11)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba[..., :3].copy())
        plt.close(fig)

    images += [images[-1]] * hold  # pause on the converged result
    imageio.mimsave(out_path, images, fps=fps, loop=0)
    return out_path


def _make_bridge(count, young, poisson, device):
    """Build a 15:1 triangulated-grid bridge pinned at its left/right edges."""
    ny = 1 + count
    nx = 15 * (ny - 1) + 1
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    gx, gy = np.meshgrid(xs, ys)
    V = np.stack([gx.ravel(), gy.ravel()], axis=1)
    V[:, 0] *= (nx - 1) / (ny - 1)  # stretch to 15:1
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            tris.append([idx, idx + 1, idx + nx + 1])
            tris.append([idx, idx + nx + 1, idx + nx])
    F = np.array(tris, dtype=np.int32)
    x_min, x_max = V[:, 0].min(), V[:, 0].max()
    free = [i for i in range(V.shape[0]) if abs(V[i, 0] - x_min) > 1e-8 and abs(V[i, 0] - x_max) > 1e-8]
    return V, F, np.array(free, dtype=np.int32)


def profile(device, counts=(2, 4, 8, 16, 32), young=2e3, poisson=0.49):
    """Print per-granularity step time (ms) versus mesh size on ``device``.

    Times each building block (assembly, forward solve, sensitivity, gradient,
    Gauss-Newton step) so their runtime/memory asymptotics can be inspected. See
    the performance section of the design doc for representative numbers.
    """
    import time  # noqa: PLC0415

    dev = wp.get_device(device)

    def timed(fn, reps, warmup=3):
        for _ in range(warmup):
            fn()
        wp.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        wp.synchronize_device(dev)
        return (time.perf_counter() - t0) / reps * 1e3

    header = f"{'count':>5} {'#verts':>8} {'#tris':>8} {'#free':>7} | {'assemA':>8} {'load':>7} {'fwd':>9} {'Gff':>8} {'grad':>9} {'GN':>10}  (ms)"
    print(header)
    for count in counts:
        V, F, free = _make_bridge(count, young, poisson, dev)
        bp = BridgeProblem(V, F, free, young, poisson, device=dev)
        _, u, _, _ = bp.forward(tol=1e-8)
        t_a = timed(bp.assemble_stiffness, 20)
        t_l = timed(bp.assemble_load, 20)
        t_f = timed(lambda bp=bp: bp.forward(tol=1e-8), 10)
        t_g = timed(lambda bp=bp, u=u: bp.assemble_Gff(u), 10)
        t_gr = timed(lambda bp=bp: bp.gradient_free(tol=1e-8), 8)
        t_gn = timed(lambda bp=bp: bp.gauss_newton_step(tol=1e-8, solve_tol=1e-8), 6)
        print(
            f"{count:>5} {V.shape[0]:>8} {F.shape[0]:>8} {bp.num_free:>7} | "
            f"{t_a:>8.3f} {t_l:>7.3f} {t_f:>9.3f} {t_g:>8.3f} {t_gr:>9.3f} {t_gn:>10.3f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--count", type=int, default=4, help="Grid resolution parameter (ny = 1 + count).")
    parser.add_argument("--method", choices=("gn", "gd", "adam"), default="gn", help="Optimizer.")
    parser.add_argument("--num-iters", type=int, default=None, help="Iteration cap (defaults per method).")
    parser.add_argument("--step-size", type=float, default=None, help="Fixed step size (defaults per method).")
    parser.add_argument("--young", type=float, default=2e3, help="Young's modulus.")
    parser.add_argument("--poisson", type=float, default=0.49, help="Poisson's ratio.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Relative loss tolerance for convergence.")
    parser.add_argument("--gif", type=str, default=None, help="Render a convergence gif to this path (matplotlib).")
    parser.add_argument(
        "--gif-every", type=int, default=1, help="Record a gif frame every N iterations (use >1 for Adam)."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print per-granularity step timing vs. mesh size and exit."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration loss output.")
    args = parser.parse_args()

    if args.profile:
        with wp.ScopedDevice(args.device):
            profile(args.device)
        raise SystemExit

    num_iters = args.num_iters or {"gn": 20, "gd": 500, "adam": 4000}[args.method]

    with wp.ScopedDevice(args.device):
        V, F, free = _make_bridge(args.count, args.young, args.poisson, args.device)
        bridge = BridgeProblem(V, F, free, args.young, args.poisson)
        result = bridge.optimize(
            method=args.method,
            num_iters=num_iters,
            step_size=args.step_size,
            tol=args.tol,
            record_every=args.gif_every if args.gif else 0,
            quiet=args.quiet,
        )
        print(
            f"RESULT method={result['method']} count={args.count} nV={V.shape[0]} "
            f"iters={result['iters']} initial_loss={result['initial_loss']:.6e} "
            f"final_loss={result['final_loss']:.6e} best_loss={result['best_loss']:.6e} "
            f"max_spike={result['max_spike_ratio']:.3f} converged={result['converged']} diverged={result['diverged']}"
        )

        if args.gif:
            path = render_convergence_gif(
                result["frames"], F, args.young, args.poisson, args.gif,
                frame_stride=args.gif_every, method_label=args.method,
            )  # fmt: skip
            print(f"wrote {path} ({len(result['frames'])} frames)")
