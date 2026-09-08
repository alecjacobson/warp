# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inverse elasticity shape optimization
#
# Optimizes the rest shape of a pinned 2D elastic bridge so that, once it sags
# under gravity, its deformed shape matches a flat target. A constant-strain
# triangle (CST), plane-strain linear-elasticity forward model computes the
# gravity-sagged shape; the rest-shape gradient of the mean-squared shape error
# is obtained by the adjoint method (autodiff through the assembly + a manual
# adjoint for the linear solve, via the implicit function theorem), and the shape
# is optimized with Warp's Adam optimizer.
#
# The physics is a pure-Warp reimplementation of Jacobson's
# gauss-newton-sensitivity-analysis reference and is validated against it.

import numpy as np
import warp_cudss

import warp as wp
import warp.sparse as wps
from warp.optim import Adam

# Physics runs in double precision; only Warp's Adam state is single precision.
scalar = wp.float64
vec2 = wp.types.vector(2, scalar)
mat22 = wp.types.matrix((2, 2), scalar)
mat23 = wp.types.matrix((2, 3), scalar)
mat33 = wp.types.matrix((3, 3), scalar)
mat36 = wp.types.matrix((3, 6), scalar)
mat66 = wp.types.matrix((6, 6), scalar)
vec6 = wp.types.vector(6, scalar)


# ---------------------------------------------------------------------------
# Element operators (CST, plane strain) -- transcribed from the C++ reference.
# ---------------------------------------------------------------------------


@wp.func
def element_stiffness(p0: vec2, p1: vec2, p2: vec2, young: scalar, poisson: scalar) -> mat66:
    """6x6 element stiffness ``K_e = area * B^T C B`` (engineering strain)."""
    Dm = mat22(p0[0] - p2[0], p0[1] - p2[1],
               p1[0] - p2[0], p1[1] - p2[1])  # fmt: skip
    G = wp.inverse(Dm) * mat23(scalar(1.0), scalar(0.0), scalar(-1.0),
                               scalar(0.0), scalar(1.0), scalar(-1.0))  # fmt: skip
    B = mat36(
        G[0, 0], scalar(0.0), G[0, 1], scalar(0.0), G[0, 2], scalar(0.0),
        scalar(0.0), G[1, 0], scalar(0.0), G[1, 1], scalar(0.0), G[1, 2],
        G[1, 0], G[0, 0], G[1, 1], G[0, 1], G[1, 2], G[0, 2],
    )  # fmt: skip
    lam = young * poisson / ((scalar(1.0) + poisson) * (scalar(1.0) - scalar(2.0) * poisson))
    mu = young / (scalar(2.0) * (scalar(1.0) + poisson))
    C = mat33(
        lam + scalar(2.0) * mu, lam, scalar(0.0),
        lam, lam + scalar(2.0) * mu, scalar(0.0),
        scalar(0.0), scalar(0.0), mu,
    )  # fmt: skip
    area = wp.abs(wp.determinant(Dm)) / scalar(2.0)
    return area * (wp.transpose(B) * C * B)


@wp.func
def element_area(p0: vec2, p1: vec2, p2: vec2) -> scalar:
    Dm = mat22(p0[0] - p2[0], p0[1] - p2[1],
               p1[0] - p2[0], p1[1] - p2[1])  # fmt: skip
    return wp.abs(wp.determinant(Dm)) / scalar(2.0)


# ---------------------------------------------------------------------------
# Assembly / solve kernels
# ---------------------------------------------------------------------------


@wp.kernel
def assemble_triplets(
    tris: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=vec2),
    young: scalar,
    poisson: scalar,
    vert_to_free: wp.array(dtype=wp.int32),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=mat22),
):
    """Emit free-free 2x2 stiffness blocks as triplets (pinned couplings -> (0,0) zero)."""
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    Ke = element_stiffness(verts[i0], verts[i1], verts[i2], young, poisson)
    ids = wp.vec3i(i0, i1, i2)
    for a in range(3):
        fa = vert_to_free[ids[a]]
        for b in range(3):
            fb = vert_to_free[ids[b]]
            slot = t * 9 + a * 3 + b
            if fa >= 0 and fb >= 0:
                rows[slot] = fa
                cols[slot] = fb
                blocks[slot] = mat22(Ke[2 * a + 0, 2 * b + 0], Ke[2 * a + 0, 2 * b + 1],
                                     Ke[2 * a + 1, 2 * b + 0], Ke[2 * a + 1, 2 * b + 1])  # fmt: skip
            else:
                rows[slot] = 0
                cols[slot] = 0
                blocks[slot] = mat22(scalar(0.0))


@wp.kernel
def scatter_stiffness_inplace(
    tris: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=vec2),
    young: scalar,
    poisson: scalar,
    dst: wp.array(dtype=wp.int32),  # triplet -> compact block index (-1 if pinned)
    values: wp.array3d(dtype=scalar),  # BSR scalar_values, shape (nnz, 2, 2)
):
    """Refill the fixed-pattern stiffness values in place from the current shape."""
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    Ke = element_stiffness(verts[i0], verts[i1], verts[i2], young, poisson)
    for a in range(3):
        for b in range(3):
            blk = dst[t * 9 + a * 3 + b]
            if blk >= 0:
                wp.atomic_add(values, blk, 0, 0, Ke[2 * a + 0, 2 * b + 0])
                wp.atomic_add(values, blk, 0, 1, Ke[2 * a + 0, 2 * b + 1])
                wp.atomic_add(values, blk, 1, 0, Ke[2 * a + 1, 2 * b + 0])
                wp.atomic_add(values, blk, 1, 1, Ke[2 * a + 1, 2 * b + 1])


@wp.kernel
def accumulate_vertex_mass(tris: wp.array2d(dtype=wp.int32), verts: wp.array(dtype=vec2),
                           mass: wp.array(dtype=scalar)):  # fmt: skip
    """Lumped mass: each vertex gets area/3 from every incident triangle."""
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    a3 = element_area(verts[i0], verts[i1], verts[i2]) / scalar(3.0)
    wp.atomic_add(mass, i0, a3)
    wp.atomic_add(mass, i1, a3)
    wp.atomic_add(mass, i2, a3)


@wp.kernel
def build_free_load(mass: wp.array(dtype=scalar), f_ext: wp.array(dtype=vec2),
                    vert_to_free: wp.array(dtype=wp.int32), load: wp.array(dtype=vec2)):  # fmt: skip
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        load[f] = mass[v] * f_ext[v]


@wp.kernel
def scatter_solution(q: wp.array(dtype=vec2), vert_to_free: wp.array(dtype=wp.int32),
                     verts: wp.array(dtype=vec2), u_full: wp.array(dtype=vec2),
                     U: wp.array(dtype=vec2)):  # fmt: skip
    """Full-length displacement (0 at pins) and deformed positions U = V + u."""
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        u_full[v] = q[f]
        U[v] = verts[v] + q[f]
    else:
        u_full[v] = vec2(scalar(0.0))
        U[v] = verts[v]


@wp.kernel
def gather_free_residual(v_target: wp.array(dtype=vec2), U: wp.array(dtype=vec2),
                         vert_to_free: wp.array(dtype=wp.int32), r_free: wp.array(dtype=vec2)):  # fmt: skip
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        r_free[f] = v_target[v] - U[v]


@wp.kernel
def residual_force(
    tris: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=vec2),
    young: scalar,
    poisson: scalar,
    u_full: wp.array(dtype=vec2),
    f_ext: wp.array(dtype=vec2),
    vert_to_free: wp.array(dtype=wp.int32),
    w: wp.array(dtype=vec2),
):
    """Equilibrium residual force ``w[free] = sum_e (M_e f_e - K_e u_e)``.

    Differentiable w.r.t. ``verts`` (``u_full`` held fixed). Seeding ``w.grad``
    with the adjoint ``lambda`` and backpropagating gives ``G_ff^T lambda`` at the
    free vertices -- the geometry sensitivity, without assembling it explicitly.
    """
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    p0, p1, p2 = verts[i0], verts[i1], verts[i2]
    Ke = element_stiffness(p0, p1, p2, young, poisson)
    a3 = element_area(p0, p1, p2) / scalar(3.0)
    ue = vec6(u_full[i0][0], u_full[i0][1], u_full[i1][0], u_full[i1][1], u_full[i2][0], u_full[i2][1])
    Kue = Ke * ue
    ids = wp.vec3i(i0, i1, i2)
    for a in range(3):
        f = vert_to_free[ids[a]]
        if f >= 0:
            mf = a3 * f_ext[ids[a]]
            wp.atomic_add(w, f, vec2(mf[0] - Kue[2 * a + 0], mf[1] - Kue[2 * a + 1]))


@wp.kernel
def combine_gradient(v_target: wp.array(dtype=vec2), U: wp.array(dtype=vec2),
                     gff_lambda: wp.array(dtype=vec2), free_verts: wp.array(dtype=wp.int32),
                     scale: scalar, grad_free: wp.array(dtype=wp.float32)):  # fmt: skip
    """grad_free = -scale (r_free + G_ff^T lambda), flattened to float32 for Adam."""
    i = wp.tid()
    v = free_verts[i]
    g = -scale * ((v_target[v] - U[v]) + gff_lambda[v])
    grad_free[2 * i + 0] = wp.float32(g[0])
    grad_free[2 * i + 1] = wp.float32(g[1])


@wp.kernel
def scatter_free_params(params: wp.array(dtype=wp.float32), free_verts: wp.array(dtype=wp.int32),
                        verts: wp.array(dtype=vec2)):  # fmt: skip
    i = wp.tid()
    v = free_verts[i]
    verts[v] = vec2(scalar(params[2 * i]), scalar(params[2 * i + 1]))


# --- Gauss-Newton assembly (the sensitivity G_ff, the T = A + G_ff system) ---


@wp.func
def element_residual(p0: vec2, p1: vec2, p2: vec2, young: scalar, poisson: scalar,
                     u_e: vec6, f_e: vec6) -> vec6:  # fmt: skip
    """Element equilibrium residual force ``M_e f_e - K_e u_e`` (lumped mass)."""
    return (element_area(p0, p1, p2) / scalar(3.0)) * f_e - element_stiffness(p0, p1, p2, young, poisson) * u_e


@wp.kernel
def scatter_sensitivity_inplace(
    tris: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=vec2),
    young: scalar,
    poisson: scalar,
    u_full: wp.array(dtype=vec2),
    f_ext: wp.array(dtype=vec2),
    eps: scalar,
    dst: wp.array(dtype=wp.int32),
    values: wp.array3d(dtype=scalar),  # G_ff scalar_values (nnz, 2, 2), zeroed first
):
    """Refill G_ff, ``G_e[:,a] = d(M_e f_e - K_e u_e)/dx_a`` by central differences."""
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    p0, p1, p2 = verts[i0], verts[i1], verts[i2]
    u_e = vec6(u_full[i0][0], u_full[i0][1], u_full[i1][0], u_full[i1][1], u_full[i2][0], u_full[i2][1])
    f_e = vec6(f_ext[i0][0], f_ext[i0][1], f_ext[i1][0], f_ext[i1][1], f_ext[i2][0], f_ext[i2][1])
    Ge = mat66(scalar(0.0))
    for a in range(6):
        comp = a % 2
        d = vec2(wp.where(comp == 0, eps, scalar(0.0)), wp.where(comp == 1, eps, scalar(0.0)))
        d0 = wp.where(a // 2 == 0, d, vec2(scalar(0.0)))
        d1 = wp.where(a // 2 == 1, d, vec2(scalar(0.0)))
        d2 = wp.where(a // 2 == 2, d, vec2(scalar(0.0)))
        col = (element_residual(p0 + d0, p1 + d1, p2 + d2, young, poisson, u_e, f_e)
               - element_residual(p0 - d0, p1 - d1, p2 - d2, young, poisson, u_e, f_e)) / (scalar(2.0) * eps)  # fmt: skip
        for i in range(6):
            Ge[i, a] = col[i]
    for bi in range(3):
        for bj in range(3):
            blk = dst[t * 9 + bi * 3 + bj]
            if blk >= 0:
                wp.atomic_add(values, blk, 0, 0, Ge[2 * bi + 0, 2 * bj + 0])
                wp.atomic_add(values, blk, 0, 1, Ge[2 * bi + 0, 2 * bj + 1])
                wp.atomic_add(values, blk, 1, 0, Ge[2 * bi + 1, 2 * bj + 0])
                wp.atomic_add(values, blk, 1, 1, Ge[2 * bi + 1, 2 * bj + 1])


@wp.kernel
def add_blocks(a: wp.array(dtype=mat22), b: wp.array(dtype=mat22), out: wp.array(dtype=mat22)):
    k = wp.tid()
    out[k] = a[k] + b[k]


@wp.kernel
def sub_free(a: wp.array(dtype=vec2), b: wp.array(dtype=vec2), out: wp.array(dtype=vec2)):
    i = wp.tid()
    out[i] = a[i] - b[i]


@wp.kernel
def apply_free_step(step: scalar, p_step: wp.array(dtype=vec2), free_verts: wp.array(dtype=wp.int32),
                    verts: wp.array(dtype=vec2)):  # fmt: skip
    i = wp.tid()
    verts[free_verts[i]] = verts[free_verts[i]] + step * p_step[i]


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class InverseElasticity:
    """Bridge whose rest shape is optimized so its gravity-sagged shape is flat."""

    def __init__(self, V, F, fixed, young=2e3, poisson=0.49, gravity=-9.8, lr=0.02, device=None):
        self.device = wp.get_device(device)
        self.young, self.poisson = float(young), float(poisson)
        self.num_verts = int(V.shape[0])
        self.num_tris = int(F.shape[0])

        fixed_set = {int(i) for i in fixed}
        free = np.array([i for i in range(self.num_verts) if i not in fixed_set], dtype=np.int32)
        self.num_free = int(free.size)
        v2f = np.full(self.num_verts, -1, dtype=np.int32)
        v2f[free] = np.arange(self.num_free, dtype=np.int32)

        d = self.device
        self.verts = wp.array(V.astype(np.float64), dtype=vec2, device=d, requires_grad=True)
        self.tris = wp.array(F.astype(np.int32), dtype=wp.int32, device=d)
        self.free_verts = wp.array(free, dtype=wp.int32, device=d)
        self.vert_to_free = wp.array(v2f, dtype=wp.int32, device=d)
        self.v_target = wp.array(V.astype(np.float64), dtype=vec2, device=d)  # flat initial shape
        f_ext = np.zeros((self.num_verts, 2), dtype=np.float64)
        f_ext[:, 1] = gravity
        self.f_ext = wp.array(f_ext, dtype=vec2, device=d)

        self._build_sparsity()

        # Reusable buffers.
        nf, nv = self.num_free, self.num_verts
        self.mass = wp.zeros(nv, dtype=scalar, device=d)
        self.load = wp.zeros(nf, dtype=vec2, device=d)
        self.q = wp.zeros(nf, dtype=vec2, device=d)
        self.u_full = wp.zeros(nv, dtype=vec2, device=d)
        self.U = wp.empty(nv, dtype=vec2, device=d)
        self.r_free = wp.zeros(nf, dtype=vec2, device=d)
        self.lam = wp.zeros(nf, dtype=vec2, device=d)
        self.w = wp.zeros(nf, dtype=vec2, device=d, requires_grad=True)
        self.grad_free = wp.zeros(2 * nf, dtype=wp.float32, device=d)

        # Warp's Adam optimizer operates on single-precision scalar parameters.
        self.params = wp.array(V[free].reshape(-1).astype(np.float32), dtype=wp.float32, device=d)
        self.optimizer = Adam([self.params], lr=lr)

        # cuDSS sparse direct solver: the stiffness A is SPD with a fixed sparsity
        # pattern, so we factor it once per shape and solve both the forward and
        # adjoint systems with that one factorization. Symbolic analysis + the first
        # factorization happen here (not capturable); refactor()/solve() are.
        self._assemble()
        self.solver = warp_cudss.CudssSolver(mtype="spd", device=d)
        self.solver.setup(self.A, self.q, self.load)

    def _build_sparsity(self):
        d = self.device
        nt = self.num_tris
        rows = wp.empty(nt * 9, dtype=wp.int32, device=d)
        cols = wp.empty(nt * 9, dtype=wp.int32, device=d)
        blocks = wp.empty(nt * 9, dtype=mat22, device=d)
        wp.launch(assemble_triplets, dim=nt,
                  inputs=[self.tris, self.verts, scalar(self.young), scalar(self.poisson),
                          self.vert_to_free, rows, cols, blocks], device=d)  # fmt: skip
        self.A = wps.bsr_from_triplets(self.num_free, self.num_free, rows, cols, blocks)
        nnz = self.A.nnz_sync()
        offsets = self.A.offsets.numpy()
        columns = self.A.columns.numpy()[:nnz]
        tris = self.tris.numpy()
        v2f = self.vert_to_free.numpy()
        dst = np.full(nt * 9, -1, dtype=np.int32)
        for t in range(nt):
            for a in range(3):
                fa = int(v2f[tris[t, a]])
                if fa < 0:
                    continue
                beg, end = int(offsets[fa]), int(offsets[fa + 1])
                rc = columns[beg:end]
                for b in range(3):
                    fb = int(v2f[tris[t, b]])
                    if fb < 0:
                        continue
                    k = int(np.searchsorted(rc, fb))
                    if k < len(rc) and int(rc[k]) == fb:
                        dst[t * 9 + a * 3 + b] = beg + k
        self.scatter_dst = wp.array(dst, dtype=wp.int32, device=d)

    # -- forward / gradient --

    def _assemble(self):
        self.A.values.zero_()
        wp.launch(scatter_stiffness_inplace, dim=self.num_tris,
                  inputs=[self.tris, self.verts, scalar(self.young), scalar(self.poisson),
                          self.scatter_dst, self.A.scalar_values], device=self.device)  # fmt: skip
        self.mass.zero_()
        wp.launch(accumulate_vertex_mass, dim=self.num_tris, inputs=[self.tris, self.verts, self.mass], device=self.device)
        wp.launch(build_free_load, dim=self.num_verts,
                  inputs=[self.mass, self.f_ext, self.vert_to_free, self.load], device=self.device)  # fmt: skip

    def forward(self):
        """Assemble and factor the stiffness, then solve for the free displacements."""
        self._assemble()
        self.solver.refactor(self.A)
        self.solver.solve(x=self.q, b=self.load)
        wp.launch(scatter_solution, dim=self.num_verts,
                  inputs=[self.q, self.vert_to_free, self.verts, self.u_full, self.U], device=self.device)  # fmt: skip
        return self.U

    def loss(self):
        self.forward()
        diff = self.v_target.numpy() - self.U.numpy()
        return float(np.mean(diff**2))

    def compute_gradient(self):
        """Fill ``self.grad_free`` (float32) with the rest-shape gradient of the loss."""
        self.forward()
        self.r_free.zero_()
        wp.launch(gather_free_residual, dim=self.num_verts,
                  inputs=[self.v_target, self.U, self.vert_to_free, self.r_free], device=self.device)  # fmt: skip
        # Adjoint solve K lambda = r_free reuses the factorization from forward().
        self.solver.solve(x=self.lam, b=self.r_free)
        # G_ff^T lambda = d(lambda . residual_force)/d verts, via autodiff.
        self.w.zero_()
        self.verts.grad.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(residual_force, dim=self.num_tris,
                      inputs=[self.tris, self.verts, scalar(self.young), scalar(self.poisson),
                              self.u_full, self.f_ext, self.vert_to_free, self.w], device=self.device)  # fmt: skip
        self.w.grad.assign(self.lam)
        tape.backward()
        wp.launch(combine_gradient, dim=self.num_free,
                  inputs=[self.v_target, self.U, self.verts.grad, self.free_verts,
                          scalar(2.0 / (2 * self.num_verts)), self.grad_free], device=self.device)  # fmt: skip
        return self.grad_free

    def gradient_np(self):
        """Per-free-vertex loss gradient in double precision (n_free, 2)."""
        self.compute_gradient()
        scale = 2.0 / (2 * self.num_verts)
        r = self.r_free.numpy()
        gl = self.verts.grad.numpy()[self.free_verts.numpy()]
        return -scale * (r + gl)

    # -- Gauss-Newton --

    def _assemble_T(self):
        """Refill G_ff and T = A + G_ff in place (both share A's pattern)."""
        self.Gff.values.zero_()
        wp.launch(scatter_sensitivity_inplace, dim=self.num_tris,
                  inputs=[self.tris, self.verts, scalar(self.young), scalar(self.poisson),
                          self.u_full, self.f_ext, scalar(1e-6), self.scatter_dst, self.Gff.scalar_values], device=self.device)  # fmt: skip
        wp.launch(add_blocks, dim=self.A.values.shape[0], inputs=[self.A.values, self.Gff.values, self.T.values], device=self.device)

    def _setup_gn(self):
        if getattr(self, "_gn_ready", False):
            return
        d = self.device
        self.Gff = wps.bsr_copy(self.A)
        self.T = wps.bsr_copy(self.A)
        self.p_step = wp.zeros(self.num_free, dtype=vec2, device=d)
        self.gn_rhs = wp.zeros(self.num_free, dtype=vec2, device=d)
        self.gn_w = wp.zeros(self.num_free, dtype=vec2, device=d)
        self._gn_ready = True

    def gauss_newton_step(self):
        """One Gauss-Newton step via the square route: T w = G_ff r, p = r - w."""
        self._setup_gn()
        self.forward()
        self.r_free.zero_()
        wp.launch(gather_free_residual, dim=self.num_verts,
                  inputs=[self.v_target, self.U, self.vert_to_free, self.r_free], device=self.device)  # fmt: skip
        self._assemble_T()
        self.gn_rhs.zero_()
        wps.bsr_mv(self.Gff, self.r_free, self.gn_rhs)
        s = warp_cudss.solve(self.T, self.gn_rhs, self.gn_w, mtype="general")
        s.release()
        wp.launch(sub_free, dim=self.num_free, inputs=[self.r_free, self.gn_w, self.p_step], device=self.device)
        return self.p_step

    def gauss_newton_optimize(self, num_iters=20, step_size=1.0, tol=1e-8, record_every=0, quiet=True):
        init = self.loss()
        self.frames = [(0, self.verts.numpy().copy(), self.U.numpy().copy())] if record_every else []
        converged = False
        it = 0
        while it < num_iters:
            self.gauss_newton_step()
            wp.launch(apply_free_step, dim=self.num_free,
                      inputs=[scalar(step_size), self.p_step, self.free_verts, self.verts], device=self.device)  # fmt: skip
            it += 1
            loss = self.loss()  # forward at the updated shape -> self.U
            if record_every and it % record_every == 0:
                self.frames.append((it, self.verts.numpy().copy(), self.U.numpy().copy()))
            if not quiet:
                print(f"  gn iter {it:3d}  loss {loss:.6e}", flush=True)
            if loss < tol * init:
                converged = True
                break
        return {"iters": it, "initial_loss": init, "final_loss": loss, "converged": converged}

    # -- optimization --

    def step(self):
        wp.launch(scatter_free_params, dim=self.num_free,
                  inputs=[self.params, self.free_verts, self.verts], device=self.device)  # fmt: skip
        self.compute_gradient()
        self.optimizer.step([self.grad_free])

    def _loss_from_U(self):
        """Loss from the deformed shape already computed by the last forward."""
        diff = self.v_target.numpy() - self.U.numpy()
        return float(np.mean(diff**2))

    def optimize(self, num_iters=800, tol=1e-8, check_every=25, record_every=0, quiet=True):
        wp.launch(scatter_free_params, dim=self.num_free,
                  inputs=[self.params, self.free_verts, self.verts], device=self.device)  # fmt: skip
        init = self.loss()
        self.frames = []  # (iter, rest V, deformed U) snapshots for visualization
        converged = False
        it = 0
        while it < num_iters:
            self.step()  # compute_gradient leaves self.U for the current shape
            it += 1
            if record_every and (it % record_every == 0 or it == 1):
                self.frames.append((it, self.verts.numpy().copy(), self.U.numpy().copy()))
            if it % check_every == 0 or it == num_iters:
                loss = self._loss_from_U()
                if not quiet:
                    print(f"  iter {it:4d}  loss {loss:.6e}", flush=True)
                if loss < tol * init:
                    converged = True
                    break
        return {"iters": it, "initial_loss": init, "final_loss": self._loss_from_U(), "converged": converged}


def make_bridge(count):
    """15:1 triangulated-grid bridge, pinned at its left/right edges."""
    ny = 1 + count
    nx = 15 * (ny - 1) + 1
    gx, gy = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    V = np.stack([gx.ravel(), gy.ravel()], axis=1)
    V[:, 0] *= (nx - 1) / (ny - 1)
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * nx + i
            tris.append([idx, idx + 1, idx + nx + 1])
            tris.append([idx, idx + nx + 1, idx + nx])
    F = np.array(tris, dtype=np.int32)
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    fixed = np.array([i for i in range(V.shape[0]) if abs(V[i, 0] - xmin) < 1e-8 or abs(V[i, 0] - xmax) < 1e-8],
                     dtype=np.int32)  # fmt: skip
    return V.astype(np.float64), F, fixed


# ---------------------------------------------------------------------------
# Optional convergence visualization (polyscope headless). Self-contained and
# safe to delete: nothing above depends on it.
# ---------------------------------------------------------------------------


def per_face_von_mises(V, U, F, young, poisson):
    """Per-triangle von Mises stress from the CST strain of displacement U - V."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    C = np.array([[lam + 2 * mu, lam, 0.0], [lam, lam + 2 * mu, 0.0], [0.0, 0.0, mu]])
    Mg = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    vm = np.zeros(len(F))
    for f, tri in enumerate(F):
        Dm = Mg @ V[tri]
        G = np.linalg.inv(Dm) @ Mg
        B = np.array([[G[0, 0], 0, G[0, 1], 0, G[0, 2], 0],
                      [0, G[1, 0], 0, G[1, 1], 0, G[1, 2]],
                      [G[1, 0], G[0, 0], G[1, 1], G[0, 1], G[1, 2], G[0, 2]]])  # fmt: skip
        s = C @ (B @ (U[tri] - V[tri]).reshape(-1))
        vm[f] = np.sqrt(s[0] ** 2 - s[0] * s[1] + s[1] ** 2 + 3.0 * s[2] ** 2)
    return vm


def render_convergence_gif(frames, F, young, poisson, out_path, max_frames=60, fps=12):
    """Render a headless gif of the rest shape (top) and gravity-deformed shape
    (bottom, colored by von Mises stress) over the optimization."""
    import tempfile  # noqa: PLC0415

    import imageio.v2 as imageio  # noqa: PLC0415
    import polyscope as ps  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    if len(frames) > max_frames:  # keep <= max_frames, evenly spaced (last always in)
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in dict.fromkeys(idx)]

    ps.set_use_prefs_file(False)
    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")

    F3 = np.asarray(F, dtype=np.int32)
    span_y = max(U[:, 1].max() - U[:, 1].min() for _, _, U in frames)
    gap = span_y + 1.0  # stack the rest shape this far above the deformed shape
    all_U = np.vstack([U for _, _, U in frames])
    cx = 0.5 * (all_U[:, 0].min() + all_U[:, 0].max())
    cy = 0.5 * gap
    vmax = max(per_face_von_mises(V, U, F3, young, poisson).max() for _, V, U in frames)

    def to3d(P2, dy):
        return np.column_stack([P2[:, 0], P2[:, 1] + dy, np.zeros(len(P2))])

    tmp = tempfile.mkdtemp()
    shots = []
    for k, (_, V, U) in enumerate(frames):
        ps.register_surface_mesh("rest", to3d(V, gap), F3, color=(0.55, 0.68, 0.9), edge_width=0.5)
        defo = ps.register_surface_mesh("deformed", to3d(U, 0.0), F3, edge_width=0.5)
        defo.add_scalar_quantity("von Mises", per_face_von_mises(V, U, F3, young, poisson),
                                 defined_on="faces", vminmax=(0.0, vmax), cmap="viridis", enabled=True)  # fmt: skip
        ps.look_at((cx, cy, 2.0 * gap), (cx, cy, 0.0))
        p = f"{tmp}/f{k:04d}.png"
        ps.screenshot(p, transparent_bg=False)
        shots.append(np.asarray(Image.open(p).convert("RGB")))

    # Fixed crop to the content (non-white) bounding box across all frames, +margin.
    content = np.stack([(s < 245).any(axis=2) for s in shots]).any(axis=0)
    ys, xs = np.where(content)
    m = 20
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, shots[0].shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, shots[0].shape[1])
    imgs = []
    for (it, _, _), s in zip(frames, shots, strict=True):
        img = Image.fromarray(s[y0:y1, x0:x1])
        ImageDraw.Draw(img).text((10, 8), f"iteration {it}", fill=(20, 20, 20))
        imgs.append(np.asarray(img))
    imgs += [imgs[-1]] * fps  # hold the converged frame ~1s
    imageio.mimsave(out_path, imgs, fps=fps, loop=0)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--method", choices=("adam", "gn"), default="adam", help="Optimizer.")
    parser.add_argument("--num-iters", type=int, default=None, help="Iteration cap (defaults per method).")
    parser.add_argument("--step-size", type=float, default=1.0,
                        help="Gauss-Newton step size (1.0 is safe through count=8; finer meshes "
                             "need a smaller step, e.g. ~0.25 for count>=16, to avoid overshoot).")  # fmt: skip
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--gif", type=str, default=None, help="Render a headless convergence gif to this path.")
    parser.add_argument("--record-every", type=int, default=25, help="Iterations between recorded gif frames.")
    parser.add_argument("--fps", type=int, default=12, help="Gif frames per second.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        V, F, fixed = make_bridge(args.count)
        problem = InverseElasticity(V, F, fixed)
        if args.method == "gn":
            result = problem.gauss_newton_optimize(num_iters=args.num_iters or 30, step_size=args.step_size,
                                                   tol=args.tol, record_every=1 if args.gif else 0, quiet=args.quiet)  # fmt: skip
        else:
            result = problem.optimize(num_iters=args.num_iters or 8000, tol=args.tol,
                                      record_every=args.record_every if args.gif else 0, quiet=args.quiet)  # fmt: skip
        if args.gif:
            path = render_convergence_gif(problem.frames, F, problem.young, problem.poisson, args.gif, fps=args.fps)
            print(f"wrote {path} ({min(len(problem.frames), 60)} frames)")
        print(f"RESULT method={args.method} count={args.count} nV={V.shape[0]} iters={result['iters']} "
              f"initial_loss={result['initial_loss']:.6e} final_loss={result['final_loss']:.6e} "
              f"converged={result['converged']}")  # fmt: skip
