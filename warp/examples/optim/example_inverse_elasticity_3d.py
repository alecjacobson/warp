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
import warp_cudss

import warp as wp
import warp.sparse as wps

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


# ---------------------------------------------------------------------------
# Assembly / solve kernels (3x3 blocks; 4 nodes per tet)
# ---------------------------------------------------------------------------


@wp.func
def _block(Ke: mat12, a: int, b: int) -> mat33:
    return mat33(Ke[3 * a + 0, 3 * b + 0], Ke[3 * a + 0, 3 * b + 1], Ke[3 * a + 0, 3 * b + 2],
                 Ke[3 * a + 1, 3 * b + 0], Ke[3 * a + 1, 3 * b + 1], Ke[3 * a + 1, 3 * b + 2],
                 Ke[3 * a + 2, 3 * b + 0], Ke[3 * a + 2, 3 * b + 1], Ke[3 * a + 2, 3 * b + 2])  # fmt: skip


@wp.kernel
def assemble_triplets(tets: wp.array2d(dtype=wp.int32), verts: wp.array(dtype=vec3), young: scalar, poisson: scalar,
                      vert_to_free: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
                      cols: wp.array(dtype=wp.int32), blocks: wp.array(dtype=mat33)):  # fmt: skip
    t = wp.tid()
    Ke = element_stiffness(verts[tets[t, 0]], verts[tets[t, 1]], verts[tets[t, 2]], verts[tets[t, 3]], young, poisson)
    for a in range(4):
        fa = vert_to_free[tets[t, a]]
        for b in range(4):
            fb = vert_to_free[tets[t, b]]
            slot = t * 16 + a * 4 + b
            if fa >= 0 and fb >= 0:
                rows[slot] = fa
                cols[slot] = fb
                blocks[slot] = _block(Ke, a, b)
            else:
                rows[slot] = 0
                cols[slot] = 0
                blocks[slot] = mat33(scalar(0.0))


@wp.kernel
def scatter_stiffness_inplace(tets: wp.array2d(dtype=wp.int32), verts: wp.array(dtype=vec3), young: scalar,
                              poisson: scalar, dst: wp.array(dtype=wp.int32), values: wp.array3d(dtype=scalar)):  # fmt: skip
    t = wp.tid()
    Ke = element_stiffness(verts[tets[t, 0]], verts[tets[t, 1]], verts[tets[t, 2]], verts[tets[t, 3]], young, poisson)
    for a in range(4):
        for b in range(4):
            blk = dst[t * 16 + a * 4 + b]
            if blk >= 0:
                for i in range(3):
                    for j in range(3):
                        wp.atomic_add(values, blk, i, j, Ke[3 * a + i, 3 * b + j])


@wp.kernel
def accumulate_vertex_mass(tets: wp.array2d(dtype=wp.int32), verts: wp.array(dtype=vec3), mass: wp.array(dtype=scalar)):
    """Lumped mass: each node gets vol/4 from every incident tet."""
    t = wp.tid()
    v4 = element_volume(verts[tets[t, 0]], verts[tets[t, 1]], verts[tets[t, 2]], verts[tets[t, 3]]) / scalar(4.0)
    for a in range(4):
        wp.atomic_add(mass, tets[t, a], v4)


@wp.kernel
def build_free_load(mass: wp.array(dtype=scalar), f_ext: wp.array(dtype=vec3),
                    vert_to_free: wp.array(dtype=wp.int32), load: wp.array(dtype=vec3)):  # fmt: skip
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        load[f] = mass[v] * f_ext[v]


@wp.kernel
def scatter_solution(q: wp.array(dtype=vec3), vert_to_free: wp.array(dtype=wp.int32), verts: wp.array(dtype=vec3),
                     u_full: wp.array(dtype=vec3), U: wp.array(dtype=vec3)):  # fmt: skip
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        u_full[v] = q[f]
        U[v] = verts[v] + q[f]
    else:
        u_full[v] = vec3(scalar(0.0))
        U[v] = verts[v]


@wp.kernel
def gather_free_residual(v_target: wp.array(dtype=vec3), U: wp.array(dtype=vec3),
                         vert_to_free: wp.array(dtype=wp.int32), r_free: wp.array(dtype=vec3)):  # fmt: skip
    v = wp.tid()
    f = vert_to_free[v]
    if f >= 0:
        r_free[f] = v_target[v] - U[v]


# --- Gauss-Newton assembly (sensitivity G_ff, the T = A + G_ff system) ---


@wp.func
def element_residual(p0: vec3, p1: vec3, p2: vec3, p3: vec3, young: scalar, poisson: scalar,
                     u_e: vec12, f_e: vec12) -> vec12:  # fmt: skip
    """Element equilibrium residual force ``M_e f_e - K_e u_e`` (lumped mass)."""
    vol4 = element_volume(p0, p1, p2, p3) / scalar(4.0)
    return vol4 * f_e - element_stiffness(p0, p1, p2, p3, young, poisson) * u_e


@wp.kernel
def scatter_sensitivity_inplace(tets: wp.array2d(dtype=wp.int32), verts: wp.array(dtype=vec3), young: scalar,
                                poisson: scalar, u_full: wp.array(dtype=vec3), f_ext: wp.array(dtype=vec3),
                                eps: scalar, dst: wp.array(dtype=wp.int32), values: wp.array3d(dtype=scalar)):  # fmt: skip
    """Refill G_ff, ``G_e[:,a] = d(M_e f_e - K_e u_e)/dx_a`` by central differences."""
    t = wp.tid()
    i0, i1, i2, i3 = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    p0, p1, p2, p3 = verts[i0], verts[i1], verts[i2], verts[i3]
    u_e = vec12(u_full[i0][0], u_full[i0][1], u_full[i0][2], u_full[i1][0], u_full[i1][1], u_full[i1][2],
               u_full[i2][0], u_full[i2][1], u_full[i2][2], u_full[i3][0], u_full[i3][1], u_full[i3][2])  # fmt: skip
    f_e = vec12(f_ext[i0][0], f_ext[i0][1], f_ext[i0][2], f_ext[i1][0], f_ext[i1][1], f_ext[i1][2],
               f_ext[i2][0], f_ext[i2][1], f_ext[i2][2], f_ext[i3][0], f_ext[i3][1], f_ext[i3][2])  # fmt: skip
    Ge = mat12(scalar(0.0))
    for a in range(12):
        comp = a % 3
        d = vec3(wp.where(comp == 0, eps, scalar(0.0)), wp.where(comp == 1, eps, scalar(0.0)),
                 wp.where(comp == 2, eps, scalar(0.0)))  # fmt: skip
        node = a // 3
        d0 = wp.where(node == 0, d, vec3(scalar(0.0)))
        d1 = wp.where(node == 1, d, vec3(scalar(0.0)))
        d2 = wp.where(node == 2, d, vec3(scalar(0.0)))
        d3 = wp.where(node == 3, d, vec3(scalar(0.0)))
        col = (element_residual(p0 + d0, p1 + d1, p2 + d2, p3 + d3, young, poisson, u_e, f_e)
               - element_residual(p0 - d0, p1 - d1, p2 - d2, p3 - d3, young, poisson, u_e, f_e)) / (scalar(2.0) * eps)  # fmt: skip
        for i in range(12):
            Ge[i, a] = col[i]
    for an in range(4):
        for bn in range(4):
            blk = dst[t * 16 + an * 4 + bn]
            if blk >= 0:
                for i in range(3):
                    for j in range(3):
                        wp.atomic_add(values, blk, i, j, Ge[3 * an + i, 3 * bn + j])


@wp.kernel
def add_blocks(a: wp.array3d(dtype=scalar), b: wp.array3d(dtype=scalar), out: wp.array3d(dtype=scalar)):
    """T = A + G_ff on the BSR scalar_values (what cuDSS reads on refactor)."""
    k = wp.tid()
    for i in range(3):
        for j in range(3):
            out[k, i, j] = a[k, i, j] + b[k, i, j]


@wp.kernel
def sub_free(a: wp.array(dtype=vec3), b: wp.array(dtype=vec3), out: wp.array(dtype=vec3)):
    i = wp.tid()
    out[i] = a[i] - b[i]


@wp.kernel
def apply_free_step(step: scalar, p_step: wp.array(dtype=vec3), free_verts: wp.array(dtype=wp.int32),
                    verts: wp.array(dtype=vec3)):  # fmt: skip
    i = wp.tid()
    verts[free_verts[i]] = verts[free_verts[i]] + step * p_step[i]


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class InverseElasticity3D:
    """3D bar whose rest shape is optimized so its gravity-sagged shape is flat."""

    def __init__(self, V, T, fixed, young=2e3, poisson=0.3, gravity=-9.8, device=None):
        self.device = wp.get_device(device)
        self.young, self.poisson = float(young), float(poisson)
        self.num_verts = int(V.shape[0])
        self.num_tets = int(T.shape[0])

        fixed_set = {int(i) for i in fixed}
        free = np.array([i for i in range(self.num_verts) if i not in fixed_set], dtype=np.int32)
        self.num_free = int(free.size)
        v2f = np.full(self.num_verts, -1, dtype=np.int32)
        v2f[free] = np.arange(self.num_free, dtype=np.int32)

        d = self.device
        self.verts = wp.array(V.astype(np.float64), dtype=vec3, device=d, requires_grad=True)
        self.tets = wp.array(T.astype(np.int32), dtype=wp.int32, device=d)
        self.free_verts = wp.array(free, dtype=wp.int32, device=d)
        self.vert_to_free = wp.array(v2f, dtype=wp.int32, device=d)
        self.v_target = wp.array(V.astype(np.float64), dtype=vec3, device=d)  # flat initial shape
        f_ext = np.zeros((self.num_verts, 3), dtype=np.float64)
        f_ext[:, 2] = gravity  # gravity along -z (the "height")
        self.f_ext = wp.array(f_ext, dtype=vec3, device=d)

        self._build_sparsity()

        nf, nv = self.num_free, self.num_verts
        self.mass = wp.zeros(nv, dtype=scalar, device=d)
        self.load = wp.zeros(nf, dtype=vec3, device=d)
        self.q = wp.zeros(nf, dtype=vec3, device=d)
        self.u_full = wp.zeros(nv, dtype=vec3, device=d)
        self.U = wp.empty(nv, dtype=vec3, device=d)
        self.r_free = wp.zeros(nf, dtype=vec3, device=d)

        self._assemble()
        self.solver = warp_cudss.CudssSolver(mtype="spd", device=d)
        self.solver.setup(self.A, self.q, self.load)

    def _build_sparsity(self):
        d = self.device
        nt = self.num_tets
        rows = wp.empty(nt * 16, dtype=wp.int32, device=d)
        cols = wp.empty(nt * 16, dtype=wp.int32, device=d)
        blocks = wp.empty(nt * 16, dtype=mat33, device=d)
        wp.launch(assemble_triplets, dim=nt,
                  inputs=[self.tets, self.verts, scalar(self.young), scalar(self.poisson),
                          self.vert_to_free, rows, cols, blocks], device=d)  # fmt: skip
        self.A = wps.bsr_from_triplets(self.num_free, self.num_free, rows, cols, blocks)
        nnz = self.nnz = self.A.nnz_sync()
        offsets = self.A.offsets.numpy()
        columns = self.A.columns.numpy()[:nnz]
        tets = self.tets.numpy()
        v2f = self.vert_to_free.numpy()
        dst = np.full(nt * 16, -1, dtype=np.int32)
        for t in range(nt):
            for a in range(4):
                fa = int(v2f[tets[t, a]])
                if fa < 0:
                    continue
                beg, end = int(offsets[fa]), int(offsets[fa + 1])
                rc = columns[beg:end]
                for b in range(4):
                    fb = int(v2f[tets[t, b]])
                    if fb < 0:
                        continue
                    k = int(np.searchsorted(rc, fb))
                    if k < len(rc) and int(rc[k]) == fb:
                        dst[t * 16 + a * 4 + b] = beg + k
        self.scatter_dst = wp.array(dst, dtype=wp.int32, device=d)

    def _assemble(self):
        self.A.scalar_values.zero_()
        wp.launch(scatter_stiffness_inplace, dim=self.num_tets,
                  inputs=[self.tets, self.verts, scalar(self.young), scalar(self.poisson),
                          self.scatter_dst, self.A.scalar_values], device=self.device)  # fmt: skip
        self.mass.zero_()
        wp.launch(accumulate_vertex_mass, dim=self.num_tets, inputs=[self.tets, self.verts, self.mass], device=self.device)
        wp.launch(build_free_load, dim=self.num_verts,
                  inputs=[self.mass, self.f_ext, self.vert_to_free, self.load], device=self.device)  # fmt: skip

    def forward(self):
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

    # -- Gauss-Newton --

    def _assemble_T(self):
        """Refill G_ff and T = A + G_ff on the scalar_values (what cuDSS reads)."""
        self.Gff.scalar_values.zero_()
        wp.launch(scatter_sensitivity_inplace, dim=self.num_tets,
                  inputs=[self.tets, self.verts, scalar(self.young), scalar(self.poisson),
                          self.u_full, self.f_ext, scalar(1e-6), self.scatter_dst, self.Gff.scalar_values], device=self.device)  # fmt: skip
        wp.launch(add_blocks, dim=self.nnz,
                  inputs=[self.A.scalar_values, self.Gff.scalar_values, self.T.scalar_values], device=self.device)  # fmt: skip

    def _setup_gn(self):
        if getattr(self, "_gn_ready", False):
            return
        d = self.device
        self.Gff = wps.bsr_copy(self.A)
        self.T = wps.bsr_copy(self.A)
        self.p_step = wp.zeros(self.num_free, dtype=vec3, device=d)
        self.gn_rhs = wp.zeros(self.num_free, dtype=vec3, device=d)
        self.gn_w = wp.zeros(self.num_free, dtype=vec3, device=d)
        self.forward()
        self._assemble_T()
        self.T_solver = warp_cudss.CudssSolver(mtype="general", device=d)
        self.T_solver.setup(self.T, self.gn_w, self.gn_rhs)
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
        self.T_solver.refactor(self.T)
        self.T_solver.solve(x=self.gn_w, b=self.gn_rhs)
        wp.launch(sub_free, dim=self.num_free, inputs=[self.r_free, self.gn_w, self.p_step], device=self.device)
        return self.p_step

    def gauss_newton_optimize(self, num_iters=20, step_size=1.0, tol=1e-8, record_every=0, quiet=True):
        """Damped Gauss-Newton on the rest shape with a fixed ``step_size``.

        The square-route system ``T = A + G_ff`` becomes ill-conditioned as the mesh
        refines, so the step size must shrink with resolution to avoid overshoot: ``1.0``
        works at coarse resolution, while finer meshes need progressively smaller steps.
        """
        init = self.loss()
        self.frames = [(0, self.verts.numpy().copy(), self.U.numpy().copy())] if record_every else []
        converged = False
        it = 0
        while it < num_iters:
            self.gauss_newton_step()
            wp.launch(apply_free_step, dim=self.num_free,
                      inputs=[scalar(step_size), self.p_step, self.free_verts, self.verts], device=self.device)  # fmt: skip
            it += 1
            loss = self.loss()
            if record_every and it % record_every == 0:
                self.frames.append((it, self.verts.numpy().copy(), self.U.numpy().copy()))
            if not quiet:
                print(f"  gn iter {it:3d}  loss {loss:.6e}", flush=True)
            if loss < tol * init:
                converged = True
                break
        return {"iters": it, "initial_loss": init, "final_loss": loss, "converged": converged}


# ---------------------------------------------------------------------------
# Optional convergence visualization (polyscope headless). Self-contained and
# safe to delete: nothing above depends on it.
# ---------------------------------------------------------------------------


def per_tet_von_mises(V, U, T, young, poisson):
    """Per-tet von Mises stress from the linear-tet strain of displacement U - V.

    Vectorized over tets (batched 3x3 inverse) so it stays fast on fine meshes."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d = lam + 2 * mu
    C = np.array([[d, lam, lam, 0, 0, 0], [lam, d, lam, 0, 0, 0], [lam, lam, d, 0, 0, 0],
                  [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]])  # fmt: skip
    Mg = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], dtype=float)
    P = V[T]  # (nT, 4, 3)
    Dm = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=1)  # (nT, 3, 3)
    G = np.linalg.inv(Dm) @ Mg  # (nT, 3, 4): dN_j/dx_i
    nT = len(T)
    B = np.zeros((nT, 6, 12))
    for j in range(4):
        B[:, 0, 3 * j] = G[:, 0, j]; B[:, 1, 3 * j + 1] = G[:, 1, j]; B[:, 2, 3 * j + 2] = G[:, 2, j]  # noqa: E702
        B[:, 3, 3 * j] = G[:, 1, j]; B[:, 3, 3 * j + 1] = G[:, 0, j]  # noqa: E702
        B[:, 4, 3 * j + 1] = G[:, 2, j]; B[:, 4, 3 * j + 2] = G[:, 1, j]  # noqa: E702
        B[:, 5, 3 * j] = G[:, 2, j]; B[:, 5, 3 * j + 2] = G[:, 0, j]  # noqa: E702
    disp = (U[T] - P).reshape(nT, 12)
    s = np.einsum("ij,nj->ni", C, np.einsum("nij,nj->ni", B, disp))  # (nT, 6) stress
    return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                   + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))  # fmt: skip


def render_convergence_gif(frames, T, young, poisson, out_path, fps=3):
    """Headless gif of the gravity-deformed shape (colored by von Mises stress)
    in front of its optimized rest shape, both sitting on a shadowed ground plane,
    over the Gauss-Newton iterations. The two shapes are placed in the same scene
    and separated along the camera's depth axis (not composited side by side)."""
    import tempfile  # noqa: PLC0415

    import imageio.v2 as imageio  # noqa: PLC0415
    import polyscope as ps  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    ps.set_use_prefs_file(False)
    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_window_size(1600, 1200)
    ps.set_SSAA_factor(3)  # supersample for crisp edges on the fine mesh
    ps.set_up_dir("z_up")
    ps.set_background_color((1.0, 1.0, 1.0))
    # Soft contact shadows on a ground plane held at a fixed height across frames.
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_blur_iters(6)
    ps.set_shadow_darkness(0.35)
    ps.set_ground_plane_height_mode("manual")

    Ti = np.asarray(T, dtype=np.int32)
    V0 = frames[0][1]
    span_x = V0[:, 0].max() - V0[:, 0].min()
    span_y = V0[:, 1].max() - V0[:, 1].min()
    depth = span_y + 2.0  # push the rest shape this far behind the deformed one (camera depth)
    vmax = max(per_tet_von_mises(V, U, Ti, young, poisson).max() for _, V, U in frames)
    z_floor = min(min(V[:, 2].min(), U[:, 2].min()) for _, V, U in frames) - 0.05
    ps.set_ground_plane_height(z_floor)

    # Fixed 3/4 view looking roughly along -y: the long (x) axis runs left-to-right,
    # z is up, and the deformed/rest shapes recede front-to-back in depth (y).
    cx = 0.5 * (V0[:, 0].min() + V0[:, 0].max())
    cy = 0.5 * (V0[:, 1].min() + V0[:, 1].max()) + 0.5 * depth
    cz = 0.5 * (z_floor + 1.0)

    def behind(P, dy):
        Q = P.copy(); Q[:, 1] += dy  # noqa: E702
        return Q

    tmp = tempfile.mkdtemp()
    shots = []
    for k, (_, V, U) in enumerate(frames):
        ps.register_volume_mesh("rest", behind(V, depth), tets=Ti, color=(0.6, 0.72, 0.92), edge_width=0.25)
        defo = ps.register_volume_mesh("deformed", U, tets=Ti, edge_width=0.25)
        defo.add_scalar_quantity("von Mises", per_tet_von_mises(V, U, Ti, young, poisson),
                                 defined_on="cells", vminmax=(0.0, vmax), cmap="viridis", enabled=True)  # fmt: skip
        if k == 0:
            ps.look_at((cx + 0.25 * span_x, cy - 1.9 * span_x, cz + 0.6 * span_x), (cx, cy, cz))
        p = f"{tmp}/f{k:04d}.png"
        # Draw twice: the ground-plane shadow map occasionally isn't populated on the
        # first draw after re-registering the meshes; the second capture has it.
        ps.screenshot(p, transparent_bg=False)
        ps.screenshot(p, transparent_bg=False)
        shots.append(np.asarray(Image.open(p).convert("RGB")))

    content = np.stack([(s < 245).any(axis=2) for s in shots]).any(axis=0)
    ys, xs = np.where(content)
    m = 20
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, shots[0].shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, shots[0].shape[1])
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    imgs = []
    for (it, _, _), s in zip(frames, shots, strict=True):
        img = Image.fromarray(s[y0:y1, x0:x1])
        ImageDraw.Draw(img).text((22, 16), f"iteration {it}", fill=(20, 20, 20), font=font)
        imgs.append(np.asarray(img))
    imgs += [imgs[-1]] * fps  # hold the converged frame ~1s
    imageio.mimsave(out_path, imgs, fps=fps, loop=0)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=1.0,
                        help="Gauss-Newton step size; must shrink as the mesh refines "
                             "(see gauss_newton_optimize). Too large a step diverges.")  # fmt: skip
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--gif", type=str, default=None, help="Render a headless convergence gif to this path.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        V, T, fixed = make_box(args.count)
        problem = InverseElasticity3D(V, T, fixed)
        result = problem.gauss_newton_optimize(num_iters=args.num_iters, step_size=args.step_size,
                                               tol=args.tol, record_every=1 if args.gif else 0, quiet=args.quiet)  # fmt: skip
        if args.gif:
            path = render_convergence_gif(problem.frames, T, problem.young, problem.poisson, args.gif)
            print(f"wrote {path} ({len(problem.frames)} frames)")
        print(f"RESULT count={args.count} nV={V.shape[0]} nT={T.shape[0]} iters={result['iters']} "
              f"initial_loss={result['initial_loss']:.6e} final_loss={result['final_loss']:.6e} "
              f"converged={result['converged']}")  # fmt: skip
