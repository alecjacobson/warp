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
