# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Barrier (particle-vs-mesh IPC contact)
#
# Particles fall under gravity onto a static mesh and a ground plane. Each
# particle is advanced independently (no particle-particle contact) as an
# incremental-potential minimization, in the style of C-IPC:
#
#     min_p  1/2 ||p - p_hat||^2 + c * kappa * B(d(p))
#
# with a log-barrier B keeping the distance d(p) above d_hat, a predictor
# p_hat and coefficient c that depend on the time integrator (backward Euler
# or Newmark), and an in-kernel Newton solve with backtracking line search.
#
# The distance d(p) = min(distance-to-mesh, distance-to-ground). Its gradient
# and, crucially, its *Hessian* come from the point-mesh signed distance whose
# closest feature (face/edge/vertex) is classified after the fact from the
# barycentric coordinates the mesh query returns -- the mesh query is a primal
# oracle and its analytic derivatives are supplied in closed form
# (``signed_distance_derivs`` / ``feature_tangent_projector`` below). One Warp
# thread solves one particle; the whole step is a single kernel launch.
#
# This is a Warp port of the test_gravity.m prototype. Rendering uses polyscope
# in headless (EGL) mode and writes an animated GIF.
#
# Extra dependencies (not required by Warp): polyscope, imageio. Run with e.g.
#   uv run --with polyscope --with imageio python -m warp.examples.optim.example_sdf_barrier
###########################################################################

import argparse
import os

import numpy as np

import warp as wp
import warp.examples

# Integrator tags (kept as ints so they can be passed to the kernel).
BACKWARD_EULER = wp.constant(0)
NEWMARK = wp.constant(1)

# Tolerance for calling a returned barycentric coordinate "zero", used to
# classify the closest feature (face/edge/vertex) after the fact from the
# ``(u, v)`` the mesh query returns.
#
# TAU_REPR repairs only the floating-point residue of an *exact* edge hit. Under
# strict IEEE binary32 the native closest-point routine (see
# ``closest_point_to_triangle`` in warp/native/intersect.h) sets one coordinate
# to zero on an edge, but packs the result as two floats and lets the third be
# reconstructed by subtraction; that reconstruction can leave a residue of up to
# 2^-25 = 1/4 FLT_EPSILON. Measured against the actual CPU and CUDA builds, the
# worst-case edge residue is exactly 2^-25, so the comparison must be inclusive
# (``<=``). This is a representation-repair constant, not a geometric tolerance:
# a face point whose true third weight is within 2^-25 of zero is deliberately
# classified as an edge (a face can lie arbitrarily close to an edge, so every
# positive tolerance absorbs a thin face collar into the edge case).
#
# TAU_GEOM is an optional, application-chosen geometric collar. Leave it at 0 for
# the narrowest post-facto convention; raise it to intentionally treat a band of
# near-edge faces as edges (collar behavior like the native sign query).
TAU_REPR = float.fromhex("0x1p-25")  # 2.9802322e-08
TAU_GEOM = 0.0
TAU = max(TAU_REPR, TAU_GEOM)


@wp.func
def feature_tangent_projector(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, u: float, v: float, n: wp.vec3) -> wp.mat33:
    """Projector ``T`` onto the directions the closest point is free to slide along.

    ==================  ================================  ============================
    Closest feature     Projector ``T``                   Resulting Hessian
    ==================  ================================  ============================
    face interior       ``I - n nᵀ``  (2 dims)            ``0`` (a plane is flat)
    edge                ``t tᵀ``      (1 dim, edge ``t``)  rank 1
    vertex              ``0``         (0 dims)             ``(s/dist)(I - n nᵀ)``, rank 2
    ==================  ================================  ============================

    The closest feature is recovered from the barycentric weights the mesh query
    returns. With ``w = 1 - u - v``, the weights map to the triangle vertices as
    ``u -> vertex 0, v -> vertex 1, w -> vertex 2`` (matching
    :func:`warp.mesh_eval_position`). Counting weights within :data:`TAU` of zero
    gives the feature: two zeros -> vertex, one zero -> edge (spanned by the two
    non-zero-weight vertices), none -> face interior. See :data:`TAU_REPR` for why
    that tolerance is what it is.

    ``.. note::`` **This recovers the triangle feature, not the surface feature.**
    The per-simplex curvature is exact only in the full-dimensional part of a
    feature's normal cone, and discrete curvature measures that cone's size:
    ``pi - dihedral`` for an edge, the angle defect ``2*pi - sum(theta)`` for a
    vertex. A flat feature has a collapsed cone, so the formula is spurious there:

    * a coplanar internal edge (dihedral ~ pi, e.g. a face diagonal) is read as an
      edge and given rank-1 curvature where the surface is flat and the Hessian is
      really zero;
    * a zero-defect vertex (a flat fan, or a subdivision point on a straight
      crease) is read as a vertex and given rank-2 curvature where the true feature
      is a face (zero) or an edge (rank 1).

    Fixing this needs adjacency / discrete curvature to demote flat features, which
    a single query does not return. But it only bites on the measure-zero set of
    points whose closest point lands exactly on such a feature -- probability zero
    for generic sampling -- so this example leaves it unhandled.
    """
    w = 1.0 - u - v
    zu = wp.abs(u) <= wp.static(TAU)
    zv = wp.abs(v) <= wp.static(TAU)
    zw = wp.abs(w) <= wp.static(TAU)

    # Two near-zero weights: closest point is a vertex, fixed as the query moves.
    if (zv and zw) or (zu and zw) or (zu and zv):
        return wp.mat33(0.0)

    # One near-zero weight: closest point slides along the edge of the other two
    # vertices. T removes that tangent direction from the curvature.
    if zu or zv or zw:
        if zw:
            t = wp.normalize(p1 - p0)  # weight of vertex 2 vanished -> edge (0, 1)
        elif zv:
            t = wp.normalize(p2 - p0)  # weight of vertex 1 vanished -> edge (0, 2)
        else:
            t = wp.normalize(p2 - p1)  # weight of vertex 0 vanished -> edge (1, 2)
        return wp.outer(t, t)

    # Face interior: the two free tangent directions are the whole tangent plane.
    return wp.identity(n=3, dtype=float) - wp.outer(n, n)


@wp.func
def signed_distance_derivs(mesh: wp.uint64, p: wp.vec3, max_dist: float):
    """Signed distance and its analytic gradient/Hessian at ``p`` (plain floats).

    Returns ``(hit, value, grad, hess)`` where ``hit`` is 0 when no surface lies
    within ``max_dist``. ``grad`` is the outward unit normal; ``hess`` is
    ``(s/dist)(I - n nᵀ - T)`` with ``T`` from :func:`feature_tangent_projector`.
    """
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    if not q.result:
        return 0, 0.0, wp.vec3(), wp.mat33()

    c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    r = p - c
    dist = wp.length(r)
    n = r / dist  # unit direction from the surface toward the query point
    s = q.sign  # +1 outside, -1 inside

    value = s * dist  # signed distance
    grad = s * n  # gradient of the signed distance is the outward unit normal

    m = wp.mesh_get(mesh)
    p0 = m.points[m.indices[q.face * 3 + 0]]
    p1 = m.points[m.indices[q.face * 3 + 1]]
    p2 = m.points[m.indices[q.face * 3 + 2]]
    tangent = feature_tangent_projector(p0, p1, p2, q.u, q.v, n)
    normal_proj = wp.identity(n=3, dtype=float) - wp.outer(n, n)
    hess = (s / dist) * (normal_proj - tangent)
    return 1, value, grad, hess


@wp.func
def distance_and_derivs(mesh: wp.uint64, p: wp.vec3, max_dist: float):
    """min(mesh SDF, ground plane z=0) with the closer feature's grad and Hessian."""
    hit, d_mesh, grad_mesh, hess_mesh = signed_distance_derivs(mesh, p, max_dist)
    d_ground = p[2]  # signed distance to the z = 0 ground plane
    if hit == 1 and d_mesh < d_ground:
        return d_mesh, grad_mesh, hess_mesh
    return d_ground, wp.vec3(0.0, 0.0, 1.0), wp.mat33(0.0)


@wp.func
def distance_value(mesh: wp.uint64, p: wp.vec3, max_dist: float) -> float:
    """Value-only distance, for the line search."""
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    d_mesh = float(1.0e30)
    if q.result:
        c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d_mesh = q.sign * wp.length(p - c)
    return wp.min(d_mesh, p[2])


@wp.func
def barrier_value(d: float, d_hat: float) -> float:
    """B(d) = -(d - d_hat)^2 log(d / d_hat) for 0 < d < d_hat, else 0 (inf if d <= 0)."""
    if d <= 0.0:
        return 1.0e30  # infeasible; large (not inf) so the line search rejects cleanly
    if d >= d_hat:
        return 0.0
    diff = d - d_hat
    return -diff * diff * wp.log(d / d_hat)


@wp.func
def barrier_derivs(d: float, dddp: wp.vec3, d2ddp2: wp.mat33, d_hat: float):
    """Barrier value, gradient, and Hessian in p, given d(p)'s derivatives.

    Chain rule: dB/dp = B'(d) dd/dp, d2B/dp2 = B''(d) (dd/dp)(dd/dp)^T + B'(d) d2d/dp2.
    """
    if d >= d_hat:
        return 0.0, wp.vec3(0.0, 0.0, 0.0), wp.mat33(0.0)
    diff = d - d_hat
    logdd = wp.log(d / d_hat)
    b = -diff * diff * logdd
    dbdd = -2.0 * diff * logdd - diff * diff / d
    d2bdd = -2.0 * logdd - 4.0 * diff / d + diff * diff / (d * d)
    dbdp = dbdd * dddp
    d2bdp2 = d2bdd * wp.outer(dddp, dddp) + dbdd * d2ddp2
    return b, dbdp, d2bdp2


@wp.func
def objective(mesh: wp.uint64, p: wp.vec3, p_hat: wp.vec3, ck: float, d_hat: float, max_dist: float) -> float:
    diff = p - p_hat
    d = distance_value(mesh, p, max_dist)
    return 0.5 * wp.dot(diff, diff) + ck * barrier_value(d, d_hat)


@wp.kernel(enable_backward=False)
def step_kernel(
    P: wp.array[wp.vec3],
    P_dot: wp.array[wp.vec3],
    P_ddot: wp.array[wp.vec3],
    mesh: wp.uint64,
    dt: float,
    gravity: wp.vec3,
    d_hat: float,
    max_dist: float,
    integrator: int,
    beta: float,
    gamma: float,
    max_inner: int,
    max_ls: int,
    P_out: wp.array[wp.vec3],
    P_dot_out: wp.array[wp.vec3],
    P_ddot_out: wp.array[wp.vec3],
):
    i = wp.tid()
    on = P[i]
    vn = P_dot[i]
    an = P_ddot[i]

    # Predictor p_hat and barrier coefficient c (integrator specific).
    if integrator == BACKWARD_EULER:
        p_hat = on + dt * vn + dt * dt * gravity
        c = dt * dt
    else:  # NEWMARK
        p_hat = on + dt * vn + dt * dt * (0.5 - beta) * an + beta * dt * dt * gravity
        c = beta * dt * dt

    # Adaptive barrier stiffness kappa, from the initial inertial vs barrier
    # gradient balance at p^n (clamped; 1 when there is no active contact).
    d0, g0, h0 = distance_and_derivs(mesh, on, max_dist)
    _b0, dbdp0, _hb0 = barrier_derivs(d0, g0, h0, d_hat)
    gc = c * dbdp0
    inertial = on - p_hat
    gcc = wp.dot(gc, gc)
    kappa = float(1.0)
    if gcc > 1.0e-20:
        kappa = wp.clamp(-wp.dot(gc, inertial) / gcc, 1.0e-5, 1.0e5)
    ck = c * kappa

    # Newton with backtracking line search on the incremental potential.
    p = on
    ident = wp.identity(n=3, dtype=float)
    for _it in range(max_inner):
        d, gd, hd = distance_and_derivs(mesh, p, max_dist)
        _bcur, dbdp, d2bdp2 = barrier_derivs(d, gd, hd, d_hat)
        g = (p - p_hat) + ck * dbdp
        if wp.length(g) < 1.0e-6:
            break
        H = ident + ck * d2bdp2
        dp = -(wp.inverse(H) @ g)
        if wp.dot(g, dp) >= 0.0:  # not a descent direction: fall back to gradient
            dp = -g

        f0 = objective(mesh, p, p_hat, ck, d_hat, max_dist)
        slope = wp.dot(g, dp)
        a = float(1.0)
        stepped = int(0)
        for _ls in range(max_ls):
            pt = p + a * dp
            if objective(mesh, pt, p_hat, ck, d_hat, max_dist) <= f0 + 0.3 * a * slope:
                p = pt
                stepped = 1
                break
            a = a * 0.5
        if stepped == 0:
            break

    # Velocity / acceleration update (integrator specific).
    if integrator == BACKWARD_EULER:
        v_next = (p - on) / dt
        a_next = an
    else:  # NEWMARK: invert the position update for a^{n+1}, then update velocity.
        a_next = (p - on - dt * vn - dt * dt * (0.5 - beta) * an) / (beta * dt * dt)
        v_next = vn + dt * ((1.0 - gamma) * an + gamma * a_next)

    P_out[i] = p
    P_dot_out[i] = v_next
    P_ddot_out[i] = a_next


def load_bunny():
    """Load the bundled bunny, normalized to the unit box with its base at z = 0."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
    V = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    F = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    # The bunny's native up-axis is +y; rotate +90 deg about x so it stands
    # upright (feet down) relative to gravity (-z): (x, y, z) -> (x, -z, y).
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    V = V @ Rx.T
    V = V - V.min(axis=0)
    V = V / np.abs(V).max()
    return V, F


class Example:
    def __init__(self, num_particles=1000, integrator="newmark", beta=1.0, gamma=1.0, seed=0):
        self.integrator = 1 if integrator == "newmark" else 0  # matches NEWMARK / BACKWARD_EULER
        self.beta = beta
        self.gamma = gamma
        self.d_hat = 0.01
        self.max_dist = 1.0e6
        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        self.V, self.F = load_bunny()
        self.mesh = wp.Mesh(
            points=wp.array(self.V, dtype=wp.vec3),
            indices=wp.array(self.F.reshape(-1), dtype=wp.int32),
        )

        rng = np.random.default_rng(seed)
        xy = rng.standard_normal((num_particles, 2)) * 0.1 + self.V[:, :2].mean(axis=0)
        z = self.V[:, 2].max() + self.d_hat * 10.0 + self.d_hat * rng.standard_normal((num_particles, 1))
        P0 = np.hstack([xy, z]).astype(np.float32)

        # Categorical colors (ColorBrewer Set1 + extras) assigned per particle.
        palette = (
            np.array(
                [0xE41A1C, 0x377EB8, 0x4DAF4A, 0x984EA3, 0xFF7F00, 0xFFFF33, 0xA65628, 0xF781BF],
                dtype=np.uint32,
            )[:, None]
            >> np.array([16, 8, 0], dtype=np.uint32)
        ) & 0xFF
        self.colors = (palette / 255.0)[rng.integers(0, len(palette), num_particles)].astype(np.float32)

        self.P = wp.array(P0, dtype=wp.vec3)
        self.P_dot = wp.zeros(num_particles, dtype=wp.vec3)
        self.P_ddot = wp.array(np.tile(self.gravity, (num_particles, 1)).astype(np.float32), dtype=wp.vec3)
        self.P_out = wp.zeros_like(self.P)
        self.P_dot_out = wp.zeros_like(self.P_dot)
        self.P_ddot_out = wp.zeros_like(self.P_ddot)

    def step(self, dt, max_inner=40, max_ls=40):
        wp.launch(
            step_kernel,
            dim=len(self.P),
            inputs=[
                self.P,
                self.P_dot,
                self.P_ddot,
                self.mesh.id,
                dt,
                self.gravity,
                self.d_hat,
                self.max_dist,
                self.integrator,
                self.beta,
                self.gamma,
                max_inner,
                max_ls,
            ],
            outputs=[self.P_out, self.P_dot_out, self.P_ddot_out],
        )
        self.P, self.P_out = self.P_out, self.P
        self.P_dot, self.P_dot_out = self.P_dot_out, self.P_dot
        self.P_ddot, self.P_ddot_out = self.P_ddot_out, self.P_ddot

    def positions(self):
        return self.P.numpy()


def main(
    device=None,
    num_particles=1000,
    integrator="newmark",
    beta=1.0,
    gamma=1.0,
    nsubsteps=10,
    fps=30,
    t_max=3.0,
    out=None,
):
    import imageio.v2 as imageio  # noqa: PLC0415
    import polyscope as ps  # noqa: PLC0415

    if out is None:
        out = os.path.join(os.path.dirname(__file__), f"../../../media/gravity_bunny_{integrator}.gif")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with wp.ScopedDevice(device):
        example = Example(num_particles, integrator, beta, gamma)

        ps.set_allow_headless_backends(True)
        ps.set_program_name("warp gravity")
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("z_up")
        ps.register_surface_mesh("mesh", example.V, example.F, color=(0.85, 0.72, 0.55), smooth_shade=True)
        pc = ps.register_point_cloud("particles", example.positions(), radius=0.006)
        pc.add_color_quantity("color", example.colors, enabled=True)
        center = example.V.mean(axis=0)
        ps.look_at((center[0] + 2.0, center[1] - 2.5, center[2] + 1.5), tuple(center))

        dt = 1.0 / fps / nsubsteps
        n_frames = int(t_max * fps)
        frames = []
        for _frame in range(n_frames):
            for _ in range(nsubsteps):
                example.step(dt)
            pc.update_point_positions(example.positions())
            # Opaque RGB frames: a transparent background would make the GIF
            # composite each frame over the last, smearing the moving particles.
            frames.append(ps.screenshot_to_buffer(transparent_bg=False)[:, :, :3])

        imageio.mimsave(out, frames, duration=1000.0 / fps, loop=0)
        print(f"wrote {out} ({len(frames)} frames)")
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--integrator", choices=["newmark", "backward-euler"], default="newmark")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nsubsteps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_known_args()[0]

    main(
        device=args.device,
        num_particles=args.num_particles,
        integrator=args.integrator,
        beta=args.beta,
        gamma=args.gamma,
        nsubsteps=args.nsubsteps,
        fps=args.fps,
        t_max=args.t_max,
        out=args.out,
    )
