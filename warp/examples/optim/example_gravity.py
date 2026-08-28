# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Gravity (particle-vs-mesh IPC contact)
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
# The distance d(p) = min(distance-to-mesh, distance-to-ground) and, crucially,
# its gradient and *Hessian* come from the feature-classified signed-distance
# derivatives in example_sdf_hessian.py -- the mesh query is a primal oracle and
# its analytic derivatives are injected by hand. One Warp thread solves one
# particle; the whole step is a single kernel launch.
#
# This is a Warp port of the test_gravity.m prototype. Rendering uses polyscope
# in headless (EGL) mode and writes an animated GIF.
#
# Extra dependencies (not required by Warp): polyscope, imageio. Run with e.g.
#   uv run --with polyscope --with imageio python -m warp.examples.optim.example_gravity
###########################################################################

import argparse
import os

import numpy as np

import warp as wp
import warp.examples
from warp.examples.optim.example_sdf_hessian import signed_distance_derivs

# Integrator tags (kept as ints so they can be passed to the kernel).
BACKWARD_EULER = wp.constant(0)
NEWMARK = wp.constant(1)


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
        pc = ps.register_point_cloud("particles", example.positions(), radius=0.006, color=(0.4, 0.2, 0.1))
        center = example.V.mean(axis=0)
        ps.look_at((center[0] + 2.0, center[1] - 2.5, center[2] + 1.5), tuple(center))

        dt = 1.0 / fps / nsubsteps
        n_frames = int(t_max * fps)
        frames = []
        tmp = out + ".frame.png"
        for _frame in range(n_frames):
            for _ in range(nsubsteps):
                example.step(dt)
            pc.update_point_positions(example.positions())
            ps.screenshot(tmp)
            frames.append(imageio.imread(tmp))
        if os.path.exists(tmp):
            os.remove(tmp)

        imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
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
