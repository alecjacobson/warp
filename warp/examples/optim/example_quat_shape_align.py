# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Quaternion Shape Alignment (one query vs a dataset)
#
# Matches one small query molecule against a dataset of candidate molecules
# by finding, for each candidate, the rigid rotation that best overlays the
# query onto it, then ranking candidates by the aligned residual.
#
# The overlay is scored with a Cauchy robust loss on the per-atom distances,
#
#     rho(s) = 0.5 * delta^2 * log(1 + s / delta^2).
#
# This is the point of the example: a plain sum-of-squared-distances fit is the
# Wahba/Procrustes problem, which has a closed-form solution from the SVD of a
# 3x3 cross-covariance -- no Newton needed. The robust loss is nonconvex, so
# there is no such direct eigen solution; the rotation must be found iteratively,
# and each step needs the exact Hessian.
#
# Translation is removed by centering each molecule, leaving a pure
# orientation fit. The orientation is a unit quaternion, optimized
# intrinsically on the rotation manifold: each Newton step works in a
# 3-vector tangent chart
#
#     q(dtheta) = exp_map(dtheta) * q,
#
# so the local problem is a 3x3 dense solve. The tangent gradient and the
# 3x3 Hessian come from a single forward pass of a second-order quaternion
# jet (wp.JetSpace2) -- no tape, no hand-derivatives, no reverse sweep.
#
# The whole dataset is fit in parallel: one thread per candidate runs its own
# Newton loop, with the exact Hessian's eigenvalues clamped to a positive floor
# (it can be indefinite far from the optimum) and an Armijo line search. A
# finite-difference gate up front checks the jet's gradient and Hessian against
# central differences on the active device.
#
###########################################################################

import numpy as np

import warp as wp

# Width 3: the SO(3) tangent chart. One forward pass yields the 3x3 Hessian.
J = wp.JetSpace2(3, wp.float64)

DTHETA_CLAMP = wp.float64(0.5)  # cap on a single Newton step, in radians
EIG_FLOOR = wp.float64(1.0e-6)  # floor for clamped Hessian eigenvalues
ARMIJO_C = wp.float64(1.0e-4)  # sufficient-decrease constant for the line search

# Cauchy robust-loss scale. A finite delta makes the per-atom loss nonconvex, so
# the fit is not a linear least-squares problem and has no closed-form
# Procrustes/SVD solution -- which is exactly what motivates an iterative Newton
# solve and the exact 3x3 Hessian computed here.
DELTA_NP = 0.5
DELTA = wp.float64(DELTA_NP)


@wp.func
def align_energy(
    dtheta: J.vec3,
    q0: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> J.scalar:
    """Cauchy robust alignment energy as a second-order jet.

    sum_a rho(s_a), s_a = || R(q) query_a - cand[c, a] ||^2,
    rho(s) = 0.5 * delta^2 * log(1 + s / delta^2).

    Nonconvex in the rotation, so it is not a linear least-squares (Wahba)
    problem and cannot be solved by a direct Procrustes/SVD fit; the exact
    Hessian drives an intrinsic Newton solve instead.
    """
    q = J.exp_map(dtheta) * q0
    d2 = DELTA * DELTA
    e = J.constant(wp.float64(0.0))
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        s = wp.dot(r, r)
        e = e + wp.float64(0.5) * d2 * wp.log(wp.float64(1.0) + s / d2)
    return e


@wp.func
def rotvec_to_quat(v: wp.vec3d) -> wp.quatd:
    """Rotation-vector exp map for the (plain, non-jet) pose update."""
    a = wp.length(v)
    if a < wp.float64(1.0e-12):
        return wp.quatd(0.0, 0.0, 0.0, 1.0)
    return wp.quat_from_axis_angle(v / a, a)


@wp.func
def energy_value(
    q: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> wp.float64:
    """Plain (non-jet) Cauchy energy, used to accept or reject a trial step."""
    d2 = DELTA * DELTA
    e = wp.float64(0.0)
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        s = wp.dot(r, r)
        e += wp.float64(0.5) * d2 * wp.log(wp.float64(1.0) + s / d2)
    return e


@wp.kernel
def grad_hess_at_identity(
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
    grad: wp.array[wp.vec3d],
    hess: wp.array[wp.mat33d],
):
    dtheta = J.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
    e = align_energy(dtheta, wp.quatd(0.0, 0.0, 0.0, 1.0), query, cand, c, n)
    grad[0] = e.grad
    hess[0] = e.hess


@wp.kernel
def fit_dataset(
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    n: int,
    iters: int,
    out_q: wp.array[wp.quatd],
    out_rmsd: wp.array[wp.float64],
):
    c = wp.tid()  # one thread per candidate
    q = wp.quatd(0.0, 0.0, 0.0, 1.0)

    for _it in range(iters):
        dtheta = J.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
        e = align_energy(dtheta, q, query, cand, c, n)
        g = e.grad
        f0 = e.value

        # PD-modified Newton direction (as in the reference MATLAB): the robust
        # Hessian can be indefinite away from the optimum, so symmetrize it,
        # clamp its eigenvalues to a positive floor -- flipping negative-
        # curvature directions rather than inflating the whole diagonal the way
        # Levenberg-Marquardt would -- and solve in the eigenbasis.
        hs = wp.float64(0.5) * (e.hess + wp.transpose(e.hess))
        eig_q, eig_d = wp.eig3(hs)
        gq = wp.transpose(eig_q) * g
        y = wp.vec3d(
            gq[0] / wp.max(eig_d[0], EIG_FLOOR),
            gq[1] / wp.max(eig_d[1], EIG_FLOOR),
            gq[2] / wp.max(eig_d[2], EIG_FLOOR),
        )
        step = -(eig_q * y)

        sn = wp.length(step)
        if sn > DTHETA_CLAMP:
            step = step * (DTHETA_CLAMP / sn)

        # Armijo backtracking along the clamped Newton direction.
        gts = wp.dot(g, step)
        alpha = wp.float64(1.0)
        accepted = int(0)
        for _ls in range(30):
            # rotvec_to_quat(...) and q are unit, so the product is unit in
            # exact arithmetic; normalize only mops up floating-point drift.
            qn = wp.normalize(rotvec_to_quat(alpha * step) * q)
            if energy_value(qn, query, cand, c, n) <= f0 + ARMIJO_C * alpha * gts:
                q = qn
                accepted = int(1)
                break
            alpha = wp.float64(0.5) * alpha

        if accepted == 0:
            break

    # Report a plain geometric RMSD (not the robust objective) so the ranking
    # is an interpretable distance.
    ssd = wp.float64(0.0)
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        ssd += wp.dot(r, r)
    out_rmsd[c] = wp.sqrt(ssd / wp.float64(n))
    out_q[c] = q


# --------------------------------------------------------------------------
# Dataset generation and a NumPy reference for the finite-difference gate.
# --------------------------------------------------------------------------


def _rand_rotation(rng, max_angle=np.pi):
    v = rng.standard_normal(3)
    ang = rng.uniform(0.0, max_angle)
    axis = v / np.linalg.norm(v)
    return np.array([*(np.sin(ang / 2) * axis), np.cos(ang / 2)])  # [x, y, z, w]


def _qrot(q, X):
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (R @ X.T).T


def make_dataset(n_atoms=12, n_candidates=64, n_matches=4, seed=0):
    """One query molecule and a dataset of candidates.

    ``n_matches`` candidates are rotated near-copies of the query (small
    structural noise); the rest are unrelated molecules. Everything is
    centered, so only orientation distinguishes them.
    """
    rng = np.random.default_rng(seed)

    query = rng.standard_normal((n_atoms, 3))
    query -= query.mean(0)

    cands = np.empty((n_candidates, n_atoms, 3))
    is_match = np.zeros(n_candidates, dtype=bool)
    match_ids = rng.choice(n_candidates, size=n_matches, replace=False)

    for c in range(n_candidates):
        if c in match_ids:
            noise = 0.02 * rng.standard_normal((n_atoms, 3))  # same molecule, slightly perturbed
            base = query + noise
            is_match[c] = True
            # A modest generating rotation, as in the reference MATLAB tests, so
            # the intrinsic Newton fit reliably reaches the global optimum.
            rot = _rand_rotation(rng, max_angle=np.deg2rad(50.0))
        else:
            base = rng.standard_normal((n_atoms, 3))  # a different molecule
            rot = _rand_rotation(rng)
        base -= base.mean(0)
        cands[c] = _qrot(rot, base)

    return query, cands, is_match


def _fd_grad_hess(query, cand_c, h=1e-5):
    # Cauchy robust energy around dtheta = 0, matching align_energy.
    def energy(dtheta):
        a = np.linalg.norm(dtheta)
        if a < 1e-12:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = dtheta / a
            q = np.array([*(np.sin(a / 2) * axis), np.cos(a / 2)])
        r = _qrot(q, query) - cand_c
        s = np.sum(r * r, axis=1)
        return np.sum(0.5 * DELTA_NP**2 * np.log(1.0 + s / DELTA_NP**2))

    g = np.zeros(3)
    H = np.zeros((3, 3))
    E = np.eye(3)
    f0 = energy(np.zeros(3))
    for i in range(3):
        g[i] = (energy(h * E[i]) - energy(-h * E[i])) / (2 * h)
        H[i, i] = (energy(h * E[i]) - 2 * f0 + energy(-h * E[i])) / h**2
    for i in range(3):
        for j in range(i + 1, 3):
            H[i, j] = (
                energy(h * E[i] + h * E[j])
                - energy(h * E[i] - h * E[j])
                - energy(-h * E[i] + h * E[j])
                + energy(-h * E[i] - h * E[j])
            ) / (4 * h**2)
            H[j, i] = H[i, j]
    return g, H


class Example:
    def __init__(self, n_atoms=12, n_candidates=64, n_matches=4, iters=25, seed=0):
        self.iters = iters
        query, cands, is_match = make_dataset(n_atoms, n_candidates, n_matches, seed)
        self.n_atoms = n_atoms
        self.n_candidates = n_candidates
        self.is_match = is_match
        self.query_np = query
        self.cands_np = cands

        self.query = wp.array([wp.vec3d(*row) for row in query], dtype=wp.vec3d)
        self.cands = wp.array(
            [[wp.vec3d(*row) for row in cands[c]] for c in range(n_candidates)],
            dtype=wp.vec3d,
        )

    def verify_derivatives(self, c=0, tol=1e-6):
        """Gate the jet gradient/Hessian against finite differences on-device."""
        grad = wp.zeros(1, dtype=wp.vec3d)
        hess = wp.zeros(1, dtype=wp.mat33d)
        wp.launch(grad_hess_at_identity, dim=1, inputs=[self.query, self.cands, c, self.n_atoms], outputs=[grad, hess])
        wp.synchronize_device()

        g_jet = np.array(grad.numpy()[0])
        H_jet = np.array(hess.numpy()[0])
        g_fd, H_fd = _fd_grad_hess(self.query_np, self.cands_np[c])

        g_err = np.max(np.abs(g_jet - g_fd))
        h_err = np.max(np.abs(H_jet - H_fd))
        sym = np.max(np.abs(H_jet - H_jet.T))
        print(f"finite-difference gate (candidate {c}):")
        print(f"  |grad_jet - grad_fd|_inf = {g_err:.3e}")
        print(f"  |hess_jet - hess_fd|_inf = {h_err:.3e}")
        print(f"  Hessian asymmetry        = {sym:.3e}")
        assert g_err < 1e-4 and h_err < 1e-3, "jet derivatives disagree with finite differences"
        return g_err, h_err

    def fit(self):
        out_q = wp.zeros(self.n_candidates, dtype=wp.quatd)
        out_rmsd = wp.zeros(self.n_candidates, dtype=wp.float64)
        wp.launch(
            fit_dataset,
            dim=self.n_candidates,
            inputs=[self.query, self.cands, self.n_atoms, self.iters],
            outputs=[out_q, out_rmsd],
        )
        wp.synchronize_device()
        return out_q.numpy(), out_rmsd.numpy()

    def report(self, rmsd, top=6):
        order = np.argsort(rmsd)
        print(f"\nbest {top} matches of {self.n_candidates} candidates (aligned RMSD):")
        print("  rank  candidate   RMSD      true match?")
        for rank, c in enumerate(order[:top]):
            print(f"  {rank + 1:>4}  {c:>9}   {rmsd[c]:.5f}   {'yes' if self.is_match[c] else 'no'}")

        # A clean separation: every planted match should rank above every decoy.
        match_rmsd = rmsd[self.is_match]
        decoy_rmsd = rmsd[~self.is_match]
        print(f"\nplanted matches: max RMSD {match_rmsd.max():.5f}")
        print(f"decoys:          min RMSD {decoy_rmsd.min():.5f}")
        separated = match_rmsd.max() < decoy_rmsd.min()
        print(f"matches separated from decoys: {separated}")
        return separated


def main(device=None):
    with wp.ScopedDevice(device):
        example = Example()
        example.verify_derivatives()
        _, rmsd = example.fit()
        example.report(rmsd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    args = parser.parse_known_args()[0]

    main(device=args.device)
