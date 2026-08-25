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
# Passing --render-gif PATH animates the per-iteration fit as a grid of
# candidates, each colored by its current alignment error with the best fit
# boxed (needs the optional polyscope and Pillow packages; runs headless).
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


@wp.struct
class StepResult:
    q: wp.quatd
    improved: wp.int32


@wp.func
def newton_step(
    q: wp.quatd,
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    c: int,
    n: int,
) -> StepResult:
    """One eigenvalue-clamped Newton step with an Armijo line search."""
    dtheta = J.seed_vec3(wp.vec3d(0.0, 0.0, 0.0), 0, 1, 2)
    e = align_energy(dtheta, q, query, cand, c, n)
    g = e.grad
    f0 = e.value

    # PD-modified Newton direction (as in the reference MATLAB): the robust
    # Hessian can be indefinite away from the optimum, so symmetrize it, clamp
    # its eigenvalues to a positive floor -- flipping negative-curvature
    # directions rather than inflating the whole diagonal the way
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
    for _ls in range(30):
        # rotvec_to_quat(...) and q are unit, so the product is unit in exact
        # arithmetic; normalize only mops up floating-point drift.
        qn = wp.normalize(rotvec_to_quat(alpha * step) * q)
        if energy_value(qn, query, cand, c, n) <= f0 + ARMIJO_C * alpha * gts:
            return StepResult(qn, wp.int32(1))
        alpha = wp.float64(0.5) * alpha

    return StepResult(q, wp.int32(0))


@wp.func
def rmsd(q: wp.quatd, query: wp.array[wp.vec3d], cand: wp.array2d[wp.vec3d], c: int, n: int) -> wp.float64:
    """Plain geometric RMSD (not the robust objective), an interpretable score."""
    ssd = wp.float64(0.0)
    for a in range(n):
        r = wp.quat_rotate(q, query[a]) - cand[c, a]
        ssd += wp.dot(r, r)
    return wp.sqrt(ssd / wp.float64(n))


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
        res = newton_step(q, query, cand, c, n)
        q = res.q
        if res.improved == 0:
            break

    out_rmsd[c] = rmsd(q, query, cand, c, n)
    out_q[c] = q


@wp.kernel
def newton_step_inplace(
    query: wp.array[wp.vec3d],
    cand: wp.array2d[wp.vec3d],
    n: int,
    q_io: wp.array[wp.quatd],
):
    # One Newton step in place -- drives the rendered animation one frame at a
    # time. Thread c touches only index c, so the read-modify-write is safe.
    c = wp.tid()
    q_io[c] = newton_step(q_io[c], query, cand, c, n).q


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


def _nn_tree_bonds(points):
    """A nearest-neighbor spanning tree, so the molecule renders with edges.

    Correspondence is shared across query and candidates, so one bond list drawn
    on both makes the overlay easy to read. Rendering only.
    """
    n = len(points)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    in_tree = [0]
    remaining = set(range(1, n))
    edges = []
    while remaining:
        best_i, best_j, best_d = 0, -1, np.inf
        for i in in_tree:
            for j in remaining:
                if d[i, j] < best_d:
                    best_i, best_j, best_d = i, j, d[i, j]
        edges.append((best_i, best_j))
        in_tree.append(best_j)
        remaining.discard(best_j)
    return np.array(edges, dtype=np.int32)


def make_dataset(n_atoms=12, n_candidates=64, n_matches=4, seed=0):
    """One query molecule and a dataset of candidate molecules.

    Molecules are generated as small self-avoiding 3D chains with approximately
    fixed bond lengths and tetrahedral bond angles. This gives much more
    molecule-like ball-and-stick geometry than independent Gaussian points,
    while keeping the example deliberately simple.

    ``n_matches`` candidates are rotated, slightly distorted copies of the query;
    the remaining candidates are independently generated decoys.

    Everything is centered, so the fitting problem remains rotation-only.
    """
    rng = np.random.default_rng(seed)

    def random_molecule():
        # These are not intended as chemically exact units, just pleasant
        # ball-and-stick proportions.
        bond_length = 0.75
        bond_jitter = 0.05

        # For three consecutive atoms A-B-C, tetrahedral chemistry gives
        # angle ABC ~= 109.5 degrees. Since `prev` points A->B and the new
        # direction points B->C, their vector angle is 180 - 109.5 = 70.5 deg.
        turn_angle = np.deg2rad(70.5)
        angle_jitter = np.deg2rad(8.0)

        # Keep non-neighboring atoms from collapsing onto one another.
        min_separation = 0.62 * bond_length

        X = np.zeros((n_atoms, 3), dtype=np.float64)

        if n_atoms <= 1:
            return X

        # Arbitrary first bond; the entire molecule will subsequently be
        # subjected to a random rotation anyway.
        X[1] = np.array([bond_length, 0.0, 0.0])

        for i in range(2, n_atoms):
            prev = X[i - 1] - X[i - 2]
            prev /= np.linalg.norm(prev)

            accepted = False
            for _attempt in range(64):
                # Random unit direction perpendicular to the previous bond.
                u = rng.standard_normal(3)
                u -= np.dot(u, prev) * prev
                un = np.linalg.norm(u)
                if un < 1.0e-10:
                    continue
                u /= un

                theta = turn_angle + rng.normal(scale=angle_jitter)
                length = bond_length * (1.0 + rng.normal(scale=bond_jitter))

                direction = np.cos(theta) * prev + np.sin(theta) * u
                p = X[i - 1] + length * direction

                # Adjacent atoms are supposed to be close; only test against
                # atoms before the immediate predecessor.
                if i <= 2:
                    accepted = True
                else:
                    d = np.linalg.norm(X[: i - 1] - p, axis=1)
                    accepted = np.all(d > min_separation)

                if accepted:
                    X[i] = p
                    break

            if not accepted:
                # Extremely unlikely fallback: accept the last proposal rather
                # than complicating this toy generator with backtracking.
                X[i] = p

        # Make the generated molecule itself randomly oriented. This prevents
        # the query/decoys from sharing an accidental preferred construction
        # direction.
        X = _qrot(_rand_rotation(rng), X)

        X -= X.mean(axis=0)
        return X

    # ----------------------------------------------------------------------
    # Query molecule
    # ----------------------------------------------------------------------

    query = random_molecule()

    # ----------------------------------------------------------------------
    # Candidate dataset
    # ----------------------------------------------------------------------

    cands = np.empty((n_candidates, n_atoms, 3), dtype=np.float64)
    is_match = np.zeros(n_candidates, dtype=bool)

    n_matches = min(n_matches, n_candidates)
    match_ids = set(rng.choice(n_candidates, size=n_matches, replace=False))

    for c in range(n_candidates):
        if c in match_ids:
            # A true match has the same underlying geometry, with small
            # per-atom structural distortion. Subtracting the mean again
            # removes the tiny translation introduced by the noise.
            noise = 0.025 * rng.standard_normal((n_atoms, 3))
            base = query + noise
            base -= base.mean(axis=0)

            is_match[c] = True

            # Keep planted matches within a reasonably large but reliable
            # Newton basin from the identity initialization.
            rot = _rand_rotation(rng, max_angle=np.deg2rad(50.0))

        else:
            # Independently generated molecule: similar geometric statistics
            # and scale, but genuinely different shape.
            base = random_molecule()
            rot = _rand_rotation(rng)

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
        self.bonds = _nn_tree_bonds(query)

        self.query = wp.array([wp.vec3d(*row) for row in query], dtype=wp.vec3d)
        self.cands = wp.array(
            [[wp.vec3d(*row) for row in cands[c]] for c in range(n_candidates)],
            dtype=wp.vec3d,
        )

    def verify_derivatives(self, c=0, grad_tol=1e-4, hess_tol=1e-3):
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
        assert g_err < grad_tol and h_err < hess_tol, "jet derivatives disagree with finite differences"
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

    def record_trajectory(self):
        """Run the fit one Newton step at a time, recording every pose.

        Returns an array of shape ``(iters + 1, n_candidates, 4)``.
        """
        q_io = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n_candidates, 1)),
            dtype=wp.quatd,
        )
        poses = [q_io.numpy().copy()]
        for _ in range(self.iters):
            wp.launch(newton_step_inplace, dim=self.n_candidates, inputs=[self.query, self.cands, self.n_atoms, q_io])
            wp.synchronize_device()
            poses.append(q_io.numpy().copy())
        return np.stack(poses)

    def render(self, gif_path, fps=8, size=900):
        """Animate the per-iteration fit as a grid and save a GIF.

        Each cell shows one candidate (gray, static) with the query overlaid and
        rotated by that candidate's current estimate, colored by the current
        alignment error; the best-fit cell is boxed. Requires ``polyscope`` and
        ``Pillow``.
        """
        try:
            import polyscope as ps  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as err:
            raise ImportError(
                "Rendering requires the 'polyscope' and 'Pillow' packages. "
                "Install them with 'pip install polyscope Pillow'."
            ) from err

        poses = self.record_trajectory()
        frames_n, n_cand, _ = poses.shape
        n = self.n_atoms

        # Grid layout in the XY plane, one cell per candidate.
        cols = int(np.ceil(np.sqrt(n_cand)))
        span = float(np.abs(self.query_np).max())
        spacing = 2.8 * span
        offsets = np.array([[(c % cols) * spacing, -(c // cols) * spacing, 0.0] for c in range(n_cand)])

        edges = np.concatenate([self.bonds + c * n for c in range(n_cand)], axis=0)
        targets = np.concatenate([self.cands_np[c] + offsets[c] for c in range(n_cand)], axis=0)

        # Per-frame fitted atom positions and per-candidate error.
        fitted = np.zeros((frames_n, n_cand * n, 3))
        errs = np.zeros((frames_n, n_cand))
        for f in range(frames_n):
            for c in range(n_cand):
                fr = _qrot(poses[f, c], self.query_np)
                errs[f, c] = np.sqrt(np.mean(np.sum((fr - self.cands_np[c]) ** 2, axis=1)))
                fitted[f, c * n : (c + 1) * n] = fr + offsets[c]
        err_atom = np.repeat(errs, n, axis=1)
        vmax = float(np.percentile(errs[0], 85))

        def ring(center):
            h = 0.5 * spacing
            corners = center + np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
            return corners

        ring_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)

        ps.set_program_name("quat shape alignment")
        ps.set_use_prefs_file(False)
        ps.set_allow_headless_backends(True)
        ps.init()
        if hasattr(ps, "set_window_size"):
            ps.set_window_size(size, size)
        ps.set_ground_plane_mode("none")
        ps.set_view_projection_mode("orthographic")
        ps.set_SSAA_factor(2)

        gray = (0.72, 0.72, 0.74)
        atom_r, bond_r = 0.10 * span, 0.045 * span

        tgt_pc = ps.register_point_cloud("target atoms", targets, radius=atom_r, color=gray)
        tgt_pc.set_radius(atom_r, relative=False)
        ps.register_curve_network("target bonds", targets, edges, radius=bond_r, color=gray).set_radius(
            bond_r, relative=False
        )

        fit_pc = ps.register_point_cloud("fit atoms", fitted[0])
        fit_pc.set_radius(atom_r, relative=False)
        fit_pc.add_scalar_quantity("error", err_atom[0], enabled=True, cmap="viridis", vminmax=(0.0, vmax))
        fit_cn = ps.register_curve_network("fit bonds", fitted[0], edges)
        fit_cn.set_radius(bond_r, relative=False)
        fit_cn.add_scalar_quantity(
            "error", err_atom[0], defined_on="nodes", enabled=True, cmap="viridis", vminmax=(0.0, vmax)
        )

        win = ps.register_curve_network(
            "best fit", ring(offsets[int(np.argmin(errs[0]))]), ring_edges, color=(0.95, 0.45, 0.1)
        )
        win.set_radius(0.35 * bond_r, relative=False)

        center = offsets.mean(0)
        eye = center + spacing * cols * np.array([0.12, -0.18, 1.2])
        ps.look_at(tuple(eye), tuple(center))

        frames = []
        for f in range(frames_n):
            fit_pc.update_point_positions(fitted[f])
            fit_pc.add_scalar_quantity("error", err_atom[f], enabled=True, cmap="viridis", vminmax=(0.0, vmax))
            fit_cn.update_node_positions(fitted[f])
            fit_cn.add_scalar_quantity(
                "error", err_atom[f], defined_on="nodes", enabled=True, cmap="viridis", vminmax=(0.0, vmax)
            )
            win.update_node_positions(ring(offsets[int(np.argmin(errs[f]))]))
            buf = ps.screenshot_to_buffer(transparent_bg=False)
            frame = buf[..., :3]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))

        # Crop the shared whitespace margin so the grid fills the frame. The
        # union of per-frame content boxes keeps every rotating molecule inside.
        boxes = [np.asarray(f.convert("L")) < 250 for f in frames]
        rows = np.any([m.any(axis=1) for m in boxes], axis=0)
        cols_mask = np.any([m.any(axis=0) for m in boxes], axis=0)
        ys, xs = np.where(rows)[0], np.where(cols_mask)[0]
        if len(ys) and len(xs):
            # Clamp against the frame itself, not `size`: on a HiDPI display the
            # screenshot buffer is a multiple of the requested window size, and
            # clamping to `size` would crop away everything past the top-left.
            fw, fh = frames[0].size
            m = 16
            box = (
                max(int(xs[0]) - m, 0),
                max(int(ys[0]) - m, 0),
                min(int(xs[-1]) + m, fw),
                min(int(ys[-1]) + m, fh),
            )
            frames = [f.crop(box) for f in frames]

        # Per-frame duration in ms; hold longer on the converged final frame.
        durations = [int(1000 / fps)] * frames_n
        durations[-1] += 1500
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"wrote {gif_path} ({frames_n} frames, {n_cand} candidates)")


def main(device=None, render_gif=None):
    with wp.ScopedDevice(device):
        if render_gif is not None:
            example = Example(n_candidates=64, n_matches=6)
            example.verify_derivatives()
            example.render(render_gif, size=2000)
        else:
            example = Example()
            example.verify_derivatives()
            _, rmsd = example.fit()
            example.report(rmsd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--render-gif",
        type=str,
        default=None,
        help="Render the per-iteration fit as an animated grid GIF to this path (needs polyscope, Pillow).",
    )
    args = parser.parse_known_args()[0]

    main(device=args.device, render_gif=args.render_gif)
