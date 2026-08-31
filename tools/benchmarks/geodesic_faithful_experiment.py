# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful reference to test whether geodesic *multiple-samples-per-cell*
actually helps over single-per-cell.

This is a slow, unambiguous host (NumPy) reference of Bowers et al.'s Program 1:
trial-major dart throwing (the k loop is *outside* the phase-group loop), a
per-cell candidate list, 27 phase groups, and a 5x5x5 conflict search. Euclidean
vs geodesic and single- vs multiple-samples-per-cell are toggles that share the
exact same resolver, so the only thing that varies between runs is the thing
under test. It runs on small meshes where quality (sample count / feature
coverage), not speed, is what matters.

    uv run tools/benchmarks/geodesic_faithful_experiment.py
"""

from collections import defaultdict

import numpy as np

SQRT3 = np.sqrt(3.0)


def geodesic_distance(p1, n1, p2, n2):
    d = p2 - p1
    de = np.linalg.norm(d)
    if de == 0.0:
        return 0.0
    v = d / de
    c1 = float(np.clip(n1 @ v, -1.0, 1.0))
    c2 = float(np.clip(n2 @ v, -1.0, 1.0))
    if abs(c1 - c2) < 1e-6:
        return de / np.sqrt(max(1.0 - c1 * c1, 1e-12))
    return de * (np.arcsin(c1) - np.arcsin(c2)) / (c1 - c2)


def poisson_faithful(pos, nrm, radius, *, geodesic, multi, seed=0):
    """Program 1 (trial-major dart throwing). Returns accepted candidate indices.

    pos, nrm are the presampled candidate positions and unit normals (in random
    order). Cell size is radius/sqrt(3); a cell holds one sample (``multi=False``)
    or a list (``multi=True``). k = the largest per-cell candidate count, i.e. we
    try every presampled point -- the paper's stated ideal for k.
    """
    n = len(pos)
    mu = radius / SQRT3
    lo = pos.min(0)
    cells = np.floor((pos - lo) / mu).astype(np.int64)

    cell_cands = defaultdict(list)  # cell -> candidate indices, in (random) list order
    for i in range(n):
        cell_cands[tuple(cells[i])].append(i)

    # Group valid cells by phase (coordinate mod 3), phases in a random order.
    rng = np.random.default_rng(seed)
    phases = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    rng.shuffle(phases)
    cells_by_phase = defaultdict(list)
    for cell in cell_cands:
        cells_by_phase[(cell[0] % 3, cell[1] % 3, cell[2] % 3)].append(cell)

    cell_samples = defaultdict(list)  # cell -> accepted indices
    r_sq = radius * radius

    def is_free(i, ci):
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nb = (ci[0] + dx, ci[1] + dy, ci[2] + dz)
                    for s in cell_samples[nb]:
                        d = pos[s] - pos[i]
                        if d @ d < r_sq:  # Euclidean prune (dg >= de)
                            if not geodesic:
                                return False
                            if geodesic_distance(pos[i], nrm[i], pos[s], nrm[s]) < radius:
                                return False
        return True

    k = max(len(v) for v in cell_cands.values())
    accepted = []
    for t in range(k):
        for phase in phases:
            for cell in cells_by_phase[phase]:
                if not multi and len(cell_samples[cell]) >= 1:
                    continue
                cands = cell_cands[cell]
                if t >= len(cands):
                    continue
                i = cands[t]
                if is_free(i, cells[i]):
                    cell_samples[cell].append(i)
                    accepted.append(i)
    return np.array(accepted, dtype=np.int64)


def presample_slab(gap, n_per_sheet, radius, seed=0):
    """Two disjoint unit sheets a distance ``gap`` apart, opposed normals."""
    rng = np.random.default_rng(seed)
    top_xy = rng.random((n_per_sheet, 2))
    bot_xy = rng.random((n_per_sheet, 2))
    top = np.column_stack([top_xy, np.full(n_per_sheet, gap)]).astype(np.float64)
    bot = np.column_stack([bot_xy, np.zeros(n_per_sheet)]).astype(np.float64)
    pos = np.vstack([top, bot])
    nrm = np.vstack([np.tile([0, 0, 1.0], (n_per_sheet, 1)), np.tile([0, 0, -1.0], (n_per_sheet, 1))])
    # Shuffle so the two sheets are interleaved in list order (random dart order).
    perm = rng.permutation(len(pos))
    return pos[perm], nrm[perm], 2.0  # area = two unit sheets


def presample_fold(dihedral_deg, n_per_face, radius, seed=0):
    """Two unit half-planes meeting at the y-axis with the given dihedral angle
    (a convex ridge). A sharp ridge gives large normal differences at the crease."""
    rng = np.random.default_rng(seed)
    half = np.radians(dihedral_deg) / 2.0
    # Face A: rotate +x plane down by (90-half) about y; use direction vectors.
    # Points param by (s in [0,1] along face, y in [0,1]); face A dir = (sin half, 0, cos half), B = (-sin half,0,cos half)
    dirA = np.array([np.sin(half), 0.0, np.cos(half)])
    dirB = np.array([-np.sin(half), 0.0, np.cos(half)])
    nA = np.array([-np.cos(half), 0.0, np.sin(half)])
    nB = np.array([np.cos(half), 0.0, np.sin(half)])
    sA = rng.random(n_per_face)
    yA = rng.random(n_per_face)
    sB = rng.random(n_per_face)
    yB = rng.random(n_per_face)
    A = sA[:, None] * dirA[None, :] + np.column_stack([np.zeros(n_per_face), yA, np.zeros(n_per_face)])
    B = sB[:, None] * dirB[None, :] + np.column_stack([np.zeros(n_per_face), yB, np.zeros(n_per_face)])
    pos = np.vstack([A, B])
    nrm = np.vstack([np.tile(nA, (n_per_face, 1)), np.tile(nB, (n_per_face, 1))])
    perm = rng.permutation(len(pos))
    return pos[perm], nrm[perm], 2.0


def run(pos, nrm, area, radius, label):
    n_eucl = len(poisson_faithful(pos, nrm, radius, geodesic=False, multi=False))
    n_gsin = len(poisson_faithful(pos, nrm, radius, geodesic=True, multi=False))
    n_gmul = len(poisson_faithful(pos, nrm, radius, geodesic=True, multi=True))
    n_est = area / (0.8660254 * radius * radius)
    print(
        f"{label:26s} eucl={n_eucl:4d}  geo-single={n_gsin:4d}  geo-MULTI={n_gmul:4d}  "
        f"(multi/single={n_gmul / max(n_gsin, 1):.2f}x, /eucl={n_gmul / max(n_eucl, 1):.2f}x)  n_est~{n_est:.0f}"
    )


def main():
    radius = 0.1
    cell = radius / SQRT3
    print(f"radius={radius}, cell=r/sqrt3={cell:.4f}, geodesic-separable only when de>r/1.571={radius / 1.571:.4f}\n")
    print("Thin slab (two parallel sheets), sweep gap d:")
    for d in (0.02, 0.03, 0.045, 0.057, 0.065, 0.085):
        pos, nrm, area = presample_slab(d, 1200, radius, seed=0)
        tag = "SAME-cell" if d < cell else "diff-cell"
        run(pos, nrm, area, radius, f"  d={d:.3f} ({tag})")

    print("\nSharp convex fold (two half-planes at a ridge), sweep dihedral angle:")
    for ang in (20, 40, 60, 90):
        pos, nrm, area = presample_fold(ang, 1400, radius, seed=0)
        run(pos, nrm, area, radius, f"  dihedral={ang} deg")


if __name__ == "__main__":
    main()
