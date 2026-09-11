# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ``warp.geometry.poisson_disk_sample`` on the shared bunny mesh.

Runs in the Warp environment. Reports, per radius, our default candidate budget
(fast) and a higher budget chosen to match PhysicsNeMo's near-maximal point
count, so the head-to-head can be read at matched counts::

    uv run --with scipy tools/benchmarks/poisson_vs_physicsnemo/run_ours.py

See ``README.md`` for the full cross-environment workflow and results.
"""

import numpy as np
from common import RADIUS_FRACTIONS, best_of, load_mesh, min_dist_over_radius

import warp as wp
from warp.geometry import poisson_disk_sample

# Our default budget (12) is the fast setting; ~48 lifts the set toward maximal
# to match a dart-throwing sampler's count on this mesh (see README).
MULTIPLIERS = (12, 48)


def main():
    wp.init()
    device = "cuda:0"
    verts, faces, diag = load_mesh()
    v = wp.array(verts, dtype=wp.vec3, device=device)
    f = wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)

    print(f"warp.geometry.poisson_disk_sample  |  device={device}  bbox_diag={diag:.4f}")
    print(f"{'radius':>10} {'frac':>6} {'mult':>5} {'points':>8} {'best_ms':>9} {'pts/s':>12} {'min-d/r':>8}")
    for frac in RADIUS_FRACTIONS:
        r = frac * diag
        for mult in MULTIPLIERS:

            def once(r=r, mult=mult):
                out = poisson_disk_sample(v, f, radius=r, candidate_multiplier=mult, device=device)
                wp.synchronize_device()
                return out

            once()  # prime, then time
            best = best_of(once)
            pts = poisson_disk_sample(v, f, radius=r, candidate_multiplier=mult, device=device)[2].numpy()
            q = min_dist_over_radius(pts, r)
            print(f"{r:10.4f} {frac:6.1%} {mult:5d} {len(pts):8d} {best * 1e3:9.2f} {len(pts) / best:12.0f} {q:8.3f}")


if __name__ == "__main__":
    main()
