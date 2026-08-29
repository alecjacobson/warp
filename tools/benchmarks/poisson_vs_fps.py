# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare warp.geometry Poisson-disk sampling against farthest-point sampling.

Both methods select points from the *same* dense candidate pool (generated once
with warp.geometry.UniformSampler), and FPS is asked for exactly as many points
as the Poisson-disk sampler produced, so the comparison is apples-to-apples. We
report runtime, the minimum inter-sample distance, and the pair-correlation
function (the blue-noise spectrum). Both samplers run entirely in Warp on the
GPU -- no PyTorch.

    uv run --with usd-core --with matplotlib tools/benchmarks/poisson_vs_fps.py
"""

import os
import time

import numpy as np
from pxr import Usd, UsdGeom
from warp_fps import farthest_point_sampling_warp_batchsort

import warp as wp
import warp.examples
import warp.geometry as geo
from warp._src.geometry import _eval_positions_kernel

wp.init()
DEVICE = "cuda:0"


def time_candidate_gen(points, faces, num_candidates) -> float:
    """Time only the candidate-pool generation (UniformSampler + positions).

    FPS receives this pool ready-made, so charging it to the Poisson sampler but
    not to FPS is the *conservative* choice -- it makes our sampler look slower.
    """
    best = np.inf
    for _ in range(3):
        wp.synchronize_device()
        t0 = time.perf_counter()
        us = geo.UniformSampler(points, faces, device=DEVICE)
        cf, cuv = us.sample(num_candidates, seed=0)
        cp = wp.empty(num_candidates, dtype=wp.vec3, device=DEVICE)
        wp.launch(_eval_positions_kernel, dim=num_candidates, inputs=[us.mesh.id, cf, cuv], outputs=[cp], device=DEVICE)
        wp.synchronize_device()
        best = min(best, time.perf_counter() - t0)
    return best


def validate_against_author():
    """Reproduce the FPS author's reference point (N=1e6, k=1024).

    Reported by the author: 26.50 ms on an RTX 3090 Ti. A comparable time here
    confirms the vendored FPS runs at full speed (no accidental host stalls).
    """
    from warp_fps import farthest_point_sampling_warp_batchsort  # noqa: PLC0415

    rng = np.random.default_rng(0)
    p = rng.standard_normal((1_000_000, 3)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    farthest_point_sampling_warp_batchsort(p.copy(), 1024, return_time=True)  # warm up
    best = min(farthest_point_sampling_warp_batchsort(p.copy(), 1024, return_time=True)[1] for _ in range(5))
    print(f"[validation] FPS N=1e6 k=1024: {best * 1000:.1f} ms  (author: 26.5 ms on RTX 3090 Ti)\n")


def load_mesh(name):
    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), name))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            return (
                np.array(m.GetPointsAttr().Get(), dtype=np.float32),
                np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32),
            )
    raise RuntimeError("no mesh in " + name)


def min_distance(pts: np.ndarray, cell: float) -> float:
    """Smallest distance between two distinct points, via a uniform cell hash."""
    keys = np.floor((pts - pts.min(0)) / cell).astype(np.int64)
    buckets: dict = {}
    for idx, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(idx)
    best = np.inf
    for (cx, cy, cz), members in buckets.items():
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near.extend(buckets.get((cx + dx, cy + dy, cz + dz), ()))
        d = np.linalg.norm(pts[members][:, None] - pts[near][None], axis=-1)
        d[d == 0] = np.inf
        if d.size:
            best = min(best, float(d.min()))
    return best


def candidate_positions(pds) -> wp.array:
    """The exact candidate positions the Poisson sampler drew from."""
    cand_f, cand_uv = pds._sampler.sample(pds.num_candidates, seed=0)
    cand = wp.empty(pds.num_candidates, dtype=wp.vec3, device=DEVICE)
    wp.launch(
        _eval_positions_kernel,
        dim=pds.num_candidates,
        inputs=[pds._sampler.mesh.id, cand_f, cand_uv],
        outputs=[cand],
        device=DEVICE,
    )
    return cand


def main():
    validate_against_author()

    points, faces = load_mesh("bunny.usd")

    # "solve" = thinning the shared candidate pool to M points, the apples-to-apples
    # step (FPS is timed the same way -- it is handed the pool). "total" adds the
    # candidate generation FPS gets for free.
    header = (
        f"{'radius':>7} {'M(out)':>8} {'N(cand)':>9} {'PDS_solve':>9} {'PDS_total':>9} {'FPS_ms':>9} "
        f"{'FPS/solve':>9} {'md_p/r':>7} {'md_f/r':>7} {'peakP':>6} {'peakF':>6}"
    )
    print(header)

    # Warm up both samplers so JIT compilation is not counted in the timings.
    _pw = geo.PoissonDiskSampler(points, faces, radius=0.02, seed=0, device=DEVICE)
    farthest_point_sampling_warp_batchsort(candidate_positions(_pw).numpy(), 256, return_time=True)
    wp.synchronize_device()

    rows = []
    for radius in (0.02, 0.01, 0.005):
        # --- Our Poisson-disk sampler, end to end (best of 3) ---
        t_pds = np.inf
        for _ in range(3):
            wp.synchronize_device()
            t0 = time.perf_counter()
            pds = geo.PoissonDiskSampler(points, faces, radius=radius, seed=0, device=DEVICE)
            wp.synchronize_device()
            t_pds = min(t_pds, time.perf_counter() - t0)
        m = pds.num_samples
        area = pds.total_area

        # Split out candidate generation so PDS "solve" matches what FPS is timed on.
        t_gen = time_candidate_gen(points, faces, pds.num_candidates)
        t_solve = max(t_pds - t_gen, 0.0)

        cand_np = candidate_positions(pds).numpy()

        # --- Warp FPS (Kaolin algorithm) on the same pool, matched count (best of 3) ---
        t_fps = np.inf
        for _ in range(3):
            idx, dt = farthest_point_sampling_warp_batchsort(cand_np, m, return_time=True)
            t_fps = min(t_fps, dt)
        fps_np = cand_np[idx]
        fps_pts = wp.array(fps_np, dtype=wp.vec3, device=DEVICE)

        P = pds.points.numpy()
        md_p, md_f = min_distance(P, radius), min_distance(fps_np, radius)
        rp, gp = geo.pair_correlation(pds.points, area, r_max=4.0 * radius, num_bins=60, device=DEVICE)
        rf, gf = geo.pair_correlation(fps_pts, area, r_max=4.0 * radius, num_bins=60, device=DEVICE)
        wp.synchronize_device()

        print(
            f"{radius:>7.3f} {m:>8d} {pds.num_candidates:>9d} {t_solve * 1000:>8.1f}m {t_pds * 1000:>8.1f}m "
            f"{t_fps * 1000:>8.1f}m {t_fps / max(t_solve, 1e-9):>8.0f}x "
            f"{md_p / radius:>7.3f} {md_f / radius:>7.3f} {gp.max():>6.2f} {gf.max():>6.2f}"
        )
        rows.append((radius, m, rp, gp, rf, gf))

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        radius, m, rp, gp, rf, gf = rows[-1]
        plt.figure(figsize=(7, 4))
        plt.plot(rp / radius, gp, label="Poisson-disk (warp.geometry)", lw=2)
        plt.plot(rf / radius, gf, label="farthest-point (Warp)", lw=2)
        plt.axhline(1.0, color="gray", ls=":", lw=1)
        plt.axvline(1.0, color="gray", ls=":", lw=1, label="Poisson radius")
        plt.xlabel("pair distance / radius")
        plt.ylabel("pair correlation g(r)")
        plt.title(f"Blue-noise spectrum on the bunny ({m} points)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("/tmp/poisson_vs_fps_pcf.png", dpi=110)
        print("\nwrote /tmp/poisson_vs_fps_pcf.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
