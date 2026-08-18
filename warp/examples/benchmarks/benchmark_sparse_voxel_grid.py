# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmark: sparse voxel grid (libigl-style flood fill in pure Warp)
#
# Two modes:
#
#   --mode compare (default)
#       Extract a mesh two ways and find the resolution crossover:
#         sparse : wp.geometry.sparse_voxel_grid discovers the near-surface
#                  cells from a seed, then wp.geometry.sparse_marching_cubes_
#                  from_cells meshes exactly those cells.
#         dense  : evaluate the field on a full grid and run the dense
#                  IsoSurfaceMarchingCubes.
#       Both produce the same mesh; everything stays on device. The implicit
#       function is a NON-SDF (a quadric sphere |p|^2-1), highlighting that
#       sparse_voxel_grid needs only a seed and sign changes -- no Lipschitz
#       / distance property -- unlike sparse_marching_cubes.
#
#   --mode profile
#       Phase breakdown of sparse_voxel_grid (traversal / vertex build /
#       end-to-end), production-vs-wavefront round counts, and a
#       batch_steps sweep, on signed distance functions.
#
# Note: requires a CUDA-capable device for representative timings.
###########################################################################

import argparse
import statistics
import time

import numpy as np

import warp as wp
import warp.geometry as wg
from warp._src.marching_cubes import MC_CUBE_CORNER_OFFSETS
from warp._src.sparse_voxel_grid import (
    _CORNER_OFFSETS,
    _DEFAULT_STACK_CAP,
    _build_vertices,
    _next_pow2,
    _traverse,
    _traverse_wavefront,
)

# =============================================================================
# Implicit functions
# =============================================================================


@wp.func
def sphere(p: wp.vec3) -> wp.float32:  # signed distance (for the profile mode)
    return wp.length(p) - 1.0


@wp.func
def torus_sdf(p: wp.vec3) -> wp.float32:
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - 0.7, p[1])
    return wp.length(q) - 0.25


# Quadric sphere: a genuine NON-SDF implicit (|p|^2 - 1, whose gradient magnitude
# 2|p| is not unit, so it is not a signed distance function). Its zero set is a
# unit sphere -- a 2-D surface whose bounding box is a full cube, so the dense
# cost (~box volume) grows a full power faster than the sparse cost (~surface
# area). Seed cell (0,0,0) with p0 on the surface straddles it.
SPHERE_SEED_P0 = (1.0, 0.0, 0.0)


@wp.func
def quadric_sphere(p: wp.vec3) -> wp.float32:
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1.0


@wp.kernel(enable_backward=False)
def quadric_sphere_field(field: wp.array3d(dtype=float), origin: wp.vec3, eps: float):
    i, j, k = wp.tid()
    field[i, j, k] = quadric_sphere(origin + eps * wp.vec3(float(i), float(j), float(k)))


# Algebraic torus: another NON-SDF (degree-4 polynomial). Its bounding box is
# still fully 3-D (all axes ~ n), so the dense cost is still O(n^3) and sparse
# still wins asymptotically -- but the box is thinner in y, so its smaller volume
# pushes the crossover to a higher resolution than the (cube-box) sphere.
_TORUS_R = 0.7
_TORUS_r = 0.25
TORUS_SEED_P0 = (_TORUS_R + _TORUS_r, 0.0, 0.0)  # (0.95, 0, 0) -- on the surface


@wp.func
def algebraic_torus(p: wp.vec3) -> wp.float32:
    s = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + wp.static(_TORUS_R * _TORUS_R - _TORUS_r * _TORUS_r)
    return s * s - wp.static(4.0 * _TORUS_R * _TORUS_R) * (p[0] * p[0] + p[2] * p[2])


@wp.kernel(enable_backward=False)
def algebraic_torus_field(field: wp.array3d(dtype=float), origin: wp.vec3, eps: float):
    i, j, k = wp.tid()
    field[i, j, k] = algebraic_torus(origin + eps * wp.vec3(float(i), float(j), float(k)))


# Non-SDF implicits for the compare mode: (scalar_func, dense_field_kernel, seed p0).
IMPLICITS = {
    "sphere": (quadric_sphere, quadric_sphere_field, SPHERE_SEED_P0),
    "torus": (algebraic_torus, algebraic_torus_field, TORUS_SEED_P0),
}


SDFS = {"sphere": (sphere, (1.0, 0.0, 0.0)), "torus": (torus_sdf, (0.95, 0.0, 0.0))}

# Permutation mapping a marching-cubes corner index to the libigl sparse-voxel
# corner index with the same {0,1}^3 offset, so sparse_voxel_grid's per-vertex
# field values (CS, indexed by CI in libigl order) can be reused directly as the
# marching-cubes corner values -- no re-evaluation.
_MC_FROM_LIBIGL_PERM = np.array([_CORNER_OFFSETS.index(off) for off in MC_CUBE_CORNER_OFFSETS], dtype=np.int32)


@wp.kernel(enable_backward=False)
def _gather_mc_corner_values(
    ci: wp.array(dtype=wp.int32, ndim=2),
    cs: wp.array(dtype=wp.float32),
    perm: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    m = wp.tid()
    for k in range(8):
        out[m, k] = cs[ci[m, perm[k]]]


# =============================================================================
# Timing
# =============================================================================


def _time(fn, device, iters):
    fn()
    wp.synchronize_device(device)
    samples = []
    for _ in range(iters):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _caps(expected):
    return expected, _next_pow2(4 * expected), _next_pow2(8 * expected)


# =============================================================================
# Mode: sparse-voxel-grid + from_cells  vs  dense marching cubes
# =============================================================================


def sparse_pipeline(func, p0, eps, expected, perm, device):
    """Discover near-surface cells, then mesh exactly those cells. On device."""
    _cv, cs, ci, cells = wg.sparse_voxel_grid(p0, func, eps, expected, device=device, return_cells=True)
    m = ci.shape[0]
    if m == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)
    corner_values = wp.empty((m, 8), dtype=wp.float32, device=device)
    wp.launch(_gather_mc_corner_values, dim=m, inputs=[ci, cs, perm], outputs=[corner_values], device=device)
    origin = wp.vec3(p0[0] - 0.5 * eps, p0[1] - 0.5 * eps, p0[2] - 0.5 * eps)
    return wg.sparse_marching_cubes_from_cells(cells, corner_values, origin=origin, cell_width=eps, device=device)


def _dense_setup(p0, eps, lo, hi):
    """Dense grid over the surface's cell bounding box, aligned to the sparse lattice.

    ``lo``/``hi`` are the min/max integer cell coordinates the sparse traversal
    discovered; the grid's corner nodes span ``[lo, hi]`` per axis, so it covers
    exactly the surface cells and shares the sparse corner lattice -- giving an
    identical mesh and the tightest fair dense baseline.
    """
    res = tuple(int(hi[a] - lo[a]) for a in range(3))
    origin = wp.vec3(*(p0[a] + eps * (float(lo[a]) - 0.5) for a in range(3)))
    upper = wp.vec3(*(origin[a] + eps * res[a] for a in range(3)))
    return res, origin, upper


def dense_pipeline(field_kernel, origin, upper, res, eps, device):
    """Evaluate the field on the full grid, then dense marching cubes. On device."""
    field = wp.empty((res[0] + 1, res[1] + 1, res[2] + 1), dtype=wp.float32, device=device)
    wp.launch(field_kernel, dim=field.shape, inputs=[field, origin, float(eps)], device=device)
    return wg.IsoSurfaceMarchingCubes.extract(
        field, 0.0, domain_bounds_lower_corner=origin, domain_bounds_upper_corner=upper
    )


def compare_dense(device, iters, max_dense_res, implicit):
    func, field_kernel, p0 = IMPLICITS[implicit]
    perm = wp.array(_MC_FROM_LIBIGL_PERM, dtype=wp.int32, device=device)

    print(f"\nDevice: {device}   implicit: {implicit} (non-SDF), seed cell (0,0,0) at p0={p0}\n")
    header = (
        f"{'1/eps':>6} {'dense res':>12} {'tris':>10} {'match':>6} "
        f"{'sparse ms':>10} {'dense ms':>10} {'speedup':>9} {'winner':>8}"
    )
    print(header)
    print("-" * len(header))

    dense_budget_ms = 20_000.0  # keep the worst dense run under ~20 s
    last_dense = None  # (grid_nodes, ms) of the previous dense run, for projection

    crossover = None
    for inv_eps in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536):
        eps = 1.0 / inv_eps
        expected = 24 * inv_eps * inv_eps

        # Setup (untimed): discover the cells once to get the surface bounding box
        # and the sparse triangle count.
        _, _, _, cells = wg.sparse_voxel_grid(p0, func, eps, expected, device=device, return_cells=True)
        cells_np = cells.numpy()
        lo, hi = cells_np.min(0), cells_np.max(0) + 1
        res, origin, upper = _dense_setup(p0, eps, lo, hi)
        res_str = f"{res[0]}x{res[1]}x{res[2]}"
        nodes = (res[0] + 1) * (res[1] + 1) * (res[2] + 1)

        _sv, si = sparse_pipeline(func, p0, eps, expected, perm, device)
        wp.synchronize_device(device)
        sparse_tris = len(si) // 3
        sparse_ms = (
            _time(
                lambda eps=eps, expected=expected: sparse_pipeline(func, p0, eps, expected, perm, device), device, iters
            )
            * 1e3
        )

        # Decide whether to run dense: skip if the grid is too big, or if the
        # projected runtime (scaled from the last dense run by node count) would
        # exceed the budget; the actual launch is guarded against out-of-memory.
        projected = last_dense[1] * nodes / last_dense[0] if last_dense else 0.0
        dense_ms = None
        skip = "skip"
        if max(res) > max_dense_res:
            skip = "res-cap"
        elif projected > dense_budget_ms:
            skip = f">{int(dense_budget_ms / 1000)}s"
        else:
            try:
                dv, di = dense_pipeline(field_kernel, origin, upper, res, eps, device)
                wp.synchronize_device(device)
                dense_tris = len(di) // 3
                match = "yes" if dense_tris == sparse_tris else f"{dense_tris}!"
                del dv, di
                dense_ms = (
                    _time(
                        lambda origin=origin, upper=upper, res=res, eps=eps: dense_pipeline(
                            field_kernel, origin, upper, res, eps, device
                        ),
                        device,
                        iters,
                    )
                    * 1e3
                )
                last_dense = (nodes, dense_ms)
            except Exception:
                skip = "n/a"

        if dense_ms is not None:
            speedup = dense_ms / sparse_ms
            winner = "sparse" if speedup > 1.0 else "dense"
            if crossover is None and speedup > 1.0:
                crossover = (inv_eps, res_str)
            print(
                f"{inv_eps:>6} {res_str:>12} {sparse_tris:>10,} {match:>6} "
                f"{sparse_ms:>10.3f} {dense_ms:>10.3f} {speedup:>8.2f}x {winner:>8}"
            )
        else:
            print(
                f"{inv_eps:>6} {res_str:>12} {sparse_tris:>10,} {'-':>6} "
                f"{sparse_ms:>10.3f} {skip:>10} {'--':>9} {'sparse':>8}"
            )

    if crossover is not None:
        print(f"\nCrossover: sparse overtakes dense at 1/eps = {crossover[0]} (dense grid {crossover[1]}).")
    else:
        print("\nNo crossover in the swept range (dense faster throughout, or all skipped).")
    print(
        "  Both paths produce the same mesh (triangle counts checked) and stay on device.\n"
        "  Sparse cost ~O(surface); dense cost ~O(res^3), so sparse wins as resolution grows.\n"
    )


# =============================================================================
# Mode: sparse_voxel_grid phase profiling
# =============================================================================


def profile_phases(device, sdf_name, iters, batch_steps):
    func, p0 = SDFS[sdf_name]
    print(f"\nDevice: {device}   SDF: {sdf_name}\n")
    header = (
        f"{'eps':>8} {'cells M':>9} {'verts':>9} {'rounds':>7} {'max_spill':>10} "
        f"{'A trav ms':>10} {'B+C ms':>9} {'D e2e ms':>9} {'cells/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for inv_eps in (16, 32, 64, 128):
        eps = 1.0 / inv_eps
        expected = 40 * inv_eps * inv_eps
        surf_cap, spill_cap, hash_cap = _caps(expected)

        surface, m, stats = _traverse(
            func,
            wp.vec3(p0),
            eps,
            0.0,
            (0, 0, 0),
            surf_cap,
            spill_cap,
            hash_cap,
            batch_steps,
            _DEFAULT_STACK_CAP,
            device,
        )
        _cv, _cs, _ci, n_unique = _build_vertices(func, surface, m, wp.vec3(p0), eps, device)
        wp.synchronize_device(device)

        a = (
            _time(
                lambda eps=eps, sc=surf_cap, pc=spill_cap, hc=hash_cap: _traverse(
                    func, wp.vec3(p0), eps, 0.0, (0, 0, 0), sc, pc, hc, batch_steps, _DEFAULT_STACK_CAP, device
                ),
                device,
                iters,
            )
            * 1e3
        )
        bc = (
            _time(
                lambda surface=surface, m=m, eps=eps: _build_vertices(func, surface, m, wp.vec3(p0), eps, device),
                device,
                iters,
            )
            * 1e3
        )
        d = (
            _time(
                lambda eps=eps, expected=expected: wg.sparse_voxel_grid(p0, func, eps, expected, device=device),
                device,
                iters,
            )
            * 1e3
        )
        print(
            f"{eps:>8.4f} {m:>9,} {n_unique:>9,} {stats['spill_round_count']:>7} {stats['max_spill_count']:>10,} "
            f"{a:>10.3f} {bc:>9.3f} {d:>9.3f} {m / (d / 1e3):>12,.0f}"
        )

    eps = 1.0 / 64.0
    surf_cap, spill_cap, hash_cap = _caps(40 * 64 * 64)
    _, mp, sp = _traverse(
        func, wp.vec3(p0), eps, 0.0, (0, 0, 0), surf_cap, spill_cap, hash_cap, batch_steps, _DEFAULT_STACK_CAP, device
    )
    _, _, sw = _traverse_wavefront(func, wp.vec3(p0), eps, 0.0, (0, 0, 0), surf_cap, spill_cap, hash_cap, device)
    print("\nProduction multi-step vs one-wave-per-launch reference (eps=1/64):")
    print(
        f"  multi-step: {sp['spill_round_count']:>3} rounds   wavefront: {sw['spill_round_count']:>3} rounds   (M={mp:,})"
    )

    print("\nbatch_steps sweep (eps=1/64):")
    for bs in (1, 4, 16, 64):
        _, _, s = _traverse(
            func, wp.vec3(p0), eps, 0.0, (0, 0, 0), surf_cap, spill_cap, hash_cap, bs, _DEFAULT_STACK_CAP, device
        )
        t = (
            _time(
                lambda bs=bs: _traverse(
                    func,
                    wp.vec3(p0),
                    eps,
                    0.0,
                    (0, 0, 0),
                    surf_cap,
                    spill_cap,
                    hash_cap,
                    bs,
                    _DEFAULT_STACK_CAP,
                    device,
                ),
                device,
                iters,
            )
            * 1e3
        )
        print(
            f"  batch_steps={bs:>3}: {s['spill_round_count']:>3} rounds  {t:8.3f} ms  max_spill={s['max_spill_count']:,}"
        )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", choices=("compare", "profile"), default="compare")
    parser.add_argument("--sdf", choices=("sphere", "torus"), default="sphere", help="Field for --mode profile.")
    parser.add_argument(
        "--implicit", choices=("sphere", "torus"), default="sphere", help="Non-SDF implicit for --mode compare."
    )
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--batch-steps", type=int, default=16)
    parser.add_argument("--max-dense-res", type=int, default=850, help="Skip dense above this grid resolution.")
    args = parser.parse_known_args()[0]

    device = wp.get_device(args.device)
    if args.mode == "compare":
        compare_dense(device, args.iters, args.max_dense_res, args.implicit)
    else:
        profile_phases(device, args.sdf, args.iters, args.batch_steps)


if __name__ == "__main__":
    main()
