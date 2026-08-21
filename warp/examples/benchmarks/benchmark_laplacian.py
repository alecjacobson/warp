# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare cotangent Laplacian assembly via ``warp.geometry`` and via ``warp.fem``.

The cotangent Laplacian of a triangle mesh is, from the physics point of view,
the P1 finite-element stiffness matrix of the Laplacian bilinear form
``int(grad(u) . grad(v))``. So the same operator can be assembled two ways:

  - :func:`warp.geometry.laplacian`, which evaluates the closed-form cotangent
    weights per triangle and pushes them straight into COO triplets;
  - :func:`warp.fem.integrate`, which builds a function space over the mesh and
    integrates the bilinear form with quadrature.

Both return the operator in the same positive semi-definite sign convention, so
the two agree outright.

Keeping the comparison fair takes some care, because most of the runtime of
either path is neither cotangents nor quadrature -- it is the shared
``warp.sparse`` assembly underneath. This benchmark therefore:

  - matches the accumulation precision. ``fem.integrate`` accumulates in
    ``float64`` by default, while ``laplacian`` is ``float32`` throughout, so
    the FEM side is asked for ``float32``;
  - matches the topology policy, timing a compact rebuild against a compact
    rebuild and a pattern refill against a pattern refill, rather than
    comparing one path's fast route against the other's slow one;
  - reports FEM under both its default construction and its faster
    row-compression construction, so the general machinery is not judged on a
    setting a real user would not pick;
  - verifies the two operators agree before timing them at all.

Setup is the subtlest of these. ``fem.Trimesh3D`` eagerly builds edge topology,
which radix-sorts the mesh's vertex indices, and the virtual fields build
per-element node restrictions. None of that depends on vertex positions, so a
caller with fixed connectivity pays it once -- but it is also what lets FEM
assemble without a global sort later, so hiding it makes the row-compression
construction look free when it is not. The benchmark therefore reports three
regimes, and which one applies depends on what the caller can amortize:

  - cold: nothing precomputed, both sides starting from positions and indices;
  - warm rebuild: connectivity-derived setup reused, sparsity pattern rebuilt;
  - topology reuse: sparsity pattern reused, coefficients overwritten.

Read the small and large resolutions differently. On a GPU the smaller meshes
are bound by host-side dispatch rather than by the mesh itself, so their
timings measure launch overhead and barely move as the mesh grows; the largest
resolution is where the numbers reflect actual assembly throughput.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

import numpy as np

import warp as wp
import warp.fem as fem
import warp.geometry
import warp.sparse

# Benchmark configuration
RESOLUTIONS = [64, 256, 1024]  # Grid resolution per side; the mesh has 2 * res^2 triangles
ITERATIONS = 20  # Number of timed iterations per variant
WARM_UP = 3  # Number of untimed warm-up iterations

# Match laplacian, which computes and stores float32 throughout.
ACCUMULATE_DTYPE = wp.float32


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Laplacian bilinear form, whose P1 stiffness matrix is the cotangent Laplacian."""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


def grid_surface(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An ``n`` by ``n`` grid of quads split into triangles, lifted out of the plane.

    The lift keeps the surface genuinely three-dimensional and gives the
    triangles a range of shapes, so neither assembly path sees a degenerate
    flat special case.
    """
    x = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    zz = 0.3 * np.sin(3.0 * xx) * np.cos(2.0 * yy)
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    lower, upper = idx[:-1, :-1], idx[1:, 1:]
    right, above = idx[1:, :-1], idx[:-1, 1:]
    indices = np.concatenate(
        [
            np.stack([lower, right, upper], axis=-1).reshape(-1, 3),
            np.stack([lower, upper, above], axis=-1).reshape(-1, 3),
        ]
    ).astype(np.int32)
    return points, indices


def make_fem_fields(points: wp.array, indices_2d: wp.array):
    """Build the FEM geometry, space, and virtual fields for the mesh.

    This depends only on connectivity, so a caller re-assembling a deforming
    mesh pays it once. It is inside the timed region for the cold regime and
    hoisted out of the warm ones.
    """
    geo = fem.Trimesh3D(tri_vertex_indices=indices_2d, positions=points)
    space = fem.make_polynomial_space(geo, degree=1)
    domain = fem.Cells(geometry=geo)
    return {
        "u": fem.make_trial(space=space, domain=domain),
        "v": fem.make_test(space=space, domain=domain),
    }


def relative_discrepancy(L, K, device) -> float:
    """Return the relative discrepancy between ``L`` and ``K``, via a matrix-vector product.

    Comparing the action on a random vector avoids densifying matrices that
    reach a million rows at the larger resolutions.
    """
    rng = np.random.default_rng(0)
    x = wp.array(rng.standard_normal(L.nrow).astype(np.float32), dtype=wp.float32, device=device)

    Lx = warp.sparse.bsr_mv(L, x).numpy()
    Kx = warp.sparse.bsr_mv(K, x).numpy()

    scale = np.abs(Lx).max()
    return float(np.abs(Lx - Kx).max() / scale) if scale > 0.0 else float("inf")


def time_variant(fn, device) -> tuple[float, float]:
    """Return the mean and standard deviation, in milliseconds, of ``ITERATIONS`` calls."""
    for _ in range(WARM_UP):
        fn()
    wp.synchronize_device(device)

    timings = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    return mean(timings), stdev(timings) if len(timings) > 1 else 0.0


def report(title: str, results: dict[str, tuple[float, float]], baseline: str) -> None:
    """Print one regime's timings, with ``baseline`` anchored at 1.00x."""
    print(f"\n    {title}")
    print(f"      {'variant':<52}{'mean (ms)':>11}{'stdev':>9}{'vs fem':>11}")
    reference = results[baseline][0]
    for name, (avg, dev) in results.items():
        marker = "  <- 1.00x anchor" if name == baseline else ""
        print(f"      {name:<52}{avg:>11.3f}{dev:>9.3f}{avg / reference:>10.2f}x{marker}")


def run_resolution(res: int, device) -> None:
    points_np, indices_np = grid_surface(res)
    num_points = points_np.shape[0]
    num_triangles = indices_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    indices_2d = wp.array(indices_np, dtype=int, device=device)

    fem_fields = make_fem_fields(points, indices_2d)

    # Correctness first: comparing the performance of two operators is only
    # meaningful once they are known to be the same operator.
    L = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.laplacian(points, indices, L)
    K = fem.integrate(stiffness_form, fields=fem_fields, accumulate_dtype=ACCUMULATE_DTYPE)

    print(f"\n  resolution {res}: {num_points} vertices, {num_triangles} triangles")
    print(f"    nnz: geometry {L.nnz_sync()}, fem {K.nnz_sync()}")
    print(f"    max relative discrepancy of (L - K) x: {relative_discrepancy(L, K, device):.3e}")

    def integrate(output, **bsr_options):
        return fem.integrate(
            stiffness_form,
            fields=fem_fields,
            output=output,
            accumulate_dtype=ACCUMULATE_DTYPE,
            bsr_options=bsr_options or None,
        )

    # Nothing precomputed: both sides start from positions and indices alone.
    # For FEM that means paying for the geometry, space, and virtual fields,
    # which is where its edge topology and its sort live.
    def fem_cold():
        return fem.integrate(
            stiffness_form,
            fields=make_fem_fields(points, indices_2d),
            accumulate_dtype=ACCUMULATE_DTYPE,
            bsr_options={"construction": "auto"},
        )

    report(
        "cold -- nothing precomputed",
        {
            "geometry.laplacian()": time_variant(lambda: warp.geometry.laplacian(points, indices), device),
            "geometry.laplacian(construction='row_compress')": time_variant(
                lambda: warp.geometry.laplacian(points, indices, construction="row_compress"), device
            ),
            "fem: Trimesh3D + space + fields + integrate": time_variant(fem_cold, device),
        },
        baseline="fem: Trimesh3D + space + fields + integrate",
    )

    # Connectivity-derived setup reused, sparsity pattern rebuilt each call.
    K_rebuild = fem.integrate(stiffness_form, fields=fem_fields, accumulate_dtype=ACCUMULATE_DTYPE)
    K_compress = fem.integrate(stiffness_form, fields=fem_fields, accumulate_dtype=ACCUMULATE_DTYPE)
    report(
        "warm rebuild -- FEM setup reused, sparsity pattern rebuilt",
        {
            "geometry.laplacian(out=)": time_variant(lambda: warp.geometry.laplacian(points, indices, L), device),
            "geometry.laplacian(out=, construction='row_compress')": time_variant(
                lambda: warp.geometry.laplacian(points, indices, L, construction="row_compress"), device
            ),
            "fem.integrate(output=)": time_variant(lambda: integrate(K_rebuild), device),
            "fem.integrate(output=, construction='auto')": time_variant(
                lambda: integrate(K_compress, construction="auto"), device
            ),
        },
        # Anchored on FEM's faster construction rather than its default, so the
        # comparison is against the best the general machinery can do.
        baseline="fem.integrate(output=, construction='auto')",
    )

    # Both sides keep an existing pattern and overwrite only the coefficients.
    K_masked = fem.integrate(stiffness_form, fields=fem_fields, accumulate_dtype=ACCUMULATE_DTYPE)
    report(
        "topology reuse -- existing pattern refilled",
        {
            "geometry.laplacian(out=, reuse_topology=True)": time_variant(
                lambda: warp.geometry.laplacian(points, indices, L, reuse_topology=True), device
            ),
            "fem.integrate(output=, topology='masked')": time_variant(
                lambda: integrate(K_masked, topology="masked"), device
            ),
        },
        baseline="fem.integrate(output=, topology='masked')",
    )

    # Every timed variant must still hold the operator verified above, or its
    # timing is measuring the wrong computation.
    print(
        f"    final discrepancy vs geometry: fem/masked {relative_discrepancy(L, K_masked, device):.3e}"
        f", fem/row-compress {relative_discrepancy(L, K_compress, device):.3e}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Device to run on. Defaults to the default device.")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=RESOLUTIONS,
        help="Grid resolutions per side to benchmark.",
    )
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        device = wp.get_device()
        print(f"Cotangent Laplacian assembly on '{device}'")
        print(f"Accumulation in {ACCUMULATE_DTYPE.__name__} on both sides.")
        print("FEM setup is timed in the cold regime and reused in the warm ones.")

        for res in args.resolutions:
            run_resolution(res, device)


if __name__ == "__main__":
    main()
