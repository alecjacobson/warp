# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Bounded Biharmonic Weights (Wang & Solomon-style)
#
# Computes smooth, bounded skinning weights by optimizing a per-edge metric
# rather than solving a constrained biharmonic QP.
#
# The weights for each handle are the harmonic function of a DEC-factored
# Laplacian  L(s) = d0^T diag(s) d0  with the handle as Dirichlet data. For any
# positive metric s > 0, harmonic weights with indicator boundary data are
# automatically in [0, 1] (maximum principle) and sum to one across handles
# (the harmonic extension of the constant 1 is 1), so the Bounded-Biharmonic-
# Weights bound and partition-of-unity constraints hold for free -- no QP.
#
# Plain harmonic weights (s = the cotangent weights) are not smooth, though. So,
# following Wang & Solomon's "Fast Quasi-Harmonic Weights", we optimize the
# metric s > 0 to minimize the biharmonic energy  sum_h w_h^T (L0 M^-1 L0) w_h
# of the resulting weights, where L0 is the fixed embedding Laplacian. The
# gradient through the harmonic solve is the adjoint VJP of
# warp.geometry.MetricHarmonicSolver, and s is kept positive by optimizing its
# logarithm with Adam.
#
# The result stays bounded and a partition of unity throughout while its
# biharmonic (smoothness) energy drops.
#
# Usage:
#   uv run --with usd-core warp/examples/geometry/example_bounded_biharmonic_weights.py
###########################################################################

import time

import numpy as np

import warp as wp
import warp.geometry
import warp.optim
import warp.render
import warp.sparse


@wp.kernel
def _exp(src: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = wp.exp(src[wp.tid()])


@wp.kernel
def _multiply(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = a[wp.tid()] * b[wp.tid()]


@wp.kernel
def _scale_2d(src: wp.array2d(dtype=float), factor: float, out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    out[i, j] = factor * src[i, j]


@wp.kernel
def _reciprocal(src: wp.array(dtype=float), out: wp.array(dtype=float)):
    out[wp.tid()] = 1.0 / src[wp.tid()]


@wp.kernel
def _get_column(src: wp.array2d(dtype=float), col: int, out: wp.array(dtype=float)):
    out[wp.tid()] = src[wp.tid(), col]


@wp.kernel
def _set_column(dst: wp.array2d(dtype=float), col: int, src: wp.array(dtype=float)):
    dst[wp.tid(), col] = src[wp.tid()]


def build_grid(resolution: int):
    """A flat square triangle grid in the z=0 plane and four interior handle vertices."""
    axis = np.linspace(0.0, 1.0, resolution + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).astype(np.float32)

    grid = np.arange((resolution + 1) ** 2).reshape(resolution + 1, resolution + 1)
    lower, upper, right, above = grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1], grid[:-1, 1:]
    triangles = np.concatenate(
        [np.stack([lower, right, upper], -1).reshape(-1, 3), np.stack([lower, upper, above], -1).reshape(-1, 3)]
    ).astype(np.int32)

    def nearest(fx, fy):
        return int(grid[round(fx * resolution), round(fy * resolution)])

    handles = np.array([nearest(0.25, 0.25), nearest(0.75, 0.25), nearest(0.25, 0.75), nearest(0.75, 0.75)], np.int32)
    return points, triangles, handles


def biharmonic_energy_operator(laplacian, mass_inverse, device):
    """Return a callable applying the biharmonic operator ``L M^-1 L`` column by column."""

    def apply(weights_2d):
        n, num_handles = weights_2d.shape
        out = wp.zeros((n, num_handles), dtype=wp.float32, device=device)
        column = wp.empty(n, dtype=wp.float32, device=device)
        for h in range(num_handles):
            wp.launch(_get_column, dim=n, inputs=[weights_2d, h], outputs=[column], device=device)
            scaled = warp.sparse.bsr_mv(laplacian, column)
            wp.launch(_multiply, dim=n, inputs=[scaled, mass_inverse], outputs=[scaled], device=device)
            energy_column = warp.sparse.bsr_mv(laplacian, scaled)
            wp.launch(_set_column, dim=n, inputs=[out, h], outputs=[energy_column], device=device)
        return out

    return apply


def _report(label, weights, apply_biharmonic):
    w = weights.numpy()
    energy = float(np.sum(w * apply_biharmonic(weights).numpy()))
    partition = np.abs(w.sum(axis=1) - 1.0).max()
    print(
        f"  {label:>8}: biharmonic energy {energy:10.4f}   weights in [{w.min():+.4f}, {w.max():+.4f}]   |sum-1| {partition:.2e}"
    )
    return energy


def render_polyscope(points_np, triangles_np, weights, screenshot_path):
    """Render the weight fields with polyscope, headless, and save a screenshot."""
    import polyscope as ps  # noqa: PLC0415

    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_ground_plane_mode("none")
    mesh = ps.register_surface_mesh("mesh", points_np, triangles_np, edge_width=1.0)
    w = weights.numpy()
    for handle in range(w.shape[1]):
        mesh.add_scalar_quantity(f"weight {handle}", w[:, handle], enabled=(handle == 0), cmap="viridis")
    ps.look_at((0.5, 0.5, 1.6), (0.5, 0.5, 0.0))
    ps.screenshot(screenshot_path, transparent_bg=False)
    print(f"Saved polyscope screenshot to {screenshot_path}")


def main(
    stage_path="example_bounded_biharmonic_weights.usd",
    resolution=48,
    num_iters=80,
    lr=0.1,
    polyscope_screenshot=None,
    headless=False,
):
    device = wp.get_device()
    points_np, triangles_np, handles_np = build_grid(resolution)
    num_points = points_np.shape[0]
    num_handles = handles_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(triangles_np.flatten(), dtype=wp.int32, device=device)

    # Fixed embedding operators for the smoothness loss.
    d0, cotangent = warp.geometry.dec_operators(points, indices)
    embedding_laplacian = warp.geometry.laplacian(points, indices)
    mass = warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI)
    mass_inverse = wp.empty(num_points, dtype=wp.float32, device=device)
    wp.launch(
        _reciprocal, dim=num_points, inputs=[warp.sparse.bsr_get_diag(mass)], outputs=[mass_inverse], device=device
    )
    apply_biharmonic = biharmonic_energy_operator(embedding_laplacian, mass_inverse, device)

    boundary = wp.array(handles_np, dtype=wp.int32, device=device)
    boundary_values = wp.array(np.eye(num_handles, dtype=np.float32), dtype=wp.float32, device=device)
    solver = warp.geometry.MetricHarmonicSolver(d0, num_points, boundary, tol=1e-6, max_iters=4 * num_points)

    # Optimize the logarithm of the metric so it stays strictly positive. Start
    # from the cotangent weights, clamped positive.
    log_metric = wp.array(np.log(np.maximum(cotangent.numpy(), 1e-4)), dtype=wp.float32, device=device)
    optimizer = warp.optim.Adam([log_metric], lr=lr)

    metric = wp.empty_like(log_metric)
    num_edges = cotangent.shape[0]
    print(f"Bounded biharmonic weights: {num_points} vertices, {triangles_np.shape[0]} triangles, {num_edges} edges")
    print(f"Solving {num_handles} weights over {num_iters} metric-optimization iterations")
    wp.launch(_exp, dim=metric.shape[0], inputs=[log_metric], outputs=[metric], device=device)
    solver.prepare(metric)
    initial_weights = solver.solve(boundary_values)
    _report("initial", initial_weights, apply_biharmonic)

    weights = initial_weights
    wp.synchronize_device(device)
    start_time = time.perf_counter()
    for iteration in range(num_iters):
        wp.launch(_exp, dim=metric.shape[0], inputs=[log_metric], outputs=[metric], device=device)
        solver.prepare(metric)
        # Warm-start from the previous iteration's weights: the metric moves only a
        # little per step, so the solution is nearly unchanged.
        weights = solver.solve(boundary_values, warm_start=weights)

        energy = apply_biharmonic(weights)  # B @ W
        grad_weights = wp.empty_like(energy)  # d(w^T B w)/dw = 2 B w
        wp.launch(_scale_2d, dim=energy.shape, inputs=[energy, 2.0], outputs=[grad_weights], device=device)
        grad_metric = solver.vjp(weights, grad_weights)
        # Chain rule for the log parameterization: d/d(log s) = s * d/ds.
        grad_log = wp.empty_like(grad_metric)
        wp.launch(_multiply, dim=grad_log.shape[0], inputs=[grad_metric, metric], outputs=[grad_log], device=device)
        optimizer.step([grad_log])

        if (iteration + 1) % max(1, num_iters // 5) == 0:
            _report(f"iter {iteration + 1}", weights, apply_biharmonic)

    wp.launch(_exp, dim=metric.shape[0], inputs=[log_metric], outputs=[metric], device=device)
    solver.prepare(metric)
    final_weights = solver.solve(boundary_values, warm_start=weights)
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start_time
    _report("final", final_weights, apply_biharmonic)

    print(
        f"Computed {num_handles} weights on {num_points} vertices in {num_iters} iterations: "
        f"{elapsed:.3f} s total, {1000.0 * elapsed / num_iters:.1f} ms/iteration"
    )

    if stage_path:
        renderer = wp.render.UsdRenderer(stage_path)
        # Color the mesh by the first handle's weight, before and after optimizing.
        for frame, weights in enumerate([initial_weights, final_weights]):
            w0 = weights.numpy()[:, 0]
            colors = np.stack([w0, 0.25 * np.ones_like(w0), 1.0 - w0], axis=-1).astype(np.float32)
            renderer.begin_frame(float(frame))
            renderer.render_mesh("weights", points_np, triangles_np, colors=colors, smooth_shading=False)
            renderer.end_frame()
        renderer.save()

    if polyscope_screenshot:
        render_polyscope(points_np, triangles_np, final_weights, polyscope_screenshot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_bounded_biharmonic_weights.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--resolution", type=int, default=48, help="Grid resolution per side.")
    parser.add_argument("--num-iters", type=int, default=80, help="Number of metric-optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate on the log-metric.")
    parser.add_argument("--headless", action="store_true", help="Run without producing a USD stage.")
    parser.add_argument(
        "--polyscope-screenshot",
        type=str,
        default=None,
        help="If set, render the final weights with polyscope (headless) and save a screenshot to this path.",
    )
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        main(
            stage_path=None if args.headless else args.stage_path,
            resolution=args.resolution,
            num_iters=args.num_iters,
            lr=args.lr,
            polyscope_screenshot=args.polyscope_screenshot,
            headless=args.headless,
        )
