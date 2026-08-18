# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.geometry as wg
from warp._src.sparse_voxel_grid import (
    _CORNER_OFFSETS,
    _NEIGHBOR_OFFSETS,
    COORD_MAX,
    _sparse_voxel_grid_wavefront_reference,
)
from warp.tests.unittest_utils import *

_CORNERS = np.array(_CORNER_OFFSETS, dtype=np.float32)


# =============================================================================
# Implicit functions (Warp @wp.func + NumPy float32 counterparts)
# =============================================================================


@wp.func
def sphere(p: wp.vec3) -> wp.float32:
    return wp.length(p) - 1.0


def np_sphere(P):
    return (np.sqrt((P * P).sum(-1)) - np.float32(1.0)).astype(np.float32)


@wp.func
def ellipsoid(p: wp.vec3) -> wp.float32:
    q = wp.vec3(p[0] / 1.3, p[1] / 0.8, p[2] / 1.0)
    return wp.length(q) - 1.0


def np_ellipsoid(P):
    q = P / np.array([1.3, 0.8, 1.0], dtype=np.float32)
    return (np.sqrt((q * q).sum(-1)) - np.float32(1.0)).astype(np.float32)


@wp.func
def torus(p: wp.vec3) -> wp.float32:
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - 0.7, p[1])
    return wp.length(q) - 0.25


def np_torus(P):
    a = np.sqrt(P[..., 0] ** 2 + P[..., 2] ** 2).astype(np.float32) - np.float32(0.7)
    return (np.sqrt(a * a + P[..., 1] ** 2) - np.float32(0.25)).astype(np.float32)


@wp.func
def wavy(p: wp.vec3) -> wp.float32:
    # A bumpy sphere: bounded, connected, more complex than a quadric.
    bump = 0.15 * wp.sin(4.0 * p[0]) * wp.sin(4.0 * p[1]) * wp.sin(4.0 * p[2])
    return wp.length(p) - (1.0 + bump)


def np_wavy(P):
    bump = np.float32(0.15) * np.sin(4.0 * P[..., 0]) * np.sin(4.0 * P[..., 1]) * np.sin(4.0 * P[..., 2])
    return (np.sqrt((P * P).sum(-1)) - (np.float32(1.0) + bump)).astype(np.float32)


@wp.func
def two_spheres(p: wp.vec3) -> wp.float32:
    a = wp.length(p - wp.vec3(-2.0, 0.0, 0.0)) - 0.6
    b = wp.length(p - wp.vec3(2.0, 0.0, 0.0)) - 0.6
    return wp.min(a, b)


@wp.func
def quadric_zero(p: wp.vec3) -> wp.float32:
    # Corners of cell (0,0,0) at p0=0, eps=1 are (+/-0.5)^3 -> sum of squares
    # exactly 0.75 in float32, so every corner is exactly the isovalue.
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 0.75


# =============================================================================
# Sparse CPU oracle (float32 to match GPU sign decisions)
# =============================================================================


def cpu_oracle(np_func, p0, eps, seed=(0, 0, 0), threshold=0.0):
    # NOTE: evaluated in float32 to match the GPU's sign decisions. The test
    # parameters below keep every cube corner well away from the isovalue (no
    # corner within ~1e-4), so float32 vs GPU rounding never flips a sign; this
    # keeps the exact set-equality comparison robust rather than merely lucky.
    p0 = np.array(p0, dtype=np.float32)
    eps = np.float32(eps)
    threshold = np.float32(threshold)

    def active(c):
        pos = (p0 + eps * (np.array(c, dtype=np.float32) + _CORNERS - np.float32(0.5))).astype(np.float32)
        s = np.sign(np_func(pos) - threshold)
        return not np.all(s == s[0])

    if not active(seed):
        return set()
    seen = {seed}
    result = set()
    stack = [seed]
    while stack:
        c = stack.pop()
        if not active(c):
            continue
        result.add(c)
        for d in _NEIGHBOR_OFFSETS:
            nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return result


def _cell_set(cells_array):
    return set(map(tuple, cells_array.numpy())) if cells_array.shape[0] > 0 else set()


def _validate_geometry(test, np_func, p0, eps, cells, cv, cs, ci):
    cells_np = cells.numpy()
    cv_np = cv.numpy()
    ci_np = ci.numpy()
    cs_np = cs.numpy()
    m = ci_np.shape[0]
    test.assertEqual(cells_np.shape[0], m)
    test.assertEqual(ci_np.shape[1], 8)

    # Each cell's 8 corners resolve to the correct world positions in libigl order.
    expected = np.array(p0, dtype=np.float64) + eps * (cells_np[:, None, :] + np.array(_CORNER_OFFSETS)[None] - 0.5)
    got = cv_np[ci_np]
    test.assertLess(np.abs(expected - got).max(), 1e-5)

    # Corners are de-duplicated (no two unique vertices coincide).
    test.assertEqual(cv_np.shape[0], len({tuple(v) for v in np.round(cv_np * 1e5).astype(np.int64)}))

    # CS equals the field at CV.
    test.assertLess(np.abs(cs_np - np_func(cv_np.astype(np.float32))).max(), 1e-4)

    # Indices in range and every vertex used.
    test.assertTrue((ci_np >= 0).all() and (ci_np < cv_np.shape[0]).all())
    test.assertTrue((np.unique(ci_np) == np.arange(cv_np.shape[0])).all())


# =============================================================================
# Tests
# =============================================================================


def test_svg_matches_oracle(test, device):
    """Active cells and geometry match a sparse CPU oracle for several surfaces."""
    cases = [
        (sphere, np_sphere, (1.0, 0.0, 0.0), 1.0 / 16.0),
        (ellipsoid, np_ellipsoid, (1.3, 0.0, 0.0), 1.0 / 16.0),
        (torus, np_torus, (0.95, 0.0, 0.0), 1.0 / 20.0),
        (wavy, np_wavy, (1.0, 0.0, 0.0), 1.0 / 20.0),
    ]
    for wp_func, np_func, p0, eps in cases:
        with test.subTest(func=wp_func.key):
            cv, cs, ci, cells = wg.sparse_voxel_grid(p0, wp_func, eps, 12000, device=device, return_cells=True)
            gpu = _cell_set(cells)
            oracle = cpu_oracle(np_func, p0, eps)
            test.assertGreater(len(oracle), 0)
            test.assertEqual(gpu, oracle)
            _validate_geometry(test, np_func, p0, eps, cells, cv, cs, ci)


def test_svg_matches_wavefront_reference(test, device):
    """Production (multi-step) traversal matches the test-only wavefront path."""
    p0, eps = (1.0, 0.0, 0.0), 1.0 / 16.0
    _, _, _, cells = wg.sparse_voxel_grid(p0, sphere, eps, 12000, device=device, return_cells=True)
    ref_cells, ref_stats = _sparse_voxel_grid_wavefront_reference(p0, sphere, eps, 12000, device=device)
    prod = _cell_set(cells)
    test.assertEqual(prod, _cell_set(ref_cells))
    # Multi-step performs several expansions per launch, so it uses fewer rounds.
    _, _, _, _, stats = wg.sparse_voxel_grid(
        p0, sphere, eps, 12000, device=device, return_cells=True, return_stats=True
    )
    test.assertLess(stats["spill_round_count"], ref_stats["spill_round_count"])


def test_svg_deterministic_set(test, device):
    """The active-cell set is identical across repeated runs despite nondeterministic order."""
    p0, eps = (1.0, 0.0, 0.0), 1.0 / 16.0
    _, _, _, first = wg.sparse_voxel_grid(p0, sphere, eps, 12000, device=device, return_cells=True)
    ref = _cell_set(first)
    for _ in range(3):
        _, _, _, again = wg.sparse_voxel_grid(p0, sphere, eps, 12000, device=device, return_cells=True)
        test.assertEqual(_cell_set(again), ref)


def test_svg_invalid_seed_empty(test, device):
    """A seed cell that does not straddle the surface yields an empty result."""
    cv, cs, ci = wg.sparse_voxel_grid((5.0, 5.0, 5.0), sphere, 1.0 / 16.0, 100, device=device)
    test.assertEqual(cv.shape[0], 0)
    test.assertEqual(cs.shape[0], 0)
    test.assertEqual(ci.shape[0], 0)


def test_svg_threshold(test, device):
    """A non-zero threshold extracts that level set, and CS stays the raw field.

    ``sphere`` is the SDF ``|p| - 1``; ``threshold=0.25`` is the ``|p| = 1.25``
    level set. libigl's ``CS`` holds the raw field value (not field - threshold),
    so a downstream extractor can re-apply the same isovalue.
    """
    p0, eps, the = (1.25, 0.0, 0.0), 1.0 / 16.0, 0.25
    cv, cs, ci, cells = wg.sparse_voxel_grid(p0, sphere, eps, 12000, threshold=the, device=device, return_cells=True)

    test.assertEqual(_cell_set(cells), cpu_oracle(np_sphere, p0, eps, threshold=the))
    _validate_geometry(test, np_sphere, p0, eps, cells, cv, cs, ci)  # asserts CS == raw np_sphere(CV)

    # The surface corners bracket the (non-zero) threshold in the raw field.
    cs_np = cs.numpy()
    test.assertLess(cs_np.min(), the)
    test.assertGreater(cs_np.max(), the)


def test_svg_exact_zero_corners(test, device):
    """All-zero corners share a sign (0) and are inactive, matching libigl's sgn."""
    # p0=0, eps=1: cell (0,0,0) corners have squared-norm exactly 0.75 -> value 0.
    *_, cells = wg.sparse_voxel_grid((0.0, 0.0, 0.0), quadric_zero, 1.0, 64, device=device, return_cells=True)
    test.assertEqual(cells.shape[0], 0)  # seed inactive -> empty


def test_svg_negative_coordinates(test, device):
    """Traversal works through negative cell coordinates."""
    p0, eps = (-1.0, 0.0, 0.0), 1.0 / 16.0
    *_, cells = wg.sparse_voxel_grid(p0, sphere, eps, 12000, device=device, return_cells=True)
    gpu = _cell_set(cells)
    test.assertEqual(gpu, cpu_oracle(np_sphere, p0, eps))
    test.assertLess(cells.numpy().min(), 0)


def test_svg_disconnected_components(test, device):
    """Only the surface component connected to the seed is returned."""
    p0, eps = (-1.4, 0.0, 0.0), 1.0 / 16.0  # seed near the left sphere at x=-2
    _, _, _, cells = wg.sparse_voxel_grid(p0, two_spheres, eps, 12000, device=device, return_cells=True)
    cells_np = cells.numpy()
    test.assertGreater(cells_np.shape[0], 0)
    # All returned cells belong to the left sphere (x < 0), not the right one.
    world_x = p0[0] + eps * cells_np[:, 0]
    test.assertTrue((world_x < 0.0).all())
    test.assertEqual(_cell_set(cells), cpu_oracle(_np_two_spheres, p0, eps))


def _np_two_spheres(P):
    a = np.sqrt(((P - np.array([-2.0, 0.0, 0.0], dtype=np.float32)) ** 2).sum(-1)) - np.float32(0.6)
    b = np.sqrt(((P - np.array([2.0, 0.0, 0.0], dtype=np.float32)) ** 2).sum(-1)) - np.float32(0.6)
    return np.minimum(a, b).astype(np.float32)


def test_svg_capacity_overflow_errors(test, device):
    """Hash / surface / spill exhaustion raise a clean error, never truncate."""
    p0, eps = (1.0, 0.0, 0.0), 1.0 / 16.0
    with test.assertRaises(wg.SparseVoxelGridError):
        wg.sparse_voxel_grid(p0, sphere, eps, 12000, visited_capacity=64, device=device)
    with test.assertRaises(wg.SparseVoxelGridError):
        wg.sparse_voxel_grid(p0, sphere, eps, 12000, surface_capacity=100, device=device)
    with test.assertRaises(wg.SparseVoxelGridError):
        wg.sparse_voxel_grid(p0, sphere, eps, 12000, spill_capacity=64, device=device)


def test_svg_coordinate_overflow_error(test, device):
    """A seed beyond the packable coordinate range raises a clean error."""
    with test.assertRaises(wg.SparseVoxelGridError):
        wg.sparse_voxel_grid((0.0, 0.0, 0.0), sphere, 1.0, 100, seed=(COORD_MAX + 1, 0, 0), device=device)


def test_svg_argument_validation(test, device):
    with test.assertRaises(TypeError):
        wg.sparse_voxel_grid((0.0, 0.0, 0.0), 42, 0.1, 100, device=device)
    with test.assertRaises(ValueError):
        wg.sparse_voxel_grid((0.0, 0.0, 0.0), sphere, 0.0, 100, device=device)
    with test.assertRaises(ValueError):
        wg.sparse_voxel_grid((0.0, 0.0, 0.0), sphere, 0.1, 0, device=device)


def test_svg_sparse_scaling(test, device):
    """Active-cell count grows ~O(n^2) (surface area), not O(n^3) (volume).

    Doubling the resolution should grow the discovered cell count by roughly 4x,
    confirming the traversal tracks the surface rather than the ambient volume.
    Buffers are sized from the (surface-proportional) cell-count hint, never from
    an n^3 ambient extent.
    """
    counts = {}
    for n in (16, 32, 64):
        eps = 2.0 / n  # fixed physical sphere, spacing ~ 1/n
        _, _, _, _, stats = wg.sparse_voxel_grid(
            (1.0, 0.0, 0.0), sphere, eps, 40 * n * n, device=device, return_cells=True, return_stats=True
        )
        counts[n] = stats["surface_count"]
        test.assertGreater(counts[n], 0)

    for n in (16, 32):
        ratio = counts[2 * n] / counts[n]
        test.assertGreater(ratio, 3.0)  # sub-cubic (O(n^2) ~ 4, not O(n^3) ~ 8)
        test.assertLess(ratio, 5.5)


devices = get_test_devices()


class TestSparseVoxelGrid(unittest.TestCase):
    pass


add_function_test(TestSparseVoxelGrid, "test_svg_matches_oracle", test_svg_matches_oracle, devices=devices)
add_function_test(
    TestSparseVoxelGrid, "test_svg_matches_wavefront_reference", test_svg_matches_wavefront_reference, devices=devices
)
add_function_test(TestSparseVoxelGrid, "test_svg_deterministic_set", test_svg_deterministic_set, devices=devices)
add_function_test(TestSparseVoxelGrid, "test_svg_invalid_seed_empty", test_svg_invalid_seed_empty, devices=devices)
add_function_test(TestSparseVoxelGrid, "test_svg_threshold", test_svg_threshold, devices=devices)
add_function_test(TestSparseVoxelGrid, "test_svg_exact_zero_corners", test_svg_exact_zero_corners, devices=devices)
add_function_test(TestSparseVoxelGrid, "test_svg_negative_coordinates", test_svg_negative_coordinates, devices=devices)
add_function_test(
    TestSparseVoxelGrid, "test_svg_disconnected_components", test_svg_disconnected_components, devices=devices
)
add_function_test(
    TestSparseVoxelGrid, "test_svg_capacity_overflow_errors", test_svg_capacity_overflow_errors, devices=devices
)
add_function_test(
    TestSparseVoxelGrid, "test_svg_coordinate_overflow_error", test_svg_coordinate_overflow_error, devices=devices
)
add_function_test(TestSparseVoxelGrid, "test_svg_argument_validation", test_svg_argument_validation, devices=devices)
add_function_test(TestSparseVoxelGrid, "test_svg_sparse_scaling", test_svg_sparse_scaling, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
