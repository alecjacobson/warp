# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel Poisson-disk surface sampling and its pair-correlation analysis."""

import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.unittest_utils import *


def _plane(n=64, size=2.0):
    """A flat square mesh of ``size x size`` in the z=0 plane."""
    xs = np.linspace(0.0, size, n)
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    points = np.stack([xv, yv, np.zeros_like(xv)], axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + (j + 1)
            d = i * n + (j + 1)
            faces.extend([a, b, c, a, c, d])
    return points, np.array(faces, dtype=np.int32), size * size


def _min_pairwise_distance(pts: np.ndarray) -> float:
    """Smallest distance between any two distinct points, via a uniform cell hash
    so it stays cheap for large point sets."""
    if len(pts) < 2:
        return np.inf
    # Cell size = a rough spacing estimate; a point's nearest neighbor is in its
    # own or an adjacent cell.
    span = pts.max(axis=0) - pts.min(axis=0)
    approx = max(np.mean(span) / max(len(pts) ** (1.0 / 2.0), 1.0), 1e-6)
    cell = approx
    keys = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
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
        a = pts[members]
        b = pts[near]
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        d[d == 0.0] = np.inf
        if d.size:
            best = min(best, float(d.min()))
    return best


def test_returns_valid_output(test, device):
    points, faces, _ = _plane(48, 2.0)
    f, uv, pos = geo.poisson_disk_sample(points, faces, radius=0.12, seed=0, device=device)
    wp.synchronize_device()

    test.assertEqual(f.dtype, wp.int32)
    test.assertEqual(uv.dtype, wp.vec2)
    test.assertEqual(pos.dtype, wp.vec3)
    test.assertEqual(f.shape[0], pos.shape[0])
    test.assertGreater(pos.shape[0], 0)

    uvn = uv.numpy()
    test.assertTrue(np.all(uvn >= 0.0))
    test.assertTrue(np.all(uvn[:, 0] + uvn[:, 1] <= 1.0 + 1e-4))
    # A planar mesh: all samples lie in the plane.
    np.testing.assert_allclose(pos.numpy()[:, 2], 0.0, atol=1e-6)


def test_minimum_distance(test, device):
    # The defining property: no two samples are closer than the radius.
    points, faces, _ = _plane(64, 2.0)
    radius = 0.1
    _, _, pos = geo.poisson_disk_sample(points, faces, radius=radius, seed=3, device=device)
    wp.synchronize_device()
    mind = _min_pairwise_distance(pos.numpy())
    test.assertGreaterEqual(mind, radius - 1e-4)


def test_maximal_coverage(test, device):
    # Maximality: essentially every point of the surface is within `radius` of a
    # sample, so no further sample could be inserted.
    points, faces, _ = _plane(64, 2.0)
    radius = 0.1
    sampler = geo.PoissonDiskSampler(points, faces, radius=radius, seed=2, device=device)
    f, uv = geo.uniformly_sample(points, faces, 20000, seed=99, device=device)
    probe = wp.empty(20000, dtype=wp.vec3, device=device)
    wp.launch(_eval_probe, dim=20000, inputs=[sampler._sampler.mesh.id, f, uv], outputs=[probe], device=device)
    wp.synchronize_device()

    S = sampler.points.numpy()
    U = probe.numpy()
    # Nearest sample distance for each probe point, computed in blocks.
    covered = 0
    block = 2000
    for i in range(0, len(U), block):
        d = np.linalg.norm(U[i : i + block, None, :] - S[None, :, :], axis=-1).min(axis=1)
        covered += int(np.sum(d < radius))
    frac = covered / len(U)
    test.assertGreater(frac, 0.95)


@wp.kernel(enable_backward=False)
def _eval_probe(
    mesh: wp.uint64,
    faces: wp.array(dtype=wp.int32),
    uv: wp.array(dtype=wp.vec2),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = uv[i]
    out[i] = wp.mesh_eval_position(mesh, faces[i], p[0], p[1])


def test_determinism(test, device):
    points, faces, _ = _plane(48, 2.0)
    a = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=7, device=device)
    b = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=7, device=device)
    c = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=8, device=device)
    wp.synchronize_device()

    test.assertEqual(a.num_samples, b.num_samples)
    np.testing.assert_array_equal(a.points.numpy(), b.points.numpy())
    # A different seed gives a different set.
    test.assertTrue(a.num_samples != c.num_samples or not np.array_equal(a.points.numpy(), c.points.numpy()))


def test_radius_controls_count(test, device):
    # Halving the radius should roughly quadruple the sample count on a plane.
    points, faces, _ = _plane(64, 2.0)
    n_big = geo.PoissonDiskSampler(points, faces, radius=0.2, seed=0, device=device).num_samples
    n_small = geo.PoissonDiskSampler(points, faces, radius=0.1, seed=0, device=device).num_samples
    wp.synchronize_device()
    test.assertGreater(n_small, n_big)
    # Count tracks 1/radius^2; expect roughly 4x within a generous band.
    test.assertGreater(n_small / n_big, 2.5)
    test.assertLess(n_small / n_big, 6.0)


def test_count_near_theoretical(test, device):
    # The sample count should be a sizable fraction of the hexagonal-packing max.
    points, faces, area = _plane(80, 2.0)
    radius = 0.08
    n = geo.PoissonDiskSampler(points, faces, radius=radius, seed=1, device=device).num_samples
    wp.synchronize_device()
    n_max = area / (0.8660254 * radius * radius)
    ratio = n / n_max
    test.assertGreater(ratio, 0.4)
    test.assertLessEqual(ratio, 1.0 + 1e-6)


def test_pair_correlation_blue_noise(test, device):
    points, faces, _ = _plane(80, 2.0)
    radius = 0.08
    sampler = geo.PoissonDiskSampler(points, faces, radius=radius, seed=5, device=device)
    r, g = sampler.pair_correlation(num_bins=40)
    wp.synchronize_device()

    # No pairs closer than the radius: g is ~0 well inside the Poisson disk.
    inside = g[r < 0.85 * radius]
    test.assertLess(float(inside.mean()), 0.05)
    # A blue-noise peak appears just past the radius.
    test.assertGreater(float(g.max()), 1.1)


def test_invalid_radius_raises(test, device):
    points, faces, _ = _plane(16, 1.0)
    with test.assertRaises(ValueError):
        geo.PoissonDiskSampler(points, faces, radius=0.0, device=device)


def test_scale_regression(test, device):
    # Regression at scale: a fine radius drives a large candidate pool (~10^6 on
    # CUDA), verifying the sampler still returns a valid, maximal set at size.
    n = 96 if device.is_cpu else 200
    radius = 0.06 if device.is_cpu else 0.02
    points, faces, area = _plane(n, 4.0)

    sampler = geo.PoissonDiskSampler(points, faces, radius=radius, seed=0, device=device)
    wp.synchronize_device()

    num = sampler.num_samples
    n_max = area / (0.8660254 * radius * radius)
    test.assertGreater(num, 0)
    test.assertLessEqual(num / n_max, 1.0 + 1e-6)
    # The blue-noise minimum-distance property must still hold at scale.
    mind = _min_pairwise_distance(sampler.points.numpy())
    test.assertGreaterEqual(mind, radius - 1e-4)


devices = get_test_devices()


class TestGeometryPoisson(unittest.TestCase):
    pass


add_function_test(TestGeometryPoisson, "test_returns_valid_output", test_returns_valid_output, devices=devices)
add_function_test(TestGeometryPoisson, "test_minimum_distance", test_minimum_distance, devices=devices)
add_function_test(TestGeometryPoisson, "test_maximal_coverage", test_maximal_coverage, devices=devices)
add_function_test(TestGeometryPoisson, "test_determinism", test_determinism, devices=devices)
add_function_test(TestGeometryPoisson, "test_radius_controls_count", test_radius_controls_count, devices=devices)
add_function_test(TestGeometryPoisson, "test_count_near_theoretical", test_count_near_theoretical, devices=devices)
add_function_test(
    TestGeometryPoisson, "test_pair_correlation_blue_noise", test_pair_correlation_blue_noise, devices=devices
)
add_function_test(TestGeometryPoisson, "test_invalid_radius_raises", test_invalid_radius_raises, devices=devices)
add_function_test(TestGeometryPoisson, "test_scale_regression", test_scale_regression, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
