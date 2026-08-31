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


def _icosphere(subdiv=3, radius=2.0):
    """A subdivided icosphere of the given radius, centered at the origin."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    verts = [np.array(v, dtype=np.float64) for v in verts]
    for _ in range(subdiv):
        mid: dict = {}
        new_faces = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(verts)
                verts.append((verts[a] + verts[b]) * 0.5)
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    v = np.array(verts)
    v = (v / np.linalg.norm(v, axis=1, keepdims=True) * radius).astype(np.float32)
    return v, np.array(faces, dtype=np.int32).reshape(-1)


def _two_sheets(n=40, size=1.0, gap=0.065):
    """Two disjoint parallel unit sheets a distance ``gap`` apart (a thin slab),
    with opposite winding so their normals point apart."""

    def sheet(z, flip):
        xs = np.linspace(0.0, size, n)
        xv, yv = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([xv, yv, np.full_like(xv, z)], axis=-1).reshape(-1, 3).astype(np.float32)
        f = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b = i * n + j, (i + 1) * n + j
                c, d = (i + 1) * n + (j + 1), i * n + (j + 1)
                f += [a, c, b, a, d, c] if flip else [a, b, c, a, c, d]
        return pts, np.array(f, dtype=np.int32)

    p0, f0 = sheet(0.0, False)
    p1, f1 = sheet(gap, True)
    return np.vstack([p0, p1]), np.concatenate([f0, f1 + len(p0)])


@wp.kernel
def _geodesic_distance_kernel(
    p1: wp.array(dtype=wp.vec3),
    n1: wp.array(dtype=wp.vec3),
    p2: wp.array(dtype=wp.vec3),
    n2: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = geo.geodesic_distance(p1[i], n1[i], p2[i], n2[i])


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


def test_invalid_num_candidates_raises(test, device):
    points, faces, _ = _plane(16, 1.0)
    with test.assertRaises(ValueError):
        geo.PoissonDiskSampler(points, faces, radius=0.1, num_candidates=0, device=device)


def test_function_matches_class(test, device):
    # The convenience function and the class must share defaults, so identical
    # arguments and seed give identical results.
    points, faces, _ = _plane(48, 2.0)
    f, uv, pos = geo.poisson_disk_sample(points, faces, radius=0.12, seed=1, device=device)
    sampler = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=1, device=device)
    wp.synchronize_device()
    np.testing.assert_array_equal(f.numpy(), sampler.faces.numpy())
    np.testing.assert_array_equal(pos.numpy(), sampler.points.numpy())
    np.testing.assert_array_equal(uv.numpy(), sampler.uv.numpy())


def test_pair_correlation_invalid_r_max_raises(test, device):
    points, faces, _ = _plane(24, 1.0)
    sampler = geo.PoissonDiskSampler(points, faces, radius=0.1, device=device)
    with test.assertRaises(ValueError):
        sampler.pair_correlation(r_max=0.0)


def test_geodesic_distance_exact_on_sphere(test, device):
    # The approximation is exact on a sphere: dg equals the arc length R*theta.
    R = 3.0
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2000, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.standard_normal((2000, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    p1 = wp.array(R * a, dtype=wp.vec3, device=device)
    p2 = wp.array(R * b, dtype=wp.vec3, device=device)
    n1 = wp.array(a.astype(np.float32), dtype=wp.vec3, device=device)
    n2 = wp.array(b.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.empty(2000, dtype=wp.float32, device=device)
    wp.launch(_geodesic_distance_kernel, dim=2000, inputs=[p1, n1, p2, n2], outputs=[out], device=device)
    wp.synchronize_device()

    true_geodesic = R * np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))
    np.testing.assert_allclose(out.numpy(), true_geodesic, rtol=2e-5, atol=1e-4)


def test_geodesic_flat_equals_euclidean(test, device):
    # On a flat sheet the normals are constant, so geodesic == Euclidean and the
    # result should match the Euclidean sampler exactly for the same seed.
    points, faces, _ = _plane(48, 2.0)
    e = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=3, geodesic=False, device=device)
    g = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=3, geodesic=True, device=device)
    wp.synchronize_device()
    test.assertEqual(e.num_samples, g.num_samples)
    np.testing.assert_array_equal(e.points.numpy(), g.points.numpy())


def test_geodesic_spacing_on_sphere(test, device):
    # On a sphere the approximation is exact, so geodesic-mode samples must be at
    # least `radius` apart in TRUE geodesic distance (arc length).
    R = 2.0
    verts, faces = _icosphere(subdiv=4, radius=R)
    radius = 0.35
    sampler = geo.PoissonDiskSampler(verts, faces, radius=radius, seed=0, geodesic=True, device=device)
    wp.synchronize_device()
    P = sampler.points.numpy()
    test.assertGreater(sampler.num_samples, 0)
    dirs = P / np.linalg.norm(P, axis=1, keepdims=True)
    # Nearest-neighbor true geodesic distance for each sample.
    for i in range(0, len(P), 512):
        block = dirs[i : i + 512]
        cos = np.clip(block @ dirs.T, -1.0, 1.0)
        cos[np.arange(len(block)), np.arange(i, i + len(block))] = -1.0  # exclude self
        nn_geo = R * np.arccos(cos.max(axis=1))
        test.assertGreaterEqual(float(nn_geo.min()), radius - 2e-2)


def test_geodesic_helps_thin_feature(test, device):
    # On a thin slab whose gap is a bit over one grid cell, geodesic mode should
    # place at least as many samples as Euclidean (it stops the two sheets from
    # over-separating across the gap).
    radius = 0.1
    points, faces = _two_sheets(n=40, size=1.0, gap=0.065)
    n_eucl = geo.PoissonDiskSampler(points, faces, radius=radius, seed=0, geodesic=False, device=device).num_samples
    n_geo = geo.PoissonDiskSampler(points, faces, radius=radius, seed=0, geodesic=True, device=device).num_samples
    wp.synchronize_device()
    test.assertGreaterEqual(n_geo, n_eucl)


def test_face_areas_forwarded(test, device):
    # face_areas passes through to the internal UniformSampler; supplying the true
    # areas must reproduce the default result for the same seed.
    points, faces, _ = _plane(48, 2.0)
    tri = faces.reshape(-1, 3)
    v0, v1, v2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    areas = (0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)).astype(np.float32)

    default = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=0, device=device)
    provided = geo.PoissonDiskSampler(points, faces, radius=0.12, seed=0, face_areas=areas, device=device)
    wp.synchronize_device()
    test.assertEqual(default.num_samples, provided.num_samples)
    np.testing.assert_array_equal(default.points.numpy(), provided.points.numpy())


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
add_function_test(
    TestGeometryPoisson, "test_invalid_num_candidates_raises", test_invalid_num_candidates_raises, devices=devices
)
add_function_test(TestGeometryPoisson, "test_function_matches_class", test_function_matches_class, devices=devices)
add_function_test(TestGeometryPoisson, "test_face_areas_forwarded", test_face_areas_forwarded, devices=devices)
add_function_test(
    TestGeometryPoisson,
    "test_geodesic_distance_exact_on_sphere",
    test_geodesic_distance_exact_on_sphere,
    devices=devices,
)
add_function_test(
    TestGeometryPoisson, "test_geodesic_flat_equals_euclidean", test_geodesic_flat_equals_euclidean, devices=devices
)
add_function_test(
    TestGeometryPoisson, "test_geodesic_spacing_on_sphere", test_geodesic_spacing_on_sphere, devices=devices
)
add_function_test(
    TestGeometryPoisson, "test_geodesic_helps_thin_feature", test_geodesic_helps_thin_feature, devices=devices
)
add_function_test(
    TestGeometryPoisson,
    "test_pair_correlation_invalid_r_max_raises",
    test_pair_correlation_invalid_r_max_raises,
    devices=devices,
)
add_function_test(TestGeometryPoisson, "test_scale_regression", test_scale_regression, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
