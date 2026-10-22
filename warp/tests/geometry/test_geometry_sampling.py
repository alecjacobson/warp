# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uniform surface sampling: warp.geometry.uniformly_sample and UniformSampler."""

import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.unittest_utils import *


def _two_triangles():
    """Build two disjoint triangles with areas 1 and 3.

    Total area is 4, so face 1 is drawn three times as often as face 0.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],  # triangle 0: area 1
            [5.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],  # triangle 1: area 3
        ],
        dtype=np.float32,
    )
    faces = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    return points, faces


def _icosahedron():
    """Build a unit icosahedron: 12 vertices, 20 equal-area triangles."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
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
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
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
        ],
        dtype=np.int32,
    ).reshape(-1)
    return verts, faces


def test_returns_valid_faces_and_barycentrics(test, device):
    points, faces = _two_triangles()
    tri, uv = geo.uniformly_sample(points, faces, 4096, seed=0, device=device)

    tri_np = tri.numpy()
    uv_np = uv.numpy()

    test.assertEqual(tri.dtype, wp.int32)
    test.assertEqual(uv.dtype, wp.vec2)
    # Faces are valid indices.
    test.assertTrue(np.all(tri_np >= 0) and np.all(tri_np < 2))
    # Barycentric coordinates lie in the unit triangle.
    u, v = uv_np[:, 0], uv_np[:, 1]
    test.assertTrue(np.all(u >= 0.0) and np.all(v >= 0.0))
    test.assertTrue(np.all(u + v <= 1.0 + 1e-5))


def test_area_weighting(test, device):
    # A triangle should be selected with probability proportional to its area.
    points, faces = _two_triangles()
    tri, _ = geo.uniformly_sample(points, faces, 400000, seed=123, device=device)

    frac_face1 = float(np.mean(tri.numpy() == 1))
    # Expected 3/4; allow a small statistical margin.
    np.testing.assert_allclose(frac_face1, 0.75, atol=0.01)


def test_within_triangle_uniform(test, device):
    # On a single triangle the mean barycentric u and v should approach 1/3.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([0, 1, 2], dtype=np.int32)
    _, uv = geo.uniformly_sample(points, faces, 500000, seed=5, device=device)

    means = uv.numpy().mean(axis=0)
    np.testing.assert_allclose(means, [1.0 / 3.0, 1.0 / 3.0], atol=0.005)


def test_sample_wrong_output_shape_raises(test, device):
    """Raise ValueError when output arrays do not match the number of samples."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sampler = geo.UniformSampler(points, np.array([0, 1, 2], dtype=np.int32), device=device)

    buffer = wp.full(64, -1, dtype=wp.int32, device=device)
    out_uv = wp.empty(64, dtype=wp.vec2, device=device)

    with test.assertRaisesRegex(ValueError, "out_faces"):
        sampler.sample(64, out_faces=buffer[:4], out_uv=out_uv)
    with test.assertRaisesRegex(ValueError, "out_uv"):
        sampler.sample(8, out_faces=wp.empty(8, dtype=wp.int32, device=device), out_uv=out_uv)

    # Nothing may have been written past the short view.
    np.testing.assert_array_equal(buffer.numpy(), -1)


def test_sample_output_on_other_device_raises(test, device):
    """Raise ValueError when an output array lives on a different device."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sampler = geo.UniformSampler(points, np.array([0, 1, 2], dtype=np.int32), device=device)

    other = "cpu" if wp.get_device(device).is_cuda else "cuda:0"
    if not wp.is_device_available(other):
        test.skipTest("No second device available")
    with test.assertRaisesRegex(ValueError, "out_faces"):
        sampler.sample(
            4,
            out_faces=wp.empty(4, dtype=wp.int32, device=other),
            out_uv=wp.empty(4, dtype=wp.vec2, device=device),
        )


def test_embed_sample_matches_barycentric_interpolation(test, device):
    """Evaluate sample positions by barycentric interpolation of the triangle vertices."""
    points, faces = _two_triangles()
    sampler = geo.UniformSampler(points, faces, device=device)
    f, uv = sampler.sample(1000, seed=4)
    positions = sampler.sample_points(1000, seed=4).numpy()

    f = f.numpy()
    uv = uv.numpy()
    tri = np.asarray(faces).reshape(-1, 3)[f]
    pts = np.asarray(points, dtype=np.float32)
    expected = pts[tri[:, 0]] * uv[:, :1] + pts[tri[:, 1]] * uv[:, 1:] + pts[tri[:, 2]] * (1.0 - uv[:, :1] - uv[:, 1:])
    np.testing.assert_allclose(positions, expected, atol=1e-6)

    # The device state follows vertices that move in place.
    sampler.points.assign(sampler.points.numpy() + np.array([0.0, 0.0, 1.0], dtype=np.float32))
    moved = sampler.sample_points(1000, seed=4).numpy()
    np.testing.assert_allclose(moved, positions + np.array([0.0, 0.0, 1.0]), atol=1e-6)


@wp.kernel(enable_backward=False)
def _mesh_builtin_positions_kernel(
    mesh: wp.uint64,
    faces: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = wp.mesh_eval_position(mesh, faces[i], uv[i][0], uv[i][1])


def test_embed_sample_matches_mesh_eval_position(test, device):
    """Match the barycentric convention of wp.mesh_eval_position."""
    points, faces = _two_triangles()
    sampler = geo.UniformSampler(points, faces, device=device)
    f, uv = sampler.sample(1000, seed=4)

    mesh = wp.Mesh(
        points=sampler.points, indices=wp.array(np.asarray(faces).reshape(-1), dtype=wp.int32, device=device)
    )
    expected = wp.empty(1000, dtype=wp.vec3, device=device)
    wp.launch(_mesh_builtin_positions_kernel, dim=1000, inputs=[mesh.id, f, uv], outputs=[expected], device=device)

    np.testing.assert_allclose(sampler.sample_points(1000, seed=4).numpy(), expected.numpy(), atol=1e-6)


def test_faces_as_2d_array(test, device):
    """Accept triangle indices as a (num_triangles, 3) Warp array."""
    points, faces = _two_triangles()
    faces_2d = wp.array(np.asarray(faces, dtype=np.int32).reshape(-1, 3), dtype=wp.int32, device=device)
    sampler = geo.UniformSampler(points, faces_2d, device=device)
    test.assertEqual(sampler.num_triangles, len(np.asarray(faces).reshape(-1, 3)))
    test.assertEqual(sampler.sample_points(16).shape, (16,))

    # Invalid shapes are rejected.
    with test.assertRaisesRegex(ValueError, "multiple of 3"):
        geo.UniformSampler(points, np.zeros((2, 4), dtype=np.int32), device=device)
    with test.assertRaisesRegex(ValueError, "multiple of 3"):
        geo.UniformSampler(points, np.zeros(5, dtype=np.int32), device=device)


def test_sample_points_on_surface(test, device):
    # Points sampled on a planar mesh must lie in that plane.
    points, faces = _two_triangles()
    sampler = geo.UniformSampler(points, faces, device=device)
    pos = sampler.sample_points(10000, seed=2).numpy()

    test.assertEqual(pos.shape, (10000, 3))
    np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-6)

    # Points on the icosahedron must lie on the unit sphere.
    verts, ico_faces = _icosahedron()
    sampler = geo.UniformSampler(verts, ico_faces, device=device)
    pos = sampler.sample_points(10000, seed=4).numpy()
    radii = np.linalg.norm(pos, axis=1)
    # Points sit on flat faces inscribed in the sphere, so radius <= 1.
    test.assertTrue(np.all(radii <= 1.0 + 1e-5))
    test.assertTrue(np.all(radii > 0.7))


def test_seed_behavior(test, device):
    points, faces = _two_triangles()
    sampler = geo.UniformSampler(points, faces, device=device)

    a_tri, a_uv = sampler.sample(1000, seed=42)
    b_tri, b_uv = sampler.sample(1000, seed=42)
    _c_tri, c_uv = sampler.sample(1000, seed=43)

    # Same seed reproduces the same samples.
    np.testing.assert_array_equal(a_tri.numpy(), b_tri.numpy())
    np.testing.assert_array_equal(a_uv.numpy(), b_uv.numpy())
    # A different seed gives different samples.
    test.assertFalse(np.array_equal(a_uv.numpy(), c_uv.numpy()))


def test_draw_sample_in_kernel(test, device):
    # The device function warp.geometry.draw_sample can be called from a user kernel.
    points, faces = _two_triangles()
    sampler = geo.UniformSampler(points, faces, device=device)

    @wp.kernel
    def sample_faces_kernel(
        state: geo.UniformSamplerState,
        seed: int,
        out_faces: wp.array[wp.int32],
    ):
        tid = wp.tid()
        rng = wp.rand_init(seed, tid)
        s = geo.draw_sample(state, rng)
        out_faces[tid] = s.face

    out_faces = wp.empty(4096, dtype=wp.int32, device=device)
    wp.launch(sample_faces_kernel, dim=4096, inputs=[sampler.state, 9], outputs=[out_faces], device=device)

    f = out_faces.numpy()
    test.assertTrue(np.all((f == 0) | (f == 1)))
    test.assertTrue(np.any(f == 0) and np.any(f == 1))


def test_zero_area_mesh_raises(test, device):
    """Raise ValueError when the mesh has zero total surface area."""
    points = np.zeros((3, 3), dtype=np.float32)
    with test.assertRaises(ValueError):
        geo.UniformSampler(points, np.array([0, 1, 2], dtype=np.int32), device=device)


def test_cdf_resolves_equal_areas_on_large_mesh(test, device):
    """Keep the area distribution accurate to double precision on a mesh with many triangles."""
    n = 500
    xs = np.arange(n + 1, dtype=np.float32)
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    points = np.stack([xv, yv, np.zeros_like(xv)], axis=-1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = (i * (n + 1) + j).ravel()
    faces = np.stack([a, a + n + 1, a + n + 2, a, a + n + 2, a + 1], axis=-1).ravel().astype(np.int32)

    sampler = geo.UniformSampler(points, faces, device=device)
    cdf = sampler.cdf.numpy()
    num_triangles = faces.shape[0] // 3
    test.assertEqual(cdf.dtype, np.float64)
    # Equal-area triangles: every CDF step is 1 / num_triangles. A float32 CDF would be
    # off by about ``num_triangles * 6e-8`` (3% here) in relative terms.
    steps = np.diff(cdf, prepend=0.0) * num_triangles
    np.testing.assert_allclose(steps, 1.0, rtol=1.0e-9)


def test_update_matches_rebuild(test, device):
    """Rebuild the area distribution from vertices that moved in place."""
    points, faces = _two_triangles()
    vertices = wp.array(points, dtype=wp.vec3, device=device)
    sampler = geo.UniformSampler(vertices, faces, device=device)

    moved = points.copy()
    moved[4] = [5.0, 0.0, 0.0]  # triangle 1 collapses to a zero-area sliver
    moved[1] = [3.0, 0.0, 0.0]  # triangle 0 grows from area 1 to 3
    vertices.assign(moved)
    sampler.update()

    rebuilt = geo.UniformSampler(moved, faces, device=device)
    np.testing.assert_allclose(sampler.cdf.numpy(), rebuilt.cdf.numpy(), rtol=1.0e-12)
    # The device-side area follows the update, while the host snapshot keeps its value.
    np.testing.assert_allclose(sampler.total_area_device.numpy(), [rebuilt.total_area], rtol=1.0e-12)
    test.assertAlmostEqual(sampler.total_area, 4.0, places=6)
    test.assertAlmostEqual(rebuilt.total_area, 3.0, places=6)
    # Every draw lands on triangle 0 now that triangle 1 has no area.
    draws, _ = sampler.sample(2000, seed=4)
    test.assertTrue(np.all(draws.numpy() == 0))


def test_update_in_cuda_graph(test, device):
    """Replay a captured update on vertices that have moved."""
    points, faces = _two_triangles()
    vertices = wp.array(points, dtype=wp.vec3, device=device)
    sampler = geo.UniformSampler(vertices, faces, device=device)

    # Load the module that holds the captured kernels up front; loading during capture
    # crashes on CUDA drivers older than 12.3.
    wp.load_module("warp.geometry.sampling", device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        sampler.update()

    moved = points.copy()
    moved[1] = [3.0, 0.0, 0.0]
    vertices.assign(moved)
    wp.capture_launch(capture.graph)

    rebuilt = geo.UniformSampler(moved, faces, device=device)
    np.testing.assert_allclose(sampler.cdf.numpy(), rebuilt.cdf.numpy(), rtol=1.0e-12)
    np.testing.assert_allclose(sampler.total_area_device.numpy(), [rebuilt.total_area], rtol=1.0e-12)


def test_update_reports_zero_area_on_check(test, device):
    """Raise ValueError from check_error after an update collapses the whole mesh."""
    points, faces = _two_triangles()
    vertices = wp.array(points, dtype=wp.vec3, device=device)
    sampler = geo.UniformSampler(vertices, faces, device=device)

    vertices.zero_()
    sampler.update()
    with test.assertRaises(ValueError):
        sampler.check_error()


def test_empty_faces_raises(test, device):
    points = np.zeros((3, 3), dtype=np.float32)
    with test.assertRaises(ValueError):
        geo.UniformSampler(points, np.array([], dtype=np.int32), device=device)


def test_face_areas_match_computed(test, device):
    # Supplying the correct per-triangle areas must reproduce the sampler built
    # without them, bit for bit, for the same seed.
    points, faces = _two_triangles()
    areas = np.array([1.0, 3.0], dtype=np.float32)  # the true areas of _two_triangles

    default = geo.UniformSampler(points, faces, device=device)
    provided = geo.UniformSampler(points, faces, face_areas=areas, device=device)
    # Also accept a warp.array of areas.
    as_wp = geo.UniformSampler(
        points, faces, face_areas=wp.array(areas, dtype=wp.float32, device=device), device=device
    )

    np.testing.assert_allclose(default.total_area, 4.0)
    np.testing.assert_allclose(provided.total_area, 4.0)
    np.testing.assert_array_equal(default.cdf.numpy(), provided.cdf.numpy())
    np.testing.assert_array_equal(default.cdf.numpy(), as_wp.cdf.numpy())

    df, du = default.sample(4096, seed=5)
    pf, pu = provided.sample(4096, seed=5)
    np.testing.assert_array_equal(df.numpy(), pf.numpy())
    np.testing.assert_array_equal(du.numpy(), pu.numpy())


def test_face_areas_override_weighting(test, device):
    # The provided areas drive the weighting: passing equal areas for a mesh whose
    # triangles have geometric areas 1 and 3 must sample the two faces equally.
    points, faces = _two_triangles()
    tri, _ = geo.uniformly_sample(points, faces, 200000, seed=1, device=device)  # geometric weighting -> ~3/4 face 1
    sampler = geo.UniformSampler(points, faces, face_areas=np.array([1.0, 1.0], dtype=np.float32), device=device)
    eq_tri, _ = sampler.sample(200000, seed=1)

    np.testing.assert_allclose(float(np.mean(tri.numpy() == 1)), 0.75, atol=0.01)
    np.testing.assert_allclose(float(np.mean(eq_tri.numpy() == 1)), 0.5, atol=0.01)


def test_face_areas_wrong_length_raises(test, device):
    points, faces = _two_triangles()  # 2 triangles
    with test.assertRaises(ValueError):
        geo.UniformSampler(points, faces, face_areas=np.array([1.0, 2.0, 3.0], dtype=np.float32), device=device)


def test_face_areas_negative_or_nonfinite_raises(test, device):
    points, faces = _two_triangles()
    for bad in ([-1.0, 3.0], [1.0, np.inf], [np.nan, 3.0]):
        with test.assertRaises(ValueError):
            geo.UniformSampler(points, faces, face_areas=np.array(bad, dtype=np.float32), device=device)


devices = get_test_devices()
cuda_devices_with_mempool = get_cuda_test_devices_with_mempool()


class TestGeometrySampling(unittest.TestCase):
    pass


add_function_test(
    TestGeometrySampling,
    "test_returns_valid_faces_and_barycentrics",
    test_returns_valid_faces_and_barycentrics,
    devices=devices,
)
add_function_test(TestGeometrySampling, "test_area_weighting", test_area_weighting, devices=devices)
add_function_test(TestGeometrySampling, "test_within_triangle_uniform", test_within_triangle_uniform, devices=devices)
add_function_test(TestGeometrySampling, "test_sample_points_on_surface", test_sample_points_on_surface, devices=devices)
add_function_test(
    TestGeometrySampling,
    "test_sample_wrong_output_shape_raises",
    test_sample_wrong_output_shape_raises,
    devices=devices,
)
add_function_test(
    TestGeometrySampling,
    "test_sample_output_on_other_device_raises",
    test_sample_output_on_other_device_raises,
    devices=devices,
)
add_function_test(
    TestGeometrySampling,
    "test_embed_sample_matches_barycentric_interpolation",
    test_embed_sample_matches_barycentric_interpolation,
    devices=devices,
)
add_function_test(
    TestGeometrySampling,
    "test_embed_sample_matches_mesh_eval_position",
    test_embed_sample_matches_mesh_eval_position,
    devices=devices,
)
add_function_test(TestGeometrySampling, "test_faces_as_2d_array", test_faces_as_2d_array, devices=devices)
add_function_test(TestGeometrySampling, "test_seed_behavior", test_seed_behavior, devices=devices)
add_function_test(TestGeometrySampling, "test_draw_sample_in_kernel", test_draw_sample_in_kernel, devices=devices)
add_function_test(
    TestGeometrySampling,
    "test_cdf_resolves_equal_areas_on_large_mesh",
    test_cdf_resolves_equal_areas_on_large_mesh,
    devices=devices,
)
add_function_test(TestGeometrySampling, "test_update_matches_rebuild", test_update_matches_rebuild, devices=devices)
add_function_test(
    TestGeometrySampling, "test_update_in_cuda_graph", test_update_in_cuda_graph, devices=cuda_devices_with_mempool
)
add_function_test(
    TestGeometrySampling,
    "test_update_reports_zero_area_on_check",
    test_update_reports_zero_area_on_check,
    devices=devices,
)
add_function_test(TestGeometrySampling, "test_zero_area_mesh_raises", test_zero_area_mesh_raises, devices=devices)
add_function_test(TestGeometrySampling, "test_empty_faces_raises", test_empty_faces_raises, devices=devices)
add_function_test(
    TestGeometrySampling, "test_face_areas_match_computed", test_face_areas_match_computed, devices=devices
)
add_function_test(
    TestGeometrySampling, "test_face_areas_override_weighting", test_face_areas_override_weighting, devices=devices
)
add_function_test(
    TestGeometrySampling, "test_face_areas_wrong_length_raises", test_face_areas_wrong_length_raises, devices=devices
)
add_function_test(
    TestGeometrySampling,
    "test_face_areas_negative_or_nonfinite_raises",
    test_face_areas_negative_or_nonfinite_raises,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
