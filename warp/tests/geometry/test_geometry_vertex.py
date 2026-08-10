# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-vertex operations: vertex normals and discrete Gaussian curvature."""

import math
import unittest

import numpy as np

import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *


def _bumpy_grid(seed=3):
    """An irregularly tessellated, non-planar patch.

    Both properties matter: the weighting schemes only disagree when the incident
    triangles differ in both area and corner angle.
    """
    p_np, f_np = U.planar_grid(6, 6, jitter=0.12, rng=np.random.default_rng(seed))
    p_np = p_np.copy()
    p_np[:, 2] = 0.3 * np.sin(3.0 * p_np[:, 0]) * np.cos(3.0 * p_np[:, 1])
    return p_np.astype(np.float32), f_np


##########################################################################
## vertex_normals
##########################################################################


def test_vertex_normals_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(31), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)

    for weighting in geo.VertexNormalWeighting:
        for normalized in (False, True):
            with test.subTest(weighting=weighting.name, normalized=normalized):
                np.testing.assert_allclose(
                    geo.vertex_normals(points, indices, weighting=weighting, normalized=normalized).numpy(),
                    U.ref_vertex_normals(p_np, f_np, weighting=weighting, normalized=normalized),
                    rtol=1e-4,
                    atol=1e-5,
                )


def test_vertex_normals_schemes_differ(test, device):
    # Guards against the weighting parameter silently becoming a no-op.
    p_np, f_np = _bumpy_grid()
    points, indices = U.to_warp(p_np, f_np, device)

    results = {
        w: geo.vertex_normals(points, indices, weighting=w, normalized=True).numpy() for w in geo.VertexNormalWeighting
    }
    schemes = list(geo.VertexNormalWeighting)
    for i in range(len(schemes)):
        for j in range(i + 1, len(schemes)):
            diff = np.abs(results[schemes[i]] - results[schemes[j]]).max()
            test.assertGreater(diff, 1e-3, f"{schemes[i].name} and {schemes[j].name} produced the same normals")


def test_vertex_normals_sphere_is_radial(test, device):
    # On a sphere every weighting scheme must give outward radial normals.
    p_np, f_np = U.icosphere(3)
    points, indices = U.to_warp(p_np, f_np, device)
    radial = p_np / np.linalg.norm(p_np, axis=1, keepdims=True)

    for weighting in geo.VertexNormalWeighting:
        with test.subTest(weighting=weighting.name):
            normals = geo.vertex_normals(points, indices, weighting=weighting, normalized=True).numpy()
            np.testing.assert_allclose(normals, radial, atol=0.02)


def test_vertex_normals_normalized_unit_length(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(37), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)
    for weighting in geo.VertexNormalWeighting:
        with test.subTest(weighting=weighting.name):
            normals = geo.vertex_normals(points, indices, weighting=weighting, normalized=True).numpy()
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-5)


def test_vertex_normals_area_weighting_matches_face_sum(test, device):
    # AREA weighting is defined as the plain sum of unnormalized face normals.
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(41), subdivisions=1)
    points, indices = U.to_warp(p_np, f_np, device)

    face_normals = geo.triangle_normals(points, indices).numpy()
    expected = np.zeros_like(p_np, dtype=np.float64)
    np.add.at(expected, f_np.reshape(-1, 3).ravel(), np.repeat(face_normals, 3, axis=0))

    got = geo.vertex_normals(points, indices, weighting=geo.VertexNormalWeighting.AREA).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


def test_vertex_normals_weighting_accepts_int(test, device):
    # VertexNormalWeighting is an IntEnum, so the raw value must work too.
    p_np, f_np = U.icosphere(1)
    points, indices = U.to_warp(p_np, f_np, device)

    from_enum = geo.vertex_normals(points, indices, weighting=geo.VertexNormalWeighting.ANGLE).numpy()
    from_int = geo.vertex_normals(points, indices, weighting=int(geo.VertexNormalWeighting.ANGLE)).numpy()
    # Normals are accumulated with atomics, so two launches need not agree bitwise.
    np.testing.assert_allclose(from_enum, from_int, rtol=1e-6, atol=1e-6)


def test_vertex_normals_invalid_weighting_raises(test, device):
    points, indices = U.to_warp(*U.icosphere(0), device)
    with test.assertRaises(ValueError):
        geo.vertex_normals(points, indices, weighting=99)


##########################################################################
## vertex_gaussian_curvature
##########################################################################


def test_curvature_gauss_bonnet_genus_0(test, device):
    # Integrated Gaussian curvature over a closed surface is 2*pi*chi, which is
    # 4*pi for any genus-0 mesh regardless of tessellation.
    for name, mesh in (
        ("tetrahedron", U.tetrahedron()),
        ("cube", U.unit_cube()),
        ("icosphere_0", U.icosphere(0)),
        ("icosphere_3", U.icosphere(3)),
        ("perturbed", U.perturbed_icosphere(np.random.default_rng(43), subdivisions=2)),
    ):
        with test.subTest(mesh=name):
            curvature = geo.vertex_gaussian_curvature(*U.to_warp(*mesh, device)).numpy()
            np.testing.assert_allclose(curvature.sum(), 4.0 * math.pi, rtol=1e-4)


def test_curvature_gauss_bonnet_torus(test, device):
    # A torus has Euler characteristic 0, so its total curvature vanishes.
    curvature = geo.vertex_gaussian_curvature(*U.to_warp(*U.torus(), device)).numpy()
    np.testing.assert_allclose(curvature.sum(), 0.0, atol=1e-3)
    # The vanishing total is a cancellation of real positive and negative
    # curvature on the outer and inner rings, not an all-zero field.
    test.assertGreater(curvature.max(), 0.01)
    test.assertLess(curvature.min(), -0.01)


def test_curvature_cube_corners(test, device):
    # Three right angles meet at each cube corner: 2*pi - 3*(pi/2) = pi/2.
    curvature = geo.vertex_gaussian_curvature(*U.to_warp(*U.unit_cube(), device)).numpy()
    np.testing.assert_allclose(curvature, np.full(8, math.pi / 2.0), rtol=1e-5)


def test_curvature_flat_interior_is_zero(test, device):
    # Interior vertices of a planar patch have zero angle defect. Boundary
    # vertices are not treated specially, so they are excluded here.
    nx = ny = 5
    p_np, f_np = U.planar_grid(nx, ny)
    curvature = geo.vertex_gaussian_curvature(*U.to_warp(p_np, f_np, device)).numpy().reshape(nx, ny)
    np.testing.assert_allclose(curvature[1:-1, 1:-1], 0.0, atol=1e-6)


def test_curvature_reference(test, device):
    p_np, f_np = U.perturbed_icosphere(np.random.default_rng(47), subdivisions=2)
    points, indices = U.to_warp(p_np, f_np, device)
    np.testing.assert_allclose(
        geo.vertex_gaussian_curvature(points, indices).numpy(),
        U.ref_vertex_gaussian_curvature(p_np, f_np),
        rtol=1e-4,
        atol=1e-5,
    )


def test_curvature_is_scale_and_rigid_invariant(test, device):
    # The angle defect is dimensionless: it is unchanged by rigid motion and by
    # uniform scaling (unlike a curvature *density*, which would scale as 1/s^2).
    p_np, f_np = U.icosphere(2)
    base = geo.vertex_gaussian_curvature(*U.to_warp(p_np, f_np, device)).numpy()

    axis = np.array([0.4, -0.2, 0.9])
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + math.sin(0.7) * k + (1.0 - math.cos(0.7)) * (k @ k)

    moved = (p_np @ rot.T + np.array([1.0, 2.0, -0.5])).astype(np.float32)
    np.testing.assert_allclose(
        geo.vertex_gaussian_curvature(*U.to_warp(moved, f_np, device)).numpy(), base, rtol=1e-4, atol=1e-5
    )

    scaled = (p_np * 7.0).astype(np.float32)
    np.testing.assert_allclose(
        geo.vertex_gaussian_curvature(*U.to_warp(scaled, f_np, device)).numpy(), base, rtol=1e-4, atol=1e-5
    )


devices = get_test_devices()


class TestGeometryVertex(unittest.TestCase):
    pass


add_function_test(TestGeometryVertex, "test_vertex_normals_reference", test_vertex_normals_reference, devices=devices)
add_function_test(
    TestGeometryVertex, "test_vertex_normals_schemes_differ", test_vertex_normals_schemes_differ, devices=devices
)
add_function_test(
    TestGeometryVertex, "test_vertex_normals_sphere_is_radial", test_vertex_normals_sphere_is_radial, devices=devices
)
add_function_test(
    TestGeometryVertex,
    "test_vertex_normals_normalized_unit_length",
    test_vertex_normals_normalized_unit_length,
    devices=devices,
)
add_function_test(
    TestGeometryVertex,
    "test_vertex_normals_area_weighting_matches_face_sum",
    test_vertex_normals_area_weighting_matches_face_sum,
    devices=devices,
)
add_function_test(
    TestGeometryVertex,
    "test_vertex_normals_weighting_accepts_int",
    test_vertex_normals_weighting_accepts_int,
    devices=devices,
)
add_function_test(
    TestGeometryVertex,
    "test_vertex_normals_invalid_weighting_raises",
    test_vertex_normals_invalid_weighting_raises,
    devices=devices,
)
add_function_test(
    TestGeometryVertex, "test_curvature_gauss_bonnet_genus_0", test_curvature_gauss_bonnet_genus_0, devices=devices
)
add_function_test(
    TestGeometryVertex, "test_curvature_gauss_bonnet_torus", test_curvature_gauss_bonnet_torus, devices=devices
)
add_function_test(TestGeometryVertex, "test_curvature_cube_corners", test_curvature_cube_corners, devices=devices)
add_function_test(
    TestGeometryVertex, "test_curvature_flat_interior_is_zero", test_curvature_flat_interior_is_zero, devices=devices
)
add_function_test(TestGeometryVertex, "test_curvature_reference", test_curvature_reference, devices=devices)
add_function_test(
    TestGeometryVertex,
    "test_curvature_is_scale_and_rigid_invariant",
    test_curvature_is_scale_and_rigid_invariant,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
