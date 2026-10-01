# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Barycentric lumped mass matrix assembly: values, structure, and gradients."""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp.geometry
import warp.sparse
from warp.tests.unittest_utils import *

# Two triangles forming a unit square. Vertices 0 and 2 belong to both
# triangles, vertices 1 and 3 to one each, so lumping accumulates.
_SQUARE_POINTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
_SQUARE_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

# A single corner tetrahedron of volume 1/6.
_TET_POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
_TET_INDICES = np.array([[0, 1, 2, 3]], dtype=np.int32)


def _diagonal(mat) -> np.ndarray:
    diag = np.zeros(mat.shape[0], dtype=np.float64)
    rows = mat.uncompress_rows().numpy()
    cols = mat.columns.numpy()
    vals = mat.values.numpy().reshape(-1)
    for k in range(mat.nnz_sync()):
        if rows[k] == cols[k]:
            diag[rows[k]] += vals[k]
    return diag


def _max_off_diagonal(mat) -> float:
    rows = mat.uncompress_rows().numpy()
    cols = mat.columns.numpy()
    vals = mat.values.numpy().reshape(-1)
    worst = 0.0
    for k in range(mat.nnz_sync()):
        if rows[k] != cols[k]:
            worst = max(worst, abs(float(vals[k])))
    return worst


@wp.kernel
def _diag_weighted_sum(
    values: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    count: int,
    out_loss: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k < count:
        wp.atomic_add(out_loss, 0, values[k] * weights[k])


def test_triangle_massmatrix(test, device):
    points = wp.array(_SQUARE_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_SQUARE_TRIS.flatten(), dtype=wp.int32, device=device)

    mass = warp.geometry.massmatrix(points, indices)
    diag = _diagonal(mass)

    # Each triangle has area 0.5 and gives area/3 to each vertex.
    expected = np.array([2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]) * 0.5
    assert_np_equal(diag.astype(np.float32), expected.astype(np.float32), tol=1e-5)
    test.assertLess(_max_off_diagonal(mass), 1e-9)
    # The diagonal sums to the total mesh area.
    test.assertAlmostEqual(float(diag.sum()), 1.0, places=5)


def test_tet_massmatrix(test, device):
    points = wp.array(_TET_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_TET_INDICES, dtype=wp.vec4i, device=device)

    mass = warp.geometry.massmatrix(points, indices)
    diag = _diagonal(mass)

    # Volume 1/6 split equally among four vertices.
    expected = np.full(4, (1.0 / 6.0) / 4.0)
    assert_np_equal(diag.astype(np.float32), expected.astype(np.float32), tol=1e-6)
    test.assertLess(_max_off_diagonal(mass), 1e-9)
    test.assertAlmostEqual(float(diag.sum()), 1.0 / 6.0, places=6)


def test_voronoi_massmatrix_obtuse(test, device):
    """The mixed Voronoi rule splits an obtuse triangle half/quarter and sums to its area."""
    # Obtuse at vertex 2: the vectors from it to the other two have negative dot.
    points_np = np.array([[0, 0, 0], [2, 0, 0], [0.5, 0.3, 0]], dtype=np.float32)
    tri = np.array([[0, 1, 2]], dtype=np.int32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(tri, dtype=wp.vec3i, device=device)

    diag = _diagonal(warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI))
    area = 0.3  # 0.5 * base(2) * height(0.3)
    # Obtuse vertex gets half the area, the other two a quarter each.
    assert_np_equal(diag.astype(np.float32), np.array([area / 4, area / 4, area / 2], np.float32), tol=1e-6)
    test.assertAlmostEqual(float(diag.sum()), area, places=6)


def test_voronoi_massmatrix_total_area_and_positivity(test, device):
    """Voronoi lumping is positive and sums to the mesh area, and differs from barycentric."""
    rng = np.random.default_rng(4)
    points_np = _SQUARE_POINTS + 0.1 * rng.standard_normal(_SQUARE_POINTS.shape).astype(np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_SQUARE_TRIS.flatten(), dtype=wp.int32, device=device)

    voronoi = _diagonal(warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI))
    barycentric = _diagonal(warp.geometry.massmatrix(points, indices))

    test.assertTrue((voronoi > 0).all())
    # Both lumpings preserve the total area.
    test.assertAlmostEqual(float(voronoi.sum()), float(barycentric.sum()), places=5)
    # But they distribute it differently.
    test.assertGreater(np.abs(voronoi - barycentric).max(), 1e-3)


def test_voronoi_massmatrix_rejects_tets(test, device):
    points = wp.array(_TET_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_TET_INDICES, dtype=wp.vec4i, device=device)
    with test.assertRaises(NotImplementedError):
        warp.geometry.massmatrix(points, indices, kind=warp.geometry.MassMatrixType.VORONOI)


def test_massmatrix_dispatch(test, device):
    """Flat ``int32`` and typed index arrays produce the same mass matrix."""
    points = wp.array(_TET_POINTS, dtype=wp.vec3, device=device)
    flat = wp.array(_TET_INDICES.flatten(), dtype=wp.int32, device=device)
    typed = wp.array(_TET_INDICES, dtype=wp.vec4i, device=device)

    from_flat = _diagonal(warp.geometry.massmatrix(points, flat, simplex_size=4))
    from_typed = _diagonal(warp.geometry.massmatrix(points, typed))
    assert_np_equal(from_flat.astype(np.float32), from_typed.astype(np.float32), tol=1e-7)

    with test.assertRaisesRegex(ValueError, "fixes the simplex size"):
        warp.geometry.massmatrix(points, typed, simplex_size=3)


def test_massmatrix_out_reuse(test, device):
    points = wp.array(_SQUARE_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_SQUARE_TRIS.flatten(), dtype=wp.int32, device=device)

    out = warp.sparse.bsr_zeros(4, 4, wp.float32, device=device)
    returned = warp.geometry.massmatrix(points, indices, out)
    test.assertIs(returned, out)

    reference = _diagonal(warp.geometry.massmatrix(points, indices))
    assert_np_equal(_diagonal(out).astype(np.float32), reference.astype(np.float32), tol=1e-6)


def test_massmatrix_gradient(test, device):
    """Gradients of the lumped mass entries match central finite differences."""
    rng = np.random.default_rng(1)
    points_np = (_TET_POINTS + 0.1 * rng.standard_normal(_TET_POINTS.shape)).astype(np.float32)
    indices = wp.array(_TET_INDICES.flatten(), dtype=wp.int32, device=device)

    probe = warp.geometry.massmatrix(wp.array(points_np, dtype=wp.vec3, device=device), indices, simplex_size=4)
    count = probe.nnz_sync()
    weights = wp.array(rng.standard_normal(count).astype(np.float32), dtype=wp.float32, device=device)

    def loss_of(positions_np, tape=None):
        points = wp.array(positions_np, dtype=wp.vec3, device=device, requires_grad=tape is not None)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=tape is not None)
        context = tape if tape is not None else contextlib.nullcontext()
        with context:
            mass = warp.geometry.massmatrix(points, indices, simplex_size=4)
            wp.launch(
                _diag_weighted_sum, dim=count, inputs=[mass.values, weights, count], outputs=[loss], device=device
            )
        return points, loss

    tape = wp.Tape()
    points, loss = loss_of(points_np, tape)
    tape.backward(loss=loss)
    analytic = points.grad.numpy()

    test.assertTrue(np.isfinite(analytic).all())
    test.assertGreater(np.abs(analytic).max(), 0.0)

    eps = 1e-3
    numeric = np.zeros_like(points_np)
    for i in range(points_np.shape[0]):
        for c in range(3):
            forward = points_np.copy()
            backward = points_np.copy()
            forward[i, c] += eps
            backward[i, c] -= eps
            numeric[i, c] = (loss_of(forward)[1].numpy()[0] - loss_of(backward)[1].numpy()[0]) / (2.0 * eps)

    assert_np_equal(analytic, numeric, tol=2e-2)


devices = get_test_devices()


class TestGeometryMassMatrix(unittest.TestCase):
    pass


add_function_test(TestGeometryMassMatrix, "test_triangle_massmatrix", test_triangle_massmatrix, devices=devices)
add_function_test(TestGeometryMassMatrix, "test_tet_massmatrix", test_tet_massmatrix, devices=devices)
add_function_test(
    TestGeometryMassMatrix, "test_voronoi_massmatrix_obtuse", test_voronoi_massmatrix_obtuse, devices=devices
)
add_function_test(
    TestGeometryMassMatrix,
    "test_voronoi_massmatrix_total_area_and_positivity",
    test_voronoi_massmatrix_total_area_and_positivity,
    devices=devices,
)
add_function_test(
    TestGeometryMassMatrix,
    "test_voronoi_massmatrix_rejects_tets",
    test_voronoi_massmatrix_rejects_tets,
    devices=devices,
)
add_function_test(TestGeometryMassMatrix, "test_massmatrix_dispatch", test_massmatrix_dispatch, devices=devices)
add_function_test(TestGeometryMassMatrix, "test_massmatrix_out_reuse", test_massmatrix_out_reuse, devices=devices)
add_function_test(TestGeometryMassMatrix, "test_massmatrix_gradient", test_massmatrix_gradient, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
