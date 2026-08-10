# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import unittest

import numpy as np

import warp as wp
import warp.geometry
import warp.sparse
from warp.tests.unittest_utils import *

# A regular octahedron: 6 vertices, 8 triangles, each vertex incident to 4
# triangles. Small enough to hardcode but with enough connectivity to
# exercise accumulation of contributions from multiple triangles per vertex.
_OCTAHEDRON_POINTS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)
_OCTAHEDRON_INDICES = np.array(
    [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ],
    dtype=np.int32,
)


def _reference_cotmatrix(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Dense reference cotangent Laplacian, matching libigl's igl::cotmatrix.

    Edge ``e`` of each triangle is opposite vertex ``e``: edge 0 = (v1, v2),
    edge 1 = (v2, v0), edge 2 = (v0, v1).
    """
    n = points.shape[0]
    v0, v1, v2 = points[indices[:, 0]], points[indices[:, 1]], points[indices[:, 2]]

    l2 = np.empty((indices.shape[0], 3), dtype=np.float64)
    l2[:, 0] = np.sum((v2 - v1) ** 2, axis=-1)
    l2[:, 1] = np.sum((v0 - v2) ** 2, axis=-1)
    l2[:, 2] = np.sum((v1 - v0) ** 2, axis=-1)

    double_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)

    c = np.empty_like(l2)
    c[:, 0] = (l2[:, 1] + l2[:, 2] - l2[:, 0]) / double_area / 4.0
    c[:, 1] = (l2[:, 2] + l2[:, 0] - l2[:, 1]) / double_area / 4.0
    c[:, 2] = (l2[:, 0] + l2[:, 1] - l2[:, 2]) / double_area / 4.0

    L = np.zeros((n, n), dtype=np.float64)
    edges = ((1, 2), (2, 0), (0, 1))
    for e, (a, b) in enumerate(edges):
        src, dst = indices[:, a], indices[:, b]
        np.add.at(L, (src, dst), c[:, e])
        np.add.at(L, (dst, src), c[:, e])
        np.add.at(L, (src, src), -c[:, e])
        np.add.at(L, (dst, dst), -c[:, e])

    return L


def _grid_surface(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An ``n`` by ``n`` grid of quads split into triangles, lifted out of the plane.

    The lift keeps the surface genuinely three-dimensional, so the comparison
    against ``warp.fem`` exercises tangential gradients rather than a flat 2D
    special case, and gives triangles a range of shapes.
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


def _bsr_to_dense(bsr) -> np.ndarray:
    dense = np.zeros(bsr.shape, dtype=np.float32)
    offsets = bsr.offsets.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy().reshape(-1)
    for row in range(bsr.nrow):
        for k in range(offsets[row], offsets[row + 1]):
            dense[row, columns[k]] += values[k]
    return dense


def test_cotmatrix(test, device):
    rng = np.random.default_rng(123)
    points_np = _OCTAHEDRON_POINTS + 0.05 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    L = warp.geometry.cotmatrix(points, indices)

    test.assertEqual(L.shape, (points_np.shape[0], points_np.shape[0]))

    dense = _bsr_to_dense(L)
    reference = _reference_cotmatrix(points_np, _OCTAHEDRON_INDICES)

    assert_np_equal(dense, reference, tol=1e-4)
    # The cotangent Laplacian is symmetric with each row summing to zero.
    assert_np_equal(dense, dense.T, tol=1e-6)
    assert_np_equal(dense.sum(axis=-1), np.zeros(dense.shape[0]), tol=1e-4)


def test_cotmatrix_out(test, device):
    """A caller-supplied output matrix produces the same result and reuses its storage."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    expected = _bsr_to_dense(warp.geometry.cotmatrix(points, indices))

    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    result = warp.geometry.cotmatrix(points, indices, out)

    test.assertIs(result, out)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # The first call grows the matrix to hold the 9 * num_triangles triplets, so a
    # second call on a mesh of the same topology must not reallocate its storage.
    columns_ptr, values_ptr = out.columns.ptr, out.values.ptr
    warp.geometry.cotmatrix(points, indices, out)

    test.assertEqual(out.columns.ptr, columns_ptr)
    test.assertEqual(out.values.ptr, values_ptr)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # Blocks already present in the output are discarded rather than accumulated.
    warp.sparse.bsr_set_diag(out, 1.0)
    warp.geometry.cotmatrix(points, indices, out)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)


@wp.kernel
def _weighted_value_sum(
    values: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    count: int,
    out_loss: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k < count:
        wp.atomic_add(out_loss, 0, values[k] * weights[k])


def test_cotmatrix_gradient(test, device):
    """Gradients through the triplet path match central finite differences.

    The loss is a fixed random combination of the matrix coefficients, which
    depends on every triangle and so exercises the whole assembly rather than a
    single entry.
    """
    rng = np.random.default_rng(11)
    points_np = _OCTAHEDRON_POINTS + 0.05 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    # The sparsity pattern is fixed by connectivity, so one weight vector stays
    # valid across the perturbed evaluations below.
    probe = warp.geometry.cotmatrix(wp.array(points_np, dtype=wp.vec3, device=device), indices)
    count = probe.nnz_sync()
    weights = wp.array(rng.standard_normal(count).astype(np.float32), dtype=wp.float32, device=device)

    def loss_of(positions_np: np.ndarray, tape: wp.Tape | None = None):
        points = wp.array(positions_np, dtype=wp.vec3, device=device, requires_grad=tape is not None)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=tape is not None)

        context = tape if tape is not None else contextlib.nullcontext()
        with context:
            L = warp.geometry.cotmatrix(points, indices)
            wp.launch(
                _weighted_value_sum,
                dim=count,
                inputs=[L.values, weights, count],
                outputs=[loss],
                device=device,
            )
        return points, loss

    tape = wp.Tape()
    points, loss = loss_of(points_np, tape)
    tape.backward(loss=loss)

    analytic = points.grad.numpy()
    test.assertTrue(np.isfinite(analytic).all(), "gradient contains non-finite entries")
    test.assertGreater(np.abs(analytic).max(), 0.0, "gradient is identically zero")

    eps = 1e-3
    numeric = np.zeros_like(points_np)
    for i in range(points_np.shape[0]):
        for c in range(3):
            forward = points_np.copy()
            backward = points_np.copy()
            forward[i, c] += eps
            backward[i, c] -= eps
            hi = loss_of(forward)[1].numpy()[0]
            lo = loss_of(backward)[1].numpy()[0]
            numeric[i, c] = (hi - lo) / (2.0 * eps)

    # Loose tolerance: the finite differences are taken in float32.
    assert_np_equal(analytic, numeric, tol=2e-2)


def test_cotmatrix_gradient_guards(test, device):
    """Combinations that cannot deliver gradients are rejected, not silently zeroed."""
    num_points = _OCTAHEDRON_POINTS.shape[0]
    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "not differentiable"):
        warp.geometry.cotmatrix(points, indices, construction="row_compress")

    with test.assertRaisesRegex(ValueError, "no gradient would reach"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
        warp.geometry.cotmatrix(points, indices, out)

    # The same call succeeds once the output can carry gradients.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    out.values.requires_grad = True
    warp.geometry.cotmatrix(points, indices, out)

    # Neither guard fires when gradients were never requested.
    plain = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    warp.geometry.cotmatrix(plain, indices, construction="row_compress")
    warp.geometry.cotmatrix(plain, indices, warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device))


def test_cotmatrix_row_compress(test, device):
    """Row-compressed construction produces the same matrix as the triplet path."""
    points_np, indices_np = _grid_surface(5)
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    expected = warp.geometry.cotmatrix(points, indices)
    compressed = warp.geometry.cotmatrix(points, indices, construction="row_compress")

    test.assertEqual(compressed.shape, expected.shape)
    # The reserved per-row capacity is an upper bound, so the compression must
    # actually pack the rows back down to the true non-zero count.
    test.assertEqual(compressed.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(compressed), _bsr_to_dense(expected), tol=1e-5)

    # Same again, writing into a caller-supplied matrix.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    result = warp.geometry.cotmatrix(points, indices, out, construction="row_compress")

    test.assertIs(result, out)
    test.assertEqual(out.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(out), _bsr_to_dense(expected), tol=1e-5)


def test_cotmatrix_row_compress_irregular(test, device):
    """The row-compressed path assumes nothing about manifoldness or valence.

    Row capacity comes from counting triangle incidences, so a mesh with a
    boundary, a non-manifold edge, and a wide range of vertex valences must
    still assemble exactly as the triplet path does.
    """
    points_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.2],
            [0.5, 0.2, 1.0],
            [2.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    # Edge (0, 1) is shared by three triangles, making it non-manifold; every
    # other edge is a boundary edge; vertex 5 touches a single triangle.
    indices_np = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [1, 5, 2]], dtype=np.int32)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    expected = warp.geometry.cotmatrix(points, indices)
    compressed = warp.geometry.cotmatrix(points, indices, construction="row_compress")

    test.assertEqual(compressed.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(compressed), _bsr_to_dense(expected), tol=1e-5)


def test_cotmatrix_construction_validation(test, device):
    """An unrecognized construction policy is rejected rather than ignored."""
    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "Unsupported `construction` policy"):
        warp.geometry.cotmatrix(points, indices, construction="sort")


def test_cotmatrix_reuse_topology(test, device):
    """Refilling an existing pattern reproduces a full rebuild, including after the mesh moves."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    # Seed the pattern with a full rebuild, then refill it.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.cotmatrix(points, indices, out)
    expected = _bsr_to_dense(out)

    out.values.zero_()
    warp.geometry.cotmatrix(points, indices, out, reuse_topology=True)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # The pattern depends only on connectivity, so deforming the mesh must still
    # give the same answer as a rebuild -- this is the case the option exists for.
    rng = np.random.default_rng(7)
    moved_np = points_np + 0.1 * rng.standard_normal(points_np.shape).astype(np.float32)
    moved = wp.array(moved_np, dtype=wp.vec3, device=device)

    warp.geometry.cotmatrix(moved, indices, out, reuse_topology=True)
    rebuilt = warp.geometry.cotmatrix(moved, indices)

    assert_np_equal(_bsr_to_dense(out), _bsr_to_dense(rebuilt), tol=1e-6)


def test_cotmatrix_reuse_topology_validation(test, device):
    """Reusing a pattern that cannot exist yet is rejected rather than silently zeroing."""
    num_points = _OCTAHEDRON_POINTS.shape[0]

    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "requires `out_cotmatrix`"):
        warp.geometry.cotmatrix(points, indices, reuse_topology=True)

    with test.assertRaisesRegex(ValueError, "it is empty"):
        empty = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
        warp.geometry.cotmatrix(points, indices, empty, reuse_topology=True)


def test_cotmatrix_out_validation(test, device):
    """A mismatched output matrix is rejected rather than silently misused."""
    num_points = _OCTAHEDRON_POINTS.shape[0]

    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "must have scalar type"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.float64, device=device)
        warp.geometry.cotmatrix(points, indices, out)

    with test.assertRaisesRegex(ValueError, "must have scalar \\(1x1\\) blocks"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.mat22, device=device)
        warp.geometry.cotmatrix(points, indices, out)

    with test.assertRaisesRegex(ValueError, "must have shape"):
        out = warp.sparse.bsr_zeros(num_points + 1, num_points + 1, wp.float32, device=device)
        warp.geometry.cotmatrix(points, indices, out)


def test_cotmatrix_capturability(test, device):
    """Assembly is CUDA-graph capturable, both allocating and into a supplied matrix."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    expected = _bsr_to_dense(warp.geometry.cotmatrix(points, indices))

    # Pre-size the output outside the capture so replay never has to grow it.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.cotmatrix(points, indices, out)
    out.values.zero_()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        warp.geometry.cotmatrix(points, indices, out)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)


devices = get_test_devices()
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestGeometry(unittest.TestCase):
    pass


add_function_test(TestGeometry, "test_cotmatrix", test_cotmatrix, devices=devices)
add_function_test(TestGeometry, "test_cotmatrix_out", test_cotmatrix_out, devices=devices)
add_function_test(TestGeometry, "test_cotmatrix_out_validation", test_cotmatrix_out_validation, devices=devices)
add_function_test(TestGeometry, "test_cotmatrix_gradient", test_cotmatrix_gradient, devices=devices)
add_function_test(TestGeometry, "test_cotmatrix_gradient_guards", test_cotmatrix_gradient_guards, devices=devices)
add_function_test(TestGeometry, "test_cotmatrix_row_compress", test_cotmatrix_row_compress, devices=devices)
add_function_test(
    TestGeometry,
    "test_cotmatrix_row_compress_irregular",
    test_cotmatrix_row_compress_irregular,
    devices=devices,
)
add_function_test(
    TestGeometry,
    "test_cotmatrix_construction_validation",
    test_cotmatrix_construction_validation,
    devices=devices,
)
add_function_test(TestGeometry, "test_cotmatrix_reuse_topology", test_cotmatrix_reuse_topology, devices=devices)
add_function_test(
    TestGeometry,
    "test_cotmatrix_reuse_topology_validation",
    test_cotmatrix_reuse_topology_validation,
    devices=devices,
)
add_function_test(
    TestGeometry,
    "test_cotmatrix_capturability",
    test_cotmatrix_capturability,
    devices=cuda_devices_with_mempool,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
