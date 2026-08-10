# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
add_function_test(
    TestGeometry,
    "test_cotmatrix_capturability",
    test_cotmatrix_capturability,
    devices=cuda_devices_with_mempool,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
