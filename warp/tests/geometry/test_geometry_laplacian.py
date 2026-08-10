# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cotangent Laplacian assembly: values, sparsity construction, and gradients."""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp.fem as fem
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


def _reference_laplacian(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Dense reference cotangent Laplacian, in the positive semi-definite convention.

    This is the negation of libigl's ``igl::cotmatrix``.

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
        np.add.at(L, (src, dst), -c[:, e])
        np.add.at(L, (dst, src), -c[:, e])
        np.add.at(L, (src, src), c[:, e])
        np.add.at(L, (dst, dst), c[:, e])

    return L


def _reference_adjacency(indices: np.ndarray, num_points: int) -> np.ndarray:
    """Dense reference vertex adjacency matrix.

    Assignment rather than accumulation, so an edge shared by two triangles
    still counts once. That is the distinction the Warp implementation has to
    get right by deduplicating its pattern.
    """
    A = np.zeros((num_points, num_points), dtype=np.float64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        A[indices[:, a], indices[:, b]] = 1.0
        A[indices[:, b], indices[:, a]] = 1.0
    return A


def _reference_uniform_laplacian(indices: np.ndarray, num_points: int) -> np.ndarray:
    A = _reference_adjacency(indices, num_points)
    return np.diag(A.sum(axis=-1)) - A


def _grid_surface(n: int, lift: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """An ``n`` by ``n`` grid of quads split into triangles, lifted out of the plane.

    The lift keeps the surface genuinely three-dimensional, so the comparison
    against ``warp.fem`` exercises tangential gradients rather than a flat 2D
    special case, and gives triangles a range of shapes. Without it the
    triangles are right isoceles, so every diagonal edge is opposite a right
    angle in both of its triangles and carries an exactly vanishing cotangent
    weight.
    """
    x = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    zz = 0.3 * np.sin(3.0 * xx) * np.cos(2.0 * yy) if lift else np.zeros_like(xx)
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


@fem.integrand
def _stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Laplacian bilinear form, whose P1 stiffness matrix is the cotangent Laplacian."""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


def _fem_stiffness(points: wp.array, indices: np.ndarray, device) -> "warp.sparse.BsrMatrix":
    """Assemble the P1 Laplacian stiffness matrix for the same mesh via ``warp.fem``."""
    geo = fem.Trimesh3D(
        tri_vertex_indices=wp.array(indices, dtype=int, device=device),
        positions=points,
    )
    space = fem.make_polynomial_space(geo, degree=1)
    domain = fem.Cells(geometry=geo)
    return fem.integrate(
        _stiffness_form,
        fields={
            "u": fem.make_trial(space=space, domain=domain),
            "v": fem.make_test(space=space, domain=domain),
        },
    )


def _bsr_to_dense(bsr) -> np.ndarray:
    dense = np.zeros(bsr.shape, dtype=np.float32)
    offsets = bsr.offsets.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy().reshape(-1)
    for row in range(bsr.nrow):
        for k in range(offsets[row], offsets[row + 1]):
            dense[row, columns[k]] += values[k]
    return dense


def test_laplacian(test, device):
    rng = np.random.default_rng(123)
    points_np = _OCTAHEDRON_POINTS + 0.05 * rng.standard_normal(_OCTAHEDRON_POINTS.shape).astype(np.float32)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    L = warp.geometry.laplacian(points, indices)

    test.assertEqual(L.shape, (points_np.shape[0], points_np.shape[0]))

    dense = _bsr_to_dense(L)
    reference = _reference_laplacian(points_np, _OCTAHEDRON_INDICES)

    assert_np_equal(dense, reference, tol=1e-4)
    # The cotangent Laplacian is symmetric with each row summing to zero.
    assert_np_equal(dense, dense.T, tol=1e-6)
    assert_np_equal(dense.sum(axis=-1), np.zeros(dense.shape[0]), tol=1e-4)

    # ... and positive semi-definite, with the constant vector in its kernel.
    eigenvalues = np.linalg.eigvalsh(dense.astype(np.float64))
    test.assertGreater(eigenvalues.min(), -1e-5)
    test.assertLess(abs(eigenvalues[0]), 1e-5)
    test.assertGreater(eigenvalues[1], 1e-3)


def test_laplacian_out(test, device):
    """A caller-supplied output matrix produces the same result and reuses its storage."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    expected = _bsr_to_dense(warp.geometry.laplacian(points, indices))

    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    result = warp.geometry.laplacian(points, indices, out)

    test.assertIs(result, out)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # The first call grows the matrix to hold the 9 * num_triangles triplets, so a
    # second call on a mesh of the same topology must not reallocate its storage.
    columns_ptr, values_ptr = out.columns.ptr, out.values.ptr
    warp.geometry.laplacian(points, indices, out)

    test.assertEqual(out.columns.ptr, columns_ptr)
    test.assertEqual(out.values.ptr, values_ptr)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # Blocks already present in the output are discarded rather than accumulated.
    warp.sparse.bsr_set_diag(out, 1.0)
    warp.geometry.laplacian(points, indices, out)
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


def test_laplacian_gradient(test, device):
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
    probe = warp.geometry.laplacian(wp.array(points_np, dtype=wp.vec3, device=device), indices)
    count = probe.nnz_sync()
    weights = wp.array(rng.standard_normal(count).astype(np.float32), dtype=wp.float32, device=device)

    def loss_of(positions_np: np.ndarray, tape: wp.Tape | None = None):
        points = wp.array(positions_np, dtype=wp.vec3, device=device, requires_grad=tape is not None)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=tape is not None)

        context = tape if tape is not None else contextlib.nullcontext()
        with context:
            L = warp.geometry.laplacian(points, indices)
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


def test_laplacian_gradient_guards(test, device):
    """Combinations that cannot deliver gradients are rejected, not silently zeroed."""
    num_points = _OCTAHEDRON_POINTS.shape[0]
    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "not differentiable"):
        warp.geometry.laplacian(points, indices, construction="row_compress")

    with test.assertRaisesRegex(ValueError, "no gradient would reach"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
        warp.geometry.laplacian(points, indices, out)

    # The same call succeeds once the output can carry gradients.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    out.values.requires_grad = True
    warp.geometry.laplacian(points, indices, out)

    # Neither guard fires when gradients were never requested.
    plain = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    warp.geometry.laplacian(plain, indices, construction="row_compress")
    warp.geometry.laplacian(plain, indices, warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device))


def test_laplacian_row_compress(test, device):
    """Row-compressed construction produces the same matrix as the triplet path."""
    points_np, indices_np = _grid_surface(5)
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    expected = warp.geometry.laplacian(points, indices)
    compressed = warp.geometry.laplacian(points, indices, construction="row_compress")

    test.assertEqual(compressed.shape, expected.shape)
    # The reserved per-row capacity is an upper bound, so the compression must
    # actually pack the rows back down to the true non-zero count.
    test.assertEqual(compressed.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(compressed), _bsr_to_dense(expected), tol=1e-5)

    # Same again, writing into a caller-supplied matrix.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    result = warp.geometry.laplacian(points, indices, out, construction="row_compress")

    test.assertIs(result, out)
    test.assertEqual(out.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(out), _bsr_to_dense(expected), tol=1e-5)


def test_laplacian_row_compress_irregular(test, device):
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

    expected = warp.geometry.laplacian(points, indices)
    compressed = warp.geometry.laplacian(points, indices, construction="row_compress")

    test.assertEqual(compressed.nnz_sync(), expected.nnz_sync())
    assert_np_equal(_bsr_to_dense(compressed), _bsr_to_dense(expected), tol=1e-5)


def test_laplacian_construction_validation(test, device):
    """An unrecognized construction policy is rejected rather than ignored."""
    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "Unsupported `construction` policy"):
        warp.geometry.laplacian(points, indices, construction="sort")


def test_laplacian_reuse_topology(test, device):
    """Refilling an existing pattern reproduces a full rebuild, including after the mesh moves."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    # Seed the pattern with a full rebuild, then refill it.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.laplacian(points, indices, out)
    expected = _bsr_to_dense(out)

    out.values.zero_()
    warp.geometry.laplacian(points, indices, out, reuse_topology=True)
    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)

    # The pattern depends only on connectivity, so deforming the mesh must still
    # give the same answer as a rebuild -- this is the case the option exists for.
    rng = np.random.default_rng(7)
    moved_np = points_np + 0.1 * rng.standard_normal(points_np.shape).astype(np.float32)
    moved = wp.array(moved_np, dtype=wp.vec3, device=device)

    warp.geometry.laplacian(moved, indices, out, reuse_topology=True)
    rebuilt = warp.geometry.laplacian(moved, indices)

    assert_np_equal(_bsr_to_dense(out), _bsr_to_dense(rebuilt), tol=1e-6)


def test_laplacian_pattern_keeps_vanishing_weights(test, device):
    """The pattern is topological, so an edge whose cotangent weight is exactly zero keeps its entry.

    A flat grid of right isoceles triangles puts a right angle opposite every
    diagonal edge, in both of the triangles sharing it, so those edges assemble
    to exactly zero. Pruning them would leave the pattern unable to represent
    the mesh once it deforms.
    """
    points_np, indices_np = _grid_surface(3, lift=False)
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    # The mesh does have vanishing weights, or the rest of the test proves nothing.
    flat = _bsr_to_dense(warp.geometry.laplacian(points, indices))
    connectivity = _reference_uniform_laplacian(indices_np, num_points) != 0.0
    test.assertTrue(np.any((flat == 0.0) & connectivity))

    for construction in ("triplets", "row_compress"):
        cotangent = warp.geometry.laplacian(points, indices, construction=construction)
        # Same entries as the connectivity-only pattern, none dropped.
        assert_np_equal(_bsr_to_dense(cotangent) != 0.0, flat != 0.0)
        test.assertEqual(int(cotangent.offsets.numpy()[num_points]), int(np.count_nonzero(connectivity)))

        # Refilling that pattern after the mesh moves must still match a rebuild:
        # the edges that vanished at rest no longer do.
        rng = np.random.default_rng(11)
        moved_np = points_np + 0.1 * rng.standard_normal(points_np.shape).astype(np.float32)
        moved = wp.array(moved_np, dtype=wp.vec3, device=device)

        warp.geometry.laplacian(moved, indices, cotangent, reuse_topology=True)
        assert_np_equal(_bsr_to_dense(cotangent), _reference_laplacian(moved_np, indices_np), tol=1e-4)

        # And the pattern is shared across weightings, so the degrees a uniform
        # refill counts off it are the true vertex degrees.
        warp.geometry.laplacian(None, indices, cotangent, weighting=uniform, reuse_topology=True)
        assert_np_equal(_bsr_to_dense(cotangent), _reference_uniform_laplacian(indices_np, num_points), tol=1e-6)


def test_laplacian_reuse_topology_validation(test, device):
    """Reusing a pattern that cannot exist yet is rejected rather than silently zeroing."""
    num_points = _OCTAHEDRON_POINTS.shape[0]

    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "requires `out_laplacian`"):
        warp.geometry.laplacian(points, indices, reuse_topology=True)

    with test.assertRaisesRegex(ValueError, "it is empty"):
        empty = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
        warp.geometry.laplacian(points, indices, empty, reuse_topology=True)


def test_laplacian_out_validation(test, device):
    """A mismatched output matrix is rejected rather than silently misused."""
    num_points = _OCTAHEDRON_POINTS.shape[0]

    points = wp.array(_OCTAHEDRON_POINTS, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "must have scalar type"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.float64, device=device)
        warp.geometry.laplacian(points, indices, out)

    with test.assertRaisesRegex(ValueError, "must have scalar \\(1x1\\) blocks"):
        out = warp.sparse.bsr_zeros(num_points, num_points, wp.mat22, device=device)
        warp.geometry.laplacian(points, indices, out)

    with test.assertRaisesRegex(ValueError, "must have shape"):
        out = warp.sparse.bsr_zeros(num_points + 1, num_points + 1, wp.float32, device=device)
        warp.geometry.laplacian(points, indices, out)


def test_laplacian_capturability(test, device):
    """Assembly is CUDA-graph capturable, both allocating and into a supplied matrix."""
    points_np = _OCTAHEDRON_POINTS
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(_OCTAHEDRON_INDICES.flatten(), dtype=wp.int32, device=device)

    expected = _bsr_to_dense(warp.geometry.laplacian(points, indices))

    # Pre-size the output outside the capture so replay never has to grow it.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.laplacian(points, indices, out)
    out.values.zero_()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        warp.geometry.laplacian(points, indices, out)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    assert_np_equal(_bsr_to_dense(out), expected, tol=1e-6)


##########################################################################
## Uniform weighting and vertex adjacency
##########################################################################


def test_uniform_laplacian(test, device):
    """The uniform Laplacian is ``D - A`` of the mesh's edge graph.

    A grid surface has boundary vertices of lower degree than interior ones and
    interior edges shared by two triangles, so it catches both a wrong degree
    and an edge counted once per incident triangle.
    """
    points_np, indices_np = _grid_surface(5)
    num_points = points_np.shape[0]

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    reference = _reference_uniform_laplacian(indices_np, num_points)

    for construction in ("triplets", "row_compress"):
        L = warp.geometry.laplacian(
            points,
            indices,
            weighting=warp.geometry.LaplacianWeighting.UNIFORM,
            construction=construction,
        )
        dense = _bsr_to_dense(L)

        test.assertEqual(L.shape, (num_points, num_points))
        assert_np_equal(dense, reference, tol=1e-6)
        # Symmetric, rows summing to zero, and positive semi-definite.
        assert_np_equal(dense, dense.T, tol=1e-6)
        assert_np_equal(dense.sum(axis=-1), np.zeros(num_points), tol=1e-6)
        test.assertGreater(np.linalg.eigvalsh(dense.astype(np.float64)).min(), -1e-5)

        # Off-diagonals are exactly -1: a doubled interior edge would read -2.
        off_diagonal = dense[~np.eye(num_points, dtype=bool)]
        test.assertEqual(set(np.unique(off_diagonal).tolist()), {-1.0, 0.0})


def test_uniform_laplacian_differs_from_cotangent(test, device):
    """The two weightings share a sparsity pattern but not their values."""
    points_np, indices_np = _grid_surface(4)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    cotangent = warp.geometry.laplacian(points, indices)
    uniform = warp.geometry.laplacian(points, indices, weighting=warp.geometry.LaplacianWeighting.UNIFORM)

    test.assertEqual(cotangent.nnz_sync(), uniform.nnz_sync())
    dense_cotangent = _bsr_to_dense(cotangent)
    dense_uniform = _bsr_to_dense(uniform)
    assert_np_equal((dense_cotangent != 0.0).astype(np.float32), (dense_uniform != 0.0).astype(np.float32), tol=0)
    test.assertGreater(np.abs(dense_cotangent - dense_uniform).max(), 0.1)


def test_uniform_laplacian_without_points(test, device):
    """Uniform weighting reads no positions, so ``points`` may be omitted."""
    points_np, indices_np = _grid_surface(4)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    reference = _reference_uniform_laplacian(indices_np, num_points)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    # Explicit vertex count.
    explicit = warp.geometry.laplacian(None, indices, weighting=uniform, num_points=num_points, device=device)
    assert_np_equal(_bsr_to_dense(explicit), reference, tol=1e-6)

    # Vertex count recovered from the largest index.
    inferred = warp.geometry.laplacian(None, indices, weighting=uniform, device=device)
    test.assertEqual(inferred.shape, (num_points, num_points))
    assert_np_equal(_bsr_to_dense(inferred), reference, tol=1e-6)

    # Vertex count taken from the supplied output matrix.
    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    from_out = warp.geometry.laplacian(None, indices, out, weighting=uniform)
    test.assertIs(from_out, out)
    assert_np_equal(_bsr_to_dense(out), reference, tol=1e-6)

    # A mesh whose last vertices are unreferenced needs the count spelled out,
    # since no index mentions them.
    padded = warp.geometry.laplacian(None, indices, weighting=uniform, num_points=num_points + 3, device=device)
    test.assertEqual(padded.shape, (num_points + 3, num_points + 3))
    assert_np_equal(_bsr_to_dense(padded)[:num_points, :num_points], reference, tol=1e-6)


def test_uniform_laplacian_argument_validation(test, device):
    points_np, indices_np = _grid_surface(3)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    with test.assertRaisesRegex(ValueError, "`points` is required for cotangent weighting"):
        warp.geometry.laplacian(None, indices, num_points=points_np.shape[0], device=device)

    with test.assertRaisesRegex(ValueError, "Pass only one of them"):
        warp.geometry.laplacian(points, indices, weighting=uniform, num_points=points_np.shape[0] + 1)

    with test.assertRaisesRegex(ValueError, "must be non-negative"):
        warp.geometry.laplacian(None, indices, weighting=uniform, num_points=-1, device=device)

    with test.assertRaises(ValueError):
        warp.geometry.laplacian(points, indices, weighting="uniform")

    # Passing a matching count alongside positions is allowed.
    warp.geometry.laplacian(points, indices, weighting=uniform, num_points=points_np.shape[0])


def test_uniform_laplacian_gradient_is_zero(test, device):
    """Uniform weighting is connectivity-only, so it is constant in ``points``.

    The guards that reject non-differentiable cotangent paths must not fire
    here: a zero gradient is the correct answer, not a dropped one.
    """
    points_np, indices_np = _grid_surface(3)
    points = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    # Neither guard applies, including the one on row-compressed construction.
    warp.geometry.laplacian(points, indices, weighting=uniform, construction="row_compress")
    out = warp.sparse.bsr_zeros(points_np.shape[0], points_np.shape[0], wp.float32, device=device)
    warp.geometry.laplacian(points, indices, out, weighting=uniform)

    tape = wp.Tape()
    with tape:
        L = warp.geometry.laplacian(points, indices, weighting=uniform)

    count = L.nnz_sync()
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    weights = wp.full(count, 1.0, dtype=wp.float32, device=device)
    wp.launch(_weighted_value_sum, dim=count, inputs=[L.values, weights, count], outputs=[loss], device=device)

    tape.backward(loss=loss)
    assert_np_equal(points.grad.numpy(), np.zeros_like(points_np), tol=0)


def test_uniform_laplacian_reuse_topology(test, device):
    points_np, indices_np = _grid_surface(4)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    reference = _reference_uniform_laplacian(indices_np, num_points)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.laplacian(None, indices, out, weighting=uniform, num_points=num_points)

    out.values.zero_()
    warp.geometry.laplacian(None, indices, out, weighting=uniform, num_points=num_points, reuse_topology=True)
    assert_np_equal(_bsr_to_dense(out), reference, tol=1e-6)

    with test.assertRaisesRegex(ValueError, "requires `out_laplacian`"):
        warp.geometry.laplacian(None, indices, weighting=uniform, num_points=num_points, reuse_topology=True)

    empty = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    with test.assertRaisesRegex(ValueError, "it is empty"):
        warp.geometry.laplacian(None, indices, empty, weighting=uniform, reuse_topology=True)


def test_uniform_laplacian_capturability(test, device):
    """The uniform path is CUDA-graph capturable once the vertex count is known."""
    points_np, indices_np = _grid_surface(4)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    reference = _reference_uniform_laplacian(indices_np, num_points)
    uniform = warp.geometry.LaplacianWeighting.UNIFORM

    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    warp.geometry.laplacian(None, indices, out, weighting=uniform, num_points=num_points)
    out.values.zero_()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        warp.geometry.laplacian(None, indices, out, weighting=uniform, num_points=num_points)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    assert_np_equal(_bsr_to_dense(out), reference, tol=1e-6)


def test_vertex_adjacency_matrix(test, device):
    """Adjacency holds one entry per directed edge, with no self-loops."""
    points_np, indices_np = _grid_surface(5)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    reference = _reference_adjacency(indices_np, num_points)

    for construction in ("triplets", "row_compress"):
        A = warp.geometry.vertex_adjacency_matrix(
            indices, num_points=num_points, construction=construction, device=device
        )
        dense = _bsr_to_dense(A)

        test.assertEqual(A.shape, (num_points, num_points))
        assert_np_equal(dense, reference, tol=1e-6)
        assert_np_equal(dense, dense.T, tol=1e-6)
        assert_np_equal(np.diag(dense), np.zeros(num_points), tol=0)
        # The diagonal is absent from the pattern, not merely zero.
        test.assertEqual(A.nnz_sync(), int(reference.sum()))


def test_vertex_adjacency_matrix_completes_uniform_laplacian(test, device):
    """``L == D - A``, which is what ties the two entry points together."""
    points_np, indices_np = _grid_surface(4)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    L = warp.geometry.laplacian(
        None, indices, weighting=warp.geometry.LaplacianWeighting.UNIFORM, num_points=num_points, device=device
    )
    A = warp.geometry.vertex_adjacency_matrix(indices, num_points=num_points, device=device)

    dense_L = _bsr_to_dense(L)
    dense_A = _bsr_to_dense(A)

    assert_np_equal(dense_L, np.diag(dense_A.sum(axis=-1)) - dense_A, tol=1e-6)
    # Every vertex of a grid surface has neighbors, so no row is trivially empty.
    test.assertGreater(dense_A.sum(axis=-1).min(), 0.0)


def test_vertex_adjacency_matrix_out_and_reuse(test, device):
    points_np, indices_np = _grid_surface(4)
    num_points = points_np.shape[0]

    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)
    reference = _reference_adjacency(indices_np, num_points)

    out = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    result = warp.geometry.vertex_adjacency_matrix(indices, out)

    test.assertIs(result, out)
    assert_np_equal(_bsr_to_dense(out), reference, tol=1e-6)

    out.values.zero_()
    warp.geometry.vertex_adjacency_matrix(indices, out, reuse_topology=True)
    assert_np_equal(_bsr_to_dense(out), reference, tol=1e-6)

    # The vertex count is recovered from the largest index when nothing else says.
    inferred = warp.geometry.vertex_adjacency_matrix(indices, device=device)
    test.assertEqual(inferred.shape, (num_points, num_points))
    assert_np_equal(_bsr_to_dense(inferred), reference, tol=1e-6)


def test_vertex_adjacency_matrix_validation(test, device):
    points_np, indices_np = _grid_surface(3)
    num_points = points_np.shape[0]
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "Unsupported `construction` policy"):
        warp.geometry.vertex_adjacency_matrix(indices, num_points=num_points, construction="sort", device=device)

    with test.assertRaisesRegex(ValueError, "requires `out_adjacency`"):
        warp.geometry.vertex_adjacency_matrix(indices, num_points=num_points, reuse_topology=True, device=device)

    empty = warp.sparse.bsr_zeros(num_points, num_points, wp.float32, device=device)
    with test.assertRaisesRegex(ValueError, "it is empty"):
        warp.geometry.vertex_adjacency_matrix(indices, empty, reuse_topology=True)

    with test.assertRaisesRegex(ValueError, "must have shape"):
        out = warp.sparse.bsr_zeros(num_points + 1, num_points + 1, wp.float32, device=device)
        warp.geometry.vertex_adjacency_matrix(indices, out, num_points=num_points)


def test_laplacian_matches_fem_stiffness(test, device):
    """The cotangent Laplacian is the P1 Laplacian stiffness matrix.

    Seen as a bilinear form, ``laplacian`` assembles ``int(grad(u) . grad(v))``
    over the mesh. ``warp.fem`` assembles the same form from the physics side,
    by quadrature rather than in closed form, and in the same positive
    semi-definite sign convention, so the two agree outright.
    """
    points_np, indices_np = _grid_surface(6)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np.flatten(), dtype=wp.int32, device=device)

    L = warp.geometry.laplacian(points, indices)
    K = _fem_stiffness(points, indices_np, device)

    test.assertEqual(L.shape, K.shape)
    # Both operators couple exactly the vertex pairs that share an edge, so the
    # sparsity patterns must agree and not merely the dense values.
    test.assertEqual(L.nnz_sync(), K.nnz_sync())

    dense_L = _bsr_to_dense(L)
    dense_K = _bsr_to_dense(K)

    assert_np_equal(dense_L, dense_K, tol=1e-4)
    # Guard against both being trivially zero, which would satisfy the above.
    test.assertGreater(np.abs(dense_L).max(), 1.0)


devices = get_test_devices()
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestGeometryLaplacian(unittest.TestCase):
    pass


add_function_test(TestGeometryLaplacian, "test_laplacian", test_laplacian, devices=devices)
add_function_test(TestGeometryLaplacian, "test_laplacian_out", test_laplacian_out, devices=devices)
add_function_test(
    TestGeometryLaplacian, "test_laplacian_out_validation", test_laplacian_out_validation, devices=devices
)
add_function_test(TestGeometryLaplacian, "test_laplacian_gradient", test_laplacian_gradient, devices=devices)
add_function_test(
    TestGeometryLaplacian, "test_laplacian_gradient_guards", test_laplacian_gradient_guards, devices=devices
)
add_function_test(TestGeometryLaplacian, "test_laplacian_row_compress", test_laplacian_row_compress, devices=devices)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_row_compress_irregular",
    test_laplacian_row_compress_irregular,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_construction_validation",
    test_laplacian_construction_validation,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian, "test_laplacian_reuse_topology", test_laplacian_reuse_topology, devices=devices
)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_pattern_keeps_vanishing_weights",
    test_laplacian_pattern_keeps_vanishing_weights,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_reuse_topology_validation",
    test_laplacian_reuse_topology_validation,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_matches_fem_stiffness",
    test_laplacian_matches_fem_stiffness,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_laplacian_capturability",
    test_laplacian_capturability,
    devices=cuda_devices_with_mempool,
)
add_function_test(TestGeometryLaplacian, "test_uniform_laplacian", test_uniform_laplacian, devices=devices)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_differs_from_cotangent",
    test_uniform_laplacian_differs_from_cotangent,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_without_points",
    test_uniform_laplacian_without_points,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_argument_validation",
    test_uniform_laplacian_argument_validation,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_gradient_is_zero",
    test_uniform_laplacian_gradient_is_zero,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_reuse_topology",
    test_uniform_laplacian_reuse_topology,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_uniform_laplacian_capturability",
    test_uniform_laplacian_capturability,
    devices=cuda_devices_with_mempool,
)
add_function_test(TestGeometryLaplacian, "test_vertex_adjacency_matrix", test_vertex_adjacency_matrix, devices=devices)
add_function_test(
    TestGeometryLaplacian,
    "test_vertex_adjacency_matrix_completes_uniform_laplacian",
    test_vertex_adjacency_matrix_completes_uniform_laplacian,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_vertex_adjacency_matrix_out_and_reuse",
    test_vertex_adjacency_matrix_out_and_reuse,
    devices=devices,
)
add_function_test(
    TestGeometryLaplacian,
    "test_vertex_adjacency_matrix_validation",
    test_vertex_adjacency_matrix_validation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
