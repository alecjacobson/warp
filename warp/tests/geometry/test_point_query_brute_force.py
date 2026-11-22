# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np

import warp as wp
import warp.geometry
from warp._src.geometry import point_query
from warp.tests.geometry import test_point_query_batched as batched
from warp.tests.geometry.test_point_query import reference
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

BACKEND = wp.geometry.PointQueryBackend.BRUTE_FORCE


def test_values(test, device):
    rng = np.random.default_rng(7348)
    # Both axes cross tile boundaries; tails must never mix adjacent clouds.
    for batches, m, n in ((3, 259, 257), (1, 5, 7), (2, 0, 3), (0, 3, 4), (2, 3, 0)):
        targets = rng.normal(size=(batches, m, 3)).astype(np.float32)
        queries = rng.normal(size=(batches, n, 3)).astype(np.float32)
        points = wp.array(targets, dtype=wp.vec3, device=device)
        samples = wp.array(queries, dtype=wp.vec3, device=device)
        for k in (1, 3, 11):
            result = wp.geometry.knn(points, samples, k, backend=BACKEND)
            batched.check_batches(test, targets, queries, k, result)
    # Also exercise the nonbatched ABI with caller-provided int64 outputs.
    targets = rng.normal(size=(19, 3)).astype(np.float32)
    queries = rng.normal(size=(17, 3)).astype(np.float32)
    points = wp.array(targets, dtype=wp.vec3, device=device)
    samples = wp.array(queries, dtype=wp.vec3, device=device)
    indices = wp.empty((17, 3), dtype=wp.int64, device=device)
    result = wp.geometry.knn(points, samples, 3, indices=indices, backend=BACKEND, squared=False)
    ids, d = reference(targets, queries, 3, np.inf, [-1] * 17)
    test.assertIs(result[1], indices)
    np.testing.assert_array_equal(indices.numpy(), ids)
    np.testing.assert_allclose(result[2].numpy(), np.sqrt(d), rtol=2e-6)


def test_filtering(test, device):
    # Ties span tiles and must prefer the lowest original index.
    targets = np.zeros((2, 260, 3), dtype=np.float32)
    targets[:, :, 0] = 1.0
    targets[:, 0] = 0.0
    queries = np.array([[[0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0], [2, 0, 0]]] * 2, dtype=np.float32)
    points = wp.array(targets, dtype=wp.vec3, device=device)
    samples = wp.array(queries, dtype=wp.vec3, device=device)
    excludes = wp.array([[-1, -1, -1, 1]] * 2, dtype=int, device=device)
    for cutoff in (np.inf, 1.0, 0.0, -1.0, float("nan"), 1e-30):
        result = wp.geometry.knn(points, samples, 3, backend=BACKEND, max_dist=cutoff, exclude_indices=excludes)
        # Compare native Warp as well as the independent ordinary-cutoff reference;
        # native specifically preserves coincidence for a positive underflowing radius.
        control = wp.geometry.knn(points, samples, 3, max_dist=cutoff, exclude_indices=excludes)
        for value, expected in zip(result[1:], control[1:], strict=True):
            np.testing.assert_array_equal(value.numpy(), expected.numpy())
    np.testing.assert_array_equal(result[3].numpy(), [[1, 0, 0, 0]] * 2)


def test_owner(test, device):
    points = wp.array(np.zeros((33, 5, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    queries = wp.array(np.ones((33, 3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    # No per-cloud host loop, tree build, or batched array views on either call.
    with (
        mock.patch.object(point_query, "_batch_view", side_effect=AssertionError("Per-cloud host work")),
        mock.patch.object(wp, "Bvh", side_effect=AssertionError("Tree construction")),
    ):
        owner = wp.geometry.PointIndex(points, backend=BACKEND)
        result = wp.geometry.knn(owner, queries, 1)
        test.assertIs(result[0], owner)
        points.fill_(wp.vec3(1.0))
        owner.refit()  # No-op: coordinates are read directly on every query.
        owner.rebuild()
        result = wp.geometry.knn(owner, queries, 1)
        np.testing.assert_array_equal(result[2].numpy(), np.zeros((33, 3, 1)))
    with test.assertRaises(ValueError):
        _ = owner.bvhs
    with test.assertRaises(ValueError):
        wp.geometry.knn(owner, queries, 1, backend=wp.geometry.PointQueryBackend.WARP)
    for options in ({"leaf_size": 4}, {"ordered": False}, {"query_stride": 1}):
        with test.assertRaises(ValueError):
            wp.geometry.knn(points, queries, 1, backend=BACKEND, **options)
    raw = point_query._build_point_index(points, backend=BACKEND)
    result = point_query._knn(points, queries, 1, bvh=raw)
    test.assertIs(result[0], raw)
    with test.assertRaises(TypeError):
        point_query._knn(points, queries, 1, bvh=raw, backend=wp.geometry.PointQueryBackend.WARP)
    with test.assertRaises(ValueError):
        point_query._knn(wp.clone(points), queries, 1, bvh=raw)


def test_gradients(test, device):
    with mock.patch.object(batched, "query_backends", return_value=[BACKEND]):
        batched.test_batched_chamfer_gradients(test, device)
        batched.test_coincident_euclidean_gradients(test, device)


def test_capture(test, device):
    rng = np.random.default_rng(912)
    points = wp.array(rng.normal(size=(3, 259, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    queries = wp.array(rng.normal(size=(3, 257, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    # One-shot owner/output creation is capturable, including with APIC.
    wp.geometry.knn(points, queries, 3, backend=BACKEND)
    for apic in (False, True):
        with wp.ScopedCapture(device=device, apic=apic, force_module_load=False) as capture:
            result = wp.geometry.knn(points, queries, 3, backend=BACKEND)
        points.fill_(wp.vec3(0.0))
        queries.fill_(wp.vec3(1.0))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(result[1].numpy(), np.broadcast_to([0, 1, 2], (3, 257, 3)))
        np.testing.assert_array_equal(result[2].numpy(), np.full((3, 257, 3), 3.0))
        np.testing.assert_array_equal(result[3].numpy(), np.full((3, 257), 3))


def test_capture_gradients(test, device):
    rng = np.random.default_rng(916)
    p = rng.normal(size=(2, 17, 3)).astype(np.float32)
    q = rng.normal(size=(2, 13, 3)).astype(np.float32)
    points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
    queries = wp.array(q, dtype=wp.vec3, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    first = wp.geometry.knn(points, queries, 1, backend=BACKEND)
    second = wp.geometry.knn(queries, points, 1, backend=BACKEND)

    def operation():
        for array in (points, queries, first[2], second[2]):
            array.grad.zero_()
        loss.zero_()
        with wp.Tape() as tape:
            for owner, samples, output in ((first[0], queries, first), (second[0], points, second)):
                wp.geometry.knn(owner, samples, 1, indices=output[1], distances=output[2], counts=output[3])
                wp.launch(
                    batched.chamfer_sum_kernel, samples.shape, [output[2], loss, 1.0 / samples.size], device=device
                )
        tape.backward(loss)

    operation()
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        operation()
    q += np.float32(0.25)
    queries.assign(q)
    expected_loss, expected_p, expected_q = batched.chamfer_reference(p, q)
    for _ in range(2):
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=2e-6)
        np.testing.assert_allclose(points.grad.numpy(), expected_p, atol=2e-6)
        np.testing.assert_allclose(queries.grad.numpy(), expected_q, atol=2e-6)


class TestPointQueryBruteForce(unittest.TestCase):
    pass


for name, function in (
    ("values", test_values),
    ("filtering", test_filtering),
    ("owner", test_owner),
    ("gradients", test_gradients),
):
    add_function_test(TestPointQueryBruteForce, "test_" + name, function, devices=get_test_devices())
add_function_test(TestPointQueryBruteForce, "test_capture", test_capture, devices=get_cuda_test_devices())

add_function_test(
    TestPointQueryBruteForce, "test_capture_gradients", test_capture_gradients, devices=get_cuda_test_devices()
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
