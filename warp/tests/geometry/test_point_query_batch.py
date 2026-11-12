# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.geometry.test_point_query import reference
from warp.tests.unittest_utils import add_function_test, get_test_devices


def check(test, p, q, k, outputs, radius=np.inf, exclude=None, squared=True):
    if exclude is None:
        exclude = np.full(len(q), -1, np.int32)
    _, indices, distances, counts = outputs
    expected_i, expected_d = reference(p, q, k, radius, exclude)
    np.testing.assert_array_equal(indices.numpy(), expected_i)
    np.testing.assert_allclose(distances.numpy(), expected_d if squared else np.sqrt(expected_d), rtol=2e-6)
    np.testing.assert_array_equal(counts.numpy(), (expected_i >= 0).sum(axis=1))


def test_batch(test, device):
    rng = np.random.default_rng(983)
    for n, nq, k in ((37, 13, 1), (37, 13, 5), (3, 11, 5), (0, 7, 5), (9, 0, 5)):
        p = np.round(rng.normal(size=(n, 3))).astype(np.float32)
        q = rng.normal(size=(nq, 3)).astype(np.float32)
        points, queries = wp.array(p, dtype=wp.vec3, device=device), wp.array(q, dtype=wp.vec3, device=device)
        result = wp.point_query_knn_batch(points, queries, k)
        check(test, p, q, k, result)
        test.assertIs(result[0].lowers, points)
        test.assertIs(result[0].uppers, points)


def test_order_cutoff_reuse(test, device):
    rng = np.random.default_rng(827)
    p = rng.normal(size=(129, 3)).astype(np.float32)
    p[:, 2] = 0
    q = p[:67].copy()
    q[0] = np.nan
    q[1] = np.inf
    exclude = np.arange(len(q), dtype=np.int32)
    points, queries = wp.array(p, dtype=wp.vec3, device=device), wp.array(q, dtype=wp.vec3, device=device)
    excluded = wp.array(exclude, device=device)
    for stride in (1, 4, 32):
        for squared in (True, False):
            ids = wp.empty((len(q), 5), dtype=wp.int64, device=device)
            ds = wp.empty((len(q), 5), dtype=float, device=device)
            counts = wp.empty(len(q), dtype=int, device=device)
            result = wp.point_query_knn_batch(
                points,
                queries,
                5,
                max_dist=0.7,
                exclude_indices=excluded,
                indices=ids,
                distances=ds,
                counts=counts,
                squared=squared,
                ordered=True,
                query_stride=stride,
            )
            test.assertIs(result[1], ids)
            test.assertIs(result[2], ds)
            test.assertIs(result[3], counts)
            check(test, p, q, 5, result, 0.7, exclude, squared)
            moved = p * np.float32(1.3)
            points.assign(moved)
            result[0].refit()
            again = wp.point_query_knn_batch(
                points,
                queries,
                5,
                bvh=result[0],
                max_dist=0.7,
                exclude_indices=excluded,
                indices=ids,
                distances=ds,
                counts=counts,
                squared=squared,
                ordered=True,
                query_stride=stride,
            )
            test.assertIs(again[0], result[0])
            check(test, moved, q, 5, again, 0.7, exclude, squared)
            points.assign(p)


@wp.kernel
def sum_distances(distances: wp.array2d[float], loss: wp.array[float]):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, distances[i, j])


def test_gradients(test, device):
    p = np.array([[0, 0, 0], [2, 1, 0], [4, 0, 0], [10, 1, 0]], np.float32)
    q = np.array([[0.2, 0.1, 0], [0.3, -0.1, 0]], np.float32)
    for ordered in (False, True):
        points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
        queries = wp.array(q, dtype=wp.vec3, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        with wp.Tape() as tape:
            result = wp.point_query_knn_batch(points, queries, 2, ordered=ordered)
            wp.launch(sum_distances, (len(q), 2), inputs=[result[2], loss], device=device)
        tape.backward(loss)
        expected_q, expected_p = np.zeros_like(q), np.zeros_like(p)
        ids, _ = reference(p, q, 2, np.inf, [-1, -1])
        for t, row in enumerate(ids):
            for i in row:
                expected_q[t] += 2 * (q[t] - p[i])
                expected_p[i] -= 2 * (q[t] - p[i])
        np.testing.assert_allclose(queries.grad.numpy(), expected_q, atol=2e-6)
        np.testing.assert_allclose(points.grad.numpy(), expected_p, atol=2e-6)


def test_capture(test, device):
    rng = np.random.default_rng(118)
    p, q = rng.normal(size=(513, 3)).astype(np.float32), rng.normal(size=(65, 3)).astype(np.float32)
    points, queries = wp.array(p, dtype=wp.vec3, device=device), wp.array(q, dtype=wp.vec3, device=device)
    stream = wp.Stream(device)
    with wp.ScopedStream(stream):
        for ordered in (False, True):
            warm = wp.point_query_knn_batch(points, queries, 5, ordered=ordered)
            with wp.ScopedCapture(stream=stream) as capture:
                result = wp.point_query_knn_batch(points, queries, 5, ordered=ordered)
            for _ in range(3):
                p = p * np.float32(1.1)
                q = q + np.float32(0.013)
                points.assign(p)
                queries.assign(q)
                wp.capture_launch(capture.graph, stream=stream)
                check(test, p, q, 5, result)
            # Capture refit and a query writing into caller-owned output buffers.
            with wp.ScopedCapture(stream=stream) as reuse:
                result[0].refit()
                again = wp.point_query_knn_batch(
                    points,
                    queries,
                    5,
                    bvh=result[0],
                    indices=warm[1],
                    distances=warm[2],
                    counts=warm[3],
                    ordered=ordered,
                )
            p = p + np.float32(0.1)
            points.assign(p)
            wp.capture_launch(reuse.graph, stream=stream)
            check(test, p, q, 5, again)


class TestPointQueryBatch(unittest.TestCase):
    def test_validation(self):
        points = wp.zeros(3, dtype=wp.vec3, device="cpu")
        bvh = wp.Bvh(points, points)
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                wp.point_query_knn_batch(points, points, k)
        for kwargs in (
            {"query_stride": 3},
            {"query_stride": 0},
            {"query_stride": 64},
            {"squared": "false"},
            {"ordered": 1},
            {"indices": wp.empty((3, 2), dtype=wp.float32, device="cpu")},
            {"exclude_indices": wp.zeros(2, dtype=int, device="cpu")},
            {"bvh": bvh, "leaf_size": 4},
        ):
            with self.assertRaises(ValueError):
                wp.point_query_knn_batch(points, points, 2, **kwargs)
        with self.assertRaises(ValueError):
            wp.point_query_knn_batch(points, wp.zeros((2, 3), dtype=wp.vec3, device="cpu"), 2)


devices = get_test_devices()
for name, fn in (("batch", test_batch), ("order_cutoff_reuse", test_order_cutoff_reuse), ("gradients", test_gradients)):
    add_function_test(TestPointQueryBatch, "test_" + name, fn, devices=devices)
add_function_test(TestPointQueryBatch, "test_capture", test_capture, devices=[d for d in devices if d.is_cuda])


if __name__ == "__main__":
    unittest.main(verbosity=2)
