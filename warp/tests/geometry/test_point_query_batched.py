# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp._src import context, point_query
from warp.tests.geometry.test_point_query import reference
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def query_backends(device):
    backends = [wp.PointQueryBackend.WARP]
    if device.is_cuda and wp.is_cubql_available():
        backends.append(wp.PointQueryBackend.CUBQL)
    return backends


def check_batches(test, targets, queries, k, result):
    indices, distances, counts = (value.numpy() for value in result[1:])
    test.assertEqual(indices.shape, (*queries.shape[:2], k))
    test.assertEqual(counts.shape, queries.shape[:2])
    for batch in range(len(targets)):
        expected_indices, expected_distances = reference(
            targets[batch], queries[batch], k, np.inf, np.full(len(queries[batch]), -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(indices[batch], expected_indices)
        np.testing.assert_allclose(distances[batch], expected_distances, rtol=3e-6, atol=1e-7)
        np.testing.assert_array_equal(counts[batch], np.minimum(k, len(targets[batch])))


def test_prebuilt_indexes(test, device):
    """Build targets without queries and reuse indexes with different query counts and k."""
    rng = np.random.default_rng(725)
    for backend in query_backends(device):
        for shape in ((37,), (2, 37), (2, 0), (0, 37)):
            targets = rng.normal(size=(*shape, 3)).astype(np.float32)
            points = wp.array(targets, dtype=wp.vec3, device=device)
            # Construction must not hide a query or allocate neighbor outputs.
            with (
                mock.patch.object(point_query, "knn", side_effect=AssertionError("Unexpected query")),
                mock.patch.object(point_query, "_query_kernel", side_effect=AssertionError("Unexpected query kernel")),
                mock.patch.object(context.runtime.core, "wp_point_query_cubql_query_device") as native_query,
            ):
                indexes = wp.build_point_index(points, k=1, backend=backend)
                native_query.assert_not_called()
            index_type = wp.CubqlPointIndex if backend == wp.PointQueryBackend.CUBQL else wp.Bvh
            test.assertIsInstance(indexes, tuple if len(shape) == 2 else index_type)
            for query_count, k in ((7, 5), (13, 1)):
                queries = rng.normal(size=(*shape[:-1], query_count, 3)).astype(np.float32)
                query_points = wp.array(queries, dtype=wp.vec3, device=device)
                with mock.patch.object(index_type, "__init__", side_effect=AssertionError("Unexpected construction")):
                    result = wp.knn(points, query_points, k, bvh=indexes)
                test.assertIs(result[0], indexes)
                if len(shape) == 2:
                    check_batches(test, targets, queries, k, result)
                else:
                    check_batches(
                        test,
                        targets[None],
                        queries[None],
                        k,
                        (None, *(value.reshape((1, *value.shape)) for value in result[1:])),
                    )


def test_batched_values(test, device):
    """Keep batches isolated and preserve single-cloud, empty, and partial results."""
    rng = np.random.default_rng(981)
    for backend in query_backends(device):
        for batches, target_count, query_count, k in (
            (2, 37, 13, 5),
            (1, 4, 3, 11),
            (2, 0, 7, 3),
            (2, 5, 0, 1),
            (0, 7, 3, 1),
        ):
            targets = rng.normal(size=(batches, target_count, 3)).astype(np.float32)
            queries = rng.normal(size=(batches, query_count, 3)).astype(np.float32)
            if batches == 2 and target_count:
                # Spatially overlapping clouds with different local index order.
                targets[1] = targets[0, ::-1]
            points = wp.array(targets, dtype=wp.vec3, device=device)
            query_points = wp.array(queries, dtype=wp.vec3, device=device)
            ids = wp.empty((batches, query_count, k), dtype=wp.int64, device=device)
            result = wp.knn(points, query_points, k, backend=backend, indices=ids)
            test.assertIs(result[1], ids)
            test.assertEqual(len(result[0]), batches)
            check_batches(test, targets, queries, k, result)
            old_handles = tuple(index.id for index in result[0])
            queries = queries + np.float32(0.13)
            query_points.assign(queries)
            reused = wp.knn(points, query_points, k, bvh=result[0], indices=ids, distances=result[2], counts=result[3])
            test.assertIs(reused[0], result[0])
            test.assertEqual(tuple(index.id for index in reused[0]), old_handles)
            check_batches(test, targets, queries, k, reused)
            if batches == 1 and target_count and query_count:
                single = wp.knn(points[0], query_points[0], k, backend=backend)
                np.testing.assert_array_equal(single[1].numpy(), reused[1].numpy()[0])


@wp.kernel
def chamfer_sum_kernel(distances: wp.array3d[float], loss: wp.array[float], weight: float):
    batch, query = wp.tid()
    wp.atomic_add(loss, 0, distances[batch, query, 0] * weight)


def chamfer_reference(targets, queries, squared=True):
    total = 0.0
    target_grad = np.zeros_like(targets)
    query_grad = np.zeros_like(queries)
    for batch in range(len(targets)):
        for references, samples, reference_grad, sample_grad in (
            (targets, queries, target_grad, query_grad),
            (queries, targets, query_grad, target_grad),
        ):
            ids, distances = reference(references[batch], samples[batch], 1, np.inf, [-1] * samples.shape[1])
            weight = 1.0 / (len(targets) * samples.shape[1])
            total += np.sum(distances if squared else np.sqrt(distances)) * weight
            for row, index in enumerate(ids[:, 0]):
                delta = 2.0 * (samples[batch, row] - references[batch, index]) * weight
                if not squared:
                    delta /= 2 * np.sqrt(distances[row, 0])
                sample_grad[batch, row] += delta
                reference_grad[batch, index] -= delta
    return total, target_grad, query_grad


def test_batched_chamfer_gradients(test, device):
    """Differentiate both Chamfer directions into the original batched arrays."""
    rng = np.random.default_rng(1729)
    targets = rng.normal(size=(2, 9, 3)).astype(np.float32)
    queries = rng.normal(size=(2, 5, 3)).astype(np.float32)
    for squared in (True, False):
        expected_loss, expected_targets, expected_queries = chamfer_reference(targets, queries, squared)
        shifted = queries.copy()
        epsilon = np.float32(1e-3)
        shifted[1, 2, 0] += epsilon
        plus = chamfer_reference(targets, shifted, squared)[0]
        shifted[1, 2, 0] -= 2 * epsilon
        minus = chamfer_reference(targets, shifted, squared)[0]
        for backend in query_backends(device):
            points = wp.array(targets, dtype=wp.vec3, device=device, requires_grad=True)
            query_points = wp.array(queries, dtype=wp.vec3, device=device, requires_grad=True)
            fixed = wp.build_point_index(points, num_queries=query_points.shape[1], k=1, backend=backend)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            with wp.Tape() as tape:
                forward = wp.knn(points, query_points, 1, bvh=fixed, squared=squared)
                backward = wp.knn(query_points, points, 1, backend=backend, squared=squared)
                wp.launch(chamfer_sum_kernel, queries.shape[:2], [forward[2], loss, 1.0 / 10], device=device)
                wp.launch(chamfer_sum_kernel, targets.shape[:2], [backward[2], loss, 1.0 / 18], device=device)
            tape.backward(loss)
            np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=2e-6)
            np.testing.assert_allclose(points.grad.numpy(), expected_targets, atol=2e-6)
            np.testing.assert_allclose(query_points.grad.numpy(), expected_queries, atol=2e-6)
            np.testing.assert_allclose(query_points.grad.numpy()[1, 2, 0], (plus - minus) / (2 * epsilon), atol=2e-4)


def test_batched_filtering(test, device):
    """Apply cutoffs and exclusions within each ordered Warp query batch."""
    rng = np.random.default_rng(172)
    targets = np.round(rng.normal(size=(2, 23, 3))).astype(np.float32)
    queries = targets[:, :7].copy()
    excludes = np.tile(np.arange(7, dtype=np.int32), (2, 1))
    result = wp.knn(
        wp.array(targets, dtype=wp.vec3, device=device),
        wp.array(queries, dtype=wp.vec3, device=device),
        4,
        exclude_indices=wp.array(excludes, device=device),
        max_dist=1.1,
        ordered=True,
        query_stride=4,
    )
    for batch in range(2):
        ids, distances = reference(targets[batch], queries[batch], 4, 1.1, excludes[batch])
        np.testing.assert_array_equal(result[1].numpy()[batch], ids)
        np.testing.assert_allclose(result[2].numpy()[batch], distances, atol=1e-7)


def test_batched_capture(test, device):
    """Replay independent query batches after modifying queries and indexed targets."""
    rng = np.random.default_rng(746)
    targets = rng.normal(size=(2, 129, 3)).astype(np.float32)
    queries = rng.normal(size=(2, 17, 3)).astype(np.float32)
    points = wp.array(targets, dtype=wp.vec3, device=device)
    query_points = wp.array(queries, dtype=wp.vec3, device=device)
    for backend in query_backends(device):
        result = wp.knn(points, query_points, 11, backend=backend)
        with wp.ScopedCapture(device=device) as capture:
            replayed = wp.knn(
                points, query_points, 11, bvh=result[0], indices=result[1], distances=result[2], counts=result[3]
            )
        for _ in range(2):
            queries += np.float32(0.17)
            query_points.assign(queries)
            wp.capture_launch(capture.graph)
            check_batches(test, targets, queries, 11, replayed)
        if backend == wp.PointQueryBackend.CUBQL:
            with wp.ScopedCapture(device=device):
                with test.assertRaisesRegex(ValueError, "cannot be captured"):
                    wp.knn(points, query_points, 1, backend=backend)
                with test.assertRaisesRegex(ValueError, "cannot be captured"):
                    wp.build_point_index(points, backend=backend)
        else:
            # Compile the ordered path before capturing full construction.
            warm = wp.knn(points, query_points, 11, ordered=True)
            with wp.ScopedCapture(device=device) as build_capture:
                rebuilt_indexes = wp.build_point_index(points, num_queries=query_points.shape[1], k=11)
                rebuilt = wp.knn(points, query_points, 11, bvh=rebuilt_indexes, ordered=True)
            targets *= np.float32(1.3)
            points.assign(targets)
            wp.capture_launch(build_capture.graph)
            check_batches(test, targets, queries, 11, rebuilt)
            with wp.ScopedCapture(device=device) as refit_capture:
                for index in warm[0]:
                    index.refit()
                refitted = wp.knn(points, query_points, 11, bvh=warm[0], ordered=True)
            targets += np.float32(0.21)
            points.assign(targets)
            wp.capture_launch(refit_capture.graph)
            check_batches(test, targets, queries, 11, refitted)


def test_cubql_route(test, device):
    """Exercise native cuBQL dispatch, capacity specializations, rebuilds, and validation."""
    if not wp.is_cubql_available():
        test.skipTest("cuBQL is unavailable")
    rng = np.random.default_rng(885)
    targets = rng.normal(size=(77, 3)).astype(np.float32)
    queries = rng.normal(size=(8, 3)).astype(np.float32)
    points = wp.array(targets, dtype=wp.vec3, device=device)
    query_points = wp.array(queries, dtype=wp.vec3, device=device)
    for objective in wp.PointQueryObjective:
        for k in (1, 5, 11, 32, 64):
            with mock.patch.object(
                context.runtime.core,
                "wp_point_query_cubql_query_device",
                wraps=context.runtime.core.wp_point_query_cubql_query_device,
            ) as native_query:
                result = wp.knn(points, query_points, k, backend=wp.PointQueryBackend.CUBQL, objective=objective)
                native_query.assert_called_once()
            test.assertIsInstance(result[0], wp.CubqlPointIndex)
            ids, distances = reference(targets, queries, k, np.inf, [-1] * len(queries))
            np.testing.assert_array_equal(result[1].numpy(), ids)
            np.testing.assert_allclose(result[2].numpy(), distances, rtol=3e-6)
            result[0].rebuild()
    previous = result[0].id
    with mock.patch.object(context.runtime.core, "wp_point_query_cubql_create_device", return_value=0):
        with test.assertRaisesRegex(RuntimeError, "Failed to construct"):
            result[0].rebuild()
    test.assertEqual(result[0].id, previous)
    test.assertFalse(
        context.runtime.core.wp_point_query_cubql_query_device(
            result[0].id,
            None,
            query_points.ptr,
            len(queries),
            1,
            32,
            32,
            result[1].ptr,
            result[2].ptr,
            result[3].ptr,
            False,
            True,
        )
    )
    test.assertIn("target coordinates", context.runtime.get_error_string())
    targets += np.float32(0.4)
    points.assign(targets)
    result[0].rebuild()
    rebuilt = wp.knn(points, query_points, 1, bvh=result[0], squared=False)
    ids, distances = reference(targets, queries, 1, np.inf, [-1] * len(queries))
    np.testing.assert_array_equal(rebuilt[1].numpy(), ids)
    np.testing.assert_allclose(rebuilt[2].numpy(), np.sqrt(distances), rtol=3e-6)
    for kwargs in (
        {"k": 65},
        {"max_dist": 1.0},
        {"ordered": True},
        {"leaf_size": 4},
        {"exclude_indices": wp.full(len(queries), -1, dtype=int, device=device)},
    ):
        with test.assertRaises(ValueError):
            wp.knn(points, query_points, backend=wp.PointQueryBackend.CUBQL, **({"k": 1} | kwargs))
    replacement = wp.clone(points)
    with test.assertRaisesRegex(ValueError, "original target allocation"):
        wp.knn(replacement, query_points, 1, bvh=result[0])

    # cuBQL can select any equal-distance indices, but must return distinct ones.
    duplicates = wp.zeros(97, dtype=wp.vec3, device=device)
    nonfinite_queries = wp.array([[1, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]], dtype=wp.vec3, device=device)
    stream = wp.Stream(device)
    stream.wait_stream(device.stream)
    with wp.ScopedStream(stream):
        tied = wp.knn(duplicates, nonfinite_queries, 11, backend=wp.PointQueryBackend.CUBQL)
        np.testing.assert_array_equal(tied[3].numpy(), [11, 0, 0])
        test.assertEqual(len(np.unique(tied[1].numpy()[0])), 11)
        np.testing.assert_array_equal(tied[2].numpy()[0], 1.0)
        np.testing.assert_array_equal(tied[1].numpy()[1:], -1)
        np.testing.assert_array_equal(tied[2].numpy()[1:], np.inf)


class TestPointQueryBatched(unittest.TestCase):
    def test_legacy_alias(self):
        """Preserve the old signature, outputs, and index reuse while reporting deprecation."""
        points = wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3, device="cpu")
        index = wp.build_point_index(points)
        test_result = wp.knn(points, points, 1, bvh=index)
        with mock.patch.object(point_query, "log_warning") as warning:
            old_result = wp.point_query_knn_batch(
                points, points, 1, bvh=index, indices=test_result[1], distances=test_result[2], counts=test_result[3]
            )
        warning.assert_called_once()
        self.assertEqual(warning.call_args.kwargs["category"], DeprecationWarning)
        self.assertIn("wp.knn()", warning.call_args.args[0])
        self.assertEqual(inspect.signature(wp.knn), inspect.signature(wp.point_query_knn_batch))
        for old, new in zip(old_result, test_result, strict=True):
            self.assertIs(old, new)

    def test_build_validation(self):
        """Reject invalid construction hints even for zero batches."""
        points = wp.zeros((0, 5), dtype=wp.vec3, device="cpu")
        for kwargs in ({"num_queries": -1}, {"num_queries": True}, {"k": 0}, {"leaf_size": 0}, {"leaf_size": True}):
            with self.assertRaises(ValueError):
                wp.build_point_index(points, **kwargs)
        for kwargs in ({"backend": "warp"}, {"objective": "reuse"}):
            with self.assertRaises(TypeError):
                wp.build_point_index(points, **kwargs)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wp.build_point_index(points, backend=wp.PointQueryBackend.CUBQL)

    def test_batched_validation(self):
        """Reject mismatched batches, index sets, and aliased output storage."""
        points = wp.zeros((2, 5), dtype=wp.vec3, device="cpu")
        queries = wp.zeros((2, 3), dtype=wp.vec3, device="cpu")
        for shape in ((3, 3), (3,), (2, 3, 1)):
            with self.assertRaises(ValueError):
                wp.knn(points, wp.zeros(shape, dtype=wp.vec3, device="cpu"), 1)
        with self.assertRaises(ValueError):
            wp.knn(points, queries, 1, bvh=())
        with self.assertRaises(TypeError):
            wp.knn(points, queries, 1, backend="cubql")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            wp.knn(points, queries, 1, backend=wp.PointQueryBackend.CUBQL)
        output = wp.empty((2, 3), dtype=int, device="cpu")
        with self.assertRaisesRegex(ValueError, "overlap"):
            wp.knn(points, queries, 1, indices=output.reshape((2, 3, 1)), counts=output)


add_function_test(TestPointQueryBatched, "test_batched_values", test_batched_values, devices=get_test_devices())
add_function_test(TestPointQueryBatched, "test_prebuilt_indexes", test_prebuilt_indexes, devices=get_test_devices())
add_function_test(TestPointQueryBatched, "test_batched_filtering", test_batched_filtering, devices=get_test_devices())
add_function_test(
    TestPointQueryBatched, "test_batched_chamfer_gradients", test_batched_chamfer_gradients, devices=get_test_devices()
)
add_function_test(TestPointQueryBatched, "test_batched_capture", test_batched_capture, devices=get_cuda_test_devices())
add_function_test(TestPointQueryBatched, "test_cubql_route", test_cubql_route, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
