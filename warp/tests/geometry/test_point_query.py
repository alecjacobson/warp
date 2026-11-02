# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import unittest

import numpy as np

import warp as wp
from warp.examples.benchmarks.point_query_common import (
    make_ordered_query,
    make_query,
    make_tiled,
    make_tiled_batched,
    query_codes,
)
from warp.tests.unittest_utils import add_function_test, get_test_devices


@functools.cache
def query_kernel(k):
    @wp.kernel(module="unique")
    def query(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        radius: float,
        exclude: wp.array[int],
        indices: wp.array2d[int],
        distances: wp.array2d[float],
        counts: wp.array[int],
        nearest: wp.array[int],
    ):
        t = wp.tid()
        for r in range(wp.static(k)):
            indices[t, r] = -1
            distances[t, r] = wp.inf
        search = wp.point_query_knn(bvh, points, queries[t], radius, wp.static(k), exclude[t])
        i = int(-1)
        count = int(0)
        while wp.point_query_next(search, i):
            indices[t, count] = i
            distances[t, count] = wp.length_sq(queries[t] - points[i])
            count += 1
        counts[t] = count
        old = i
        wp.expect_eq(wp.point_query_next(search, i), False)
        wp.expect_eq(i, old)
        nearest[t] = wp.point_query_nearest(bvh, points, queries[t], radius, exclude[t])

    return query


def reference(points, queries, k, radius, exclude):
    indices = np.full((len(queries), k), -1, dtype=np.int32)
    distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        for t, q in enumerate(queries):
            if not radius > 0 or not np.isfinite(q).all():
                continue
            d = (q - points) ** 2
            d2 = (d[:, 0] + d[:, 1]) + d[:, 2]
            valid = np.flatnonzero((d2 < np.float32(radius) ** np.float32(2)) & (np.arange(len(points)) != exclude[t]))
            selected = valid[np.lexsort((valid, d2[valid]))][:k]
            indices[t, : len(selected)] = selected
            distances[t, : len(selected)] = d2[selected]
    return indices, distances


def check(test, device, p, q, k=8, radius=np.inf, exclude=None, constructor=None, leaf_size=4, bvh=None, points=None):
    p = np.asarray(p, dtype=np.float32).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float32).reshape(-1, 3)
    if exclude is None:
        exclude = np.full(len(q), -1, dtype=np.int32)
    if points is None:
        points = wp.array(p, dtype=wp.vec3, device=device)
    if bvh is None:
        bvh = wp.Bvh(points, points, constructor=constructor, leaf_size=leaf_size)
    queries = wp.array(q, dtype=wp.vec3, device=device)
    excluded = wp.array(exclude, dtype=int, device=device)
    indices = wp.empty((len(q), k), dtype=int, device=device)
    distances = wp.empty((len(q), k), dtype=float, device=device)
    counts = wp.empty(len(q), dtype=int, device=device)
    nearest = wp.empty(len(q), dtype=int, device=device)
    args = [bvh.id, points, queries, float(radius), excluded, indices, distances, counts, nearest]
    wp.launch(query_kernel(k), len(q), inputs=args, device=device)
    expected_i, expected_d = reference(p, q, k, radius, exclude)
    np.testing.assert_array_equal(indices.numpy(), expected_i)
    np.testing.assert_allclose(distances.numpy(), expected_d, rtol=2e-6)
    np.testing.assert_array_equal(counts.numpy(), (expected_i >= 0).sum(axis=1))
    np.testing.assert_array_equal(nearest.numpy(), expected_i[:, 0])
    return bvh, points, args


def test_clouds(test, device):
    rng = np.random.default_rng(741)
    for n in (0, 1, 3, 33, 513):
        p = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        q = rng.uniform(-2, 2, (35, 3)).astype(np.float32)
        for k in (1, 4, 8, 32, 64):
            check(test, device, p, q, k)
    p = rng.normal(size=(321, 3)).astype(np.float32)
    centers = rng.uniform(-1, 1, (8, 3)).astype(np.float32)
    clustered = centers[rng.integers(8, size=len(p))] + np.float32(0.001) * p
    outliers = clustered.copy()
    outliers[-3:] = [[100, 0, 0], [0, -100, 0], [0, 0, 100]]
    for geometry in (p, p * [1, 1, 0], p * [1, 0, 0], np.zeros_like(p), np.round(p), clustered, outliers):
        for constructor in ("sah", "median", "lbvh", "cubql"):
            if device.is_cpu and constructor == "lbvh":
                continue
            for leaf in (1, 4, 8):
                check(test, device, geometry, p[:23], constructor=constructor, leaf_size=leaf)


def test_boundaries(test, device):
    p = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]], np.float32)
    q = np.array([[0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0], [1e10, 0, 0]], np.float32)
    for radius in (
        0,
        -1,
        np.nan,
        np.inf,
        1,
        np.nextafter(np.float32(1), np.float32(0)),
        np.nextafter(np.float32(1), np.float32(2)),
    ):
        for k in (1, 3, 8):
            check(test, device, p, q, k, radius, exclude=np.array([4, -1, -1, -1], np.int32))
    check(test, device, p, p, 8, exclude=np.arange(len(p), dtype=np.int32))
    check(test, device, [[1.8e19, 0, 0]], [[0, 0, 0]], radius=2e19)
    # Cross-leaf ties must select original IDs even when the tree reorders primitives.
    p = np.tile(p[:4], (32, 1))
    check(test, device, p, [[0, 0, 0]], 8, leaf_size=1)


@wp.func
def nested(bvh: wp.uint64, points: wp.array[wp.vec3], q: wp.vec3):
    return wp.point_query_knn(bvh, points, q, wp.inf, 8)


@wp.kernel
def iterators(bvh: wp.uint64, points: wp.array[wp.vec3]):
    a = nested(bvh, points, wp.vec3(0.0))
    b = wp.point_query_knn(bvh, points, wp.vec3(0.0), wp.inf, 1)
    i = int(-1)
    j = int(-1)
    wp.expect_eq(wp.point_query_next(a, i), True)
    wp.expect_eq(wp.point_query_next(b, j), True)
    wp.expect_eq(i, j)
    saved = j
    wp.expect_eq(wp.point_query_next(b, j), False)
    wp.expect_eq(j, saved)
    wp.expect_eq(wp.point_query_next(a, i), True)
    wp.expect_neq(i, j)
    # Value copies have independent cursors; early exit is valid.
    c = a
    wp.expect_eq(wp.point_query_next(a, i), True)
    wp.expect_eq(wp.point_query_next(c, j), True)
    wp.expect_eq(i, j)


def test_iterators(test, device):
    points = wp.array([[i, 0, 0] for i in range(10)], dtype=wp.vec3, device=device)
    bvh = wp.Bvh(points, points)
    wp.launch(iterators, 1, inputs=[bvh.id, points], device=device)
    wp.synchronize_device(device)


@wp.kernel
def fused_loss(bvh: wp.uint64, points: wp.array[wp.vec3], queries: wp.array[wp.vec3], loss: wp.array[float]):
    t = wp.tid()
    search = wp.point_query_knn(bvh, points, queries[t], wp.inf, 3)
    i = int(-1)
    while wp.point_query_next(search, i):
        wp.atomic_add(loss, 0, wp.length_sq(queries[t] - points[i]))


@wp.kernel
def nearest_loss(bvh: wp.uint64, points: wp.array[wp.vec3], queries: wp.array[wp.vec3], loss: wp.array[float]):
    t = wp.tid()
    i = wp.point_query_nearest(bvh, points, queries[t], wp.inf)
    if i >= 0:
        wp.atomic_add(loss, 0, wp.length_sq(queries[t] - points[i]))


def test_gradients(test, device):
    p = np.array([[0, 0, 0], [2, 1, 0], [4, 0, 0], [10, 1, 0]], np.float32)
    q = np.array([[0.2, 0.1, 0], [0.3, -0.1, 0]], np.float32)
    for kernel, k in ((nearest_loss, 1), (fused_loss, 3)):
        points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
        queries = wp.array(q, dtype=wp.vec3, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        bvh = wp.Bvh(points, points)
        with wp.Tape() as tape:
            wp.launch(kernel, len(q), inputs=[bvh.id, points, queries, loss], device=device)
        tape.backward(loss)
        expected_q = np.zeros_like(q)
        expected_p = np.zeros_like(p)
        ids, _ = reference(p, q, k, np.inf, [-1, -1])
        for t in range(len(q)):
            for i in ids[t]:
                expected_q[t] += 2 * (q[t] - p[i])
                expected_p[i] -= 2 * (q[t] - p[i])
        np.testing.assert_allclose(queries.grad.numpy(), expected_q, atol=2e-6)
        np.testing.assert_allclose(points.grad.numpy(), expected_p, atol=2e-6)


def test_motion_capture(test, device):
    rng = np.random.default_rng(19)
    p = rng.normal(size=(129, 3)).astype(np.float32)
    q = rng.normal(size=(17, 3)).astype(np.float32)
    for constructor in ("sah", "median", "cubql") + (("lbvh",) if device.is_cuda else ()):
        bvh, points, args = check(test, device, p, q, constructor=constructor)
        moved = p + rng.normal(scale=0.2, size=p.shape).astype(np.float32)
        points.assign(moved)
        old_id = bvh.id
        bvh.refit()
        test.assertEqual(bvh.id, old_id)
        check(test, device, moved, q, bvh=bvh, points=points)
        bvh.rebuild()
        check(test, device, moved, q, bvh=bvh, points=points)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                bvh.refit()
                wp.launch(query_kernel(8), len(q), inputs=args, device=device)
            for _ in range(2):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(args[5].numpy(), reference(moved, q, 8, np.inf, [-1] * len(q))[0])


def test_baselines(test, device):

    rng = np.random.default_rng(910)
    for n in (0, 7, 257):
        p = rng.normal(size=(n, 3)).astype(np.float32)
        if n > 7:
            p[::3] = [1, 0, 0]
        q = np.concatenate((np.zeros((1, 3), np.float32), p[:3], [[np.nan, 0, 0]])).astype(np.float32)
        for k in (1, 8):
            bvh, points, args = check(test, device, p, q, k)
            args = args[:-1]
            keys = wp.empty(2 * len(q), dtype=int, device=device)
            order = wp.empty_like(keys)
            wp.launch(query_codes, len(q), inputs=[bvh.id, args[2], keys, order], device=device)
            wp.utils.radix_sort_pairs(keys, order, len(q))
            wp.launch(make_ordered_query(k), len(q), inputs=[*args[:5], order, *args[5:]], device=device)
            np.testing.assert_array_equal(args[5].numpy(), reference(p, q, k, np.inf, [-1] * len(q))[0])
            # Also validate every baseline with exclusion and a partial cutoff.
            if n > 7:
                args[3] = 0.8
                args[4].fill_(0)
            expected, _ = reference(p, q, k, args[3], args[4].numpy())
            for method in ("bvh", "bvh_sorted", "brute", "brute_sorted", "sphere"):
                # A one-slot stack forces complete restart on the larger tree.
                wp.launch(make_query(k, method, 1), len(q), inputs=args, device=device)
                np.testing.assert_array_equal(args[5].numpy(), expected)
            for bp, block in ((32, 32), (128, 64)):
                tiled, merge = make_tiled(k, bp)
                for partitions in (1, 3):
                    pi = wp.empty((len(q), partitions, k), dtype=int, device=device)
                    pd = wp.empty((len(q), partitions, k), dtype=float, device=device)
                    wp.launch_tiled(
                        tiled,
                        dim=(len(q), partitions),
                        inputs=[points, args[2], args[3], args[4], partitions, pi, pd],
                        block_dim=block,
                        device=device,
                    )
                    wp.launch(merge, len(q), inputs=[pi, pd, *args[5:]], device=device)
                    np.testing.assert_array_equal(args[5].numpy(), expected)


def test_k_diagnostics(test, device):
    for value in (0, -2):

        @wp.kernel(module="unique")
        def invalid(bvh: wp.uint64, points: wp.array[wp.vec3]):
            search = wp.point_query_knn(bvh, points, wp.vec3(0.0), wp.inf, wp.static(value))  # noqa: B023 - compiled before the next loop iteration

        with test.assertRaisesRegex(Exception, "compile-time positive integer"):
            wp.load_module(module=invalid.module, device=device)

    @wp.kernel(module="unique")
    def runtime_k(bvh: wp.uint64, points: wp.array[wp.vec3], k: int):
        search = wp.point_query_knn(bvh, points, wp.vec3(0.0), wp.inf, k)

    with test.assertRaisesRegex(Exception, "compile-time positive integer"):
        wp.load_module(module=runtime_k.module, device=device)


@wp.kernel
def stored_loss(points: wp.array[wp.vec3], queries: wp.array[wp.vec3], ids: wp.array2d[int], loss: wp.array[float]):
    t, r = wp.tid()
    i = ids[t, r]
    if i >= 0:
        wp.atomic_add(loss, 0, wp.length_sq(queries[t] - points[i]))


def test_stored_gradients(test, device):
    p = np.array([[0, 0, 0], [2, 1, 0]], np.float32)
    q = np.array([[0.2, 0.1, 0], [0.3, -0.1, 0]], np.float32)
    points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
    queries = wp.array(q, dtype=wp.vec3, device=device, requires_grad=True)
    bvh = wp.Bvh(points, points)
    _, _, args = check(test, device, p, q, 8, bvh=bvh, points=points)
    ids = args[5]
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(stored_loss, (len(q), 8), inputs=[points, queries, ids, loss], device=device)
    tape.backward(loss)
    gp = points.grad.numpy()
    gq = queries.grad.numpy()

    # Central differences use independently selected neighbors, away from switches.
    def value(pp, qq):
        ii, dd = reference(pp, qq, 8, np.inf, [-1] * len(qq))
        return dd[ii >= 0].sum(dtype=np.float64)

    eps = 1.0e-3
    for array, grad in ((p, gp), (q, gq)):
        numerical = np.empty_like(array)
        for row in range(len(array)):
            for axis in range(3):
                old = array[row, axis]
                array[row, axis] = old + eps
                plus = value(p, q)
                array[row, axis] = old - eps
                minus = value(p, q)
                array[row, axis] = old
                numerical[row, axis] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(grad, numerical, atol=7.0e-4, rtol=1.0e-3)


@wp.kernel
def empty_region(bvh: wp.uint64):
    region = wp.bvh_query_aabb(bvh, wp.vec3(-1.0), wp.vec3(1.0))
    i = int(123)
    wp.expect_eq(wp.bvh_query_next(region, i), False)
    wp.expect_eq(i, 123)


def test_empty_builders(test, device):
    for constructor in ("sah", "median", "cubql") + (("lbvh",) if device.is_cuda else ()):
        bvh, points, _ = check(test, device, [], [[0, 0, 0]], constructor=constructor)
        bvh.refit()
        bvh.rebuild()
        check(test, device, [], [[0, 0, 0]], bvh=bvh, points=points)
        wp.launch(empty_region, 1, inputs=[bvh.id], device=device)
    wp.synchronize_device(device)


def test_scales_readers(test, device):
    rng = np.random.default_rng(7)
    p = rng.normal(size=(73, 3)).astype(np.float32)
    q = rng.normal(size=(17, 3)).astype(np.float32)
    for scale in (1.0e-18, 1.0e-8, 1.0e8, 1.0e18):
        check(test, device, p * scale, q * scale, radius=float(2 * scale))
    if device.is_cuda:
        _bvh, _points, args = check(test, device, p, q)
        all_args = []
        streams = [wp.Stream(device), wp.Stream(device)]
        for stream in streams:
            a = args[:5] + [wp.empty_like(out) for out in args[5:]]
            all_args.append(a)
            with wp.ScopedStream(stream):
                wp.launch(query_kernel(8), len(q), inputs=a, device=device)
        for stream in streams:
            wp.synchronize_stream(stream)
        for a in all_args:
            np.testing.assert_array_equal(a[5].numpy(), reference(p, q, 8, np.inf, [-1] * len(q))[0])


def test_query_tiles(test, device):
    rng = np.random.default_rng(44)
    p = rng.normal(size=(259, 3)).astype(np.float32)
    q = rng.normal(size=(7, 3)).astype(np.float32)
    q[-1] = np.nan
    pw = wp.array(p, dtype=wp.vec3, device=device)
    qw = wp.array(q, dtype=wp.vec3, device=device)
    excludes = np.array([0, 1, -1, -1, 2, -1, -1], np.int32)
    ew = wp.array(excludes, dtype=int, device=device)
    for k in (1, 8):
        expected, distances = reference(p, q, k, 0.8, excludes)
        oi = wp.empty((len(q), k), dtype=int, device=device)
        od = wp.empty((len(q), k), dtype=float, device=device)
        counts = wp.empty(len(q), dtype=int, device=device)
        for bq in (2, 4):
            tiled, merge = make_tiled_batched(k, 128, bq)
            pi = wp.empty((len(q), 3, k), dtype=int, device=device)
            pd = wp.empty((len(q), 3, k), dtype=float, device=device)
            wp.launch_tiled(
                tiled,
                dim=((len(q) + bq - 1) // bq, 3),
                inputs=[pw, qw, 0.8, ew, 3, pi, pd],
                block_dim=64,
                device=device,
            )
            wp.launch(merge, len(q), inputs=[pi, pd, oi, od, counts], device=device)
            np.testing.assert_array_equal(oi.numpy(), expected)
            np.testing.assert_allclose(od.numpy(), distances, rtol=2.0e-6)


class TestPointQuery(unittest.TestCase):
    pass


for name, fn in (
    ("query_tiles", test_query_tiles),
    ("stored_gradients", test_stored_gradients),
    ("empty_builders", test_empty_builders),
    ("scales_readers", test_scales_readers),
    ("baselines", test_baselines),
    ("k_diagnostics", test_k_diagnostics),
    ("clouds", test_clouds),
    ("boundaries", test_boundaries),
    ("iterators", test_iterators),
    ("gradients", test_gradients),
    ("motion_capture", test_motion_capture),
):
    add_function_test(TestPointQuery, "test_" + name, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
