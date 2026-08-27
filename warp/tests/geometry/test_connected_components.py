# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# Independent CPU reference (serial union-find over simplices, no Warp)
# ---------------------------------------------------------------------------


def _reference(simplices, num_points, simplex_size=3):
    """Return ``(labels, num_components)`` from a serial union-find over simplices.

    Every vertex of a simplex is unioned with its first, which for connected
    components induces the same partition as any spanning set of intra-simplex
    edges.
    """
    flat = np.asarray(simplices, dtype=np.int64).reshape(-1)
    parent = list(range(num_points))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for s in range(len(flat) // simplex_size):
        base = s * simplex_size
        first = int(flat[base])
        for c in range(1, simplex_size):
            union(first, int(flat[base + c]))

    roots = sorted({find(v) for v in range(num_points)})
    remap = {r: c for c, r in enumerate(roots)}
    labels = np.array([remap[find(v)] for v in range(num_points)], dtype=np.int32)
    return labels, len(roots)


def _partition(labels):
    """Set of vertex groups induced by a label array (order-independent)."""
    groups = {}
    for v, lbl in enumerate(labels):
        groups.setdefault(int(lbl), set()).add(v)
    return {frozenset(s) for s in groups.values()}


def _compute(simplices, num_points, device, simplex_size=3):
    arr = np.asarray(simplices, dtype=np.int32).reshape(-1, simplex_size)
    indices = wp.array(arr, dtype=wp.int32, device=device)
    labels, k = warp.geometry.connected_components(indices, num_points=num_points, device=device)
    return labels.numpy(), k


def _assert_matches_reference(test, simplices, num_points, device, simplex_size=3):
    labels, k = _compute(simplices, num_points, device, simplex_size=simplex_size)
    ref_labels, ref_k = _reference(simplices, num_points, simplex_size=simplex_size)

    test.assertEqual(k, ref_k, "component count mismatch")
    # Labels must be a contiguous [0, k) range...
    if num_points > 0:
        test.assertEqual(set(labels.tolist()), set(range(k)))
    # ...and induce exactly the reference partition (label *values* may differ).
    test.assertEqual(_partition(labels), _partition(ref_labels), "partition mismatch")
    return labels, k


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _grid_mesh(nx, ny):
    """Triangulated ``nx x ny`` grid (one connected component)."""

    def vid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            bl, br, tr, tl = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append((bl, br, tr))
            tris.append((bl, tr, tl))
    return tris, (nx + 1) * (ny + 1)


def _strip(num_triangles):
    """A long triangle strip: a single component with large graph diameter."""
    return [(i, i + 1, i + 2) for i in range(num_triangles)], num_triangles + 2


def _random_soup(rng, num_points, num_triangles):
    tris = []
    for _ in range(num_triangles):
        t = rng.integers(0, num_points, size=3)
        tris.append((int(t[0]), int(t[1]), int(t[2])))
    return tris


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_triangle(test, device):
    _, k = _assert_matches_reference(test, [(0, 1, 2)], 3, device)
    test.assertEqual(k, 1)


def test_two_disjoint_triangles(test, device):
    _, k = _assert_matches_reference(test, [(0, 1, 2), (3, 4, 5)], 6, device)
    test.assertEqual(k, 2)


def test_bowtie_single_component(test, device):
    # Two triangles sharing only vertex 0 are still one edge-connected component.
    _, k = _assert_matches_reference(test, [(0, 1, 2), (0, 3, 4)], 5, device)
    test.assertEqual(k, 1)


def test_nonmanifold_edge_single_component(test, device):
    _, k = _assert_matches_reference(test, [(0, 1, 2), (0, 1, 3), (0, 1, 4)], 5, device)
    test.assertEqual(k, 1)


def test_long_strip(test, device):
    # Exercises convergence over a large-diameter graph.
    tris, num_points = _strip(200)
    _, k = _assert_matches_reference(test, tris, num_points, device)
    test.assertEqual(k, 1)


def test_isolated_vertices(test, device):
    _, k = _assert_matches_reference(test, [(0, 1, 2)], 6, device)
    test.assertEqual(k, 4)  # one triangle component + three isolated vertices


def test_degenerate_triangles(test, device):
    # (0,0,1) connects 0-1 via its non-degenerate edge; (2,2,2) leaves 2 isolated.
    _, k = _assert_matches_reference(test, [(0, 0, 1), (2, 2, 2)], 3, device)
    test.assertEqual(k, 2)


def test_grid_single_component(test, device):
    tris, num_points = _grid_mesh(8, 6)
    _, k = _assert_matches_reference(test, tris, num_points, device)
    test.assertEqual(k, 1)


def test_multiple_grid_components(test, device):
    tris_a, na = _grid_mesh(4, 4)
    tris_b, nb = _grid_mesh(3, 5)
    tris = list(tris_a) + [(a + na, b + na, c + na) for (a, b, c) in tris_b]
    _, k = _assert_matches_reference(test, tris, na + nb, device)
    test.assertEqual(k, 2)


def test_empty_mesh(test, device):
    for num_points in (0, 5):
        _, k = _assert_matches_reference(test, np.zeros((0, 3), dtype=np.int32), num_points, device)
        test.assertEqual(k, num_points)  # every vertex is its own component


def test_permutation_and_relabel_invariance(test, device):
    rng = np.random.default_rng(3)
    tris = [(0, 1, 2), (2, 3, 4), (5, 6, 7), (0, 5, 8)]
    num_points = 9
    base_labels, base_k = _compute(tris, num_points, device)

    shuffled = [tris[i] for i in rng.permutation(len(tris))]
    s_labels, s_k = _compute(shuffled, num_points, device)
    test.assertEqual(s_k, base_k)
    test.assertEqual(_partition(s_labels), _partition(base_labels))

    relabel = rng.permutation(num_points)
    relabeled = [tuple(int(relabel[v]) for v in t) for t in tris]
    _, r_k = _compute(relabeled, num_points, device)
    test.assertEqual(r_k, base_k)


def test_infer_num_points(test, device):
    tris = [(0, 1, 2), (2, 3, 4)]
    arr = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    indices = wp.array(arr, dtype=wp.int32, device=device)
    labels, k = warp.geometry.connected_components(indices, device=device)
    ref_labels, ref_k = _reference(tris, 5)
    test.assertEqual(k, ref_k)
    test.assertEqual(_partition(labels.numpy()), _partition(ref_labels))


def test_random_soups(test, device):
    rng = np.random.default_rng(20260818)
    for _ in range(80):
        num_points = int(rng.integers(2, 25))
        num_triangles = int(rng.integers(0, 40))
        tris = _random_soup(rng, num_points, num_triangles)
        _assert_matches_reference(test, tris, num_points, device)


def test_segments(test, device):
    # simplex_size=2: a path 0-1-2, an isolated edge 5-6, isolated vertices 3,4.
    _, k = _assert_matches_reference(test, [0, 1, 1, 2, 5, 6], 7, device, simplex_size=2)
    test.assertEqual(k, 4)


def test_tetrahedra(test, device):
    # Two disjoint tets, then two tets sharing a vertex.
    _, k = _assert_matches_reference(test, [0, 1, 2, 3, 4, 5, 6, 7], 8, device, simplex_size=4)
    test.assertEqual(k, 2)
    _, k = _assert_matches_reference(test, [0, 1, 2, 3, 3, 4, 5, 6], 7, device, simplex_size=4)
    test.assertEqual(k, 1)


def test_points(test, device):
    # simplex_size=1: no edges, every vertex is its own component.
    _, k = _assert_matches_reference(test, [0, 1, 2, 3], 4, device, simplex_size=1)
    test.assertEqual(k, 4)


def test_random_soups_multi_size(test, device):
    rng = np.random.default_rng(2026_08_26)
    for simplex_size in (1, 2, 3, 4, 5):
        for _ in range(25):
            num_points = int(rng.integers(2, 25))
            num_simplices = int(rng.integers(0, 40))
            flat = rng.integers(0, num_points, size=num_simplices * simplex_size).astype(np.int32)
            _assert_matches_reference(test, flat, num_points, device, simplex_size=simplex_size)


def test_graph_capture(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")

    tris = [(0, 1, 2), (2, 3, 4), (5, 6, 7), (8, 9, 10)]
    num_points = 11
    ref_labels, ref_k = _reference(tris, num_points)
    idx = wp.array(np.asarray(tris, dtype=np.int32).reshape(-1, 3), dtype=wp.int32, device=device)

    # Eager mode returns a Python int; a warm-up also sizes scratch before capture.
    _, warmup_k = warp.geometry.connected_components(idx, num_points=num_points, device=device)
    test.assertIsInstance(warmup_k, int)
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as cap:
        labels, count = warp.geometry.connected_components(idx, num_points=num_points, device=device)

    # During capture the count is a device array, resolved only after replay.
    test.assertTrue(hasattr(count, "numpy"))
    wp.capture_launch(cap.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(count.numpy()[0]), ref_k)
    test.assertEqual(_partition(labels.numpy()), _partition(ref_labels))

    # The graph is reusable: a second replay reproduces the result.
    wp.capture_launch(cap.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(count.numpy()[0]), ref_k)


def test_invalid_inputs(test, device):
    # Not 2-D.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.connected_components(bad)
    # Index out of range.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([[0, 1, 5]], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.connected_components(bad, num_points=3)
    # Negative index.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([[0, 1, -1]], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.connected_components(bad, num_points=3)
    # Wrong dtype.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([[0, 1, 2]], dtype=np.int64), dtype=wp.int64, device=device)
        warp.geometry.connected_components(bad, num_points=3)


devices = get_test_devices()


class TestConnectedComponents(unittest.TestCase):
    pass


add_function_test(TestConnectedComponents, "test_single_triangle", test_single_triangle, devices=devices)
add_function_test(TestConnectedComponents, "test_two_disjoint_triangles", test_two_disjoint_triangles, devices=devices)
add_function_test(
    TestConnectedComponents, "test_bowtie_single_component", test_bowtie_single_component, devices=devices
)
add_function_test(
    TestConnectedComponents,
    "test_nonmanifold_edge_single_component",
    test_nonmanifold_edge_single_component,
    devices=devices,
)
add_function_test(TestConnectedComponents, "test_long_strip", test_long_strip, devices=devices)
add_function_test(TestConnectedComponents, "test_isolated_vertices", test_isolated_vertices, devices=devices)
add_function_test(TestConnectedComponents, "test_degenerate_triangles", test_degenerate_triangles, devices=devices)
add_function_test(TestConnectedComponents, "test_grid_single_component", test_grid_single_component, devices=devices)
add_function_test(
    TestConnectedComponents, "test_multiple_grid_components", test_multiple_grid_components, devices=devices
)
add_function_test(TestConnectedComponents, "test_empty_mesh", test_empty_mesh, devices=devices)
add_function_test(
    TestConnectedComponents,
    "test_permutation_and_relabel_invariance",
    test_permutation_and_relabel_invariance,
    devices=devices,
)
add_function_test(TestConnectedComponents, "test_infer_num_points", test_infer_num_points, devices=devices)
add_function_test(TestConnectedComponents, "test_random_soups", test_random_soups, devices=devices)
add_function_test(TestConnectedComponents, "test_segments", test_segments, devices=devices)
add_function_test(TestConnectedComponents, "test_tetrahedra", test_tetrahedra, devices=devices)
add_function_test(TestConnectedComponents, "test_points", test_points, devices=devices)
add_function_test(
    TestConnectedComponents, "test_random_soups_multi_size", test_random_soups_multi_size, devices=devices
)
add_function_test(TestConnectedComponents, "test_graph_capture", test_graph_capture, devices=devices)
add_function_test(TestConnectedComponents, "test_invalid_inputs", test_invalid_inputs, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
