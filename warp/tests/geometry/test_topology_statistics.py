# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.geometry
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# Independent NumPy/Python reference (does not use the Warp implementation)
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = (
    "num_vertices",
    "num_triangles",
    "num_edges",
    "num_boundary_edges",
    "num_nonmanifold_edges",
    "num_misoriented_edges",
    "num_nonmanifold_vertices",
    "num_unreferenced_vertices",
    "num_degenerate_triangles",
)

_PREDICATE_FIELDS = (
    "is_edge_manifold",
    "is_closed_edge_manifold",
    "is_vertex_manifold",
    "is_manifold",
    "is_closed_manifold",
    "is_oriented",
)


def _reference(tris, num_points):
    """Compute topology statistics from scratch with Python dicts/sets."""
    tris = [tuple(int(v) for v in t) for t in tris]
    num_triangles = len(tris)

    kept = []
    num_degenerate = 0
    for i, j, k in tris:
        if i == j or j == k or k == i:
            num_degenerate += 1
        else:
            kept.append((i, j, k))

    # Canonical undirected edge -> (count, signed_count).
    edges = {}
    for i, j, k in kept:
        for a, b in ((i, j), (j, k), (k, i)):
            u, v = (a, b) if a < b else (b, a)
            count, signed = edges.get((u, v), (0, 0))
            count += 1
            signed += 1 if a == u else -1
            edges[(u, v)] = (count, signed)

    num_boundary = sum(1 for c, _ in edges.values() if c == 1)
    num_nonmanifold_edges = sum(1 for c, _ in edges.values() if c > 2)
    num_misoriented = sum(1 for c, s in edges.values() if c == 2 and s != 0)

    # Incident faces per referenced vertex.
    incident = {}
    for i, j, k in kept:
        for v in (i, j, k):
            incident.setdefault(v, []).append({i, j, k} - {v})

    num_unreferenced = num_points - len(incident)

    # A vertex is manifold iff its incident faces form one edge-connected fan:
    # two faces are adjacent when they share a neighbor of the vertex.
    num_nonmanifold_vertices = 0
    for neighbors in incident.values():
        n = len(neighbors)
        seen = [False] * n
        components = 0
        for start in range(n):
            if seen[start]:
                continue
            components += 1
            stack = [start]
            seen[start] = True
            while stack:
                cur = stack.pop()
                for other in range(n):
                    if not seen[other] and (neighbors[cur] & neighbors[other]):
                        seen[other] = True
                        stack.append(other)
        if components > 1:
            num_nonmanifold_vertices += 1

    scalars = {
        "num_vertices": num_points,
        "num_triangles": num_triangles,
        "num_edges": len(edges),
        "num_boundary_edges": num_boundary,
        "num_nonmanifold_edges": num_nonmanifold_edges,
        "num_misoriented_edges": num_misoriented,
        "num_nonmanifold_vertices": num_nonmanifold_vertices,
        "num_unreferenced_vertices": num_unreferenced,
        "num_degenerate_triangles": num_degenerate,
    }

    is_edge_manifold = num_nonmanifold_edges == 0 and num_degenerate == 0
    is_closed_edge_manifold = is_edge_manifold and num_boundary == 0
    is_vertex_manifold = num_nonmanifold_vertices == 0 and num_unreferenced == 0 and num_degenerate == 0
    predicates = {
        "is_edge_manifold": is_edge_manifold,
        "is_closed_edge_manifold": is_closed_edge_manifold,
        "is_vertex_manifold": is_vertex_manifold,
        "is_manifold": is_edge_manifold and is_vertex_manifold,
        "is_closed_manifold": is_closed_edge_manifold and is_vertex_manifold,
        "is_oriented": num_misoriented == 0,
    }
    return scalars, predicates


def _compute(tris, num_points, device):
    flat = np.asarray(tris, dtype=np.int32).reshape(-1)
    indices = wp.array(flat, dtype=wp.int32, device=device)
    return warp.geometry.triangle_mesh_topology_statistics(indices, num_points=num_points, device=device)


def _assert_matches_reference(test, tris, num_points, device):
    stats = _compute(tris, num_points, device)
    scalars, predicates = _reference(tris, num_points)
    for field, expected in scalars.items():
        test.assertEqual(getattr(stats, field), expected, f"{field} mismatch")
    for field, expected in predicates.items():
        test.assertEqual(getattr(stats, field), expected, f"{field} mismatch")
    return stats


# ---------------------------------------------------------------------------
# Deterministic meshes
# ---------------------------------------------------------------------------

# Consistently outward-oriented tetrahedron.
_TET = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def _fan_mesh(num_triangles):
    """A disk fan with a high-valence center vertex 0 and a closed rim loop.

    The center vertex 0 has valence ``num_triangles``; the rim vertices
    ``1..num_triangles`` form a boundary loop, so every rim edge is a boundary
    edge and every spoke edge is interior.
    """
    n = num_triangles
    return [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_triangle(test, device):
    stats = _assert_matches_reference(test, [(0, 1, 2)], 3, device)
    test.assertEqual(stats.num_edges, 3)
    test.assertEqual(stats.num_boundary_edges, 3)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    test.assertEqual(stats.num_misoriented_edges, 0)
    test.assertTrue(stats.is_edge_manifold)
    test.assertFalse(stats.is_closed_edge_manifold)
    test.assertTrue(stats.is_vertex_manifold)
    test.assertTrue(stats.is_manifold)
    test.assertFalse(stats.is_closed_manifold)
    test.assertTrue(stats.is_oriented)


def test_quad(test, device):
    stats = _assert_matches_reference(test, [(0, 1, 2), (0, 2, 3)], 4, device)
    test.assertEqual(stats.num_edges, 5)
    test.assertEqual(stats.num_boundary_edges, 4)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    test.assertEqual(stats.num_misoriented_edges, 0)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertTrue(stats.is_manifold)
    test.assertTrue(stats.is_oriented)


def test_misoriented_shared_edge(test, device):
    # Flip the second triangle so both traverse the shared edge (0, 2) the same way.
    stats = _assert_matches_reference(test, [(0, 1, 2), (0, 3, 2)], 4, device)
    test.assertEqual(stats.num_edges, 5)
    test.assertEqual(stats.num_boundary_edges, 4)
    test.assertEqual(stats.num_misoriented_edges, 1)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertFalse(stats.is_oriented)
    test.assertTrue(stats.is_manifold)  # orientation is independent of manifoldness


def test_closed_tetrahedron(test, device):
    stats = _assert_matches_reference(test, _TET, 4, device)
    test.assertEqual(stats.num_edges, 6)
    test.assertEqual(stats.num_boundary_edges, 0)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    test.assertEqual(stats.num_misoriented_edges, 0)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertTrue(stats.is_closed_manifold)
    test.assertTrue(stats.is_oriented)


def test_flipped_tetrahedron_face(test, device):
    tet = list(_TET)
    a, b, c = tet[0]
    tet[0] = (a, c, b)  # flip one face
    stats = _assert_matches_reference(test, tet, 4, device)
    # The flipped face shares all three of its edges, so three edges misorient.
    test.assertEqual(stats.num_misoriented_edges, 3)
    test.assertEqual(stats.num_boundary_edges, 0)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    test.assertFalse(stats.is_oriented)
    test.assertTrue(stats.is_closed_manifold)


def test_nonmanifold_edge_vertex_manifold(test, device):
    # Three triangles sharing edge (0, 1); vertices 0 and 1 still form one fan.
    tris = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
    stats = _assert_matches_reference(test, tris, 5, device)
    test.assertEqual(stats.num_nonmanifold_edges, 1)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertFalse(stats.is_edge_manifold)
    test.assertTrue(stats.is_vertex_manifold)
    test.assertFalse(stats.is_manifold)


def test_bowtie_vertex(test, device):
    tris = [(0, 1, 2), (0, 3, 4)]
    stats = _assert_matches_reference(test, tris, 5, device)
    test.assertEqual(stats.num_boundary_edges, 6)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    test.assertEqual(stats.num_nonmanifold_vertices, 1)  # vertex 0
    test.assertFalse(stats.is_vertex_manifold)
    test.assertFalse(stats.is_manifold)


def test_disconnected_components(test, device):
    tris = [(0, 1, 2), (3, 4, 5)]
    stats = _assert_matches_reference(test, tris, 6, device)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertEqual(stats.num_unreferenced_vertices, 0)
    test.assertTrue(stats.is_manifold)


def test_unreferenced_vertex(test, device):
    stats = _assert_matches_reference(test, [(0, 1, 2)], 4, device)
    test.assertEqual(stats.num_unreferenced_vertices, 1)
    test.assertFalse(stats.is_vertex_manifold)
    test.assertFalse(stats.is_manifold)


def test_empty_mesh(test, device):
    for num_points in (0, 5):
        stats = _assert_matches_reference(test, np.zeros((0, 3), dtype=np.int32), num_points, device)
        test.assertEqual(stats.num_triangles, 0)
        test.assertEqual(stats.num_edges, 0)
        test.assertTrue(stats.is_edge_manifold)
        test.assertTrue(stats.is_oriented)
        # An empty vertex set is vacuously vertex-manifold; unreferenced vertices are not.
        test.assertEqual(stats.is_vertex_manifold, num_points == 0)


def test_degenerate_faces(test, device):
    tris = [(0, 0, 1), (0, 1, 0), (2, 2, 2)]
    stats = _assert_matches_reference(test, tris, 3, device)
    test.assertEqual(stats.num_degenerate_triangles, 3)
    test.assertEqual(stats.num_edges, 0)
    test.assertEqual(stats.num_triangles, 3)
    test.assertFalse(stats.is_edge_manifold)
    test.assertFalse(stats.is_vertex_manifold)
    test.assertFalse(stats.is_manifold)


def test_degenerate_mixed_with_valid(test, device):
    # A valid closed tetrahedron plus a degenerate triangle referencing a new vertex.
    tris = [*_TET, (4, 4, 4)]
    stats = _assert_matches_reference(test, tris, 5, device)
    test.assertEqual(stats.num_degenerate_triangles, 1)
    test.assertEqual(stats.num_edges, 6)
    # Vertex 4 only appears in the (skipped) degenerate face, so it is unreferenced.
    test.assertEqual(stats.num_unreferenced_vertices, 1)
    test.assertFalse(stats.is_manifold)


def test_high_valence_fan(test, device):
    tris = _fan_mesh(256)
    num_points = 257
    stats = _assert_matches_reference(test, tris, num_points, device)
    test.assertEqual(stats.num_nonmanifold_vertices, 0)
    test.assertEqual(stats.num_nonmanifold_edges, 0)
    # Center vertex has valence 256; the rim loop of 256 edges is the boundary.
    test.assertEqual(stats.num_boundary_edges, 256)
    test.assertTrue(stats.is_manifold)
    test.assertFalse(stats.is_closed_manifold)
    test.assertTrue(stats.is_oriented)


def test_permutation_invariance(test, device):
    rng = np.random.default_rng(7)
    tris = [*_TET, (0, 1, 2), (0, 1, 3), *_fan_mesh(16)]
    num_points = int(np.max(tris)) + 1
    baseline = _compute(tris, num_points, device)

    # Reorder triangles.
    order = rng.permutation(len(tris))
    shuffled = [tris[i] for i in order]
    permuted = _compute(shuffled, num_points, device)
    for field in _SCALAR_FIELDS + _PREDICATE_FIELDS:
        test.assertEqual(getattr(permuted, field), getattr(baseline, field), f"{field} not permutation-invariant")

    # Relabel vertices with a random bijection.
    relabel = rng.permutation(num_points)
    relabeled = [tuple(int(relabel[v]) for v in t) for t in tris]
    relabeled_stats = _compute(relabeled, num_points, device)
    for field in _SCALAR_FIELDS + _PREDICATE_FIELDS:
        test.assertEqual(getattr(relabeled_stats, field), getattr(baseline, field), f"{field} not relabel-invariant")


def _random_soup(rng, num_points, num_triangles):
    tris = []
    for _ in range(num_triangles):
        while True:
            t = rng.integers(0, num_points, size=3)
            if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]:
                break
        tris.append(tuple(int(v) for v in t))
    return tris


def test_random_soups(test, device):
    rng = np.random.default_rng(20240817)
    for _ in range(60):
        num_points = int(rng.integers(3, 20))
        num_triangles = int(rng.integers(1, 30))
        tris = _random_soup(rng, num_points, num_triangles)
        _assert_matches_reference(test, tris, num_points, device)


def test_random_soups_infer_num_points(test, device):
    # Exercise the num_points=None inference path against the reference.
    rng = np.random.default_rng(99)
    for _ in range(20):
        num_points = int(rng.integers(3, 15))
        tris = _random_soup(rng, num_points, int(rng.integers(1, 20)))
        # Ensure the largest vertex is referenced so inference matches num_points.
        tris.append((num_points - 1, (num_points - 2) % num_points, (num_points - 3) % num_points))
        flat = np.asarray(tris, dtype=np.int32).reshape(-1)
        indices = wp.array(flat, dtype=wp.int32, device=device)
        stats = warp.geometry.triangle_mesh_topology_statistics(indices, device=device)
        scalars, predicates = _reference(tris, num_points)
        for field, expected in {**scalars, **predicates}.items():
            test.assertEqual(getattr(stats, field), expected, f"{field} mismatch")


def test_invalid_inputs(test, device):
    # Non-multiple-of-three length.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.triangle_mesh_topology_statistics(bad)
    # Index out of range.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([0, 1, 5], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.triangle_mesh_topology_statistics(bad, num_points=3)
    # Negative index.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([0, 1, -1], dtype=np.int32), dtype=wp.int32, device=device)
        warp.geometry.triangle_mesh_topology_statistics(bad, num_points=3)
    # Wrong dtype.
    with test.assertRaises(ValueError):
        bad = wp.array(np.array([0, 1, 2], dtype=np.int64), dtype=wp.int64, device=device)
        warp.geometry.triangle_mesh_topology_statistics(bad, num_points=3)


devices = get_test_devices()


class TestTopologyStatistics(unittest.TestCase):
    pass


add_function_test(TestTopologyStatistics, "test_single_triangle", test_single_triangle, devices=devices)
add_function_test(TestTopologyStatistics, "test_quad", test_quad, devices=devices)
add_function_test(TestTopologyStatistics, "test_misoriented_shared_edge", test_misoriented_shared_edge, devices=devices)
add_function_test(TestTopologyStatistics, "test_closed_tetrahedron", test_closed_tetrahedron, devices=devices)
add_function_test(
    TestTopologyStatistics, "test_flipped_tetrahedron_face", test_flipped_tetrahedron_face, devices=devices
)
add_function_test(
    TestTopologyStatistics,
    "test_nonmanifold_edge_vertex_manifold",
    test_nonmanifold_edge_vertex_manifold,
    devices=devices,
)
add_function_test(TestTopologyStatistics, "test_bowtie_vertex", test_bowtie_vertex, devices=devices)
add_function_test(TestTopologyStatistics, "test_disconnected_components", test_disconnected_components, devices=devices)
add_function_test(TestTopologyStatistics, "test_unreferenced_vertex", test_unreferenced_vertex, devices=devices)
add_function_test(TestTopologyStatistics, "test_empty_mesh", test_empty_mesh, devices=devices)
add_function_test(TestTopologyStatistics, "test_degenerate_faces", test_degenerate_faces, devices=devices)
add_function_test(
    TestTopologyStatistics, "test_degenerate_mixed_with_valid", test_degenerate_mixed_with_valid, devices=devices
)
add_function_test(TestTopologyStatistics, "test_high_valence_fan", test_high_valence_fan, devices=devices)
add_function_test(TestTopologyStatistics, "test_permutation_invariance", test_permutation_invariance, devices=devices)
add_function_test(TestTopologyStatistics, "test_random_soups", test_random_soups, devices=devices)
add_function_test(
    TestTopologyStatistics, "test_random_soups_infer_num_points", test_random_soups_infer_num_points, devices=devices
)
add_function_test(TestTopologyStatistics, "test_invalid_inputs", test_invalid_inputs, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
