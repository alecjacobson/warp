# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the mesh intersection example on CPU and available CUDA devices.

Run with ``uv run -m warp.tests.geometry.test_mesh_intersect_hash``.
"""

import unittest

import numpy as np

import warp as wp
from warp.examples.geometry.example_mesh_intersect_hash import intersect_models, make_box
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def intersect_exhaustive(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array2d[int],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array2d[int],
    hits: wp.array2d[int],
):
    i, j = wp.tid()
    hits[i, j] = wp.intersect_tri_tri(
        vertices_a[faces_a[i, 0]],
        vertices_a[faces_a[i, 1]],
        vertices_a[faces_a[i, 2]],
        vertices_b[faces_b[j, 0]],
        vertices_b[faces_b[j, 1]],
        vertices_b[faces_b[j, 2]],
    )


def check_mesh_pair(test, device, model_a, model_b, *, cell_size=0.5, grid_dim=16):
    """Compare the example's hashed result against an independent exhaustive reference."""
    candidates_host, hashed_host = intersect_models(model_a, model_b, device, cell_size=cell_size, grid_dim=grid_dim)
    arrays = []
    for vertices, faces in (model_a, model_b):
        arrays.extend((wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces, dtype=int, device=device)))
    shape = (len(model_a[1]), len(model_b[1]))
    exhaustive = wp.zeros(shape, dtype=int, device=device)
    wp.launch(intersect_exhaustive, dim=shape, inputs=[*arrays, exhaustive], device=device)
    exhaustive_host = exhaustive.numpy()
    np.testing.assert_array_equal(hashed_host, exhaustive_host)
    test.assertTrue(np.all(candidates_host[exhaustive_host != 0] == 1), "Broad phase missed or duplicated a hit")
    test.assertLessEqual(int(candidates_host.max()), 1, "A hash bucket returned a pair more than once")
    return int(np.count_nonzero(candidates_host)), int(np.count_nonzero(hashed_host))


def test_box_meshes(test, device):
    model_a = make_box()
    cases = (
        ("overlapping", make_box(translation=(0.7, 0.3, 0.2), angle=0.31), True),
        ("separated", make_box(translation=(4.0, 0.0, 0.0)), False),
        ("contained", make_box(scale=0.25, angle=0.31), False),
        # Coarse triangles must be found even when they span multiple cells.
        ("unequal_triangles", make_box(subdivisions=1, translation=(0.7, 0.3, 0.2), angle=0.31), True),
    )
    for name, model_b, expected_hit in cases:
        with test.subTest(case=name):
            candidates, hits = check_mesh_pair(test, device, model_a, model_b)
            total = len(model_a[1]) * len(model_b[1])
            test.assertEqual(hits > 0, expected_hit)
            test.assertLess(candidates, total)
            print(f"{device} {name}: {candidates}/{total} candidate pairs, {hits} intersecting pairs")


def test_hash_collisions(test, device):
    # Negative coordinates and a tiny table force distant cells into the same buckets.
    model_a = make_box(translation=(-10.0, -10.0, -10.0))
    model_b = make_box(subdivisions=1, translation=(-9.3, -9.7, -9.8), angle=0.31)
    candidates, hits = check_mesh_pair(test, device, model_a, model_b, cell_size=0.125, grid_dim=2)
    test.assertGreater(hits, 0)
    test.assertLess(candidates, len(model_a[1]) * len(model_b[1]))


def test_screenshot_models(test, device):
    model_a = make_box(subdivisions=6)
    model_b = make_box(subdivisions=6, translation=(0.7, 0.3, 0.2), angle=0.31)
    candidates, hits = check_mesh_pair(test, device, model_a, model_b)
    test.assertGreater(hits, 0)
    test.assertLess(candidates, len(model_a[1]) * len(model_b[1]))


def test_triangle_contacts(test, device):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    model_a = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), faces)
    cases = (
        ("crossing", [[0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], 1),
        ("coplanar", [[0.1, 0.1, 0], [0.6, 0.1, 0], [0.1, 0.6, 0]], 1),
        ("touching", [[1, 0, 0], [2, 0, 1], [2, 1, 0]], 1),
        ("separated", [[0, 0, 1], [1, 0, 1], [0, 1, 1]], 0),
    )
    for name, vertices, expected_hits in cases:
        with test.subTest(case=name):
            model_b = (np.array(vertices, dtype=np.float32), faces)
            _, hits = check_mesh_pair(test, device, model_a, model_b, cell_size=0.25)
            test.assertEqual(hits, expected_hits)


class TestMeshIntersectHash(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestMeshIntersectHash, "test_box_meshes", test_box_meshes, devices=devices, check_output=False)
add_function_test(TestMeshIntersectHash, "test_hash_collisions", test_hash_collisions, devices=devices)
add_function_test(TestMeshIntersectHash, "test_screenshot_models", test_screenshot_models, devices=devices)
add_function_test(TestMeshIntersectHash, "test_triangle_contacts", test_triangle_contacts, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
