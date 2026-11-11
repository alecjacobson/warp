# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-specific oracle/data checks; intentionally outside the core suite."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy
from warp.examples.benchmarks.point_query_datasets import prepare


class TestComparisonOracle(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [3, 0, 0]], dtype=np.float32)
        self.q = np.zeros((1, 3), dtype=np.float32)

    def test_boundary_tie_is_interchangeable(self):
        for indices in ([0, 1], [0, 2]):
            result = accuracy(self.p, self.q, np.array([indices]), np.array([[0.0, 1.0]]), 2)
            self.assertEqual(result["tie_aware_recall_at_k"], 1.0)
            self.assertEqual(result["distance_rank_mismatch_rows"], 0)

    def test_missing_interior_neighbor(self):
        result = accuracy(self.p, self.q, np.array([[1, 2]]), np.array([[1.0, 1.0]]), 2)
        self.assertEqual(result["tie_aware_recall_at_k"], 0.5)
        self.assertEqual(result["distance_rank_mismatch_rows"], 1)

    def test_duplicate_results(self):
        result = accuracy(self.p, self.q, np.array([[0, 0]]), np.array([[0.0, 0.0]]), 2)
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["distance_rank_mismatch_rows"], 1)

    def test_reported_distances_are_checked(self):
        result = accuracy(self.p, self.q, np.array([[0, 1]]), np.array([[0.0, 0.5]]), 2)
        self.assertEqual(result["distance_rank_mismatch_rows"], 0)
        self.assertEqual(result["reported_distance_mismatch_rows"], 1)

    def test_invalid_indices(self):
        result = accuracy(self.p, self.q, np.array([[0, 4]]), np.array([[0.0, 1.0]]), 2)
        self.assertFalse(result["valid_indices_and_finite_distances"])


class TestComparisonData(unittest.TestCase):
    def test_scan_holdout_and_size_cap(self):
        source = np.arange(60, dtype=np.float64).reshape(20, 3)
        with patch("warp.examples.benchmarks.point_query_datasets.read_scan", return_value=(source, {})):
            p, q, metadata = prepare(Path("."), "bunny_scan", 100, 4, 0, 17)
        self.assertEqual(len(p), 16)
        self.assertEqual(len(q), 4)
        self.assertEqual(metadata["requested_N"], 100)
        self.assertFalse(set(map(tuple, p)) & set(map(tuple, q)))

    def test_deterministic_noise_and_inputs(self):
        p, q, metadata = prepare(Path("."), "volume", 101, 13, 0.01, 31)
        p2, q2, metadata2 = prepare(Path("."), "volume", 101, 13, 0.01, 31)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(q, q2)
        self.assertEqual(metadata["points_sha256"], metadata2["points_sha256"])
        clean, _, _ = prepare(Path("."), "volume", 101, 13, 0, 31)
        self.assertGreater(float(np.std(p - clean)), 0.005)
        self.assertLess(float(np.std(p - clean)), 0.015)

    def test_surface_area_weighting(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [5, 0, 0], [3, 2, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        with patch("warp.examples.benchmarks.point_query_datasets.read_bunny", return_value=(vertices, faces)):
            p, _, metadata = prepare(Path("."), "bunny_surface", 10000, 100, 0, 31)
        original_x = p[:, 0] * metadata["scale"] + metadata["center"][0]
        self.assertAlmostEqual(float(np.mean(original_x >= 3)), 0.8, delta=0.025)


if __name__ == "__main__":
    unittest.main()
