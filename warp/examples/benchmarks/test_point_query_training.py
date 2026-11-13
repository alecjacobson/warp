# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the independent Chamfer loss/gradient oracle used by the benchmark."""

import unittest

import numpy as np

from warp.examples.benchmarks.benchmark_point_query_training import chamfer_reference


class TestChamferReference(unittest.TestCase):
    def test_finite_difference(self):
        """Match central differences of exhaustive symmetric Chamfer in two batches."""
        rng = np.random.default_rng(7831)
        points, queries = rng.normal(size=(2, 11, 3)), rng.normal(size=(2, 7, 3))

        def exhaustive_loss(decoded):
            distances = np.sum((decoded[:, :, None] - points[:, None]) ** 2, axis=-1)
            return distances.min(axis=2).mean() + distances.min(axis=1).mean()

        distances = np.sum((queries[:, :, None] - points[:, None]) ** 2, axis=-1)
        loss, gradient = chamfer_reference(points, queries, distances.argmin(axis=2), distances.argmin(axis=1))
        self.assertAlmostEqual(loss, exhaustive_loss(queries), places=12)
        epsilon = 1e-6
        numerical = np.zeros_like(queries)
        for index in np.ndindex(queries.shape):
            plus, minus = queries.copy(), queries.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (exhaustive_loss(plus) - exhaustive_loss(minus)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-7, atol=2e-9)

    def test_batch_isolation(self):
        """Keep reverse gradient accumulation within the corresponding batch."""
        points = np.array([[[0.0, 0, 0], [2, 0, 0]], [[100.0, 0, 0], [102, 0, 0]]])
        queries = np.array([[[0.5, 0, 0]], [[101.5, 0, 0]]])
        _, gradient = chamfer_reference(points, queries, np.array([[0], [1]]), np.zeros((2, 2), dtype=int))
        np.testing.assert_allclose(gradient, np.zeros_like(queries), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
