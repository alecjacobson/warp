# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track construction-inclusive and reused-target point-query costs on CUDA."""

import numpy as np

import warp as wp
import warp.geometry

from .benchmarks_utils import setup_once


class PointQuery:
    params = (["warp", "cubql_reuse", "cubql_build"], [1, 4], [1024, 100_000], [1, 11])
    param_names = ["configuration", "batches", "targets", "k"]
    repeat = 10
    number = 1

    @setup_once
    def setup(self, configuration, batches, targets, k):
        if not wp.is_cuda_available():
            raise NotImplementedError("Point-query benchmarks require CUDA")
        self.device = wp.get_device("cuda:0")
        if configuration != "warp" and (not wp.is_cubql_available() or not self.device.is_mempool_supported):
            raise NotImplementedError("Native cuBQL requires a cuBQL-enabled build and CUDA memory pools")
        self.backend = (
            wp.geometry.PointQueryBackend.WARP if configuration == "warp" else wp.geometry.PointQueryBackend.CUBQL
        )
        self.objective = (
            wp.geometry.PointQueryObjective.BUILD_AND_QUERY
            if configuration == "cubql_build"
            else wp.geometry.PointQueryObjective.REUSE
        )
        rng = np.random.default_rng(234)
        self.points = wp.array(
            rng.normal(size=(batches, targets, 3)).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.queries = wp.array(
            rng.normal(size=(batches, 1024, 3)).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.result = wp.geometry.knn(self.points, self.queries, k, backend=self.backend, objective=self.objective)
        self.outputs = {"indices": self.result[1], "distances": self.result[2], "counts": self.result[3]}
        # Warm compilation and retain the owner and outputs through graph replay.
        with wp.ScopedCapture(device=self.device) as capture:
            wp.geometry.knn(self.result[0], self.queries, k, **self.outputs)
        self.graph = capture.graph
        wp.synchronize_device(self.device)

    def time_build_query(self, configuration, batches, targets, k):
        result = wp.geometry.knn(
            self.points, self.queries, k, backend=self.backend, objective=self.objective, **self.outputs
        )
        wp.synchronize_device(self.device)
        # Include index retirement in the one-shot workload.
        del result

    def time_reused_query(self, configuration, batches, targets, k):
        wp.geometry.knn(self.result[0], self.queries, k, **self.outputs)
        wp.synchronize_device(self.device)

    def time_reused_graph(self, configuration, batches, targets, k):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
