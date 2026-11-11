# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture complete Warp/PhysicsNeMo calls and replay after both inputs change."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy
from warp.examples.benchmarks.benchmark_point_query_hybrid import physics_isotropic
from warp.examples.benchmarks.point_query_backends import Provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    ws = wp.Stream("cuda:0")
    stream = wp.stream_to_torch(ws)
    report = {}
    with wp.ScopedStream(ws), torch.cuda.stream(stream), torch.no_grad():
        physics = Provider("physicsnemo", Path("../artifacts/comparison/pilot"))
        iso, digest = physics_isotropic(physics, args.output.parent / "physics-isotropic")
        tuned, tuned_digest = physics_isotropic(physics, args.output.parent / "physics-tuned", tuned=True)
        report["physics_tuned_sha256"] = tuned_digest
        for name, provider in [
            ("warp_configured", Provider("warp_configured", None)),
            ("physicsnemo", physics),
            ("physicsnemo_isotropic", iso),
            ("physicsnemo_tuned", tuned),
        ]:
            rng = np.random.default_rng(7308)
            p = rng.normal(size=(4096, 3)).astype(np.float32) * np.array([1, 1, 0.001], np.float32)
            q = rng.normal(size=(64, 3)).astype(np.float32) * np.array([1, 1, 0.001], np.float32)
            pt, qt = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
            problem = provider.bind(pt, qt, 11)
            for _ in range(3):
                result = problem.one_shot()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if name == "warp_configured":
                    wp.capture_begin(force_module_load=False, external=True)
                try:
                    result = problem.one_shot()
                finally:
                    if name == "warp_configured":
                        wp.capture_end()
            validations = []
            for _ in range(3):
                pt[:, 1].mul_(1.1)
                qt.add_(0.003)
                graph.replay()
                torch.cuda.synchronize()
                output = result[1]
                validations.append(
                    accuracy(pt.cpu().numpy(), qt.cpu().numpy(), output[0].cpu().numpy(), output[1].cpu().numpy(), 11)
                )
            report[name] = {
                "supported": True,
                "capture_scope": "build and query",
                "changed_point_and_query_validations": validations,
            }
        report["physics_isotropic_sha256"] = digest
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
