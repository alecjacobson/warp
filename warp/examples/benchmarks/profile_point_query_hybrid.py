# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile matching Warp and PhysicsNeMo build/query phases on frozen inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.point_query_backends import Provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="sphere-n1000000-q8192-sigma0-k11")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = Path("../data/knn-comparison/pilot/manifest.json")
    case = next(c for c in json.loads(manifest.read_text())["cases"] if c["id"] == args.case)
    d = np.load(manifest.parent / case["file"])
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    ws = wp.Stream("cuda:0")
    with wp.ScopedStream(ws), torch.cuda.stream(wp.stream_to_torch(ws)), torch.no_grad():
        p, q = torch.as_tensor(d["points"], device="cuda"), torch.as_tensor(d["queries"], device="cuda")
        old = Provider("warp_configured", None).bind(p, q, case["k"])
        physics = Provider("physicsnemo", Path("../artifacts/comparison/pilot")).library
        for _ in range(3):
            _result = old.one_shot()
            _other = physics.knn_impl(p, q, case["k"])
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            with torch.profiler.record_function("warp_build"):
                index = old.build()
            with torch.profiler.record_function("warp_search"):
                _result = old.search(index)
            with torch.profiler.record_function("physics_build"):
                tree = physics.build_morton_tree(p)
            with torch.profiler.record_function("physics_plan"):
                order, window = physics.morton_query_plan(tree, q)
            with torch.profiler.record_function("physics_search"):
                _other = physics.tree_knn_query(tree, q, case["k"], order, window)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / "trace.json"))
        (args.output / "activity.txt").write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=90))


if __name__ == "__main__":
    main()
