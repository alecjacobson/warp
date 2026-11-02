# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect kernel activity, graph timings, and an optional CUPTI trace for point queries.

Example: uv run --no-sync -m warp.examples.benchmarks.profile_point_query
    --output ../knn-profile --leaf-size 32 --ordered --trace
The optional trace uses PyTorch's profiler; event timings are collected separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query import cloud, measure, validate
from warp.examples.benchmarks.point_query_common import make_api_query, make_ordered_query, query_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--leaf-size", type=int, default=4)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.ordered and args.query_stride != 1:
        parser.error("Use either ordered or sparse query launches")
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device("cuda:0")
    p, q = cloud("volume", args.points, args.queries, 0.0, 531)
    points = wp.array(p, dtype=wp.vec3, device=device)
    queries = wp.array(q, dtype=wp.vec3, device=device)
    excludes = wp.full(len(q), -1, dtype=int, device=device)
    ids = wp.empty((len(q), args.k), dtype=wp.int64, device=device)
    ds = wp.empty((len(q), args.k), dtype=float, device=device)
    counts = wp.empty(len(q), dtype=int, device=device)
    keys = wp.empty(2 * len(q), dtype=int, device=device)
    order = wp.empty_like(keys)
    kernel = make_ordered_query(args.k, True) if args.ordered else make_api_query(args.k, True, args.query_stride)
    tree = wp.Bvh(points, points, constructor="lbvh", leaf_size=args.leaf_size)

    def plan():
        if args.ordered:
            wp.launch(query_codes, len(q), inputs=[tree.id, queries, keys, order], device=device)
            wp.utils.radix_sort_pairs(keys, order, len(q))

    def query():
        inputs = [tree.id, points, queries, np.inf, excludes] + ([order] if args.ordered else []) + [ids, ds, counts]
        wp.launch(
            kernel,
            len(q) * args.query_stride,
            inputs=inputs,
            block_dim=32 if args.query_stride > 1 else 256,
            device=device,
        )

    plan()
    query()
    tree.rebuild()
    wp.synchronize_device(device)
    result = {
        "device": device.name,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    # Built-in timers expose Warp activities. CUPTI below also exposes CUB's
    # internal kernels and CUDA runtime allocations/copies/synchronization.
    with wp.ScopedTimer("construction", cuda_filter=wp.TIMING_ALL) as timer:
        built = wp.Bvh(points, points, constructor="lbvh", leaf_size=args.leaf_size)
    result["construction_activities"] = [
        {"name": t.name, "ms": t.elapsed, "kind": t.filter} for t in timer.timing_results
    ]
    result["query_graph"], _ = measure(query, device, 7, graph=True)

    def rebuilt_query():
        tree.rebuild()
        plan()
        query()

    result["rebuild_plan_query_graph"], _ = measure(rebuilt_query, device, 7, graph=True)
    result["validation"] = validate(p, q, ids.numpy(), args.k)
    if result["validation"]["float32_mismatch_rows"]:
        raise AssertionError(result["validation"])
    if args.trace:
        import torch  # noqa: PLC0415 - optional profiler dependency

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            with torch.profiler.record_function("warp_construction"):
                built = wp.Bvh(points, points, constructor="lbvh", leaf_size=args.leaf_size)
            with torch.profiler.record_function("warp_query_plan"):
                plan()
            with torch.profiler.record_function("warp_query"):
                query()
            # This explicit profiler boundary is not part of construction/query.
            wp.synchronize_device(device)
        profiler.export_chrome_trace(str(args.output / "trace.json"))
        (args.output / "activity.txt").write_text(
            profiler.key_averages().table(sort_by="cuda_time_total", row_limit=80)
        )
    result["built_id_nonzero"] = bool(built.id)
    (args.output / "profile.json").write_text(json.dumps(result, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
