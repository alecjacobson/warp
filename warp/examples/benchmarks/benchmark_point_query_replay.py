# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure scalar and public array queries, including CUDA graph replay.

Use the refresh comparison's timer, stream, output contract and environment.
Torch is used by the shared timer only; query implementations use Warp arrays.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy, environment, measure
from warp.examples.benchmarks.point_query_common import make_api_query, make_ordered_query, query_codes


def validate(p, q, result, k):
    return accuracy(p, q, result[1].numpy(), result[2].numpy(), k)


class Query:
    def __init__(self, points, queries, k, method):
        self.points, self.queries, self.k, self.method = points, queries, k, method
        self.nq = len(queries)
        self.config = wp.point_query_config(len(points), self.nq, points.device)
        self.excludes = wp.full(self.nq, -1, dtype=int, device=points.device)
        self.stride = 1 if method == "warp_scalar" else self.config.query_stride
        self.ordered = method != "warp_scalar" and self.config.ordered
        self.kernel = make_ordered_query(k, True) if self.ordered else make_api_query(k, True, self.stride)
        self.backend = (
            wp.PointQueryBackend.CUBQL
            if method in ("warp_cubql_reuse", "warp_cubql_build")
            else wp.PointQueryBackend.WARP
        )
        self.objective = (
            wp.PointQueryObjective.BUILD_AND_QUERY if method == "warp_cubql_build" else wp.PointQueryObjective.REUSE
        )

    def build(self):
        if self.method == "warp_scalar":
            # Literal simple usage: preserve the actual default leaf size (1).
            return wp.Bvh(self.points, self.points)
        if self.method == "warp_bvh_cubql":
            return wp.Bvh(self.points, self.points, constructor="cubql", leaf_size=self.config.leaf_size)
        if self.method in ("warp_batch", "warp_cubql_reuse", "warp_cubql_build"):
            return wp.build_point_index(
                self.points, num_queries=self.nq, k=self.k, backend=self.backend, objective=self.objective
            )
        return wp.Bvh(self.points, self.points, leaf_size=self.config.leaf_size)

    def outputs(self):
        return (
            wp.empty((self.nq, self.k), dtype=wp.int64, device=self.points.device),
            wp.empty((self.nq, self.k), dtype=float, device=self.points.device),
            wp.empty(self.nq, dtype=int, device=self.points.device),
        )

    def call(self, bvh=None, outputs=None):
        ids, distances, counts = self.outputs() if outputs is None else outputs
        if self.method in ("warp_batch", "warp_cubql_reuse", "warp_cubql_build", "warp_bvh_cubql"):
            if bvh is None and self.method == "warp_bvh_cubql":
                bvh = self.build()
            return wp.knn(
                self.points,
                self.queries,
                self.k,
                bvh=bvh,
                indices=ids,
                distances=distances,
                counts=counts,
                squared=False,
                backend=self.backend,
                objective=self.objective,
            )
        if bvh is None:
            bvh = self.build()
        extra = []
        if self.ordered:
            keys = wp.empty(2 * self.nq, dtype=int, device=self.points.device)
            order = wp.empty_like(keys)
            wp.launch(query_codes, self.nq, inputs=[bvh.id, self.queries, keys, order])
            wp.utils.radix_sort_pairs(keys, order, self.nq)
            extra = [order]
        wp.launch(
            self.kernel,
            self.nq * self.stride,
            inputs=[bvh.id, self.points, self.queries, wp.inf, self.excludes, *extra, ids, distances, counts],
            block_dim=32 if self.stride > 1 else 256,
        )
        return bvh, ids, distances, counts


def graph_measure(problem, p, q, index, repeats):
    # Caller-owned outputs avoid output allocation nodes on the reused path.
    outputs = problem.outputs()
    warm = problem.call(index, outputs)
    wp.synchronize_device()
    start = time.perf_counter()
    with wp.ScopedCapture() as capture:
        result = problem.call(index, outputs)
    setup_ms = 1000 * (time.perf_counter() - start)

    def replay():
        wp.capture_launch(capture.graph)
        return result

    # Prove that replay reads changed values, not the original capture inputs.
    changed_q = q + np.float32(0.013)
    changed_p = p * np.float32(1.01) if index is None else p
    problem.queries.assign(changed_q)
    if index is None:
        problem.points.assign(changed_p)
    changed_validation = validate(changed_p, changed_q, replay(), problem.k)
    problem.points.assign(p)
    problem.queries.assign(q)
    timing, _ = measure(replay, repeats)
    validation = validate(p, q, result, problem.k)
    # Graph instantiation/upload may be lazy; the two timer warmups exclude it.
    timing["capture_wall_ms"] = setup_ms
    timing["validation"] = validation
    timing["changed_input_validation"] = changed_validation
    del warm
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("warp_scalar", "warp_tuned", "warp_batch", "warp_cubql_reuse", "warp_cubql_build", "warp_bvh_cubql"),
        default=("warp_scalar", "warp_tuned", "warp_batch"),
    )
    args = parser.parse_args()
    wp.init()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if "warp_bvh_cubql" in args.methods:
        # Initialize the upstream allocator before applying the application policy.
        initial_points = wp.zeros(8, dtype=wp.vec3, device="cuda:0")
        initial_index = wp.Bvh(initial_points, initial_points, constructor="cubql")
        del initial_index, initial_points
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    stream = wp.Stream("cuda:0")
    source_files = (
        "warp/_src/point_query.py",
        "warp/native/bvh.cu",
        "warp/native/bvh_cubql.cu",
        "warp/native/point_query.h",
        "warp/bin/warp.so",
        "warp/examples/benchmarks/benchmark_point_query_replay.py",
        "warp/examples/benchmarks/point_query_common.py",
        "warp/examples/benchmarks/point_query_backends.py",
        "warp/examples/benchmarks/benchmark_point_query_comparison.py",
    )
    report = {
        "environment": environment(),
        "repeats": args.repeats,
        "source_sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": wp.get_device("cuda:0").name,
        "sm_count": wp.get_device("cuda:0").sm_count,
        "primary_timing": "shared comparison timer; synchronized wall including events and submission",
        "graph_timing": "single replay plus synchronization; capture/instantiation/warmup excluded",
        "output_contract": "sorted int64 indices and float32 Euclidean distances, GPU resident",
        "allocation": "Warp arrays, 512 MiB pool retention; previous comparison used Torch output allocation",
        "shared_stream": "Warp-owned blocking stream, borrowed by Torch timer",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.manifest.read_text())["cases"]
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        for case in cases[: args.limit]:
            path = args.manifest.parent / case["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != case["file_sha256"]:
                raise ValueError("Dataset checksum mismatch")
            with np.load(path) as data:
                p, q = data["points"], data["queries"][: case["Q"]]
            points, queries = wp.array(p, dtype=wp.vec3, device="cuda:0"), wp.array(q, dtype=wp.vec3, device="cuda:0")
            row = {"case": case, "variants": {}}
            for method in args.methods:
                problem = Query(points, queries, case["k"], method)
                build, index = measure(problem.build, args.repeats)
                query_timing, result = measure(lambda problem=problem, index=index: problem.call(index), args.repeats)
                query_valid = validate(p, q, result, case["k"])
                one_shot, result = measure(problem.call, args.repeats)
                valid = validate(p, q, result, case["k"])
                graph_query = graph_measure(problem, p, q, index, args.repeats)
                build_capturable = problem.backend == wp.PointQueryBackend.WARP and method != "warp_bvh_cubql"
                graph_full = graph_measure(problem, p, q, None, args.repeats) if build_capturable else None
                row["variants"][method] = {
                    "build": build,
                    "reused_index_search": query_timing,
                    "one_shot": one_shot,
                    "validation": valid,
                    "reused_validation": query_valid,
                    "graph_one_shot": graph_full,
                    "graph_one_shot_status": "measured" if build_capturable else "construction_not_capturable",
                    "graph_reused_index_search": graph_query,
                    "config": {
                        "backend": problem.backend.name,
                        "objective": problem.objective.name,
                        "constructor": "cubql" if method == "warp_bvh_cubql" else "automatic",
                        "leaf_size": (
                            index.leaf_size
                            if problem.backend == wp.PointQueryBackend.CUBQL
                            else 1
                            if method == "warp_scalar"
                            else problem.config.leaf_size
                        ),
                        "query_stride": problem.stride if problem.backend == wp.PointQueryBackend.WARP else "automatic",
                        "ordered": problem.ordered if problem.backend == wp.PointQueryBackend.WARP else False,
                    },
                }
                print(
                    case["id"],
                    method,
                    "eager",
                    round(one_shot["wall_median_ms"], 4),
                    round(query_timing["wall_median_ms"], 4),
                    "graph",
                    round(graph_full["wall_median_ms"], 4) if graph_full else "unsupported",
                    round(graph_query["wall_median_ms"], 4),
                    flush=True,
                )
                del index, result, problem
                gc.collect()
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del points, queries
            gc.collect()


if __name__ == "__main__":
    main()
