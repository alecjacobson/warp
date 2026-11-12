# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Warp kNN with LBVH and cuBQL SAH construction at matching settings."""

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import environment, measure
from warp.examples.benchmarks.benchmark_point_query_replay import Query, graph_measure, validate


class BuilderQuery(Query):
    def __init__(self, points, queries, k, constructor):
        super().__init__(points, queries, k, "warp_batch")
        self.constructor = constructor

    def build(self):
        return wp.Bvh(self.points, self.points, constructor=self.constructor, leaf_size=self.config.leaf_size)

    def call(self, bvh=None, outputs=None):
        if bvh is None:
            bvh = self.build()
        return super().call(bvh, outputs)


def capture_probe(args):
    """Run a full-build probe in its own process, outside timing workers."""
    rng = np.random.default_rng(715)
    p, q = rng.normal(size=(513, 3)).astype(np.float32), rng.normal(size=(65, 3)).astype(np.float32)
    points, queries = wp.array(p, dtype=wp.vec3), wp.array(q, dtype=wp.vec3)
    problem = BuilderQuery(points, queries, 11, args.capture_probe)
    report = {"constructor": args.capture_probe, "environment": environment()}
    try:
        report["timing_and_validation"] = graph_measure(problem, p, q, None, 3)
        report["supported"] = True
    except Exception as error:
        report.update(supported=False, error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def benchmark(args):
    sources = (
        "warp/examples/benchmarks/benchmark_point_query_builders.py",
        "warp/examples/benchmarks/benchmark_point_query_replay.py",
        "warp/examples/benchmarks/benchmark_point_query_comparison.py",
        "warp/_src/point_query.py",
        "warp/native/point_query.h",
        "warp/native/bvh.cu",
        "warp/native/bvh_cubql.cu",
        "warp/bin/warp.so",
    )
    # Initialize cuBQL's memory resource before setting the shared pool policy.
    dummy = wp.zeros(4, dtype=wp.vec3)
    warm_tree = wp.Bvh(dummy, dummy, constructor="cubql")
    del warm_tree, dummy
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    report = {
        "environment": environment(),
        "repeats": 9,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": wp.get_device().name,
        "sm_count": wp.get_device().sm_count,
        "output_contract": "GPU int64 indices, float32 Euclidean distances; same batch query kernels",
        "timing": "shared comparison timer, nine synchronized wall/event samples after two warmups",
        "policy": "same leaf size, query stride, block size and ordering; explicit construction for both paths",
        "pool_release_threshold_bytes": wp.get_mempool_release_threshold("cuda:0"),
        "cases": [],
    }
    cases = json.loads(args.manifest.read_text())["cases"]
    for case in cases[: args.limit]:
        path = args.manifest.parent / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["file_sha256"]:
            raise ValueError("Dataset checksum mismatch")
        with np.load(path) as arrays:
            p, q = arrays["points"], arrays["queries"][: case["Q"]]
        points, queries = wp.array(p, dtype=wp.vec3), wp.array(q, dtype=wp.vec3)
        row = {"case": case, "variants": {}}
        for constructor in ("lbvh", "cubql"):
            problem = BuilderQuery(points, queries, case["k"], constructor)
            build, index = measure(problem.build, 9)
            query, result = measure(lambda problem=problem, index=index: problem.call(index), 9)
            query_valid = validate(p, q, result, case["k"])
            full, result = measure(problem.call, 9)
            valid = validate(p, q, result, case["k"])
            graph_query = graph_measure(problem, p, q, index, 9)
            # cuBQL's existing construction/conversion path has host readbacks.
            graph_full = graph_measure(problem, p, q, None, 9) if constructor == "lbvh" else None
            row["variants"]["warp_batch_" + constructor] = {
                "build": build,
                "one_shot": full,
                "reused_index_search": query,
                "validation": valid,
                "reused_validation": query_valid,
                "graph_reused_index_search": graph_query,
                "graph_one_shot": graph_full,
                "config": {**vars(problem.config), "constructor": constructor},
            }
            print(
                case["id"],
                constructor,
                "ordinary",
                full["wall_median_ms"],
                query["wall_median_ms"],
                "query graph",
                graph_query["wall_median_ms"],
                flush=True,
            )
            del problem, index, result
            gc.collect()
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del points, queries
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-probe", choices=("lbvh", "cubql"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        if args.capture_probe:
            capture_probe(args)
        else:
            benchmark(args)


if __name__ == "__main__":
    main()
