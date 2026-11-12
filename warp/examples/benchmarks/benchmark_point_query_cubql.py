# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark upstream cuBQL defaults and settings selected on separate inputs."""

import argparse
import ctypes
import gc
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp._src.point_query import _init_bounds, _query_bounds, _query_codes
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy, environment, measure
from warp.examples.benchmarks.point_query_datasets import prepare

BUILDERS = ("spatial_median", "radix", "rebin_radix", "sah", "elh")
DEFAULT = {"builder": 0, "leaf": 0, "stride": 1, "block": 256, "ordered": False}
REGIMES = {
    "tiny": (1000, 32, 32),
    "medium_k1": (100000, 1024, 1),
    "medium_k32": (100000, 1024, 32),
    "million": (1000000, 8192, 11),
}


def regime(case):
    return "tiny" if case["N"] == 1000 else "million" if case["N"] >= 1000000 else f"medium_k{case['k']}"


def passed(v):
    return (
        v.get("valid_shape")
        and v.get("valid_indices_and_finite_distances")
        and all(
            v.get(k, 1) == 0
            for k in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        )
    )


class Library:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path.resolve()))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.upstream_build.argtypes = [ptr, integer, integer, integer, ptr]
        self.lib.upstream_build.restype = ptr
        self.lib.upstream_destroy.argtypes = [ptr, ptr]
        self.lib.upstream_destroy.restype = None
        self.lib.upstream_query.argtypes = [ptr, ptr, ptr, integer, integer, integer, integer, ptr, ptr, ptr, ptr, ptr]
        self.lib.upstream_query.restype = integer


class Tree:
    def __init__(self, problem, config):
        self.problem = problem
        self.handle = problem.library.lib.upstream_build(
            problem.points.ptr, len(problem.points), config["builder"], config["leaf"], wp.get_stream().cuda_stream
        )

    def __del__(self):
        if self.handle:
            self.problem.library.lib.upstream_destroy(self.handle, wp.get_stream().cuda_stream)
            self.handle = None


class Problem:
    def __init__(self, library, p, q, k):
        self.library, self.p, self.q, self.k = library, p, q, k
        self.points, self.queries = (
            wp.array(p, dtype=wp.vec3, device="cuda:0"),
            wp.array(q, dtype=wp.vec3, device="cuda:0"),
        )

    def outputs(self):
        return (
            wp.empty((len(self.q), self.k), dtype=wp.int64),
            wp.empty((len(self.q), self.k), dtype=float),
            wp.empty(len(self.q), dtype=int),
        )

    def build(self, config):
        return Tree(self, config)

    def query(self, tree, config, outputs=None):
        outputs = self.outputs() if outputs is None else outputs
        order = None
        if config["ordered"]:
            keys = wp.empty(2 * len(self.q), dtype=int)
            order = wp.empty_like(keys)
            bounds = wp.empty((2, 3), dtype=float)
            workers = min(len(self.q), 256)
            wp.launch(_init_bounds, 3, inputs=[bounds])
            wp.launch(_query_bounds, workers, inputs=[self.queries, bounds, workers])
            wp.launch(_query_codes, len(self.q), inputs=[self.queries, bounds, keys, order])
            wp.utils.radix_sort_pairs(keys, order, len(self.q))
        error = self.library.lib.upstream_query(
            tree.handle,
            self.points.ptr,
            self.queries.ptr,
            len(self.q),
            self.k,
            config["stride"],
            config["block"],
            order.ptr if order is not None else None,
            *(a.ptr for a in outputs),
            wp.get_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"cuBQL query CUDA error {error}")
        return outputs

    def call(self, config):
        tree = self.build(config)
        return tree, self.query(tree, config)

    def validate(self, outputs, checked=32, q=None):
        return accuracy(
            self.p, self.q if q is None else q, outputs[0].numpy(), outputs[1].numpy(), self.k, checked=checked
        )


def key(config):
    return json.dumps(config, sort_keys=True)


def tune(library, args):
    report = {
        "environment": environment(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("point_query_cubql_upstream.cu"))
        },
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "search": "40 builder/leaf screens; top two per objective plus default tree; 48 launch/order settings per shortlisted tree; top three per objective plus defaults remeasured with nine samples and 32 oracle rows; select separate fixed policies by N/Q/k regime",
        "seed": 20261112,
        "repeats": 5,
        "builders": BUILDERS,
        "leaf_sizes": [0, 1, 2, 4, 8, 16, 32, 64],
        "strides": [1, 2, 4, 8, 16, 32],
        "block_sizes": [32, 64, 128, 256],
        "ordered": [False, True],
        "regimes": {},
    }
    for name, (n, nq, k) in REGIMES.items():
        training, metadata = [], []
        for family, sigma in itertools.product(("sphere", "volume"), (0, 0.01)):
            p, q, meta = prepare(args.manifest.parent.parent, family, n, nq, sigma, report["seed"])
            training.append(Problem(library, p, q, k))
            metadata.append(meta)
        base = wp.point_query_config(n, nq, "cuda:0")
        stage_a = []
        for builder, leaf in itertools.product(range(len(BUILDERS)), report["leaf_sizes"]):
            config = {
                "builder": builder,
                "leaf": leaf,
                "stride": base.query_stride,
                "block": base.block_dim,
                "ordered": False,
            }
            samples = []
            for problem in training:
                bt, tree = measure(lambda problem=problem, config=config: problem.build(config), 5)
                qt, output = measure(lambda problem=problem, config=config, tree=tree: problem.query(tree, config), 5)
                valid = problem.validate(output, checked=2)
                samples.append({"build": bt, "query": qt, "validation": valid})
                del tree, output
            stage_a.append({"config": config, "samples": samples})
            print("screen", name, BUILDERS[builder], leaf, flush=True)
        valid_a = [row for row in stage_a if all(passed(s["validation"]) for s in row["samples"])]

        def score(row, phase):
            return statistics.geometric_mean(
                s["query"]["wall_median_ms"] + (s["build"]["wall_median_ms"] if phase == "one_shot" else 0)
                for s in row["samples"]
            )

        # Retain top two trees for each objective plus the upstream default tree.
        shortlist = {
            key(r["config"]): r
            for phase in ("one_shot", "reused_index_search")
            for r in sorted(valid_a, key=lambda r, phase=phase: score(r, phase))[:2]
        }
        default_tree = next(r for r in valid_a if r["config"]["builder"] == r["config"]["leaf"] == 0)
        shortlist[key(default_tree["config"])] = default_tree
        stage_b = []
        for screen in shortlist.values():
            trees = [p.build(screen["config"]) for p in training]
            for stride, block, ordered in itertools.product(
                report["strides"], report["block_sizes"], report["ordered"]
            ):
                config = dict(screen["config"], stride=stride, block=block, ordered=ordered)
                samples = []
                for i, (problem, tree) in enumerate(zip(training, trees, strict=True)):
                    qt, output = measure(
                        lambda problem=problem, tree=tree, config=config: problem.query(tree, config), 5
                    )
                    valid = problem.validate(output, checked=2)
                    samples.append({"build": screen["samples"][i]["build"], "query": qt, "validation": valid})
                stage_b.append({"config": config, "samples": samples})
            del trees
            print("launch", name, screen["config"], flush=True)
        valid_b = [r for r in stage_b if all(passed(s["validation"]) for s in r["samples"])]
        finalists = {
            key(r["config"]): r["config"]
            for phase in ("one_shot", "reused_index_search")
            for r in sorted(valid_b, key=lambda r, phase=phase: score(r, phase))[:3]
        }
        finalists[key(DEFAULT)] = DEFAULT
        final = []
        for config in finalists.values():
            samples = []
            for problem in training:
                tree = problem.build(config)
                qt, output = measure(lambda problem=problem, tree=tree, config=config: problem.query(tree, config), 9)
                qvalid = problem.validate(output)
                full_time, result = measure(lambda problem=problem, config=config: problem.call(config), 9)
                valid = problem.validate(result[1])
                samples.append(
                    {"one_shot": full_time, "reused_index_search": qt, "validation": valid, "reused_validation": qvalid}
                )
                del tree, output, result
            final.append({"config": config, "samples": samples})
        valid_final = [
            r for r in final if all(passed(s["validation"]) and passed(s["reused_validation"]) for s in r["samples"])
        ]
        selected = {
            phase: min(
                valid_final,
                key=lambda r, phase=phase: statistics.geometric_mean(s[phase]["wall_median_ms"] for s in r["samples"]),
            )["config"]
            for phase in ("one_shot", "reused_index_search")
        }
        report["regimes"][name] = {
            "training": metadata,
            "tree_screen": stage_a,
            "launch_screen": stage_b,
            "finalists": final,
            "selected": selected,
        }
        args.tuning.write_text(json.dumps(report, indent=2) + "\n")
        print("SELECTED", name, selected, flush=True)
        del training
        gc.collect()
    return report


def benchmark(library, args, tuning):
    metadata = {
        "environment": environment(),
        "repeats": 9,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "provider": {
            "exact_algorithm": True,
            "query": "unmodified cuBQL::points::findKNN",
            "builders": BUILDERS,
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        },
        "output_contract": "int64 indices and float32 Euclidean distances; direct GPU output conversion",
        "tuning_sha256": hashlib.sha256(args.tuning.read_bytes()).hexdigest(),
    }
    reports = {method: dict(metadata, method=method, cases=[]) for method in ("cubql_defaults", "cubql_tuned")}
    for case in json.loads(args.manifest.read_text())["cases"]:
        path = args.manifest.parent / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["file_sha256"]:
            raise ValueError("Dataset checksum mismatch")
        with np.load(path) as arrays:
            p, q = arrays["points"], arrays["queries"][: case["Q"]]
        problem = Problem(library, p, q, case["k"])
        for method, report in reports.items():
            selected = (
                tuning["regimes"][regime(case)]["selected"]
                if method == "cubql_tuned"
                else dict.fromkeys(("one_shot", "reused_index_search"), DEFAULT)
            )
            config = selected["reused_index_search"]
            bt, tree = measure(lambda problem=problem, config=config: problem.build(config), 9)
            qt, output = measure(lambda problem=problem, tree=tree, config=config: problem.query(tree, config), 9)
            qvalid = problem.validate(output)
            full_time, result = measure(
                lambda problem=problem, selected=selected: problem.call(selected["one_shot"]), 9
            )
            valid = problem.validate(result[1])
            outputs = problem.outputs()
            problem.query(tree, config, outputs)
            with wp.ScopedCapture() as capture:
                problem.query(tree, config, outputs)

            def replay(graph=capture.graph, outputs=outputs):
                wp.capture_launch(graph)
                return outputs

            changed_q = q + np.float32(0.013)
            problem.queries.assign(changed_q)
            replay()
            changed = problem.validate(outputs, q=changed_q)
            problem.queries.assign(q)
            gt, _ = measure(replay, 9)
            graph_valid = problem.validate(outputs)
            row = dict(
                case,
                status="ok",
                build=bt,
                reused_index_search=qt,
                one_shot=full_time,
                validation=valid,
                reused_validation=qvalid,
                configuration=selected,
                graph_reused_index_search=dict(gt, validation=graph_valid, changed_input_validation=changed),
            )
            report["cases"].append(row)
            (args.output / f"{method}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                "MEASURE",
                method,
                case["id"],
                full_time["wall_median_ms"],
                qt["wall_median_ms"],
                "checks",
                passed(valid),
                passed(qvalid),
                passed(changed),
                passed(graph_valid),
                flush=True,
            )
            del capture, tree, output, outputs, result
        del problem
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--reuse-tuning", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.tuning.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    torch.set_grad_enabled(False)
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    stream = wp.Stream("cuda:0")
    library = Library(args.library)
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        tuning = json.loads(args.tuning.read_text()) if args.reuse_tuning else tune(library, args)
        benchmark(library, args, tuning)


if __name__ == "__main__":
    main()
