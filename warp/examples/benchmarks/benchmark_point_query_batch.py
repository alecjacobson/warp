# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the Warp-only batch API with configured scalar kernels; no Torch import."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy
from warp.examples.benchmarks.point_query_common import make_api_query, make_ordered_query, query_codes


def measure(fn, repeats):
    for _ in range(2):
        result = fn()
    wp.synchronize_device("cuda:0")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        wp.synchronize_device("cuda:0")
        samples.append((time.perf_counter() - start) * 1000)
    return {"wall_median_ms": statistics.median(samples), "wall_samples_ms": samples}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in (
                "warp/_src/point_query.py",
                "warp/native/point_query.h",
                "warp/native/bvh.cu",
                "warp/examples/benchmarks/benchmark_point_query_batch.py",
            )
        },
        "repeats": args.repeats,
        "timing": "synchronized wall; build, output allocation and query; float32 squared distances/int32 IDs",
        "device": wp.get_device("cuda:0").name,
        "sm_count": wp.get_device("cuda:0").sm_count,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in json.loads(args.manifest.read_text())["cases"]:
        if args.filter and args.filter not in case["id"]:
            continue
        path = args.manifest.parent / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["file_sha256"]:
            raise ValueError("Dataset checksum mismatch")
        data = np.load(path)
        p, q = data["points"], data["queries"][: case["Q"]]
        points, queries = wp.array(p, dtype=wp.vec3, device="cuda:0"), wp.array(q, dtype=wp.vec3, device="cuda:0")
        config = wp.point_query_config(len(p), len(q), points.device)
        excludes = wp.full(len(q), -1, dtype=int, device="cuda:0")
        k = case["k"]

        def scalar(index=None, dense=False, config=config, q=q, k=k, points=points, queries=queries, excludes=excludes):
            leaf = 4 if dense else config.leaf_size
            stride = 1 if dense else config.query_stride
            ordered = False if dense else config.ordered
            if index is None:
                index = wp.Bvh(points, points, leaf_size=leaf)
            ids = wp.empty((len(q), k), dtype=int, device="cuda:0")
            ds = wp.empty((len(q), k), dtype=float, device="cuda:0")
            counts = wp.empty(len(q), dtype=int, device="cuda:0")
            extra = []
            if ordered:
                keys = wp.empty(2 * len(q), dtype=int, device="cuda:0")
                order = wp.empty_like(keys)
                wp.launch(query_codes, len(q), inputs=[index.id, queries, keys, order])
                wp.utils.radix_sort_pairs(keys, order, len(q))
                extra = [order]
            kernel = make_ordered_query(k) if ordered else make_api_query(k, query_stride=stride)
            wp.launch(
                kernel,
                len(q) * stride,
                inputs=[index.id, points, queries, wp.inf, excludes, *extra, ids, ds, counts],
                block_dim=32 if stride > 1 else 256,
            )
            return index, ids, ds, counts

        row = {"case": case, "config": vars(config), "variants": {}}
        for name, fn in [
            ("batch", lambda points=points, queries=queries, k=k: wp.point_query_knn_batch(points, queries, k)),
            ("configured_scalar", scalar),
            ("dense_scalar", lambda scalar=scalar: scalar(dense=True)),
        ]:
            timing, result = measure(fn, args.repeats)
            ids, ds = result[1].numpy(), result[2].numpy()
            valid = accuracy(p, q, ids, np.sqrt(ds), k)
            index = result[0]
            search = (
                (
                    lambda points=points, queries=queries, k=k, index=index: wp.point_query_knn_batch(
                        points, queries, k, bvh=index
                    )
                )
                if name == "batch"
                else (lambda scalar=scalar, index=index, name=name: scalar(index, name == "dense_scalar"))
            )
            query_timing, query_result = measure(search, args.repeats)
            query_valid = accuracy(p, q, query_result[1].numpy(), np.sqrt(query_result[2].numpy()), k)
            row["variants"][name] = {
                "one_shot": timing,
                "reused_index_search": query_timing,
                "accuracy": valid,
                "reused_accuracy": query_valid,
            }
            print(
                case["id"],
                name,
                round(timing["wall_median_ms"], 3),
                round(query_timing["wall_median_ms"], 3),
                valid["distance_rank_mismatch_rows"],
                flush=True,
            )
        report["cases"].append(row)
        report["torch_imported"] = "torch" in sys.modules
        if report["torch_imported"]:
            raise RuntimeError("Warp-only benchmark imported Torch")
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
