# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recheck the original point-query matrix after profile-guided optimizations.

The configured comparison uses leaf size 32 and one active query per warp for
Q <= 32, or leaf size 32 plus Morton ordering for million-point batched clouds.
Other cases keep the original leaf size 4 and natural query order. These are
explicit experimental configurations, not an automatic public API dispatch rule.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query import cloud, load_physicsnemo, measure, scan_points, validate
from warp.examples.benchmarks.point_query_common import make_api_query, make_ordered_query, query_codes


def run_variant(points, queries, excludes, k, leaf, stride, ordered, repeats):
    device = points.device
    nq = len(queries)
    kernel = make_ordered_query(k, True) if ordered else make_api_query(k, True, stride)
    block = 32 if stride > 1 else 256
    build, tree = measure(lambda: wp.Bvh(points, points, constructor="lbvh", leaf_size=leaf), device, repeats)
    ids = wp.empty((nq, k), dtype=wp.int64, device=device)
    ds = wp.empty((nq, k), dtype=float, device=device)
    counts = wp.empty(nq, dtype=int, device=device)
    keys = wp.empty(2 * nq, dtype=int, device=device) if ordered else None
    order = wp.empty_like(keys) if ordered else None

    def plan(tree=tree, keys=keys, order=order):
        if ordered:
            wp.launch(query_codes, nq, inputs=[tree.id, queries, keys, order], device=device)
            wp.utils.radix_sort_pairs(keys, order, nq)

    def query():
        inputs = [tree.id, points, queries, np.inf, excludes] + ([order] if ordered else []) + [ids, ds, counts]
        wp.launch(kernel, nq * stride, inputs=inputs, block_dim=block, device=device)

    planning, _ = measure(plan, device, repeats)
    timing, _ = measure(query, device, repeats, graph=True)

    def one_shot():
        new_tree = wp.Bvh(points, points, constructor="lbvh", leaf_size=leaf)
        new_ids = wp.empty((nq, k), dtype=wp.int64, device=device)
        new_ds = wp.empty((nq, k), dtype=float, device=device)
        new_counts = wp.empty(nq, dtype=int, device=device)
        new_keys = wp.empty(2 * nq, dtype=int, device=device) if ordered else None
        new_order = wp.empty_like(new_keys) if ordered else None
        plan(new_tree, new_keys, new_order)
        inputs = (
            [new_tree.id, points, queries, np.inf, excludes]
            + ([new_order] if ordered else [])
            + [new_ids, new_ds, new_counts]
        )
        wp.launch(kernel, nq * stride, inputs=inputs, block_dim=block, device=device)
        return new_tree, new_ids, new_ds, new_counts, new_keys, new_order

    one, result = measure(one_shot, device, repeats)

    def rebuild_query():
        tree.rebuild()
        plan()
        query()

    rebuild_graph, _ = measure(rebuild_query, device, repeats, graph=True)
    return (
        {
            "leaf_size": leaf,
            "query_stride": stride,
            "ordered": ordered,
            "build": build,
            "query_plan": planning if ordered else None,
            "query": timing,
            "one_shot": one,
            "rebuild_plan_query_graph": rebuild_graph,
        },
        result[1].numpy(),
        ids.numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("reports/knn/archive/baseline/results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--losses-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--mempool-release-threshold-mib",
        type=int,
        help="Retain freed CUDA pool memory across synchronization (MiB); omit to preserve the current policy",
    )
    args = parser.parse_args()
    if args.mempool_release_threshold_mib is not None and args.mempool_release_threshold_mib < 0:
        parser.error("The memory-pool release threshold must be nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device("cuda:0")
    initial_threshold = wp.get_mempool_release_threshold(device)
    if args.mempool_release_threshold_mib is not None:
        wp.set_mempool_release_threshold(device, args.mempool_release_threshold_mib * 1024**2)
    original = json.loads(args.original.read_text())
    baseline, sha = load_physicsnemo(args.output.parent / "physicsnemo")
    import torch  # noqa: PLC0415 - optional comparison dependency

    scan = scan_points(args.scan)
    report = {
        "device": device.name,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "original_results_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
        "physicsnemo_source_sha256": sha,
        "native_bvh_sha256": hashlib.sha256(Path("warp/native/bvh.cu").read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            Path("warp/examples/benchmarks/point_query_common.py").read_bytes()
        ).hexdigest(),
        "repeats": args.repeats,
        "mempool_release_threshold_bytes": wp.get_mempool_release_threshold(device),
        "initial_mempool_release_threshold_bytes": initial_threshold,
        "torch_allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "cases": [],
    }
    for case_id, old in enumerate(original["cases"], 1):
        old_pr = old["variants"]["physicsnemo"]["one_shot"]["median_ms"]
        lost = old["one_shot"]["median_ms"] > old_pr
        if args.losses_only and not lost:
            continue
        n, nq, k = (old[key] for key in ("N", "Q", "k"))
        p, q = cloud(old["distribution"], n, nq, old["gaussian_sigma"], old["seed"], scan)
        points = wp.array(p, dtype=wp.vec3, device=device)
        queries = wp.array(q, dtype=wp.vec3, device=device)
        excludes = wp.full(nq, -1, dtype=int, device=device)
        row = {key: old[key] for key in ("N", "Q", "k", "distribution", "gaussian_sigma", "seed")}
        row.update(case_id=case_id, originally_lost=lost, original_one_shot=old["one_shot"])
        print(f"Case {case_id}: {old['distribution']} sigma={old['gaussian_sigma']} N={n} Q={nq} k={k}", flush=True)
        for name, leaf, stride, ordered in (
            ("same_configuration", 4, 1, False),
            ("configured", 32 if nq <= 32 or n >= 1_000_000 else 4, 32 if nq <= 32 else 1, nq > 32 and n >= 1_000_000),
        ):
            result, ids, graph_ids = run_variant(points, queries, excludes, k, leaf, stride, ordered, args.repeats)
            result["validation"] = validate(p, q, ids, k)
            result["graph_validation"] = validate(p, q, graph_ids, k)
            if result["validation"]["float32_mismatch_rows"] or result["graph_validation"]["float32_mismatch_rows"]:
                raise AssertionError(result)
            row[name] = result
        tp, tq = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
        one, result = measure(lambda tp=tp, tq=tq, k=k: baseline.knn_impl(tp, tq, k), device, args.repeats)
        check = validate(p, q, result[0].cpu().numpy(), k)
        if check["float32_mismatch_rows"]:
            raise AssertionError(check)
        row["physicsnemo"] = {"one_shot": one, "validation": check}
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(
            f"  prior={old['one_shot']['median_ms']:.4f} "
            f"same={row['same_configuration']['one_shot']['median_ms']:.4f} "
            f"configured={row['configured']['one_shot']['median_ms']:.4f} "
            f"PR={one['median_ms']:.4f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
