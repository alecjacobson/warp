# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose case 25 with alternating CUDA pool policies and paired implementations.

Run with --output DIRECTORY. Optional --trace captures additional samples with
CUPTI; these are kept separate from unprofiled timings. The release threshold is
restored on exit. Inputs, query algorithm, and one-shot allocation semantics stay
fixed while the pool alternates between zero and the requested retention policy.
"""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query import cloud, load_physicsnemo, validate
from warp.examples.benchmarks.point_query_common import make_ordered_query, query_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--retained-mib", type=int, default=512)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-repeats", type=int, default=300)
    args = parser.parse_args()
    if min(args.repeats, args.retained_mib, args.trace_repeats) <= 0:
        parser.error("Repeat counts and retained MiB must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device("cuda:0")
    baseline, sha = load_physicsnemo(args.output / "physicsnemo")
    import torch  # noqa: PLC0415 - optional comparison/profiling dependency

    p, q = cloud("volume", 1_000_000, 100_000, 0.0, 531)
    points = wp.array(p, dtype=wp.vec3, device=device)
    queries = wp.array(q, dtype=wp.vec3, device=device)
    excludes = wp.full(len(q), -1, dtype=int, device=device)
    tp, tq = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
    kernel = make_ordered_query(1, True)
    parts = {}

    def one_shot():
        t0 = time.perf_counter()
        tree = wp.Bvh(points, points, constructor="lbvh", leaf_size=32)
        t1 = time.perf_counter()
        ids = wp.empty((len(q), 1), dtype=wp.int64, device=device)
        distances = wp.empty((len(q), 1), dtype=float, device=device)
        counts = wp.empty(len(q), dtype=int, device=device)
        keys = wp.empty(2 * len(q), dtype=int, device=device)
        order = wp.empty_like(keys)
        t2 = time.perf_counter()
        wp.launch(query_codes, len(q), inputs=[tree.id, queries, keys, order], device=device)
        wp.utils.radix_sort_pairs(keys, order, len(q))
        t3 = time.perf_counter()
        wp.launch(
            kernel,
            len(q),
            inputs=[tree.id, points, queries, np.inf, excludes, order, ids, distances, counts],
            device=device,
        )
        t4 = time.perf_counter()
        parts.update(
            build_submit_ms=1000 * (t1 - t0),
            outputs_submit_ms=1000 * (t2 - t1),
            plan_submit_ms=1000 * (t3 - t2),
            query_submit_ms=1000 * (t4 - t3),
        )
        return tree, ids, distances, counts, keys, order

    functions = {"warp": one_shot, "physicsnemo": lambda: baseline.knn_impl(tp, tq, 1)}
    results = {}
    state = {}
    gc_start = 0.0
    report = {
        "device": device.name,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_bvh_sha256": hashlib.sha256(Path("warp/native/bvh.cu").read_bytes()).hexdigest(),
        "physicsnemo_source_sha256": sha,
        "initial_mempool_release_threshold_bytes": wp.get_mempool_release_threshold(device),
        "torch_allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "repeats": args.repeats,
        "trace_repeats": args.trace_repeats if args.trace else 0,
        "gc_events": [],
        "phases": [],
    }

    def on_gc(phase, info):
        nonlocal gc_start
        if phase == "start":
            gc_start = time.perf_counter()
        else:
            report["gc_events"].append({**state, "ms": 1000 * (time.perf_counter() - gc_start), **info})

    begin = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)

    def sample(name):
        wp.synchronize_device(device)
        start = time.perf_counter()
        wp.record_event(begin)
        t1 = time.perf_counter()
        new = functions[name]()
        t2 = time.perf_counter()
        # Include retirement of the previous output, as in the original helper.
        results[name] = new
        t3 = time.perf_counter()
        wp.record_event(end)
        t4 = time.perf_counter()
        wp.synchronize_device(device)
        t5 = time.perf_counter()
        row = {
            **state,
            "gpu_ms": wp.get_event_elapsed_time(begin, end),
            "wall_ms": 1000 * (t5 - start),
            "submit_ms": 1000 * (t2 - t1),
            "retire_ms": 1000 * (t3 - t2),
            "sync_ms": 1000 * (t5 - t4),
        }
        if name == "warp":
            row.update(parts)
        return row

    gc.callbacks.append(on_gc)
    try:
        for phase, mib in enumerate((0, args.retained_mib, 0, args.retained_mib)):
            wp.set_mempool_release_threshold(device, mib * 1024**2)
            state.update(phase=phase, sample=-1, profiled=False)
            for name, fn in functions.items():
                state["implementation"] = name
                for _ in range(3):
                    results[name] = fn()
            wp.synchronize_device(device)
            row = {"phase": phase, "release_threshold_mib": mib, "samples": [], "trace_samples": []}
            report["phases"].append(row)
            for i in range(args.repeats):
                names = ("warp", "physicsnemo") if i % 2 == 0 else ("physicsnemo", "warp")
                for name in names:
                    state.update(sample=i, implementation=name, profiled=False)
                    row["samples"].append(sample(name))
            if args.trace:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as profiler:
                    for i in range(args.trace_repeats):
                        state.update(sample=i, implementation="warp", profiled=True)
                        with torch.profiler.record_function(f"sample-{i}"):
                            row["trace_samples"].append(sample("warp"))
                profiler.export_chrome_trace(str(args.output / f"phase-{phase}-trace.json"))
            state["sample"] = -1
            row["validation"] = {
                "warp": validate(p, q, results["warp"][1].numpy(), 1),
                "physicsnemo": validate(p, q, results["physicsnemo"][0].cpu().numpy(), 1),
            }
            if any(check["float32_mismatch_rows"] for check in row["validation"].values()):
                raise AssertionError(row["validation"])
            row["warp_pool_used_bytes"] = wp.get_mempool_used_mem_current(device)
            row["torch_reserved_bytes"] = torch.cuda.memory_reserved()
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
            medians = {
                name: statistics.median(s["gpu_ms"] for s in row["samples"] if s["implementation"] == name)
                for name in functions
            }
            print(f"Phase {phase}, threshold {mib} MiB: {medians}", flush=True)
    finally:
        gc.callbacks.remove(on_gc)
        wp.set_mempool_release_threshold(device, report["initial_mempool_release_threshold_bytes"])


if __name__ == "__main__":
    main()
