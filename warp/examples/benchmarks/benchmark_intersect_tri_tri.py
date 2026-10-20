# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native and pure Warp triangle intersection using warmed CUDA graphs.

Run with ``uv run -m warp.examples.benchmarks.benchmark_intersect_tri_tri``.
Timings exclude compilation, data generation, host transfers, and broad phase.
Each reported sample measures a graph of repeated kernel launches with CUDA
events. Native/pure timing order alternates between samples. Outputs must match
before timing; performance is reported, never asserted.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

import warp as wp
import warp.geometry
from warp.examples.geometry.example_mesh_intersect_hash import intersect_models, make_box


def make_kernel(pure):
    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(points: wp.array2d[Any], result: wp.array[int]):
        i = wp.tid()
        if wp.static(pure):
            result[i] = wp.geometry.intersect_tri_tri(
                points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]
            )
        else:
            result[i] = wp.intersect_tri_tri(
                points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]
            )

    return evaluate


def make_workloads(count, device):
    rng = np.random.default_rng(20261020)
    random = rng.uniform(-1.0, 1.0, (count, 6, 3))
    separated = random.copy()
    separated[:, 3:, 0] += 4.0
    coplanar = random.copy()
    coplanar[:, :, 2] = 0.0
    # Real candidate pairs from the rendered example, shuffled and repeated.
    a = make_box(subdivisions=6)
    b = make_box(subdivisions=6, translation=(0.7, 0.3, 0.2), angle=0.31)
    candidates, _ = intersect_models(a, b, device)
    pairs = np.argwhere(candidates)
    pairs = pairs[rng.permutation(len(pairs))]
    mesh = np.concatenate((a[0][a[1][pairs[:, 0]]], b[0][b[1][pairs[:, 1]]]), axis=1)
    mesh = mesh[np.arange(count) % len(mesh)]
    return {"random": random, "separated": separated, "coplanar": coplanar, "mesh_candidates": mesh}


def benchmark(points, kernels, dtype, device, repeats, samples):
    data = wp.array(points, dtype=dtype, device=device)
    outputs = [wp.empty(len(points), dtype=int, device=device) for _ in kernels]
    graphs = []
    for kernel, out in zip(kernels, outputs, strict=True):
        for _ in range(5):
            wp.launch(kernel, dim=len(points), inputs=[data, out], device=device)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(repeats):
                wp.launch(kernel, dim=len(points), inputs=[data, out], device=device)
        graphs.append(capture.graph)
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(outputs[0].numpy(), outputs[1].numpy())
    hits = int(np.count_nonzero(outputs[0].numpy()))
    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    times = [[], []]
    with wp.ScopedDevice(device):
        for sample in range(samples):
            for index in (0, 1) if sample % 2 == 0 else (1, 0):
                wp.record_event(start)
                wp.capture_launch(graphs[index])
                wp.record_event(end)
                times[index].append(wp.get_event_elapsed_time(start, end) / repeats)
    return {
        "pairs": len(points),
        "hits": hits,
        "native_ms": median(times[0]),
        "pure_ms": median(times[1]),
        "speedup": median(times[0]) / median(times[1]),
        "native_samples_ms": times[0],
        "pure_samples_ms": times[1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=262144)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("outputs/intersect_tri_tri_benchmark.json"))
    args = parser.parse_args()
    if min(args.pairs, args.repeats, args.samples) < 1:
        parser.error("--pairs, --repeats, and --samples must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("CUDA is required for the event/graph benchmark")
    kernels = (make_kernel(False), make_kernel(True))
    workloads = make_workloads(args.pairs, device)
    results = {}
    for dtype in (wp.vec3, wp.vec3d):
        for name, points in workloads.items():
            key = f"{dtype._wp_scalar_type_.__name__}/{name}"
            result = benchmark(points, kernels, dtype, device, args.repeats, args.samples)
            results[key] = result
            print(
                f"{key}: native {result['native_ms']:.4f} ms, pure {result['pure_ms']:.4f} ms, "
                f"speedup {result['speedup']:.3f}x"
            )
    report = {
        "device": device.name,
        "warp_version": wp.__version__,
        "pairs": args.pairs,
        "repeats": args.repeats,
        "samples": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
