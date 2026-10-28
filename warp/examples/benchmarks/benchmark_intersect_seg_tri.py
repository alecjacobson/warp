# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pure Warp segment intersection with native ray-triangle routines.

Native adapters call the existing float32 implementations in intersect.h with
direction p1-p0 and restrict the ray parameter to [0, 1]. They do not add coplanar
handling, tolerance, or backface correction. Differences are reported, not hidden.
CUDA graph timings exclude compilation, transfers, and broad phase; they measure
boolean output only, letting the compiler discard unused barycentric/normal outputs.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

import warp as wp
import warp.geometry
from warp.examples.geometry.example_mesh_intersect_hash import intersect_models, make_box, mesh_edges


@wp.func_native("""
float t, u, v, sign;
bool hit = wp::intersect_ray_tri_woop(p0, p1-p0, a, b, c, t, u, v, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_woop(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


@wp.func_native("""
float t, u, v, w, sign;
bool hit = wp::intersect_ray_tri_moller(p0, p1-p0, a, b, c, t, u, v, w, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_moller(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


@wp.func_native("""
float t, u, v, w, sign;
bool hit = wp::intersect_ray_tri_rtcd(p0, p1-p0, a, b, c, t, u, v, w, sign, nullptr);
return hit && t >= 0.0f && t <= 1.0f;
""")
def native_rtcd(p0: wp.vec3, p1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool: ...


def make_kernel(predicate):
    @wp.kernel(module="unique", enable_backward=False)
    def evaluate(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
        i = wp.tid()
        hits[i] = predicate(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])

    return evaluate


def make_workloads(count, device):
    rng = np.random.default_rng(20261028)
    random = rng.uniform(-1, 1, (count, 5, 3)).astype(np.float32)
    separated = random.copy()
    separated[:, 2:, 0] += 4.0
    coplanar = random.copy()
    coplanar[:, :, 2] = 0.0
    a = mesh_edges(make_box(subdivisions=6))
    b = make_box(subdivisions=6, translation=(0.7, 0.3, 0.2), angle=0.31)
    # Keep this throughput workload on geometric hash candidates only.
    candidates, _ = intersect_models(a, b, device, tol=0.0)
    pairs = np.argwhere(candidates)
    pairs = pairs[rng.permutation(len(pairs))]
    mesh = np.concatenate((a[0][a[1][pairs[:, 0]]], b[0][b[1][pairs[:, 1]]]), axis=1)
    mesh = mesh[np.arange(count) % len(mesh)]
    return {"random": random, "separated": separated, "coplanar": coplanar, "mesh_candidates": mesh}


def benchmark(points, kernels, device, repeats, samples):
    data = wp.array(points, dtype=wp.vec3, device=device)
    graphs, outputs, device_outputs = [], [], []
    for kernel in kernels.values():
        out = wp.empty(len(points), dtype=bool, device=device)
        for _ in range(5):
            wp.launch(kernel, dim=len(points), inputs=[data, out], device=device)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(repeats):
                wp.launch(kernel, dim=len(points), inputs=[data, out], device=device)
        graphs.append(capture.graph)
        # Captured launches retain pointers, so keep the allocations alive until
        # all graphs have finished replaying.
        device_outputs.append(out)
        outputs.append(out.numpy())
    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    times = [[] for _ in graphs]
    with wp.ScopedDevice(device):
        for sample in range(samples):
            # Rotate measurement order to distribute clock/temperature drift.
            for offset in range(len(graphs)):
                index = (sample + offset) % len(graphs)
                wp.record_event(start)
                wp.capture_launch(graphs[index])
                wp.record_event(end)
                times[index].append(wp.get_event_elapsed_time(start, end) / repeats)
    result = {}
    for index, (name, kernel) in enumerate(kernels.items()):
        np.testing.assert_array_equal(device_outputs[index].numpy(), outputs[index])
        result[name] = {
            "kernel_properties": wp.get_cuda_kernel_properties(kernel, device=device),
            "median_ms": median(times[index]),
            "samples_ms": times[index],
            "hits": int(outputs[index].sum()),
            "disagreements_with_pure": int(np.count_nonzero(outputs[index] != outputs[0])),
            "pure_only": int(np.count_nonzero(outputs[0] & ~outputs[index])),
            "native_only": int(np.count_nonzero(~outputs[0] & outputs[index])),
            "speedup_of_pure": median(times[index]) / median(times[0]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=262144)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("outputs/intersect_seg_tri_benchmark.json"))
    args = parser.parse_args()
    if min(args.pairs, args.repeats, args.samples) < 1:
        parser.error("--pairs, --repeats, and --samples must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("CUDA is required for the event/graph benchmark")
    kernels = {
        name: make_kernel(predicate)
        for name, predicate in (
            ("pure", wp.geometry.intersect_seg_tri),
            ("woop", native_woop),
            ("moller", native_moller),
            ("rtcd", native_rtcd),
        )
    }
    results = {}
    for name, points in make_workloads(args.pairs, device).items():
        result = benchmark(points, kernels, device, args.repeats, args.samples)
        results[name] = result
        for routine, stats in result.items():
            print(
                f"{name}/{routine}: {stats['median_ms'] * 1000:.2f} us, "
                f"{stats['hits']} hits, {stats['disagreements_with_pure']} disagreements, "
                f"pure speedup {stats['speedup_of_pure']:.3f}x"
            )
    report = {
        "device": device.name,
        "warp_version": wp.__version__,
        "precision": "float32",
        "pairs": args.pairs,
        "repeats": args.repeats,
        "samples": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
