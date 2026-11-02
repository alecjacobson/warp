# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supplementary query-tile reuse, motion schedules, and allocation measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query import cloud, load_physicsnemo, measure, validate
from warp.examples.benchmarks.point_query_common import make_api_query, make_tiled, make_tiled_batched


@wp.kernel(enable_backward=False)
def move(points: wp.array[wp.vec3], velocity: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] += velocity[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    device = wp.get_device("cuda:0")
    results = {"device": device.name, "tiles": [], "motion": [], "memory": []}
    for n, nq, k in ((32, 35, 8), (1000, 1003, 8), (100_000, 1, 8)):
        p, q = cloud("volume", n, nq, 0.0, 931)
        pw, qw = wp.array(p, dtype=wp.vec3), wp.array(q, dtype=wp.vec3)
        ex = wp.full(nq, -1, dtype=int)
        oi, od = wp.empty((nq, k), dtype=int), wp.empty((nq, k), dtype=float)
        counts = wp.empty(nq, dtype=int)
        for bp, block in ((64, 32), (128, 64), (256, 128)):
            for bq in (1, 2, 4):
                for partitions in (1, 32):
                    tiled, merge = make_tiled(k, bp) if bq == 1 else make_tiled_batched(k, bp, bq)
                    pi, pd = wp.empty((nq, partitions, k), dtype=int), wp.empty((nq, partitions, k), dtype=float)

                    def execute(
                        tiled=tiled,
                        merge=merge,
                        bq=bq,
                        partitions=partitions,
                        pi=pi,
                        pd=pd,
                        block=block,
                        nq=nq,
                        pw=pw,
                        qw=qw,
                        ex=ex,
                        oi=oi,
                        od=od,
                        counts=counts,
                    ):
                        wp.launch_tiled(
                            tiled,
                            dim=((nq + bq - 1) // bq, partitions),
                            inputs=[pw, qw, np.inf, ex, partitions, pi, pd],
                            block_dim=block,
                        )
                        wp.launch(merge, nq, inputs=[pi, pd, oi, od, counts])

                    timing, _ = measure(execute, device, 7, graph=True)
                    check = validate(p, q, oi.numpy(), k)
                    if check["float32_mismatch_rows"]:
                        raise AssertionError(check)
                    results["tiles"].append(
                        {
                            "N": n,
                            "Q": nq,
                            "k": k,
                            "Bp": bp,
                            "Bq": bq,
                            "block": block,
                            "partitions": partitions,
                            "scratch_bytes": nq * partitions * k * 8,
                            "query": timing,
                            "validation": check,
                        }
                    )
    p, q = cloud("volume", 100_000, 8192, 0.0, 931)
    initial = wp.array(p, dtype=wp.vec3)
    points = wp.clone(initial)
    queries = wp.array(q, dtype=wp.vec3)
    rng = np.random.default_rng(91)
    velocity = wp.array(rng.normal(scale=0.002, size=p.shape).astype(np.float32), dtype=wp.vec3)
    bvh = wp.Bvh(points, points, leaf_size=4)
    ex = wp.full(len(q), -1, dtype=int)
    oi, od, counts = wp.empty((len(q), 11), dtype=int), wp.empty((len(q), 11), dtype=float), wp.empty(len(q), dtype=int)
    kernel = make_api_query(11)
    for rebuild_every in (0, 5, 1):

        def cycle(rebuild_every=rebuild_every):
            wp.copy(points, initial)
            for step in range(10):
                wp.launch(move, len(p), inputs=[points, velocity])
                if rebuild_every and (step + 1) % rebuild_every == 0:
                    bvh.rebuild()
                else:
                    bvh.refit()
                wp.launch(kernel, len(q), inputs=[bvh.id, points, queries, np.inf, ex, oi, od, counts])

        # Capture reset, ten updates, maintenance and queries; include all work.
        timing, _ = measure(cycle, device, 7, graph=True)
        final = points.numpy()
        check = validate(final, q, oi.numpy(), 11)
        if check["float32_mismatch_rows"]:
            raise AssertionError(check)
        results["motion"].append(
            {
                "N": len(p),
                "Q": len(q),
                "k": 11,
                "steps": 10,
                "rebuild_every": rebuild_every,
                "velocity_sigma": 0.002,
                "cycle": timing,
                "validation": check,
            }
        )
    # Allocation tracking is outside timed regions. Reports include native BVH
    # buffers and construction scratch, not just Python-owned arrays.
    for n in (100_000, 1_000_000):
        records = []
        with wp.ScopedMemoryTracker("point_query", report_func=records.append) as tracker:
            tracker.clear()
            pp, qq = cloud("volume", n, 8192, 0.0, 931)
            ppw, qqw = wp.array(pp, dtype=wp.vec3), wp.array(qq, dtype=wp.vec3)
            tree = wp.Bvh(ppw, ppw, leaf_size=4)
            ix = wp.empty((8192, 11), dtype=int)
            ds = wp.empty((8192, 11), dtype=float)
            ct = wp.empty(8192, dtype=int)
            wp.launch(kernel, 8192, inputs=[tree.id, ppw, qqw, np.inf, ex, ix, ds, ct])
            wp.synchronize_device(device)
        results["memory"].append({"backend": "Warp LBVH", "N": n, "Q": 8192, "k": 11, "allocation_report": records[0]})
    baseline, _ = load_physicsnemo(args.output.parent / "physicsnemo")
    import torch  # noqa: PLC0415 - optional benchmark dependency

    for n in (100_000, 1_000_000):
        pp, qq = cloud("volume", n, 8192, 0.0, 931)
        tp, tq = torch.tensor(pp, device="cuda"), torch.tensor(qq, device="cuda")
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = baseline.knn_impl(tp, tq, 11)
        torch.cuda.synchronize()
        results["memory"].append(
            {
                "backend": "PhysicsNeMo",
                "N": n,
                "Q": 8192,
                "k": 11,
                "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - base,
                "input_bytes": tp.numel() * tp.element_size() + tq.numel() * tq.element_size(),
            }
        )
        del output
    args.output.write_text(json.dumps(results, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
