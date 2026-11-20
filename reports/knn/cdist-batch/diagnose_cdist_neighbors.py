# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect changed nearest neighbors under matrix-based cdist arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import warp as wp
import warp.geometry
from warp.examples.benchmarks.benchmark_point_query_cdist import chunked_ids, read_inputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--points", type=int, nargs="+", default=[4096, 65536])
parser.add_argument("--batches", type=int, default=32)
options = parser.parse_args()
options.output.parent.mkdir(parents=True, exist_ok=True)
wp.init()
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
records = []
for n in options.points:
    args = argparse.Namespace(archive=options.archive, batches=options.batches, n=n)
    p, q = read_inputs(args)
    with wp.ScopedDevice("cuda:0"), torch.cuda.stream(wp.stream_to_torch(wp.get_stream())):
        tp, tq = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
        pa, qa = wp.from_torch(tp, dtype=wp.vec3), wp.from_torch(tq, dtype=wp.vec3)
        f = wp.geometry.knn(pa, qa, 1)
        r = wp.geometry.knn(qa, pa, 1)
        ref = [f[1].numpy()[..., 0], r[1].numpy()[..., 0]]
        rows = min(1024, 1024**3 // (options.batches * n * 4))
        mm = [a.cpu().numpy() for a in chunked_ids(tp, tq, rows, "use_mm_for_euclid_dist")]
        row = {"points": n, "chunk_rows": rows, "directions": []}
        for source, target, a, b in ((q, p, ref[0], mm[0]), (p, q, ref[1], mm[1])):
            batches, queries = np.where(a != b)
            chosen = target[batches, b[batches, queries]].astype(np.float64)
            expected = target[batches, a[batches, queries]].astype(np.float64)
            position = source[batches, queries].astype(np.float64)
            mm_d = np.sum((position - chosen) ** 2, axis=1)
            ref_d = np.sum((position - expected) ** 2, axis=1)
            regret = mm_d - ref_d
            sampled = []
            for t in np.linspace(0, len(batches) - 1, min(16, len(batches)), dtype=int):
                bi, qi = batches[t], queries[t]
                ds = np.sum((target[bi].astype(np.float64) - source[bi, qi].astype(np.float64)) ** 2, axis=1)
                sampled.append(
                    {
                        "reference_is_min": bool(ds[a[bi, qi]] == ds.min()),
                        "mm_is_min": bool(ds[b[bi, qi]] == ds.min()),
                        "mm_gap": float(ds[b[bi, qi]] - ds.min()),
                    }
                )
            row["directions"].append(
                {
                    "changed_ids": len(batches),
                    "total_ids": options.batches * n,
                    "exact_distance_ties": int(np.count_nonzero(regret == 0)),
                    "strictly_farther": int(np.count_nonzero(regret > 0)),
                    "closer_than_reference": int(np.count_nonzero(regret < 0)),
                    "same_coordinates": int(np.count_nonzero(np.all(chosen == expected, axis=1))),
                    "max_squared_distance_regret": float(regret.max(initial=0)),
                    "sampled_changed_exhaustive": sampled,
                }
            )
        records.append(row)
        options.output.write_text(json.dumps(records, indent=2) + "\n")
        print(
            n,
            [
                (d["changed_ids"], d["strictly_farther"], d["exact_distance_ties"], d["max_squared_distance_regret"])
                for d in row["directions"]
            ],
            flush=True,
        )
        del tp, tq, pa, qa, f, r, ref, mm
    torch.cuda.empty_cache()
