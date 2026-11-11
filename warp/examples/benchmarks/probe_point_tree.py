# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count traversal work for axis-scaled and isotropic Morton point trees; do not time these kernels."""

import argparse
import json
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.point_query_hybrid import make_hybrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../data/knn-comparison/pilot/manifest.json"))
    parser.add_argument("--case", default="sphere-n1000000-q8192-sigma0-k11")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = next(c for c in json.loads(args.manifest.read_text())["cases"] if c["id"] == args.case)
    k = case["k"]
    if k < 3:
        raise ValueError("The diagnostic requires k >= 3 for its three counter slots")
    d = np.load(args.manifest.parent / case["file"])
    wp.init()
    p = wp.array(d["points"], dtype=wp.vec3, device="cuda:0")
    q = wp.array(d["queries"][: case["Q"]], dtype=wp.vec3, device="cuda:0")
    other = wp.clone(p)
    order = wp.array(np.arange(len(q), dtype=np.int32), device="cuda:0")
    ids = wp.empty((len(q), k), dtype=wp.int64, device="cuda:0")
    ds = wp.empty((len(q), k), dtype=float, device="cuda:0")
    result = {
        "case": case,
        "counter_names": ["traversal_iterations", "point_distances", "candidate_insert_calls"],
        "variants": {},
    }
    for leaf in (4, 8, 32):
        for isotropic, upper in ((False, other), (True, p)):
            tree = wp.Bvh(p, upper, constructor="lbvh", leaf_size=leaf)
            kernel = make_hybrid(k, diagnostics=True)
            wp.launch(kernel, len(q), inputs=[tree.id, p, p, order, q, order, ids, ds])
            counters = ids.numpy()[:, :3]
            name = f"isotropic{int(isotropic)}-leaf{leaf}"
            result["variants"][name] = {
                "mean": counters.mean(0).tolist(),
                "p99": np.quantile(counters, 0.99, axis=0).tolist(),
                "max": counters.max(0).tolist(),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
