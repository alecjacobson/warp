# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate unseen seeds and a nested-query sweep for the hybrid comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from warp.examples.benchmarks.point_query_datasets import FAMILIES, SOURCES, prepare, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../data/knn-comparison/hybrid-holdout"))
    parser.add_argument("--source-dir", type=Path, default=Path("../data/knn-comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"suite": "hybrid-holdout", "sources": SOURCES, "numpy_version": np.__version__, "cases": []}
    for seed in (7801, 9109):
        for family in FAMILIES:
            for sigma in (0.0, 0.01):
                sizes = [(100_000, 1024, 32)]
                if family in ("sphere", "volume"):
                    sizes.append((1_000_000, 100_000, 11))
                for n, nq, k in sizes:
                    p, q, meta = prepare(args.source_dir, family, n, nq, sigma, seed)
                    filename = f"{family}-n{n}-q{nq}-sigma{sigma:g}-seed{seed}.npz"
                    np.savez_compressed(args.output / filename, points=p, queries=q)
                    digest = sha256(args.output / filename)
                    for prefix in (1024, 8192, 100_000) if n == 1_000_000 else (1024,):
                        case = dict(meta, Q=prefix, k=k, file=filename, file_sha256=digest)
                        case["queries_sha256"] = hashlib.sha256(q[:prefix].tobytes()).hexdigest()
                        case["id"] = f"{family}-n{n}-q{prefix}-sigma{sigma:g}-seed{seed}-k{k}"
                        result["cases"].append(case)
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
