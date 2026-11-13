# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the restored kNN comparison sequentially with isolated provider processes.

Dependencies, native adapters, and datasets must already exist in the workspace.
The default runs one workload as an installation check. Use --full-matrix for
the report's complete 40-workload, nine-sample experiment.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
EXTERNAL = (
    "physicsnemo",
    "cubql",
    "cupy_kdtree",
    "keops",
    "faiss_flat",
    "cuvs_brute_force",
    "cuvs_unexpanded",
    "cuvs_minkowski",
    "pytorch3d",
)


def main():
    """Save every process outcome and command, including capture failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--only", nargs="+", help="Select named jobs to rerun after diagnosing a failure")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = workspace / ("run" if args.full_matrix else "smoke")
    output.mkdir(parents=True, exist_ok=True)
    manifest = workspace / "data/pilot/manifest.json"
    libraries = workspace / "libraries"
    dependencies = [
        "--cubql-library",
        str(libraries / "cubql_custom.so"),
        "--pytorch3d-path",
        str(workspace / "third_party/pytorch3d"),
        "--faiss-path",
        str(workspace / "third_party/faiss/build-knn/faiss/python"),
    ]
    common = ["--manifest", str(manifest)]
    limit = [] if args.full_matrix else ["--limit", "1"]
    repeats = ["--repeats", "9" if args.full_matrix else "2"]
    jobs = [
        (
            "warp-public",
            "benchmark_point_query_replay",
            [
                *common,
                "--output",
                str(output / "warp-public.json"),
                *limit,
                *repeats,
                "--methods",
                "warp_scalar",
                "warp_batch",
                "warp_bvh_cubql",
                "warp_cubql_reuse",
                "warp_cubql_build",
            ],
        ),
        (
            "cubql-upstream",
            "benchmark_point_query_cubql",
            [
                *common,
                "--output",
                str(output),
                "--library",
                str(libraries / "cubql_upstream.so"),
                "--tuning",
                str(workspace / "frozen-cubql-tuning.json"),
                "--reuse-tuning",
                *limit,
            ],
        ),
    ]
    for method in EXTERNAL:
        jobs.append(
            (
                method,
                "benchmark_point_query_comparison",
                [*common, "--output", str(output), "--worker", method, *dependencies, *repeats, *limit],
            )
        )
    for method in EXTERNAL:
        for phase in ("query", "full"):
            name = f"{method}-graph-{phase}"
            jobs.append(
                (
                    name,
                    "benchmark_point_query_graphs",
                    [
                        *common,
                        "--output",
                        str(output / f"{name}.json"),
                        "--method",
                        method,
                        "--phase",
                        phase,
                        *dependencies,
                        *repeats,
                        *limit,
                    ],
                )
            )
    if args.only:
        unknown = set(args.only) - {name for name, _, _ in jobs}
        if unknown:
            parser.error(f"Unknown jobs: {sorted(unknown)}")
        jobs = [job for job in jobs if job[0] in args.only]
    blas = workspace / "openblas/usr/lib/x86_64-linux-gnu"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{blas}:{blas / 'openblas-pthread'}:" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["CUDA_PATH"] = env["CUDA_HOME"] = str(args.cuda.resolve())
    receipt_path = output / "processes.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    for name, module, arguments in jobs:
        command = [
            sys.executable,
            "-m",
            "warp.examples.benchmarks.point_query_comparison_environment",
            "--python",
            sys.executable,
            "--cuda",
            str(args.cuda.resolve()),
            "--module",
            "warp.examples.benchmarks." + module,
            "--",
            *arguments,
        ]
        receipt = {"name": name, "command": command, "started_at": datetime.now(UTC).isoformat()}
        print("Running", name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=REPOSITORY,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                receipt["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                receipt.update(status="timeout", seconds=args.timeout)
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipts.append(receipt)
        receipt_path.write_text(json.dumps(receipts, indent=2) + "\n")
        print("Finished", name, receipt.get("returncode", receipt.get("status")), flush=True)


if __name__ == "__main__":
    main()
