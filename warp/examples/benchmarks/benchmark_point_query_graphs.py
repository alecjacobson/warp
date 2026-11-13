# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure one external kNN provider's graph replay in an isolated process.

Run query and construction-inclusive capture in separate processes. A capture
error terminates this worker after saving its diagnostic; do not continue timing
in a CUDA context whose capture failed.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy, environment, measure
from warp.examples.benchmarks.point_query_backends import Provider


def main():
    """Capture a provider operation and validate original and changed inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--phase", choices=("query", "full"), required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cubql-library", type=Path)
    parser.add_argument("--pytorch3d-path", type=Path)
    parser.add_argument("--faiss-path", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for path in (args.pytorch3d_path, args.faiss_path):
        if path is not None:
            sys.path.insert(0, str(path.resolve()))
    wp.init()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = wp.Stream("cuda:0")
    torch_stream = wp.stream_to_torch(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": args.method,
        "phase": args.phase,
        "environment": environment(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": wp.get_device("cuda:0").name,
        "sm_count": wp.get_device("cuda:0").sm_count,
        "repeats": args.repeats,
        "timing": "single graph replay; shared synchronized wall/event timer; capture and warmup excluded",
        "cases": [],
    }
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(wp.ScopedStream(stream))
            stack.enter_context(torch.cuda.stream(torch_stream))
            if args.method.startswith("cuvs_") or args.method == "cupy_kdtree":
                import cupy as cp  # noqa: PLC0415

                stack.enter_context(cp.cuda.ExternalStream(torch_stream.cuda_stream))
            if args.method.startswith("cuvs_"):
                import rmm  # noqa: PLC0415

                resource = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=0, release_threshold=512 * 1024**2)
                rmm.mr.set_current_device_resource(resource)
            provider = Provider(args.method, args.output.parent, args.cubql_library)
            report["provider"] = provider.metadata
            wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
            cases = json.loads(args.manifest.read_text())["cases"]
            for case in cases[: args.limit]:
                row = {"id": case["id"], "status": "preparing"}
                report["cases"].append(row)
                source = args.manifest.parent / case["file"]
                if hashlib.sha256(source.read_bytes()).hexdigest() != case["file_sha256"]:
                    raise ValueError(f"Dataset checksum mismatch: {source}")
                with np.load(source) as data:
                    points_host, queries_host = data["points"], data["queries"]
                points = torch.tensor(points_host, device="cuda")
                queries = torch.tensor(queries_host, device="cuda")
                problem = provider.bind(points, queries, case["k"])
                index = problem.build() if args.phase == "query" else None
                full_call = args.phase == "full" or not problem.has_index
                report["query_scope"] = "complete operator" if not problem.has_index else "reused index"
                for _ in range(3):
                    warm = problem.one_shot() if full_call else (index, problem.search(index))
                torch.cuda.synchronize()
                del warm
                graph = torch.cuda.CUDAGraph()
                row["status"] = "capturing"
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                start = time.perf_counter()
                with torch.cuda.graph(graph, stream=torch_stream):
                    uses_warp = args.method.startswith("warp_") or args.method == "cubql"
                    if uses_warp:
                        wp.capture_begin(force_module_load=False, external=True)
                    try:
                        result = problem.one_shot() if full_call else (index, problem.search(index))
                    finally:
                        if uses_warp:
                            wp.capture_end()
                capture_ms = 1000 * (time.perf_counter() - start)

                def replay(graph=graph, result=result):
                    graph.replay()
                    return result

                changed_queries = queries_host + np.float32(0.013)
                changed_points = points_host * np.float32(1.01) if full_call else points_host
                queries.copy_(torch.as_tensor(changed_queries, device="cuda"))
                if full_call:
                    points.copy_(torch.as_tensor(changed_points, device="cuda"))
                replay()
                changed = accuracy(
                    changed_points,
                    changed_queries,
                    result[1][0].cpu().numpy(),
                    result[1][1].cpu().numpy(),
                    case["k"],
                )
                points.copy_(torch.as_tensor(points_host, device="cuda"))
                queries.copy_(torch.as_tensor(queries_host, device="cuda"))
                timing, result = measure(replay, args.repeats)
                validation = accuracy(
                    points_host, queries_host, result[1][0].cpu().numpy(), result[1][1].cpu().numpy(), case["k"]
                )
                row.update(
                    status="ok",
                    timing=timing,
                    capture_wall_ms=capture_ms,
                    validation=validation,
                    changed_input_validation=changed,
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(args.method, args.phase, case["id"], timing["wall_median_ms"], flush=True)
                del replay, result, graph, index, problem, points, queries
                gc.collect()
                torch.cuda.synchronize()
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if report["cases"]:
            failed = report["cases"][-1]
            failed.update(failed_stage=failed["status"], status="error", error=repr(error))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
