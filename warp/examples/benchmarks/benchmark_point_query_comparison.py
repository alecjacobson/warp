# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Broad GPU kNN comparison over checksummed, immutable dataset cases.

Run each library in a fresh subprocess, sequentially. Public calls return int64
indices and float32 Euclidean distances. Wall times include host submission and
synchronize the device; CUDA-event times are supplemental. Graph probes run in
separate processes so an unsupported capture cannot contaminate later timings.
"""

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def accuracy(p, q, ids, distances, k, checked=32):
    rows = np.unique(np.linspace(0, len(q) - 1, min(checked, len(q)), dtype=int))
    shape_ok = ids.shape == distances.shape == (len(q), k)
    if not shape_ok:
        return {"valid_shape": False, "indices_shape": list(ids.shape), "distances_shape": list(distances.shape)}
    valid = bool(np.all((ids >= 0) & (ids < len(p))) and np.isfinite(distances).all())
    if not valid:
        return {"valid_shape": True, "valid_indices_and_finite_distances": False}
    duplicate_rows = int(np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0, axis=1).sum())
    unsorted_rows = int(np.any(np.diff(distances, axis=1) < -1.0e-7, axis=1).sum())
    recalls, rank_errors, max_rel, output_errors = [], 0, 0.0, 0
    p64 = p.astype(np.float64)
    for row in rows:
        d2 = np.sum((p64 - q[row].astype(np.float64)) ** 2, axis=1)
        selected = np.argpartition(d2, k - 1)[:k]
        expected = np.sort(d2[selected])
        actual = np.sort(d2[ids[row]])
        # Equal-distance neighbors are interchangeable for cross-library scoring.
        boundary = expected[-1]
        interior = np.flatnonzero(d2 < boundary)
        chosen = set(ids[row].tolist())
        correct = sum(int(i) in chosen for i in interior)
        correct += min(k - len(interior), sum(d2[int(i)] == boundary for i in chosen))
        recalls.append(correct / k)
        rank_errors += int(not np.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-12))
        actual_distances = np.sqrt(d2[ids[row]])
        max_rel = max(
            max_rel, float(np.max(np.abs(distances[row] - actual_distances) / np.maximum(actual_distances, 1.0e-12)))
        )
        output_errors += int(not np.allclose(distances[row], actual_distances, rtol=5.0e-4, atol=1.0e-6))
    return {
        "valid_shape": True,
        "valid_indices_and_finite_distances": True,
        "checked_rows": len(rows),
        "duplicate_rows": duplicate_rows,
        "unsorted_rows": unsorted_rows,
        "tie_aware_recall_at_k": float(np.mean(recalls)),
        "distance_rank_mismatch_rows": rank_errors,
        "reported_distance_mismatch_rows": output_errors,
        "max_reported_distance_relative_error": max_rel,
        "rank_rtol": 1.0e-5,
        "rank_atol_squared": 1.0e-12,
    }


def measure(fn, repeats):
    import torch  # noqa: PLC0415

    for _ in range(2):
        result = fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall, gpu = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        wall.append(1000 * (time.perf_counter() - start))
        gpu.append(begin.elapsed_time(end))
    return {
        "wall_median_ms": statistics.median(wall),
        "wall_min_ms": min(wall),
        "wall_max_ms": max(wall),
        "wall_samples_ms": wall,
        "event_median_ms": statistics.median(gpu),
        "event_samples_ms": gpu,
    }, result


def environment():
    packages = {}
    for name in (
        "torch",
        "numpy",
        "cupy-cuda12x",
        "cuvs-cu12",
        "libcuvs-cu12",
        "rmm-cu12",
        "faiss-gpu-cu12",
        "pykeops",
        "pytorch3d",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark_point_query_comparison.py", "point_query_backends.py")
        },
        "native_bvh_sha256": hashlib.sha256(Path("warp/native/bvh.cu").read_bytes()).hexdigest(),
    }


def worker(args):
    import torch  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415
    from warp.examples.benchmarks.point_query_backends import Provider  # noqa: PLC0415

    if args.pytorch3d_path:
        sys.path.insert(0, str(args.pytorch3d_path.resolve()))
    if args.faiss_path:
        sys.path.insert(0, str(args.faiss_path.resolve()))
    wp.init()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    wp.set_mempool_release_threshold("cuda:0", args.pool_mib * 1024**2)
    stream = torch.cuda.Stream()
    wp_stream = wp.stream_from_torch(stream)
    report = {
        "method": args.worker,
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "primary_timing": "device-synchronized wall time",
        "warp_pool_release_threshold_bytes": wp.get_mempool_release_threshold("cuda:0"),
        "torch_allocator": torch.cuda.memory.get_allocator_backend(),
        "cases": [],
    }
    if args.pytorch3d_path:
        report["pytorch3d_revision"] = subprocess.check_output(
            ["git", "-C", str(args.pytorch3d_path), "rev-parse", "HEAD"], text=True
        ).strip()
    if args.faiss_path:
        report["faiss_revision"] = subprocess.check_output(
            ["git", "-C", str(args.faiss_path), "rev-parse", "HEAD"], text=True
        ).strip()
    path = args.output / (args.worker + ("-graph.json" if args.graph_probe else ".json"))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.cuda.stream(stream))
            stack.enter_context(wp.ScopedStream(wp_stream))
            if args.worker.startswith("cuvs_") or args.worker == "cupy_kdtree":
                import cupy as cp  # noqa: PLC0415

                stack.enter_context(cp.cuda.ExternalStream(stream.cuda_stream))
                report["cupy_allocator"] = "default caching pool"
            if args.worker.startswith("cuvs_"):
                import rmm  # noqa: PLC0415

                resource = rmm.mr.CudaAsyncMemoryResource(
                    initial_pool_size=0, release_threshold=args.pool_mib * 1024**2
                )
                rmm.mr.set_current_device_resource(resource)
                report["rmm_allocator"] = {
                    "name": "CudaAsyncMemoryResource",
                    "initial_pool_bytes": 0,
                    "release_threshold_bytes": args.pool_mib * 1024**2,
                }
            provider = Provider(args.worker, args.output, args.cubql_library)
            report["provider"] = provider.metadata
            if args.graph_probe:
                rng = np.random.default_rng(117)
                p, q = rng.normal(size=(4096, 3)).astype(np.float32), rng.normal(size=(64, 3)).astype(np.float32)
                pt, qt = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
                problem = provider.bind(pt, qt, 11)
                index = problem.build()
                for _ in range(3):
                    result = problem.search(index)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    uses_warp = args.worker.startswith("warp_") or args.worker in ("physicsnemo", "cubql")
                    if uses_warp:
                        wp.capture_begin(force_module_load=False, external=True)
                    try:
                        result = problem.search(index)
                    finally:
                        if uses_warp:
                            wp.capture_end()
                qt.add_(0.003)
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                report["capture"] = {
                    "supported": True,
                    "changed_query_validation": accuracy(
                        p, qt.cpu().numpy(), result[0].cpu().numpy(), result[1].cpu().numpy(), 11
                    ),
                }
            else:
                manifest = json.loads(args.manifest.read_text())
                report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
                cases = manifest["cases"][: args.limit] if args.limit else manifest["cases"]
                for case in cases:
                    row = {
                        "id": case["id"],
                        "N": case["N"],
                        "Q": case["Q"],
                        "k": case["k"],
                        "family": case["family"],
                        "sigma": case["sigma"],
                    }
                    report["cases"].append(row)
                    try:
                        source = args.manifest.parent / case["file"]
                        if hashlib.sha256(source.read_bytes()).hexdigest() != case["file_sha256"]:
                            raise ValueError(f"Dataset checksum mismatch: {source}")
                        with np.load(source) as arrays:
                            p, q = arrays["points"], arrays["queries"]
                        t0 = time.perf_counter()
                        pt, qt = torch.tensor(p, device="cuda"), torch.tensor(q, device="cuda")
                        torch.cuda.synchronize()
                        row["input_transfer_wall_ms"] = 1000 * (time.perf_counter() - t0)
                        problem = provider.bind(pt, qt, case["k"])
                        if problem.has_index:
                            row["build"], index = measure(problem.build, args.repeats)
                            row["reused_index_search"], output = measure(
                                lambda problem=problem, index=index: problem.search(index), args.repeats
                            )
                            row["reused_validation"] = accuracy(
                                p, q, output[0].cpu().numpy(), output[1].cpu().numpy(), case["k"]
                            )
                            del index, output
                        else:
                            row["build"], row["reused_index_search"] = None, None
                        row["one_shot"], result = measure(problem.one_shot, args.repeats)
                        output = result[1]
                        row["validation"] = accuracy(p, q, output[0].cpu().numpy(), output[1].cpu().numpy(), case["k"])
                        row["output_contract"] = {
                            "index_dtype": str(output[0].dtype),
                            "distance_dtype": str(output[1].dtype),
                            "device": str(output[0].device),
                            "distance": "Euclidean",
                        }
                        if (
                            output[0].dtype != torch.int64
                            or output[1].dtype != torch.float32
                            or not output[0].is_cuda
                            or not output[1].is_cuda
                        ):
                            raise AssertionError(row["output_contract"])
                        free, total = torch.cuda.mem_get_info()
                        row["memory_snapshot"] = {
                            "device_used_bytes": total - free,
                            "torch_reserved_bytes": torch.cuda.memory_reserved(),
                            "note": "post-warmup snapshot including caches/context, not transient peak",
                        }
                        row["status"] = "ok"
                        print(
                            args.worker,
                            case["id"],
                            round(row["one_shot"]["wall_median_ms"], 4),
                            "ms",
                            row["validation"],
                            flush=True,
                        )
                        del result, output, problem, pt, qt
                        gc.collect()
                        torch.cuda.synchronize()
                    except Exception as error:
                        row.update(status="error", error=repr(error), traceback=traceback.format_exc())
                        print(args.worker, case["id"], repr(error), flush=True)
                    path.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        if args.graph_probe:
            report["capture"] = {"supported": False, "error": repr(error)}
        print(traceback.format_exc(), flush=True)
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "warp_default",
            "warp_configured",
            "physicsnemo",
            "cubql",
            "cupy_kdtree",
            "keops",
            "faiss_flat",
            "cuvs_brute_force",
            "cuvs_unexpanded",
            "cuvs_minkowski",
            "pytorch3d",
            "cuvs_cagra",
        ],
    )
    parser.add_argument("--cubql-library", type=Path)
    parser.add_argument("--pytorch3d-path", type=Path)
    parser.add_argument("--faiss-path", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pool-mib", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout per method process, in seconds")
    parser.add_argument("--worker")
    parser.add_argument("--graph-probe", action="store_true")
    parser.add_argument("--skip-graph-probes", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.pool_mib < 0:
        parser.error("repeats must be positive and pool-mib must be nonnegative")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.methods:
        base = [
            sys.executable,
            "-m",
            "warp.examples.benchmarks.benchmark_point_query_comparison",
            "--manifest",
            str(args.manifest),
            "--output",
            str(args.output),
            "--worker",
            name,
            "--repeats",
            str(args.repeats),
            "--pool-mib",
            str(args.pool_mib),
        ]
        for option, value in (
            ("--cubql-library", args.cubql_library),
            ("--pytorch3d-path", args.pytorch3d_path),
            ("--faiss-path", args.faiss_path),
            ("--limit", args.limit),
        ):
            if value is not None:
                base += [option, str(value)]
        for probe in [True, False] if not args.skip_graph_probes else [False]:
            label = name + ("-graph" if probe else "")
            print("Running", label, flush=True)
            with (args.output / (label + ".log")).open("w") as log:
                try:
                    process = subprocess.run(
                        base + (["--graph-probe"] if probe else []),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                    record = {"name": label, "returncode": process.returncode}
                except subprocess.TimeoutExpired:
                    record = {"name": label, "status": "timeout", "seconds": args.timeout}
            records.append(record)
            (args.output / "processes.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Results:", args.output)


if __name__ == "__main__":
    main()
