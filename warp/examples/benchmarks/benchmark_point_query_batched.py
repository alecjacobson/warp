# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure fixed-target batched kNN and symmetric Chamfer on supplied pulmonary clouds."""

import argparse
import gc
import hashlib
import json
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy
from warp.examples.geometry.example_point_query_chamfer import mean_distance_kernel


def measure(operation, repeats, device):
    result = None
    for _ in range(2):
        result = operation()
    wp.synchronize_device(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        wp.synchronize_device(device)
        samples.append(1000 * (time.perf_counter() - start))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}, result


def validate(targets, queries, result, k):
    indices, distances, counts = (array.numpy() for array in result[1:])
    np.testing.assert_array_equal(counts, k)
    checks = []
    for batch in range(len(targets)):
        check = accuracy(targets[batch], queries[batch], indices[batch], np.sqrt(distances[batch]), k)
        if not (
            check["valid_shape"]
            and check["valid_indices_and_finite_distances"]
            and all(
                check[name] == 0
                for name in (
                    "duplicate_rows",
                    "unsorted_rows",
                    "distance_rank_mismatch_rows",
                    "reported_distance_mismatch_rows",
                )
            )
        ):
            raise RuntimeError(f"Batched point-query validation failed: {check}")
        checks.append(check)
    return checks


def benchmark_case(targets_host, queries_host, k, backend, objective, repeats, device):
    points = wp.array(targets_host, dtype=wp.vec3, device=device)
    queries = wp.array(queries_host, dtype=wp.vec3, device=device)
    wp.synchronize_device(device)
    baseline = wp.get_mempool_used_mem_current(device)
    initial = wp.point_query_knn_batch(points, queries, k, backend=backend, objective=objective)
    wp.synchronize_device(device)
    retained_bytes = wp.get_mempool_used_mem_current(device) - baseline
    checks = validate(targets_host, queries_host, initial, k)
    del initial

    def build_and_query():
        return wp.point_query_knn_batch(points, queries, k, backend=backend, objective=objective)

    build_time, result = measure(build_and_query, repeats, device)
    indexes = result[0]

    def query():
        return wp.point_query_knn_batch(points, queries, k, bvh=indexes)

    query_time, reused = measure(query, repeats, device)
    with wp.ScopedCapture(device=device) as capture:
        graph_result = wp.point_query_knn_batch(
            points, queries, k, bvh=indexes, indices=reused[1], distances=reused[2], counts=reused[3]
        )
    graph_time, _ = measure(lambda: wp.capture_launch(capture.graph), repeats, device)
    changed = queries_host + np.float32(0.003)
    queries.assign(changed)
    wp.capture_launch(capture.graph)
    graph_checks = validate(targets_host, changed, graph_result, k)
    queries.assign(queries_host)

    report = {
        "backend": backend.name,
        "objective": objective.name,
        "build_and_query": build_time,
        "reused_query": query_time,
        "graph_query": graph_time,
        "validation": checks,
        "changed_query_graph_validation": graph_checks,
        "input_bytes": targets_host.nbytes + queries_host.nbytes,
        "retained_index_and_output_pool_bytes": retained_bytes,
        "pairwise_float32_tensor_bytes": 4 * len(points) * targets_host.shape[1] * queries_host.shape[1],
    }
    if k == 1:
        decoded = wp.array(queries_host, dtype=wp.vec3, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        reverse_backend = backend

        def chamfer_step():
            loss.zero_()
            decoded.grad.zero_()
            with wp.Tape() as tape:
                forward = wp.point_query_knn_batch(points, decoded, 1, bvh=indexes)
                reverse = wp.point_query_knn_batch(
                    decoded, points, 1, backend=reverse_backend, objective=wp.PointQueryObjective.BUILD_AND_QUERY
                )
                wp.launch(mean_distance_kernel, decoded.shape, [forward[2], loss, 1.0 / decoded.size], device=device)
                wp.launch(mean_distance_kernel, points.shape, [reverse[2], loss, 1.0 / points.size], device=device)
            tape.backward(loss)
            return loss

        report["symmetric_chamfer_forward_backward"], _ = measure(chamfer_step, repeats, device)
        report["chamfer_loss"] = float(loss.numpy()[0])
        if not np.isfinite(report["chamfer_loss"]) or not np.isfinite(decoded.grad.numpy()).all():
            raise RuntimeError("Chamfer loss or gradients are nonfinite")
        if backend == wp.PointQueryBackend.CUBQL and objective == wp.PointQueryObjective.REUSE:
            reverse_backend = wp.PointQueryBackend.WARP
            report["mixed_chamfer_forward_backward"], _ = measure(chamfer_step, repeats, device)
            np.testing.assert_allclose(float(loss.numpy()[0]), report["chamfer_loss"], rtol=2e-5)
            if not np.isfinite(decoded.grad.numpy()).all():
                raise RuntimeError("Mixed-backend Chamfer gradients are nonfinite")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    wp.init()
    device = wp.get_device("cuda:0")
    wp.set_mempool_release_threshold(device, 512 * 1024**2)
    rng = np.random.default_rng(20261113)
    clouds = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".npz"):
                with archive.open(name) as stream, np.load(stream, allow_pickle=False) as data:
                    clouds[Path(name).stem] = np.ascontiguousarray(data["points"], dtype=np.float32)
    report = {
        "device": device.name,
        "sm_count": device.sm_count,
        "warp_version": wp.__version__,
        "numpy_version": np.__version__,
        "repeats": args.repeats,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "warp/_src/point_query.py",
                "warp/native/bvh_cubql.cu",
                "warp/native/bvh.cu",
                "warp/bin/warp.so",
                "warp/examples/benchmarks/benchmark_point_query_batched.py",
            )
        },
        "cloud_sizes": {name: len(points) for name, points in clouds.items()},
        "memory_scope": "Live default CUDA pool bytes for retained index and outputs after synchronization; excludes inputs and transient construction scratch. Not a peak-memory measurement.",
        "query_generation": "Deterministic target samples with independent Gaussian noise of sigma=0.01 in source units (cm for pulmonary inputs); no target normalization or downsampling except the explicitly named mixed subset.",
        "cases": [],
    }
    cases = []
    for name, points in clouds.items():
        for query_count in (1024, 8192):
            cases.append((f"{name}_full_q{query_count}", points[None], query_count, 1))
    largest = max(clouds.values(), key=len)
    for batches in (2, 4):
        cases.append((f"largest_full_b{batches}", np.repeat(largest[None], batches, axis=0), 8192, 1))
    mixed = np.stack([points[rng.choice(len(points), 65536, replace=False)] for points in list(clouds.values())[:4]])
    cases.append(("mixed_subset_b4_m65536", mixed, 8192, 1))
    for size, query_count, k in ((1000, 32, 32), (100000, 1024, 32), (1000000, 8192, 11)):
        cases.append(
            (f"synthetic_volume_m{size}_k{k}", rng.uniform(-1, 1, (1, size, 3)).astype(np.float32), query_count, k)
        )
    variants = (
        ("warp", wp.PointQueryBackend.WARP, wp.PointQueryObjective.REUSE),
        ("cubql_reuse", wp.PointQueryBackend.CUBQL, wp.PointQueryObjective.REUSE),
        ("cubql_build", wp.PointQueryBackend.CUBQL, wp.PointQueryObjective.BUILD_AND_QUERY),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, targets, query_count, k in cases:
        if args.filter not in name:
            continue
        queries = np.stack([points[rng.choice(len(points), query_count)] for points in targets])
        queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)
        row = {
            "name": name,
            "batches": len(targets),
            "targets": targets.shape[1],
            "queries": query_count,
            "k": k,
            "variants": {},
        }
        for label, backend, objective in variants:
            gc.collect()
            result = benchmark_case(targets, queries, k, backend, objective, args.repeats, device)
            row["variants"][label] = result
            print(
                name,
                label,
                "build/query/graph ms",
                *(round(result[key]["median_ms"], 4) for key in ("build_and_query", "reused_query", "graph_query")),
                flush=True,
            )
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
