# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed-target batched Chamfer with PyTorch cdist on supplied clouds.

Run each method/shape in a separate process so allocator or CUDA failures remain
isolated. Torch is an optional dependency of this benchmark, not of Warp's API.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import warp as wp
import warp.geometry
from warp.examples.benchmarks.benchmark_point_query_chamfer import load_clouds, validate_neighbors
from warp.examples.geometry.example_point_query_chamfer import mean_distance_kernel

METHODS = (
    "cdist",
    "cdist_direct",
    "chunked_direct",
    "chunked_mm",
    "warp",
    "cubql_warp",
    "warp_torch",
    "cubql_torch",
    "warp_build_both",
    "warp_torch_build_both",
)


def selected_loss(points, queries, forward, reverse):
    nearest_targets = points.gather(1, forward.unsqueeze(-1).expand(-1, -1, 3))
    nearest_queries = queries.gather(1, reverse.unsqueeze(-1).expand(-1, -1, 3))
    return (queries - nearest_targets).square().sum(-1).mean() + (points - nearest_queries).square().sum(-1).mean()


def chunked_ids(points, queries, rows, mode):
    """Visit each pair once, retaining the two directional minima on device."""
    batches, n, _ = queries.shape
    m = points.shape[1]
    forward = torch.empty((batches, n), device=queries.device, dtype=torch.int64)
    reverse = torch.zeros((batches, m), device=queries.device, dtype=torch.int64)
    reverse_distance = torch.full((batches, m), float("inf"), device=queries.device)
    with torch.no_grad():
        for start in range(0, n, rows):
            distance = torch.cdist(queries[:, start : start + rows], points, compute_mode=mode)
            forward[:, start : start + rows] = distance.argmin(dim=2)
            values, indices = distance.min(dim=1)
            better = values < reverse_distance
            reverse = torch.where(better, indices + start, reverse)
            reverse_distance = torch.minimum(reverse_distance, values)
            del distance
    return forward, reverse


def numpy_reference(points, queries, forward, reverse):
    """Independently accumulate selected-pair squared loss/decoded gradients."""
    p, q = points.astype(np.float64), queries.astype(np.float64)
    gradient = np.zeros_like(q)
    loss = 0.0
    for batch in range(len(p)):
        first = q[batch] - p[batch, forward[batch]]
        second = q[batch, reverse[batch]] - p[batch]
        loss += (first * first).sum() / (len(p) * q.shape[1])
        loss += (second * second).sum() / (len(p) * p.shape[1])
        gradient[batch] += 2 * first / (len(p) * q.shape[1])
        np.add.at(gradient[batch], reverse[batch], 2 * second / (len(p) * p.shape[1]))
    return float(loss), gradient


def read_inputs(args):
    clouds = [c for c in load_clouds(args.archive) if len(c) >= args.n]
    if not clouds:
        raise ValueError("No supplied cloud is large enough")
    selected = [clouds[b % len(clouds)] for b in range(args.batches)]
    rng = np.random.default_rng(np.random.SeedSequence([2348, args.batches, args.n]))
    p = np.stack([c[rng.choice(len(c), args.n, replace=False)] for c in selected])
    q = np.stack([c[rng.choice(len(c), args.n, replace=False)] for c in selected])
    return p, q


def run_worker(args):
    wp.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = wp.get_device("cuda:0")
    p, q = read_inputs(args)
    rows = min(args.chunk_rows, max(1, args.chunk_mib * 1024**2 // (args.batches * args.n * 4)))
    build_both = args.method.endswith("build_both")
    record = {
        "method": args.method,
        "batches": args.batches,
        "points": args.n,
        "chunk_rows": rows if args.method.startswith("chunked") else None,
        "points_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        "device": device.name,
        "sm_count": device.sm_count,
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": wp.__version__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (__file__, "warp/_src/geometry/point_query.py", "warp/bin/warp.so")
        },
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "tf32": False,
        "gradient": "decoded queries only; fixed target index outside timing; reverse index rebuilt each call",
        "timing": "synchronized wall, two warmups, no graph replay, uploads and compilation excluded",
    }
    if build_both:
        record["gradient"] = "decoded queries only; both target and reverse indexes rebuilt each call"
    if args.method in ("cdist", "cdist_direct"):
        matrix_bytes = args.batches * args.n**2 * 4
        if matrix_bytes > record["device_memory_bytes"]:
            record.update(
                status="does_not_fit",
                matrix_bytes=matrix_bytes,
                reason="Output matrix alone exceeds device memory; not launched",
            )
            return record
    # The forced direct-distance diagnostic already takes seconds at 4K on
    # this workload. Keep its larger cases explicit rather than extrapolating
    # unmeasured timings; the default matrix-based chunked baseline runs all sizes.
    if args.method == "chunked_direct" and args.n > args.direct_max_points:
        record.update(
            status="not_run", reason=f"Forced direct-distance diagnostic limited to N<={args.direct_max_points}"
        )
        return record

    with wp.ScopedDevice(device), torch.cuda.stream(wp.stream_to_torch(device.stream)):
        wp.set_mempool_release_threshold(device, 512 * 1024**2)
        points = torch.tensor(p, device="cuda")
        queries = torch.tensor(q, device="cuda", requires_grad=True)
        wp_points = wp.from_torch(points, dtype=wp.vec3)
        wp_queries = wp.from_torch(queries, dtype=wp.vec3, requires_grad=False)
        # Build the correctness oracle outside timing, then independently check
        # sampled exhaustive distances and reconstruct all gradients in NumPy.
        oracle_first = wp.geometry.knn(wp_points, wp_queries, 1)
        oracle_second = wp.geometry.knn(wp_queries, wp_points, 1)
        validate_neighbors(p, q, oracle_first)
        validate_neighbors(q, p, oracle_second)
        expected_loss, expected_gradient = numpy_reference(
            p, q, oracle_first[1].numpy()[..., 0], oracle_second[1].numpy()[..., 0]
        )
        del oracle_first, oracle_second

        is_warp = args.method.startswith(("warp", "cubql"))
        tape_mode = args.method in ("warp", "cubql_warp", "warp_build_both")
        if is_warp:
            backend = (
                wp.geometry.PointQueryBackend.CUBQL
                if args.method.startswith("cubql")
                else wp.geometry.PointQueryBackend.WARP
            )
            fixed = wp_points if build_both else wp.geometry.PointIndex(wp_points, num_queries=args.n, backend=backend)
            if tape_mode:
                wp_queries = wp.from_torch(queries, dtype=wp.vec3, requires_grad=True)
            forward_ids = wp.empty((args.batches, args.n, 1), dtype=wp.int64)
            reverse_ids = wp.empty_like(forward_ids)
            forward_dist = wp.empty(forward_ids.shape, dtype=float, requires_grad=tape_mode)
            reverse_dist = wp.empty_like(forward_dist, requires_grad=tape_mode)
            forward_count = wp.empty((args.batches, args.n), dtype=int)
            reverse_count = wp.empty_like(forward_count)
            loss_array = wp.zeros(1, dtype=float, requires_grad=True)

        def operation():
            queries.grad = None
            if is_warp:
                if tape_mode:
                    wp_queries.grad.zero_()
                    forward_dist.grad.zero_()
                    reverse_dist.grad.zero_()
                    loss_array.zero_()
                tape = wp.Tape() if tape_mode else None
                if tape is not None:
                    tape.__enter__()
                try:
                    first = wp.geometry.knn(
                        fixed, wp_queries, 1, indices=forward_ids, distances=forward_dist, counts=forward_count
                    )
                    second = wp.geometry.knn(
                        wp_queries, wp_points, 1, indices=reverse_ids, distances=reverse_dist, counts=reverse_count
                    )
                    if tape_mode:
                        wp.launch(mean_distance_kernel, wp_queries.shape, [first[2], loss_array, 1.0 / wp_queries.size])
                        wp.launch(mean_distance_kernel, wp_points.shape, [second[2], loss_array, 1.0 / wp_points.size])
                finally:
                    if tape is not None:
                        tape.__exit__(None, None, None)
                if tape_mode:
                    tape.backward(loss_array)
                    result = loss_array, wp_queries.grad
                else:
                    loss = selected_loss(points, queries, wp.to_torch(first[1])[..., 0], wp.to_torch(second[1])[..., 0])
                    loss.backward()
                    result = loss, queries.grad
                # The changing reverse owner is retired before returning; retain
                # all referenced arrays until its submitted operations are safe.
                del first, second
                return result
            if args.method.startswith("chunked"):
                mode = "donot_use_mm_for_euclid_dist" if args.method == "chunked_direct" else "use_mm_for_euclid_dist"
                forward, reverse = chunked_ids(points, queries, rows, mode)
                loss = selected_loss(points, queries, forward, reverse)
            else:
                if args.method == "cdist_direct":
                    distances = torch.cdist(queries, points, compute_mode="donot_use_mm_for_euclid_dist")
                else:
                    distances = torch.cdist(queries, points)
                loss = distances.min(dim=2).values.square().mean() + distances.min(dim=1).values.square().mean()
            loss.backward()
            return loss, queries.grad

        for _ in range(2):
            result = operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = operation()
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
        if tape_mode:
            loss = float(result[0].numpy()[0])
            gradient = result[1].numpy().astype(np.float64)
        else:
            loss = float(result[0].detach().cpu())
            gradient = result[1].detach().cpu().numpy().astype(np.float64)
        delta = gradient - expected_gradient
        relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(expected_gradient), 1e-15))
        maximum_error = float(np.max(np.abs(delta)))
        tolerance = 1e-8 + 5e-4 * float(np.max(np.abs(expected_gradient)))
        passed = bool(
            np.isclose(loss, expected_loss, rtol=2e-4, atol=1e-7) and relative_l2 <= 5e-4 and maximum_error <= tolerance
        )
        record.update(
            status="ok" if passed else "accuracy_failed",
            median_ms=statistics.median(samples),
            samples_ms=samples,
            validation={
                "loss": loss,
                "reference_loss": expected_loss,
                "gradient_relative_l2": relative_l2,
                "gradient_max_error": maximum_error,
                "gradient_max_tolerance": tolerance,
                "passed": passed,
            },
            torch_allocator_peak_bytes=torch.cuda.max_memory_allocated(),
        )
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--points", nargs="+", type=int, default=[1024, 4096, 16384, 32768, 65536])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--chunk-mib", type=int, default=1024)
    parser.add_argument(
        "--direct-max-points",
        type=int,
        default=4096,
        help="Largest forced-direct chunked diagnostic; increase explicitly for slow larger cases",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    if min(args.batches, args.repeats, args.chunk_rows, args.chunk_mib, args.direct_max_points, *args.points) < 1:
        parser.error("Counts, repeats and chunk limits must be positive")
    if args.worker and (args.method is None or args.n is None or args.n < 1):
        parser.error("Workers require --method and a positive --n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            record = run_worker(args)
        except Exception as error:
            record = {
                "method": args.method,
                "batches": args.batches,
                "points": args.n,
                "status": "runtime_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(record, flush=True)
    else:
        records = []
        for n in args.points:
            for method in args.methods:
                path = args.output.parent / f"{args.output.stem}-{method}-{n}.json"
                command = [
                    sys.executable,
                    "-m",
                    "warp.examples.benchmarks.benchmark_point_query_cdist",
                    "--worker",
                    "--archive",
                    str(args.archive),
                    "--output",
                    str(path),
                    "--batches",
                    str(args.batches),
                    "--n",
                    str(n),
                    "--method",
                    method,
                    "--repeats",
                    str(args.repeats),
                    "--chunk-rows",
                    str(args.chunk_rows),
                    "--chunk-mib",
                    str(args.chunk_mib),
                    "--direct-max-points",
                    str(args.direct_max_points),
                ]
                path.unlink(missing_ok=True)
                with path.with_suffix(".log").open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                record = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"method": method, "points": n, "status": "worker_failed"}
                )
                record["worker_exit_code"] = process.returncode
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                print(n, method, record["status"], record.get("median_ms"), flush=True)


if __name__ == "__main__":
    main()
