# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare public batched kNN and fixed-target Chamfer without modifying providers."""

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy, environment
from warp.examples.geometry.example_point_query_chamfer import mean_distance_kernel

METHODS = (
    "warp",
    "cubql_reuse",
    "cubql_build",
    "torch_mm",
    "torch_direct",
    "torch_selected",
    "pytorch3d",
    "pytorch3d_tuned",
    "keops",
)


def sha256(path):
    """Hash an evidence file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(operation, repeats):
    """Measure synchronized wall time after two warmup calls."""
    result = None
    for _ in range(2):
        result = operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}, result


def checks_pass(checks):
    """Require all sampled rank/distance and full-output structural checks."""
    return bool(checks) and all(
        check.get("valid_shape")
        and check.get("valid_indices_and_finite_distances")
        and all(
            check.get(key, 1) == 0
            for key in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        )
        for check in checks
    )


def query_checks(points, queries, indices, squared_distances, k):
    """Check every batch independently against exhaustive sampled references."""
    return [
        accuracy(p, q, ids, np.sqrt(distances), k)
        for p, q, ids, distances in zip(points, queries, indices, squared_distances, strict=True)
    ]


def chamfer_reference(points, queries, forward_ids, reverse_ids):
    """Compute float64 loss and decoded gradients for fixed nearest-neighbor IDs."""
    points, queries = points.astype(np.float64), queries.astype(np.float64)
    batches, targets, _ = points.shape
    query_count = queries.shape[1]
    gradient = np.zeros_like(queries)
    loss = 0.0
    for batch in range(batches):
        forward_delta = queries[batch] - points[batch, forward_ids[batch]]
        reverse_delta = queries[batch, reverse_ids[batch]] - points[batch]
        loss += np.sum(forward_delta**2) / (batches * query_count)
        loss += np.sum(reverse_delta**2) / (batches * targets)
        gradient[batch] += 2 * forward_delta / (batches * query_count)
        np.add.at(gradient[batch], reverse_ids[batch], 2 * reverse_delta / (batches * targets))
    return loss, gradient


def prepare(args):
    """Restore the saved pulmonary workload sequence and independent references."""
    rng = np.random.default_rng(20261113)
    clouds = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".npz"):
                with archive.open(name) as stream, np.load(stream, allow_pickle=False) as arrays:
                    clouds[Path(name).stem] = np.ascontiguousarray(arrays["points"], dtype=np.float32)
    cases = []
    for name, points in clouds.items():
        for query_count in (1024, 8192):
            cases.append((f"{name}_full_q{query_count}", points[None], query_count, 1))
    largest = max(clouds.values(), key=len)
    for batches in (2, 4):
        cases.append((f"largest_full_b{batches}", np.repeat(largest[None], batches, axis=0), 8192, 1))
    mixed = np.stack([p[rng.choice(len(p), 65536, replace=False)] for p in list(clouds.values())[:4]])
    cases.append(("mixed_subset_b4_m65536", mixed, 8192, 1))
    for size, query_count, k in ((1000, 32, 32), (100000, 1024, 32), (1000000, 8192, 11)):
        cases.append(
            (f"synthetic_volume_m{size}_k{k}", rng.uniform(-1, 1, (1, size, 3)).astype(np.float32), query_count, k)
        )
    generated = []
    for name, points, query_count, k in cases:
        queries = np.stack([p[rng.choice(len(p), query_count)] for p in points])
        queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)
        generated.append((name, points, queries, k))
    fixture_rng = np.random.default_rng(510203)
    generated.append(
        (
            "fixture_b2_m73_n41",
            fixture_rng.normal(size=(2, 73, 3)).astype(np.float32),
            fixture_rng.normal(size=(2, 41, 3)).astype(np.float32),
            1,
        )
    )
    args.data.mkdir(parents=True, exist_ok=True)
    manifest = {"archive_sha256": sha256(args.archive), "seed": 20261113, "cases": [], "reference_checks": []}
    for name, points, queries, k in generated:
        target_array, query_array = wp.array(points, dtype=wp.vec3), wp.array(queries, dtype=wp.vec3)
        forward = wp.knn(target_array, query_array, k)
        ids, distances = forward[1].numpy(), forward[2].numpy()
        checks = query_checks(points, queries, ids, distances, k)
        if not checks_pass(checks):
            raise ValueError(f"Reference query checks failed: {name}")
        arrays = {"points": points, "queries": queries, "reference_indices": ids, "reference_distances": distances}
        if k == 1:
            reverse = wp.knn(query_array, target_array, 1)
            reverse_ids = reverse[1].numpy()[..., 0]
            reverse_checks = query_checks(queries, points, reverse[1].numpy(), reverse[2].numpy(), 1)
            if not checks_pass(reverse_checks):
                raise ValueError(f"Reverse reference checks failed: {name}")
            loss, gradient = chamfer_reference(points, queries, ids[..., 0], reverse_ids)
            arrays.update(reference_loss=loss, reference_gradient=gradient)
            del reverse
        path = args.data / (name + ".npz")
        np.savez_compressed(path, **arrays)
        row = {
            "name": name,
            "batches": len(points),
            "targets": points.shape[1],
            "queries": queries.shape[1],
            "k": k,
            "file": path.name,
            "file_sha256": sha256(path),
            "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
        }
        if name.startswith("fixture"):
            manifest["fixture"] = row
        else:
            manifest["cases"].append(row)
        manifest["reference_checks"].append(
            {"name": name, "forward": checks, "reverse": reverse_checks if k == 1 else None}
        )
        print("Prepared", name, flush=True)
        del forward, target_array, query_array
    (args.data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def regime(case):
    """Select a shape regime without consulting evaluation timings."""
    return f"b{case['batches']}_q{'small' if case['queries'] <= 2048 else 'large'}_k{case['k']}"


def tune(args):
    """Choose public PyTorch3D kernel versions using independent synthetic inputs."""
    from pytorch3d.ops import knn_points  # noqa: PLC0415

    manifest = json.loads((args.data / "manifest.json").read_text())
    regimes = {regime(case): case for case in [*manifest["cases"], manifest["fixture"]]}
    tuning = {
        "seed": 198704,
        "selection": "minimum median forward call, independently per direction; no evaluation inputs",
        "regimes": {},
    }
    generator = torch.Generator(device="cuda").manual_seed(tuning["seed"])
    for key, case in regimes.items():
        batches, k = case["batches"], case["k"]
        query_count = 1024 if case["queries"] <= 2048 else 8192
        points = torch.randn((batches, 131072, 3), device="cuda", generator=generator)
        queries = torch.randn((batches, query_count, 3), device="cuda", generator=generator)
        valid_versions = range(4 if k <= 4 else 3)
        entry = {"shape": [batches, query_count, 131072], "directions": {}}
        for direction, left, right, count in (("forward", queries, points, k), ("reverse", points, queries, 1)):
            if direction == "reverse" and k != 1:
                continue
            measurements = {}
            for version in valid_versions:
                with torch.no_grad():
                    timing, _ = measure(
                        lambda version=version, left=left, right=right, count=count: knn_points(
                            left, right, K=count, version=version
                        ),
                        5,
                    )
                measurements[str(version)] = timing
            entry["directions"][direction] = {
                "version": int(min(measurements, key=lambda v: measurements[v]["median_ms"])),
                "candidates": measurements,
            }
        tuning["regimes"][key] = entry
        print("Tuned", key, {k: v["version"] for k, v in entry["directions"].items()}, flush=True)
    args.output.write_text(json.dumps(tuning, indent=2) + "\n")


class Problem:
    """Bind a batch to unchanged provider APIs and standard loss reductions."""

    def __init__(self, method, phase, points, queries, k, tuning):
        self.method, self.phase, self.k = method, phase, k
        self.points, self.queries = points, queries
        self.tuning = tuning
        self.is_warp = method in ("warp", "cubql_reuse", "cubql_build")
        if self.is_warp:
            self.target_array = wp.from_torch(points, dtype=wp.vec3)
            self.query_array = wp.from_torch(queries, dtype=wp.vec3, requires_grad=phase == "chamfer")
            self.backend = wp.PointQueryBackend.WARP if method == "warp" else wp.PointQueryBackend.CUBQL
            self.objective = (
                wp.PointQueryObjective.BUILD_AND_QUERY
                if method == "cubql_build" and phase == "query"
                else wp.PointQueryObjective.REUSE
            )
            self.indexes = wp.build_point_index(
                self.target_array, num_queries=queries.shape[1], k=k, backend=self.backend, objective=self.objective
            )
            self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        elif method.startswith("pytorch3d"):
            from pytorch3d.loss import chamfer_distance  # noqa: PLC0415
            from pytorch3d.ops import knn_points  # noqa: PLC0415

            self.knn_points, self.chamfer_distance = knn_points, chamfer_distance
        elif method == "keops":
            from pykeops.torch import LazyTensor  # noqa: PLC0415

            self.lazy_tensor = LazyTensor

    def query(self):
        """Return sorted int64 neighbor indices and float32 squared distances."""
        if self.is_warp:
            ids = wp.empty((*self.queries.shape[:2], self.k), dtype=wp.int64)
            result = wp.knn(self.target_array, self.query_array, self.k, bvh=self.indexes, indices=ids)
            return result[1], result[2]
        with torch.no_grad():
            if self.method.startswith("pytorch3d"):
                version = self.tuning["forward"]["version"] if self.method.endswith("tuned") else -1
                result = self.knn_points(self.queries, self.points, K=self.k, version=version)
                return result.idx, result.dists
            if self.method == "keops":
                delta = self.lazy_tensor(self.queries[:, :, None, :]) - self.lazy_tensor(self.points[:, None, :, :])
                distances, ids = (delta**2).sum(-1).Kmin_argKmin(self.k, dim=2)
                return ids.long(), distances
            mode = "use_mm_for_euclid_dist" if self.method == "torch_mm" else "donot_use_mm_for_euclid_dist"
            matrix = torch.cdist(self.queries, self.points, compute_mode=mode)
            if self.k == 1:
                distances, ids = matrix.min(dim=2, keepdim=True)
            else:
                distances, ids = matrix.topk(self.k, dim=2, largest=False, sorted=True)
            return ids, distances.square()

    def chamfer(self):
        """Compute sum-of-means squared Chamfer and decoded-point gradients."""
        if self.is_warp:
            self.loss.zero_()
            self.query_array.grad.zero_()
            reverse_backend = wp.PointQueryBackend.CUBQL if self.method == "cubql_build" else wp.PointQueryBackend.WARP
            with wp.Tape() as tape:
                forward = wp.knn(self.target_array, self.query_array, 1, bvh=self.indexes)
                reverse = wp.knn(
                    self.query_array,
                    self.target_array,
                    1,
                    backend=reverse_backend,
                    objective=wp.PointQueryObjective.BUILD_AND_QUERY,
                )
                wp.launch(
                    mean_distance_kernel, self.query_array.shape, [forward[2], self.loss, 1.0 / self.query_array.size]
                )
                wp.launch(
                    mean_distance_kernel, self.target_array.shape, [reverse[2], self.loss, 1.0 / self.target_array.size]
                )
            tape.backward(self.loss)
            return self.loss, self.query_array.grad
        self.queries.grad = None
        if self.method == "pytorch3d":
            loss = self.chamfer_distance(
                self.queries, self.points, norm=2, batch_reduction="mean", point_reduction="mean"
            )[0]
        elif self.method == "pytorch3d_tuned":
            forward = self.knn_points(self.queries, self.points, K=1, version=self.tuning["forward"]["version"])
            reverse = self.knn_points(self.points, self.queries, K=1, version=self.tuning["reverse"]["version"])
            loss = forward.dists.mean() + reverse.dists.mean()
        elif self.method in ("keops", "torch_selected"):
            with torch.no_grad():
                if self.method == "keops":
                    delta = self.lazy_tensor(self.queries[:, :, None, :]) - self.lazy_tensor(self.points[:, None, :, :])
                    distances = (delta**2).sum(-1)
                    forward_ids, reverse_ids = distances.argmin(dim=2).long(), distances.argmin(dim=1).long()
                else:
                    distances = torch.cdist(self.queries, self.points, compute_mode="donot_use_mm_for_euclid_dist")
                    forward_ids = distances.argmin(dim=2, keepdim=True)
                    reverse_ids = distances.argmin(dim=1).unsqueeze(-1)
                    del distances
            forward_points = self.points.gather(1, forward_ids.expand(-1, -1, 3))
            reverse_points = self.queries.gather(1, reverse_ids.expand(-1, -1, 3))
            loss = (self.queries - forward_points).square().sum(-1).mean() + (
                self.points - reverse_points
            ).square().sum(-1).mean()
        else:
            mode = "use_mm_for_euclid_dist" if self.method == "torch_mm" else "donot_use_mm_for_euclid_dist"
            matrix = torch.cdist(self.queries, self.points, compute_mode=mode)
            loss = matrix.min(dim=2).values.square().mean() + matrix.min(dim=1).values.square().mean()
        loss.backward()
        return loss, self.queries.grad


def numpy_result(array):
    """Read a provider output after timing."""
    return array.numpy() if isinstance(array, wp.array) else array.detach().cpu().numpy()


def run_case(args, case, tuning):
    """Time one case and preserve correctness or allocation failures explicitly."""
    path = args.data / case["file"]
    if sha256(path) != case["file_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    with np.load(path) as arrays:
        points_host, queries_host = arrays["points"], arrays["queries"]
        points = torch.tensor(points_host, device="cuda")
        queries = torch.tensor(queries_host, device="cuda", requires_grad=args.phase == "chamfer")
        start = time.perf_counter()
        problem = Problem(args.method, args.phase, points, queries, case["k"], tuning)
        torch.cuda.synchronize()
        setup_ms = 1000 * (time.perf_counter() - start)
        operation = problem.query if args.phase == "query" else problem.chamfer
        timing, result = measure(operation, args.repeats)
        row = {"name": case["name"], "status": "ok", "setup_ms": setup_ms, "timing": timing}
        if args.phase == "query":
            ids, distances = map(numpy_result, result)
            row["validation"] = query_checks(points_host, queries_host, ids, distances, case["k"])
            row["passed"] = checks_pass(row["validation"])
            if ids.dtype != np.int64 or distances.dtype != np.float32:
                raise TypeError(f"Output contract mismatch: {ids.dtype}, {distances.dtype}")
        else:
            loss, gradient = map(numpy_result, result)
            reference_loss, reference_gradient = float(arrays["reference_loss"]), arrays["reference_gradient"]
            delta = gradient.astype(np.float64) - reference_gradient
            relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(reference_gradient), 1e-15))
            max_error = float(np.max(np.abs(delta)))
            tolerance = 1e-8 + 5e-4 * float(np.max(np.abs(reference_gradient)))
            row["validation"] = {
                "loss": float(loss.reshape(-1)[0]),
                "reference_loss": reference_loss,
                "gradient_relative_l2_error": relative_l2,
                "gradient_max_abs_error": max_error,
                "gradient_max_abs_tolerance": tolerance,
                "gradient_relative_l2_tolerance": 5e-4,
                "loss_rtol": 2e-4,
                "loss_atol": 1e-7,
            }
            row["passed"] = bool(
                np.isfinite(gradient).all()
                and relative_l2 <= 5e-4
                and max_error <= tolerance
                and np.isclose(float(loss.reshape(-1)[0]), reference_loss, rtol=2e-4, atol=1e-7)
            )
        return row


def main():
    """Prepare data, select public settings, or execute an isolated provider worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "tune", "worker"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("pulmonary_point_clouds.zip"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--pytorch3d-path", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--phase", choices=("query", "chamfer"))
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--case", help="Run one workload in a fresh process after a provider failure")
    args = parser.parse_args()
    sys.path.insert(0, str(args.pytorch3d_path.resolve()))
    wp.init()
    wp.set_device("cuda:0")
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        if args.action == "prepare":
            prepare(args)
            return
        if args.action == "tune":
            tune(args)
            return
        manifest_path = args.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        tuning = json.loads(args.tuning.read_text())["regimes"]
        report = {
            "method": args.method,
            "phase": args.phase,
            "environment": environment(),
            "device": wp.get_device().name,
            "sm_count": wp.get_device().sm_count,
            "repeats": args.repeats,
            "source_sha256": sha256(__file__),
            "manifest_sha256": sha256(manifest_path),
            "tuning_sha256": sha256(args.tuning),
            "cases": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cases = [manifest["fixture"]] if args.smoke else manifest["cases"]
        if args.case:
            cases = [case for case in cases if case["name"] == args.case]
            if not cases:
                raise ValueError(f"Unknown workload: {args.case}")
        for case in cases:
            if args.phase == "chamfer" and case["k"] != 1:
                continue
            try:
                row = run_case(args, case, tuning[regime(case)]["directions"])
            except torch.cuda.OutOfMemoryError as error:
                row = {"name": case["name"], "status": "out_of_memory", "error": str(error), "passed": False}
            except RuntimeError as error:
                message = str(error)
                status = (
                    "allocator_nvml_error"
                    if "NVML_SUCCESS == r" in message and "CUDACachingAllocator" in message
                    else "launch_error"
                    if "invalid configuration argument" in message
                    else "runtime_error"
                )
                row = {"name": case["name"], "status": status, "error": message, "passed": False}
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                # End this context. A coordinator may resume later cases in fresh processes.
                raise
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                args.method,
                args.phase,
                row["name"],
                row["status"],
                row["passed"],
                row.get("timing", {}).get("median_ms"),
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
