# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure batched Chamfer forward/backward and fixed-index queries on CUDA.

Synthetic measurements are not reproductions of customer pulmonary scans.
"""

import argparse
import cProfile
import hashlib
import json
import statistics
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry
from warp.examples.geometry.example_point_query_chamfer import mean_distance_kernel


def measure(fn, repeats, device):
    for _ in range(2):
        fn()
    wp.synchronize_device(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        samples.append(1000 * (time.perf_counter() - start))
    return {"wall_median_ms": statistics.median(samples), "wall_samples_ms": samples}


def load_clouds(archive_path):
    """Read point arrays without extracting files or allowing pickled payloads."""
    clouds = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".npz"):
                with archive.open(name) as stream, np.load(stream, allow_pickle=False) as arrays:
                    points = np.ascontiguousarray(arrays["points"], dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid finite 3D point cloud: {name}")
                clouds.append(points)
    if not clouds:
        raise ValueError("The archive must contain .npz files with points arrays")
    return clouds


def validate_neighbors(targets, queries, output):
    """Check all output IDs and sampled exhaustive float32 nearest distances."""
    ids, ds, counts = (a.numpy() for a in output[1:])
    if not np.all((ids >= 0) & (ids < targets.shape[1])) or not np.all(counts == 1):
        raise ValueError("Invalid nearest-neighbor IDs or counts")
    for batch in range(len(targets)):
        for row in np.linspace(0, queries.shape[1] - 1, min(8, queries.shape[1]), dtype=int):
            delta = targets[batch] - queries[batch, row]
            d = np.sum(delta * delta, axis=1)
            np.testing.assert_allclose(d[ids[batch, row, 0]], d.min(), rtol=3e-5, atol=1e-7)
            np.testing.assert_allclose(ds[batch, row, 0], d.min(), rtol=3e-5, atol=1e-7)


def run(args):
    wp.init()
    device = wp.get_device(args.device)
    clouds = load_clouds(args.archive) if args.archive else None
    result = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": device.name,
        "sm_count": device.sm_count,
        "seed": 2348,
        "seed_policy": "SeedSequence([2348, batches, points])",
        "source": "subsampled archive clouds" if clouds else "synthetic Gaussian volume",
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest() if clouds else None,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("warp/_src/geometry/point_query.py", "warp/native/bvh_cubql.cu", "warp/bin/warp.so", __file__)
        },
        "repeats": args.repeats,
        "timing": "synchronized wall; two warmups; uploads and compilation excluded",
        "cases": [],
    }
    with wp.ScopedDevice(device):
        wp.set_mempool_release_threshold(device, 512 * 1024**2)
        for batches in args.batches:
            for n in args.points:
                rng = np.random.default_rng(np.random.SeedSequence([2348, batches, n]))
                if clouds:
                    eligible = [c for c in clouds if len(c) >= n]
                    if not eligible:
                        raise ValueError(f"No archive cloud has at least {n} points")
                    selected = [eligible[b % len(eligible)] for b in range(batches)]
                    p = np.stack([c[rng.choice(len(c), n, replace=False)] for c in selected])
                    q = np.stack([c[rng.choice(len(c), n, replace=False)] for c in selected])
                else:
                    p = rng.normal(size=(batches, n, 3)).astype(np.float32)
                    q = rng.normal(size=(batches, n, 3)).astype(np.float32)
                for backend_name in args.backends:
                    backend = getattr(wp.geometry.PointQueryBackend, backend_name.upper())
                    targets = wp.array(p, dtype=wp.vec3, requires_grad=True)
                    queries = wp.array(q, dtype=wp.vec3, requires_grad=True)
                    fixed = wp.geometry.PointIndex(targets, num_queries=n, backend=backend)
                    query_targets = wp.geometry.PointIndex(queries, num_queries=n, backend=backend)
                    loss = wp.zeros(1, dtype=float, requires_grad=True)
                    outputs = [wp.geometry.knn(fixed, queries, 1), wp.geometry.knn(query_targets, targets, 1)]
                    buffers = [{"indices": o[1], "distances": o[2], "counts": o[3]} for o in outputs]
                    validate_neighbors(p, q, outputs[0])
                    validate_neighbors(q, p, outputs[1])

                    def query(fixed=fixed, queries=queries, buffers=buffers):
                        wp.geometry.knn(fixed, queries, 1, **buffers[0])

                    def chamfer(
                        reuse,
                        fixed=fixed,
                        query_targets=query_targets,
                        queries=queries,
                        targets=targets,
                        loss=loss,
                        buffers=buffers,
                        backend=backend,
                    ):
                        loss.zero_()
                        with wp.Tape() as tape:
                            first = wp.geometry.knn(fixed, queries, 1, **buffers[0])
                            second = wp.geometry.knn(
                                query_targets if reuse else queries, targets, 1, backend=backend, **buffers[1]
                            )
                            wp.launch(mean_distance_kernel, queries.shape, [first[2], loss, 1.0 / queries.size])
                            wp.launch(mean_distance_kernel, targets.shape, [second[2], loss, 1.0 / targets.size])
                        tape.backward(loss)
                        tape.zero()

                    query()
                    with wp.ScopedCapture(device=device) as capture:
                        query()
                    row = {
                        "batches": batches,
                        "points": n,
                        "backend": backend_name,
                        "points_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
                        "queries_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                        "sampled_accuracy": "passed, up to 8 exhaustive rows per batch and direction",
                        "query": measure(query, args.repeats, device),
                        "query_graph": measure(
                            lambda graph=capture.graph: wp.capture_launch(graph), args.repeats, device
                        ),
                        "chamfer_reused": measure(lambda fn=chamfer: fn(True), args.repeats, device),
                        "chamfer_rebuild_reverse": measure(lambda fn=chamfer: fn(False), args.repeats, device),
                    }
                    if not np.isfinite(loss.numpy()).all():
                        raise ValueError("Nonfinite Chamfer loss")
                    result["cases"].append(row)
                    print(
                        batches,
                        n,
                        backend_name,
                        {k: round(v["wall_median_ms"], 3) for k, v in row.items() if isinstance(v, dict)},
                        flush=True,
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    if args.profile and batches == max(args.batches) and n == min(args.points):
                        profile = cProfile.Profile()
                        profile.runcall(chamfer, False)
                        profile.dump_stats(str(args.output.with_suffix(f".{backend_name}.prof")))
                    del query, chamfer, capture, outputs, buffers, fixed, query_targets
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--archive", type=Path, help="ZIP of .npz files containing points arrays; subsample without replacement"
    )
    parser.add_argument("--points", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--backends", nargs="+", choices=["warp", "cubql"], default=["warp", "cubql"])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.points) < 1 or min(args.batches) < 1:
        parser.error("repeats, point counts, and batch counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
