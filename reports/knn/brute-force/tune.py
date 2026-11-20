# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep internal target tiles and thread blocks; validate every selected ID."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry
from warp._src.geometry.point_query_brute_force import query_kernel
from warp.examples.benchmarks.benchmark_point_query_chamfer import load_clouds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
wp.init()
clouds = load_clouds(args.archive)
records = []
for kind, b, n, k in [
    ("volume", 32, 1024, 1),
    ("noisy_volume", 32, 1024, 1),
    ("noisy_pulmonary", 32, 1024, 1),
    ("surface", 32, 1024, 1),
    ("noisy_surface", 32, 1024, 1),
    ("pulmonary", 32, 1024, 1),
    ("pulmonary", 1, 1024, 1),
    ("pulmonary", 32, 4096, 1),
    ("volume", 4, 259, 11),
]:
    rng = np.random.default_rng(np.random.SeedSequence([2348, b, n]))
    if "pulmonary" in kind:
        cs = [clouds[i % len(clouds)] for i in range(b)]
        p = np.stack([c[rng.choice(len(c), n, replace=False)] for c in cs])
        q = np.stack([c[rng.choice(len(c), n, replace=False)] for c in cs])
    else:
        p = rng.uniform(-1, 1, size=(b, n, 3)).astype("f")
        q = rng.uniform(-1, 1, size=(b, n, 3)).astype("f")
        if "surface" in kind:
            p = rng.normal(size=(b, n, 3)).astype("f")
            q = rng.normal(size=(b, n, 3)).astype("f")
        if "surface" in kind:
            p /= np.linalg.norm(p, axis=2, keepdims=True)
            q /= np.linalg.norm(q, axis=2, keepdims=True)
    if kind.startswith("noisy_"):
        p += rng.normal(scale=0.02, size=p.shape).astype("f")
        q += rng.normal(scale=0.02, size=q.shape).astype("f")
    pp = wp.array(p, dtype=wp.vec3)
    qq = wp.array(q, dtype=wp.vec3)
    oracle = wp.geometry.knn(pp, qq, k)
    expected = oracle[1].numpy()
    ids = wp.empty((b, n, k), dtype=int)
    ds = wp.empty(ids.shape, dtype=float)
    cts = wp.empty((b, n), dtype=int)
    inputs = [pp.flatten(), qq.flatten(), n, n, float("inf"), None, ids.flatten(), ds.flatten(), cts.flatten()]
    candidates = [(tile, block) for tile in (0, 32, 64, 128, 256, 512) for block in (64, 128, 256)]
    rng.shuffle(candidates)
    for tile, block in candidates:
        kernel = query_kernel(k, wp.int32, True, False, tile)

        def run(kernel=kernel, b=b, n=n, block=block, inputs=inputs):
            wp.launch(kernel, (b, (n + block - 1) // block, block), inputs, block_dim=block, record_tape=False)

        run()
        np.testing.assert_array_equal(ids.numpy(), expected)
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                run()
        samples = []
        for _ in range(7):
            wp.synchronize()
            start = time.perf_counter()
            wp.capture_launch(capture.graph)
            wp.synchronize()
            samples.append((time.perf_counter() - start) * 100)
        row = {
            "kind": kind,
            "batches": b,
            "n": n,
            "k": k,
            "tile": tile,
            "block": block,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "correct": True,
        }
        records.append(row)
        print(kind, b, n, k, tile, block, row["median_ms"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "device": wp.get_device().name,
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing": "synchronized wall / 10 captured launches; 7 repetitions",
                "kernel_sha256": hashlib.sha256(
                    Path("warp/_src/geometry/point_query_brute_force.py").read_bytes()
                ).hexdigest(),
                "cases": records,
            },
            indent=2,
        )
    )
