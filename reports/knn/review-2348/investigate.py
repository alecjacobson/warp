# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternate original and reviewed Python implementations on identical inputs."""

import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

import warp as wp
from warp._src.geometry import point_query as reviewed
from warp.examples.benchmarks import benchmark_point_query_api_revision as harness

root = Path(__file__).resolve().parent
old_path = root / "original_point_query.py"
old_path.write_bytes(
    subprocess.check_output(
        ["git", "show", "fb53789563df025bf6d4de6569b730b7fd2692dd:warp/_src/geometry/point_query.py"]
    )
)
spec = importlib.util.spec_from_file_location("original_point_query", old_path)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
rng = np.random.default_rng(20261113)
clouds = {}
with zipfile.ZipFile("pulmonary_point_clouds.zip") as archive:
    for name in sorted(archive.namelist()):
        if name.endswith(".npz"):
            with archive.open(name) as stream, np.load(stream, allow_pickle=False) as data:
                clouds[Path(name).stem] = np.ascontiguousarray(data["points"], dtype=np.float32)
cases = []
for name, points in clouds.items():
    for count in (1024, 8192):
        cases.append((f"{name}_full_q{count}", points[None], count, 1))
largest = max(clouds.values(), key=len)
for batches in (2, 4):
    cases.append((f"largest_full_b{batches}", np.repeat(largest[None], batches, axis=0), 8192, 1))
mixed = np.stack([p[rng.choice(len(p), 65536, replace=False)] for p in list(clouds.values())[:4]])
cases.append(("mixed_subset_b4_m65536", mixed, 8192, 1))
for size, count, k in ((1000, 32, 32), (100000, 1024, 32), (1000000, 8192, 11)):
    cases.append((f"synthetic_volume_m{size}_k{k}", rng.uniform(-1, 1, (1, size, 3)).astype(np.float32), count, k))
selected = {
    "0115_H_PULM_ALGS_full_q8192": ("CUBQL", "BUILD_AND_QUERY"),
    "0116_H_PULM_ALGS_full_q1024": ("CUBQL", "BUILD_AND_QUERY"),
    "0117_H_PULM_ALGS_full_q1024": ("CUBQL", "REUSE"),
    "0102_H_PULM_H_full_q1024": ("WARP", "REUSE"),
    "synthetic_volume_m1000_k32": ("CUBQL", "BUILD_AND_QUERY"),
}
inputs = []
for name, targets, count, k in cases:
    queries = np.stack([p[rng.choice(len(p), count)] for p in targets])
    queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)
    if name in selected:
        inputs.append((name, targets, queries, k))
wp.init()
device = wp.get_device("cuda:0")
wp.set_mempool_release_threshold(device, 512 * 1024**2)
records = []
for round_index in range(3):
    for name, targets, queries, k in inputs:
        order = [("original", original), ("reviewed", reviewed)]
        if round_index % 2:
            order.reverse()
        for label, module in order:
            gc.collect()
            harness.geometry = module
            backend, objective = selected[name]
            result = harness.benchmark_case(
                targets,
                queries,
                k,
                getattr(module.PointQueryBackend, backend),
                getattr(module.PointQueryObjective, objective),
                31,
                device,
            )
            records.append(
                {
                    "case": name,
                    "round": round_index,
                    "implementation": label,
                    "backend": backend,
                    "objective": objective,
                    "result": result,
                }
            )
            print(
                name,
                round_index,
                label,
                {p: round(t["median_ms"], 5) for p, t in result.items() if isinstance(t, dict) and "median_ms" in t},
                flush=True,
            )
            (root / "investigation.json").write_text(json.dumps(records, indent=2) + "\n")
print("Original source SHA256:", hashlib.sha256(old_path.read_bytes()).hexdigest())
