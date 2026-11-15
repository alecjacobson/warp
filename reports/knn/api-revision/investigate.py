# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recheck the two largest increases on identical queries with alternating API order."""

import gc
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import warp as wp
import warp.geometry as geometry
from warp._src.geometry import point_query
from warp.examples.benchmarks import benchmark_point_query_api_revision as revision
from warp.examples.benchmarks import benchmark_point_query_batched as retained

rng = np.random.default_rng(20261113)
clouds = {}
with zipfile.ZipFile("pulmonary_point_clouds.zip") as archive:
    for name in sorted(archive.namelist()):
        if name.endswith(".npz"):
            with archive.open(name) as stream, np.load(stream, allow_pickle=False) as data:
                clouds[Path(name).stem] = np.ascontiguousarray(data["points"], dtype=np.float32)
# Match all RNG consumption before query generation in the complete benchmark.
for points in list(clouds.values())[:4]:
    rng.choice(len(points), 65536, replace=False)
for size in (1000, 100000, 1000000):
    rng.uniform(-1, 1, (1, size, 3)).astype(np.float32)
cases = []
for name, points in clouds.items():
    for count in (1024, 8192):
        queries = points[rng.choice(len(points), count)][None].copy()
        queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)
        if name == "0105_H_PULM_H":
            cases.append((f"{name}_full_q{count}", points[None], queries))
retained.wp = SimpleNamespace(
    **(
        vars(wp)
        | {
            "knn": point_query._knn,
            "PointQueryBackend": geometry.PointQueryBackend,
            "PointQueryObjective": geometry.PointQueryObjective,
        }
    )
)
wp.init()
device = wp.get_device("cuda:0")
wp.set_mempool_release_threshold(device, 512 * 1024**2)
records = []
for round_index in range(3):
    for name, targets, queries in cases:
        order = [("geometry", revision.benchmark_case), ("engine", retained.benchmark_case)]
        if round_index % 2:
            order.reverse()
        for api, benchmark in order:
            gc.collect()
            result = benchmark(
                targets, queries, 1, geometry.PointQueryBackend.WARP, geometry.PointQueryObjective.REUSE, 31, device
            )
            records.append({"case": name, "round": round_index, "api": api, "result": result})
            print(
                name,
                round_index,
                api,
                {
                    key: value["median_ms"]
                    for key, value in result.items()
                    if isinstance(value, dict) and "median_ms" in value
                },
                flush=True,
            )
Path(__file__).with_name("investigation.json").write_text(json.dumps(records, indent=2) + "\n")
