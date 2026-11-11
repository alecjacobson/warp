# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled Warp/PhysicsNeMo hybrid ablations, keeping the old baseline intact."""

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy, environment, measure
from warp.examples.benchmarks.point_query_backends import Provider
from warp.examples.benchmarks.point_query_hybrid import Hybrid


def physics_isotropic(provider, directory, tuned=False):
    """Generate explicit PR variants for metric scaling and sparse launches."""
    source = Path(provider.library.__file__).read_text()
    old = "extent = points.max(dim=0).values - lo"
    if source.count(old) != 1:
        raise ValueError("Pinned PhysicsNeMo extent expression changed")
    source = source.replace(old, "extent = (points.max(dim=0).values - lo).max().expand(3)")
    name = "physicsnemo_tuned" if tuned else "physicsnemo_isotropic"
    if tuned:
        source = source.replace(
            "def get_tree_knn_kernel(capacity: int, static_k: bool):",
            "def get_tree_knn_kernel(capacity: int, static_k: bool, stride: int = 1):",
        )
        source = source.replace("key = (capacity, static_k)", "key = (capacity, static_k, stride)")
        source = source.replace(
            "qi = query_order[wp.tid()]",
            "tid = wp.tid()\n        if tid % wp.static(stride) != 0:\n            return\n        qi = query_order[tid // wp.static(stride)]",
        )
        source = source.replace(
            "kernel = get_tree_knn_kernel(*_kernel_key(k))",
            "stride = 1\n    while stride < 32 and m * stride < 512 * wp.device_from_torch(queries.device).sm_count:\n        stride *= 2\n    kernel = get_tree_knn_kernel(*_kernel_key(k), stride)",
        )
        source = source.replace("dim=m,", "dim=m * stride,\n            block_dim=32 if stride > 1 else 256,")
        source = source.replace(
            "query_order = torch.argsort(q_codes).int()",
            "query_order = (torch.arange(len(queries), dtype=torch.int32, device=queries.device)\n        if len(queries) < 512 * wp.device_from_torch(queries.device).sm_count\n        else torch.argsort(q_codes).int())",
        )
    source = source.replace("physicsnemo::knn_warp", f"physicsnemo::{name}_experiment")
    path = directory / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    variant = copy.copy(provider)
    variant.library = module
    return variant, hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../data/knn-comparison/pilot/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--isotropic", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--morton-seed", action="store_true")
    parser.add_argument("--carry", action="store_true")
    parser.add_argument("--runs", action="store_true")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--physics-isotropic", action="store_true")
    parser.add_argument("--blocks", action="store_true")
    parser.add_argument("--physics-tuned", action="store_true")
    args = parser.parse_args()
    wp.init()
    wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
    torch.set_grad_enabled(False)
    warp_stream = wp.Stream("cuda:0")
    stream = wp.stream_to_torch(warp_stream)
    out = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = environment()
    metadata.update(shared_stream="Warp-owned blocking stream", settings={k: str(v) for k, v in vars(args).items()})
    args.output.with_suffix(".environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        physics = Provider("physicsnemo", Path("../artifacts/comparison/pilot"))
        original = Provider("warp_configured", None)
        tuned = None
        if args.physics_tuned:
            tuned, digest = physics_isotropic(physics, args.output.parent / "physics-tuned", tuned=True)
            metadata["physics_tuned_sha256"] = digest
        iso = None
        if args.physics_isotropic:
            iso, digest = physics_isotropic(physics, args.output.parent / "physics-isotropic")
            metadata["physics_isotropic_sha256"] = digest
            args.output.with_suffix(".environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        args.output.with_suffix(".environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cases = json.loads(args.manifest.read_text())["cases"]
        for case in cases:
            if args.filter and not any(s in case["id"] for s in args.filter.split(",")):
                continue
            data = np.load(args.manifest.parent / case["file"])
            p, q = data["points"], data["queries"][: case["Q"]]
            pt, qt = torch.as_tensor(p, device="cuda"), torch.as_tensor(q, device="cuda")
            variants = [
                ("warp_configured", original.bind(pt, qt, case["k"])),
                ("physicsnemo", physics.bind(pt, qt, case["k"])),
            ]
            if iso is not None:
                variants.append(("physicsnemo_isotropic", iso.bind(pt, qt, case["k"])))
            if tuned is not None:
                variants.append(("physicsnemo_tuned", tuned.bind(pt, qt, case["k"])))
            if args.blocks:
                from warp.examples.benchmarks.point_query_common import make_api_query  # noqa: PLC0415

                for block in (32, 64, 128, 256):
                    for stride in (1, 2, 4, 8, 16, 32) if block == 32 else (1,):
                        problem = original.bind(pt, qt, case["k"])
                        problem.block_dim = block
                        problem.stride = stride
                        problem.ordered = False
                        problem.kernel = make_api_query(case["k"], True, stride)
                        variants.append((f"warp-block{block}-stride{stride}", problem))
            configs = (
                itertools.product([4, 8, 32], [False, True], [False, True], [False, True], ["heap", "sorted"])
                if args.sweep
                else [(8, True, True, True, "heap")]
            )
            if args.carry:
                configs = itertools.product([4, 8, 32], [False, True], [False, True], ["carry"], ["heap"])
            if args.runs:
                configs = itertools.product([4, 8, 32], [False, True], [False], [False], ["heap"])
            if args.cached:
                configs = itertools.product([4, 8, 32], [False, True], [False, True], ["cached"], ["heap"])
            if args.baseline_only or args.blocks:
                configs = []
            for leaf, ordered, packed, seed, selector in configs:
                seeding = "morton" if args.morton_seed and seed else seed
                name = f"l{leaf}-o{int(ordered)}-p{int(packed)}-s{seeding}-{selector}"
                variants.append(
                    (name, Hybrid(pt, qt, case["k"], leaf, ordered, packed, seeding, selector, args.isotropic))
                )
            row = {"case": case, "variants": {}}
            for name, problem in variants:
                fn = problem.one_shot
                timing, (_, result) = measure(fn, args.repeats)
                valid = accuracy(p, q, result[0].cpu().numpy(), result[1].cpu().numpy(), case["k"])
                row["variants"][name] = {
                    "one_shot": timing,
                    "accuracy": valid,
                    "configuration": {
                        key: getattr(problem, key, None) for key in ("leaf", "stride", "block_dim", "ordered")
                    },
                }
                if not name.startswith("physicsnemo"):
                    bt, index = measure(problem.build, args.repeats)
                    st, _ = measure(lambda problem=problem, index=index: problem.search(index), args.repeats)
                    row["variants"][name].update(build=bt, search=st)
                print(
                    case["id"],
                    name,
                    round(timing["wall_median_ms"], 3),
                    valid["distance_rank_mismatch_rows"],
                    flush=True,
                )
            out.append(row)
            args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
