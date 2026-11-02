# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible CPU/CUDA point-query timings, exhaustive checks, and PR comparison.

Run from the checkout: uv run --no-sync -m warp.examples.benchmarks.benchmark_point_query
    --output ../artifacts/results.json --scan ../data/DemoICPPointClouds.zip --physicsnemo
Requires NumPy; --physicsnemo additionally requires PyTorch. Timings exclude JIT.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.point_query_common import (
    make_api_query,
    make_ordered_query,
    make_query,
    make_tiled,
    query_codes,
)
from warp.examples.benchmarks.point_query_cubql import CubqlLibrary

PHYSICSNEMO_REV = "bc1c912528f0409c6853dbd33e00346a9184185c"
PHYSICSNEMO_URL = f"https://raw.githubusercontent.com/NVIDIA/physicsnemo/{PHYSICSNEMO_REV}/physicsnemo/nn/functional/neighbors/knn/_warp_impl.py"


def load_physicsnemo(directory):
    """Preserve source and change only the two framework imports, not algorithms."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "physicsnemo_original.py"
    if not raw.exists():
        urllib.request.urlretrieve(PHYSICSNEMO_URL, raw)
    source = raw.read_text()
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    (directory / "physicsnemo_adapter.py").write_text("""import torch
import warp as wp
class FunctionSpec:
    @staticmethod
    def warp_launch_context(tensor):
        stream = wp.stream_from_torch(torch.cuda.current_stream(tensor.device))
        return stream.device, stream
    @staticmethod
    def warp_stream_scope(stream):
        return wp.ScopedStream(stream)
def validate_inputs(points, queries):
    assert points.device == queries.device and points.dtype == queries.dtype
""")
    source = source.replace(
        "from physicsnemo.core.function_spec import FunctionSpec", "from physicsnemo_adapter import FunctionSpec"
    )
    source = source.replace("from .utils import validate_inputs", "from physicsnemo_adapter import validate_inputs")
    adapted = directory / "physicsnemo_standalone.py"
    adapted.write_text(source)
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("physicsnemo_standalone", adapted)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, sha


def scan_points(path):
    """Read the unresampled XYZ rows of Open3D's Redwood RGB-D scan fragment."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("cloud_bin_0.pcd"))
        raw = archive.read(name)
    header, data = raw.split(b"DATA binary\n", 1)
    fields = next(line for line in header.decode().splitlines() if line.startswith("FIELDS ")).split()[1:]
    count = int(next(line for line in header.decode().splitlines() if line.startswith("POINTS ")).split()[1])
    values = np.frombuffer(data, dtype="<f4").reshape(count, len(fields))
    return values[:, [fields.index(a) for a in ("x", "y", "z")]].copy()


def cloud(kind, n, q, noise, seed, scan=None):
    rng = np.random.default_rng(seed)
    if kind == "scan":
        p = scan.copy()
        n = len(p)
        p = (p - p.mean(axis=0)) / np.max(np.ptp(p, axis=0))
        queries = p[rng.integers(n, size=q)] + rng.normal(scale=0.003, size=(q, 3))
    elif kind == "surface":
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        queries = rng.normal(size=(q, 3))
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    elif kind == "volume":
        p = rng.uniform(-1, 1, (n, 3))
        queries = rng.uniform(-1, 1, (q, 3))
    elif kind == "clusters":
        centers = rng.uniform(-1, 1, (8, 3))
        p = centers[rng.integers(8, size=n)] + rng.normal(scale=0.015, size=(n, 3))
        queries = rng.uniform(-1, 1, (q, 3))
    else:
        raise ValueError(kind)
    if noise:
        p += rng.normal(scale=noise, size=p.shape)
        queries += rng.normal(scale=noise, size=queries.shape)
    return np.asarray(p, np.float32), np.asarray(queries, np.float32)


def measure(fn, device, repeats, graph=False):
    for _ in range(2):
        result = fn()
    wp.synchronize_device(device)
    iterations = 1
    if graph and device.is_cuda:
        iterations = 20
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(iterations):
                fn()

        def fn():
            return wp.capture_launch(capture.graph)

    gpu, wall = [], []
    if device.is_cuda:
        begin, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
    for _ in range(repeats):
        wp.synchronize_device(device)
        start = time.perf_counter()
        if device.is_cuda:
            wp.record_event(begin)
        result = fn()
        if device.is_cuda:
            wp.record_event(end)
        wp.synchronize_device(device)
        wall.append(1000 * (time.perf_counter() - start))
        gpu.append((wp.get_event_elapsed_time(begin, end) if device.is_cuda else wall[-1]) / iterations)
    return {
        "median_ms": statistics.median(gpu),
        "min_ms": min(gpu),
        "max_ms": max(gpu),
        "wall_median_ms": statistics.median(wall) / iterations,
        "samples_ms": gpu,
        "graph_iterations": iterations,
    }, result


_oracle_cache = None


def validate(p, q, ids, k):
    # Reuse the oracle within one immutable workload, outside every timed region.
    global _oracle_cache
    if _oracle_cache is None or _oracle_cache[0] is not p or _oracle_cache[1] is not q or _oracle_cache[2] != k:
        rows = np.unique(np.linspace(0, len(q) - 1, min(32, len(q)), dtype=int))
        original = np.arange(len(p))
        expected32, expected64 = [], []
        p64 = p.astype(np.float64)
        for row in rows:
            d = (p - q[row]) ** 2
            d2 = (d[:, 0] + d[:, 1]) + d[:, 2]
            expected32.append(np.lexsort((original, d2))[:k])
            d64 = np.sum((p64 - q[row]) ** 2, axis=1)
            expected64.append(np.lexsort((original, d64))[:k])
        _oracle_cache = p, q, k, rows, np.array(expected32), np.array(expected64)
    _, _, _, rows, expected32, expected64 = _oracle_cache
    actual = ids[rows, : min(k, len(p))]
    return {
        "checked_rows": len(rows),
        "float32_mismatch_rows": int(np.any(expected32 != actual, axis=1).sum()),
        "float64_mismatch_rows": int(np.any(expected64 != actual, axis=1).sum()),
    }


def run_case(p, q, k, device, repeats, baseline=None, tune=False, cubql=None):
    n, nq = len(p), len(q)
    transfer, arrays = measure(
        lambda: (wp.array(p, dtype=wp.vec3, device=device), wp.array(q, dtype=wp.vec3, device=device)), device, repeats
    )
    points, queries = arrays
    excludes = wp.full(nq, -1, dtype=int, device=device)
    oi = wp.empty((nq, k), dtype=int, device=device)
    od = wp.empty((nq, k), dtype=float, device=device)
    counts = wp.empty(nq, dtype=int, device=device)
    out = {"N": n, "Q": nq, "k": k, "transfer_allocation": transfer, "variants": {}}
    variants = out["variants"]
    constructors = ("lbvh", "cubql") if device.is_cuda else ("sah", "median", "cubql")
    for constructor in constructors:
        for leaf in (1, 4, 8) if tune else (4,):
            build, tree = measure(
                lambda constructor=constructor, leaf=leaf: wp.Bvh(
                    points, points, constructor=constructor, leaf_size=leaf
                ),
                device,
                repeats,
            )
            args = [tree.id, points, queries, np.inf, excludes, oi, od, counts]
            for method in ("api", "bvh", "bvh_sorted"):
                kernel = make_api_query(k) if method == "api" else make_query(k, method)

                def call(kernel=kernel, args=args):
                    return wp.launch(kernel, nq, inputs=args, device=device)

                timing, _ = measure(call, device, repeats, graph=True)
                check = validate(p, q, oi.numpy(), k)
                if check["float32_mismatch_rows"]:
                    raise AssertionError((constructor, leaf, method, check))
                variants[f"{method}/{constructor}/leaf{leaf}"] = {"query": timing, "build": build, "validation": check}
            out64 = wp.empty((nq, k), dtype=wp.int64, device=device)
            keys = wp.empty(2 * nq, dtype=int, device=device)
            order = wp.empty_like(keys)

            def query_plan(tree=tree, keys=keys, order=order):
                wp.launch(query_codes, nq, inputs=[tree.id, queries, keys, order], device=device)
                wp.utils.radix_sort_pairs(keys, order, nq)

            plan, _ = measure(query_plan, device, repeats)
            for ordered in (False, True):
                kernel64 = make_ordered_query(k, True) if ordered else make_api_query(k, True)
                args64 = (
                    [tree.id, points, queries, np.inf, excludes] + ([order] if ordered else []) + [out64, od, counts]
                )
                timing, _ = measure(
                    lambda kernel64=kernel64, args64=args64: wp.launch(kernel64, nq, inputs=args64, device=device),
                    device,
                    repeats,
                    graph=True,
                )
                check = validate(p, q, out64.numpy(), k)
                if check["float32_mismatch_rows"]:
                    raise AssertionError(("ordered" if ordered else "api64", check))
                name = "ordered64" if ordered else "api64"
                variants[f"{name}/{constructor}/leaf{leaf}"] = {"query": timing, "build": build, "validation": check}
                if ordered:
                    variants[f"{name}/{constructor}/leaf{leaf}"]["query_plan"] = plan
            refit, _ = measure(tree.refit, device, repeats, graph=True)
            rebuild, _ = measure(tree.rebuild, device, repeats)
            variants[f"bvh/{constructor}/leaf{leaf}"].update(refit=refit, rebuild=rebuild)
    # One-shot device-resident allocation/build/query time, with equivalent full outputs.
    kernel = make_api_query(k, euclidean64=True)

    def one_shot():
        tree = wp.Bvh(points, points, constructor=constructors[0], leaf_size=4)
        idx = wp.empty((nq, k), dtype=wp.int64, device=device)
        ds = wp.empty((nq, k), dtype=float, device=device)
        ct = wp.empty(nq, dtype=int, device=device)
        wp.launch(kernel, nq, inputs=[tree.id, points, queries, np.inf, excludes, idx, ds, ct], device=device)
        return tree, idx, ds, ct

    out["one_shot"], _ = measure(one_shot, device, repeats)
    ordered_kernel = make_ordered_query(k, True)

    def ordered_one_shot():
        tree = wp.Bvh(points, points, constructor=constructors[0], leaf_size=4)
        idx = wp.empty((nq, k), dtype=wp.int64, device=device)
        ds = wp.empty((nq, k), dtype=float, device=device)
        ct = wp.empty(nq, dtype=int, device=device)
        keys = wp.empty(2 * nq, dtype=int, device=device)
        order = wp.empty_like(keys)
        wp.launch(query_codes, nq, inputs=[tree.id, queries, keys, order], device=device)
        wp.utils.radix_sort_pairs(keys, order, nq)
        wp.launch(
            ordered_kernel, nq, inputs=[tree.id, points, queries, np.inf, excludes, order, idx, ds, ct], device=device
        )
        return tree, idx, ds, ct, keys, order

    out["ordered_one_shot"], _ = measure(ordered_one_shot, device, repeats)
    tree = wp.Bvh(points, points, constructor=constructors[0], leaf_size=4)
    args = [tree.id, points, queries, np.inf, excludes, oi, od, counts]
    for method in ("brute", "brute_sorted", "sphere"):
        if method != "sphere" and n * nq > 20_000_000:
            continue
        kernel = make_query(k, method)
        timing, _ = measure(
            lambda kernel=kernel: wp.launch(kernel, nq, inputs=args, device=device), device, repeats, graph=True
        )
        check = validate(p, q, oi.numpy(), k)
        if check["float32_mismatch_rows"]:
            raise AssertionError((method, check))
        variants[method] = {"query": timing, "validation": check}
    if n * nq <= 20_000_000:
        for bp, block, partitions in ((128, 64, 1), (128, 64, 32), (256, 128, 32)):
            tiled, merge = make_tiled(k, bp)
            pi = wp.empty((nq, partitions, k), dtype=int, device=device)
            pd = wp.empty((nq, partitions, k), dtype=float, device=device)

            def tiled_call(tiled=tiled, merge=merge, partitions=partitions, pi=pi, pd=pd, block=block):
                wp.launch_tiled(
                    tiled,
                    dim=(nq, partitions),
                    inputs=[points, queries, np.inf, excludes, partitions, pi, pd],
                    block_dim=block,
                    device=device,
                )
                wp.launch(merge, nq, inputs=[pi, pd, oi, od, counts], device=device)

            timing, _ = measure(tiled_call, device, repeats, graph=True)
            check = validate(p, q, oi.numpy(), k)
            if check["float32_mismatch_rows"]:
                raise AssertionError(("tiled", check))
            variants[f"tiled/bp{bp}/block{block}/partitions{partitions}"] = {
                "query": timing,
                "validation": check,
                "scratch_bytes": nq * partitions * k * 8,
            }
    if cubql is not None:
        build, ct = measure(lambda: cubql.build(points), device, repeats)
        timing, _ = measure(lambda: ct.query(queries, k, np.inf, excludes, oi, od, counts), device, repeats, graph=True)
        check = validate(p, q, oi.numpy(), k)
        if check["float32_mismatch_rows"]:
            raise AssertionError(("cuBQL", check))
        variants["cubql_direct"] = {"build": build, "query": timing, "validation": check}
    if baseline is not None:
        torch = importlib.import_module("torch")

        tp = torch.tensor(p, device="cuda")
        tq = torch.tensor(q, device="cuda")
        build, bt = measure(lambda: baseline.build_morton_tree(tp), device, repeats)
        plan, (order, window) = measure(lambda: baseline.morton_query_plan(bt, tq), device, repeats)
        # Launch the unchanged PR query kernel into preallocated outputs for the
        # reusable-index timing; include its normal allocations in one-shot timing.
        result = baseline.tree_knn_query(bt, tq, k, order, window)
        native_args = [
            wp.from_torch(bt["tree_points"], dtype=wp.vec3),
            wp.from_torch(bt["order"]),
            wp.from_torch(bt["node_lo"], dtype=wp.vec3),
            wp.from_torch(bt["node_hi"], dtype=wp.vec3),
        ]
        native_args += [wp.from_torch(bt[key]) for key in ("left", "right", "leaf_start", "leaf_count")]
        native_args += [
            wp.from_torch(tq, dtype=wp.vec3),
            wp.from_torch(order),
            wp.from_torch(window),
            k,
            wp.from_torch(result[0]),
            wp.from_torch(result[1]),
        ]
        on_kernel = baseline.get_tree_knn_kernel(*baseline._kernel_key(k))
        query, _ = measure(
            lambda: wp.launch(on_kernel, nq, inputs=native_args, device=device), device, repeats, graph=True
        )
        check = validate(p, q, result[0].cpu().numpy(), k)
        one, _ = measure(lambda: baseline.knn_impl(tp, tq, k), device, repeats)
        variants["physicsnemo"] = {
            "build": build,
            "query_plan": plan,
            "query": query,
            "one_shot": one,
            "validation": check,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--cubql-library", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physicsnemo", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if args.physicsnemo and not device.is_cuda:
        parser.error("The PhysicsNeMo backend requires CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline, source_sha = load_physicsnemo(args.output.parent / "physicsnemo") if args.physicsnemo else (None, None)
    scan = scan_points(args.scan) if args.scan else None
    cubql = None
    if args.cubql_library:
        cubql = CubqlLibrary(args.cubql_library)
    report = {
        "warp_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "physicsnemo_revision": PHYSICSNEMO_REV,
        "physicsnemo_source_sha256": source_sha,
        "device": device.name,
        "arch": device.arch if device.is_cuda else platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "cubql_library_sha256": hashlib.sha256(args.cubql_library.read_bytes()).hexdigest()
        if args.cubql_library
        else None,
        "precision": "float32, direct differences, noncontracting distance/bounds helper",
        "scan_sha256": hashlib.sha256(args.scan.read_bytes()).hexdigest() if args.scan else None,
        "cases": [],
    }
    sizes = [(32, 32, 8), (1000, 1000, 8), (100_000, 1, 8), (100_000, 8192, 11)]
    if not args.quick:
        sizes += [
            (1_000_000, 100_000, 1),
            (1_000_000, 100_000, 11),
            (128, 1, 64),
            (1000, 32, 4),
            (1000, 32, 16),
            (1000, 32, 32),
        ]
    for kind in ("surface", "volume", "scan") if scan is not None else ("surface", "volume"):
        for noise in (0.0, 0.01):
            for n, nq, k in [(len(scan), 8192, 11)] if kind == "scan" else sizes:
                print(f"Running {kind} noise={noise} N={n} Q={nq} k={k}", flush=True)
                p, q = cloud(kind, n, nq, noise, 531, scan)
                case = run_case(p, q, k, device, args.repeats, baseline, args.tune, cubql)
                case.update(distribution=kind, gaussian_sigma=noise, seed=531)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2))
                on = case["variants"].get("physicsnemo", {}).get("one_shot", {}).get("median_ms")
                print(f"  one-shot {case['one_shot']['median_ms']:.4f} ms; PhysicsNeMo {on}", flush=True)


if __name__ == "__main__":
    main()
