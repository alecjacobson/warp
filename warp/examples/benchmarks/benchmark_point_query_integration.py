# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Warp's native cuBQL integration with the standalone upstream adapter."""

import argparse
import ctypes
import gc
import hashlib
import json
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np

import warp as wp
from warp._src import context
from warp._src.point_query import _cubql_config
from warp.examples.benchmarks.benchmark_point_query_comparison import accuracy


class UpstreamLibrary:
    def __init__(self, path):
        self.core = ctypes.CDLL(str(path.resolve()))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.core.upstream_build.argtypes = [pointer, integer, integer, integer, pointer]
        self.core.upstream_build.restype = pointer
        self.core.upstream_destroy.argtypes = [pointer, pointer]
        self.core.upstream_destroy.restype = None
        self.core.upstream_query.argtypes = [
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            integer,
            integer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.core.upstream_query.restype = integer


def measure(operations, repeats, device, iterations=1):
    """Interleave warmed wall-clock samples with deterministic randomized order."""
    for operation in operations.values():
        for _ in range(3):
            operation()
    wp.synchronize_device(device)
    samples = {name: [] for name in operations}
    rng = np.random.default_rng(17)
    for _ in range(repeats):
        for name in rng.permutation(list(operations)):
            operation = operations[name]
            start = time.perf_counter()
            for _ in range(iterations):
                result = operation()
            wp.synchronize_device(device)
            samples[name].append((time.perf_counter() - start) * 1000 / iterations)
            del result
    return {name: {"median_ms": statistics.median(values), "samples_ms": values} for name, values in samples.items()}


def validate(targets, queries, outputs, k):
    indices, distances, counts = (array.numpy() for array in outputs)
    np.testing.assert_array_equal(counts, k)
    checks = []
    for batch in range(len(targets)):
        check = accuracy(targets[batch], queries[batch], indices[batch], distances[batch], k)
        if not (
            check.get("valid_shape")
            and check.get("valid_indices_and_finite_distances")
            and all(
                check.get(name, 1) == 0
                for name in (
                    "duplicate_rows",
                    "unsorted_rows",
                    "distance_rank_mismatch_rows",
                    "reported_distance_mismatch_rows",
                )
            )
        ):
            raise RuntimeError(f"Integration comparison failed validation: {check}")
        checks.append(check)
    return checks


def run_case(targets, samples, k, objective, libraries, repeats, device):
    points = wp.array(targets, dtype=wp.vec3, device=device)
    queries = wp.array(samples, dtype=wp.vec3, device=device)
    batches, num_points = points.shape
    num_queries = queries.shape[1]
    indexes = wp.build_point_index(
        points, num_queries=num_queries, k=k, backend=wp.PointQueryBackend.CUBQL, objective=objective
    )
    warp_indexes = wp.build_point_index(points, num_queries=num_queries, k=k)
    builder, leaf, stride, block = _cubql_config(num_points, num_queries, k, objective)
    upstream_builder = 4 if builder == 0 else builder
    point_views = [points[batch] for batch in range(batches)]
    query_views = [queries[batch] for batch in range(batches)]
    stream = wp.get_stream(device).cuda_stream
    trees = {}
    pool_threshold = wp.get_mempool_release_threshold(device)
    try:
        for name, library in libraries.items():
            trees[name] = []
            for cloud in point_views:
                handle = library.core.upstream_build(cloud.ptr, num_points, upstream_builder, leaf, stream)
                if not handle:
                    raise RuntimeError("Standalone cuBQL construction returned a null handle")
                trees[name].append(handle)
        # The upstream default resource changes the pool threshold. Restore the
        # same explicit application policy before timing any of the query routes.
        wp.set_mempool_release_threshold(device, pool_threshold)

        def allocate_outputs():
            return (
                wp.empty((batches, num_queries, k), dtype=wp.int64, device=device),
                wp.empty((batches, num_queries, k), dtype=float, device=device),
                wp.empty((batches, num_queries), dtype=int, device=device),
            )

        def public_query(outputs, index_set):
            wp.knn(
                points,
                queries,
                k,
                bvh=index_set,
                squared=False,
                indices=outputs[0],
                distances=outputs[1],
                counts=outputs[2],
            )
            return outputs

        def direct_query(outputs, name):
            for batch in range(batches):
                pointers = [output.ptr + batch * output.strides[0] for output in outputs]
                if name == "integrated_native":
                    success = context.runtime.core.wp_point_query_cubql_query_device(
                        indexes[batch].id,
                        point_views[batch].ptr,
                        query_views[batch].ptr,
                        num_queries,
                        k,
                        stride,
                        block,
                        *pointers,
                        True,
                        False,
                    )
                    if not success:
                        raise RuntimeError(context.runtime.get_error_string())
                else:
                    status = libraries[name].core.upstream_query(
                        trees[name][batch],
                        point_views[batch].ptr,
                        query_views[batch].ptr,
                        num_queries,
                        k,
                        stride,
                        block,
                        None,
                        *pointers,
                        stream,
                    )
                    if status:
                        raise RuntimeError(f"Standalone cuBQL query returned CUDA error {status}")
            return outputs

        operations = {}
        outputs_by_route = {}
        for name in ("integrated_public", "integrated_native", "warp", *libraries):
            outputs = allocate_outputs()
            outputs_by_route[name] = outputs
            if name in ("integrated_public", "warp"):
                selected = indexes if name == "integrated_public" else warp_indexes
                operations[name] = lambda outputs=outputs, selected=selected: public_query(outputs, selected)
            else:
                operations[name] = lambda outputs=outputs, name=name: direct_query(outputs, name)

        checks = {name: validate(targets, samples, operation(), k) for name, operation in operations.items()}
        ordinary = measure(operations, repeats, device)
        allocated = {
            "integrated_public": lambda: public_query(allocate_outputs(), indexes),
            "warp": lambda: public_query(allocate_outputs(), warp_indexes),
        }
        allocated.update({name: lambda name=name: direct_query(allocate_outputs(), name) for name in libraries})
        allocation_times = measure(allocated, repeats, device)

        graphs = {}
        for name, operation in operations.items():
            with wp.ScopedCapture(device=device) as capture:
                operation()
            graphs[name] = capture.graph
        graph_times = measure(
            {name: lambda graph=graph: wp.capture_launch(graph) for name, graph in graphs.items()},
            repeats,
            device,
            iterations=32,
        )
        shifted = samples + np.float32(0.003)
        queries.assign(shifted)
        for graph in graphs.values():
            wp.capture_launch(graph)
        graph_checks = {name: validate(targets, shifted, outputs, k) for name, outputs in outputs_by_route.items()}
        return {
            "objective": objective.name,
            "config": {"builder": upstream_builder, "leaf": leaf, "stride": stride, "block": block},
            "preallocated": ordinary,
            "allocated": allocation_times,
            "graph": graph_times,
            "validation": checks,
            "changed_query_graph_validation": graph_checks,
        }
    finally:
        wp.synchronize_device(device)
        for name, handles in trees.items():
            for handle in handles:
                libraries[name].core.upstream_destroy(handle, stream)
        wp.synchronize_device(device)
        wp.set_mempool_release_threshold(device, pool_threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--matched-library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    wp.init()
    device = wp.get_device("cuda:0")
    wp.set_mempool_release_threshold(device, 512 * 1024**2)
    libraries = {
        "standalone_matched": UpstreamLibrary(args.matched_library),
        "standalone_sm120": UpstreamLibrary(args.native_library),
    }
    rng = np.random.default_rng(20261114)
    cases = []
    for size, queries, k in ((1000, 32, 32), (100000, 1024, 1), (100000, 1024, 32), (1000000, 8192, 11)):
        points = rng.uniform(-1, 1, (1, size, 3)).astype(np.float32)
        cases.append((f"synthetic_m{size}_q{queries}_k{k}", points, queries, k))
    with zipfile.ZipFile(args.archive) as archive:
        largest = None
        for name in sorted(archive.namelist()):
            if name.endswith(".npz"):
                with archive.open(name) as stream, np.load(stream, allow_pickle=False) as data:
                    points = np.ascontiguousarray(data["points"], dtype=np.float32)[None]
                for queries in (1024, 8192):
                    cases.append((f"{Path(name).stem}_q{queries}", points, queries, 1))
                if largest is None or points.shape[1] > largest.shape[1]:
                    largest = points
        if largest is None:
            raise ValueError("The archive must contain point-cloud NPZ files")
        cases.append(("largest_full_b4", np.repeat(largest, 4, axis=0), 8192, 1))
    sources = (
        "warp/_src/point_query.py",
        "warp/native/bvh_cubql.cu",
        "warp/bin/warp.so",
        "warp/examples/benchmarks/point_query_cubql_upstream.cu",
        __file__,
    )
    report = {
        "device": device.name,
        "sm_count": device.sm_count,
        "repeats": args.repeats,
        "source_hashes": {str(name): hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        "standalone_hashes": {
            "matched": hashlib.sha256(args.matched_library.read_bytes()).hexdigest(),
            "sm120": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
        },
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "method": "Paired randomized-order synchronized wall samples. Preallocated and allocation-inclusive ordinary calls; graph samples average 32 replays. Same float32 points, int64 outputs, Euclidean distances and build/launch settings. Matched standalone uses compute_75 PTX like the quick native build; sm120 is also measured. Construction excluded; pool retention restored to 512 MiB after upstream builds.",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, points, count, k in cases:
        # Generate before filtering to make filtered reruns reproducible.
        samples = np.stack([cloud[rng.choice(len(cloud), count)] for cloud in points])
        samples += rng.normal(0, 0.01, samples.shape).astype(np.float32)
        if args.filter not in name:
            continue
        row = {
            "name": name,
            "batches": len(points),
            "targets": points.shape[1],
            "queries": count,
            "k": k,
            "objectives": {},
        }
        for objective in wp.PointQueryObjective:
            gc.collect()
            result = run_case(points, samples, k, objective, libraries, args.repeats, device)
            row["objectives"][objective.name] = result
            print(
                name,
                objective.name,
                "graph ms",
                {key: round(value["median_ms"], 5) for key, value in result["graph"].items()},
                flush=True,
            )
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
