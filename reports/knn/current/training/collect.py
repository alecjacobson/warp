# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit completed workers and collect reproducible comparison evidence without GPU work."""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import shutil
import statistics
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    """Hash an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(condition, message):
    """Reject incomplete or inconsistent evidence."""
    if not condition:
        raise ValueError(message)


def main():
    """Verify samples, sources and data before copying the saved measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".benchmarks/knn"))
    args = parser.parse_args()
    workspace = args.workspace
    experiment = workspace / "training"
    manifest_path = experiment / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checked(len(manifest["cases"]) == 24, "Expected 24 evaluation workloads")
    pulmonary = json.loads(Path("reports/knn/current/batched.json").read_text())
    checked(manifest["archive_sha256"] == pulmonary["archive_sha256"], "Pulmonary archive changed")
    for case in [*manifest["cases"], manifest["fixture"]]:
        path = experiment / "data" / case["file"]
        checked(digest(path) == case["file_sha256"], f"Input file changed: {path}")
        with np.load(path) as arrays:
            for key in ("points", "queries"):
                checked(
                    hashlib.sha256(arrays[key].tobytes()).hexdigest() == case[key + "_sha256"],
                    f"Input coordinates changed: {case['name']}, {key}",
                )
    sources = experiment / "source-snapshots"
    sources.mkdir(exist_ok=True)
    harness = Path("warp/examples/benchmarks/benchmark_point_query_training.py")
    (sources / (digest(harness) + ".py.gz")).write_bytes(gzip.compress(harness.read_bytes(), mtime=0))
    source_asts = {}
    compared_nodes = [
        "measure",
        "Problem",
        "run_case",
        "chamfer_reference",
        "query_checks",
        "checks_pass",
        "prepare",
        "regime",
        "tune",
    ]
    for path in sources.glob("*.py.gz"):
        body = gzip.decompress(path.read_bytes())
        version = hashlib.sha256(body).hexdigest()
        checked(path.name == version + ".py.gz", "Source snapshot hash mismatch")
        source_asts[version] = {
            node.name: ast.dump(node)
            for node in ast.parse(body).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in compared_nodes
        }
    checked(all(value == next(iter(source_asts.values())) for value in source_asts.values()), "Timed code changed")
    timing_count, case_count, versions = 0, 0, set()
    result_files = sorted((experiment / "run").glob("*-*.json"))
    checked(len(result_files) == 17, "Expected eight query and nine Chamfer reports")
    environment = json.loads(result_files[0].read_text())["environment"]
    for path in result_files:
        report = json.loads(path.read_text())
        checked(report["environment"] == environment, f"Worker environment changed: {path}")
        expected = {c["name"] for c in manifest["cases"] if report["phase"] == "query" or c["k"] == 1}
        checked({c["name"] for c in report["cases"]} == expected, f"Missing cases: {path}")
        checked(len(report["cases"]) == len(expected), f"Duplicate cases: {path}")
        checked(report["manifest_sha256"] == digest(manifest_path), f"Wrong manifest: {path}")
        checked(report["tuning_sha256"] == digest(experiment / "tuning.json"), f"Wrong tuning: {path}")
        checked(report["sm_count"] == 94 and report["repeats"] == 9, f"Wrong experiment: {path}")
        for row in report["cases"]:
            versions.add(row.get("source_sha256", report["source_sha256"]))
            if row["status"] == "ok":
                samples = row["timing"]["samples_ms"]
                checked(len(samples) == 9 and all(np.isfinite(s) and s > 0 for s in samples), f"Bad samples: {path}")
                checked(statistics.median(samples) == row["timing"]["median_ms"], f"Wrong median: {path}")
                timing_count += 1
            else:
                checked(not row["passed"] and "timing" not in row and bool(row["error"]), f"Unclear failure: {path}")
            case_count += 1
    checked(versions <= source_asts.keys(), "Missing source snapshot")
    checked(case_count == 381, "Wrong total cell count")
    previous = json.loads(Path("reports/knn/current/provenance.json").read_text())
    retained = {}
    for name, artifact in previous["artifacts"].items():
        checked(digest(Path("reports/knn") / name) == artifact["sha256"], f"Retained artifact changed: {name}")
        for source in artifact["source_checks"]:
            checked(digest(Path(source["path"])) == source["sha256"], f"Runtime source changed: {source['path']}")
            retained[source["path"]] = source["sha256"]
    archive = json.loads(Path("reports/knn/archive/manifest.json").read_text())
    historical_count = 0
    for item in archive["files"].values():
        path = Path(item["path"])
        if path.suffix in (".md", ".rst", ".html"):
            continue
        checked(digest(path) == item["sha256"], f"Historical evidence changed: {path}")
        historical_count += 1
    dependency = workspace / "third_party/pytorch3d"
    revision = subprocess.check_output(["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(dependency), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    checked(not dirty, "PyTorch3D tracked source changed")
    old = json.loads(Path("reports/knn/current/libraries/provenance.json").read_text())
    checked(revision == old["dependencies"]["pytorch3d"]["revision"], "PyTorch3D pin changed")
    binary = dependency / "pytorch3d/_C.cpython-312-x86_64-linux-gnu.so"
    checked(digest(binary) == old["binary_sha256"][str(binary)], "PyTorch3D binary changed")
    package_checks = {}
    for name in ("torch", "pykeops", "keopscore"):
        package = distribution(name)
        count = 0
        for entry in package.files:
            if entry.suffix != ".py" or not entry.hash:
                continue
            if name == "torch" and str(entry) not in ("torch/functional.py", "torch/autograd/__init__.py"):
                continue
            actual = base64.urlsafe_b64encode(hashlib.sha256(Path(package.locate_file(entry)).read_bytes()).digest())
            checked(
                entry.hash.mode == "sha256" and actual.decode().rstrip("=") == entry.hash.value,
                f"Installed package source changed: {entry}",
            )
            count += 1
        package_checks[name] = {"version": package.version, "python_files_matching_wheel_record": count}
    for source in (manifest_path, experiment / "tuning.json"):
        shutil.copyfile(source, ROOT / source.name)
    for source in (experiment / "run").rglob("*"):
        if not source.is_file():
            continue
        target = ROOT / "results" / source.relative_to(experiment / "run")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".log":
            target.with_suffix(".log.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        else:
            shutil.copyfile(source, target)
    for source in sources.glob("*.py.gz"):
        (ROOT / "source-snapshots").mkdir(exist_ok=True)
        shutil.copyfile(source, ROOT / "source-snapshots" / source.name)
    (ROOT / "logs").mkdir(exist_ok=True)
    for source in (experiment / "logs").glob("*.log"):
        (ROOT / "logs" / (source.name + ".gz")).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    smoke = [json.loads(p.read_text()) for p in sorted((experiment / "smoke").glob("*-*.json"))]
    checked(len(smoke) == 17 and all(r["cases"][0]["passed"] for r in smoke), "Fixture validation incomplete")
    (ROOT / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    receipts = json.loads((experiment / "run/processes.json").read_text())
    provenance = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "runtime_base_commit": json.loads(result_files[0].read_text())["environment"]["git_revision"],
        "hardware": old["hardware"],
        "toolkit": old["toolkit"],
        "source_sha256": retained
        | {
            str(path): digest(path)
            for path in [
                harness,
                Path("warp/examples/benchmarks/benchmark_point_query_comparison.py"),
                Path("warp/examples/benchmarks/point_query_comparison_environment.py"),
                Path("warp/examples/benchmarks/test_point_query_training.py"),
                *(Path("reports/knn/current/training") / name for name in ("collect.py", "run.py", "summarize.py")),
            ]
        },
        "harness_versions": sorted(versions),
        "unchanged_ast_nodes_across_snapshots": compared_nodes,
        "harness_change": "Add single-case filtering and runtime-error records; provider, timing, oracle and tuning code unchanged.",
        "manifest_sha256": digest(manifest_path),
        "tuning_sha256": digest(experiment / "tuning.json"),
        "dependencies": {
            "torch": "2.7.1+cu128",
            "pykeops": "2.3",
            "pytorch3d": {"revision": revision, "tracked_source_clean": True, "binary_sha256": digest(binary)},
        },
        "package_source_checks": package_checks,
        "measurements": {
            "case_cells": case_count,
            "nine_sample_medians_verified": timing_count,
            "worker_attempts": len(receipts),
            "first_started_at": min(r["started_at"] for r in receipts),
            "last_finished_at": max(r["finished_at"] for r in receipts),
            "small_fixture_provider_phase_checks": len(smoke),
        },
        "historical_raw_artifacts_verified_unchanged": historical_count,
        "scope": "Native batching and symmetric fixed-target Chamfer gradients; ordinary calls only. Failed checks/errors are retained and excluded from speed ratios. No external source changes or replacement kernels.",
        "artifact_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and (
                p.name in ("manifest.json", "tuning.json", "smoke.json")
                or p.relative_to(ROOT).parts[0] in ("results", "logs", "source-snapshots")
            )
        },
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["measurements"], indent=2))


if __name__ == "__main__":
    main()
