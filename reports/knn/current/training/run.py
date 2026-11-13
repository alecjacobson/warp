# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run unchanged providers sequentially for the batched query/Chamfer report."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

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


def main():
    """Record isolated worker outcomes with explicit timeouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".benchmarks/knn"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--complete", action="store_true", help="Measure missing cases in fresh processes and preserve initial attempts"
    )
    args = parser.parse_args()
    if args.complete and args.smoke:
        parser.error("--complete resumes evaluation cases; run --smoke separately")
    valid_workers = {
        f"{method}-{phase}"
        for method in METHODS
        for phase in ("query", "chamfer")
        if method != "torch_selected" or phase != "query"
    }
    if args.only and not set(args.only) <= valid_workers:
        parser.error(f"Unknown worker names: {sorted(set(args.only) - valid_workers)}")
    workspace = args.workspace.resolve()
    output = workspace / "training" / ("smoke" if args.smoke else "run")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "processes.json"
    receipts = json.loads(receipt_path.read_text()) if args.complete and receipt_path.exists() else []
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "4"
    for method in METHODS:
        for phase in ("query", "chamfer"):
            if method == "torch_selected" and phase == "query":
                continue
            name = method + "-" + phase
            if args.only and name not in args.only:
                continue
            command = [
                sys.executable,
                "-m",
                "warp.examples.benchmarks.point_query_comparison_environment",
                "--python",
                sys.executable,
                "--cuda",
                "/home/horde/cuda/12.9.1",
                "--module",
                "warp.examples.benchmarks.benchmark_point_query_training",
                "--",
                "worker",
                "--data",
                str(workspace / "training/data"),
                "--tuning",
                str(workspace / "training/tuning.json"),
                "--pytorch3d-path",
                str(workspace / "third_party/pytorch3d"),
                "--method",
                method,
                "--phase",
                phase,
                "--output",
                str(output / (name + ".json")),
                "--repeats",
                "2" if args.smoke else "9",
            ]
            if args.smoke:
                command.append("--smoke")
            if args.complete:
                result_path = output / (name + ".json")
                previous = json.loads(result_path.read_text()) if result_path.exists() else None
                existing = {case["name"]: case for case in previous["cases"]} if previous else {}
                manifest = json.loads((workspace / "training/data/manifest.json").read_text())
                expected = [case for case in manifest["cases"] if phase == "query" or case["k"] == 1]
                missing = [case for case in expected if case["name"] not in existing]
                if not missing:
                    continue
                attempts = output / "attempts" / name
                attempts.mkdir(parents=True, exist_ok=True)
                if previous:
                    if not (attempts / "initial.json").exists():
                        shutil.copyfile(result_path, attempts / "initial.json")
                        shutil.copyfile(output / (name + ".log"), attempts / "initial.log")
                    for case in existing.values():
                        case.setdefault("source_sha256", previous["source_sha256"])
                for case in missing:
                    target = attempts / (case["name"] + ".json")
                    case_command = list(command)
                    case_command[case_command.index("--output") + 1] = str(target)
                    case_command += ["--case", case["name"]]
                    receipt = {
                        "name": name,
                        "case": case["name"],
                        "command": case_command,
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                    print("Completing", name, case["name"], flush=True)
                    with target.with_suffix(".log").open("w") as log:
                        process = subprocess.run(
                            case_command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3600, check=False
                        )
                    receipt.update(returncode=process.returncode, finished_at=datetime.now(UTC).isoformat())
                    receipts.append(receipt)
                    receipt_path.write_text(json.dumps(receipts, indent=2) + "\n")
                    if not target.exists():
                        raise RuntimeError(f"Worker left no diagnostic: {name}, {case['name']}")
                    supplemental = json.loads(target.read_text())
                    row = supplemental["cases"][0]
                    row["source_sha256"] = supplemental["source_sha256"]
                    existing[case["name"]] = row
                    if previous is None:
                        previous = supplemental
                    previous["cases"] = [existing[c["name"]] for c in expected if c["name"] in existing]
                    previous["source_versions"] = sorted({r["source_sha256"] for r in existing.values()})
                    result_path.write_text(json.dumps(previous, indent=2) + "\n")
                continue
            receipt = {"name": name, "command": command, "started_at": datetime.now(UTC).isoformat()}
            print("Running", name, flush=True)
            with (output / (name + ".log")).open("w") as log:
                try:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3600, check=False
                    )
                    receipt["returncode"] = process.returncode
                except subprocess.TimeoutExpired:
                    receipt["status"] = "timeout"
            receipt["finished_at"] = datetime.now(UTC).isoformat()
            receipts.append(receipt)
            (output / "processes.json").write_text(json.dumps(receipts, indent=2) + "\n")
            print("Finished", name, receipt.get("returncode", receipt.get("status")), flush=True)


if __name__ == "__main__":
    main()
