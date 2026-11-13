# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the matched batched-query and fixed-target Chamfer comparison."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
METHODS = {
    "warp": ("backend=WARP", "`backend=wp.PointQueryBackend.WARP`"),
    "cubql_reuse": (
        "backend=CUBQL\nobjective=REUSE",
        "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`",
    ),
    "cubql_build": (
        "backend=CUBQL\nobjective=BUILD_AND_QUERY",
        "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY`",
    ),
    "torch_mm": (
        "torch.cdist\ncompute_mode=use_mm_for_euclid_dist",
        '`torch.cdist(compute_mode="use_mm_for_euclid_dist")`',
    ),
    "torch_direct": (
        "torch.cdist\ncompute_mode=donot_use_mm_for_euclid_dist",
        '`torch.cdist(compute_mode="donot_use_mm_for_euclid_dist")`',
    ),
    "torch_selected": (
        "torch.cdist direct\nselected-pair autograd",
        "`torch.cdist` direct mode under `torch.no_grad()`; gather selected pairs, reduce squared distances, backward",
    ),
    "pytorch3d": ("PyTorch3D\nautomatic version", "Public PyTorch3D automatic kernel selection"),
    "pytorch3d_tuned": (
        "PyTorch3D\nindependently tuned version",
        "Public `knn_points(version=...)`, selected on independent synthetic inputs",
    ),
    "keops": (
        "KeOps\nnative batch dimensions",
        "Public batched `LazyTensor` reductions; selected-pair autograd for Chamfer",
    ),
}


def main():
    """Save colored ratios, every timing/status, and passing-pair aggregates."""
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    all_rows, summary = [], {}
    lines = [
        "# Batched query and Chamfer timings",
        "",
        "All ratios are method / the matching public `backend=wp.PointQueryBackend.WARP` reference. "
        "Values above 1 favor the reference. Only passing paired cases enter geometric means. "
        "Passing counts differ; geometric means over different subsets are not head-to-head rankings. "
        "See [the report](../../comparisons.md#batched-queries-and-chamfer-against-external-libraries), "
        "[every median/status](cases.csv), and [reproduction](reproduce.md).",
    ]
    for phase in ("query", "chamfer"):
        cases = sorted(
            [c for c in manifest["cases"] if phase == "query" or c["k"] == 1], key=lambda c: c["batches"] == 1
        )
        methods = {name: labels for name, labels in METHODS.items() if phase == "chamfer" or name != "torch_selected"}
        if phase == "chamfer":
            methods["warp"] = (
                "Fixed: backend=WARP\nReverse: backend=WARP",
                "Fixed and reverse: `backend=wp.PointQueryBackend.WARP`",
            )
            methods["cubql_reuse"] = (
                "Fixed: backend=CUBQL, objective=REUSE\nReverse: backend=WARP",
                "Fixed: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; "
                "reverse: `backend=wp.PointQueryBackend.WARP`",
            )
            methods["cubql_build"] = (
                "Fixed: backend=CUBQL, objective=REUSE\nReverse: backend=CUBQL\nobjective=BUILD_AND_QUERY",
                "Fixed: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`; "
                "reverse: `backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY`",
            )
            methods["pytorch3d"] = (
                "pytorch3d.loss\nchamfer_distance()",
                '`pytorch3d.loss.chamfer_distance(norm=2, batch_reduction="mean", point_reduction="mean")`',
            )
        reports = {}
        for method in methods:
            report = json.loads((ROOT / "results" / f"{method}-{phase}.json").read_text())
            if report["manifest_sha256"] != manifest_hash or report["sm_count"] != 94 or report["repeats"] != 9:
                raise ValueError(f"Mismatched experiment: {method}, {phase}")
            reports[method] = {case["name"]: case for case in report["cases"]}
            if set(reports[method]) != {case["name"] for case in cases}:
                raise ValueError(f"Incomplete measurements: {method}, {phase}")
        values = np.full((len(cases), len(methods)), np.nan)
        labels = {}
        summary[phase] = {"case_count": len(cases), "method_count": len(methods), "methods": {}}
        lines.extend(
            [
                "",
                f"## {'Reused-target query' if phase == 'query' else 'Symmetric Chamfer forward + backward'}",
                "",
                "| Configuration | Passing pairs | Method/reference geometric mean | Method wins |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for column, method in enumerate(methods):
            ratios, statuses = [], {}
            for row, case in enumerate(cases):
                result = reports[method][case["name"]]
                baseline = reports["warp"][case["name"]]
                ratio = None
                status = (
                    result["status"] if result["status"] != "ok" else "pass" if result["passed"] else "checks_failed"
                )
                if status == "pass" and not baseline["passed"]:
                    status = "reference_checks_failed"
                if status == "pass":
                    ratio = result["timing"]["median_ms"] / baseline["timing"]["median_ms"]
                    ratios.append(ratio)
                    values[row, column] = np.log2(ratio)
                    labels[row, column] = f"{ratio:.2f}x\n{result['timing']['median_ms']:.4f} ms"
                else:
                    labels[row, column] = {
                        "out_of_memory": "OOM",
                        "allocator_nvml_error": "allocator\nerror",
                        "launch_error": "launch\nerror",
                    }.get(status, status.replace("_", "\n"))
                statuses[status] = statuses.get(status, 0) + 1
                all_rows.append(
                    {
                        "case": case["name"],
                        "batches": case["batches"],
                        "targets": case["targets"],
                        "queries": case["queries"],
                        "k": case["k"],
                        "phase": phase,
                        "method": method,
                        "status": status,
                        "median_ms": result.get("timing", {}).get("median_ms"),
                        "method_over_reference": ratio,
                    }
                )
            aggregate = {
                "passing_pairs": len(ratios),
                "method_over_reference_geomean": statistics.geometric_mean(ratios) if ratios else None,
                "method_wins": sum(value < 1 for value in ratios),
                "statuses": statuses,
            }
            summary[phase]["methods"][method] = aggregate
            value = f"{aggregate['method_over_reference_geomean']:.3f}x" if ratios else "—"
            lines.append(
                f"| {methods[method][1]} | {len(ratios)}/{len(cases)} | {value} | {aggregate['method_wins']} |"
            )
        with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "warp-knn-training"}):
            figure, axis = plt.subplots(figsize=(22, 0.4 * len(cases) + 3.8))
            figure.subplots_adjust(left=0.24, right=0.95, top=0.90, bottom=0.24)
            colors = plt.get_cmap("RdBu").copy()
            colors.set_bad("#e8e8e8")
            chart = axis.imshow(values, cmap=colors, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6), aspect="auto")
            axis.set_yticks(
                range(len(cases)),
                [f"{c['name']}\nB={c['batches']}  M={c['targets']:,}  N={c['queries']:,}  k={c['k']}" for c in cases],
                fontsize=8,
            )
            for row, label in enumerate(axis.get_yticklabels()):
                if cases[row]["batches"] > 1:
                    label.set_fontweight("bold")
            axis.axhline(sum(c["batches"] > 1 for c in cases) - 0.5, color="#555555", linewidth=1.5)
            axis.set_xticks(
                range(len(methods)), [label[0] for label in methods.values()], rotation=25, ha="right", fontsize=9
            )
            for (row, column), label in labels.items():
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if np.isfinite(value) and (value < -1.5 or value > 4.5) else "black",
                )
            title = (
                "Batched fixed-target queries"
                if phase == "query"
                else "Symmetric Chamfer: fixed targets, forward + backward"
            )
            figure.suptitle(
                title + f" · {len(cases)} workloads · 94-SM Blackwell MIG\n"
                "Fixed target indexes prebuilt; B=2/4 rows first. Method / backend=wp.PointQueryBackend.WARP.",
                fontsize=14,
            )
            bar = figure.colorbar(chart, ax=axis, fraction=0.022, pad=0.02, ticks=range(-3, 7))
            bar.ax.set_yticklabels(["≤1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x", "16x", "32x", "≥64x"])
            figure.text(
                0.5,
                0.025,
                "Blue >1: reference faster. Red <1: column faster. Gray: failed checks or runtime errors.\n"
                "Cells show the ratio and median milliseconds for the entire batch; nine synchronized wall-clock samples. "
                "Backend/objective names use wp.PointQueryBackend / wp.PointQueryObjective.\n"
                + (
                    "Sorted int64 IDs and float32 squared distances. Complete native batched calls; no Python loop over external batches."
                    if phase == "query"
                    else "Sum of mean squared distances; decoded gradients included. Reverse indexes rebuilt each step. "
                    "PyTorch3D/KeOps remain unmodified; tuned versions use independent inputs."
                ),
                ha="center",
                fontsize=10,
            )
            path = ROOT / f"{phase}.svg"
            figure.savefig(path, metadata={"Date": None})
            plt.close(figure)
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    with (ROOT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
