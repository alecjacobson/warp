# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate colored cross-library tables from the current saved measurements."""

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
RESULTS = ROOT / "results"
METHODS = {
    "warp_scalar": ("wp.point_query_knn()\nscalar, dense launch", "`wp.point_query_knn()`; default `wp.Bvh()`"),
    "warp_batch": ("wp.knn()\nbackend=WARP", "`backend=wp.PointQueryBackend.WARP`"),
    "warp_bvh_cubql": (
        'wp.knn(backend=WARP)\nBvh(constructor="cubql")',
        '`backend=wp.PointQueryBackend.WARP`, `bvh=wp.Bvh(..., constructor="cubql")`',
    ),
    "warp_cubql_reuse": (
        "wp.knn(backend=CUBQL)\nobjective=REUSE",
        "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`",
    ),
    "warp_cubql_build": (
        "wp.knn(backend=CUBQL)\nobjective=BUILD_AND_QUERY",
        "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY`",
    ),
    "physicsnemo": ("PhysicsNeMo\npinned PR", "PhysicsNeMo, pinned PR implementation"),
    "cubql": ("Standalone cuBQL\ncustom top-k", "Standalone cuBQL, custom top-k adapter"),
    "cubql_defaults": ("Standalone cuBQL\nfindKNN defaults", "Standalone cuBQL `findKNN`, default settings"),
    "cubql_tuned": ("Standalone cuBQL\nfindKNN frozen tuning", "Standalone cuBQL `findKNN`, frozen tuning"),
    "cupy_kdtree": ("CuPy KDTree", "CuPy `KDTree`"),
    "keops": ("KeOps", "KeOps `Kmin_argKmin`"),
    "faiss_flat": ("Faiss Flat", "Faiss `GpuIndexFlatL2`"),
    "cuvs_brute_force": ("cuVS expanded L2", "cuVS brute force, expanded squared L2"),
    "cuvs_unexpanded": ("cuVS direct L2", "cuVS brute force, direct squared L2"),
    "cuvs_minkowski": ("cuVS Minkowski p=2", "cuVS brute force, Minkowski p=2"),
    "pytorch3d": ("PyTorch3D", "PyTorch3D `knn_points`"),
}
PHASES = {
    "one_shot": ("Build + query, ordinary calls", "one-shot.svg"),
    "reused_index_search": ("Reused-index query, ordinary calls", "reused-query.svg"),
    "graph_one_shot": ("Build + query, CUDA graph replay", "graph-one-shot.svg"),
    "graph_reused_index_search": ("Reused-index query, CUDA graph replay", "graph-reused-query.svg"),
}


def passed(check):
    """Require structural, rank, and distance checks to pass."""
    return bool(
        check.get("valid_shape")
        and check.get("valid_indices_and_finite_distances")
        and all(
            check.get(key, 1) == 0
            for key in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        )
    )


def load_reports():
    """Load the complete ordinary matrix and separately recorded graph runs."""
    public = json.loads((RESULTS / "warp-public.json").read_text())
    reports = {}
    for method in METHODS:
        if method.startswith("warp_"):
            reports[method] = {
                "cases": [dict(row["case"], **row["variants"][method]) for row in public["cases"]],
                "manifest_sha256": public["manifest_sha256"],
                "device": public["device"],
                "sm_count": public["sm_count"],
            }
        else:
            reports[method] = json.loads((RESULTS / f"{method}.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest_sha256 = hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
    expected = {case["id"] for case in manifest["cases"]}
    for method, report in reports.items():
        if report["manifest_sha256"] != manifest_sha256 or report["device"] != public["device"]:
            raise ValueError(f"Mismatched inputs or hardware: {method}")
        if {case["id"] for case in report["cases"]} != expected:
            raise ValueError(f"Incomplete ordinary measurements for {method}")
        for case in report["cases"]:
            if not case.get("one_shot"):
                raise ValueError(f"Missing ordinary timing: {method}, {case['id']}")
        report["by_id"] = {case["id"]: case for case in report["cases"]}
        for phase in ("query", "full"):
            path = RESULTS / f"{method}-graph-{phase}.json"
            if path.exists():
                graph = json.loads(path.read_text())
                if graph["manifest_sha256"] != manifest_sha256 or graph["device"] != public["device"]:
                    raise ValueError(f"Mismatched graph inputs or hardware: {method}, {phase}")
                graph["by_id"] = {case["id"]: case for case in graph["cases"]}
                report[f"graph_{phase}"] = graph
    return reports, manifest


def measurement(report, case_id, phase):
    """Return timing and a visible status without substituting ordinary graph times."""
    case = report["by_id"][case_id]
    if phase.startswith("graph_"):
        timing = case.get(phase)
        if timing:
            valid = passed(timing["validation"]) and passed(timing["changed_input_validation"])
            return timing, "pass" if valid else "checks failed"
        if case.get("graph_one_shot_status") == "construction_not_capturable" and phase == "graph_one_shot":
            return None, "unsupported"
        graph = report.get("graph_full" if phase == "graph_one_shot" else "graph_query")
        if graph is None:
            return None, "not measured"
        graph_case = graph["by_id"].get(case_id)
        if graph_case is None:
            return None, "not run"
        if graph_case.get("status") != "ok":
            stage = graph_case.get("failed_stage", graph_case.get("status"))
            return None, "capture failed" if stage == "capturing" else "setup failed"
        valid = passed(graph_case["validation"]) and passed(graph_case["changed_input_validation"])
        return graph_case["timing"], "pass" if valid else "checks failed"
    # These three adapters expose a complete operator, without a reusable index.
    selected = phase if case.get(phase) else "one_shot"
    validation = "reused_validation" if selected == "reused_index_search" else "validation"
    return case[selected], "pass" if passed(case[validation]) else "checks failed"


def main():
    """Write four colored matrices, full CSV timings, and machine-readable aggregates."""
    reports, manifest = load_reports()
    cases = manifest["cases"]
    groups = {}
    for case in cases:
        groups.setdefault((case["family"], case["N"], case["Q"], case["k"]), []).append(case["id"])
    receipts = json.loads((RESULTS / "processes.json").read_text())
    dates = sorted({record[key][:10] for record in receipts for key in ("started_at", "finished_at")})
    date_label = " to ".join(dict.fromkeys((dates[0], dates[-1])))
    sm_count = reports["warp_batch"]["sm_count"]
    summary = {
        "case_count": len(cases),
        "method_count": len(METHODS),
        "device": reports["warp_batch"]["device"],
        "sm_count": sm_count,
        "measurement_dates": dates,
        "phases": {},
    }
    csv_rows = []
    lines = [
        "# Current cross-library timings",
        "",
        "See [comparisons](../../comparisons.md) for the colored tables and methodology. "
        "[cases.csv](cases.csv) contains every median, ratio, and check status. "
        "All ratios below are method time divided by the public `backend=wp.PointQueryBackend.WARP` time; "
        "values above 1 favor that reference. Geometric means include only passing paired cases.",
    ]
    for phase, (title, filename) in PHASES.items():
        summary["phases"][phase] = {}
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method / public settings | Passing pairs | Method / reference geometric mean | Method wins |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method, report in reports.items():
            ratios = []
            statuses = {}
            for case in cases:
                timer, status = measurement(report, case["id"], phase)
                baseline, baseline_status = measurement(reports["warp_batch"], case["id"], phase)
                statuses[status] = statuses.get(status, 0) + 1
                ratio = None
                if status == baseline_status == "pass":
                    ratio = timer["wall_median_ms"] / baseline["wall_median_ms"]
                    ratios.append(ratio)
                csv_rows.append(
                    {
                        "case": case["id"],
                        "method": method,
                        "phase": phase,
                        "status": status,
                        "wall_median_ms": timer["wall_median_ms"] if timer else None,
                        "method_over_reference": ratio,
                    }
                )
            aggregate = {
                "passing_pairs": len(ratios),
                "method_over_reference_geomean": statistics.geometric_mean(ratios) if ratios else None,
                "method_wins": sum(ratio < 1 for ratio in ratios),
                "statuses": statuses,
            }
            summary["phases"][phase][method] = aggregate
            value = f"{aggregate['method_over_reference_geomean']:.3f}x" if ratios else "—"
            lines.append(
                f"| {METHODS[method][1]} | {len(ratios)}/{len(cases)} | {value} | {aggregate['method_wins']} |"
            )
        values = np.full((len(groups), len(METHODS)), np.nan)
        labels = {}
        for column, method in enumerate(METHODS):
            for row, case_ids in enumerate(groups.values()):
                ratios, statuses = [], []
                for case_id in case_ids:
                    timer, status = measurement(reports[method], case_id, phase)
                    baseline, baseline_status = measurement(reports["warp_batch"], case_id, phase)
                    statuses.append(status)
                    if status == baseline_status == "pass":
                        ratios.append(timer["wall_median_ms"] / baseline["wall_median_ms"])
                if len(ratios) == len(case_ids):
                    ratio = statistics.geometric_mean(ratios)
                    values[row, column] = np.log2(ratio)
                    labels[row, column] = f"{ratio:.2f}x"
                elif "checks failed" in statuses:
                    labels[row, column] = f"checks\n{len(ratios)}/{len(case_ids)} pass"
                else:
                    priority = ("capture failed", "setup failed", "unsupported", "not run", "not measured")
                    labels[row, column] = next(status for status in priority if status in statuses).replace(" ", "\n")
        with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "knn-current-libraries"}):
            figure, axes = plt.subplots(figsize=(24, 12))
            colors = plt.get_cmap("RdBu").copy()
            colors.set_bad("#e8e8e8")
            chart = axes.imshow(values, cmap=colors, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6), aspect="auto")
            axes.set_xticks(range(len(METHODS)), [label[0] for label in METHODS.values()], rotation=35, ha="right")
            axes.set_yticks(
                range(len(groups)),
                [
                    f"{family.replace('_', ' ')} · M={size:,} · N={queries:,} · k={k}"
                    for family, size, queries, k in groups
                ],
            )
            for (row, column), label in labels.items():
                value = values[row, column]
                color = "white" if np.isfinite(value) and (value < -1.5 or value > 4.5) else "black"
                axes.text(column, row, label, ha="center", va="center", color=color, fontsize=8)
            axes.set_title(
                title + f" · {sm_count}-SM Blackwell MIG · {date_label}\n"
                "Method / wp.knn(backend=wp.PointQueryBackend.WARP); geometric mean across clean/noisy inputs",
                pad=16,
            )
            bar = figure.colorbar(chart, ax=axes, fraction=0.02, pad=0.02, ticks=range(-3, 7))
            bar.ax.set_yticklabels(["≤1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x", "16x", "32x", "≥64x"])
            figure.text(
                0.5,
                0.015,
                "Blue >1: reference faster. Red <1: column faster. Gray: incomplete/failed pair, no paired ratio. "
                "Both cases must pass structural, rank, and distance checks for color.\n"
                "Nine synchronized samples. PhysicsNeMo, KeOps, and PyTorch3D expose complete operators. "
                "Graph columns use replay measurements only; a failed capture stops that isolated worker.",
                ha="center",
                fontsize=10,
            )
            figure.tight_layout(rect=(0, 0.055, 1, 1))
            figure.savefig(ROOT / filename, metadata={"Date": None})
            plt.close(figure)
            path = ROOT / filename
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    with (ROOT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
