# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the current kNN tables and chart from saved benchmark samples."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "warp": "`backend=wp.PointQueryBackend.WARP`",
    "cubql_reuse": "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.REUSE`",
    "cubql_build": "`backend=wp.PointQueryBackend.CUBQL`, `objective=wp.PointQueryObjective.BUILD_AND_QUERY`",
}
MODES = {
    "build_and_query": "Construction plus query",
    "reused_query": "Reused index, ordinary query including output allocation",
    "graph_query": "Reused index, single query graph replay",
}


def geometric_mean(values):
    """Compute the geometric mean of positive timing ratios."""
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    """Write aggregates, per-case timings, and a fixed-target comparison chart."""
    data = json.loads((ROOT / "batched.json").read_text())
    cases = data["cases"]
    summary = {"case_count": len(cases), "modes": {}}
    lines = [
        "# Current kNN timings",
        "",
        "Generated from [batched.json](batched.json). See [current comparisons](../comparisons.md) "
        "for methodology and recommendations. All times are median milliseconds; lower is better.",
        "",
        "The column labels below map to public arguments; all configurations use `wp.knn()`:",
        "",
        "| Column | Arguments |",
        "| --- | --- |",
    ]
    for variant, label in VARIANTS.items():
        lines.append(f"| `{variant}` | {label} |")
    for mode, title in MODES.items():
        summary["modes"][mode] = {}
        for variant in VARIANTS:
            ratios = [
                case["variants"]["warp"][mode]["median_ms"] / case["variants"][variant][mode]["median_ms"]
                for case in cases
            ]
            summary["modes"][mode][variant] = {
                "geomean_speedup_over_warp": geometric_mean(ratios),
                "wins_over_warp": sum(ratio > 1 for ratio in ratios),
                "lowest_median_cases": sum(
                    min(case["variants"], key=lambda name: case["variants"][name][mode]["median_ms"]) == variant
                    for case in cases
                ),
            }
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Workload | B | M | N | k | `warp` | `cubql_reuse` | `cubql_build` |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in cases:
            values = [case["variants"][variant][mode]["median_ms"] for variant in VARIANTS]
            times = [f"**{value:.5f}**" if value == min(values) else f"{value:.5f}" for value in values]
            lines.append(
                f"| `{case['name']}` | {case['batches']} | {case['targets']:,} | {case['queries']:,} | "
                f"{case['k']} | {' | '.join(times)} |"
            )
    chamfer_cases = [case for case in cases if case["k"] == 1]
    ratios = [
        case["variants"]["warp"]["symmetric_chamfer_forward_backward"]["median_ms"]
        / case["variants"]["cubql_reuse"]["mixed_chamfer_forward_backward"]["median_ms"]
        for case in chamfer_cases
    ]
    summary["mixed_chamfer"] = {
        "case_count": len(ratios),
        "wins_over_warp_both_directions": sum(ratio > 1 for ratio in ratios),
        "geomean_speedup_over_warp_both_directions": geometric_mean(ratios),
    }
    lines.extend(
        [
            "",
            "## Symmetric Chamfer forward and backward",
            "",
            "Fixed targets are prebuilt. The reverse direction constructs the changing decoded index each step. "
            "The mixed column uses `cubql_reuse` for fixed targets and `warp` in reverse; the final column "
            "uses `cubql_reuse` for fixed targets and `cubql_build` in reverse.",
            "",
            "| Workload | `warp` both directions | `cubql_reuse` fixed, `warp` reverse | "
            "`cubql_reuse` fixed, `cubql_build` reverse |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for case in chamfer_cases:
        variants = case["variants"]
        lines.append(
            f"| `{case['name']}` | {variants['warp']['symmetric_chamfer_forward_backward']['median_ms']:.5f} | "
            f"{variants['cubql_reuse']['mixed_chamfer_forward_backward']['median_ms']:.5f} | "
            f"{variants['cubql_reuse']['symmetric_chamfer_forward_backward']['median_ms']:.5f} |"
        )
    checks = [
        check
        for case in cases
        for variant in case["variants"].values()
        for mode in ("validation", "changed_query_graph_validation")
        for check in variant[mode]
    ]
    summary["validation"] = {
        "workload_variant_combinations": sum(len(case["variants"]) for case in cases),
        "sampled_rows": sum(check["checked_rows"] for check in checks),
        "min_tie_aware_recall_at_k": min(check["tie_aware_recall_at_k"] for check in checks),
        "all_shapes_and_indices_valid": all(
            check["valid_shape"] and check["valid_indices_and_finite_distances"] for check in checks
        ),
        **{
            key: sum(check[key] for check in checks)
            for key in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "timings.md").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "warp-knn-current"})
    figure, axes = plt.subplots(figsize=(11, 10))
    for mode, offset, color, label in (
        ("reused_query", -0.15, "#276FBF", "Ordinary query, including output allocation"),
        ("graph_query", 0.15, "#B54A16", "Single query graph replay"),
    ):
        ratios = [
            case["variants"]["warp"][mode]["median_ms"] / case["variants"]["cubql_reuse"][mode]["median_ms"]
            for case in cases
        ]
        axes.scatter(ratios, [index + offset for index in range(len(cases))], color=color, label=label, s=28)
    axes.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes.set_yticks(range(len(cases)), [case["name"] for case in cases], fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("Speedup: WARP median / CUBQL + REUSE median (higher is faster)")
    axes.set_title("Fixed-target queries through wp.knn()\n24 workloads · 94-SM Blackwell MIG · 2026-11-13")
    axes.grid(axis="x", alpha=0.2)
    axes.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "query-speedups.svg", metadata={"Date": None})
    plt.close(figure)
    chart = ROOT / "query-speedups.svg"
    chart.write_text("\n".join(line.rstrip() for line in chart.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
