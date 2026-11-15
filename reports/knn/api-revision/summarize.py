# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize the matched API/main revision measurements without pooling experiments."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
PHASES = {
    "build_and_query": "Build + query",
    "reused_query": "Reused query",
    "graph_query": "Query graph",
    "symmetric_chamfer_forward_backward": "Chamfer, same backend",
    "mixed_chamfer_forward_backward": "Chamfer, mixed backends",
}
VARIANTS = {"warp": "WARP", "cubql_reuse": "CUBQL / REUSE", "cubql_build": "CUBQL / BUILD_AND_QUERY"}


def main():
    reports = {name: json.loads((ROOT / f"{name}.json").read_text()) for name in ("before", "after", "engine")}
    before, after, engine = (reports[name] for name in ("before", "after", "engine"))
    if len({r["archive_sha256"] for r in reports.values()}) != 1:
        raise ValueError("Dataset hashes differ")
    rows = []
    memory_equal = 0
    for old_case, new_case, control_case in zip(before["cases"], after["cases"], engine["cases"], strict=True):
        for key in ("name", "batches", "targets", "queries", "k"):
            if len({case[key] for case in (old_case, new_case, control_case)}) != 1:
                raise ValueError(f"Mismatched case {key}")
        for variant in VARIANTS:
            old, new, control = (case["variants"][variant] for case in (old_case, new_case, control_case))
            memory_equal += old["retained_index_and_output_pool_bytes"] == new["retained_index_and_output_pool_bytes"]
            for phase in PHASES:
                if phase not in old:
                    continue
                values = [result[phase]["median_ms"] for result in (old, new, control)]
                rows.append(
                    {
                        "case": old_case["name"],
                        "variant": variant,
                        "phase": phase,
                        "before_ms": values[0],
                        "after_ms": values[1],
                        "engine_ms": values[2],
                        "after_over_before": values[1] / values[0],
                        "engine_over_before": values[2] / values[0],
                        "after_over_engine": values[1] / values[2],
                        "delta_us": 1000 * (values[1] - values[0]),
                    }
                )
    with (ROOT / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"timing_records": len(rows), "identical_retained_memory_pairs": memory_equal, "phases": {}}
    table = [
        "| Phase | Comparisons | New API / before | Engine / before | New API / engine | Median added time |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, label in PHASES.items():
        selected = [row for row in rows if row["phase"] == phase]
        ratios = {
            key: statistics.geometric_mean(row[key] for row in selected)
            for key in ("after_over_before", "engine_over_before", "after_over_engine")
        }
        delta = statistics.median(row["delta_us"] for row in selected)
        summary["phases"][phase] = {"count": len(selected), **ratios, "median_delta_us": delta}
        table.append(
            f"| {label} | {len(selected)} | "
            + " | ".join(f"{100 * (value - 1):+.2f}%" for value in ratios.values())
            + f" | {delta:+.2f} µs |"
        )
    summary["largest_ratio"] = max(rows, key=lambda row: row["after_over_before"])
    summary["smallest_ratio"] = min(rows, key=lambda row: row["after_over_before"])
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "tables.md").write_text(
        "# Matched before/after measurements\n\n"
        "Positive percentages mean longer runtime. Ratios are geometric means of per-case medians. "
        "Times are synchronized wall measurements, including Python and allocation work.\n\n"
        + "\n".join(table)
        + "\n\n[Every median and ratio](cases.csv). These are separate sequential runs; small differences "
        "are not confidence bounds or universal guarantees.\n"
    )

    columns = [(variant, phase) for variant in VARIANTS for phase in PHASES if not phase.startswith("mixed")]
    columns.append(("cubql_reuse", "mixed_chamfer_forward_backward"))
    names = [case["name"] for case in before["cases"]]
    lookup = {(row["case"], row["variant"], row["phase"]): row for row in rows}
    values = np.full((len(names), len(columns)), np.nan)
    for i, name in enumerate(names):
        for j, (variant, phase) in enumerate(columns):
            row = lookup.get((name, variant, phase))
            if row:
                values[i, j] = 100 * (row["after_over_before"] - 1)
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "knn-api-revision", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(17, 12))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        plotted = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6), aspect="auto")
        labels = {
            "build_and_query": "build + query",
            "reused_query": "query",
            "graph_query": "graph",
            "symmetric_chamfer_forward_backward": "Chamfer",
            "mixed_chamfer_forward_backward": "mixed Chamfer",
        }
        ax.set_xticks(range(len(columns)), [f"{VARIANTS[v]}\n{labels[p]}" for v, p in columns], rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(columns)):
                value = values[i, j]
                ax.text(
                    j,
                    i,
                    "n/a" if np.isnan(value) else f"{value:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 4.5 else "black",
                )
        ax.set_title(
            "Geometry API + NVIDIA main vs published branch (November 15)\n"
            "Blue = faster; red = slower. Each cell: change in median runtime, 15 samples.\n"
            "94-SM Blackwell MIG · 24 workloads · 300 comparisons",
            pad=18,
        )
        fig.colorbar(plotted, ax=ax, fraction=0.02, pad=0.02, label="Runtime change (%)")
        fig.tight_layout()
        fig.savefig(ROOT / "before-after.svg", metadata={"Date": None})
        plt.close(fig)
    svg = ROOT / "before-after.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
