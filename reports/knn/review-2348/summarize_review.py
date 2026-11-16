# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize the review measurements without pooling separate experiments."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

root = Path(__file__).resolve().parent
before, after = (json.loads((root / f"{x}.json").read_text()) for x in ("before", "after"))
if before["archive_sha256"] != after["archive_sha256"]:
    raise ValueError("Dataset hashes differ")
rows = []
memory = 0
for old, new in zip(before["cases"], after["cases"], strict=True):
    for key in ("name", "batches", "targets", "queries", "k"):
        if old[key] != new[key]:
            raise ValueError(f"Mismatched {key}")
    for variant, result in old["variants"].items():
        memory += (
            result["retained_index_and_output_pool_bytes"]
            == new["variants"][variant]["retained_index_and_output_pool_bytes"]
        )
        for phase, timing in result.items():
            if isinstance(timing, dict) and "median_ms" in timing:
                a = timing["median_ms"]
                b = new["variants"][variant][phase]["median_ms"]
                rows.append(
                    {
                        "case": old["name"],
                        "variant": variant,
                        "phase": phase,
                        "before_ms": a,
                        "after_ms": b,
                        "after_over_before": b / a,
                    }
                )
with (root / "cases.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
phases = {
    "build_and_query": "Build + query",
    "reused_query": "Reused query",
    "graph_query": "Query graph",
    "symmetric_chamfer_forward_backward": "Chamfer forward/backward",
    "mixed_chamfer_forward_backward": "Mixed-backend Chamfer",
}
summary = {"timing_records": len(rows), "identical_memory_pairs": memory, "phases": {}, "rechecks": []}
table = ["| Workload | Comparisons | Reviewed / original |", "| --- | ---: | ---: |"]
for phase, label in phases.items():
    subset = [row for row in rows if row["phase"] == phase]
    delta = 100 * (statistics.geometric_mean(row["after_over_before"] for row in subset) - 1)
    summary["phases"][phase] = {"count": len(subset), "percent_change": delta}
    table.append(f"| {label} | {len(subset)} | {delta:+.2f}% |")
records = json.loads((root / "investigation.json").read_text())
for row in sorted(rows, key=lambda r: r["after_over_before"], reverse=True):
    if row["after_over_before"] <= 1.05:
        continue
    original = [
        x["result"][row["phase"]]["median_ms"]
        for x in records
        if x["case"] == row["case"] and x["implementation"] == "original"
    ]
    reviewed = [
        x["result"][row["phase"]]["median_ms"]
        for x in records
        if x["case"] == row["case"] and x["implementation"] == "reviewed"
    ]
    ratios = [b / a for a, b in zip(original, reviewed, strict=True)]
    summary["rechecks"].append(
        {
            "case": row["case"],
            "phase": row["phase"],
            "initial_percent": 100 * (row["after_over_before"] - 1),
            "paired_round_percent": [100 * (r - 1) for r in ratios],
            "paired_geomean_percent": 100 * (statistics.geometric_mean(ratios) - 1),
        }
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "tables.md").write_text("\n".join(table) + "\n")
columns = [(v, p) for v in ("warp", "cubql_reuse", "cubql_build") for p in phases if not p.startswith("mixed")] + [
    ("cubql_reuse", "mixed_chamfer_forward_backward")
]
names = [x["name"] for x in before["cases"]]
lookup = {(r["case"], r["variant"], r["phase"]): 100 * (r["after_over_before"] - 1) for r in rows}
values = np.array([[lookup.get((name, v, p), np.nan) for v, p in columns] for name in names])
labels = {"warp": "WARP", "cubql_reuse": "CUBQL / REUSE", "cubql_build": "CUBQL / BUILD_AND_QUERY"}
short = {
    "build_and_query": "build + query",
    "reused_query": "query",
    "graph_query": "graph",
    "symmetric_chamfer_forward_backward": "Chamfer",
    "mixed_chamfer_forward_backward": "mixed Chamfer",
}
with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "knn-review-2348", "font.size": 9}):
    fig, ax = plt.subplots(figsize=(17, 12))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    plot = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6), aspect="auto")
    ax.set_xticks(range(len(columns)), [f"{labels[v]}\n{short[p]}" for v, p in columns], rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(columns)):
            val = values[i, j]
            ax.text(
                j,
                i,
                "n/a" if np.isnan(val) else f"{val:+.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(val) > 4.5 else "black",
            )
    ax.set_title(
        "Reviewed / original runtime: sequential full sweeps\nPositive is slower; >5% increases received paired rechecks (see report)"
    )
    fig.colorbar(plot, ax=ax, label="Runtime change (%)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(root / "before-after.svg")
    plt.close(fig)
print(json.dumps(summary, indent=2))
