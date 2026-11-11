# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize retained measurements; ratios above one favor configured Warp."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def valid(a):
    return (
        a.get("valid_shape", False)
        and a.get("valid_indices_and_finite_distances", False)
        and all(
            a.get(k, -1) == 0
            for k in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        )
    )


def aggregate(rows, method):
    ratios = [r[method] / r["warp_configured"] for r in rows if r.get(method) is not None]
    return {
        "cases": len(ratios),
        "lower_median": sum(r > 1 for r in ratios),
        "more_than_5_percent_faster": sum(r > 1.05 for r in ratios),
        "within_5_percent": sum(0.95 <= r <= 1.05 for r in ratios),
        "more_than_5_percent_slower": sum(r < 0.95 for r in ratios),
        "geomean_ratio": math.exp(sum(map(math.log, ratios)) / len(ratios)),
        "minimum_ratio": min(ratios),
        "maximum_ratio": max(ratios),
    }


def main():
    before = {r["case"]["id"]: r for r in json.loads((ROOT / "results/before.json").read_text())}
    extras = {}
    for name, file in (("cuvs_unexpanded", "cuvs_unexpanded.json"), ("warp_default", "warp_default.json")):
        data = json.loads((ROOT / "results" / file).read_text())
        for row in data["cases"]:
            if row["status"] != "ok" or not valid(row["validation"]):
                raise ValueError((name, row))
        extras[name] = {r["id"]: r["one_shot"]["wall_median_ms"] for r in data["cases"]}
    summary = {}
    all_rows = []
    for suite, filename in (("pilot", "final.json"), ("holdout", "holdout.json")):
        rows = []
        for item in json.loads((ROOT / "results" / filename).read_text()):
            case = item["case"]
            row = {"suite": suite, **{k: case.get(k) for k in ("id", "family", "N", "Q", "k", "sigma", "seed")}}
            for method, result in item["variants"].items():
                if not valid(result["accuracy"]):
                    raise ValueError((case["id"], method, result["accuracy"]))
                row[method] = result["one_shot"]["wall_median_ms"]
            if suite == "pilot":
                row["warp_before"] = before[case["id"]]["variants"]["warp_configured"]["one_shot"]["wall_median_ms"]
                for name, data in extras.items():
                    row[name] = data[case["id"]]
            rows.append(row)
        methods = ["physicsnemo", "physicsnemo_isotropic", "physicsnemo_tuned"]
        if suite == "pilot":
            methods += ["warp_before", "warp_default", "cuvs_unexpanded"]
        summary[suite] = {m: aggregate(rows, m) for m in methods}
        groups = {}
        for r in rows:
            label = f"N=1M Q={r['Q']} k={r['k']}" if r["N"] == 1_000_000 else f"Q={r['Q']} k={r['k']}"
            groups.setdefault(label, []).append(r)
        summary[suite]["regimes"] = {g: {m: aggregate(rs, m) for m in methods} for g, rs in groups.items()}
        all_rows += rows
    captures = json.loads((ROOT / "results/capture.json").read_text())
    for name, capture in captures.items():
        if isinstance(capture, dict):
            if not capture["supported"] or not all(map(valid, capture["changed_point_and_query_validations"])):
                raise ValueError((name, capture))
    cuvs_graph = json.loads((ROOT / "results/cuvs_unexpanded-graph.json").read_text())["capture"]
    if not cuvs_graph["supported"] or not valid(cuvs_graph["changed_query_validation"]):
        raise ValueError(cuvs_graph)
    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    with (ROOT / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for suite in ("pilot", "holdout"):
        table = [
            "| Reference | Warp lower median | Reference / Warp GM | >5% faster / within 5% / >5% slower |",
            "| --- | ---: | ---: | ---: |",
        ]
        for method, v in summary[suite].items():
            if method == "regimes":
                continue
            table.append(
                f"| {method} | {v['lower_median']}/{v['cases']} | {v['geomean_ratio']:.2f}x | {v['more_than_5_percent_faster']} / {v['within_5_percent']} / {v['more_than_5_percent_slower']} |"
            )
        (ROOT / f"{suite}-table.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
