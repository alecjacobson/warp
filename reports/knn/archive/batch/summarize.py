"""Validate and summarize saved Warp-only batch/scalar measurements."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    summary, rows = {}, []
    for suite, count in (("pilot", 40), ("dense", 8)):
        report = json.loads((ROOT / f"{suite}.json").read_text())
        if len(report["cases"]) != count or report["torch_imported"] or report["repeats"] != 9:
            raise ValueError(f"Incomplete or non-Warp-only run: {suite}")
        for row in report["cases"]:
            for name, v in row["variants"].items():
                for phase, validation in (("one_shot", "accuracy"), ("reused_index_search", "reused_accuracy")):
                    check = v[validation]
                    if (
                        not check.get("valid_shape")
                        or not check.get("valid_indices_and_finite_distances")
                        or any(
                            check.get(k, 1)
                            for k in (
                                "duplicate_rows",
                                "unsorted_rows",
                                "distance_rank_mismatch_rows",
                                "reported_distance_mismatch_rows",
                            )
                        )
                        or len(v[phase]["wall_samples_ms"]) != 9
                    ):
                        raise ValueError(f"Failed validation: {suite} {row['case']['id']} {name} {phase}")
                    rows.append(
                        {
                            "suite": suite,
                            "case": row["case"]["id"],
                            "method": name,
                            "phase": phase,
                            "wall_median_ms": v[phase]["wall_median_ms"],
                        }
                    )
        summary[suite] = {}
        for ref in ("configured_scalar", "dense_scalar"):
            summary[suite][ref] = {}
            for phase in ("one_shot", "reused_index_search"):
                ratios = [
                    row["variants"][ref][phase]["wall_median_ms"] / row["variants"]["batch"][phase]["wall_median_ms"]
                    for row in report["cases"]
                ]
                summary[suite][ref][phase] = {
                    "geomean_reference_over_batch": statistics.geometric_mean(ratios),
                    "min_ratio": min(ratios),
                    "max_ratio": max(ratios),
                    "batch_lower_medians": sum(r > 1 for r in ratios),
                    "within_5pct": sum(1 / 1.05 <= r <= 1.05 for r in ratios),
                    "batch_over_5pct_slower": sum(r < 1 / 1.05 for r in ratios),
                }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "cases.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
