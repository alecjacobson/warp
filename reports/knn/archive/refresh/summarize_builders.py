"""Summarize matched Warp query measurements with two BVH constructors."""

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    report = json.loads((ROOT / "results/warp-builders.json").read_text())
    provenance = json.loads((ROOT / "builders-provenance.json").read_text())
    for path, expected in provenance["retained_result_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Retained result changed: {path}")
    summary = {}
    lines = [
        "# Matched Warp kNN builder timings",
        "",
        "Both variants use the same Warp batch query kernels and requested leaf/launch settings. Only the BVH constructor changes. Values below compare fresh paired measurements; the main colored plots retain the earlier LBVH batch measurement as their common denominator.",
        "",
        "| Operation | cuBQL SAH relative to LBVH | cuBQL SAH lower median |",
        "| --- | ---: | ---: |",
    ]
    for row in report["cases"]:
        configs = [dict(v["config"]) for v in row["variants"].values()]
        for config in configs:
            config.pop("constructor")
        if configs[0] != configs[1]:
            raise ValueError("Builder settings do not match")
    for phase, name in (
        ("one_shot", "Build + query, ordinary call"),
        ("reused_index_search", "Reused query, ordinary call"),
        ("graph_reused_index_search", "Reused query, CUDA graph replay"),
    ):
        ratios = [
            r["variants"]["warp_batch_lbvh"][phase]["wall_median_ms"]
            / r["variants"]["warp_batch_cubql"][phase]["wall_median_ms"]
            for r in report["cases"]
        ]
        gm = statistics.geometric_mean(ratios)
        summary[phase] = {
            "geomean_lbvh_over_cubql": gm,
            "cubql_lower_median": sum(r > 1 for r in ratios),
            "cubql_over_5pct_faster": sum(r > 1.05 for r in ratios),
            "within_5pct": sum(1 / 1.05 <= r <= 1.05 for r in ratios),
        }
        relative = f"{gm:.2f}x faster" if gm >= 1 else f"{1 / gm:.2f}x slower"
        lines.append(f"| {name} | {relative} | {sum(r > 1 for r in ratios)}/40 |")
    old = {
        r["case"]["id"]: r["variants"]["warp_batch"]
        for r in json.loads((ROOT / "results/warp-replay.json").read_text())["cases"]
    }
    summary["retained_lbvh_over_matched_lbvh"] = {
        phase: statistics.geometric_mean(
            old[r["case"]["id"]][phase]["wall_median_ms"] / r["variants"]["warp_batch_lbvh"][phase]["wall_median_ms"]
            for r in report["cases"]
        )
        for phase in ("one_shot", "reused_index_search", "graph_reused_index_search")
    }
    lines += [
        "",
        "Geometric means over 40 cases. Counts include small differences and are not significance tests. Full-build graph capture succeeds for LBVH; cuBQL construction fails the isolated probe because it synchronizes with the host. Both query graphs pass replay checks after query values change.",
        "",
        "| Case | LBVH build + query ms | cuBQL build + query ms | LBVH reused ms | cuBQL reused ms | LBVH query graph ms | cuBQL query graph ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["cases"]:
        values = [
            r["variants"][m][phase]["wall_median_ms"]
            for phase in ("one_shot", "reused_index_search", "graph_reused_index_search")
            for m in ("warp_batch_lbvh", "warp_batch_cubql")
        ]
        lines.append("| " + r["case"]["id"] + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    (ROOT / "builders-timings.md").write_text("\n".join(lines) + "\n")
    (ROOT / "builders-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
