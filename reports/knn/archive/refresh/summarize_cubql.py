"""Validate frozen cuBQL tuning provenance and summarize both execution modes."""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    tuning_bytes = gzip.decompress((ROOT / "results/cubql-tuning.json.gz").read_bytes())
    tuning = json.loads(tuning_bytes)
    provenance = json.loads((ROOT / "cubql-provenance.json").read_text())
    require(hashlib.sha256(tuning_bytes).hexdigest() == provenance["tuning_sha256"], "Tuning hash mismatch")
    for path, expected in provenance["retained_result_sha256"].items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"Retained result changed: {path}")
    methods = {
        name: json.loads((ROOT / "results" / f"{name}.json").read_text()) for name in ("cubql_defaults", "cubql_tuned")
    }
    for report in methods.values():
        require(report["tuning_sha256"] == provenance["tuning_sha256"], "Evaluation tuning mismatch")
        require(report["provider"]["library_sha256"] == provenance["library_sha256"], "Evaluation library mismatch")
    labels = {
        "tiny": "N=1K, Q=32, k=32",
        "medium_k1": "N=100K, Q=1,024, k=1",
        "medium_k32": "N=100K, Q=1,024, k=32",
        "million": "N=1M, Q=8,192, k=11",
    }
    lines = [
        "# Selected cuBQL findKNN settings",
        "",
        "Settings were frozen on separate training inputs before the 40-case evaluation. Build-plus-query and reused-index objectives select independent trees. These are the best candidates tested in the documented hierarchical search.",
        "",
        "| Regime | Objective | Builder | Leaf threshold | Thread stride | Block size | Query ordering |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, row in tuning["regimes"].items():
        for phase, config in row["selected"].items():
            label = "Build + query" if phase == "one_shot" else "Reused index"
            lines.append(
                f"| {labels[name]} | {label} | {tuning['builders'][config['builder']]} | {config['leaf']} | {config['stride']} | {config['block']} | {config['ordered']} |"
            )
    lines += [
        "",
        "Leaf threshold 0 leaves the choice to the upstream builder. Defaults use spatial median with resolved leaf threshold 1, stride 1, block size 256 and no ordering. Every selected setting disables ordering after its full cost was measured.",
        "",
        "| Regime | Builder/leaf candidates | Launch candidates | Remeasured finalists |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in tuning["regimes"].items():
        lines.append(
            f"| {labels[name]} | {len(row['tree_screen'])} | {len(row['launch_screen'])} | {len(row['finalists'])} |"
        )
    (ROOT / "cubql-settings.md").write_text("\n".join(lines) + "\n")
    default = {c["id"]: c for c in methods["cubql_defaults"]["cases"]}
    tuned = {c["id"]: c for c in methods["cubql_tuned"]["cases"]}
    warp = {
        c["case"]["id"]: c["variants"]["warp_batch"]
        for c in json.loads((ROOT / "results/warp-replay.json").read_text())["cases"]
    }
    require(set(default) == set(tuned) == set(warp), "Case coverage mismatch")
    summary = {}
    lines = [
        "# cuBQL execution-mode comparison",
        "",
        "Geometric means across all 40 passing cases. Warp means the automatic batch API. Each row compares the same execution mode. No full-construction cuBQL graph is claimed.",
        "",
        "| Execution and objective | Defaults / tuned cuBQL | Tuned cuBQL faster than defaults | Warp / tuned cuBQL | Tuned cuBQL faster than Warp |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase, label in [
        ("one_shot", "Ordinary build + query"),
        ("reused_index_search", "Ordinary reused query"),
        ("graph_reused_index_search", "CUDA graph reused query"),
    ]:
        ratios = [default[cid][phase]["wall_median_ms"] / c[phase]["wall_median_ms"] for cid, c in tuned.items()]
        wratios = [warp[cid][phase]["wall_median_ms"] / c[phase]["wall_median_ms"] for cid, c in tuned.items()]
        summary[phase] = {
            "geomean_default_over_tuned": statistics.geometric_mean(ratios),
            "tuned_faster_than_defaults": sum(r > 1 for r in ratios),
            "geomean_warp_over_tuned": statistics.geometric_mean(wratios),
            "tuned_faster_than_warp": sum(r > 1 for r in wratios),
            "tuned_over_5pct_faster_than_warp": sum(r > 1.05 for r in wratios),
            "within_5pct_of_warp": sum(1 / 1.05 <= r <= 1.05 for r in wratios),
        }
        lines.append(
            f"| {label} | {statistics.geometric_mean(ratios):.2f}x | {sum(r > 1 for r in ratios)}/40 | {statistics.geometric_mean(wratios):.2f}x | {sum(r > 1 for r in wratios)}/40 |"
        )
    lines += [
        "",
        "A ratio above one favors tuned cuBQL. Build-plus-query and query-only use separately selected, frozen settings. Graphs use the query settings chosen from ordinary-call training; capture-specific tuning was not performed. Warp samples are retained from the previous update on the same GPU and environment, so small differences remain subject to run-to-run variation.",
    ]
    (ROOT / "cubql-timings.md").write_text("\n".join(lines) + "\n")
    (ROOT / "cubql-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
