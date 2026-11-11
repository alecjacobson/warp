"""Summarize saved comparison JSON without importing any GPU libraries."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
METHODS = (
    "warp_default",
    "warp_configured",
    "physicsnemo",
    "cubql",
    "cupy_kdtree",
    "keops",
    "faiss_flat",
    "cuvs_brute_force",
    "cuvs_unexpanded",
    "cuvs_minkowski",
    "pytorch3d",
    "cuvs_cagra",
)


def passed(v):
    return (
        v.get("valid_shape", False)
        and v.get("valid_indices_and_finite_distances", False)
        and all(
            v.get(k, 1) == 0
            for k in (
                "duplicate_rows",
                "unsorted_rows",
                "distance_rank_mismatch_rows",
                "reported_distance_mismatch_rows",
            )
        )
    )


def main():
    reports = {
        m: json.loads((ROOT / "results" / f"{m}.json").read_text())
        for m in METHODS
        if (ROOT / "results" / f"{m}.json").exists()
    }
    cases = {m: {c["id"]: c for c in r["cases"]} for m, r in reports.items()}
    warp = cases["warp_configured"]
    rows = []
    for m, r in reports.items():
        for c in r["cases"]:
            row = {k: c.get(k) for k in ("id", "family", "N", "Q", "k", "sigma", "status")}
            row["method"] = m
            for phase in ("build", "reused_index_search", "one_shot"):
                timing = c.get(phase) or {}
                row[phase + "_wall_ms"] = timing.get("wall_median_ms")
                row[phase + "_event_ms"] = timing.get("event_median_ms")
            row["passes_sampled_checks"] = passed(c.get("validation", {}))
            for field in ("tie_aware_recall_at_k", "distance_rank_mismatch_rows", "reported_distance_mismatch_rows"):
                row[field] = c.get("validation", {}).get(field)
            rows.append(row)
    with (ROOT / "cases.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    lines = [
        "| Method | Completed | Pass sampled checks | Min recall | Warp wins / comparable | Geomean other / Warp | Query graph |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for m, report in reports.items():
        completed = [c for c in report["cases"] if c.get("status") == "ok"]
        valid = [c for c in completed if passed(c["validation"])]
        pairs = [(c, warp[c["id"]]) for c in valid if passed(warp[c["id"]].get("validation", {}))]
        ratios = [c["one_shot"]["wall_median_ms"] / w["one_shot"]["wall_median_ms"] for c, w in pairs]
        all_ratios = [c["one_shot"]["wall_median_ms"] / warp[c["id"]]["one_shot"]["wall_median_ms"] for c in completed]
        graph_file = ROOT / "results" / f"{m}-graph.json"
        graph = json.loads(graph_file.read_text()).get("capture", {}) if graph_file.exists() else {}
        graph_ok = graph.get("supported", False) and passed(graph.get("changed_query_validation", {}))
        recalls = [c["validation"].get("tie_aware_recall_at_k", 0) for c in completed]
        summary[m] = {
            "completed": len(completed),
            "passed": len(valid),
            "min_recall": min(recalls, default=0),
            "comparable": len(pairs),
            "warp_wins": sum(r > 1 for r in ratios),
            "geomean_ratio": statistics.geometric_mean(ratios) if ratios else None,
            "query_graph_captured": graph.get("supported", False),
            "query_graph_passed_sampled_checks": graph_ok,
            "all_timed_warp_wins": sum(r > 1 for r in all_ratios),
            "all_timed_geomean_ratio": statistics.geometric_mean(all_ratios) if all_ratios else None,
        }
        s = summary[m]
        exact = report.get("provider", {}).get("exact_algorithm", False)
        ratio_text = f"{s['geomean_ratio']:.2f}x" if ratios and exact else "—"
        wins = f"{s['warp_wins']}/{s['comparable']}" if exact else "ANN; separate"
        lines.append(
            f"| {m} | {len(completed)} | {len(valid)} | {s['min_recall']:.5f} | {wins} | {ratio_text} | {'pass' if graph_ok else 'failed/unsupported'} |"
        )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    raw_lines = [
        "| Method | Warp faster / timed cases | Geomean other / Warp | Cases passing sampled checks |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m, s in summary.items():
        if not reports[m].get("provider", {}).get("exact_algorithm", False):
            continue
        ratio = s["all_timed_geomean_ratio"]
        raw_lines.append(
            f"| {m} | {s['all_timed_warp_wins']}/{s['completed']} | {ratio:.2f}x | {s['passed']}/{s['completed']} |"
        )
    (ROOT / "all-timed-table.md").write_text("\n".join(raw_lines) + "\n")
    query_lines = [
        "| Method | Warp query wins / comparable | Geomean other / Warp |",
        "| --- | ---: | ---: |",
    ]
    for m, report in reports.items():
        if not report.get("provider", {}).get("exact_algorithm", False):
            continue
        ratios = []
        for c in report["cases"]:
            w = warp[c["id"]]
            phase = "reused_index_search" if c.get("reused_index_search") else "one_shot"
            validation = "reused_validation" if phase == "reused_index_search" else "validation"
            if c.get(phase) and passed(c.get(validation, {})) and passed(w.get("reused_validation", {})):
                ratios.append(c[phase]["wall_median_ms"] / w["reused_index_search"]["wall_median_ms"])
        geometric = f"{statistics.geometric_mean(ratios):.2f}x" if ratios else "—"
        query_lines.append(f"| {m} | {sum(r > 1 for r in ratios)}/{len(ratios)} | {geometric} |")
    (ROOT / "query-table.md").write_text("\n".join(query_lines) + "\n")
    for phase in ("one_shot", "reused_index_search"):
        outcomes = []
        for cid, w in warp.items():
            if not w.get(phase) or not passed(w.get("validation" if phase == "one_shot" else "reused_validation", {})):
                continue
            candidates = []
            for m, method_cases in cases.items():
                if m.startswith("warp_") or not reports[m].get("provider", {}).get("exact_algorithm", False):
                    continue
                c = method_cases.get(cid, {})
                # Methods without an index perform the same search on each call.
                metric = phase if c.get(phase) else "one_shot"
                val = "reused_validation" if metric == "reused_index_search" else "validation"
                if c.get(metric) and passed(c.get(val, {})):
                    candidates.append((c[metric]["wall_median_ms"], m))
            if candidates:
                best, name = min(candidates)
                outcomes.append(
                    {
                        "id": cid,
                        "best_other": name,
                        "other_ms": best,
                        "warp_ms": w[phase]["wall_median_ms"],
                        "ratio": best / w[phase]["wall_median_ms"],
                    }
                )
        (ROOT / f"best-{phase}.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        print(
            phase,
            "Warp faster than fastest passing alternative:",
            sum(x["ratio"] > 1 for x in outcomes),
            "/",
            len(outcomes),
        )
        for x in outcomes:
            if x["ratio"] <= 1:
                print(x)


if __name__ == "__main__":
    main()
