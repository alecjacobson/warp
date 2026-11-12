"""Plot only saved CUDA graph replay timings; never substitute ordinary calls."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from plot import LABELS as ORDINARY_LABELS
from plot import METHODS as ORDINARY_METHODS
from summarize import load_reports, passed

ROOT = Path(__file__).parent
METHODS = (ORDINARY_METHODS[0], "warp_tuned", *ORDINARY_METHODS[1:])
LABELS = (ORDINARY_LABELS[0], "Warp scalar (tuned launch)", *ORDINARY_LABELS[1:])


def validate_timer(timer, context):
    for field in ("wall_samples_ms", "event_samples_ms"):
        samples = timer[field]
        if len(samples) != 9 or not all(np.isfinite(x) and x > 0 for x in samples):
            raise ValueError(f"Invalid graph samples: {context} {field}")
    if not np.isclose(timer["wall_median_ms"], statistics.median(timer["wall_samples_ms"])):
        raise ValueError(f"Invalid graph median: {context}")
    if not passed(timer["validation"]) or not passed(timer["changed_input_validation"]):
        raise ValueError(f"Graph replay checks failed: {context}")


def main():
    reports = load_reports()
    baseline = {c["id"]: c for c in reports["warp_batch"]["cases"]}
    cases = {m: {c["id"]: c for c in reports[m]["cases"]} for m in METHODS}
    for method, rows in cases.items():
        if len(rows) != len(reports[method]["cases"]) or set(rows) != set(baseline):
            raise ValueError(f"Incomplete or duplicate cases: {method}")
    failure = json.loads((ROOT / "results/cubql-full-capture.json").read_text())
    if failure["supported"] is not False:
        raise ValueError("Expected the recorded cuBQL full-build capture failure")
    groups = {}
    for c in baseline.values():
        size = "1M" if c["N"] == 1_000_000 else "1K" if c["N"] == 1000 else f"{round(c['N'] / 1000)}K"
        groups.setdefault((c["family"], size, c["k"]), []).append(c["id"])

    for phase, stem, title in (
        ("graph_one_shot", "graph-one-shot", "Build + query"),
        ("graph_reused_index_search", "graph-reused-query", "Reused-index query"),
    ):
        for cid, case in baseline.items():
            validate_timer(case[phase], f"baseline {cid} {phase}")
        values = np.full((len(groups), len(METHODS)), np.nan)
        labels = {}
        measured = 0
        for j, method in enumerate(METHODS):
            for i, ids in enumerate(groups.values()):
                ratios = []
                for cid in ids:
                    timer = cases[method][cid].get(phase)
                    if timer is not None:
                        validate_timer(timer, f"{method} {cid} {phase}")
                        ratios.append(timer["wall_median_ms"] / baseline[cid][phase]["wall_median_ms"])
                        measured += 1
                if len(ratios) == len(ids):
                    ratio = statistics.geometric_mean(ratios)
                    values[i, j] = np.log2(ratio)
                    labels[i, j] = f"{ratio:.2f}x"
                elif ratios:
                    raise ValueError(f"Partial graph coverage: {method} {phase} {ids}")
                elif method == "warp_batch_cubql" and phase == "graph_one_shot":
                    labels[i, j] = "capture\nunsupported"
                else:
                    labels[i, j] = "not\nmeasured"

        with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "point-query-graph"}):
            fig, ax = plt.subplots(figsize=(19, 11))
            cmap = plt.get_cmap("RdBu").copy()
            cmap.set_bad("#e8e8e8")
            im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6), aspect="auto")
            ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right")
            ax.set_yticks(range(len(groups)), [f"{f.replace('_', ' ')} · {n} · k={k}" for f, n, k in groups])
            for (i, j), label in labels.items():
                value = values[i, j]
                color = "white" if np.isfinite(value) and (value < -1.5 or value > 4.5) else "black"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
            ax.set_title(
                f"CUDA graph replay — {title}: method / retained Warp batch (LBVH, graph replay)\n"
                "Geometric mean across each group's clean/noisy variants; 1x = reference batch graph replay",
                pad=15,
            )
            bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
            bar.ax.set_yticklabels(["<=1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x", "16x", "32x", ">=64x"])
            fig.text(
                0.5,
                0.015,
                "Blue / >1: reference LBVH batch faster. Red / <1: column method faster. Gray: no replay timing.\n"
                "Nine synchronized single-replay wall samples; capture/instantiation excluded. Saved measurements only.\n"
                "Capture unsupported: recorded full-build failure. Not measured: no saved timing; does not establish capture support.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.075, 1, 1))
            svg = ROOT / f"{stem}.svg"
            fig.savefig(svg, metadata={"Date": None})
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
            fig.savefig(Path("/tmp") / f"knn-refresh-{stem}.png", dpi=140)
            plt.close(fig)
        print(f"{stem}: {len(groups)} rows; {measured} method/case graph timings plus {len(baseline)} baseline checks")


if __name__ == "__main__":
    main()
