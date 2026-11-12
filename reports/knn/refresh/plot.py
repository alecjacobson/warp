"""Plot saved exact-method comparisons; no GPU dependencies."""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from summarize import load_reports, passed

ROOT = Path(__file__).parent
METHODS = (
    "warp_scalar",
    "physicsnemo",
    "cubql",
    "cubql_defaults",
    "cubql_tuned",
    "cupy_kdtree",
    "keops",
    "faiss_flat",
    "cuvs_brute_force",
    "cuvs_unexpanded",
    "cuvs_minkowski",
    "pytorch3d",
)
LABELS = (
    "Warp scalar (simple)",
    "PhysicsNeMo (pinned PR)",
    "cuBQL custom top-k\n(leaf 4, dense)",
    "cuBQL findKNN defaults",
    "cuBQL findKNN\n(best tested settings)",
    "CuPy KDTree",
    "KeOps",
    "Faiss Flat",
    "cuVS sqL2",
    "cuVS direct L2",
    "cuVS p=2",
    "PyTorch3D",
)


def main(phase):
    reports = load_reports()
    baseline = {c["id"]: c for c in reports["warp_batch"]["cases"]}
    groups = {}
    for c in baseline.values():
        size = "1M" if c["N"] == 1_000_000 else "1K" if c["N"] == 1000 else f"{round(c['N'] / 1000)}K"
        groups.setdefault((c["family"], size, c["k"]), []).append(c["id"])
    values = np.full((len(groups), len(METHODS)), np.nan)
    labels = {}
    for j, m in enumerate(METHODS):
        cases = {c["id"]: c for c in reports[m]["cases"]}
        for i, (_, ids) in enumerate(groups.items()):
            ratios = []
            for cid in ids:
                c = cases.get(cid, {})
                w = baseline[cid]
                metric = phase if c.get(phase) else "one_shot"
                validation = "reused_validation" if metric == "reused_index_search" else "validation"
                warp_validation = "reused_validation" if phase == "reused_index_search" else "validation"
                if c.get(metric) and passed(c.get(validation, {})) and passed(w.get(warp_validation, {})):
                    ratios.append(c[metric]["wall_median_ms"] / w[phase]["wall_median_ms"])
            if len(ratios) == len(ids):
                ratio = statistics.geometric_mean(ratios)
                values[i, j] = np.log2(ratio)
                labels[i, j] = f"{ratio:.1f}x"
            else:
                labels[i, j] = f"checks\n{len(ratios)}/{len(ids)}"
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "point-query-comparison"}):
        fig, ax = plt.subplots(figsize=(15.5, 11))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#e8e8e8")
        im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6), aspect="auto")
        ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(groups)), [f"{f.replace('_', ' ')} · {n} · k={k}" for f, n, k in groups])
        for (i, j), label in labels.items():
            color = "white" if np.isfinite(values[i, j]) and (values[i, j] < -1.5 or values[i, j] > 4.5) else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
        title = (
            "Build + query" if phase == "one_shot" else "Reused-index query (full call for operators without an index)"
        )
        ax.set_title(
            title
            + ": alternative / Warp batch (automatic tuning, ordinary call)\nGeometric mean across each group's clean/noisy variants",
            pad=15,
        )
        bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        bar.ax.set_yticklabels(["<=1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x", "16x", "32x", ">=64x"])
        fig.text(
            0.5,
            0.015,
            "Values >1: Warp faster. Values <1: alternative faster. Gray: not all sampled accuracy checks passed.\nNine synchronized wall samples; fresh Warp/cuBQL, other timings retained. No graph replay in this plot.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        stem = "one-shot" if phase == "one_shot" else "reused-query"
        fig.savefig(ROOT / f"{stem}.svg", metadata={"Date": None})
        svg = ROOT / f"{stem}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        fig.savefig(Path("/tmp") / f"knn-refresh-{stem}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main("one_shot")
    main("reused_index_search")
