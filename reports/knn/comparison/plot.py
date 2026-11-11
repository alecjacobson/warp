"""Plot saved exact-method comparisons; no GPU dependencies."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from summarize import passed

ROOT = Path(__file__).parent
METHODS = (
    "warp_default",
    "physicsnemo",
    "cubql",
    "cupy_kdtree",
    "keops",
    "faiss_flat",
    "cuvs_brute_force",
    "cuvs_unexpanded",
    "cuvs_minkowski",
    "pytorch3d",
)
LABELS = (
    "Warp default",
    "PhysicsNeMo",
    "cuBQL",
    "CuPy KDTree",
    "KeOps",
    "Faiss Flat",
    "cuVS sqL2",
    "cuVS direct L2",
    "cuVS p=2",
    "PyTorch3D",
)


def main():
    reports = {m: json.loads((ROOT / "results" / f"{m}.json").read_text()) for m in (*METHODS, "warp_configured")}
    baseline = {c["id"]: c for c in reports["warp_configured"]["cases"]}
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
                if c.get("one_shot") and passed(c.get("validation", {})) and passed(w.get("validation", {})):
                    ratios.append(c["one_shot"]["wall_median_ms"] / w["one_shot"]["wall_median_ms"])
            if len(ratios) == len(ids):
                ratio = statistics.geometric_mean(ratios)
                values[i, j] = np.log2(ratio)
                labels[i, j] = f"{ratio:.1f}x"
            else:
                labels[i, j] = f"checks\n{len(ratios)}/{len(ids)}"
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "point-query-comparison"}):
        fig, ax = plt.subplots(figsize=(13, 11))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#e8e8e8")
        im = ax.imshow(values, cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(groups)), [f"{f.replace('_', ' ')} · {n} · k={k}" for f, n, k in groups])
        for (i, j), label in labels.items():
            color = "white" if np.isfinite(values[i, j]) and abs(values[i, j]) > 2 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
        ax.set_title(
            "One-shot latency: alternative / configured Warp\nGeometric mean across each group's clean/noisy variants",
            pad=15,
        )
        bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[-3, -2, -1, 0, 1, 2, 3])
        bar.ax.set_yticklabels(["≤⅛x", "¼x", "½x", "1x", "2x", "4x", "≥8x"])
        fig.text(
            0.5,
            0.015,
            "Blue (>1): Warp faster. Red (<1): alternative faster. Gray: not all sampled accuracy checks passed.\nFive synchronized wall-time samples per case; one GPU slice. CAGRA approximation excluded.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(ROOT / "one-shot.svg", metadata={"Date": None})
        svg = ROOT / "one-shot.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        fig.savefig(ROOT / "one-shot.png", dpi=140)


if __name__ == "__main__":
    main()
