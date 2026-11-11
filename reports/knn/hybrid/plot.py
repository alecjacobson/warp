# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot measured speed ratios; average clean/noisy variants geometrically."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).parent
matplotlib.rcParams["svg.hashsalt"] = "warp-knn-hybrid"
rows = [r for r in csv.DictReader((ROOT / "cases.csv").open()) if r["suite"] == "pilot"]
groups = {}
for row in rows:
    label = f"{row['family']} · N={int(row['N']):,} · Q={int(row['Q']):,} · k={row['k']}"
    groups.setdefault(label, []).append(row)
methods = ("warp_before", "physicsnemo", "physicsnemo_tuned", "cuvs_unexpanded")
values = np.array(
    [
        [math.exp(np.mean([math.log(float(r[m]) / float(r["warp_configured"])) for r in rs])) for m in methods]
        for rs in groups.values()
    ]
)
fig, ax = plt.subplots(figsize=(11, 9))
colors = ax.imshow(np.log2(values), cmap="RdBu", norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=6), aspect="auto")
for i in range(len(values)):
    for j in range(len(methods)):
        ax.text(
            j,
            i,
            f"{values[i, j]:.2f}x",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if np.log2(values[i, j]) > 3.0 or np.log2(values[i, j]) < -1.5 else "black",
        )
ax.set_xticks(range(4), ["Previous Warp", "PhysicsNeMo\npinned", "PhysicsNeMo\nequally tuned", "cuVS\ndirect L2"])
ax.set_yticks(range(len(groups)), list(groups), fontsize=8)
ax.set_title("Reference latency / configured Warp latency\nBlue (>1): Warp faster · red (<1): reference faster", pad=18)
fig.colorbar(colors, ax=ax, label="log₂ speed ratio", fraction=0.025)
fig.tight_layout()
fig.savefig(ROOT / "ratios.svg", metadata={"Date": None})
fig.savefig("/tmp/warp-knn-hybrid-ratios.png", dpi=140)
p = ROOT / "ratios.svg"
p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
