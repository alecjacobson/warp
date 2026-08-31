# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run warp.geometry.register_rigid on the shared benchmark problem.

uv run tools/benchmarks/icp_baselines/run_warp.py
"""

import os
import sys

import numpy as np

import warp as wp
import warp.geometry as geo

sys.path.insert(0, os.path.dirname(__file__))
from common import load, record, timed

wp.init()
DEVICE = "cuda:0"


def main():
    p = load()
    src = p["source"].astype(np.float32)
    tgt = p["target"].astype(np.float32)
    normals = p["normals"].astype(np.float32)
    faces = p["faces"].astype(np.int32)
    gt_inv = p["gt_inv"]
    mcd = float(p["max_dist"])
    sync = wp.synchronize_device

    geo.register_rigid(src, (tgt, normals), max_iters=5, tol=0.0, max_corr_dist=mcd, device=DEVICE)
    sync()

    configs = [
        ("warp_point_to_plane_cloud", (tgt, normals), {}),
        ("warp_symmetric_cloud", (tgt, normals), {"variant": "symmetric"}),
        ("warp_point_to_plane_mesh", (tgt, faces), {}),
    ]
    for name, target, kw in configs:
        out, ms = timed(
            lambda target=target, kw=kw: (
                geo.register_rigid(
                    src, target, max_iters=50, tol=1e-8, max_corr_dist=mcd, device=DEVICE, **kw
                ).transform
            ),
            sync=sync,
        )
        record(name, out, ms, gt_inv, "L40 cuda:0")


if __name__ == "__main__":
    main()
