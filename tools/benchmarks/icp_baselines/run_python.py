# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the importable Python ICP baselines on the shared benchmark problem.

Each baseline is optional and import-guarded; whatever is installed in the
current environment runs and is recorded, the rest are skipped. Typically run in
a separate venv per baseline (see this directory's README):

    uv run --python 3.12 --with open3d run_python.py         # Open3D
    LD_LIBRARY_PATH=<fast_gicp build> python run_python.py   # + fast_gicp
    python run_python.py                                     # + PyTorch3D
"""

# Each baseline is imported lazily inside its adapter so a missing library only
# skips that one method; keep those imports local.
# ruff: noqa: PLC0415

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import load, record, timed

P = load()
SRC, TGT, NRM = P["source"], P["target"], P["normals"]
GT_INV, MCD = P["gt_inv"], float(P["max_dist"])


def run_open3d():
    import open3d as o3d

    dev = o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available() else o3d.core.Device("CPU:0")
    f32 = o3d.core.float32

    def cloud(points, normals=None):
        pc = o3d.t.geometry.PointCloud(dev)
        pc.point.positions = o3d.core.Tensor(points.astype(np.float32), f32, dev)
        if normals is not None:
            pc.point.normals = o3d.core.Tensor(normals.astype(np.float32), f32, dev)
        return pc

    source, target_pl, target_pt = cloud(SRC), cloud(TGT, NRM), cloud(TGT)
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(1e-8, 1e-8, 50)
    init = o3d.core.Tensor(np.eye(4).astype(np.float32), f32, dev)
    reg = o3d.t.pipelines.registration
    for name, est, tgt in [
        ("open3d_point_to_plane", reg.TransformationEstimationPointToPlane(), target_pl),
        ("open3d_point_to_point", reg.TransformationEstimationPointToPoint(), target_pt),
    ]:
        out, ms = timed(lambda est=est, tgt=tgt: reg.icp(source, tgt, MCD, init, est, criteria))
        record(name, out.transformation.cpu().numpy(), ms, GT_INV, f"Open3D {dev}")


def run_fast_gicp():
    import pygicp

    for name, cls in [("fast_gicp_FastGICP", pygicp.FastGICP), ("fast_gicp_FastVGICP", pygicp.FastVGICP)]:

        def run(cls=cls):
            reg = cls()
            reg.set_num_threads(8)
            reg.set_max_correspondence_distance(MCD)
            reg.set_input_target(TGT.astype(np.float64))
            reg.set_input_source(SRC.astype(np.float64))
            return reg.align(np.eye(4))

        out, ms = timed(run)
        record(name, out, ms, GT_INV, "CPU 8-thread")


def run_pytorch3d():
    import torch
    from pytorch3d.ops import iterative_closest_point

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.tensor(SRC, dtype=torch.float32, device=dev)[None]
    target = torch.tensor(TGT, dtype=torch.float32, device=dev)[None]
    sync = torch.cuda.synchronize if dev == "cuda" else None

    def run():
        sol = iterative_closest_point(source, target, max_iterations=50)
        T = np.eye(4)
        T[:3, :3] = sol.RTs.R[0].cpu().numpy().T  # PyTorch3D uses X @ R
        T[:3, 3] = sol.RTs.T[0].cpu().numpy()
        return T

    out, ms = timed(run, sync=sync)
    record("pytorch3d_point_to_point", out, ms, GT_INV, f"PyTorch3D {dev}")


if __name__ == "__main__":
    for label, fn in [("Open3D", run_open3d), ("fast_gicp", run_fast_gicp), ("PyTorch3D", run_pytorch3d)]:
        try:
            fn()
        except ImportError:
            print(f"{label:<28}skipped (not installed)")
        except Exception as e:
            print(f"{label:<28}FAILED: {type(e).__name__}: {e}")
