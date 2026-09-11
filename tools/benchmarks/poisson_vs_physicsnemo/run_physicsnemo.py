# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PhysicsNeMo's Warp Poisson-disk sampler on the shared bunny mesh.

The PhysicsNeMo release that contains this op pins a torch build too new for
many drivers, so instead of installing the whole package this script fetches the
two source files of the sampler at a *pinned commit* into a git-ignored
``_vendor/`` directory and runs them verbatim. The only thing replaced is the
``FunctionSpec`` device/stream helper (a tiny stub that binds Warp to torch's
current CUDA stream) -- every GPU kernel is PhysicsNeMo's own code, unchanged.

Run in a separate environment that has torch (CUDA), Warp, and SciPy::

    python -m venv .venv && . .venv/bin/activate
    pip install "torch" warp-lang scipy --index-url <a CUDA build matching your driver>
    python tools/benchmarks/poisson_vs_physicsnemo/run_physicsnemo.py

See ``README.md`` for the full workflow, the pinned commit, and results.
"""

import importlib
import os
import sys
import urllib.request

import numpy as np
import torch
from common import RADIUS_FRACTIONS, best_of, load_mesh, min_dist_over_radius

import warp as wp

# Pinned PhysicsNeMo revision (NVIDIA/physicsnemo, Apache-2.0). Update deliberately.
PNEMO_COMMIT = "b8955662fb5339510970921e0c9cc57bb6242b1b"
PNEMO_DIR = "physicsnemo/nn/functional/geometry/mesh_poisson_disk_sample/_warp_impl"
RAW = f"https://raw.githubusercontent.com/NVIDIA/physicsnemo/{PNEMO_COMMIT}/{PNEMO_DIR}"

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "_vendor")

# Faithful stand-in for physicsnemo.core.function_spec.FunctionSpec: only the two
# members the op uses, binding Warp to the same CUDA stream torch is on so the
# copied kernels run exactly as they would inside PhysicsNeMo.
_FUNCTION_SPEC = """\
import contextlib
import torch
import warp as wp


class FunctionSpec:
    @staticmethod
    def warp_launch_context(t):
        idx = t.device.index if t.device.index is not None else 0
        dev = wp.get_device(f"cuda:{idx}")
        stream = wp.Stream(dev, cuda_stream=torch.cuda.current_stream(device=t.device).cuda_stream)
        return dev, stream

    @staticmethod
    @contextlib.contextmanager
    def warp_stream_scope(stream):
        prev = wp.get_stream(stream.device)
        wp.set_stream(stream, stream.device)
        try:
            yield
        finally:
            wp.set_stream(prev, stream.device)
"""


def _ensure_vendored():
    """Fetch op.py + _kernels.py at the pinned commit and wire them into a local
    package that uses the FunctionSpec stub. Idempotent."""
    os.makedirs(VENDOR, exist_ok=True)
    open(os.path.join(VENDOR, "__init__.py"), "w").close()
    open(os.path.join(VENDOR, "function_spec.py"), "w").write(_FUNCTION_SPEC)
    for name in ("_kernels.py", "op.py"):
        dst = os.path.join(VENDOR, name)
        if not os.path.exists(dst):
            src = urllib.request.urlopen(f"{RAW}/{name}", timeout=60).read().decode()
            if name == "op.py":
                src = src.replace(
                    "from physicsnemo.core.function_spec import FunctionSpec",
                    "from .function_spec import FunctionSpec",
                )
            open(dst, "w").write(src)
    print(f"vendored PhysicsNeMo sampler @ {PNEMO_COMMIT[:12]} -> {VENDOR}")


def main():
    wp.init()
    _ensure_vendored()
    sys.path.insert(0, HERE)
    op = importlib.import_module("_vendor.op")
    sample = op.mesh_poisson_disk_sample_warp

    verts, faces, diag = load_mesh()
    v = torch.from_numpy(verts).cuda()
    f = torch.from_numpy(faces.reshape(-1).astype(np.int64)).cuda()

    print(f"PhysicsNeMo mesh_poisson_disk_sample_warp  |  device=cuda  bbox_diag={diag:.4f}")
    print(f"{'radius':>10} {'frac':>6} {'points':>8} {'best_ms':>9} {'pts/s':>12} {'min-d/r':>8}")
    for frac in RADIUS_FRACTIONS:
        r = frac * diag

        def once(r=r):
            out = sample(v, f, min_distance=r)
            torch.cuda.synchronize()
            return out

        once()
        best = best_of(once)
        pts = sample(v, f, min_distance=r).detach().cpu().numpy()
        q = min_dist_over_radius(pts, r)
        print(f"{r:10.4f} {frac:6.1%} {len(pts):8d} {best * 1e3:9.2f} {len(pts) / best:12.0f} {q:8.3f}")


if __name__ == "__main__":
    main()
