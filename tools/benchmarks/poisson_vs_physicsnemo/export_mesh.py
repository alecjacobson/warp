# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the bundled Stanford bunny to ``mesh.npz`` so both benchmark scripts
sample the *identical* mesh (they run in separate virtual environments).

Run once, in the Warp environment (needs ``usd-core`` for the ``.usd`` asset)::

    uv run --with usd-core tools/benchmarks/poisson_vs_physicsnemo/export_mesh.py
"""

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp.examples

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            verts = np.array(m.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
            diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
            out = os.path.join(HERE, "mesh.npz")
            np.savez(out, vertices=verts, faces=faces, diag=diag)
            print(f"wrote {out}: {len(verts)} verts, {len(faces)} faces, bbox diag {diag:.4f}")
            return
    raise RuntimeError("no mesh found in bunny.usd")


if __name__ == "__main__":
    main()
