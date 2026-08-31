# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the compiled PCL baseline (see ``pcl/``) on the shared benchmark problem.

Build the binary first::

    cmake -S tools/benchmarks/icp_baselines/pcl -B <build> -DCMAKE_BUILD_TYPE=Release
    cmake --build <build> -j

Then::

    uv run --with scipy tools/benchmarks/icp_baselines/run_pcl.py <build>/pcl_icp
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import BENCH_DIR, load, record


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "pcl_icp"
    p = load()
    gt_inv = p["gt_inv"]
    mcd = str(float(p["max_dist"]))
    source = os.path.join(BENCH_DIR, "source.xyz")
    target = os.path.join(BENCH_DIR, "target.xyz")
    out = subprocess.check_output([binary, source, target, mcd]).decode()
    for line in out.strip().splitlines():
        fields = line.split()
        if fields[0] != "RESULT":
            continue
        name = fields[1]
        transform = np.array(fields[2:18], float).reshape(4, 4)
        ms = float(fields[18])
        record(name, transform, ms, gt_inv, "CPU (PCL)")


if __name__ == "__main__":
    main()
