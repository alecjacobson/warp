# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print the accumulated ICP baseline results as a table, sorted by accuracy.

uv run tools/benchmarks/icp_baselines/report.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_PATH


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"no results at {RESULTS_PATH}; run the runners first")
        return
    results = json.load(open(RESULTS_PATH))
    print(f"{'method':<28}{'rot (deg)':>10}{'trans':>11}{'time (ms)':>11}   note")
    print("-" * 74)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["rot"]):
        print(f"{name:<28}{r['rot']:>10.4f}{r['trans']:>11.5f}{r['ms']:>11.2f}   {r.get('note', '')}")


if __name__ == "__main__":
    main()
