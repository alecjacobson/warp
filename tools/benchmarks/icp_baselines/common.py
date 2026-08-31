# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared problem I/O and scoring for the ICP baseline comparison.

Every baseline registers the *same* fixed problem (written once by
``problem.py``) so accuracy and timing are comparable across libraries, even
though each runs in its own environment. Results accumulate in a JSON file that
``report.py`` prints as a table.
"""

import json
import os
import tempfile
import time

import numpy as np

BENCH_DIR = os.environ.get("ICP_BENCH_DIR", os.path.join(tempfile.gettempdir(), "warp_icp_baselines"))
PROBLEM_PATH = os.path.join(BENCH_DIR, "problem.npz")
RESULTS_PATH = os.path.join(BENCH_DIR, "results.json")


def load():
    data = np.load(PROBLEM_PATH)
    return {k: data[k] for k in data.files}


def rotation_error_deg(a, b):
    c = (np.trace(a[:3, :3] @ b[:3, :3].T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def score(transform, gt_inv):
    return rotation_error_deg(transform, gt_inv), float(np.linalg.norm(transform[:3, 3] - gt_inv[:3, 3]))


def timed(fn, repeats=3, sync=None):
    """Best-of-``repeats`` wall-clock (ms) of ``fn``; ``sync`` is called before
    stopping the clock (e.g. ``wp.synchronize_device``)."""
    best = np.inf
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if sync is not None:
            sync()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return out, best


def record(method, transform, ms, gt_inv, note=""):
    os.makedirs(BENCH_DIR, exist_ok=True)
    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    rot, trans = score(np.asarray(transform, dtype=float), gt_inv)
    results[method] = {"rot": rot, "trans": trans, "ms": ms, "note": note}
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)
    print(f"{method:<28}rot={rot:8.4f}  trans={trans:9.5f}  ms={ms:8.2f}  {note}")
