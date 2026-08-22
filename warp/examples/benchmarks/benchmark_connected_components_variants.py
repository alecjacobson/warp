# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Benchmark: parallel connected-components variants
#
# Reproducible comparison harness (non-gating) behind the algorithm choice in
# warp.geometry.connected_components. Each variant produces per-vertex root
# labels from a flat int32 triangle-index array (edges = the 3 per triangle):
#
#   LP        pure label propagation (atomic-min, no compression)
#   PJ1       hook + single grandparent pointer jump
#   hook+full hook + full path compression      <-- chosen for production
#   UF        union-find: in-kernel find (path halving) + atomic-min root hook
#   FastSV    two hooks + shortcut per round
#
# It first cross-checks every variant against a serial union-find reference
# (the correctness net), then times them across mesh topologies on CPU/CUDA and
# reports median milliseconds and round counts.
#
# Findings (this machine, L40 / x86_64):
#   * On CUDA the cost is round-count bound: full path compression converges in
#     ~2 rounds on every topology, versus ~9-18 for single-jump and hundreds to
#     thousands for pure LP on high-diameter graphs.
#   * On CPU kernels run in-order, so LP resolves in ~2 sweeps and per-round cost
#     dominates; UF's in-kernel find/retry is ~3x slower there than hook+full.
#   * hook+full is the robust cross-device pick: best-or-tied on CPU and, on
#     CUDA, within noise of UF except on pathological 1M-long strips.
#
# Run: uv run warp/examples/benchmarks/benchmark_connected_components_variants.py [--device ...]
###########################################################################

"""Parallel connected-components variant comparison (see module header)."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import warp as wp
from warp._src.utils import array_scan  # noqa: F401  (kept for parity experiments)

_CAP = 3000  # hard round cap (LP on high-diameter graphs may not converge)


# ---------------------------------------------------------------------------
# Shared kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _init(labels: wp.array[wp.int32]):
    labels[wp.tid()] = wp.tid()


@wp.kernel(enable_backward=False)
def _flatten(labels: wp.array[wp.int32], out: wp.array[wp.int32]):
    # Read-only chase to root (labels is not mutated), for uniform readout.
    v = wp.tid()
    r = v
    while labels[r] != r:
        r = labels[r]
    out[v] = r


@wp.func
def _find(labels: wp.array[wp.int32], x0: wp.int32) -> wp.int32:
    # Find with path halving (writes compress the tree on the fly).
    x = x0
    px = labels[x]
    while px != x:
        gpx = labels[px]
        labels[x] = gpx
        x = gpx
        px = labels[x]
    return x


# ---------------------------------------------------------------------------
# Variant kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _lp_step(indices: wp.array[wp.int32], labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
    # Pure label propagation: both endpoints take the min of the two labels.
    t = wp.tid()
    for c in range(3):
        a = indices[3 * t + c]
        b = indices[3 * t + (c + 1) % 3]
        la = labels[a]
        lb = labels[b]
        if la != lb:
            m = wp.min(la, lb)
            if la > m:
                old = wp.atomic_min(labels, a, m)
                if m < old:
                    wp.atomic_max(changed, 0, 1)
            if lb > m:
                old = wp.atomic_min(labels, b, m)
                if m < old:
                    wp.atomic_max(changed, 0, 1)


@wp.kernel(enable_backward=False)
def _hook(indices: wp.array[wp.int32], labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
    # Hook the endpoint with the larger label under the smaller (forest, parents).
    t = wp.tid()
    for c in range(3):
        a = indices[3 * t + c]
        b = indices[3 * t + (c + 1) % 3]
        la = labels[a]
        lb = labels[b]
        if la != lb:
            lo = wp.min(la, lb)
            hi = wp.max(la, lb)
            old = wp.atomic_min(labels, hi, lo)
            if lo < old:
                wp.atomic_max(changed, 0, 1)


@wp.kernel(enable_backward=False)
def _jump1(labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
    # Single grandparent pointer jump.
    v = wp.tid()
    p = labels[v]
    gp = labels[p]
    if gp != p:
        labels[v] = gp
        wp.atomic_max(changed, 0, 1)


@wp.kernel(enable_backward=False)
def _compress_full(labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
    # Full path compression: point directly at the root.
    v = wp.tid()
    r = labels[v]
    while labels[r] != r:
        r = labels[r]
    if r != labels[v]:
        labels[v] = r
        wp.atomic_max(changed, 0, 1)


@wp.kernel(enable_backward=False)
def _uf_step(indices: wp.array[wp.int32], labels: wp.array[wp.int32], changed: wp.array[wp.int32]):
    # Parallel union-find: find both roots (path halving) and hook via atomic-min,
    # retrying on concurrent updates until the two roots coincide.
    t = wp.tid()
    for c in range(3):
        a = indices[3 * t + c]
        b = indices[3 * t + (c + 1) % 3]
        if a == b:
            continue
        ra = _find(labels, a)
        rb = _find(labels, b)
        done = int(0)
        while ra != rb and done == 0:
            hi = wp.max(ra, rb)
            lo = wp.min(ra, rb)
            old = wp.atomic_min(labels, hi, lo)
            if old == hi:
                wp.atomic_max(changed, 0, 1)
                done = 1
            else:
                ra = _find(labels, a)
                rb = _find(labels, b)


# ---------------------------------------------------------------------------
# Drivers -- each returns flattened root labels (np array) and round count
# ---------------------------------------------------------------------------


def _make_labels(num_vertices, device):
    labels = wp.empty(num_vertices, dtype=wp.int32, device=device)
    wp.launch(_init, dim=num_vertices, inputs=[labels], device=device)
    return labels


def _finish(labels, num_vertices, device):
    out = wp.empty(num_vertices, dtype=wp.int32, device=device)
    wp.launch(_flatten, dim=num_vertices, inputs=[labels, out], device=device)
    return out


def _loop(step_launchers, changed, batch, cap):
    """Run rounds (each = all step_launchers) until `changed` stays 0 over a batch."""
    rounds = 0
    while rounds < cap:
        changed.zero_()
        for _ in range(batch):
            for launch in step_launchers:
                launch()
            rounds += 1
        if int(changed.numpy()[0]) == 0:
            break
    return rounds


def run_lp(indices, num_vertices, num_triangles, device, batch=1):
    labels = _make_labels(num_vertices, device)
    rounds = 0
    if num_triangles > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        rounds = _loop(
            [lambda: wp.launch(_lp_step, dim=num_triangles, inputs=[indices, labels, changed], device=device)],
            changed,
            batch,
            _CAP,
        )
    return _finish(labels, num_vertices, device), rounds


def run_pj1(indices, num_vertices, num_triangles, device, batch=1):
    labels = _make_labels(num_vertices, device)
    rounds = 0
    if num_triangles > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        rounds = _loop(
            [
                lambda: wp.launch(_hook, dim=num_triangles, inputs=[indices, labels, changed], device=device),
                lambda: wp.launch(_jump1, dim=num_vertices, inputs=[labels, changed], device=device),
            ],
            changed,
            batch,
            _CAP,
        )
    return _finish(labels, num_vertices, device), rounds


def run_hookfull(indices, num_vertices, num_triangles, device, batch=1):
    labels = _make_labels(num_vertices, device)
    rounds = 0
    if num_triangles > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        rounds = _loop(
            [
                lambda: wp.launch(_hook, dim=num_triangles, inputs=[indices, labels, changed], device=device),
                lambda: wp.launch(_compress_full, dim=num_vertices, inputs=[labels, changed], device=device),
            ],
            changed,
            batch,
            _CAP,
        )
    return _finish(labels, num_vertices, device), rounds


def run_uf(indices, num_vertices, num_triangles, device, batch=1):
    labels = _make_labels(num_vertices, device)
    rounds = 0
    if num_triangles > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        rounds = _loop(
            [lambda: wp.launch(_uf_step, dim=num_triangles, inputs=[indices, labels, changed], device=device)],
            changed,
            batch,
            _CAP,
        )
    return _finish(labels, num_vertices, device), rounds


def run_fastsv(indices, num_vertices, num_triangles, device, batch=1):
    # Two hooks per shortcut (FastSV-style) to cut round count.
    labels = _make_labels(num_vertices, device)
    rounds = 0
    if num_triangles > 0:
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        rounds = _loop(
            [
                lambda: wp.launch(_hook, dim=num_triangles, inputs=[indices, labels, changed], device=device),
                lambda: wp.launch(_jump1, dim=num_vertices, inputs=[labels, changed], device=device),
                lambda: wp.launch(_hook, dim=num_triangles, inputs=[indices, labels, changed], device=device),
                lambda: wp.launch(_jump1, dim=num_vertices, inputs=[labels, changed], device=device),
            ],
            changed,
            batch,
            _CAP,
        )
    return _finish(labels, num_vertices, device), rounds


VARIANTS = {
    "LP": run_lp,
    "PJ1": run_pj1,
    "hook+full": run_hookfull,
    "UF": run_uf,
    "FastSV": run_fastsv,
}


# ---------------------------------------------------------------------------
# Serial reference + partition equality
# ---------------------------------------------------------------------------


def reference(tris, num_points):
    parent = list(range(num_points))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in tris:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i, j), (j, k), (k, i)):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(v) for v in range(num_points)], dtype=np.int64)


def same_partition(a, b):
    # Two root-label arrays induce the same partition iff the maps are consistent.
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return len(set(zip(a.tolist(), b.tolist(), strict=True))) == len(set(a.tolist())) == len(set(b.tolist()))


# ---------------------------------------------------------------------------
# Mesh generators
# ---------------------------------------------------------------------------


def grid(n):
    def vid(i, j):
        return i * (n + 1) + j

    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    bl, br, tr, tl = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
    tris = np.stack([bl, br, tr, bl, tr, tl], axis=-1).reshape(-1).astype(np.int32)
    return tris, (n + 1) * (n + 1)


def strip(m):
    i = np.arange(m)
    return np.stack([i, i + 1, i + 2], axis=-1).reshape(-1).astype(np.int32), m + 2


def many_components(k):
    base = np.arange(k) * 3
    return np.stack([base, base + 1, base + 2], axis=-1).reshape(-1).astype(np.int32), k * 3


def giant_plus_small(n, k):
    g, gv = grid(n)
    s, sv = many_components(k)
    return np.concatenate([g, s + gv]).astype(np.int32), gv + sv


def hub_fan(m):
    # One high-degree hub vertex 0 shared by m triangles around a rim.
    i = np.arange(m)
    return np.stack([np.zeros(m, np.int64), 1 + i, 1 + (i + 1) % m], axis=-1).reshape(-1).astype(np.int32), m + 1


def random_soup(rng, num_points, m):
    t = rng.integers(0, num_points, size=(m, 3))
    return t.reshape(-1).astype(np.int32), num_points


# ---------------------------------------------------------------------------
# Correctness cross-check (the regression net for every variant)
# ---------------------------------------------------------------------------


def correctness(devices):
    rng = np.random.default_rng(12345)
    n_fail = 0
    n_cases = 0
    for _ in range(150):
        num_points = int(rng.integers(2, 40))
        m = int(rng.integers(0, 60))
        tris = rng.integers(0, num_points, size=(m, 3)).astype(np.int32)
        ref = reference(tris, num_points)
        for device in devices:
            idx = wp.array(tris.reshape(-1), dtype=wp.int32, device=device)
            for name, fn in VARIANTS.items():
                out, _ = fn(idx, num_points, m, device)
                n_cases += 1
                if not same_partition(out.numpy(), ref):
                    n_fail += 1
                    print(f"  FAIL {name} on {device}: V={num_points} F={m}")
    print(f"correctness: {n_cases - n_fail}/{n_cases} variant-cases passed across {devices}")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _time(fn, device, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        s.append(time.perf_counter() - t0)
    return statistics.median(s) * 1e3


def benchmark(device, batch):
    device = wp.get_device(device)
    meshes = [
        ("grid 224", *grid(224)),
        ("grid 707", *grid(707)),
        ("strip 100k", *strip(100_000)),
        ("strip 1M", *strip(1_000_000)),
        ("many 200k", *many_components(200_000)),
        ("giant+small", *giant_plus_small(500, 50_000)),
        ("hub 20k", *hub_fan(20_000)),
    ]
    print(f"\nDevice: {device}  batch={batch}  (median ms | rounds)\n")
    header = f"{'mesh':14s} {'F':>9s} {'V':>9s} " + " ".join(f"{n:>16s}" for n in VARIANTS)
    print(header)
    for name, tris, v in meshes:
        m = len(tris) // 3
        idx = wp.array(tris, dtype=wp.int32, device=device)
        cells = []
        for fn in VARIANTS.values():
            _, rounds = fn(idx, v, m, device, batch=batch)  # warm + get rounds
            capped = "+" if rounds >= _CAP else " "
            ms = _time(lambda fn=fn, idx=idx, v=v, m=m: fn(idx, v, m, device, batch=batch), device)
            cells.append(f"{ms:8.2f}/{rounds:<5d}{capped}")
        print(f"{name:14s} {m:>9d} {v:>9d} " + " ".join(f"{c:>16s}" for c in cells))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_known_args()[0]

    wp.init()
    devices = [str(d) for d in wp.get_devices()]
    if not args.skip_correctness:
        ok = correctness(devices)
        print("ALL CORRECT" if ok else "CORRECTNESS FAILURES PRESENT")
    targets = [args.device] if args.device else devices
    for d in targets:
        benchmark(d, args.batch)
