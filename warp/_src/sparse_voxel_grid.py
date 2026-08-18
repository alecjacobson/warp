# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse voxel grid discovery around the level set of an implicit function.

This is a pure-Warp analogue of libigl's ``igl::sparse_voxel_grid``. Starting
from a seed cell, it flood-fills the connected set of grid cells that straddle
the zero level set of a scalar function, using only sparse (``O(M)``) storage --
never a dense ``n^3`` ambient grid -- and produces the same data libigl does:

* ``CV`` -- unique voxel-corner positions,
* ``CS`` -- the scalar field sampled at ``CV``,
* ``CI`` -- the 8 corner indices of each intersecting voxel (libigl corner order).

The whole pipeline runs on the GPU. Traversal claims cells in a concurrent
open-addressing hash set (via ``wp.atomic_cas``) and expands them with a
multi-step block of work per launch, so the number of host round-trips is far
below the flood-fill's graph diameter. See ``design/sparse-voxel-grid.md``.
"""

from __future__ import annotations

import numpy as np

import warp as wp
from warp._src import utils as _wp_utils

# =============================================================================
# Constants
# =============================================================================

# Cell/vertex integer coordinates are packed into a uint64 hash key. We reserve
# 21 bits per axis and bias signed coordinates into an unsigned range, so a
# valid coordinate is in ``[-COORD_BIAS, COORD_BIAS - 1]``. Key 0 is reserved to
# mean "empty slot", so the packed value is stored as ``packed + 1``.
_COORD_BITS: int = 21
_COORD_BIAS: int = 1 << (_COORD_BITS - 1)  # 2**20
_COORD_LIMIT: int = 1 << _COORD_BITS  # 2**21
COORD_MIN: int = -_COORD_BIAS
COORD_MAX: int = _COORD_BIAS - 1

# Maximum linear-probe distance before declaring the hash table overloaded.
_MAX_PROBES: int = 64

_EMPTY = wp.constant(wp.uint64(0))
_MAX_PROBES_C = wp.constant(_MAX_PROBES)
_COORD_BIAS_C = wp.constant(_COORD_BIAS)
_COORD_LIMIT_C = wp.constant(_COORD_LIMIT)

# fmt: off
# The 8 cube corners in libigl's ``sparse_voxel_grid`` order (used for CI and for
# the vertex-lattice keys). Corner ``k`` of cell ``(i, j, k)`` is at lattice node
# ``(i, j, k) + _CORNER_OFFSETS[k]``.
_CORNER_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0),
    (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 0, 0),
)
# fmt: on

# The 26-neighborhood offsets (all of {-1,0,1}^3 except the origin).
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1) if (di, dj, dk) != (0, 0, 0)
)

# State array layout (a single int32 array, read back once per coarse round).
_ST_SPILL_COUNT = 0  # number of cells written to the output spill queue this round
_ST_HASH_OVERFLOW = 1
_ST_SURFACE_OVERFLOW = 2
_ST_SPILL_OVERFLOW = 3
_ST_COORD_OVERFLOW = 4
_STATE_SIZE = 5

# Default tuning constants (see the design doc; treat as tunable, not API).
_DEFAULT_BATCH_STEPS: int = 16
_DEFAULT_STACK_CAP: int = 64


# =============================================================================
# Device-cached neighbor table
# =============================================================================

_neighbor_table_cache: dict[str, wp.array] = {}


def _get_neighbor_table(device) -> wp.array:
    key = str(device)
    if key not in _neighbor_table_cache:
        _neighbor_table_cache[key] = wp.array(
            np.array(_NEIGHBOR_OFFSETS, dtype=np.int32), dtype=wp.vec3i, device=device
        )
    return _neighbor_table_cache[key]


# =============================================================================
# Scalar-function-independent device helpers
# =============================================================================


@wp.func
def _pack_cell(c: wp.vec3i, state: wp.array(dtype=wp.int32)) -> wp.uint64:
    """Pack an integer coordinate into a nonzero uint64 key (0 = coord overflow)."""
    ux = c[0] + _COORD_BIAS_C
    uy = c[1] + _COORD_BIAS_C
    uz = c[2] + _COORD_BIAS_C
    if ux < 0 or ux >= _COORD_LIMIT_C or uy < 0 or uy >= _COORD_LIMIT_C or uz < 0 or uz >= _COORD_LIMIT_C:
        wp.atomic_add(state, _ST_COORD_OVERFLOW, 1)
        return wp.uint64(0)
    packed = wp.uint64(ux) | (wp.uint64(uy) << wp.uint64(21)) | (wp.uint64(uz) << wp.uint64(42))
    return packed + wp.uint64(1)


@wp.func
def _mix64(key: wp.uint64) -> wp.uint64:
    """SplitMix64 finalizer -- a good integer hash for the packed keys."""
    x = key
    x = (x ^ (x >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    x = x ^ (x >> wp.uint64(31))
    return x


@wp.func
def _claim(
    table: wp.array(dtype=wp.uint64), key: wp.uint64, mask: wp.int32, state: wp.array(dtype=wp.int32)
) -> wp.int32:
    """Atomically claim ``key`` in the open-addressing set.

    Returns 1 if newly claimed, 0 if already present, -1 on probe overflow.
    """
    slot = wp.int32(_mix64(key) & wp.uint64(mask))
    for _probe in range(_MAX_PROBES_C):
        old = wp.atomic_cas(table, slot, _EMPTY, key)
        if old == _EMPTY:
            return 1
        if old == key:
            return 0
        slot = (slot + 1) & mask
    wp.atomic_add(state, _ST_HASH_OVERFLOW, 1)
    return -1


@wp.kernel(enable_backward=False)
def _seed_kernel(
    table: wp.array(dtype=wp.uint64),
    mask: wp.int32,
    seed: wp.vec3i,
    spill: wp.array(dtype=wp.vec3i),
    state: wp.array(dtype=wp.int32),
):
    """Claim the seed cell and place it in the first spill queue."""
    _claim(table, _pack_cell(seed, state), mask, state)
    spill[0] = seed


@wp.kernel(enable_backward=False)
def _decode_vertices_kernel(
    unique_keys: wp.array(dtype=wp.uint64),
    p0: wp.vec3,
    eps: wp.float32,
    positions: wp.array(dtype=wp.vec3),
):
    """Recover world-space corner positions from packed vertex-lattice keys."""
    tid = wp.tid()
    packed = unique_keys[tid] - wp.uint64(1)
    mask21 = wp.uint64((1 << 21) - 1)
    ux = wp.int32(packed & mask21) - _COORD_BIAS_C
    uy = wp.int32((packed >> wp.uint64(21)) & mask21) - _COORD_BIAS_C
    uz = wp.int32((packed >> wp.uint64(42)) & mask21) - _COORD_BIAS_C
    positions[tid] = p0 + eps * wp.vec3(wp.float32(ux) - 0.5, wp.float32(uy) - 0.5, wp.float32(uz) - 0.5)


@wp.kernel(enable_backward=False)
def _corner_keys_kernel(
    cells: wp.array(dtype=wp.vec3i),
    state: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.uint64),
    perm: wp.array(dtype=wp.int32),
):
    """Emit the 8 corner (vertex-lattice) keys of each cell, in libigl order."""
    m = wp.tid()
    c = cells[m]
    for k in range(8):
        u = wp.vec3i(
            c[0] + wp.static(_CORNER_OFFSETS[k][0]),
            c[1] + wp.static(_CORNER_OFFSETS[k][1]),
            c[2] + wp.static(_CORNER_OFFSETS[k][2]),
        )
        keys[m * 8 + k] = _pack_cell(u, state)
        perm[m * 8 + k] = m * 8 + k


@wp.kernel(enable_backward=False)
def _mark_first_kernel(sorted_keys: wp.array(dtype=wp.uint64), is_first: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i == 0:
        is_first[i] = 1
    else:
        is_first[i] = wp.where(sorted_keys[i] != sorted_keys[i - 1], 1, 0)


@wp.kernel(enable_backward=False)
def _scatter_unique_kernel(
    sorted_keys: wp.array(dtype=wp.uint64),
    sorted_perm: wp.array(dtype=wp.int32),
    unique_scan: wp.array(dtype=wp.int32),
    is_first: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    unique_keys: wp.array(dtype=wp.uint64),
):
    i = wp.tid()
    uid = unique_scan[i] - 1
    inverse[sorted_perm[i]] = uid
    if is_first[i] == 1:
        unique_keys[uid] = sorted_keys[i]


@wp.kernel(enable_backward=False)
def _fill_iota_kernel(out: wp.array(dtype=wp.int32)):
    out[wp.tid()] = wp.tid()


# =============================================================================
# Scalar-function-specialized kernels (traversal + field sampling)
# =============================================================================

_kernel_cache: dict[wp.Function, dict] = {}


def _get_kernels(scalar_func: wp.Function) -> dict:
    """Build (and cache) the traversal/sampling kernels specialized to ``scalar_func``."""
    if scalar_func in _kernel_cache:
        return _kernel_cache[scalar_func]

    module = wp.Module(f"warp_sparse_voxel_grid_{scalar_func.key}_{id(scalar_func)}")

    @wp.func(module=module)
    def cell_active(p0: wp.vec3, eps: wp.float32, iso: wp.float32, c: wp.vec3i) -> bool:
        # A cell is active iff its 8 corner samples do not all share the same
        # sign (matching libigl's ``sgn``, where an exact zero is its own sign).
        s0 = wp.int32(0)
        active = False
        for k in range(8):
            pos = p0 + eps * wp.vec3(
                wp.float32(c[0]) + wp.static(float(_CORNER_OFFSETS[k][0])) - 0.5,
                wp.float32(c[1]) + wp.static(float(_CORNER_OFFSETS[k][1])) - 0.5,
                wp.float32(c[2]) + wp.static(float(_CORNER_OFFSETS[k][2])) - 0.5,
            )
            v = wp.float32(scalar_func(pos)) - iso
            sk = wp.where(v > 0.0, 1, 0) - wp.where(v < 0.0, 1, 0)
            if k == 0:
                s0 = sk
            elif sk != s0:
                active = True
        return active

    @wp.func(module=module)
    def spill_push(
        nb: wp.vec3i, spill_out: wp.array(dtype=wp.vec3i), spill_cap: wp.int32, state: wp.array(dtype=wp.int32)
    ):
        si = wp.atomic_add(state, _ST_SPILL_COUNT, 1)
        if si < spill_cap:
            spill_out[si] = nb
        else:
            wp.atomic_add(state, _ST_SPILL_OVERFLOW, 1)

    @wp.kernel(module=module, enable_backward=False)
    def expand_multistep(
        spill_in: wp.array(dtype=wp.vec3i),
        n_in: wp.int32,
        p0: wp.vec3,
        eps: wp.float32,
        iso: wp.float32,
        table: wp.array(dtype=wp.uint64),
        mask: wp.int32,
        neighbors: wp.array(dtype=wp.vec3i),
        local_stack: wp.array(dtype=wp.vec3i),
        stack_cap: wp.int32,
        batch_steps: wp.int32,
        surface: wp.array(dtype=wp.vec3i),
        surface_count: wp.array(dtype=wp.int32),
        surface_cap: wp.int32,
        spill_out: wp.array(dtype=wp.vec3i),
        spill_cap: wp.int32,
        state: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        if tid >= n_in:
            return

        base = tid * stack_cap
        top = wp.int32(1)
        local_stack[base] = spill_in[tid]  # seed this thread's local flood fill
        steps = wp.int32(0)

        while top > 0 and steps < batch_steps:
            top -= 1
            c = local_stack[base + top]
            steps += 1

            if not cell_active(p0, eps, iso, c):
                continue

            idx = wp.atomic_add(surface_count, 0, 1)
            if idx < surface_cap:
                surface[idx] = c
            else:
                wp.atomic_add(state, _ST_SURFACE_OVERFLOW, 1)

            for n in range(26):
                nb = c + neighbors[n]
                key = _pack_cell(nb, state)
                if key == wp.uint64(0):
                    continue
                if _claim(table, key, mask, state) == 1:
                    if top < stack_cap:
                        local_stack[base + top] = nb
                        top += 1
                    else:
                        spill_push(nb, spill_out, spill_cap, state)

        # Drain claimed-but-unevaluated cells to the global queue for next round.
        while top > 0:
            top -= 1
            spill_push(local_stack[base + top], spill_out, spill_cap, state)

    @wp.kernel(module=module, enable_backward=False)
    def expand_wavefront(
        frontier_in: wp.array(dtype=wp.vec3i),
        n_in: wp.int32,
        p0: wp.vec3,
        eps: wp.float32,
        iso: wp.float32,
        table: wp.array(dtype=wp.uint64),
        mask: wp.int32,
        neighbors: wp.array(dtype=wp.vec3i),
        surface: wp.array(dtype=wp.vec3i),
        surface_count: wp.array(dtype=wp.int32),
        surface_cap: wp.int32,
        frontier_out: wp.array(dtype=wp.vec3i),
        spill_cap: wp.int32,
        state: wp.array(dtype=wp.int32),
    ):
        # One BFS wave per launch: evaluate this cell, then spill its newly-claimed
        # neighbors for the next wave. Test/debug reference only (see the module doc).
        tid = wp.tid()
        if tid >= n_in:
            return
        c = frontier_in[tid]
        if not cell_active(p0, eps, iso, c):
            return
        idx = wp.atomic_add(surface_count, 0, 1)
        if idx < surface_cap:
            surface[idx] = c
        else:
            wp.atomic_add(state, _ST_SURFACE_OVERFLOW, 1)
        for n in range(26):
            nb = c + neighbors[n]
            key = _pack_cell(nb, state)
            if key == wp.uint64(0):
                continue
            if _claim(table, key, mask, state) == 1:
                spill_push(nb, frontier_out, spill_cap, state)

    @wp.kernel(module=module, enable_backward=False)
    def eval_scalar(positions: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)):
        i = wp.tid()
        values[i] = wp.float32(scalar_func(positions[i]))

    kernels = {"expand": expand_multistep, "wavefront": expand_wavefront, "eval": eval_scalar, "module": module}
    _kernel_cache[scalar_func] = kernels
    return kernels


# =============================================================================
# Host driver
# =============================================================================


def _next_pow2(n: int) -> int:
    return 1 << max(1, (int(n) - 1).bit_length())


class SparseVoxelGridError(RuntimeError):
    """Raised when a sparse voxel grid capacity or coordinate limit is exceeded."""


def _check_state(state_np, surface_cap, spill_cap, hash_cap):
    if state_np[_ST_COORD_OVERFLOW] > 0:
        raise SparseVoxelGridError(
            f"Cell coordinate exceeded the packable range [{COORD_MIN}, {COORD_MAX}]. "
            "Shift the seed/origin or reduce the traversal extent."
        )
    if state_np[_ST_HASH_OVERFLOW] > 0:
        raise SparseVoxelGridError(
            f"Visited hash table overflowed (capacity {hash_cap}). Increase expected_number_of_cubes."
        )
    if state_np[_ST_SURFACE_OVERFLOW] > 0:
        raise SparseVoxelGridError(
            f"Surface-cell capacity ({surface_cap}) exceeded. Increase expected_number_of_cubes."
        )
    if state_np[_ST_SPILL_OVERFLOW] > 0:
        raise SparseVoxelGridError(f"Spill-queue capacity ({spill_cap}) exceeded. Increase expected_number_of_cubes.")


def _traverse(scalar_func, p0, eps, iso, seed, surface_cap, spill_cap, hash_cap, batch_steps, stack_cap, device):
    """Flood-fill the connected active cells from ``seed``. Returns (surface, M, stats)."""
    kernels = _get_kernels(scalar_func)
    mask = hash_cap - 1
    neighbors = _get_neighbor_table(device)

    table = wp.zeros(hash_cap, dtype=wp.uint64, device=device)
    surface = wp.empty(surface_cap, dtype=wp.vec3i, device=device)
    surface_count = wp.zeros(1, dtype=wp.int32, device=device)
    spill_a = wp.empty(spill_cap, dtype=wp.vec3i, device=device)
    spill_b = wp.empty(spill_cap, dtype=wp.vec3i, device=device)
    state = wp.zeros(_STATE_SIZE, dtype=wp.int32, device=device)

    wp.launch(_seed_kernel, dim=1, inputs=[table, mask, wp.vec3i(seed), spill_a, state], device=device)

    n_in = 1
    rounds = 0
    max_spill = 0
    while n_in > 0:
        state[_ST_SPILL_COUNT : _ST_SPILL_COUNT + 1].zero_()  # keep overflow flags cumulative
        local_stack = wp.empty(n_in * stack_cap, dtype=wp.vec3i, device=device)
        wp.launch(
            kernels["expand"],
            dim=n_in,
            inputs=[
                spill_a,
                n_in,
                wp.vec3(p0),
                wp.float32(eps),
                wp.float32(iso),
                table,
                mask,
                neighbors,
                local_stack,
                stack_cap,
                batch_steps,
                surface,
                surface_count,
                surface_cap,
                spill_b,
                spill_cap,
                state,
            ],
            device=device,
        )
        state_np = state.numpy()  # single sync point per coarse round
        _check_state(state_np, surface_cap, spill_cap, hash_cap)
        n_out = int(state_np[_ST_SPILL_COUNT])
        max_spill = max(max_spill, n_out)
        spill_a, spill_b = spill_b, spill_a
        n_in = n_out
        rounds += 1

    m = int(surface_count.numpy()[0])
    stats = {"surface_count": m, "spill_round_count": rounds, "max_spill_count": max_spill, "hash_capacity": hash_cap}
    return surface, m, stats


def _traverse_wavefront(scalar_func, p0, eps, iso, seed, surface_cap, spill_cap, hash_cap, device):
    """One-BFS-wave-per-launch traversal. Test/debug reference only -- not production.

    Uses the same sparse ``O(M)`` storage as :func:`_traverse` but reads the
    frontier count back once per BFS level, so it is simpler to reason about and
    useful for isolating bugs in the multi-step traversal. It must not be used for
    performance claims.
    """
    kernels = _get_kernels(scalar_func)
    mask = hash_cap - 1
    neighbors = _get_neighbor_table(device)

    table = wp.zeros(hash_cap, dtype=wp.uint64, device=device)
    surface = wp.empty(surface_cap, dtype=wp.vec3i, device=device)
    surface_count = wp.zeros(1, dtype=wp.int32, device=device)
    frontier_a = wp.empty(spill_cap, dtype=wp.vec3i, device=device)
    frontier_b = wp.empty(spill_cap, dtype=wp.vec3i, device=device)
    state = wp.zeros(_STATE_SIZE, dtype=wp.int32, device=device)

    wp.launch(_seed_kernel, dim=1, inputs=[table, mask, wp.vec3i(seed), frontier_a, state], device=device)

    n_in = 1
    rounds = 0
    while n_in > 0:
        state[_ST_SPILL_COUNT : _ST_SPILL_COUNT + 1].zero_()  # keep overflow flags cumulative
        wp.launch(
            kernels["wavefront"],
            dim=n_in,
            inputs=[
                frontier_a,
                n_in,
                wp.vec3(p0),
                wp.float32(eps),
                wp.float32(iso),
                table,
                mask,
                neighbors,
                surface,
                surface_count,
                surface_cap,
                frontier_b,
                spill_cap,
                state,
            ],
            device=device,
        )
        state_np = state.numpy()
        _check_state(state_np, surface_cap, spill_cap, hash_cap)
        n_in = int(state_np[_ST_SPILL_COUNT])
        frontier_a, frontier_b = frontier_b, frontier_a
        rounds += 1

    m = int(surface_count.numpy()[0])
    return surface, m, {"surface_count": m, "spill_round_count": rounds}


def _sparse_voxel_grid_wavefront_reference(
    p0, scalar_func, eps, expected_number_of_cubes, threshold=0.0, seed=(0, 0, 0), device=None
):
    """Test/debug-only wavefront traversal returning the active integer cell coordinates.

    Not part of the public API; exists solely to cross-check the production
    :func:`sparse_voxel_grid` traversal. Uses sparse storage only.
    """
    device = wp.get_device(device)
    expected = int(expected_number_of_cubes)
    surface, m, stats = _traverse_wavefront(
        scalar_func,
        wp.vec3(p0),
        float(eps),
        float(threshold),
        tuple(int(s) for s in seed),
        expected,
        _next_pow2(4 * expected),
        _next_pow2(8 * expected),
        device,
    )
    cells = surface[:m] if m > 0 else wp.empty(0, dtype=wp.vec3i, device=device)
    return cells, stats


# =============================================================================
# Vertex / index construction
# =============================================================================


def _build_vertices(scalar_func, surface, m, p0, eps, device):
    """Build (CV, CS, CI) from the ``m`` active cells in ``surface``."""
    cells = surface[:m]
    n_occ = 8 * m

    state = wp.zeros(_STATE_SIZE, dtype=wp.int32, device=device)
    keys = wp.empty(n_occ, dtype=wp.uint64, device=device)
    perm = wp.empty(n_occ, dtype=wp.int32, device=device)
    wp.launch(_corner_keys_kernel, dim=m, inputs=[cells, state], outputs=[keys, perm], device=device)

    # A cell in range can still have a ``+1`` corner just past the packable range.
    # The traversal already rejects such cells (an active boundary cell's ``+1``
    # neighbor overflows first), so this is defensive; guard it explicitly rather
    # than let a bad key (0) flow into the de-duplication.
    if int(state.numpy()[_ST_COORD_OVERFLOW]) > 0:
        raise SparseVoxelGridError(f"Cell corner exceeded the packable coordinate range [{COORD_MIN}, {COORD_MAX}].")

    # Sort (key, occurrence) by key. radix_sort_pairs needs 2*count storage.
    sort_keys = wp.empty(2 * n_occ, dtype=wp.uint64, device=device)
    sort_perm = wp.empty(2 * n_occ, dtype=wp.int32, device=device)
    wp.copy(sort_keys[:n_occ], keys)
    wp.copy(sort_perm[:n_occ], perm)
    _wp_utils.radix_sort_pairs(sort_keys, sort_perm, n_occ)
    sorted_keys = sort_keys[:n_occ]
    sorted_perm = sort_perm[:n_occ]

    is_first = wp.empty(n_occ, dtype=wp.int32, device=device)
    wp.launch(_mark_first_kernel, dim=n_occ, inputs=[sorted_keys, is_first], device=device)
    unique_scan = wp.empty(n_occ, dtype=wp.int32, device=device)
    _wp_utils.array_scan(is_first, unique_scan, inclusive=True)
    n_unique = int(unique_scan[-1:].numpy()[0])

    inverse = wp.empty(n_occ, dtype=wp.int32, device=device)
    unique_keys = wp.empty(n_unique, dtype=wp.uint64, device=device)
    wp.launch(
        _scatter_unique_kernel,
        dim=n_occ,
        inputs=[sorted_keys, sorted_perm, unique_scan, is_first],
        outputs=[inverse, unique_keys],
        device=device,
    )

    ci = inverse.reshape((m, 8))

    cv = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        _decode_vertices_kernel,
        dim=n_unique,
        inputs=[unique_keys, wp.vec3(p0), wp.float32(eps)],
        outputs=[cv],
        device=device,
    )

    cs = wp.empty(n_unique, dtype=wp.float32, device=device)
    kernels = _get_kernels(scalar_func)
    wp.launch(kernels["eval"], dim=n_unique, inputs=[cv], outputs=[cs], device=device)

    return cv, cs, ci, n_unique


# =============================================================================
# Public API
# =============================================================================


def sparse_voxel_grid(
    p0: wp.vec3 | tuple[float, float, float],
    scalar_func: wp.Function,
    eps: float,
    expected_number_of_cubes: int,
    threshold: float = 0.0,
    seed: tuple[int, int, int] = (0, 0, 0),
    batch_steps: int = _DEFAULT_BATCH_STEPS,
    stack_cap: int = _DEFAULT_STACK_CAP,
    surface_capacity: int | None = None,
    spill_capacity: int | None = None,
    visited_capacity: int | None = None,
    device: wp.DeviceLike = None,
    return_stats: bool = False,
    return_cells: bool = False,
):
    """Discover the sparse voxel grid straddling the level set of ``scalar_func``.

    Flood-fills, starting from ``seed``, the connected set of grid cells whose 8
    corner samples do not all share the same sign, using the full 26-neighborhood
    -- a pure-Warp analogue of ``igl::sparse_voxel_grid``. Only sparse ``O(M)``
    storage is used, where ``M`` is the number of intersecting cells; no dense
    ``n^3`` grid is ever allocated.

    Cell ``(i, j, k)`` is the cube centered at ``p0 + eps * (i, j, k)`` with
    corners offset by ``+/- eps/2``. If the ``seed`` cell does not straddle the
    surface, the result is empty. Only the component connected to ``seed`` is
    returned.

    Args:
        p0: World-space center of cell ``(0, 0, 0)``.
        scalar_func: A Warp ``@wp.func`` with signature ``(p: wp.vec3) -> float``
            giving the implicit field. It is evaluated inside the traversal
            kernels, which are specialized (and cached) per function.
        eps: The grid spacing (edge length of a cell).
        expected_number_of_cubes: A hint for the number of intersecting cells,
            used to size the sparse buffers (analogous to libigl's argument).
        threshold: The isovalue defining the surface (default ``0``).
        seed: Integer coordinate of the seed cell (default ``(0, 0, 0)``).
        batch_steps: Local expansion steps each thread performs per launch
            (tuning constant, not output-affecting).
        stack_cap: Per-thread local stack capacity (tuning constant).
        surface_capacity: Override for the active-cell buffer size. Defaults to
            ``expected_number_of_cubes``.
        spill_capacity: Override for the traversal frontier buffer size. Defaults
            to ``next_pow2(4 * expected_number_of_cubes)``.
        visited_capacity: Override for the visited hash-table size (rounded up to
            a power of two). Defaults to ``next_pow2(8 * expected_number_of_cubes)``.
        device: The Warp device to run on. Defaults to the current device.
        return_stats: If ``True``, also return a diagnostics dictionary.
        return_cells: If ``True``, also return the integer active-cell coordinates.

    Returns:
        ``(CV, CS, CI)`` where ``CV`` is a ``wp.array(dtype=wp.vec3)`` of unique
        corner positions, ``CS`` a ``wp.array(dtype=wp.float32)`` of field values
        at ``CV``, and ``CI`` a ``wp.array(dtype=wp.int32, ndim=2)`` of shape
        ``(M, 8)`` indexing ``CV`` in libigl corner order. Extra trailing values
        are appended for ``return_cells`` / ``return_stats``.

    Raises:
        SparseVoxelGridError: On hash/surface/spill capacity exhaustion or a cell
            coordinate outside the packable range ``[COORD_MIN, COORD_MAX]``.
        TypeError: If ``scalar_func`` is not a ``warp.Function``.
        ValueError: If ``eps`` or ``expected_number_of_cubes`` is not positive.
    """
    if not isinstance(scalar_func, wp.Function):
        raise TypeError(
            "`scalar_func` must be a warp.Function (@wp.func) with signature (p: wp.vec3) -> float, "
            f"but got {type(scalar_func)}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if expected_number_of_cubes <= 0:
        raise ValueError(f"expected_number_of_cubes must be positive, got {expected_number_of_cubes}.")

    device = wp.get_device(device)
    p0 = wp.vec3(p0)
    eps = float(eps)

    expected = int(expected_number_of_cubes)
    surface_cap = int(surface_capacity) if surface_capacity is not None else expected
    spill_cap = int(spill_capacity) if spill_capacity is not None else _next_pow2(4 * expected)
    hash_cap = _next_pow2(visited_capacity) if visited_capacity is not None else _next_pow2(8 * expected)

    surface, m, trav_stats = _traverse(
        scalar_func,
        p0,
        eps,
        float(threshold),
        tuple(int(s) for s in seed),
        surface_cap,
        spill_cap,
        hash_cap,
        int(batch_steps),
        int(stack_cap),
        device,
    )

    if m == 0:
        cv = wp.empty(0, dtype=wp.vec3, device=device)
        cs = wp.empty(0, dtype=wp.float32, device=device)
        ci = wp.empty((0, 8), dtype=wp.int32, device=device)
        n_unique = 0
    else:
        cv, cs, ci, n_unique = _build_vertices(scalar_func, surface, m, p0, eps, device)

    result = [cv, cs, ci]
    if return_cells:
        result.append(surface[:m] if m > 0 else wp.empty(0, dtype=wp.vec3i, device=device))
    if return_stats:
        stats = dict(trav_stats)
        stats.update(
            {
                "unique_corners": n_unique,
                "surface_capacity": surface_cap,
                "spill_capacity": spill_cap,
                "batch_steps": int(batch_steps),
                "stack_cap": int(stack_cap),
            }
        )
        result.append(stats)
    return tuple(result)
