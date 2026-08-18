# Sparse Voxel Grid

**Status**: Implemented

**Issue**: No tracking issue; prompted by a follow-up request to mirror
libigl's `igl::sparse_voxel_grid` (see Motivation).

## Motivation

`warp.geometry.sparse_marching_cubes` discovers near-surface cells with a
*Lipschitz octree*: it prunes top-down using a Lipschitz bound on the field.
That is ideal when the field is a signed distance function, but many workflows
instead have a **seed cell known to touch the surface** and want to *flood-fill*
outward through the cells that straddle it -- with no Lipschitz assumption. This
is exactly libigl's `igl::sparse_voxel_grid`, which starts at cell `(0,0,0)`,
tests its eight corners for a sign change, and expands through the 26-neighborhood
of intersecting cells. It is common in vision / generative-AI pipelines that
already have a marked band of voxels near an object.

This feature re-creates `igl::sparse_voxel_grid` in pure Warp, on the GPU,
producing the same data: `CV` (unique corner positions), `CS` (field values at
`CV`), and `CI` (the 8 corner indices per intersecting voxel).

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Flood-fill the connected active cells from a seed via 26-neighbor connectivity | Must | libigl semantics |
| R2  | Match libigl's cell-validity rule (corners not all the same `sgn`, exact zero is its own sign) | Must | Includes all-zero -> inactive |
| R3  | `O(M)` work and storage for `M` intersecting cells; never allocate or scan an `n^3` grid | Must | The hard asymptotic requirement |
| R4  | Production traversal performs multiple expansion steps per host round (not one round per BFS level) | Must | Reduce synchronizations |
| R5  | Prevent duplicate evaluation via atomic sparse claiming | Must | Concurrency correctness |
| R6  | Explicit, safe errors on hash / queue / coordinate / output capacity exhaustion | Must | Never truncate output |
| R7  | Build `CV`/`CS`/`CI` without traversal-order dependencies | Must | GPU order is nondeterministic |
| R8  | Evaluate the scalar field inside Warp kernels (`@wp.func`) | Must | No dense CPU/GPU grid eval |

**Non-goals**: A persistent global producer/consumer traversal kernel (a possible
later optimization). Reproducing libigl's traversal-time vertex sharing (we build
vertices in a separate order-independent pass). Meshing -- this returns voxels and
corner data, not triangles.

## Design

### Approach

Two phases, both sparse and on-device:

1. **Traversal** discovers the connected active cells. A concurrent
   open-addressing hash set (keyed by packed integer cell coordinates, claimed
   with `wp.atomic_cas`) guarantees each cell is evaluated once. Work flows
   through global *spill queues*: one host round consumes the current queue and
   produces the next. Each thread performs a **multi-step local flood fill**
   before spilling, so a round advances the frontier by many cells, not one BFS
   level.
2. **Vertex construction** turns the `M` active cells into `CV`/`CS`/`CI` with an
   order-independent sort-and-unique over integer corner keys.

The public entry point is `warp.geometry.sparse_voxel_grid(p0, scalar_func, eps,
expected_number_of_cubes, ...)`.

### Alternatives Considered

- **Block-shared cooperative tile stacks** (the originally-specified production
  architecture). Warp exposes structured tile ops (`tile_zeros`, `tile_scan`,
  shared storage) and `atomic_cas`, but not a clean block-shared *dynamic stack*
  with per-thread push/pop and a shared top counter. Rather than build a fragile
  shared-memory work-list, the implementation uses **per-thread local stacks in a
  global scratch slice** (`local_stack[tid*stack_cap : ...]`) plus a global spill
  queue. This achieves the same asymptotics and the same "multiple steps per
  launch" property (R4) with simpler, race-free invariants: a thread owns its
  stack slice exclusively, and the only atomics are the hash-set CAS and the
  append counters. A block-shared or persistent-kernel variant remains a possible
  later optimization. The benchmark measures the payoff: at `eps = 1/64` on a
  sphere the multi-step traversal uses ~31 coarse rounds versus ~157 for a
  one-wave-per-launch reference.

- **One-wave-per-launch (wavefront) as production.** Correct and sparse, but it
  synchronizes once per BFS level (`O(diameter)` host round-trips). It is kept
  only as a clearly-marked test/debug reference
  (`_sparse_voxel_grid_wavefront_reference`) for cross-checking.

- **A dense visited bitset / dense occupancy.** Prohibited by R3; it would be
  `O(n^3)`. The visited set is the sparse hash table instead.

- **`wp.HashGrid` for the visited set.** Built for spatial neighbor queries on
  float positions, not exact integer de-duplication with atomic claiming.

### Key Implementation Details

Module: `warp/_src/sparse_voxel_grid.py`; public re-exports in `warp/geometry.py`.

**Cell/vertex key packing.** Integer coordinates are biased and packed into a
`uint64`: 21 bits per axis, bias `2^20`, so a valid coordinate is in
`[-2^20, 2^20 - 1]` (exposed as `COORD_MIN`/`COORD_MAX`). Key `0` is reserved for
an empty slot, so the stored key is `packed + 1` (`uint64` avoids the `2^63`
edge). A coordinate outside the range sets a `coord_overflow` flag and aborts
cleanly rather than wrapping.

**Sparse visited set.** A power-of-two `uint64` table sized from
`expected_number_of_cubes` (default `next_pow2(8 * expected)`), never from `n`.
`_claim` mixes the key (SplitMix64 finalizer), then linear-probes with
`wp.atomic_cas(EMPTY -> key)`: a swap from `EMPTY` means *newly claimed*, hitting
the same key means *already seen*, and exceeding the probe limit sets a
`hash_overflow` flag.

**Multi-step traversal kernel** (`expand_multistep`, specialized per
`scalar_func`). One thread per input frontier cell seeds a private stack in the
global scratch slice, then repeats up to `batch_steps` times: pop a cell,
evaluate its 8 corner signs, and if active, append it to `surface_cells`
(atomic) and try to claim all 26 neighbors; newly-claimed neighbors go on the
local stack, or spill to the global `spill_out` queue if it is full. Remaining
local-stack entries drain to `spill_out` at the end. The host swaps the spill
queues and reads a single `spill_count` per round to decide termination.

**State and safety.** A 5-element `int32` state array carries the per-round spill
count and four cumulative overflow flags (hash / surface / spill / coordinate).
Only the spill count is reset each round, so overflow flags survive to the host
check, which raises `SparseVoxelGridError`. All capacity checks guard the write
*before* it happens, so output is never silently truncated.

**Vertex construction** (`_build_vertices`). For `M` active cells it emits `8*M`
corner occurrences with `uint64` vertex-lattice keys in libigl corner order,
sorts `(key, occurrence)` with `radix_sort_pairs`, marks run starts, prefix-scans
to assign one id per unique corner, scatters ids back to form `CI[M, 8]`, decodes
each unique key to `CV = p0 + eps*(u - 0.5)`, and evaluates `CS = scalar_func(CV)`
once per unique corner. This is fully order-independent (R7).

### Public API

```python
CV, CS, CI = warp.geometry.sparse_voxel_grid(
    p0,                       # world center of cell (0,0,0)
    scalar_func,              # @wp.func (p: wp.vec3) -> float
    eps,                      # grid spacing
    expected_number_of_cubes, # sizes the sparse buffers (not n^3)
    threshold=0.0, seed=(0, 0, 0), device=None,
    return_cells=False, return_stats=False,
    batch_steps=16, stack_cap=64,           # tuning
    surface_capacity=None, spill_capacity=None, visited_capacity=None,
)
```

`return_stats=True` adds a diagnostics dict (`surface_count`,
`spill_round_count`, `max_spill_count`, `unique_corners`, `hash_capacity`, ...).
Coordinate range, supported dtype (`float32`), CUDA backend, and capacity
behavior are documented on the function.

## Testing Strategy

`warp/tests/geometry/test_sparse_voxel_grid.py`, across `get_test_devices()`
(CPU + CUDA):

- **Cell-set + geometry vs. a sparse CPU oracle** (a float32 26-neighbor DFS
  reproducing libigl's sign rule) for sphere, ellipsoid, torus, and a bumpy
  ("wavy") sphere. Also validates `CV`/`CI` corner positions and `CS` values.
- **Production == wavefront reference**, and the multi-step path uses strictly
  fewer coarse rounds.
- **Determinism**: the active-cell *set* is identical across repeated runs
  despite nondeterministic traversal order.
- **Edge cases**: invalid (non-straddling) seed -> empty; all-zero corners
  inactive (exact-zero handling); negative coordinates; disconnected components
  (only the seed's component); and clean `SparseVoxelGridError` on hash, surface,
  spill, and coordinate-range overflow.
- **Sparse scaling**: with a fixed physical sphere and `eps ~ 1/n`, the active
  cell count grows ~4x per resolution doubling (surface area, `O(n^2)`), not 8x
  (`O(n^3)`). Buffers are sized from the cell-count hint, never from an ambient
  extent.

`warp/examples/benchmarks/benchmark_sparse_voxel_grid.py` has two modes.
`--mode profile` breaks down the phases (traversal / vertex build / end-to-end),
compares multi-step vs. wavefront round counts, and sweeps `batch_steps`.
`--mode compare` closes the loop against dense extraction: it feeds the
discovered cells into :func:`warp.geometry.sparse_marching_cubes_from_cells`
(reusing the sparse per-corner field values, so no re-evaluation) and compares
end-to-end against the dense :class:`warp.geometry.IsoSurfaceMarchingCubes` on
the surface's bounding-box grid. Both paths stay on device and produce the same
mesh (triangle counts are checked). It uses a NON-SDF implicit (the quadric
sphere ``|p|^2 - 1``) with a hard-coded seed, underscoring that
`sparse_voxel_grid` needs only a seed and sign changes -- no Lipschitz/distance
property. On an L40 (quadric sphere) the sparse path overtakes dense around a
``385^3`` grid, is ~5x faster by ``769^3``, and keeps extracting (an ~89 M
triangle mesh at ``3073^3``) after the dense grid becomes infeasible -- dense
cost ~O(res^3) versus sparse ~O(res^2). A thinner surface (e.g. a torus) has a
smaller dense-box constant, so its crossover lands at a higher resolution, but
the asymptotics are the same.
