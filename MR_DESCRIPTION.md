<!--
TEMPORARY FILE -- delete this in one commit before sending the MR.

Suggested MR title:
    Add sparse voxel grid flood fill to warp.geometry

Target: NVIDIA/warp `main`. No related GitHub issue (prompted by a follow-up
request to mirror libigl's igl::sparse_voxel_grid). Formatted for
.gitlab/merge_request_templates/Default.md.

Note: this branch builds on the sparse marching cubes work (GH-1803); it uses
warp.geometry.sparse_marching_cubes_from_cells in the compare benchmark. Rebase
onto that once it lands.
-->

## Description

`warp.geometry.sparse_marching_cubes` finds near-surface cells with a *Lipschitz
octree* -- ideal when the field is a signed distance function. But many workflows
instead have a **seed cell known to touch the surface** and want to flood-fill
outward through the cells that straddle it, with no Lipschitz assumption. That is
exactly libigl's `igl::sparse_voxel_grid`: start at a seed cell, test its eight
corners for a sign change, and expand through the 26-neighborhood of intersecting
cells. It is common in vision / generative-AI pipelines that already have a marked
band of voxels near an object.

This MR adds **`warp.geometry.sparse_voxel_grid`**, a pure-Warp GPU re-creation of
that algorithm. It returns the same data libigl does -- `CV` (unique corner
positions), `CS` (field values at `CV`), and `CI` (per-voxel 8 corner indices) --
and runs entirely on the GPU using only sparse `O(M)` storage, never a dense
`n^3` grid.

Design decisions, the O(M) architecture, and the one deviation from the original
spec (thread-local stacks + global spill queues instead of block-shared tile
stacks, because Warp exposes no clean block-shared dynamic stack primitive) are
documented in `design/sparse-voxel-grid.md`.

## Changes

- **`warp/_src/sparse_voxel_grid.py`** (new): the implementation.
  - A concurrent open-addressing hash set keyed by 21-bit-per-axis packed `uint64`
    cell coordinates, claimed with `wp.atomic_cas` (EMPTY=0 sentinel, key = packed+1).
  - Production traversal `expand_multistep`: one thread per frontier cell does up
    to `batch_steps` local expansions via a per-thread stack in a global scratch
    slice, spilling overflow to a global queue; the host reads one spill count per
    coarse round (not one per BFS level).
  - Order-independent vertex construction (`radix_sort_pairs` + scan + scatter over
    int64 corner keys, libigl corner order).
  - Hash / surface / spill / coordinate-range exhaustion raise
    `SparseVoxelGridError`; output is never silently truncated.
- **`warp/geometry.py`, `warp/__init__.py`**: export `sparse_voxel_grid` and
  `SparseVoxelGridError`; register the module source.
- **`warp/tests/geometry/test_sparse_voxel_grid.py`** (new, CPU + CUDA): 12 tests.
- **`warp/examples/benchmarks/benchmark_sparse_voxel_grid.py`** (new): two modes.
- **`docs/api_reference/warp_geometry.rst`**: API entries.
- **`design/sparse-voxel-grid.md`** (new) and a `changelog/` fragment.

## Checklist

- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] I added a changelog fragment if this change affects users.

## Validation summary

Ran the full test module (24 cases across CPU + CUDA on an L40); all pass.

- **Correctness vs. a sparse CPU oracle** (`test_svg_matches_oracle`) -- a float32
  26-neighbor DFS reproducing libigl's sign rule, for sphere, ellipsoid, torus,
  and a bumpy sphere. Compares the exact active-cell set and validates
  `CV`/`CI`/`CS` (corner positions, de-duplication, `CS == field(CV)`, index range,
  full vertex coverage).
- **Production == wavefront reference** (`test_svg_matches_wavefront_reference`)
  and the multi-step path uses strictly fewer coarse rounds (e.g. 30 vs 157 at
  eps=1/64).
- **Determinism** -- the active-cell *set* is identical across repeated runs
  despite nondeterministic traversal order.
- **Threshold** (`test_svg_threshold`) -- a non-zero isovalue extracts that level
  set and `CS` stays the raw field value (matching libigl).
- **Edge cases** -- invalid (non-straddling) seed -> empty; all-zero corners
  inactive (exact-zero handling); negative coordinates; disconnected components
  (only the seed's component); argument validation; and clean
  `SparseVoxelGridError` on hash, surface, spill, and coordinate-range overflow.
- **Sparse scaling** (`test_svg_sparse_scaling`) -- with a fixed physical sphere
  and `eps ~ 1/n`, the active cell count grows ~4x per resolution doubling
  (surface area, `O(n^2)`), not 8x. Buffers are sized from the cell-count hint,
  never from an `n^3` ambient extent.

I also ran an independent adversarial code review of the traversal atomics,
overflow-flag handling, integer packing, termination, and vertex de-duplication;
it found no blockers. Three minor items surfaced and were fixed: a defensive
coordinate-overflow check in the vertex builder, the missing `threshold` test
above, and a note documenting why the float32 oracle comparison is robust (test
parameters keep all corners away from the isovalue).

Benchmark (`benchmark_sparse_voxel_grid.py --mode compare`) feeds the discovered
cells into `sparse_marching_cubes_from_cells` and compares end-to-end against the
dense `IsoSurfaceMarchingCubes` on the surface's bounding-box grid; both paths stay
on device and produce the same mesh (triangle counts checked every row). On an L40
with a non-SDF implicit (`|p|^2 - 1`), the sparse path overtakes dense around a
`385^3` grid, is ~5x faster by `769^3`, and keeps extracting (an ~89 M triangle
mesh) after the dense grid is infeasible -- dense cost ~O(res^3) vs sparse
~O(res^2).

## New feature / enhancement

```python
import warp as wp
import warp.geometry as wg

@wp.func
def sphere(p: wp.vec3) -> float:
    return wp.length(p) - 1.0

# Flood-fill from a seed cell that touches the surface; CV/CS/CI describe the
# occupied voxels. No dense grid, no Lipschitz assumption -- just sign changes.
CV, CS, CI = wg.sparse_voxel_grid(
    p0=(1.0, 0.0, 0.0),   # world center of cell (0,0,0), on the surface
    scalar_func=sphere,
    eps=1.0 / 128.0,      # grid spacing
    expected_number_of_cubes=200_000,
    device="cuda:0",
)
```
