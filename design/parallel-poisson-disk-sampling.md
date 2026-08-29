# Parallel Poisson-Disk Sampling on Surfaces

**Status**: In Progress

**Issue**: N/A (feature request)

## Motivation

Poisson-disk (blue-noise) point sets have no two samples closer than a radius
`r` and fill the domain as densely as that constraint allows. On triangle-mesh
surfaces they are the standard input for stippling, remeshing seeds, object
scattering, and Monte-Carlo integration, because their spectrum suppresses
low-frequency noise and aliasing far better than uniform random sampling.

This feature implements the GPU-parallel surface sampler of Bowers, Wang, Wei
and Maletz, *"Parallel Poisson Disk Sampling with Spectrum Analysis on
Surfaces"* (ACM SIGGRAPH Asia 2010), on top of the area-weighted
`warp.geometry.UniformSampler` already in this module. It also provides the
paper's companion measurement tool -- a differential-domain radial statistic
(pair-correlation function) -- so users can verify the blue-noise quality of a
point set directly on the surface.

## Requirements

| ID  | Requirement                                                                 | Priority | Notes |
| --- | --------------------------------------------------------------------------- | -------- | ----- |
| R1  | `poisson_disk_sample(points, faces, radius)` returns a conflict-free set    | Must     | No pair closer than `radius` |
| R2  | The set is *maximal*: no candidate can be added without a conflict          | Must     | Blue-noise coverage |
| R3  | Fully parallel on GPU; also runs on CPU                                     | Must     | No host-side per-sample loop |
| R4  | Reuse `UniformSampler` for area-weighted candidate generation               | Must     | Density independent of tessellation |
| R5  | `PoissonDiskSampler` class exposes faces, uv, and positions of the result   | Must     | Mirrors `UniformSampler` |
| R6  | Deterministic for a fixed seed                                              | Should   | Reproducible tests |
| R7  | `pair_correlation` measures blue-noise quality on the surface               | Should   | The paper's spectrum analysis |
| R8  | Performance scales to millions of candidates                                | Should   | Regression + perf tests |

**Non-goals**: Geodesic distances (we use the Euclidean approximation, valid
when `r` is small relative to surface curvature -- the paper's Euclidean
variant); exact maximal Poisson-disk sampling in the continuous limit (we
produce a maximal *subset of a finite candidate pool*, which converges to the
continuous result as the pool density grows); the full 2-D power-spectrum
periodogram (the pair-correlation function is the surface-appropriate analog).

## Design

### Approach

Three stages, all on device:

1. **Candidate generation.** Draw a dense pool of `M` candidate points with
   `UniformSampler` (area-weighted, so density is uniform per unit surface area
   regardless of tessellation). `M` defaults to `candidate_multiplier` times the
   theoretical maximal sample count `N_est = total_area / (sqrt(3)/2 * r^2)`
   (hexagonal packing of disks of radius `r/2`).

2. **Parallel conflict resolution over a single-entry spatial hash.** Following
   the paper, the grid cell edge is `mu = r / sqrt(3)`, so a cell's diagonal is
   `r` and a cell holds at most one accepted sample. Only the non-empty cells
   are stored, in a **spatial hash** keyed by the integer cell id
   (`hash(cell_id) % table_size`, linear probing, table sized to `2 x` the
   candidate count so the load factor stays below 1/2). Memory therefore scales
   with the sampled *surface*, not the 3D bounding volume -- the key reason a
   plain dense grid is unusable here.

   Cells are resolved in **27 phase groups** (cell coordinates modulo 3, in a
   seed-dependent random order). Two cells in one group are at least 3 cells
   apart, so their samples are always `> r` apart and every group is one fully
   parallel pass with no inter-cell coordination. Each pass is three kernels:
   a candidate marks itself *free* if the surrounding `5x5x5` block of the hash
   holds no accepted sample within `r`; the free candidates in a cell elect the
   highest-priority one (`atomic_max` then `atomic_min` index tie-break); the
   winner is written as that cell's sample. Because priorities are fixed
   (seeded), the result is deterministic (R6).

3. **Compaction.** A prefix sum (`warp.utils.array_scan`) over the accepted mask
   scatters the surviving faces, uv, and positions into tightly packed output
   arrays.

The paper's canonical inner loop is *dart throwing*: each cell tries up to `k`
random candidates in turn, accepting the first conflict-free one. Choosing the
highest-priority conflict-free candidate in a single pass is an equivalent
parallel formulation (it considers all of a cell's candidates at once) that maps
cleanly onto Warp atomics and needs no per-cell trial loop.

### Alternatives Considered

- **`wp.HashGrid` over the candidate cloud.** Warp's built-in hash grid stores
  *every* candidate (tens per cell) and returns per-cell point lists, so each
  neighbor query scans far more points than the paper's one-sample-per-cell
  structure. A maximal-independent-set solver over it also needs a host readback
  per round to detect convergence. It was prototyped and measured slower; more
  importantly it is the wrong granularity -- the paper's insight is precisely
  that the accepted grid has a single entry per cell. Rejected in favor of the
  custom spatial hash.
- **Dense 3D grid (one int per cell).** O(1) lookups, but memory is
  O(volume) = O((L/mu)^3); for a 2D surface embedded in 3D this explodes at
  small `r` (the grid is almost all empty). Rejected for surfaces; the hash
  keeps only the O(area/mu^2) non-empty cells.
- **Relaxation / Lloyd or sample-elimination (Yuksel 2015).** Produces excellent
  spectra but is iterative and does not target a hard radius directly. Out of
  scope; the sampler's result can be fed into such a relaxer later.

### Key Implementation Details

- **Cell id is 64-bit** (`(cz*gy + cy)*gx + cx`) so the id space can exceed
  `2^31` for fine grids on large meshes, even though per-axis counts fit in 32
  bits.
- **Hash insert** uses `wp.atomic_cas` on an empty slot to claim a cell exactly
  once; each candidate then caches its slot and phase so the phase passes avoid
  recomputing the probe.
- **Status array** (`int32` per candidate): `0 = active`, `1 = accepted`. A
  `free` flag per candidate is written by the first phase pass and read by the
  next two, so the `5x5x5` neighborhood scan runs once per candidate per phase.
- **Distance test** uses squared distance against `r^2` to avoid a `sqrt`.
- **Termination is fixed** at 27 phase passes -- no convergence loop or host
  readback, unlike an MIS formulation.
- **`pair_correlation(points, area, ...)`** builds a `wp.HashGrid` over the
  sample positions and atomically histograms every ordered neighbor pair's
  distance into radial bins up to `r_max`. Each bin is normalized by the count
  expected for a uniform Poisson process of the same density,
  `N * (N/area) * 2*pi*r*dr`, so `g(r) -> 1` at large `r`, `g(r) ~ 0` below the
  Poisson radius, with the characteristic blue-noise peak just past it.

### Public API

```python
import warp.geometry as geo

# One-shot: returns faces, barycentric uv, and world positions of the samples.
faces, uv, points = geo.poisson_disk_sample(points, faces, radius=0.05)

# Reusable object.
sampler = geo.PoissonDiskSampler(points, faces, radius=0.05, seed=0)
sampler.points, sampler.faces, sampler.uv, sampler.num_samples

# Spectrum analysis on the surface.
r, g = geo.pair_correlation(sampler.points, area=sampler.total_area, r_max=0.3)
```

## Testing Strategy

- **Correctness (R1)**: brute-force check that no accepted pair is closer than
  `radius`, on a plane and on a closed mesh, CPU and CUDA.
- **Maximality (R2)**: every *rejected/unused* candidate has an accepted sample
  within `radius` (nothing could have been added).
- **Coverage / count**: the sample count is within a sane band of the
  theoretical maximal count `N_est` (roughly `0.5*N_est` to `1.0*N_est`).
- **Area weighting (R4)**: on a mesh with wildly uneven triangle areas, the
  sample density (samples per unit area) is roughly constant across regions.
- **Determinism (R6)**: identical seed reproduces identical output; different
  seed differs.
- **Spectrum (R7)**: `pair_correlation` of the result is ~0 below `radius` and
  approaches 1 at large radius; a uniform random set shows no such gap.
- **Performance (R8)**: a regression/perf test samples a large mesh with a small
  radius (hundreds of thousands of candidates) within a generous time budget,
  and reports throughput. No speedup-ratio assertions (flaky in CI).
- **Example**: a headless polyscope script renders the mesh and its Poisson-disk
  samples and writes an animated GIF sweeping the radius.
