<!-- Temporary notes: Warp vs libigl vs Sellán (spacetime continuation) comparison.
     Companion to SWEPT_VOLUME_MR.md; not necessarily for the MR body. -->

# Swept volume: Warp vs libigl vs Sellán

A three-way comparison on a **single rigid part** (none of these do articulation
easily, so the bunny is the fair common ground). Bunny (6,102 v), swept along a
20-keyframe screw trajectory (120° rotation + translation), grid spacing matched
across all three at `h = eps = 0.0221`.

The three methods represent two different philosophies:

- **Warp** (this MR) and **libigl** (`igl::swept_volume`) *stamp* the shape at
  discrete pose samples and union the per-pose signed distances on a grid.
- **Sellán** — *Swept Volumes via Spacetime Numerical Continuation*, Sellán,
  Aigerman, Jacobson, SIGGRAPH 2021 (`github.com/sgsellan/swept-volumes`) —
  computes the **continuous** swept volume of the interpolated motion via
  numerical continuation. It is effectively the ground truth that discrete
  stamping should converge to.

## Outputs and timing (20-keyframe trajectory)

| method | approach | verts / faces | volume | time |
|---|---|---|---|---|
| **Sellán** | continuous continuation | 13,478 / 26,952 | **0.381** | 18,760 ms |
| **Warp** (L40) | 20-pose stamp | 14,948 / 29,896 | 0.344 | **26 ms** |
| **libigl** (CPU) | 20-pose stamp | 23,174 / 45,204 | 0.326 | 1,237 ms |

Sellán has the **largest** volume because it is continuous — it fills the gaps
*between* the 20 poses. The discrete stamps under-approximate (Warp 0.344,
libigl 0.326); the gap closes as the temporal sampling densifies.

(libigl's larger vert count here is its high-level `swept_volume` building its
own coarser grid + its own MC; on a *matched* grid libigl ≡ Warp to 0.13 voxels,
so it converges to Sellán identically, just ~50× slower on CPU.)

## Convergence — dense stamping → continuous ground truth

Warp stamping the same screw trajectory at increasing pose counts, symmetric
Hausdorff to Sellán's continuous result (in voxels, `h = 0.0221`):

| poses | Warp time | Hausdorff to Sellán |
|---|---|---|
| 20  | 26 ms  | 3.51 vox |
| 60  | 77 ms  | 2.14 vox |
| 180 | 235 ms | **0.81 vox** |
| 540 | 702 ms | 0.70 vox |

Warp's GPU dense-stamp **converges to Sellán's continuous continuation result**
(3.5 → 0.7 voxels). The residual ~0.7-voxel floor is just the MC-vs-dual-contouring
meshing at spacing `h`, not a method disagreement.

## Take-away

At ~180 poses Warp matches the sophisticated continuous method to **sub-voxel in
235 ms — about 80× faster than Sellán computes it (18.8 s)**. On the GPU you can
brute-force dense stamping all the way to the continuous limit *faster than the
continuous method itself runs*. That is the core argument for the dense approach
in the Warp context: simplicity (pure Warp, existing mesh queries, no
continuation solver) with no accuracy penalty at achievable pose counts.

## Reproduction notes

- Sellán's code is a C++/CMake **GUI** app with no Python binding. For a headless
  run I built a small **driver** that calls its `swept_volume(V, F, Transformations,
  eps, num_seeds, dir, U, G, ...)` library function directly (skipping the GLFW
  viewer), against the repo's pinned libigl submodule and Eigen 3.3.9. Runs with
  no display.
- Grids matched by *spacing* (`h = eps`); each method builds its own grid/origin
  and meshes differently (Warp MC, libigl MC, Sellán dual contouring), so the
  comparison is surface-to-surface Hausdorff, which is robust to those choices.
- Single rigid body only: Sellán is single-shape/GUI-driven, so articulated
  assemblies (the UR10) are out of scope for this three-way — see
  `SWEPT_VOLUME_MR.md` Appendix A for the Warp-vs-libigl UR10 comparison.
