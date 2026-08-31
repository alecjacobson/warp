# Rigid Registration (Iterative Closest Point)

**Status**: In Progress

**Issue**: N/A (feature request)

## Motivation

Rigid registration -- finding the rotation and translation that best aligns a
*source* point set to a *target* surface -- is a workhorse of 3-D scanning,
SLAM, robotics, and shape analysis. The standard tool is Iterative Closest Point
(ICP): alternately (1) match each source point to its closest point on the
target and (2) solve for the rigid transform that minimizes a distance metric,
then repeat. We want a fast, batched, GPU implementation in Warp that abstracts
over point-cloud and mesh targets, defaults to the robust point-to-plane
Gauss-Newton formulation, and is competitive with the state of the art.

A key structural advantage we exploit: **the motion is rigid, so the target
never deforms.** We build the target's acceleration structure (BVH for a mesh,
hash grid for a point cloud) *once*; every iteration transforms the source and
queries the fixed target -- no rebuild.

## Requirements

| ID  | Requirement                                                                  | Priority | Notes |
| --- | ---------------------------------------------------------------------------- | -------- | ----- |
| R1  | Point-to-plane Gauss-Newton ICP as the default                               | Must     | Normals from the closest-point query |
| R2  | Target abstracted over mesh and point cloud                                  | Must     | `wp.Mesh` closest point / hash-grid NN |
| R3  | Rebuild no acceleration structure across iterations                          | Must     | Rigid motion; transform source, query fixed target |
| R4  | Publicly exposed `register_rigid(...)` plus reusable `@wp.func`s             | Must     | Jacobian, residual, closest-query, SE(3) helpers |
| R5  | Stochastic per-iteration subsampling of correspondences                      | Should   | Bouaziz et al. 2013 insight; speed + basin |
| R6  | Robust weighting (down-weight outlier correspondences)                        | Should   | Welsch / Tukey approximating sparse (l_p) ICP |
| R7  | Batching: multi-source, multi-target, multi-initialization                   | Should   | B independent 6x6 solves in parallel |
| R8  | Symmetric ICP variant (experimental)                                          | Could    | Rusinkiewicz 2019; wider basin |
| R9  | Comparison harness vs PyTorch3D, Open3D tensor, fast_gicp, PCL               | Should   | Accuracy + speed |
| R10 | Working example with a headless polyscope GIF                                | Must     | Convergence animation |

**Non-goals**: non-rigid / deformable registration; global registration
(RANSAC/feature matching) -- we assume a reasonable initial guess or provide a
multi-initialization sweep (R7); learned features.

## Literature and state of the art

- **Besl & McKay 1992; Chen & Medioni 1991.** ICP itself. Point-to-*point* (Besl)
  minimizes `||Rp+t-q||^2` (closed form via Umeyama/SVD). Point-to-*plane* (Chen &
  Medioni) minimizes `((Rp+t-q).n_q)^2` and converges much faster on smooth
  surfaces because it lets points slide along the tangent plane. We default to
  point-to-plane, linearized (small-angle) into a 6x6 Gauss-Newton system.
- **Rusinkiewicz & Levoy 2001, "Efficient variants of ICP."** Taxonomy of
  sampling / matching / weighting / rejection choices. Justifies random source
  sampling and normal-space sampling; informs R5/R6.
- **Segal, Haehnel, Thrun 2009, "Generalized ICP" (GICP).** Plane-to-plane:
  models each point with a local covariance and uses a Mahalanobis residual. A
  natural extension once we have per-point normals/covariances; noted as a future
  variant.
- **Bouaziz, Tagliasacchi, Pauly 2013, "Sparse Iterative Closest Point."**
  Replaces the l2 correspondence cost with a sparsity-inducing `l_p` (p in (0,1))
  norm solved by ADMM, so gross outliers incur little penalty and are effectively
  discarded. We take the practical insight -- **robustly discount a random subset
  of correspondences each iteration** -- as R5+R6 (stochastic subsampling plus a
  robust per-correspondence weight), which keeps ICP's simple 6x6 solve while
  gaining outlier tolerance at low per-iteration cost.
- **Rusinkiewicz 2019, "A Symmetric Objective Function for ICP."** Symmetrizes
  point-to-plane using *both* surfaces' normals and a half-rotation split,
  yielding a wider convergence basin and a linearization that is *exact* for
  exact correspondences. Its linearized system (eq. 10) has the **same 6x6
  structure** as point-to-plane, so it drops in as a variant (R8).

Baselines for R9:
- **PyTorch3D** `ops.iterative_closest_point`: batched point-to-*point* (Umeyama),
  GPU via Torch. Good speed baseline; different (weaker) metric.
- **Open3D tensor** `t.pipelines.registration.icp`: PointToPoint / PointToPlane /
  ColoredICP on CUDA tensors. The closest apples-to-apples for point-to-plane.
- **fast_gicp** (`pygicp`): CUDA (V)GICP, a strong speed/accuracy baseline.
- **PCL** `IterativeClosestPoint` / `GeneralizedIterativeClosestPoint`: CPU
  reference. Availability varies; harness records which baselines are installed.

## Design

### The core solve (point-to-plane, linearized Gauss-Newton)

With the current estimate `T = (R, t)` applied to source point `p_i`, let
`q_i, n_i` be the closest target point and its unit normal. Parameterize the
*incremental* transform by a rotation vector `a` and translation `t` (6 DOF),
`R_delta ~ I + [a]x`. The linearized point-to-plane residual is

    r_i = (p_i - q_i).n_i + a.(p_i x n_i) + t.n_i

so the 1x6 Jacobian row and scalar target are

    J_i = [ (p_i x n_i)^T , n_i^T ] ,   b_i = -(p_i - q_i).n_i

Accumulate the 6x6 normal equations `A = sum w_i J_i^T J_i`, `g = sum w_i J_i b_i`
(with robust weight `w_i`), solve `A x = g` (6x6 SPD, Cholesky), split
`x = [a; t]`, form `R_delta = expm([a]x)` (or `(I+[a]x)` re-orthonormalized), and
compose `T <- (R_delta, t_delta) . T`. Iterate to convergence.

**Symmetric variant (R8):** same 6x6 machinery, but `n_i = n_{p,i} + n_{q,i}`
(source + target normals), the rotation term uses `(p~_i + q~_i) x n_i` on
centered points, and the final transform composes the rotation twice with the
centroids (Rusinkiewicz eq. 11). Selected by a `variant=` argument.

### Target abstraction (R2) and no rebuild (R3)

A small typed dispatch over the target's closest-point query, exposed as
`@wp.func`s so users can call them in their own kernels:

- **Mesh target:** `wp.Mesh` built once. Per source point, `wp.mesh_query_point`
  returns the closest face + barycentric coords; `q` from
  `wp.mesh_eval_position`, `n` from `wp.mesh_eval_face_normal` (or interpolated).
- **Point-cloud target:** `wp.HashGrid` over target points built once; nearest
  neighbor gives `q`; `n` from precomputed target normals (estimated once, e.g.
  via local PCA, or supplied).

Each iteration a kernel applies the current `T` to the (subsampled) source points
and runs the query against the *unchanged* structure -- the rigid-motion payoff.
The query `@wp.func` returns `(q, n, valid)`; correspondences beyond a distance
threshold are rejected (`valid=False`, Rusinkiewicz rejection).

### Stochastic sampling and robustness (R5, R6)

Each iteration draws a random subset of `sample_count` source indices (seeded,
reproducible) and forms correspondences only for those -- cheaper iterations and
a mild annealing/basin-widening effect. A robust weight `w_i = welsch(r_i / s)`
(or Tukey), with scale `s` set from a robust statistic of the residuals,
approximates the sparse-`l_p` down-weighting of outliers without ADMM. Both are
optional and default on with conservative settings.

### Batching (R7)

A batch of `B` independent problems (any of multi-source, multi-target,
multi-initialization) is laid out as leading-dimension-`B` arrays. The
accumulation kernel reduces each correspondence's `J^T J` / `J^T b` into that
problem's 6x6 system (21 upper-triangular + 6 entries) via per-batch
accumulators; a `B`-wide kernel does the 6x6 Cholesky solve and transform
update. Multi-initialization shares one target structure across the batch (R3
still holds) and returns the best (lowest final residual) per source.

### Public API

```python
import warp.geometry as geo  # ICP lives in warp.geometry

# One-shot: align source points to a target (mesh or point cloud).
result = geo.register_rigid(
    source,                      # (N,3) points, or a (points, faces) mesh
    target,                      # (M,3) points [+ normals], or a mesh
    init=None,                   # (4,4) or (B,4,4) initial transforms
    variant="point_to_plane",    # or "symmetric", "point_to_point"
    max_iters=50, tol=1e-6,
    sample_count=None,           # stochastic subsample per iter; None = all
    robust="welsch",             # or None
    max_corr_dist=inf,
)
result.transform      # (4,4) or (B,4,4)
result.rmse, result.iterations, result.converged

# Exposed device functions for use in user kernels.
geo.point_plane_term(p, q, n)       # -> GaussNewtonTerm (Jacobian row + rhs)
geo.closest_on_mesh(mesh, p, dmax)  # -> ClosestPoint (point, normal, dist, valid)
```

The implementation lives in `warp.geometry` (alongside surface sampling), not a
separate module.

## Testing Strategy

- **Recovery (correctness):** apply a known random rigid transform to a mesh's
  surface samples, add small noise, and check ICP recovers the inverse to within
  a tight rotation/translation tolerance -- mesh and point-cloud targets, CPU and
  CUDA.
- **Point-to-plane vs point-to-point:** point-to-plane converges in fewer
  iterations on a smooth target (assert iteration counts / final RMSE).
- **Outliers:** with a fraction of source points replaced by outliers, the robust
  + stochastic path recovers the transform where plain l2 fails.
- **Batching:** `B` independent problems match the same problems solved
  one-by-one; multi-initialization picks the basin that reaches the global
  optimum.
- **No-rebuild invariant:** the target structure object is created once and its
  id is unchanged across iterations (assert, and it is implied by the API).
- **Determinism:** fixed seed reproduces the trajectory.
- **Symmetric variant:** wider basin -- converges from initializations where
  point-to-plane stalls.
- **Comparison harness** (`tools/benchmarks/`): accuracy (geodesic rotation error,
  translation error) and wall-clock vs PyTorch3D / Open3D / fast_gicp / PCL on a
  shared benchmark, recording which baselines are installed.
- **Example:** headless polyscope GIF of source converging onto target.

## Milestones (commit/push each)

1. Design doc (this) + `warp.geometry` ICP skeleton and SE(3)/Jacobian `wp.func`s
   with unit tests.
2. Point-to-plane GN ICP for a **mesh** target, single problem; recovery tests.
3. **Point-cloud** target (hash-grid NN + normals); recovery tests.
4. Stochastic subsampling + robust weighting; outlier tests.
5. Batching (multi-init first, then multi-source/target); batching tests.
6. Symmetric variant; basin test.
7. Comparison harness vs available baselines.
8. Example + polyscope headless GIF; docs + changelog.
