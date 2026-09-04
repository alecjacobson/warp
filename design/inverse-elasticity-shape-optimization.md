# Inverse-elasticity shape optimization example (sparse Gauss–Newton)

- **Status:** Implemented (GD / Adam / Gauss-Newton on CPU and CUDA, validated against a
  numpy/scipy host oracle and finite differences; headless convergence gif). Follow-ups:
  docs-gallery integration, larger-mesh solver robustness (Levenberg-Marquardt damping),
  and eventual replacement of `example_elastic_shape_optimization.py`.
- **Author:** AI-authored (Claude Code), directed by @alecjacobson
- **Reference:** https://github.com/alecjacobson/gauss-newton-sensitivity-analysis
  (C++/TinyAD/libigl/Eigen); derivation in that repo's `sparse_kkt_gauss_newton.md`.
- **Related work:** Zehnder, Coros, Thomaszewski, "Sparse Gauss-Newton for
  Accelerated Sensitivity Analysis," SIGGRAPH Asia 2021 (arXiv:2107.03285).

## Motivation

`warp/examples/fem/example_elastic_shape_optimization.py` is the current shape-
optimization example. It is broken in several ways and leans heavily on
`warp.fem` machinery, which makes it a poor learning reference. This document
plans a new, self-contained **pure-Warp** example that is (a) mathematically
transparent, (b) validated end-to-end against finite differences and host-side
direct solvers, and (c) a good showcase of Warp's sparse linear algebra
(`warp.sparse`), iterative solvers (`warp.optim.linear` — `cg`/`cr`/`gmres`),
and `warp.optim.Adam`, including the CPU/GPU tradeoffs between a first-order
GPU optimizer (Adam) and a sparse Gauss–Newton step.

## Problem

2D **inverse elasticity / rest-shape fitting**. A wide (15:1) triangulated grid
"bridge" is pinned at its left and right edges and sags under gravity. We
optimize the **rest shape** `V` (design variables `x = vec(V)`) so that once it
sags, the deformed shape `U = V + u(V)` matches a flat target `V_target` (the
original undeformed grid). The rest shape arches upward so gravity pulls it flat.

Notation (matching the reference derivation): `n = 2·#V`, `c = 1/√n`,
residual `r(x) = c(x + u(x) − x_target)`, loss `f(x) = ‖r‖² = mean((V_target−U)²)`.

### Forward model (linear elastostatics, constant-strain triangles)

- Per triangle: plane-strain stiffness `K_e = area · Bᵀ C B` (6×6, displacement
  ordering `[u0x,u0y,u1x,u1y,u2x,u2y]`) and lumped mass `M_e = (area/3)·I` on the
  diagonal. `C` from Young's modulus `Y` and Poisson `ν` (`λ = Yν/((1+ν)(1−2ν))`,
  `μ = Y/(2(1+ν))`). This is exactly the reference's `local_stiffness`/`local_mass`.
- Assemble global `K(V)`, `M(V)`. Load `ℓ = M·f_ext`, `f_ext = (0,−9.8)` per vertex.
- Dirichlet: pinned vertices have `u = 0`. With zero Dirichlet values the free
  solve is `A q = ℓ_free`, `A = K_ff` (SPD). `U = V + u`.

### Derivatives (sparse sensitivity)

- Sensitivity matrix `G(:,a) = ∂M/∂x_a · f_ext − ∂K/∂x_a · u` (u held fixed),
  assembled element-locally: `G_e(:,a) = ∂M_e/∂x_a·f_e − ∂K_e/∂x_a·u_e` for the 6
  local geometry coords `a`. `G_ff = G(free,free)`.
- **Gradient (adjoint):** one solve against `A`. `dV_free = (2/n)·(−(r_free +
  G_ffᵀ A⁻¹ r_free))`.
- **Gauss–Newton (sparse square route):** `T = A + G_ff`, solve `T w = G_ff·r_free`,
  then `p = r_free − w`; `dV_free = p`. `T` is sparse but **nonsymmetric /
  indefinite** — this is the crux that motivates a general solver (Warp `gmres`)
  rather than `cg`. (The reference uses Eigen `SparseLU`; the KKT route needs a
  symmetric-indefinite solver, which is why `SparseLU`/`gmres` is used instead of
  Cholesky/`cg`.)

Sign note: the reference uses `r = V_target − U` in code (so the step formulas
carry a sign flip vs. the `r = x+u−x*` convention in the derivation). We follow
the code convention and pin it down with the FD oracle.

### Why not just gradient descent

The reference proves plain **GD fails**: at the default mesh it diverges once the
loss reaches ~0.005–0.014 (raw gradient vs. GN direction cosine ~0.24 — nearly
orthogonal), because un-preconditioned per-vertex steps collapse individual
triangles toward zero area. **Gauss–Newton converges in ~4 iterations.** **Adam**
(step ~0.02) reaches GN-level precision in a few thousand iterations. Reproducing
these three behaviors is our end-to-end regression.

## Design

### Layout

```
warp/examples/optim/example_inverse_elasticity.py   # runnable example (CLI, optimize loop, viz/gif)
warp/examples/optim/inverse_elasticity_oracle.py    # numpy/scipy host reference (test oracle only)
warp/examples/optim/test_inverse_elasticity.py      # regression tests (run directly, unittest)
```

The physics (`local_stiffness`, assembly, forward solve, `G`, gradient, GN step)
lives as importable functions in the example module so the tests can exercise
each piece. Everything is `wp.array`/kernel-based; the host oracle is a small,
independent numpy/scipy reimplementation used only for validation.

### Warp implementation choices

- **Element operators:** `@wp.func local_stiffness(Vf, Y, ν) -> wp.mat66`,
  `local_mass(Vf) -> wp.mat66`, translating the reference formulas directly.
- **Element gradients `∂K_e/∂x`, `∂M_e/∂x`:** we only ever need the *matvec*
  `G_e(:,a) = ∂/∂x_a[M_e f_e − K_e u_e]` (a 6×6 Jacobian of a 6-vector element
  function). Two interchangeable routes, cross-checked against each other:
  (1) **central finite differences per element** over the 6 local coords —
  parallel, exact to FD tolerance, matches the reference's own suggestion of "six
  broadcast finite-difference directions"; (2) **Warp autodiff** of the element
  function. Start with (1) for robustness; keep (2) as a validation/showcase.
- **Assembly:** build `A = K_ff` and `G_ff` directly over free dofs (scatter only
  element entries whose both dofs are free; zero-Dirichlet makes the fixed columns
  drop out of the free equations) as `warp.sparse` BSR (1×1 scalar blocks to start;
  2×2 per-vertex blocks as an optimization later) via `bsr_from_triplets`.
- **Forward SPD solve:** `warp.optim.linear.cr` (or `cg`) on `A` with a Jacobi
  preconditioner.
- **GN solve:** form `T = A + G_ff` (`bsr_axpy`) and solve with `warp.optim.linear.gmres`
  (nonsymmetric). Report the solve residual.
- **Optimizers:** GD (`V −= α·grad`), `warp.optim.Adam` on the free-dof gradient,
  and GN (`V += α·p`). Loss/convergence/divergence tracking mirrors the reference's
  `OptimizeResult` (best loss, max spike ratio, converged/diverged flags).

### Validation strategy (regression as we go)

Two independent oracles, matching the reference test suite's philosophy:

1. **Host direct-solver oracle** (`inverse_elasticity_oracle.py`, numpy/scipy):
   full pipeline with `scipy.sparse.linalg.spsolve` for the forward solve and
   sparse LU on `T` for GN. Warp results must match it to tight tolerance
   (matvec, `u`, gradient, GN step, loss).
2. **Finite differences:** the host oracle's gradient and GN step are themselves
   checked against dense central-difference oracles (`warp.autograd.jacobian_fd`
   or direct FD), reproducing the reference's `test_main.cpp` checks
   (`gradient` err < 1e-5, `gauss_newton` err < 1e-4 at `count=2`).

Per-phase gates (each its own commit):
- P2 mesh/oracle: oracle gradient/GN match FD (< 1e-5 / 1e-4).
- P3 local ops: warp `K_e`,`M_e` match host + FD on a single triangle.
- P4 assembly/forward: warp `A·u`, forward `u` match scipy spsolve (< 1e-6 rel).
- P5 gradient: warp gradient matches host + FD oracle.
- P6 GN: warp GN step matches host LU-on-T + FD oracle; GMRES residual small.
- P7 optimizers: GN converges (loss < tol in ≲6 iters, `count=4`); Adam converges;
  GD diverges — matching the reference's documented behavior.

Tests run directly (`uv run .../test_inverse_elasticity.py`); a light end-to-end
smoke may also be registered with the example test suite. Per AGENTS.md, granular
physics tests stay with the example, not in `warp/tests/**` core.

### Visualization

Headless frames of the rest shape (arching) stacked over the deformed shape
(colored by per-face von Mises stress), assembled into a convergence gif. The
committed renderer uses matplotlib (Agg) for a labeled, equal-aspect 2D figure
with a colorbar -- a good fit for this planar problem. (Polyscope headless
rendering via EGL also works on this machine, including colormaps; matplotlib
was chosen for the built-in 2D labels/colorbar, not because polyscope was
unavailable.) The gif is committed under `docs/img/examples/` via git-LFS.

### CPU/GPU story

Adam is embarrassingly parallel and runs entirely on-GPU across many cheap
iterations; GN does few iterations but each needs a sparse assembly + a Krylov
solve. See the performance findings below for the actual tradeoff.

## Performance findings (NVIDIA L40, CUDA 12.6, float64)

Measured on this machine (throwaway harnesses; numbers are representative).

### Per-granularity step time vs. mesh size

| count | #verts | #tris | #free | assemble A | load | forward (CR) | G_ff | gradient | GN step |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 93 | 120 | 87 | 0.27 ms | 0.05 | 5.6 ms | 0.28 | 11.5 ms | 23 ms |
| 4 | 305 | 480 | 295 | 0.32 | 0.06 | 16 | 0.33 | 32 | 78 |
| 8 | 1089 | 1920 | 1071 | 0.32 | 0.06 | 33 | 0.31 | 65 | 415 |
| 16 | 4097 | 7680 | 4063 | 0.30 | 0.05 | 67 | 0.32 | 132 | 1515 |
| 32 | 15873 | 30720 | 15807 | 0.32 | 0.05 | 145 | 0.35 | 279 | **5974** |

- **Memory** scales `O(#tris)` (`A.nnz = 9·#tris` before compaction); block-diagonal
  workspace is `O(#tris)`. As expected.
- **Assembly** (A, G_ff, load) is cheap and roughly flat -- launch-bound at these
  sizes, not the bottleneck.
- **Forward/adjoint CR solves** are overhead-bound at small sizes (5.6 ms for 87
  dofs is almost entirely launch + per-iteration host residual-check sync, not
  compute) and grow ~`O(#free)` at large sizes.
- **The Gauss-Newton step is dominated by BiCGSTAB on `T = A + G_ff`** and scales
  **super-linearly** (6 s/step at count=32): `T` is ill-conditioned (`cond(T)`
  ranges ~1e5 at count=2 to ~1e8 at count=12) and the Jacobi-preconditioned
  BiCGSTAB iteration count grows with the mesh. This -- not assembly or host
  syncs -- is the real GN scaling limit, and the natural next optimization
  (a better preconditioner, or the direct sparse factorization the C++ reference
  uses, which Warp lacks natively).

### Precision (is float64 necessary?)

`cond(A)`/`cond(T)` grow to ~1e7/1e8 by count=12, so a **single-precision** direct
solve loses accuracy: relative error ~1e-3 at count≤4 but **~2-4% by count=12**.
float32 therefore "works" only for small meshes at low precision; the
high-precision GN convergence (to ~1e-9) and larger-mesh robustness that make this
example interesting **require float64**. float32 could be offered as a fast
low-precision mode but is not the right default here.

**Adam specifically** is the most float32-friendly path: it never touches the
ill-conditioned `T`, only the forward/adjoint solves (whose float32 error is
~1e-4 at small meshes), and it is gradient-driven, so a ~1e-4-accurate gradient
still lets it descend to roughly that noise floor. So float32 Adam is expected to
converge to ~1e-4-1e-6 loss (vs float64's ~1e-9) at small meshes -- fine for
Adam's role as the robust first-order baseline, and it would give Adam the ~2x
speed/memory advantage. (This is inferred from the measured float32 solve
accuracy; a definitive check needs the float32 Warp pipeline, a follow-up, since
a numpy Adam proxy through the host oracle is too slow to run to convergence.)
This is a good reason to make dtype a per-run option, with float32 available for
Adam.

### Graph capturability (host syncs)

The optimizer step is **not currently CUDA-graph capturable**: a capture attempt
fails on a device→host copy from (a) the Krylov solvers' host-side residual checks
(`check_every>0`) and (b) `bsr_from_triplets`' nnz read-back, plus (c) the
per-iteration `loss()` `.numpy()`. This matters because the as-is Adam-vs-GN
wall-clock is overhead-dominated (see below) and thus unfair to Adam. Making the
step capturable requires: solvers with `check_every=0` (device-side convergence
via conditional graph nodes, supported on CUDA ≥12.4), **fixed-pattern in-place
assembly** (build the sparsity once, then scatter-add element blocks into the
compact `values` array with a precomputed triplet→block map, avoiding
`bsr_from_triplets` per step), and a device-side loss.

Progress: the two enabling mechanisms are verified. Atomic scatter into a BSR's
`scalar_values` view works (a tiny test confirmed it), and `build_sparsity()` +
`assemble_stiffness_inplace()` refill `A` from a fixed pattern, matching the
triplet assembly to 1.5e-16 on CPU and CUDA. `check_every=0` alone already gives
~2x on the forward solve at count=8 (little at count=2, where per-call solver
graph setup ~20 ms dominates). Remaining to fully capture the step: the same
in-place treatment for `G_ff` and `T = A + G_ff` (identical pattern to `A`), a
device-side loss, and wrapping one optimizer step in a captured graph.

### Alternative Gauss-Newton solver routes to try

The C++ reference's `sparse_kkt_gauss_newton.md` lists larger, sparser
formulations of the GN step -- the 2m x 2m square `(p,w)` system
`S = [[I, I], [-G_ff, A]]` and the full 3m x 3m symmetric-indefinite KKT saddle
system -- and notes the KKT is a poor fit for CPU pivotless `LDLT` (indefinite),
so a GPU iterative solver (BiCGSTAB/GMRES, which ignore indefiniteness) is worth
trying. **However, a measurement here found these larger systems are actually
*worse*-conditioned than `T` for this problem**, so BiCGSTAB on them is likely
slower, not faster:

| count | m | cond(T) | cond(S, 2m) | cond(KKT, 3m) |
|---|---|---|---|---|
| 2 | 174 | 1.1e5 | 1.3e6 | 7.7e6 |
| 3 | 352 | 3.9e5 | 3.3e6 | 4.4e7 |
| 4 | 590 | 1.0e6 | 1.6e8 | 1.6e8 |

The saddle structure (rank-deficient `[[I,I],[I,I]]` block, zero corner) drives
the higher condition number -- also why LDLT fails. Still on the list to try
empirically on GPU (iterative convergence depends on spectrum clustering, not
just `cond`; and the notes' `c^2 = 1/n` scaling or a small Levenberg-Marquardt
`T <- T + µI` regularization may change the picture), but the naive larger
systems are not an obvious win. Improving the GN solver is better pursued via a
stronger preconditioner for `T` (which is already the best-conditioned form).

### Adam vs. Gauss-Newton wall clock (count=2, as-is)

| method | iters to `tol=1e-8` | wall clock | final loss |
|---|---|---|---|
| Gauss-Newton | 4 | 111 ms | 1.3e-10 |
| Adam (lr 0.02) | 507 | 8.9 s | 1.9e-9 |

GN is ~80× faster **as measured**, but Adam's ~17 ms/iter is almost all host-sync
/ launch overhead (its actual per-step compute at 87 dofs is microseconds). With
the graph-capture refactor above, Adam's per-step overhead collapses and the gap
is expected to narrow to roughly single-digit-x; at larger meshes GN's BiCGSTAB
cost grows, narrowing it further. A fully fair comparison should be run on the
capturable implementation.

## Open questions / risks

- **Polyscope headless** on a display-less server may require EGL; matplotlib is
  the fallback. Resolve in the viz phase.
- **`gmres` on `T`** must converge on the nonsymmetric system; if conditioning is
  poor at larger meshes, fall back to `bicgstab` or add Levenberg–Marquardt
  damping (`(c²+µ)I` on the `(p,p)` block / `T ← T + µI`-style), which the
  reference documents.
- **BSR block size / Dirichlet reindexing** details (1×1 vs 2×2 blocks, free-dof
  remap) are an implementation choice; start simple (scalar CSR-like), optimize
  later.
- Step-size safety: like the reference, the fixed GN `step_size=1.0` is not safe
  at every resolution; document and default conservatively.

## Non-goals

- Nonlinear elasticity, contact, or remeshing.
- Matching the reference's full menu of optimizers (VectorAdam, L-BFGS, Sobolev);
  GD / Adam / Gauss–Newton are the didactic core. Others may follow.
