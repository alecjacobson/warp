# Inverse-elasticity shape optimization

Design notes for `warp/examples/optim/example_inverse_elasticity.py`, a pure-Warp
GPU reimplementation of Alec Jacobson's `gauss-newton-sensitivity-analysis` C++
reference. Every layer is validated against that reference to machine precision.

## Problem

A 2D elastic bridge is pinned at its left and right edges and sags under gravity.
We optimize its **rest shape** so that the gravity-**deformed** shape matches a flat
target, minimizing `mean(|V_target - U(V)|^2)` over the free (non-pinned) vertices,
where `U(V)` is the deformed shape produced by the forward elasticity solve.

## Forward model

Constant-strain triangles, plane-strain linear elasticity. Per element,
`K_e = area * B^T C B` with the engineering-strain `B` and the plane-strain
constitutive matrix `C`; mass is lumped (`area/3` per vertex). The equilibrium
`K u = M f_ext` (Dirichlet zero at the pins) gives the displacement `u`, and
`U = V + u`. The element operators are transcribed directly from the C++ reference
and match it exactly.

## Solver: cuDSS sparse direct

The stiffness `K` is SPD but, at the demo's Poisson ratio (ν = 0.49,
near-incompressible), badly conditioned. This is the crux of the design:

- An **iterative** solve (CG) needs hundreds-to-thousands of iterations here, each
  firing many tiny kernels. At the demo's mesh sizes (hundreds–thousands of DOFs)
  that is *dispatch-bound* on the GPU and ends up **slower than the CPU** direct
  solve. Neither warm-starting (CG's iteration count is set by the spectrum, not
  the initial guess) nor Jacobi/block-Jacobi preconditioning meaningfully helps;
  only a global preconditioner (multigrid) would, at large complexity cost.
- A **sparse direct** solve is robust to the conditioning and, crucially, factors
  `K` once and reuses that factorization for both the forward solve (`K u = load`)
  and the adjoint solve (`K λ = r_free`). On GPU via
  [cuDSS](https://docs.nvidia.com/cuda/cudss/) (the `warp-cuDSS` helper), the
  factorization parallelizes well.

Result: the GPU **beats** an equivalent C++ CPU sparse-direct implementation, and
the margin grows with mesh refinement (measured, L40 vs the reference on CPU):

| count | free DOFs | C++ CPU | GPU cuDSS | speedup |
|---|---|---|---|---|
| 2 | 87 | 0.38 s | 0.22 s | 1.7× |
| 4 | 295 | 11.4 s | 1.6 s | 7× |
| 8 | 1071 | 153 s | 6.3 s | 24× |

All converge to the same ~1e-8 loss in the same iteration count as the reference.

## Gradient: adjoint via autodiff

The rest-shape gradient is the adjoint gradient of the mean-squared shape error.
Rather than assembling the geometry-sensitivity matrix `G` explicitly (error-prone),
we solve the adjoint system `K λ = r_free` (reusing the cuDSS factorization) and
obtain `G_ff^T λ = ∂(λ · R)/∂V` — where `R = M f_ext - K u` is the equilibrium
residual force — by **autodiff of a single residual-force kernel** through a
`wp.Tape` (the implicit-function-theorem adjoint). No explicit sensitivity matrix
is formed. The result matches the C++ `gradient_step` to ~1e-11 at both flat and
strongly arched shapes.

## Optimizer

`warp.optim.Adam` (single precision — the optimizer state is fp32 while the physics
stays fp64). Adam converges to `tol = 1e-8` in the same iteration counts as the
reference (~575 at count=2, ~4200 at count=4, ~11500 at count=8), flooring only at
~1e-12 due to the fp32 optimizer state — far below any meaningful threshold.

## Validation

The C++ reference is exposed to Python via a small pybind11 binding
(`forward_sim`, `loss`, `gradient_step`, `gauss_newton_step`, `adam_step`,
`optimize_adam`) and used as ground truth during development. The shipped
regression test is self-contained (a finite-difference gradient check plus a
convergence check) and does not depend on it.

## Visualization

`render_convergence_gif` (optional, polyscope headless) stacks the rest shape over
the gravity-deformed shape (colored by von Mises stress) across the optimization,
subsampled to ≤60 frames. It is self-contained and safe to delete.

## Gauss-Newton

The example also provides a Gauss-Newton optimizer (the reference's square-route
step). Per step: solve the forward equilibrium, assemble the geometry sensitivity
`G_ff` (element residual-force Jacobian, `G_e[:,a] = ∂(M_e f_e − K_e u_e)/∂x_a`,
sharing `A`'s pattern), form the nonsymmetric `T = A + G_ff`, solve
`T w = G_ff r_free` with cuDSS (`mtype="general"`), and take `p = r_free − w`.
Both the SPD forward solve and the general T-solve run on cuDSS.

GN converges **quadratically in a handful of iterations** regardless of mesh size
(3/4/6 iterations at count=2/4/8, vs Adam's 575/4200/11500), matching the C++
reference to the digit. `step_size = 1.0` is safe through count=8; finer meshes
need a smaller step (~0.25 at count≥16) to avoid overshoot — the reference's
documented behavior, confirmed against the trusted C++ (which also diverges at
step 1.0 there). Because GN takes so few iterations, the GPU advantage over the
C++ CPU appears only once the per-solve work is non-trivial: ~0.4× at count=4 (GN
is trivially fast there), 1.5× at count=8, 3.8× at count=16.

Note: the current GN T-solve uses a fresh cuDSS solver each step (re-running the
symbolic analysis), where the C++ reuses one factorization workspace. Reusing a
persistent T solver across steps (two persistent cuDSS solvers coexist fine) would
remove that redundant analysis — a straightforward future optimization.
