Add `warp/examples/optim/example_inverse_elasticity.py`, a pure-Warp 2D inverse-elasticity
shape-optimization example. A pinned bridge's rest shape is optimized so that its
gravity-sagged shape matches a flat target. The constant-strain-triangle, plane-strain
forward model is solved with the cuDSS sparse **direct** solver (one factorization per
shape, reused for the forward and adjoint solves); the rest-shape gradient is obtained by
the adjoint method (autodiff through the assembly plus a manual adjoint for the linear
solve), and the shape is optimized with either Warp's Adam optimizer or a sparse Gauss-Newton step
(``T = A + G_ff`` solved directly with cuDSS ``mtype="general"``) that converges in a few
iterations. Includes a finite-difference
gradient regression test and an optional headless (polyscope) convergence gif. The GPU solve
beats an equivalent CPU sparse-direct implementation, with the advantage growing as the mesh
is refined.
