Add `warp/examples/optim/example_inverse_elasticity.py`, a pure-Warp 2D inverse-elasticity
shape-optimization example. A pinned bridge's rest shape is optimized so that its
gravity-sagged shape matches a flat target, featuring a sparse Gauss-Newton step (`T = A + G_ff`
solved with a Warp Krylov solver) that converges in a few iterations where plain gradient
descent fails, alongside a Warp Adam baseline. Includes a numpy/scipy host oracle and
finite-difference regression tests, and a headless convergence gif.
