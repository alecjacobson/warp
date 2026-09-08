Add `warp/examples/optim/example_inverse_elasticity_3d.py`, the 3D tetrahedral analogue of
the inverse-elasticity shape-optimization example. A pinned 15:3:1 elastic bar
(tetrahedralized) has its rest shape optimized so that its gravity-sagged shape matches a
flat target. Linear tetrahedra with isotropic 3D linear elasticity are solved with the cuDSS
sparse **direct** solver, and the rest shape is optimized with a sparse Gauss-Newton step
(``T = A + G_ff`` solved directly with cuDSS ``mtype="general"``) that converges quadratically
in a few iterations. Because there is no external reference for this variant, correctness rests
on implementation-independent checks (rigid-body modes, eigenvalue count, an independent NumPy
forward, and a finite-difference Newton step built from the forward map alone), all covered by
regression tests. Includes an optional headless (polyscope) convergence gif.
