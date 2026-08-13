Add `warp.geometry` module with `warp.geometry.delaunay_edge_flip()` for parallel, in-place
Delaunay edge flipping of 2D triangle meshes, along with the supporting
`warp.geometry.triangle_triangle_adjacency()` and `warp.geometry.in_circle()` helpers. Flips run
as a priority-based maximal independent set and the convergence loop is driven on-device with
`warp.capture_while()`, so the whole routine is CUDA-graph capturable. The FEM elastic shape
optimization example now uses it in place of a host-side NumPy edge flipper.
