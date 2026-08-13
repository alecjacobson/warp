Add `warp.geometry` module with `warp.geometry.delaunay_edge_flip()` for parallel, in-place
Delaunay edge flipping of 2D triangle meshes, along with the supporting
`warp.geometry.triangle_triangle_adjacency()` and `warp.geometry.in_circle()` helpers. The FEM
elastic shape optimization example now uses it in place of a host-side NumPy edge flipper.
