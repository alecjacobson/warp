Add `warp.geometry.connected_components()`, which labels the connected components of a simplicial
mesh from a flat `int32` index array (no vertex positions), matching the edge-based connectivity of
gptoolbox's `connected_components`. A `simplex_size` argument selects the number of vertices per
simplex (`2` segments, `3` triangles by default, `4` tetrahedra, and so on). It returns per-vertex
component ids in a contiguous `[0, num_components)` range together with the component count. The
labeling runs on the Warp device (CPU or CUDA) as a parallel union-find -- simplex-parallel hooking
plus vertex-parallel full path compression, iterated on-device to a fixpoint -- so it scales to
large meshes without a serial sweep. The convergence loop is driven on-device with `warp.capture_while()`, so the routine
is CUDA-graph capturable; during capture `num_points` is required and the component count is
returned as a single-element device array.
