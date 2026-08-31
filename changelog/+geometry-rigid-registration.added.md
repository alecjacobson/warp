Add rigid registration (iterative closest point) to `warp.geometry`.
`warp.geometry.register_rigid` aligns a source point set to a target with
point-to-plane Gauss-Newton ICP, abstracting over mesh targets (`warp.Mesh` or
`(points, faces)`) and point-cloud targets (bare points, with normals estimated
by local PCA, or `(points, normals)`). Because the motion is rigid, the target's
BVH or hash grid is built once and queried each iteration -- never rebuilt. The
closest-point device functions `warp.geometry.closest_on_mesh` and
`warp.geometry.closest_on_points`, and the Gauss-Newton term builder
`warp.geometry.point_plane_term`, are exposed for use in your own kernels.
Optional stochastic subsampling and robust (Welsch) weighting follow Bouaziz et
al., "Sparse Iterative Closest Point" (2013), and a `variant="symmetric"` option
implements the wider-basin symmetric objective of Rusinkiewicz, "A Symmetric
Objective Function for ICP" (2019). `warp.geometry.register_rigid_batched`
runs many problems in parallel against a shared target for multi-initialization
(keep the best via `best_index`) and multi-source batching. The point-to-plane
residual can use the target's surface/PCA normal (`plane_normal="surface"`, the
default) or the query-closest direction (`plane_normal="closest_point"`, needing
no normals). See `warp/examples/geometry/example_icp_registration.py` and the
options ablation in `tools/benchmarks/`.
