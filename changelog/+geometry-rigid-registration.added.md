Add rigid registration (iterative closest point) to `warp.geometry`.
`warp.geometry.register_rigid` aligns a source point set to a target with
point-to-plane Gauss-Newton ICP, abstracting over mesh targets (`warp.Mesh` or
`(points, faces)`) and point-cloud targets (bare points, with normals estimated
by local PCA, or `(points, normals)`). Because the motion is rigid, the target's
BVH or hash grid is built once and queried each iteration -- never rebuilt. The
closest-point device functions `warp.geometry.closest_on_mesh` and
`warp.geometry.closest_on_points`, and the Gauss-Newton term builder
`warp.geometry.point_plane_term`, are exposed for use in your own kernels. See
`warp/examples/geometry/example_icp_registration.py`.
