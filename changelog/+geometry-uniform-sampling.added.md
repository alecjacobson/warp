Add a `warp.geometry` module for triangle-mesh processing, starting with
area-weighted uniform surface sampling. `warp.geometry.uniformly_sample` draws
points in one call, `warp.geometry.UniformSampler` amortizes the setup over many
draws, and the device function `warp.geometry.draw` (also exposed as the
`UniformSampler.draw` member `@wp.func`) samples a single point from within your
own kernels, returning a face index and barycentric coordinates like the
`wp.mesh_query_*` builtins.
