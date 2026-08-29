Add an optional geodesic distance metric to `warp.geometry.PoissonDiskSampler`
and `poisson_disk_sample` via a `geodesic=True` flag, plus the underlying
`warp.geometry.geodesic_distance` device function (the fast normal-based
approximation of Bowers et al., exact on a sphere). In geodesic mode the minimum
distance is measured along the surface, which stops samples on opposite sides of
a thin feature -- close in 3D but far along the surface -- from over-separating.
The default Euclidean path is unchanged in both behavior and performance.
