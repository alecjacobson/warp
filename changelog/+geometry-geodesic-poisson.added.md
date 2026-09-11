Add an optional on-surface distance metric to `warp.geometry.PoissonDiskSampler`
and `poisson_disk_sample` via a `geodesic=True` flag. It uses the fast
normal-based curvature correction of Bowers et al. (a local approximation of
geodesic distance, exact on a sphere) rather than a true shortest-path geodesic.
In this mode the minimum distance is measured along the surface, which stops
samples on opposite sides of a thin feature -- close in 3D but far along the
surface -- from over-separating. The default Euclidean path is unchanged in both
behavior and performance.
