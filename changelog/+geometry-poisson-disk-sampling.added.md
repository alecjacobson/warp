Add parallel Poisson-disk (blue-noise) surface sampling to `warp.geometry`,
implementing Bowers et al., "Parallel Poisson Disk Sampling with Spectrum
Analysis on Surfaces" (SIGGRAPH Asia 2010). `warp.geometry.poisson_disk_sample`
and `warp.geometry.PoissonDiskSampler` draw a point set in which no two samples
are closer than a given radius, resolving conflicts in parallel over a
single-entry spatial hash and 27 phase groups (memory scales with the surface,
not the 3-D volume). `warp.geometry.pair_correlation` measures the blue-noise
spectrum of a surface point set. See `warp/examples/geometry/`.
