Add an optional ``face_areas`` argument to ``warp.geometry.UniformSampler`` so a
caller can supply precomputed per-triangle areas instead of having the sampler
recompute them from the mesh; when omitted the areas are computed as before.
