Extend `wp.JetSpace()` and `wp.JetSpace2()` with quaternion and `vec3` payloads and a rotation-vector exponential
map, so a scalar objective written through `quat` rotations differentiates without hand-written derivatives. Seeding a
3-vector tangent and calling `J.exp_map()` gives on-manifold (unit-quaternion) derivatives -- the 3x3 tangent Hessian
of a second-order jet in a single forward pass -- while seeding the four components directly gives full-space
derivatives. See `warp/examples/optim/example_quat_shape_align.py`, which aligns one query molecule against a dataset
of candidates under a robust (Cauchy) loss by intrinsic Newton on the rotation -- a nonconvex fit with no closed-form
Procrustes solution, so the one-pass Hessian earns its keep.
