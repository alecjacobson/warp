Add `wp.JetSpace()`, which generates first-order forward-mode jet types (`scalar`, `vec2`, `vec3`) carrying derivatives
along a chosen number of directions at once. Evaluating a `@wp.func` over jets produces its gradient as a side effect,
with no hand-differentiation, and because the jet arithmetic is ordinary Warp code, `wp.Tape` can differentiate through
it to give reverse-over-forward Hessians without a second-order tape. The generated arithmetic is registered as
overloads of Warp's builtins, so jets are written with ordinary operator and builtin syntax (`a * b`, `wp.sin(a)`,
`wp.length(d)`, `v[0]`) and nothing has to be bound into the calling module. See `design/forward-mode-jets.md` and the
`optim/example_jet_hessian.py` example, which contrasts a width-k pass followed by k reverse sweeps against k width-1
Hessian-vector products.
