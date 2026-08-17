Add `wp.JetSpace()` and `wp.JetSpace2()`, which generate forward-mode jet types that carry a value together with its
derivatives along a chosen number of directions at once. `wp.JetSpace()` gives first-order jets (`scalar`, `vec2`,
`vec3`, `mat2`, `mat3`, and rectangular `mat32`/`mat23`); evaluating a `@wp.func` over them produces its gradient as a
side effect with no hand-differentiation, and because the jet arithmetic is ordinary Warp code, `wp.Tape` can
differentiate through it to give reverse-over-forward Hessians without a second-order tape. `wp.JetSpace2()` gives
second-order scalar jets that carry a value, gradient, and dense Hessian, yielding the local Hessian in a single forward
pass with no tape at all. The generated arithmetic is registered as overloads of Warp's builtins, so jets are written
with ordinary operator and builtin syntax (`a * b`, `wp.sin(a)`, `wp.length(d)`, `wp.determinant(m)`, `v[0]`) and
nothing has to be bound into the calling module. See `design/forward-mode-jets.md` and the
`optim/example_implicit_projection.py` example, which uses first-order jets to project points onto a metaball level set
and second-order jets to drive a closest-point Newton flow.
