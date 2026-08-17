<!--
SCRATCH FILE — not part of the feature. Draft MR description for the
forward-mode-jets MR. Copy into the GitLab MR body, then delete this file
before (or at) filing. Do NOT include it in the MR diff.

Title suggestion:
    Add forward-mode jets (wp.JetSpace / wp.JetSpace2)
-->

## Description

Adds forward-mode automatic differentiation ("jets") to Warp, so that
evaluating an ordinary `@wp.func` over jet values yields its derivatives as a
side effect — no hand-written gradients or Hessians.

**Motivation.** Warp differentiates in reverse mode, which gives gradients
cheaply but has no second-order tape. Simulation work usually needs a *local*
Hessian — an energy `E(x) = Σ_i g_i(S_i x)` where each `g_i` reads a handful of
scalars (a spring, a triangle, a tetrahedron). Today users derive those local
Hessians by hand (error-prone, hard to keep in sync) or give up on Newton-type
solvers. Forward-mode jets close this gap.

**Approach.** `wp.JetSpace(width, dtype)` specializes a family of jet types
(`scalar`, `vec2`, `vec3`, `mat2`, `mat3`, and rectangular `mat32`/`mat23`),
each a `@wp.struct` carrying a value plus its derivatives along `width`
directions. `wp.JetSpace2(width, dtype)` adds second-order *scalar* jets
carrying value + gradient + a dense Hessian.

The key design decision is that the generated jet arithmetic is **registered as
overloads of Warp's own builtins**, so user code uses ordinary syntax
(`a * b`, `wp.sin(a)`, `wp.length(d)`, `wp.determinant(m)`, `v[0]`) with nothing
bound into the calling module. This works because Warp resolves operators to
builtin names and does type-directed overload resolution; appending the jet
`@wp.func` overloads to those builtins puts jets on the same footing as native
types, and — because the jet arithmetic is itself ordinary Warp code —
`wp.Tape` can differentiate *through* it to give reverse-over-forward Hessians.

**Two Hessian routes.** (1) Reverse-over-first-order-jet via `wp.Tape` — reuses
Warp's autodiff, one Hessian row per backward pass; best on CPU and for long
chains. (2) Pure-forward second-order jet (`wp.JetSpace2`) — the whole local
Hessian in one forward pass with no tape; fastest on GPU for small energies.

**Alternatives considered** (detailed in `design/forward-mode-jets.md`):
`J.install(globals())`, module-level operator names, and a `@J.func` decorator
were all rejected as leaking plumbing into the user API; builtin registration
needs none of them.

See `design/forward-mode-jets.md` for the full design, requirements, coverage
matrix, and testing strategy.

## Changes

- **warp/_src/jet.py** (new): `JetSpace` / `JetSpace2` factories; jet struct
  types; arithmetic, transcendental, inverse-trig, branching, geometry, and
  matrix overloads registered on Warp builtins.
- **warp/tests/test_jet.py** (new): closed-form + finite-difference checks for
  value/gradient/Hessian, `vec3` geometry, multi-width coexistence, `float64`,
  caching, and width validation.
- **warp/tests/test_jet_ops.py** (new): finite-difference checks for the wider
  op surface — inverse-trig, `pow` variants, branching (1st and 2nd order), and
  matrix jets (`mat2` det/trace, `mat3` tet energy, `mat3` inverse, rectangular
  `mat32` triangle energy). CPU-only; see note in Validation.
- **warp/tests/unittest_suites.py**: register both modules in `default_suite`.
- **warp/examples/optim/example_implicit_projection.py** (new) + docs image +
  `docs/index.rst` gallery entry: metaball level-set projection (first-order
  jets) vs closest-point Newton flow (second-order jets).
- **warp/tests/test_examples.py**: register the example (CPU + CUDA).
- **warp/examples/benchmarks/benchmark_jet_{gradient,hessian}{,_mesh}.py**
  (new): gradient- and Hessian-strategy benchmarks; each gates its strategies
  against finite differences before timing.
- **changelog/+forward-mode-jets.added.md** (new), **design/forward-mode-jets.md**
  (new).

## Checklist

- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] I added a changelog fragment if this change affects users.

## Validation summary

- `warp/tests/test_jet.py` (`TestJet`, `TestJetSpace`): the Hessian of
  `g(a,b) = sin(a·b) + 0.1·a³ + exp(b)` is checked against both a closed-form
  NumPy Hessian and float64 second differences (independent oracles), with
  symmetry asserted separately since off-diagonals come from separate backward
  passes. Also value/gradient, `vec3` geometry via `wp.length`, indexing/`dot`/
  `cross`/`normalize`/`length_sq`, two widths coexisting (first still resolves
  after the second registers), a `float64` space to `1e-12`, and caching/width
  validation.
- `warp/tests/test_jet_ops.py` (`TestJetOps`, `TestJetMatrix`): finite-
  difference checks for inverse-trig, `pow` variants, and branching builtins
  (gradient for 1st order; gradient + Hessian for 2nd order), and for matrix
  jets (det/trace, tet energy, `mat3` inverse err ~1.4e-7, rectangular triangle
  energy grad err ~2.3e-6). **CPU-only by design**: building one module for both
  CPU and CUDA in the same process trips a module-hasher instability where the
  second device's build perturbs shared hash state; single-device keeps hashes
  consistent. CUDA correctness of these ops is covered by the finite-difference
  gates in the jet benchmarks.
- `warp/tests/test_examples.py`: `example_implicit_projection` runs to
  completion on CPU and CUDA (asserts return code 0).
- Ran the jet suite locally: all jet tests pass; `example_implicit_projection`
  converges (projection |f|→~4e-5) and the closest-point flow lands nearer the
  surface than the plain projection for all sampled points.

## New feature / enhancement

```python
import warp as wp

J = wp.JetSpace(2)  # first-order, 2 directions

@wp.func
def energy(a: J.scalar, b: J.scalar) -> J.scalar:
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)  # ordinary Warp syntax

@wp.kernel
def grad_kernel(z: wp.array2d[float], grad_g: wp.array[J.coeff]):
    i = wp.tid()
    # Seed the identity: one forward pass yields the whole local gradient.
    grad_g[i] = energy(J.seed(z[i, 0], 0), J.seed(z[i, 1], 1)).coeff

# Second-order: value + gradient + dense Hessian in one forward pass, no tape.
J2 = wp.JetSpace2(2)

@wp.func
def energy2(a: J2.scalar, b: J2.scalar) -> J2.scalar:
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)

@wp.kernel
def hess_kernel(z: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    e = energy2(J2.seed(z[i, 0], 0), J2.seed(z[i, 1], 1))
    for p in range(2):
        for q in range(2):
            hess[i, p, q] = e.hess[p, q]
```

## Follow-up work (tracked)

Deferred items, filed as issues on the development fork (not NVIDIA/warp):

- Native symmetric matrix type for the Hessian —
  https://github.com/alecjacobson/warp/issues/5
- Comparison operators dispatching through the overload table (so `a < b` works
  on a jet instead of comparing `.value`) —
  https://github.com/alecjacobson/warp/issues/10
- Second-order jets for vector/matrix payloads (may not be worthwhile; large
  compile cost) — https://github.com/alecjacobson/warp/issues/11
- Comprehensive local-Hessian benchmark —
  https://github.com/alecjacobson/warp/issues/9
