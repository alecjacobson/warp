# Forward-Mode Jets

**Status**: Implemented

**Issue**: [GH-1813](https://github.com/NVIDIA/warp/issues/1813)

## Motivation

Warp differentiates kernels in reverse mode. That gives gradients cheaply, but
second derivatives require differentiating a gradient, and Warp has no
second-order tape.

For simulation work the quantity actually needed is usually not a global
Hessian but a *local* one: an energy written as a sum of small terms,

```
E(x) = sum_i g_i(S_i x)
```

where each `g_i` reads a handful of scalars (one spring, one triangle, one
tetrahedron). The per-term Hessian is small and dense, and the assembled
Hessian is sparse. Users writing such energies today either derive the local
Hessian by hand -- error-prone and hard to keep in sync with the energy -- or
give up on Newton-type solvers.

Forward-mode jets close this gap. A jet carries a value together with its
derivatives along `width` directions at once:

```
x = value + sum_q coeff[q] * eps_q
```

Evaluating `g_i` over jets rather than floats yields `grad g_i` as a side
effect of evaluating it, with no hand-differentiation. Crucially, the jet
arithmetic is itself ordinary Warp code, so Warp's existing reverse-mode
autodiff can differentiate *through* it. Reverse-over-forward then gives the
local Hessian one row at a time, using only machinery Warp already has.

Jets are complementary to any scheme in which users supply local derivatives by
hand and a separate mechanism scatters them into a global sparse Hessian: jets
remove the need to write those local derivatives, and say nothing about the
assembly step.

## Requirements

| ID  | Requirement                                                                       | Priority | Notes                                            |
| --- | --------------------------------------------------------------------------------- | -------- | ------------------------------------------------ |
| R1  | Evaluating a `@wp.func` over jets yields its gradient with no hand-differentiation | Must     |                                                  |
| R2  | Warp's reverse mode differentiates through jet code, giving reverse-over-forward Hessians | Must | Via `wp.Tape`, or in-kernel with `wp.grad` for a local, tape-free sweep |
| R3  | Ordinary Warp syntax: `a * b`, `wp.sin(a)`, `wp.length(d)`, `v[0]`                 | Must     | No names bound into the caller's module          |
| R4  | Derivative width independent of geometric dimension                                | Must     | 6 directions over two `vec3` endpoints           |
| R5  | Several widths and dtypes coexist in one process                                   | Should   |                                                  |
| R6  | Geometry helpers: `dot`, `length`, `normalize`, `cross`                            | Should   |                                                  |
| R7  | Matrix jets: `determinant`, `trace`, `transpose`, `inverse`, matmul                | Should   | Square `mat2`/`mat3` and rectangular `mat32`/`mat23` for deformation gradients |
| R8  | Second-order scalar jets giving a dense local Hessian in one forward pass           | Should   | `wp.JetSpace2`; no tape required                 |

**Non-goals**: second-order jets for vector/matrix payloads
([#11](https://github.com/alecjacobson/warp/issues/11) -- may not be
worthwhile); quaternion jet types; a native symmetric matrix type for the
Hessian ([#5](https://github.com/alecjacobson/warp/issues/5)); comparison
operators dispatching through the overload table so `a < b` works on a jet
([#10](https://github.com/alecjacobson/warp/issues/10)); assembling the sparse
global Hessian from the local blocks; choosing a sparsity pattern or a linear
solver.

## Design

### What `width` is

`width` is the number of derivative directions a jet carries, and it is fixed at
type-specialization time: `wp.JetSpace(6)` produces types whose `coeff` is a
6-vector, unrolled into the generated code. It is a **compile-time constant, not
a runtime length**. This is the single most important thing to understand about
the API, because it decides what jets are and are not for.

The consequences follow directly:

- **Cost scales with `width`, whether or not you use it.** A first-order jet
  carries `width` derivative components through every intermediate; a
  second-order jet carries `width²`. Nothing is sparse and nothing is skipped,
  so the width you ask for is the width you pay for, in registers and in
  compile time.
- **Each distinct width is a distinct specialization.** `wp.JetSpace(6)` and
  `wp.JetSpace(9)` generate separate struct types and separate builtin
  overloads. They coexist fine (R5), but a width chosen per launch, or read
  from data, is not expressible.
- **This is the right shape for a *local* derivative.** The energies this
  targets have a small, statically known arity: a spring reads 2 `vec3` nodes
  (`width = 6`), a triangle 3 (`width = 9`), a tetrahedron 4 (`width = 12`).
  The width is a property of the element type, known when the kernel is
  written, and the resulting dense local gradient or Hessian block is exactly
  what a Newton solver wants to scatter into a sparse global matrix.
- **It is the wrong shape for a global or unbounded derivative.** Differentiating
  a loss with respect to all `n` degrees of freedom is not a `width = n` jet.
  `n` is a runtime quantity, so it cannot specialize a type; and even where it
  is known, the cost is `n` forward directions to reverse mode's single
  backward pass. Reverse mode (`wp.Tape`, `wp.grad`) remains the right tool
  there, and jets do not replace it. Jets are for the inner, fixed-arity term;
  reverse mode is for the outer sum over an unbounded number of them.

A useful test: if you can write the width as a literal next to the `@wp.func`
that consumes it, jets fit. If the width depends on the size of the problem,
they do not.

### Approach

`wp.JetSpace(width, dtype)` specializes a family of types at
code-generation time and returns them in a namespace:

| Name    | `value`         | `coeff`              |
| ------- | --------------- | -------------------- |
| `scalar`| `dtype`         | `vector(width)`      |
| `vec2`  | `vector(2)`     | `matrix(2, width)`   |
| `vec3`  | `vector(3)`     | `matrix(3, width)`   |
| `mat2`  | `matrix(2, 2)`  | `matrix(4, width)`   |
| `mat3`  | `matrix(3, 3)`  | `matrix(9, width)`   |
| `mat32` | `matrix(3, 2)`  | `matrix(6, width)`   |
| `mat23` | `matrix(2, 3)`  | `matrix(6, width)`   |

Matrix `coeff` stores the entry derivatives row-major (`coeff[r * cols + c, q]`
is `d value[r, c] / d eps_q`). Each type is a `@wp.struct`. The arithmetic is
generated as ordinary `@wp.func`
overloads over those structs, with width-dependent loops statically unrolled
via `wp.static(width)`.

The key move is what happens to those overloads: they are **registered as
overloads of Warp's own builtins**, so user code needs no jet-specific syntax:

```python
J = wp.JetSpace(6)

@wp.func
def spring_energy(x0: J.vec3, x1: J.vec3) -> J.scalar:
    d = x1 - x0
    r = wp.length(d) - 1.0
    return 0.5 * r * r
```

`spring_energy` is a plain `@wp.func`. Nothing is imported or installed.

### Why registration on builtins is necessary

Warp resolves a binary operator in `Adjoint.emit_BinOp` roughly as:

```python
name = builtin_operators[type(node.op)]        # ast.Mult -> "mul"

user_func = adj.resolve_external_reference(name)
if isinstance(user_func, Function):
    return adj.add_call(user_func, (left, right), {}, {})

return adj.add_builtin_call(name, [left, right])
```

`resolve_external_reference` is `resolve_closure_or_global(adj.func, name)`: it
looks the name up in the closure or globals **of the function being parsed**.

That is the whole problem. A jet library that generates its overloads inside a
factory function has them in the factory's local scope, with no lexical
connection to the user's module. Operators then only work if the user injects
the names first:

```python
J.install(globals())    # what this design exists to avoid
```

Beyond being ugly, that approach leaks plumbing into the API: it has to merge
overload sets when two widths share a module, guard against clobbering a
user's own `add()`, and it shadows Python's `pow` builtin in the target module.

The final line of the resolution above is the way out. Ordinary expressions
like `vec3 * float` need no `mul` in user globals -- they fall through to
Warp's builtin overload registry and resolve by argument type. Registering the
jet arithmetic there puts jets on exactly the same footing:

```
a * b  ->  builtin "mul"  ->  type-directed overload resolution  ->  mul(scalar, scalar)
```

The same applies to `wp.sin`, `wp.dot`, `wp.cross`, `wp.normalize`, and to
`v[0]`, which routes through the `extract` builtin.

### Alternatives Considered

**`J.install(globals())`.** Inject the operator names into the caller's module
before it defines any `@wp.func`. Works, but every consumer module pays for it,
and the helper needs real machinery: merging overloads across widths, a
collision guard, and an unavoidable shadowing of the `pow` builtin. Rejected as
plumbing in the user API.

**Module-level operator names in the jet module,** with every specialization
added to them, so users write `from warp.jet import add, sub, mul, div`. Avoids
the `install` call but is the same idea: Warp still requires those bare names in
the consumer's globals. Rejected.

**A jet-aware decorator, `@J.func`,** that clones the user's function with a
private `__globals__` containing the operator names, then hands it to
`wp.func`. This keeps the caller's module clean and needs no changes to Warp,
which makes it the best option for a library living *outside* Warp. It still
forces a non-standard decorator on every jet function, and a matching
`@J.kernel` for kernels containing jet expressions. Rejected once the work moved
into Warp, where builtin registration is available and needs neither.

**Second-order jets** give the Hessian in one forward pass with no tape at all.
This was originally deferred -- it does not reuse Warp's reverse-mode autodiff
the way first-order jets do, and the state grows quadratically in the width --
but the tradeoff is worth it for small local energies, so it was added as
`wp.JetSpace2` (scalar payload only). See [Second-order jets](#second-order-jets)
below. Extending it to vector/matrix payloads multiplies that quadratic state
by the component count and is left as a non-goal
([#11](https://github.com/alecjacobson/warp/issues/11)).

### Key Implementation Details

Everything lives in `warp/_src/jet.py`, exported as `wp.JetSpace`.

**Registration.** `_register(name, fn)` appends each of `fn`'s signatures to
`warp._src.context.builtin_functions[name].overloads`. A Warp function defined
several times under one name accumulates its signatures in `user_overloads`;
builtin overload resolution matches against a function's own `input_types` and
does not recurse into that dict, so each signature is appended individually. A
`@wp.func` appended to a builtin's overload list is returned by `resolve_func`
like any other match and generates a normal user-function call with its
adjoint, which is what makes R2 work.

Registered names: `add`, `sub`, `mul`, `div`, `pow`, `neg`, `pos`, `sin`,
`cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `log`, `sqrt`, `abs`,
`sign`, `min`, `max`, `clamp`, `where`, `extract`, `dot`, `length`,
`length_sq`, `normalize`, `cross`, `transpose`, `determinant`, `trace`,
`inverse`. `wp.JetSpace2` registers the same scalar arithmetic, transcendental,
and branching families (up through `where`).

The value-branching builtins (`min`, `max`, `clamp`, `where`, `abs`, `sign`)
select or pass through a derivative along with the chosen value, so a piecewise
energy stays differentiable. Comparisons themselves (`a < b`) do not overload --
Warp lowers them to raw C++ operators rather than through the builtin table --
so jet code compares `.value` explicitly; making them dispatch is tracked in
[#10](https://github.com/alecjacobson/warp/issues/10).

**What stays on the namespace.** Seeding and construction (`seed`,
`seed_vec3`, `constant`, `directional_vec3`, `make_vec3`, ...) have no builtin
counterpart, so they remain namespace members. So do `perp` and `cross2`, which
Warp does not define for `vec2`; adding them as builtin overloads would give
jets a surface Warp itself lacks.

**Global mutation.** Registration mutates a process-global table. It is
additive, happens once per `(width, dtype)` because `JetSpace` caches, and
appends after the existing overloads so builtin resolution order is unchanged.
Each `_make_jet_space` call produces fresh `Function` objects, so widths do not
interfere. The cost is that overload lists grow: `mul` already carries ~180
overloads and each jet space adds roughly a dozen, which lengthens a linear
scan during code generation only.

**No deferred annotations.** `warp/_src/jet.py` deliberately omits
`from __future__ import annotations`. Warp resolves struct annotations with
`inspect.get_annotations(eval_str=True)`, which evaluates them against module
globals; deferred annotations would turn the locally-generated types into
strings that no longer resolve.

### Usage

Seeding the identity makes one forward pass yield the whole local gradient:

```python
J = wp.JetSpace(2)

@wp.func
def local_energy(a: J.scalar, b: J.scalar) -> J.scalar:
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)

@wp.kernel
def local_gradient(z: wp.array2d[float], grad_g: wp.array[J.coeff]):
    i = wp.tid()
    grad_g[i] = local_energy(J.seed(z[i, 0], 0), J.seed(z[i, 1], 1)).coeff
```

Taping that launch and running one backward pass per gradient component gives
the local Hessian a row at a time:

```python
tape.backward(grads={grad_g: seed_row})
# z.grad[i,b] = d grad_g[i,row] / d z[i,b] = H_i[row,b]
```

### Matrix jets

Square (`mat2`, `mat3`) and rectangular (`mat32`, `mat23`) matrix jets let a
deformation gradient be a first-class jet value, so an elastic energy written in
terms of `F`, `S = FᵀF`, `det F`, and `tr S` differentiates through the whole
chain without hand-derivatives. Overloads cover `transpose`, `trace`,
`determinant`, matmul, and `inverse`. Inverse uses the closed-form differential
`d(A⁻¹) = -A⁻¹ (dA) A⁻¹`, evaluated per direction from `A⁻¹` computed once. A
`mat3` deformation gradient is built column-by-column from three seeded `vec3`
endpoints (`make_mat3`); the rectangular pair supports triangles, whose `3×2`
`F` and `S = FᵀF` come from `make_mat32` and the `(2×3)(3×2)` product.

These reuse the native `wp.inverse`/`wp.determinant` on the `value` and only
add the derivative bookkeeping, so they are cheap relative to the transcendental
overloads.

### Second-order jets

`wp.JetSpace2(width, dtype)` carries a value, a length-`width` gradient, and a
dense `width × width` Hessian, propagated by the second-order chain rule through
the same arithmetic, transcendental, and branching overloads as first-order
jets. One forward evaluation of a scalar energy yields its local Hessian with
**no tape** and no reverse sweep at all.

That is one of several ways to reach a local Hessian with this machinery, and
which one wins is energy-dependent:

| Route | Mechanism | Cost |
| ----- | --------- | ---- |
| Reverse-over-jet, tape | width-`k` forward jet for the gradient, then `k` `tape.backward()` sweeps for the rows | `k` backward launches; needs a global `requires_grad` array |
| Reverse-over-jet, in-kernel | width-1 dual for each directional derivative, reverse sweep taken *in the kernel* with `wp.grad` | one launch, register-resident, no tape and no global `.grad` |
| Pure forward | second-order jet | one launch, no reverse at all; `O(k²)` state in registers |

The in-kernel `wp.grad` route matters because it does not force a choice between
tape overhead and second-order jets. It stays local -- each element's Hessian is
computed in registers and scattered out -- so it composes with sparse assembly
the same way the pure-forward route does, without `JetSpace2`'s compile cost.

That compile cost is the main reason to prefer it. A second-order jet holds an
`O(k²)` Hessian in registers through every intermediate, and NVCC's compile time
is super-linear in `k`. How much that bites depends on the energy: for the short
elasticity kernels in `benchmark_jet_hessian_mesh.py` a tet (`k=12`) second-order
jet compiles in about 5 s and is the fastest route at runtime, while the longer
scalar-form energies in `benchmark_element_hessian.py` push the same `k=12` case
to roughly 2 minutes on a cold cache. When compile time dominates -- during
iteration, or for a wide energy -- in-kernel `wp.grad` over first-order jets gets
the same Hessian with none of that cost. The tape route remains useful on CPU and
for long chains.

The payload is **scalar only**. Each intermediate already carries an `O(width²)`
Hessian; making the payload a vector or matrix multiplies that by the component
count (`9×` for a `mat3`), which is a large compile-time and register cost for a
rare use case, so it is deferred
([#11](https://github.com/alecjacobson/warp/issues/11)). Vector-valued second
derivatives, where genuinely needed, decompose into per-component scalar jets.

## Coverage

First-order (`wp.JetSpace`) and second-order (`wp.JetSpace2`) support, by
operation family. "FD" marks families verified against finite differences in
`test_jet_ops.py`; the rest are covered by closed-form checks in `test_jet.py`.

| Family                                                    | 1st-order scalar | 1st-order vec2/vec3 | 1st-order matrix        | 2nd-order scalar |
| --------------------------------------------------------- | :--------------: | :-----------------: | :---------------------: | :--------------: |
| Arithmetic (`+ - * / ** neg pos`)                         |        ✓         |          ✓          |            ✓            |        ✓         |
| Transcendental (`sin cos tan exp log sqrt`)               |        ✓         |          –          |            –            |        ✓         |
| Inverse-trig (`asin acos atan atan2`) — FD                |        ✓         |          –          |            –            |        ✓         |
| Branching (`min max clamp where abs sign`) — FD           |        ✓         |          –          |            –            |        ✓         |
| Indexing (`v[i]`, `extract`)                              |        ✓         |          ✓          |            ✓            |        –         |
| Geometry (`dot length length_sq normalize cross`)         |        ✓         |          ✓          |            –            |        –         |
| Matrix (`transpose trace determinant inverse`, matmul)    |        –         |          –          |    ✓ (mat2/3/32/23)     |        –         |
| Reverse-over-jet Hessian via `wp.Tape`                    |        ✓         |          ✓          |            ✓            |     n/a          |
| Reverse-over-jet Hessian via in-kernel `wp.grad` (no tape) |        ✓         |          ✓          |            ✓            |     n/a          |
| Pure-forward Hessian (no tape, no reverse)                |       n/a        |         n/a         |           n/a           |        ✓         |

Not covered (tracked as issues): second-order vector/matrix jets
([#11](https://github.com/alecjacobson/warp/issues/11)); a native symmetric
matrix type for the Hessian ([#5](https://github.com/alecjacobson/warp/issues/5));
overloaded comparisons ([#10](https://github.com/alecjacobson/warp/issues/10));
quaternion jets.

## Testing Strategy

Two modules, both registered in `default_suite`:

**`warp/tests/test_jet.py`** (`TestJet`, `TestJetSpace`). Device-parametrized
tests via `add_function_test`, plus a fixed-device `TestJetSpace` for namespace
behavior. The Hessian is checked two ways against
`g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)`: a closed-form NumPy Hessian, and float64
second differences of `g`. The second derives nothing by hand, so it catches an
error in the closed form; the first is exact, so it does not depend on a step
size. Symmetry is asserted separately, since the two off-diagonals come from
separate backward passes and agreement is not automatic. Other cases: the
forward value and gradient; `vec3` geometry through `wp.length` on a spring
energy with a known closed-form gradient; `v[i]`, `wp.dot`, `wp.cross`,
`wp.normalize`, `wp.length_sq`; two widths coexisting, asserting the first still
resolves after the second registers; a `float64` space, asserted to `1e-12`; and
`JetSpace` caching and width validation.

**`warp/tests/test_jet_ops.py`** (`TestJetOps`, `TestJetMatrix`). Finite-
difference checks for the wider op surface: inverse-trig, `pow` variants, and
value-branching builtins for both first- and second-order jets (gradient and, for
second order, the Hessian); and matrix jets -- `mat2` determinant/trace, a `mat3`
tetrahedron elastic energy, `mat3` inverse, and a rectangular `mat32` triangle
energy. These are **CPU-only** (`DEVICES = ["cpu"]`, standard `unittest.TestCase`
rather than `add_function_test`): building one module for both CPU and CUDA in the
same process trips a module-hasher instability where the second device's build
perturbs shared hash state, so a kernel's symbol is looked up under a hash that
differs from the one it compiled with. Single device keeps the hashes consistent;
CUDA correctness of these ops is covered by the finite-difference gates in the jet
benchmarks.

## Benchmarks

Five benchmarks under `warp/examples/benchmarks/` accompany the feature, each
gating its strategies against finite differences before timing (which is also
the CUDA correctness coverage for the CPU-only `test_jet_ops.py` cases):

- `benchmark_jet_gradient.py` / `benchmark_jet_gradient_mesh.py` -- first-order
  gradient throughput as width `k` and element count scale, contrasting a
  width-`k` jet pass, a `wp.Tape`, and in-kernel `wp.grad`.
- `benchmark_jet_hessian.py` / `benchmark_jet_hessian_mesh.py` -- local-Hessian
  strategies: width-`k` reverse-over-jet tape, width-1 tape, in-kernel width-1
  `wp.grad`, and the pure-forward second-order jet, on real spring/triangle/
  tetrahedron energies.
- `benchmark_element_hessian.py` -- the same spring/triangle/tetrahedron
  energies written once as generic scalar `wp.func`s, so one definition feeds
  every gradient and Hessian strategy. `--forward2-max-k` skips the
  second-order jet above a chosen width, which is what makes the compile-time
  trade-off discussed under *Second-order jets* measurable rather than
  anecdotal.
