# Forward-Mode Jets

**Status**: In Progress

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

This is a different mechanism from the `wp.summand` / `wp.indexed_sum` work in
[`sparse-hessians.md`](sparse-hessians.md), which asks users to supply
derivatives via `@wp.summand_grad` / `@wp.summand_hessian`. Jets are
complementary: they remove the need to write those derivatives by hand.

## Requirements

| ID  | Requirement                                                                       | Priority | Notes                                            |
| --- | --------------------------------------------------------------------------------- | -------- | ------------------------------------------------ |
| R1  | Evaluating a `@wp.func` over jets yields its gradient with no hand-differentiation | Must     |                                                  |
| R2  | `wp.Tape` differentiates through jet code, giving reverse-over-forward Hessians    | Must     |                                                  |
| R3  | Ordinary Warp syntax: `a * b`, `wp.sin(a)`, `wp.length(d)`, `v[0]`                 | Must     | No names bound into the caller's module          |
| R4  | Derivative width independent of geometric dimension                                | Must     | 6 directions over two `vec3` endpoints           |
| R5  | Several widths and dtypes coexist in one process                                   | Should   |                                                  |
| R6  | Geometry helpers: `dot`, `length`, `normalize`, `cross`                            | Should   |                                                  |

**Non-goals**: second-order jets (jet-of-jet); matrix or quaternion jet types;
assembling the sparse global Hessian, which is what `wp.indexed_sum` covers;
choosing a sparsity pattern or a linear solver.

## Design

### Approach

`wp.JetSpace(width, dtype)` specializes a family of types at
code-generation time and returns them in a namespace:

| Name     | `value`        | `coeff`             |
| -------- | -------------- | ------------------- |
| `scalar` | `dtype`        | `vector(width)`     |
| `vec2`   | `vector(2)`    | `matrix(2, width)`  |
| `vec3`   | `vector(3)`    | `matrix(3, width)`  |

Each is a `@wp.struct`. The arithmetic is generated as ordinary `@wp.func`
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

**Second-order jets (jet-of-jet)** would give the Hessian in one forward pass
with no tape at all. It squares the coefficient storage and does not reuse
Warp's autodiff, so it is left as a non-goal.

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
`cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `extract`, `dot`, `length`,
`length_sq`, `normalize`, `cross`.

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

## Choosing a Width: k-Wide Versus k Hessian-Vector Products

A `k x k` local Hessian can be assembled two ways, and `JetSpace` deliberately
prescribes neither. Which is faster depends on the device, and the ranking
reverses between CPU and GPU.

**width-k.** One `k`-wide forward pass yields all of `grad g`. Then `k` reverse
sweeps through that widened program extract its `k x k` Jacobian, a row at a
time.

**width-1.** For each basis direction `e_j`, a constant-width forward pass
yields the scalar `Dg[e_j] = grad g . e_j`, and one reverse sweep over that
scalar yields `grad_z (grad g . e_j) = H e_j` -- column `j`, in one shot. This
is the usual "a Hessian is `k` Hessian-vector products" identity.

### Cost model

Write `C` for the primal cost and `T` for the tangent work per direction.

|         | forward       | reverse         | launches | reads of `z` |
| ------- | ------------- | --------------- | -------- | ------------ |
| width-k | `C + kT`      | `kC + k^2 T`    | `k + 1`  | 1            |
| width-1 | `k(C + T)`    | `kC + kT`       | `2k`     | `k`          |

The two trade against each other. Width-k wins the forward: it reads `z` once
and computes the primal once, carrying `k` tangents alongside, where width-1
re-reads and recomputes `k` separate times. Width-k loses the reverse: each of
its `k` sweeps differentiates a program that is itself `k` wide, giving the
`k^2 T` term.

### Measurements

Splitting the two halves apart on CPU (Apple M4, m=20000, milliseconds):

| k  | strategy | forward | reverse | total  |
| -- | -------- | ------- | ------- | ------ |
| 4  | width-k  | 0.18    | 2.35    | 2.54   |
| 4  | width-1  | 0.65    | 1.86    | 2.51   |
| 8  | width-k  | 0.45    | 52.85   | 53.30  |
| 8  | width-1  | 2.28    | 7.08    | 9.35   |
| 16 | width-k  | 1.36    | 514.65  | 516.01 |
| 16 | width-1  | 9.02    | 31.91   | 40.93  |

Both predicted effects appear. Width-k's forward advantage grows with `k`
(3.6x, 5.1x, 6.6x) and its reverse penalty grows faster (1.3x, 7.5x, 16.1x).
At k=16 the reverse is 99.7% of its total, so the reverse decides the outcome.

End to end, the winner depends on the device:

| k  | L40, m=500000        | x86_64 CPU, m=50000   |
| -- | -------------------- | --------------------- |
| 8  | width-k 3.0x faster  | width-1 5.3x faster   |
| 16 | width-k 2.7x faster  | width-1 10.3x faster  |

### Why the ranking reverses

**A CPU charges for FLOPs.** A Warp CPU kernel walks the elements on one
thread, so the `k^2 T` arithmetic is serialized and lands directly in wall
clock. The signature is exact: the width-1 advantage is 5.3x at k=8 and 10.3x
at k=16, doubling as `k` doubles, which is `k^2 T / kT = k`. The operation
count predicts the CPU result precisely.

**A GPU charges for bytes and launches.** With 500000 local terms there are
half a million threads in flight, so width-k's extra arithmetic is
latency-hidden -- it costs occupancy rather than time, as long as the `k`
tangents fit in registers. What is left to pay for is width-1's redundancy: it
reads `z` `k` times (`m k^2` words against `m k`, which is 32 MiB per forward
set at k=16, streamed from HBM rather than sitting in cache as the 1.28 MiB
CPU working set does) and issues `2k` launches against `k + 1`.

The GPU ratio corroborates this. It is flat, even shrinking -- 3.0x at k=8,
2.7x at k=16. A compute-bound device would show it falling like `1/k`, as the
CPU does. It does not, so `k^2 T` is not what is being charged.

At smaller `m` the GPU picture is launch-bound instead: at m=50000 width-k
leads by a flat 1.2-1.4x across k=2..16, close to the `2k / (k + 1)` launch
ratio.

### Open question

Width-k's GPU advantage should survive only while `k` tangents fit in registers
with enough occupancy left to hide the arithmetic. At k=16 that is comfortable
on an L40 (255 registers per thread); at k=32 it is roughly double, occupancy
drops, and the hidden `k^2 T` should become visible. The advantage already
slipped from 3.0x to 2.7x between k=8 and k=16, so the crossover is plausibly
near k=32 and likely past by k=64. That measurement has not been taken.

Running k=32 at both m=500000 and m=50000 would isolate it: if the cause is
occupancy, the large-`m` case degrades more, since that is the one relying on
many resident warps.

### A loop-structure hazard

Local energies are often written as a loop over `k` terms carrying a jet
accumulator. Warp unrolls a loop only up to `max_unroll` (default 16); past
that it stays dynamic, and **reverse mode through a dynamic loop carrying a
seeded jet returns wrong gradients with no diagnostic**. An accumulator alone
is fine; the hazard needs a loop-carried jet whose coefficients were seeded per
iteration.

Warp applies the limit per loop rather than to the total trip count, so
strip-mining a `k`-term chain into two loops of about `sqrt(k)` keeps every `k`
up to `max_unroll^2 = 256` straight-line without touching module options. At
k=100 a 10x10 pair generates 11405 lines against 869 for the rolled form, and a
bounds guard for non-factorable `k` does not prevent unrolling (k=50 unrolls as
7x8). `benchmark_jet_hessian.py` does this, and gates every configuration
against a float64 finite-difference reference before timing it.

This concerns user code that loops over jets. The jet types themselves are
correct in both passes well past this threshold.

## Testing Strategy

`warp/tests/test_jet.py`, registered in `default_suite`. Device-parametrized
tests via `add_function_test`, plus a fixed-device `TestJetSpace` for
namespace behavior.

The Hessian is checked two ways against `g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)`:
a closed-form NumPy Hessian, and float64 second differences of `g`. The second
derives nothing by hand, so it catches an error in the closed form; the first
is exact, so it does not depend on a step size. Symmetry is asserted
separately, since the two off-diagonals come from separate backward passes and
agreement is not automatic.

Other cases: the forward value and gradient; `vec3` geometry through
`wp.length` on a spring energy with a known closed-form gradient; `v[i]`,
`wp.dot`, `wp.cross`, `wp.normalize`, `wp.length_sq`; two widths coexisting,
asserting the first still resolves after the second registers; a `float64`
space, asserted to `1e-12`; and `JetSpace` caching and width validation.
