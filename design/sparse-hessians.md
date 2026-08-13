# Sparse Hessians and Jacobians

**Status**: In Progress

**Issue**: [GH-1767](https://github.com/NVIDIA/warp/issues/1767)

## Implementation status

An initial MVP of the ``@wp.summand`` / ``wp.indexed_sum`` API is implemented in
``warp/_src/summand.py`` (re-exported as ``wp.summand``, ``wp.indexed_sum``).

Done:

* Manual derivatives via ``@wp.summand_grad`` / ``@wp.summand_hessian`` using the
  argument-indexed sparse-dict convention (grad ``{i: dE/d(arg_i)}``, Hessian
  ``{(i, j): block}``, upper triangle only). The dict bodies are read from the
  AST and compiled into strict-typed ``wp.func``\ s per derivative request (the
  factory model in ``warp/_src/summand.py``); a ``wp.func`` never literally
  returns a Python dict.
* Forward evaluation of the summed energy (``value.value``), plus assembly of
  ``value.hessian[positions, positions]`` into a ``mat33``
  ``warp.sparse.BsrMatrix`` and ``value.gradient[positions]`` into a ``vec3``
  array, for a single ``wp.array(dtype=wp.vec3)`` variable over 1-, 2-, or
  3-node stencils. Assembly kernels are cached per (kind, arity).
* ``value.vjp(positions, seed)`` -- the vector-Jacobian product ``seed * dE/dx``
  for the scalar energy (so ``gradient[x] == vjp(x, 1.0)``, and ``vjp(x, -1.0)``
  is a Newton right-hand side in one pass). ``seed`` is the scalar output
  cotangent; it generalizes to a vector cotangent for the vector-valued
  (Jacobian) track.
* One non-differentiable per-element scalar parameter (e.g. a spring rest
  length) supplied through a second index array: ``wp.indexed_sum(energy,
  (edges, edge_ids))`` bound with ``total(positions, rest_lengths)``.
* Sparsity-preserving composition: ``a + b`` and ``scale * a`` build a weighted
  sum whose value/gradient/vjp/Hessian are the sums of the terms (Hessian by
  union sparsity; terms without a registered Hessian, e.g. a linear gravity
  potential, contribute zero). See ``optim/example_catenary.py``.
* A finite-difference backend: ``wp.indexed_sum(..., backend='fd')`` computes the
  gradient and Hessian by naive central differences of the energy, so no
  ``@wp.summand_grad`` / ``@wp.summand_hessian`` rules are needed (1- and 2-node
  stencils, 0 or 1 parameter). See ``optim/example_catenary_fd.py``.
* Hessian-vector products for free via ``warp.sparse.bsr_mv``.
* NumPy reference oracles with finite-difference Hessian and gradient checks
  (``warp/tests/summand_references.py``) and assembly/HVP tests
  (``warp/tests/test_summand.py``).
* Examples: ``warp/examples/optim/example_spring_hessian.py`` (zero-rest Newton)
  and ``warp/examples/optim/example_catenary.py`` (rest-length springs plus
  gravity, exercising the parameter and composition features).

Tested on both CPU and CUDA (NVIDIA L40, sm_89).

Not yet implemented (see the design below and the tracking issue): a jets
(forward-mode) backend, more than one per-element parameter and mixed element
dimensions, multi-variable off-diagonal blocks and the storage/return model for
a mixed-dtype Hessian (manifest per-pair BSRs vs. a single scalar matrix with
views -- still open), PSD projection, and the Jacobian track. The
finite-difference backend so far covers only 1- and 2-node stencils.

## Motivation

This document tracks related work and open design questions for dealing with
(sparse) Hessians in Warp.

Consider functions of the form

```
f(x) = Σᵢ₌₀ᵏ gᵢ(Sᵢ x)
```

where each local summand function `gᵢ: ℝᵐ → ℝ` and `Sᵢ ∈ ℝᵐˣⁿ` selects the
relevant local variables (and let `xᵢ = Sᵢ x`).

We're interested in efficiently evaluating the Hessian, which has the sparse
form

```
∂²f/∂x² = Σᵢ₌₀ᵏ Sᵢᵀ ∂²gᵢ/∂xᵢ² Sᵢ,
```

where `xᵢ = Sᵢ x` and each `∂²gᵢ/∂xᵢ² ∈ ℝᵐˣᵐ` is in general dense. The
assembled global matrix is sparse even though each local block is dense.

## Requirements

| ID  | Requirement                                                                                     | Priority | Notes                                        |
| --- | ----------------------------------------------------------------------------------------------- | -------- | -------------------------------------------- |
| R1  | Assemble sparse Hessians from local summands into `warp.BsrMatrix`                              | Must     | Natural block layout for `wp.array[wp.vec3]` |
| R2  | Dictionary-style interface for Hessian/Jacobian blocks and variable orderings                   | Must     | `tbd.hessian[a, b]`, `tbd.jacobian[(a, b)]`  |
| R3  | Support non-differentiable inputs, mixed element dims, and differing indexing per parameter     | Must     | `hessian_variable=False`                     |
| R4  | Sparsity-preserving composition of summand terms                                                | Should   | Union sparsity via an expression tree        |
| R5  | Per-term PSD projection for Newton's method                                                     | Should   | `psd_projection='abs'` / thresholding        |
| R6  | Matrix-free Hessian/Jacobian products (HVP, JVP, VJP)                                           | Should   | For iterative solvers                        |
| R7  | Swappable differentiation backends                                                              | Could    | Hand-written, jets, sympy, LLM, finite diff  |
| R8  | Variadic local input sizes                                                                      | Could    | e.g. mixed quad/triangle meshes              |

**Non-goals**: Operations that completely destroy Hessian sparsity (e.g.
`value**2`) will probably be forever disallowed.

## Design

### Sparsity requires thinking about layout

Assume `f(a, b)` is a scalar function of vector/tensor-valued `a` and `b`
variables. For a typical `f` in FEM simulation, the gradients will be dense
(`∂f/∂a` will be the same shape as `a` and in general all entries will be
non-zero). Using Warp we can collect these for a scalar loss `f` on the tape:

```python
with tape:
    warp.launch(kernel=loss_kernel, inputs=[a, b, f], device="cuda")

tape.backward(loss_value)
# tape.gradients is a dictionary
dfda = tape.gradients[a]
# dfda, dfdb are the same shape as a and b
dfdb = tape.gradients[b]
```

Because `∂f/∂a` is dense and the same shape as `a`, it can use the same storage
as `a`: a dense array. For example, if `a` is `wp.array[wp.vec3]` then `dfda`
can also be `wp.array[wp.vec3]`.

But often the Hessian matrix will be sparse; specifically, each "block"
`∂²f/∂a²`, `∂²f/∂a∂b`, `∂²f/∂b²` will be sparse. Realizing a dense
`wp.array[wp.array[...]]` is out of the question.

The natural sparse matrix format will be `warp.BsrMatrix`, which has a natural
layout for a given block `∂²f/∂a∂b` when `a` and `b` are `wp.array[wp.vec3]` →
each block `ij` corresponds to `∂²f/∂aᵢ∂bⱼ` where `aᵢ` and `bⱼ ∈ ℝ³`.

If we want to manifest the entire Hessian with respect to all variables `H`,
then we need to decide/specify which order variables' derivatives will be
written; does `H = ∂²f/∂[ab]²` or `H = ∂²f/∂[ba]²`? It's natural to let the user
specify this and return blocks as thin "views" or reorderings onto the entire
matrix built using a default order. An alternative would be to manifest separate
`warp.BsrMatrix`'s for any requested block.

Regardless, we would like to have a dictionary-type interface:

```python
# tbd is a to-be-determined object exposing hessian as a dictionary
d2fda2 = tbd.hessian[a, a]
d2fdab = tbd.hessian[a, b]
d2fdba = tbd.hessian[b, a]
d2fdbb = tbd.hessian[b, b]
# passing tuples for either key specifies a multi-variable ordering
H = tbd.hessian[(a, b), (a, b)]
```

### Sparse Jacobians

For non-linear least squares problems we may have losses of a similar form:

```
f(x) = Σᵢ₌₀ᵏ ‖ Gᵢ(Sᵢ x) ‖² = ‖ G(x) ‖² = ‖ ∑ᵢ Rᵢ Gᵢ(Sᵢ x) ‖²
```

where `Gᵢ: ℝᵐ → ℝᵛ` is vector-valued and `Rᵢ` scatters each local summand's
pre-squared-norm value into slots in a global vector `G`.

Similarly, we may have non-linear constraints of the same summation form:

```
G(x) = ∑ᵢ Rᵢ Gᵢ(Sᵢ x) ≥ 0
```

In both cases, we're interested in evaluating the Jacobian matrix, which has the
sparse form

```
∂G/∂x = ∑ᵢ Rᵢ ∂Gᵢ/∂xᵢ Sᵢ
```

where each `∂Gᵢ/∂xᵢ ∈ ℝᵛˣᵐ` is dense.

The same layout discussion applies for sparse blocks, with an analogous ideal
dictionary-based interface:

```python
dGda = tbd.jacobian[a]
dGdb = tbd.jacobian[b]
J = tbd.jacobian[(a, b)]
```

### Alternatives Considered (Related work)

**`warp.fem`** — Supports Gauss-Newton (already PSD) sparse Hessians, but we
must write the summand already in bilinear form. The workflow is designed for
linearized PDE solving, i.e., directly on the Euler-Lagrange equation of a
variational problem rather than by specifying a loss to optimize. This doesn't
hurt generality, but it may be uncomfortable to users expecting a loss
minimization workflow.

**TinyAD** — C++ targeting CPU. Forward-mode, fixed-size jets. Some low-level
support for extracting subblocks efficiently. Supports PSD-projected Hessians.

**`alecjacobson/indexed_sum`** — Pure PyTorch via `torch.func.hessian`
(forward-over-reverse) and `torch.func.vmap`. Still a lot on the table to
optimize, including `torch.compile` for kernel fusion and BSR/CSR with caching.
Supports PSD-projected Hessians.

### Approach: `@wp.summand` and `wp.indexed_sum`

`@wp.summand` is similar to `@wp.fem.integrand` but without assuming
FEM-specific function spaces/quadrature. Ideally, `@wp.fem.integrand` could wrap
around this more general `@wp.summand`.

Let's first consider an easier case where all the variables are in
`x: wp.array[wp.vec3]`.

```python
positions = wp.array((num_points,), dtype=wp.vec3)
edges = wp.array((num_edges,), dtype=wp.vec2i)


@wp.summand
def zero_rest_length_spring_energy(
    p0: wp.vec3,
    p1: wp.vec3,
) -> float:
    return 0.5 * (wp.length(p0 - p1) ** 2)


# create indexed_sum object with index information
total_energy_function = wp.indexed_sum(zero_rest_length_spring_energy, edges)
# invoke forward pass with variable data
value = total_energy_function(positions)
# construct sparse hessian
H = value.hessian[positions, positions]
```

Using the same `positions` and `edges` arrays, a serial implementation computing
the same value might look like:

```python
value = 0.0
for tid in range(0, num_edges):
    i0, i1 = edges[tid]
    value += zero_rest_length_spring_energy(
        positions[i0],
        positions[i1],
    )
```

In the previous spring example, all of the `zero_rest_length_spring_energy`
inputs were differentiation variables, they had the same dimension `vec3`, and
were indexed by the same row of indices `edges`.

```python
positions = wp.array((num_points,), dtype=wp.vec3)
# tag as non-differentiable
rest_lengths = wp.array((num_edges,), dtype=wp.float, hessian_variable=False)
edges = wp.array((num_edges,), dtype=wp.vec2i)
edge_ids = wp.array(wp.arange(0, num_edges), dtype=int32)


@wp.summand
def spring_energy(
    p0: wp.vec3,
    p1: wp.vec3,
    r: float,
) -> float:
    return 0.5 * (wp.length(p0 - p1) - r) ** 2


# pass indexing objects corresponding to ordered parameters to local_function
total_energy_function = wp.indexed_sum(spring_energy, (edges, edge_ids))
# pass variable arrays ordered to match indexing objects
value = total_energy_function(positions, rest_lengths)
d2value_dpositions2 = value.hessian[positions, positions]
# fails unless we change hessian_variable=True for rest_lengths above
d2value_drest_lengths2 = value.hessian[rest_lengths, rest_lengths]
```

Now, the new array of data `rest_lengths` is constant w.r.t. the Hessian, has a
different element dimension from the positions (`float` vs `vec3`), and is
indexed "per-edge" rather than "per-vertex".

Using the same data arrays, a serial implementation computing this new value
might look like:

```python
value = 0.0
for tid in range(0, num_edges):
    i0, i1 = edges[tid]
    i2 = edge_ids[tid]
    value += spring_energy(
        positions[i0],
        positions[i1],
        rest_lengths[i2],
    )
```

### Sparsity-preserving compositions

Summing multiple `wp.indexed_sum` objects simply unions their sparsity patterns,
which is often fine. For example, adding multiple values created with
`indexed_sum` creates a small expression tree, so that we don't necessarily need
to manifest and sum sparse matrices explicitly.

```python
pred_positions = wp.array((num_points,), dtype=wp.vec3, hessian_variable=False)
vertex_ids = wp.array(wp.arange(0, num_points), dtype=int32)


@wp.summand
def inertia(p0: wp.vec3, pred_p0: wp.vec3):
    return 0.5 * wp.length(p0 - pred_p0) ** 2


# each sub-energy may have different variable data or indexing
total_spring_energy = wp.indexed_sum(spring_energy, (edges, edge_ids))
total_inertia_energy = wp.indexed_sum(inertia, (vertex_ids))

# constant scalar
h = 1.0 / 30.0

value = (
    total_spring_energy(positions, rest_lengths)
    + (1.0 / h**2) * total_inertia_energy(positions, pred_positions)
)

# total_spring_energy.hessian + (1.0/h**2) * total_inertia_energy.hessian
d2value_dpositions2 = value.hessian[positions, positions]
```

Operations that completely ruin sparsity of the Hessian like

```python
value_squared = value**2
```

will probably be forever disallowed.

### PSD projection

For Newton's method it's often important to project or regularize the Hessian to
be positive semi-definite (PSD). The standard practice is to project each local
dense contribution by thresholding [Teran et al. 2005] or taking absolute value
[Chen et al. 2024] of eigenvalues.

```python
# projects each term
total_spring_energy = wp.indexed_sum(spring_energy, (edges, edge_ids), psd_projection='abs')
```

This will zealously project every spring energy term even if adding it to an
inertia term later would have been enough to make the full Hessian PSD.

How to best postpone PSD projection until composition of terms is an open
question. It seems easy if the sparsity patterns/topologies are the same or one
is contained in the other, but difficult to achieve in general. Chen et al. 2024
originally started as a project trying to apply chordal decomposition post facto
to manifested matrices, but this was not fruitful.

Caching the local Hessians for reuse might improve efficiency for schemes that
first try the unprojected Hessian and then fall back to projected. For GPU, the
memory/compute tradeoff makes this an interesting hypothesis to test.

### Swappable backends

For experimentation and perhaps inevitably case-by-case tuning, the mechanism
for differentiation should be an exposed option: `wp.indexed_sum(..., backend=...)`.

1. **hand-written** → like `wp.func_grad`, allow custom overload
   - 👍 optimal performance
   - 👎 not automatic, error prone
2. **jets** → forward-over-forward autodiff
   - 👎 theoretically suboptimal `O(n³)` performance
   - 👍 for fixed sizes can be quite efficient
3. **sympy** → symbolic differentiation
   - 👎 struggles with `if` clauses and other piecewise differentiable functions
   - 👍 common subexpression elimination could be faster than autodiff
4. **codex** → analytic gradient function via LLM
   - 👎 could output wrong result (outside its testing)
   - 👍 potentially stronger than sympy with the advantages of CSE
5. **finite-difference** → last resort / testing
   - 👍 general and easy to implement (might help as a placeholder for compiled
     `wp.func`'s that don't have second derivatives implemented)
   - 👎 terrible numerics for second derivatives

#### Jet backend: status and generality gaps

Forward-mode jets are implemented as ordinary differentiable Warp types in
``warp/_src/jet.py``: ``wp.JetSpace(width, dtype)`` (first order, carrying value
plus a ``width``-vector of coefficients, with ``scalar``/``vec2``/``vec3``
variants) and ``wp.JetSpace2(width, dtype)`` (second order / forward-over-forward,
carrying value, gradient, and the full ``width×width`` Hessian; scalars only).
The arithmetic is registered as overloads of Warp's builtins, so a local energy
is written once as a plain ``@wp.func`` in ordinary operator/`wp.length`/`wp.dot`
syntax and specializes across plain floats and every jet type. Unit tests in
``warp/tests/test_jet.py`` cover both, cross-checked against analytic
derivatives, reverse-over-forward, and finite differences.

What the benchmarks (``warp/examples/benchmarks/benchmark_jet_*``) show:

* **Gradients**: a first-order jet assembling a summed-loss vertex gradient (one
  fused forward+scatter launch, no tape) is only a *modest* win over Warp's
  reverse mode on real shared-vertex meshes — ~1.1–1.7× per launch, shrinking as
  the stencil ``k`` grows (spring k=6 > triangle k=9 > tet k=12) and with element
  count, and collapsing to near parity under warm CUDA-graph replay. Reverse mode
  is already close to optimal here; jets are not a general replacement.
* **Hessians**: this is where jets earn their place. The second-order
  (``JetSpace2``) forward pass assembles the local Hessian block with no
  second-order tape and is a GPU small-``k`` specialist. The full ``k×k`` kernel
  compiles in ~4 s (k=6), ~45 s (k=9), ~2 min (k=12) — one-time, then cached —
  and is correct at all three; the practical ceiling is compile time, not
  correctness.

Two open threads recorded as tracking issues:

* A **native symmetric matrix type** with intrinsic ops. Packing the Hessian into
  the ``k(k+1)/2`` upper triangle to halve storage/compute was prototyped and made
  compile time *5–6× worse* — the packed form hand-unrolls each op into ``O(L)``
  scalar statements (6× more generated source) instead of one whole-matrix
  intrinsic. Genuinely exploiting symmetry needs Warp-core support (GH issue #5).
* **Jet type generality gaps** blocking arbitrary energies (GH issue #6): matrix
  jet types (``mat22``/``mat33`` so ``wp.determinant``/``trace``/``inverse`` work
  on jets — the biggest gap for FEM, distinct from the storage type above);
  comparisons / ``select`` / ``min`` / ``max`` / ``atan2`` for branching and angle
  energies; ``pow(jet, jet)`` and integer exponents; shared arithmetic scaffolding
  so first- and second-order don't diverge; and ``JetSpace2`` vec/`tan`/`abs`
  parity. Every change is gated on ``test_jet.py`` plus the jet benchmarks for
  correctness and performance regressions.

### HVP: Hessian-vector product

Iterative solvers don't need the Hessian materialized. They just need products.
For some of the backends, directly computing `w = H v` rather than materializing
`H` and relying on the iterative solver to call `matmul H @ v` will be faster.

```python
# materialize BSR Matrix
H = loss.hessian[x, x]
# lambda that multiplies a given vector v by H
hvp_func = lambda v: loss.hvp((x, x), v)
```

The HVP mode should compose fine with PSD projection and sparsity-preserving
compositions. This all also applies to Jacobians, which would appreciate having
JVP and VJP routines.

## Future compatibility

### Variable input sizes

So far we've assumed that the local summand takes a fixed number of inputs, both
in terms of the dimension of each parameter (e.g., `float`, `vec3`) and the
number of parameters (e.g., a face is a triangle with three nodes → three
inputs). It would be great to support variadic inputs at least up to a paddable
amount.

As a stop-gap, the initial design allows the user to dispatch on a finite set of
sizes (e.g., for a quad-dominant mesh with some triangles, split into two calls:
one over quads and one over triangles). Obviously nicer if that is automatic and
easy to write.
