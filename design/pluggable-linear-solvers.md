# Pluggable Linear Solvers

**Status**: Proposed

**Issue**: None yet

## Motivation

Warp's sparse linear solvers in `warp.optim.linear` (`cg`, `cr`, `bicgstab`,
`gmres`) are **one-shot functions**: each call takes `(A, b, x)` and runs the
entire solve from scratch. This shape has two consequences:

1. **No amortization.** There is no way to do expensive setup once and reuse it
   across many solves -- building a preconditioner, or (the big one) factoring a
   matrix once and back-substituting many right-hand sides. A metric-optimization
   or Newton loop that solves the *same operator* against many RHS, or re-solves a
   same-pattern operator every step, pays full setup every time.

2. **No seam for a direct solver.** A sparse direct solver (cuDSS on GPU, CHOLMOD
   on CPU) is fundamentally factor-once / solve-many. Its value *only* shows up
   when setup and solve are separable. With a one-shot API there is nowhere to put
   it, so Warp callers who want a direct solve must hand-roll ctypes bindings and a
   bespoke lifecycle.

A prototype on the `alecjacobson/harmonic` branch validated the fix in a narrow
setting. `warp.geometry.MetricHarmonicSolver` introduced a minimal pluggable
`LinearSolver` seam (`prepare` / `solve` / `solve_many`) with two backends: the
built-in Jacobi-preconditioned CG (default, no extra dependencies) and an optional
cuDSS direct backend (via the external `warp-cudss` package). On an ill-conditioned
3-D quasi-harmonic problem the direct backend, driven through this seam and replayed
in a CUDA graph, ran roughly an order of magnitude faster per iteration than CG and
far faster than a CPU reference. The seam itself -- not the specific geometry code --
is the reusable idea.

This document proposes promoting that seam into a Warp-wide abstraction so that
`warp.optim.linear` (and consumers like `warp.fem`, `warp.geometry`, user code)
can **use the built-in iterative solvers by default and swap in a direct backend
such as cuDSS when it is available**, without changing call sites.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | A stateful solver object with a `prepare(A)` / `solve(b, x)` lifecycle that both iterative and direct solvers satisfy | Must | The existing one-shot `cg(A,b,x)` etc. become thin wrappers over it |
| R2  | Built-in iterative backends (CG/CR/BiCGStab/GMRES) are the default and require no optional dependencies | Must | "Works out of the box" |
| R3  | Optional direct backends (cuDSS now; CHOLMOD/cuSolverSp later) register only when their runtime library is available; absence is graceful | Must | No hard dependency; `try`-load, not import error |
| R4  | A backend declares its capabilities; callers/factory negotiate rather than assume | Must | matrix type (SPD/sym/general), dtype, multi-RHS, matrix-free vs assembled, capture-safety |
| R5  | Multi-RHS: `solve` accepts a single or batched RHS; backends that can batch do, others fall back to a per-column loop provided by the base class | Must | Batched direct solve was a 2-3x win in the prototype |
| R6  | Separable, capture-aware phases: symbolic setup is out of graph capture; numeric refresh and solve are capturable | Must | The prototype's entire perf story depends on this |
| R7  | Reuse of the symbolic/pattern setup across solves where only values change | Should | The dominant one-time cost for direct solvers; keyed on sparsity pattern, not values |
| R8  | A factory that selects a backend by name, with an `"auto"` default that prefers a direct backend only when one is available and applicable, else the iterative default | Should | `"auto"` policy is intentionally simple in v1 (see Non-goals) |
| R9  | Shared BSR to scalar-CSR expansion utility in `warp.sparse` | Should | Direct solvers consume scalar CSR; Warp assembles BSR |
| R10 | Automatic memory-bounded chunking of multi-RHS solves | Could | A large 3-D factor plus a wide solve workspace can exhaust device memory |

**Non-goals**:

- **Smart automatic solver selection.** Choosing direct vs iterative optimally
  requires knowing conditioning and fill-in, which are not cheaply knowable a
  priori. v1's `"auto"` is a coarse availability/applicability check, not a
  cost model. Selection policy can evolve on top of the stable lifecycle later.
- **Constraint handling.** Dirichlet elimination, KKT/saddle systems, penalty
  methods, and index reduction stay in the *callers* (`warp.fem`,
  `warp.geometry`, user code). The solver abstraction is a pure `A x = b`.
- **Nonlinear / optimization solvers.** Out of scope; this is the inner linear
  solve those build on.
- **Distributed / multi-GPU solves.** Out of scope for v1.

## Design

### Approach

Introduce a `LinearSolver` protocol in `warp.optim.linear` with a
**prepare / solve** lifecycle and a small **capability record**. Wrap the existing
iterative algorithms as backends implementing it, add an optional direct backend,
and provide a factory that resolves a backend by name with a conservative `"auto"`.

Interface sketch (illustrative, not final):

```python
class LinearSolver(Protocol):
    # --- capabilities (class-level, queried by the factory) ---
    requires_assembled_matrix: bool  # direct: True; iterative: False (matrix-free OK)
    supports_multi_rhs: bool  # True => override solve() for batched RHS
    supported_matrix_types: frozenset  # subset of {"spd", "symmetric", "general"}
    supported_dtypes: frozenset  # {wp.float32, wp.float64, ...}

    def prepare(self, A, *, matrix_type="spd", rhs_width=1) -> None:
        """Set up for operator A. Iterative: wrap + build preconditioner.
        Direct: (re)factor. May allocate / synchronize -> NOT graph-capturable.
        Re-runs the symbolic phase only if A's sparsity pattern changed (R7)."""

    def solve(self, b, x) -> None:
        """Solve A x = b in place. b, x are (n,) or (n, k). prepare() ran first.
        Capturable: only device work, stable buffers."""
```

The existing functions become wrappers, preserving backward compatibility:

```python
def cg(A, b, x, **kw):  # unchanged public signature
    s = CgSolver(**kw)
    s.prepare(A)
    s.solve(b, x)
```

Backend selection:

```python
make_solver(A, kind="auto", *, matrix_type="spd", rhs_width=1) -> LinearSolver
# kind: "cg" | "cr" | "gmres" | "bicgstab" | "cudss" | "direct" | "auto"
#   "auto"   -> a direct backend iff one is available AND A is assembled AND
#               matrix_type is supported; otherwise the iterative default (CG+Jacobi)
#   "direct" -> best available direct backend, or a clear error if none loaded
#   explicit named backend -> that backend, or a clear error if unavailable/inapplicable
```

The default remains the built-in iterative path. cuDSS (or any direct backend)
is a *swap*, selected by name or picked up by `"auto"` only when its shared
library loads successfully.

### Relationship to existing `warp.optim.linear` classes

This is a generalization of existing structure, **not a parallel hierarchy**.
`warp.optim.linear` already has most of the iterative half:

- **`LinearSolverState` + `CG` / `CR` / `BiCGSTAB` / `GMRES`.** These are already
  stateful, pre-allocated, re-runnable, CUDA-graph-capturable iterative solvers:
  construction allocates all temporary buffers for a fixed `(A, b, x, M)` layout,
  and `__call__(A=None, b=None, x=None, M=None)` re-runs the solver, optionally
  substituting shape/dtype/device/batch-compatible operands, with an internal
  `use_cuda_graph` capture of the iteration. A reusable instance is obtained today
  via the subclasses or `cg(..., run=False)`. So the "amortize setup, reuse buffers,
  capture the inner loop" idea (R1, R6) *already exists for iterative solvers* -- the
  new protocol must build on it, not replace it.

- **`LinearOperator` / `aslinearoperator`.** The matrix-free operator the iterative
  path consumes (any `A` that supports a matvec). Reused unchanged. The
  `requires_assembled_matrix` capability (R4) is exactly the gate that forbids a
  direct backend from accepting a bare `LinearOperator`.

- **`preconditioner(A, ptype)`.** Builds `M` as a `LinearOperator`, supplied to the
  solver as an operand.

The reconciliation, and where the existing model falls short for a direct backend:

- **Lifecycle mapping.** The `LinearSolver.prepare`/`solve` split maps onto the
  existing classes as `prepare` ~= constructing the state (allocate buffers, and --
  open question below -- build the preconditioner), `solve` ~= `__call__`. For
  `CG`/`CR`/`BiCGSTAB`/`GMRES` this is a near-mechanical refactor (or a thin adapter
  that wraps a `LinearSolverState` as the iterative backend). The question of whether
  `LinearSolver` *is* a refactor of `LinearSolverState` or an adapter *over* it is the
  main integration decision; the refactor is cleaner, the adapter is less disruptive
  to existing call sites.

- **Why a single `__call__` is insufficient.** `LinearSolverState.__call__`
  *re-runs the full iteration* each call; there is no separable factor. A direct
  backend needs `prepare` = factor (expensive, once) and `solve` = back-substitute
  (cheap, many). Expressed in today's model, "substitute `A` and call again" would
  mean "re-factor every call" -- defeating the purpose. This is precisely why the
  two-phase `prepare`/`solve` protocol is needed above the current one-method state.

- **Operand substitution vs the pattern token.** `LinearSolverState._check_compatible`
  already enforces a "same shape/dtype/device/batch layout" contract when substituting
  operands -- the existing precedent for the buffer-reuse (R1) and capture (R6)
  contracts. R7 extends it: substituting `A` with the *same sparsity pattern but new
  values* should trigger a numeric refactor in a direct backend. Today the check only
  compares shape/dtype (iterative solvers do not care about the pattern); the proposed
  sparsity-pattern token is what lets a direct backend tell "values changed" (refactor)
  from "pattern changed" (re-analyze) apart.

- **Backward compatibility.** The existing one-shot functions (`cg`/`cr`/`bicgstab`/
  `gmres`) and the `run=False` state objects keep working unchanged; they become the
  documented thin surface over the unified `LinearSolver`.

**Open question (preconditioner ownership).** Today `preconditioner()` builds `M`
and the caller passes it in as an operand. The new lifecycle could either keep `M`
caller-supplied (least disruptive, preserves the current signature) or move
preconditioner construction into the iterative backend's `prepare` (cleaner "all
setup amortized in prepare" story, but changes who owns `M`). This doc does not
decide it; it should be resolved when the iterative backends are wrapped.

### Alternatives Considered

- **Keep the one-shot functions, add a separate direct-solve function.** Rejected:
  it forks the API (iterative vs direct call sites differ), and still gives direct
  solvers no lifecycle to amortize over -- the core problem. A consumer switching
  between backends would have to restructure its code.

- **Explicit three-phase API: `analyze(pattern)` / `factor(A)` / `solve(b, x)`.**
  This most faithfully matches a direct solver's internals (symbolic / numeric /
  solve). Rejected as the *common* interface because it burdens the 90% iterative
  case with a phase they do not have, and pushes the analyze-vs-refactor bookkeeping
  onto every caller. Instead the three phases are preserved *internally*: `prepare`
  decides symbolic-vs-numeric by inspecting a **sparsity-pattern token** on the
  matrix (R7), so the public API stays two-phase while direct backends still skip
  re-analysis when the pattern is unchanged. (`update(A)` MAY be exposed as an
  explicit numeric-only refresh for callers that want to be unambiguous.)

- **Bake constraint handling into the solver** (e.g., accept boundary conditions).
  Rejected: the prototype showed constraint elimination is a distinct, caller-owned
  layer (`MetricHarmonicSolver` reduces Dirichlet BCs to an interior SPD system and
  hands the solver only `A x = b`). Keeping the solver a pure linear solve preserves
  reuse across `fem`, `geometry`, and user code, and avoids an indefinite KKT system
  that direct SPD / CG backends cannot take.

- **A smart `"auto"` cost model in v1.** Rejected for now (see Non-goals): direct
  wins decisively on ill-conditioned 3-D systems and loses on large well-conditioned
  2-D systems (heavy fill-in, few CG iterations), and the deciding factors are not
  cheaply predictable. The lifecycle is the durable contribution; selection policy
  is layered on later.

### Key Implementation Details

**Phase / capture mapping (R6).** The lifecycle maps cleanly onto what can and
cannot be inside a CUDA graph:

| Lifecycle call | Iterative backend | Direct backend | Capturable? |
| --- | --- | --- | --- |
| `prepare` (pattern new) | wrap operator, build preconditioner | symbolic analysis + numeric factorization | No (allocates / may sync) |
| `prepare` (pattern same) | rebuild preconditioner (cheap) | numeric refactorization only | Yes, in practice |
| `solve` | run the iteration (device-side convergence, no host readback) | triangular back-substitution | Yes |

Consumers run `prepare` once outside capture (or once per changed pattern), then
capture a loop body that calls the per-step numeric refresh and `solve`. The
prototype demonstrated a full optimization iteration -- assembly, refactor,
forward/adjoint solves, and the optimizer step -- captured into one graph and
replayed, with the solver's device-side state (and the iterative solver's
convergence check) living entirely on device.

**Sparsity-pattern token (R7).** `BsrMatrix` gains a cheap monotonically-increasing
*topology version* that bumps whenever `offsets`/`columns` are rebuilt (compression,
`bsr_from_triplets`, structural `bsr_mm`) but not when only `values` change
(`reuse_topology=True` refills, in-place scaling). `prepare` compares the token to
the one it last analyzed: equal => numeric-only path; changed => full symbolic +
numeric. This is what lets a direct backend reuse the expensive analysis across an
entire optimization loop and across different RHS/poses on the same mesh, while
keeping the public API two-phase.

**Capability negotiation (R4).** The factory and `"auto"` consult the capability
record to reject impossible combinations up front: a direct backend requires an
assembled matrix, so a matrix-free `LinearOperator` cannot use it; an SPD-only
backend rejects a `general` matrix; dtype must be supported. Explicitly naming an
inapplicable or unavailable backend is an error with a clear message, never a
silent fallback.

**Multi-RHS and chunking (R5, R10).** `solve` accepts `(n,)` or `(n, k)`. The base
class provides a default `(n, k)` implementation that loops `solve` over columns, so
every backend satisfies the batched interface. Backends that natively batch (cuDSS
multi-RHS) override it. Because a wide solve workspace on a large 3-D factor can
exceed device memory, the batched path auto-chunks into balanced groups sized to a
memory budget the backend can query, exposing only an optional override rather than
a required user flag (the prototype required a manual `--max-rhs`).

**Data handoff (R9).** Direct backends consume scalar CSR with stable device
pointers. A `bsr_to_scalar_csr` utility in `warp.sparse` builds the scalar sparsity
once per pattern and refreshes only values on numeric updates; it is a zero-copy
view for already-scalar (`1x1` block) matrices. Pointer stability across capture
replays is a documented contract of `prepare`/`solve`: buffers referenced by a
backend's descriptors must not be reallocated between `prepare` and its `solve`s.

**Mapping existing code.** The iterative backends wrap the existing
`LinearSolverState` subclasses (`CG`/`CR`/`BiCGSTAB`/`GMRES`), `LinearOperator`, and
`preconditioner`; see *Relationship to existing `warp.optim.linear` classes* above for
how `prepare`/`solve` map onto their construction + `__call__` and what has to change
(the pattern token, and the preconditioner-ownership question). The optional cuDSS
backend follows the `warp-cudss` prototype (dynamic `libcudss` load, Warp owns the CSR
arrays and passes device pointers, cuDSS stream set from `wp.get_stream`, analysis once
/ refactor on value change / reuse factorization across RHS).

## Testing Strategy

- **Backend parity.** For a set of SPD systems, assert the direct backend and the
  iterative backend produce the same solution to tolerance (the prototype has a
  precedent: a dense factor-once/solve-many reference backend checked against CG).
- **Lifecycle correctness.** A backend that overrides batched `solve` matches the
  per-column default on multi-RHS inputs; `prepare` then repeated `solve` with
  different RHS reuses setup and stays correct.
- **Pattern-token reuse.** After a values-only update (same token) the result
  matches a fresh full `prepare`; after a structural change (bumped token) the
  backend re-analyzes and stays correct.
- **Graph capture equivalence.** A captured `prepare`(once)+`solve`(replayed) loop
  matches the eager loop bit-for-bit for a backend whose solve is deterministic
  (the prototype verified this for the optimizer; the same harness applies to the
  solve path). Covers the device-side-convergence iterative path and the
  refactor+solve direct path.
- **Graceful availability.** With the optional backend's library absent,
  `make_solver(kind="auto")` returns the iterative default and all consumers still
  run; `kind="cudss"` raises a clear, actionable error.
- **Device coverage.** Iterative backends run on CPU and CUDA via
  `get_test_devices()`; direct backends are CUDA-only (or CPU-only for a CHOLMOD-
  style backend) and skip cleanly where unavailable. No timing-based assertions.
