# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve real symmetric standard and SPD generalized partial eigenproblems.

Solve ``A u = lambda M u`` on CPU or CUDA, with symmetric A and an optional
positive-definite mass operator M. Omitting mass selects the identity. Inputs
and results use float64; a :class:`warp.optim.linear.KrylovSolve` used for
preconditioning may work internally in float32. Matrix assembly belongs to the
caller, preferably using :mod:`warp.fem`.

Choose :class:`LOBPCG` for the smallest algebraic eigenpairs with an effective
preconditioner, or :class:`KrylovSchur` for restarted spectral selection and
shift-invert. For shift-invert, assemble ``A - sigma M`` and supply a prepared
one-RHS inverse. Generalized regular Krylov iteration requires a prepared mass
inverse. Inaccurate inner inverses are reported as failures; truncated LOBPCG
preconditioning is allowed.

Construct and prepare outside CUDA capture. Preparation allocates workspace,
compiles kernels and uploads the graph. Calling a prepared CUDA state returns
device arrays without reading convergence on the host. CPU execution uses
Warp numerical kernels and stops eagerly; CPU graph capture is rejected.
A prepared CUDA call can also be
recorded in a caller-owned graph::

    import warp as wp
    from warp.optim import eigen, linear

    correction = linear.KrylovSolve(A, nrhs=k, method="cg", tol=0.01)
    state = eigen.LOBPCG(A, k, mass=M, preconditioner=correction).prepare()
    result = state()
    result.check()  # Explicit synchronization and convergence check.

    with wp.ScopedCapture(device=A.device) as capture:
        state()
    wp.capture_launch(capture.graph)

Keep the state and borrowed operators alive until dependent device work
finishes. Results are overwritten on the next call. Order executions sharing
workspace; concurrent calls on one state are unsupported. Matrix values may
change in place, but storage, shapes and operator contracts must stay fixed.
Update borrowed factorizations and preconditioners separately when needed.

Acceptance uses fresh original-equation residuals:
``||A u - lambda M u|| <= atol + tol (||A u|| + |lambda| ||M u||)``.
Results are sorted by ascending eigenvalue and mass-normalized. Inspect
``valid``, ``converged`` and ``status`` before using a result, or call
:meth:`EigenResult.check` outside capture to raise on failure. Small residuals
alone do not prove spectral completeness for an arbitrary starting subspace.

Complex, nonsymmetric, singular-mass, batched independent eigenproblems and
reverse-mode differentiation are outside this initial API. A ``warp.Tape``
enclosing an eigen call is explicitly rejected. See
``warp/examples/optim/example_eigen.py`` for complete FEM assembly and a solve.

Inherited execution methods
---------------------------

.. automethod:: warp.optim.eigen.LOBPCG.prepare

.. automethod:: warp.optim.eigen.KrylovSchur.prepare
"""

from warp._src.optim.eigen._frontend import (
    LOBPCG,
    EigenResult,
    EigenSolverState,
    EigenStatus,
    KrylovSchur,
    krylov_schur,
    lobpcg,
)

__all__ = ["LOBPCG", "EigenResult", "EigenSolverState", "EigenStatus", "KrylovSchur", "krylov_schur", "lobpcg"]
