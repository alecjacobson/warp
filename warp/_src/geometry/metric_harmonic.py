# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable harmonic solve over an edge metric, for metric optimization.

:class:`MetricHarmonicSolver` solves the harmonic system of the DEC-factored
Laplacian ``L(s) = d0^T diag(s) d0`` with Dirichlet boundary conditions, where the
per-edge weights ``s`` (the diagonal Hodge star from
:func:`~warp.geometry.dec_operators`) are a free *metric*. It also returns the
gradient of a downstream loss with respect to ``s`` through the solve, computed by
the adjoint (implicit-function) method rather than by differentiating the solver:

    solve     ``L_uu(s) w_u = -L_ub(s) bc``
    adjoint   ``L_uu(s) lambda_u = (dLoss/dw)_u``
    gradient  ``dLoss/ds = -(d0 w) * (d0 lambda)``   (elementwise over edges)

The reduced interior operator ``L_uu(s)`` is *materialized* as a sparse matrix,
rebuilt by :meth:`~MetricHarmonicSolver.prepare` each time the metric changes, and
handed to a pluggable :class:`LinearSolver` backend. The default backend
(:class:`ConjugateGradientSolver`) runs Jacobi-preconditioned conjugate gradient; a
factorizing backend (e.g. a cuDSS Cholesky) that factors in ``prepare`` and
back-substitutes in ``solve`` drops in through the same interface. The
``prepare`` / ``solve`` / ``vjp`` split mirrors the factor-then-solve pattern a
direct solver uses, which makes it the natural seam for that swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import warp as wp
from warp._src.geometry.harmonic import _get_column, _interior_indices, _selection_matrix, _set_column
from warp._src.optim.linear import aslinearoperator, cg, preconditioner
from warp._src.sparse import BsrMatrix, bsr_copy, bsr_mm, bsr_mm_work_arrays, bsr_mv, bsr_transposed

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


@wp.kernel
def _scale_in_place(values: wp.array[float], scale: wp.array[float]):
    i = wp.tid()
    values[i] = values[i] * scale[i]


@wp.kernel
def _scale_by_row(source: wp.array[float], row_of_entry: wp.array[int], scale: wp.array[float], out: wp.array[float]):
    # ``out[k] = source[k] * scale[row(k)]`` -- scale each nonzero of a sparse
    # matrix by its row's metric weight, to form ``diag(s) d0_i``.
    k = wp.tid()
    out[k] = source[k] * scale[row_of_entry[k]]


@wp.kernel
def _accumulate_neg_product(a: wp.array[float], b: wp.array[float], out: wp.array[float]):
    # ``out -= a * b`` -- the per-edge adjoint contribution ``-(d0 w) * (d0 lambda)``.
    i = wp.tid()
    out[i] = out[i] - a[i] * b[i]


class LinearSolver(ABC):
    """Pluggable backend solving ``A x = b`` for a fixed symmetric positive-definite ``A``.

    :class:`MetricHarmonicSolver` calls :meth:`prepare` whenever the operator changes
    (once per metric update) and :meth:`solve` for every right-hand side that reuses
    it -- the forward solve of each column and the adjoint solve all share one prepared
    operator. This is the seam for a factorizing backend: a direct solver (e.g. cuDSS
    Cholesky) factors once in :meth:`prepare` and back-substitutes in :meth:`solve`,
    amortizing the factorization across the many solves of a metric-optimization step.
    """

    @abstractmethod
    def prepare(self, matrix: BsrMatrix) -> None:
        """Set up the operator ``matrix`` (e.g. factor it, or build a preconditioner).

        Called once per distinct operator. The matrix is square, symmetric, and
        positive definite.
        """

    @abstractmethod
    def solve(self, rhs: wp.array, x: wp.array) -> None:
        """Solve ``matrix x = rhs`` in place, using the incoming ``x`` as initial guess.

        ``prepare`` is guaranteed to have been called first. ``rhs`` and ``x`` are
        one-dimensional arrays matching the operator dimension; ``x`` holds a warm
        start on entry (zero if none) and the solution on return.
        """

    def solve_many(self, rhs: wp.array, x: wp.array) -> None:
        """Solve ``matrix x[:, j] = rhs[:, j]`` for every column of the two-dimensional
        ``rhs`` / ``x`` (shape ``(n, k)``), in place.

        The default loops over columns calling :meth:`solve`. A backend that can solve
        several right-hand sides together (e.g. a direct factorization reused across a
        multi-RHS triangular solve) should override this -- that is where most of the
        speedup of batching the solves comes from.
        """
        n, k = rhs.shape
        column_rhs = wp.empty(n, dtype=rhs.dtype, device=rhs.device)
        column_x = wp.empty(n, dtype=x.dtype, device=x.device)
        for j in range(k):
            wp.launch(_get_column, dim=n, inputs=[rhs, j], outputs=[column_rhs], device=rhs.device)
            wp.launch(_get_column, dim=n, inputs=[x, j], outputs=[column_x], device=x.device)
            self.solve(column_rhs, column_x)
            wp.launch(_set_column, dim=n, inputs=[x, j, column_x], device=x.device)


class ConjugateGradientSolver(LinearSolver):
    """Default backend: Jacobi-preconditioned conjugate gradient.

    Args:
        tol: Relative residual tolerance.
        max_iters: Maximum iterations. Defaults to the operator dimension.
        use_preconditioner: If ``True`` (default), use a diagonal (Jacobi)
            preconditioner. This is essential on non-uniform meshes, where it can
            cut the iteration count by an order of magnitude.
        check_every: Iterations between host-side convergence checks. ``0`` keeps the
            entire solve on the device (CUDA only, no host readback). Defaults to
            ``0`` on CUDA and ``10`` on CPU.
    """

    def __init__(
        self,
        *,
        tol: float = 1.0e-6,
        max_iters: int | None = None,
        use_preconditioner: bool = True,
        check_every: int | None = None,
    ):
        self.tol = tol
        self.max_iters = max_iters
        self.use_preconditioner = use_preconditioner
        self.check_every = check_every
        self._operator = None
        self._preconditioner = None
        self._maxiter = None
        self._check_every = 0

    def prepare(self, matrix: BsrMatrix) -> None:
        self._operator = aslinearoperator(matrix)
        self._preconditioner = preconditioner(matrix, "diag") if self.use_preconditioner else None
        self._maxiter = self.max_iters if self.max_iters is not None else matrix.shape[0]
        # Device-side convergence (no host readback) is available on CUDA; on CPU
        # fall back to periodic host checks so the solve still terminates early.
        self._check_every = self.check_every if self.check_every is not None else (0 if matrix.device.is_cuda else 10)

    def solve(self, rhs: wp.array, x: wp.array) -> None:
        if self._operator is None:
            raise RuntimeError("Call ConjugateGradientSolver.prepare() before solve().")
        cg(
            self._operator,
            rhs,
            x,
            tol=self.tol,
            maxiter=self._maxiter,
            M=self._preconditioner,
            check_every=self._check_every,
        )


class MetricHarmonicSolver:
    """Harmonic solve of ``d0^T diag(s) d0`` with Dirichlet data, differentiable in ``s``.

    The topology (``d0`` and the boundary/interior split) is fixed at construction.
    Call :meth:`prepare` with a metric ``s`` to build (materialize) the reduced
    operator, then :meth:`solve` and :meth:`vjp` as many times as needed with that
    metric. A metric-optimization loop prepares once per iteration as ``s`` changes.

    Args:
        d0: The exterior derivative from :func:`~warp.geometry.dec_operators`, of
            shape ``(num_edges, num_points)``.
        num_points: Number of mesh vertices.
        boundary: Indices of the constrained vertices (distinct), of type
            :class:`warp.int32`.
        tol: Relative residual tolerance for the default conjugate-gradient backend.
            Ignored when ``solver`` is given.
        max_iters: Maximum iterations for the default backend. Defaults to the number
            of interior vertices. Ignored when ``solver`` is given.
        use_preconditioner: Whether the default backend uses a diagonal (Jacobi)
            preconditioner (default ``True``). Ignored when ``solver`` is given.
        solver: Linear-solver backend used to solve the reduced interior system. If
            omitted, a :class:`ConjugateGradientSolver` is built from ``tol``,
            ``max_iters``, and ``use_preconditioner``. Pass a custom
            :class:`LinearSolver` (e.g. a cuDSS-backed direct solver) to factor the
            materialized operator once per metric and back-substitute each solve.
        device: Device on which to run. Defaults to the device of ``d0``.
    """

    def __init__(
        self,
        d0: BsrMatrix,
        num_points: int,
        boundary: wp.array[int],
        *,
        tol: float = 1.0e-6,
        max_iters: int | None = None,
        use_preconditioner: bool = True,
        solver: LinearSolver | None = None,
        device: DeviceLike | None = None,
    ):
        self.device = wp.get_device(device) if device is not None else d0.device
        self.d0 = d0
        self.num_points = num_points
        self.num_edges = d0.shape[0]
        self.num_interior = num_points - boundary.shape[0]
        self.interior = _interior_indices(boundary, num_points, self.num_interior, self.device)
        self._boundary_selection = _selection_matrix(boundary, num_points, wp.float32, self.device)
        self._interior_selection = _selection_matrix(self.interior, num_points, wp.float32, self.device)
        self.solver = solver or ConjugateGradientSolver(
            tol=tol, max_iters=max_iters, use_preconditioner=use_preconditioner
        )

        # ``d0_i = d0 restricted to interior columns`` (edges x interior). Then
        # ``L_uu(s) = d0_i^T diag(s) d0_i``. Precompute it and the per-nonzero edge
        # index used to scale its rows by the metric.
        self._d0_interior = bsr_mm(d0, bsr_transposed(self._interior_selection))
        self._d0_interior_T = bsr_transposed(self._d0_interior)
        self._edge_of_entry = self._d0_interior.uncompress_rows()
        self._scaled = bsr_copy(self._d0_interior)
        # Establish the sparsity of L_uu once so later rebuilds only refill values.
        self._mm_work = bsr_mm_work_arrays()
        self._reduced = bsr_mm(self._d0_interior_T, self._d0_interior, work_arrays=self._mm_work)

        self._star1: wp.array | None = None
        self._prepared = False

        # Scratch reused across solve()/vjp() so neither allocates per iteration: this
        # keeps memory flat (important for large direct factorizations) and lets the
        # whole loop be captured into a CUDA graph.
        self._sol = wp.empty(num_points, dtype=wp.float32, device=self.device)
        self._edge = wp.empty(self.num_edges, dtype=wp.float32, device=self.device)
        self._forcing = wp.empty(num_points, dtype=wp.float32, device=self.device)
        self._interior_col = wp.empty(self.num_interior, dtype=wp.float32, device=self.device)
        self._full = wp.empty(num_points, dtype=wp.float32, device=self.device)
        self._vertex_col = wp.empty(num_points, dtype=wp.float32, device=self.device)
        self._d0_w = wp.empty(self.num_edges, dtype=wp.float32, device=self.device)
        self._d0_lambda = wp.empty(self.num_edges, dtype=wp.float32, device=self.device)
        # Batched (n, k) right-hand side / solution / scattered buffers, sized lazily.
        self._rhs_batch: wp.array | None = None
        self._x_batch: wp.array | None = None
        self._scattered: wp.array | None = None
        self._batch_width = 0

    def prepare(self, star1: wp.array) -> None:
        """Materialize the reduced operator ``L_uu(s)`` for the metric ``star1``.

        Rebuilds the sparse operator (reusing the fixed sparsity pattern), hands it to
        the solver backend (which factors it or builds its preconditioner), and stores
        the metric for the Dirichlet forcing term. Call once per metric before
        :meth:`solve` / :meth:`vjp`.
        """
        wp.launch(
            _scale_by_row,
            dim=self._scaled.values.shape[0],
            inputs=[self._d0_interior.values, self._edge_of_entry, star1, self._scaled.values],
            device=self.device,
        )
        bsr_mm(self._d0_interior_T, self._scaled, self._reduced, reuse_topology=True, work_arrays=self._mm_work)
        self.solver.prepare(self._reduced)
        self._star1 = star1
        self._prepared = True

    def _apply_full_into(self, vector: wp.array, out: wp.array) -> None:
        """Apply the unreduced ``L(s) = d0^T diag(s) d0`` to ``vector``, writing ``out``."""
        bsr_mv(self.d0, vector, y=self._edge, beta=0.0)
        wp.launch(_scale_in_place, dim=self.num_edges, inputs=[self._edge, self._star1], device=self.device)
        bsr_mv(self.d0, self._edge, y=out, transpose=True, beta=0.0)

    def _ensure_batch(self, width: int) -> None:
        if self._batch_width != width:
            self._rhs_batch = wp.empty((self.num_interior, width), dtype=wp.float32, device=self.device)
            self._x_batch = wp.empty((self.num_interior, width), dtype=wp.float32, device=self.device)
            self._scattered = wp.empty((self.num_points, width), dtype=wp.float32, device=self.device)
            self._batch_width = width

    def solve(self, boundary_values: wp.array, warm_start: wp.array | None = None) -> wp.array:
        """Solve ``L(s) w = 0`` with ``w[boundary] = boundary_values``.

        :meth:`prepare` must have been called with the current metric.

        Args:
            boundary_values: Prescribed values at the constrained vertices: a
                one-dimensional array for a single function, or a two-dimensional
                ``(num_boundary, d)`` array for ``d`` functions.
            warm_start: Optional previous solution (same shape as the returned
                array) used as the solver's initial guess. In an optimization loop
                the previous iteration's solution makes an excellent warm start.

        Returns:
            The solution at every vertex, shaped like ``boundary_values``
            (``(num_points,)`` or ``(num_points, d)``). Boundary entries are exact.
        """
        self._require_prepared()
        columns_2d = boundary_values.ndim == 2
        num_functions = boundary_values.shape[1] if columns_2d else 1
        self._ensure_batch(num_functions)
        result = wp.zeros(
            (self.num_points, num_functions) if columns_2d else self.num_points, dtype=wp.float32, device=self.device
        )

        # Build every column's reduced right-hand side and keep its scattered boundary
        # data, then solve all columns together.
        for c in range(num_functions):
            column = _column_of(boundary_values, c, columns_2d, self.device)
            bsr_mv(self._boundary_selection, column, y=self._sol, transpose=True, beta=0.0)  # scatter
            self._apply_full_into(self._sol, self._forcing)
            bsr_mv(self._interior_selection, self._forcing, y=self._interior_col, alpha=-1.0, beta=0.0)
            wp.launch(
                _set_column, dim=self.num_interior, inputs=[self._rhs_batch, c, self._interior_col], device=self.device
            )
            wp.launch(_set_column, dim=self.num_points, inputs=[self._scattered, c, self._sol], device=self.device)

        if warm_start is None:
            self._x_batch.zero_()
        else:
            for c in range(num_functions):
                warm_column = _column_of(warm_start, c, columns_2d, self.device)
                bsr_mv(self._interior_selection, warm_column, y=self._interior_col, beta=0.0)
                wp.launch(
                    _set_column,
                    dim=self.num_interior,
                    inputs=[self._x_batch, c, self._interior_col],
                    device=self.device,
                )

        self.solver.solve_many(self._rhs_batch, self._x_batch)

        for c in range(num_functions):
            wp.launch(
                _get_column, dim=self.num_points, inputs=[self._scattered, c], outputs=[self._sol], device=self.device
            )
            wp.launch(
                _get_column,
                dim=self.num_interior,
                inputs=[self._x_batch, c],
                outputs=[self._interior_col],
                device=self.device,
            )
            bsr_mv(self._interior_selection, self._interior_col, y=self._sol, transpose=True, alpha=1.0, beta=1.0)
            _store(result, self._sol, c, columns_2d, self.device)

        return result

    def vjp(self, solution: wp.array, grad_solution: wp.array) -> wp.array:
        """Gradient of a loss with respect to the metric through the solve.

        :meth:`prepare` must have been called with the same metric used for the
        forward :meth:`solve` that produced ``solution``.

        Args:
            solution: The forward solution ``w``.
            grad_solution: The loss gradient ``dLoss/dw``, shaped like ``solution``.

        Returns:
            ``dLoss/dstar1``, a length-``num_edges`` array.
        """
        self._require_prepared()
        columns_2d = solution.ndim == 2
        num_functions = solution.shape[1] if columns_2d else 1
        self._ensure_batch(num_functions)
        grad_star1 = wp.zeros(self.num_edges, dtype=wp.float32, device=self.device)

        # Adjoint systems share the operator with the forward solve, so batch them too.
        for c in range(num_functions):
            g_column = _column_of(grad_solution, c, columns_2d, self.device)
            bsr_mv(self._interior_selection, g_column, y=self._interior_col, beta=0.0)
            wp.launch(
                _set_column, dim=self.num_interior, inputs=[self._rhs_batch, c, self._interior_col], device=self.device
            )

        self._x_batch.zero_()
        self.solver.solve_many(self._rhs_batch, self._x_batch)

        for c in range(num_functions):
            wp.launch(
                _get_column,
                dim=self.num_interior,
                inputs=[self._x_batch, c],
                outputs=[self._interior_col],
                device=self.device,
            )
            bsr_mv(
                self._interior_selection, self._interior_col, y=self._full, transpose=True, beta=0.0
            )  # zero on boundary
            w_column = _column_of(solution, c, columns_2d, self.device)
            bsr_mv(self.d0, w_column, y=self._d0_w, beta=0.0)
            bsr_mv(self.d0, self._full, y=self._d0_lambda, beta=0.0)
            wp.launch(
                _accumulate_neg_product,
                dim=self.num_edges,
                inputs=[self._d0_w, self._d0_lambda],
                outputs=[grad_star1],
                device=self.device,
            )

        return grad_star1

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("Call MetricHarmonicSolver.prepare(star1) before solve() or vjp().")


def _column_of(values: wp.array, c: int, columns_2d: bool, device: DeviceLike) -> wp.array:
    if not columns_2d:
        return values
    out = wp.empty(values.shape[0], dtype=wp.float32, device=device)
    wp.launch(_get_column, dim=values.shape[0], inputs=[values, c], outputs=[out], device=device)
    return out


def _store(result: wp.array, solution: wp.array, c: int, columns_2d: bool, device: DeviceLike) -> None:
    if not columns_2d:
        wp.copy(result, solution)
    else:
        wp.launch(_set_column, dim=result.shape[0], inputs=[result, c, solution], device=device)
