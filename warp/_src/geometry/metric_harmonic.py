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

The reduced interior operator ``L_uu(s)`` is *materialized* as a sparse matrix (so
conjugate gradient uses a single sparse matrix-vector product and a diagonal
Jacobi preconditioner), rebuilt by :meth:`prepare` each time the metric changes.
The ``prepare`` / ``solve`` / ``vjp`` split mirrors the factor-then-solve pattern a
direct solver uses, which makes it the natural seam for swapping in a factorizing
backend later.
"""

from __future__ import annotations

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
        tol: Relative residual tolerance for the conjugate-gradient solves.
        max_iters: Maximum conjugate-gradient iterations. Defaults to the number
            of interior vertices.
        use_preconditioner: If ``True`` (default), use a diagonal (Jacobi)
            preconditioner. This is essential on non-uniform meshes, where it can
            cut the iteration count by an order of magnitude.
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
        self.tol = tol
        self.max_iters = max_iters if max_iters is not None else self.num_interior
        self._use_preconditioner = use_preconditioner
        # Device-side convergence (no host readback) is available on CUDA; on CPU
        # fall back to periodic host checks so the solve still terminates early.
        self._check_every = 0 if self.device.is_cuda else 10

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
        self._operator = None
        self._preconditioner = None

    def prepare(self, star1: wp.array) -> None:
        """Materialize the reduced operator ``L_uu(s)`` for the metric ``star1``.

        Rebuilds the sparse operator (reusing the fixed sparsity pattern) and its
        preconditioner, and stores the metric for the Dirichlet forcing term. Call
        once per metric before :meth:`solve` / :meth:`vjp`.
        """
        wp.launch(
            _scale_by_row,
            dim=self._scaled.values.shape[0],
            inputs=[self._d0_interior.values, self._edge_of_entry, star1, self._scaled.values],
            device=self.device,
        )
        bsr_mm(self._d0_interior_T, self._scaled, self._reduced, reuse_topology=True, work_arrays=self._mm_work)
        self._star1 = star1
        self._operator = aslinearoperator(self._reduced)
        self._preconditioner = preconditioner(self._reduced, "diag") if self._use_preconditioner else None

    def _cg(self, rhs: wp.array, x: wp.array) -> None:
        cg(
            self._operator,
            rhs,
            x,
            tol=self.tol,
            maxiter=self.max_iters,
            M=self._preconditioner,
            check_every=self._check_every,
        )

    def _apply_full(self, vector: wp.array) -> wp.array:
        """Apply the unreduced ``L(s) = d0^T diag(s) d0`` to a full-length vector."""
        edge = bsr_mv(self.d0, vector)
        wp.launch(_scale_in_place, dim=self.num_edges, inputs=[edge, self._star1], device=self.device)
        return bsr_mv(self.d0, edge, transpose=True)

    def solve(self, boundary_values: wp.array, warm_start: wp.array | None = None) -> wp.array:
        """Solve ``L(s) w = 0`` with ``w[boundary] = boundary_values``.

        :meth:`prepare` must have been called with the current metric.

        Args:
            boundary_values: Prescribed values at the constrained vertices: a
                one-dimensional array for a single function, or a two-dimensional
                ``(num_boundary, d)`` array for ``d`` functions.
            warm_start: Optional previous solution (same shape as the returned
                array) used as the conjugate-gradient initial guess. In an
                optimization loop the previous iteration's solution makes an
                excellent warm start.

        Returns:
            The solution at every vertex, shaped like ``boundary_values``
            (``(num_points,)`` or ``(num_points, d)``). Boundary entries are exact.
        """
        self._require_prepared()
        columns_2d = boundary_values.ndim == 2
        num_functions = boundary_values.shape[1] if columns_2d else 1
        result = wp.zeros(
            (self.num_points, num_functions) if columns_2d else self.num_points, dtype=wp.float32, device=self.device
        )

        for c in range(num_functions):
            column = _column_of(boundary_values, c, columns_2d, self.device)
            solution = bsr_mv(self._boundary_selection, column, transpose=True)  # scatter boundary data
            forcing = self._apply_full(solution)
            rhs = bsr_mv(self._interior_selection, forcing, alpha=-1.0)

            if warm_start is None:
                x = wp.zeros(self.num_interior, dtype=wp.float32, device=self.device)
            else:
                warm_column = _column_of(warm_start, c, columns_2d, self.device)
                x = bsr_mv(self._interior_selection, warm_column)
            self._cg(rhs, x)
            solution = bsr_mv(self._interior_selection, x, y=solution, transpose=True, alpha=1.0, beta=1.0)
            _store(result, solution, c, columns_2d, self.device)

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
        grad_star1 = wp.zeros(self.num_edges, dtype=wp.float32, device=self.device)

        for c in range(num_functions):
            w_column = _column_of(solution, c, columns_2d, self.device)
            g_column = _column_of(grad_solution, c, columns_2d, self.device)

            g_interior = bsr_mv(self._interior_selection, g_column)
            adjoint = wp.zeros(self.num_interior, dtype=wp.float32, device=self.device)
            self._cg(g_interior, adjoint)
            adjoint_full = bsr_mv(self._interior_selection, adjoint, transpose=True)  # zero on the boundary

            d0_w = bsr_mv(self.d0, w_column)
            d0_lambda = bsr_mv(self.d0, adjoint_full)
            wp.launch(
                _accumulate_neg_product,
                dim=self.num_edges,
                inputs=[d0_w, d0_lambda],
                outputs=[grad_star1],
                device=self.device,
            )

        return grad_star1

    def _require_prepared(self) -> None:
        if self._operator is None:
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
