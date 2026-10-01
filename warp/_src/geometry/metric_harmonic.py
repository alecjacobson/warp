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

Everything is matrix-free: ``L(s)`` acts as ``d0^T (s * (d0 x))``, so no operator
is ever assembled and ``s`` only ever appears as an elementwise edge scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from warp._src.geometry.harmonic import _get_column, _interior_indices, _selection_matrix, _set_column
from warp._src.optim.linear import LinearOperator, cg
from warp._src.sparse import BsrMatrix, bsr_mv

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


@wp.kernel
def _scale_in_place(values: wp.array[float], scale: wp.array[float]):
    i = wp.tid()
    values[i] = values[i] * scale[i]


@wp.kernel
def _axpby(ax: wp.array[float], y: wp.array[float], alpha: float, beta: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = alpha * ax[i] + beta * y[i]


@wp.kernel
def _accumulate_neg_product(a: wp.array[float], b: wp.array[float], out: wp.array[float]):
    # ``out -= a * b`` -- the per-edge adjoint contribution ``-(d0 w) * (d0 lambda)``.
    i = wp.tid()
    out[i] = out[i] - a[i] * b[i]


class MetricHarmonicSolver:
    """Harmonic solve of ``d0^T diag(s) d0`` with Dirichlet data, differentiable in ``s``.

    The topology (``d0`` and the boundary/interior split) is fixed at construction;
    the metric ``s`` is supplied per call, so a metric-optimization loop builds the
    solver once and calls :meth:`solve` and :meth:`vjp` each iteration as ``s``
    changes. The reduced operator ``L_uu(s)`` is applied matrix-free and shared
    between the forward and adjoint solves.

    Args:
        d0: The exterior derivative from :func:`~warp.geometry.dec_operators`, of
            shape ``(num_edges, num_points)``.
        num_points: Number of mesh vertices.
        boundary: Indices of the constrained vertices (distinct), of type
            :class:`warp.int32`.
        tol: Relative residual tolerance for the conjugate-gradient solves.
        max_iters: Maximum conjugate-gradient iterations. Defaults to the number
            of interior vertices.
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

    def _reduced_operator(self, star1: wp.array) -> LinearOperator:
        """Matrix-free ``L_uu(s)``: restrict ``d0^T diag(s) d0`` to interior vertices."""
        d0 = self.d0
        selection = self._interior_selection
        n = self.num_points
        edges = self.num_edges
        interior = self.num_interior
        device = self.device

        full = wp.empty(n, dtype=wp.float32, device=device)
        edge = wp.empty(edges, dtype=wp.float32, device=device)
        back = wp.empty(n, dtype=wp.float32, device=device)
        reduced = wp.empty(interior, dtype=wp.float32, device=device)

        def matvec(x, y, z, alpha, beta):
            bsr_mv(selection, x, full, transpose=True)  # extend to all vertices
            bsr_mv(d0, full, edge)  # edge gradient
            wp.launch(_scale_in_place, dim=edges, inputs=[edge, star1], device=device)  # apply the metric
            bsr_mv(d0, edge, back, transpose=True)  # back to vertices
            bsr_mv(selection, back, reduced)  # restrict to interior
            wp.launch(_axpby, dim=interior, inputs=[reduced, y, float(alpha), float(beta)], outputs=[z], device=device)

        return LinearOperator((interior, interior), wp.float32, self.device, matvec)

    def _apply_full(self, star1: wp.array, vector: wp.array) -> wp.array:
        """Apply the unreduced ``L(s) = d0^T diag(s) d0`` to a full-length vector."""
        edge = bsr_mv(self.d0, vector)
        wp.launch(_scale_in_place, dim=self.num_edges, inputs=[edge, star1], device=self.device)
        return bsr_mv(self.d0, edge, transpose=True)

    def solve(self, star1: wp.array, boundary_values: wp.array) -> wp.array:
        """Solve ``L(s) w = 0`` with ``w[boundary] = boundary_values``.

        Args:
            star1: Per-edge metric weights, of length ``num_edges``.
            boundary_values: Prescribed values at the constrained vertices: a
                one-dimensional array for a single function, or a two-dimensional
                ``(num_boundary, d)`` array for ``d`` functions.

        Returns:
            The solution at every vertex, shaped like ``boundary_values``
            (``(num_points,)`` or ``(num_points, d)``). Boundary entries are exact.
        """
        operator = self._reduced_operator(star1)
        columns_2d = boundary_values.ndim == 2
        num_functions = boundary_values.shape[1] if columns_2d else 1
        result = wp.zeros(
            (self.num_points, num_functions) if columns_2d else self.num_points, dtype=wp.float32, device=self.device
        )

        for c in range(num_functions):
            column = _get_column_or_self(boundary_values, c, columns_2d, self.device)
            solution = bsr_mv(self._boundary_selection, column, transpose=True)  # scatter boundary data
            forcing = self._apply_full(star1, solution)
            rhs = bsr_mv(self._interior_selection, forcing, alpha=-1.0)

            x = wp.zeros(self.num_interior, dtype=wp.float32, device=self.device)
            cg(operator, rhs, x, tol=self.tol, maxiter=self.max_iters)
            solution = bsr_mv(self._interior_selection, x, y=solution, transpose=True, alpha=1.0, beta=1.0)
            _store(result, solution, c, columns_2d, self.device)

        return result

    def vjp(self, star1: wp.array, solution: wp.array, grad_solution: wp.array) -> wp.array:
        """Gradient of a loss with respect to ``star1`` through the solve.

        Args:
            star1: The metric weights used in the forward :meth:`solve`.
            solution: The forward solution ``w`` it returned.
            grad_solution: The loss gradient ``dLoss/dw``, shaped like ``solution``.

        Returns:
            ``dLoss/dstar1``, a length-``num_edges`` array.
        """
        operator = self._reduced_operator(star1)
        columns_2d = solution.ndim == 2
        num_functions = solution.shape[1] if columns_2d else 1
        grad_star1 = wp.zeros(self.num_edges, dtype=wp.float32, device=self.device)

        for c in range(num_functions):
            w_column = _get_column_or_self(solution, c, columns_2d, self.device)
            g_column = _get_column_or_self(grad_solution, c, columns_2d, self.device)

            g_interior = bsr_mv(self._interior_selection, g_column)
            adjoint = wp.zeros(self.num_interior, dtype=wp.float32, device=self.device)
            cg(operator, g_interior, adjoint, tol=self.tol, maxiter=self.max_iters)
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


def _get_column_or_self(values: wp.array, c: int, columns_2d: bool, device: DeviceLike) -> wp.array:
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
