# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harmonic and biharmonic solves on triangle and tetrahedral meshes.

The public entry point is :func:`harmonic`, the counterpart of ``igl::harmonic``.
It minimizes the ``k``-harmonic energy subject to Dirichlet boundary conditions,
building the same operator ``Q = L (M^-1 L)^(k-1)`` from the cotangent Laplacian
:func:`~warp._src.geometry.laplacian.laplacian` and the lumped mass matrix
:func:`~warp._src.geometry.mass.massmatrix`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from warp._src.geometry.laplacian import _resolve_simplex_indices, laplacian
from warp._src.geometry.mass import MassMatrixType, massmatrix
from warp._src.optim.linear import cg, preconditioner
from warp._src.sparse import (
    BsrMatrix,
    bsr_copy,
    bsr_diag,
    bsr_from_triplets,
    bsr_get_diag,
    bsr_mm,
    bsr_mv,
    bsr_transposed,
)
from warp._src.utils import array_scan

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


@wp.kernel
def _iota(out: wp.array[int]):
    out[wp.tid()] = wp.tid()


@wp.kernel
def _reciprocal(src: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    # Inverse of a positive diagonal, with zeros left as zeros so an unreferenced
    # vertex (zero mass) does not produce a non-finite entry. The mass inverse only
    # enters the biharmonic (k>=2) operator, which is assembled in double precision.
    i = wp.tid()
    v = src[i]
    out[i] = wp.where(v != wp.float64(0.0), wp.float64(1.0) / v, wp.float64(0.0))


@wp.kernel
def _to_f64(src: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float64)):
    out[wp.tid()] = wp.float64(src[wp.tid()])


@wp.kernel
def _to_f32(src: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float32)):
    out[wp.tid()] = wp.float32(src[wp.tid()])


@wp.kernel
def _get_column(src: wp.array2d(dtype=float), col: int, out: wp.array[float]):
    out[wp.tid()] = src[wp.tid(), col]


@wp.kernel
def _set_column(dst: wp.array2d(dtype=float), col: int, src: wp.array[float]):
    dst[wp.tid(), col] = src[wp.tid()]


@wp.kernel
def _mark_boundary(boundary: wp.array[int], flags: wp.array[int]):
    flags[boundary[wp.tid()]] = 1


@wp.kernel
def _free_flag(is_boundary: wp.array[int], out_free: wp.array[int]):
    i = wp.tid()
    out_free[i] = 1 - is_boundary[i]


@wp.kernel
def _scatter_interior(is_boundary: wp.array[int], positions: wp.array[int], interior: wp.array[int]):
    # ``positions`` is the exclusive prefix sum of the free (non-boundary) flags,
    # so it is the compact index each free vertex takes among the interior list.
    i = wp.tid()
    if is_boundary[i] == 0:
        interior[positions[i]] = i


def _interior_indices(boundary: wp.array, n: int, num_interior: int, device: DeviceLike) -> wp.array:
    """Compute the interior vertex indices on device, without a host readback.

    ``boundary`` must hold distinct indices; ``num_interior`` is ``n`` minus its
    length. Marking, scanning, and scattering keeps the whole operation
    CUDA-graph capturable.
    """
    is_boundary = wp.zeros(n, dtype=wp.int32, device=device)
    wp.launch(_mark_boundary, dim=boundary.shape[0], inputs=[boundary], outputs=[is_boundary], device=device)
    free = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(_free_flag, dim=n, inputs=[is_boundary], outputs=[free], device=device)
    positions = wp.empty(n, dtype=wp.int32, device=device)
    array_scan(free, positions, inclusive=False)
    interior = wp.empty(num_interior, dtype=wp.int32, device=device)
    wp.launch(_scatter_interior, dim=n, inputs=[is_boundary, positions], outputs=[interior], device=device)
    return interior


def _selection_matrix(selected: wp.array, num_columns: int, dtype: type, device: DeviceLike) -> BsrMatrix:
    """Build the ``(len(selected), num_columns)`` matrix that gathers the selected rows.

    Row ``i`` has a single ``1`` in column ``selected[i]``, so left-multiplication
    picks those entries out of a vector and its transpose scatters them back.
    """
    count = selected.shape[0]
    rows = wp.empty(count, dtype=wp.int32, device=device)
    wp.launch(_iota, dim=count, outputs=[rows], device=device)
    ones = wp.ones(count, dtype=dtype, device=device)
    return bsr_from_triplets(count, num_columns, rows, selected, ones)


def _harmonic_operator(laplacian_matrix: BsrMatrix, mass: BsrMatrix, k: int) -> BsrMatrix:
    """Assemble ``Q = L (M^-1 L)^(k-1)`` in double precision, matching ``igl::harmonic``."""
    diag = bsr_get_diag(mass)
    inverse = wp.empty_like(diag)
    wp.launch(_reciprocal, dim=diag.shape[0], inputs=[diag], outputs=[inverse], device=diag.device)
    mass_inverse = bsr_diag(inverse)

    operator = laplacian_matrix
    for _ in range(k - 1):
        operator = bsr_mm(bsr_mm(operator, mass_inverse), laplacian_matrix)
    return operator


def harmonic(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    boundary: wp.array[int],
    boundary_values: wp.array,
    *,
    k: int = 1,
    num_points: int | None = None,
    simplex_size: int | None = None,
    tol: float = 1.0e-8,
    max_iters: int | None = None,
    device: DeviceLike | None = None,
) -> wp.array:
    """Solve for a ``k``-harmonic function with Dirichlet boundary conditions.

    Minimizes the ``k``-harmonic energy ``z^T Q z`` subject to ``z[boundary] =
    boundary_values``, where ``Q = L (M^-1 L)^(k-1)`` is built from the cotangent
    Laplacian ``L`` and the lumped mass matrix ``M``. This is the construction of
    ``igl::harmonic``: ``k=1`` gives a harmonic function (it minimizes the
    Dirichlet energy and reproduces linear data), ``k=2`` a biharmonic function,
    and higher ``k`` the polyharmonic functions. The mass matrix matches
    ``igl::harmonic``'s default -- the mixed Voronoi lumping on triangle meshes and
    barycentric lumping on tetrahedral meshes.

    The constrained minimization is reduced to the unknown (interior) vertices
    and solved with conjugate gradient.

    .. warning::

        Warp's built-in conjugate-gradient solver is efficient only for ``k=1``.
        For ``k>1`` the operator squares the Laplacian, whose condition number
        grows like ``h^-4``, so conjugate gradient converges slowly and, in single
        precision, stalls well short of the solution. This function therefore
        assembles and solves ``k>1`` in double precision and allows many more
        iterations (see ``max_iters``), but the solve remains inherently expensive
        and may not reach ``tol`` on large or finely tessellated meshes. There is
        no sparse direct solver or multigrid preconditioner available here to do
        better; libigl solves the same operator with a direct factorization, which
        is why it is robust where this iterative solve is not. For biharmonic work
        at scale, prefer a coarser mesh or an external direct/multigrid solver.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Mesh connectivity, either a flat :class:`warp.int32` array or a
            typed array of :class:`warp.vec3i` (triangles) or :class:`warp.vec4i`
            (tetrahedra); see :func:`~warp.geometry.laplacian`.
        boundary: Indices of the constrained vertices, of type
            :class:`warp.int32`. The indices must be distinct.
        boundary_values: Prescribed values at the constrained vertices. A
            one-dimensional array of length ``len(boundary)`` solves for a single
            function; a two-dimensional array of shape ``(len(boundary), d)``
            solves for ``d`` functions sharing the same operator, one per column.
        k: Power of the harmonic operator. ``1`` is harmonic, ``2`` biharmonic.
        num_points: Number of vertices. Inferred from ``points`` when omitted.
        simplex_size: Vertices per element (3 or 4) when ``indices`` is a flat
            array; see :func:`~warp.geometry.laplacian`.
        tol: Relative residual tolerance for the conjugate-gradient solve.
        max_iters: Maximum conjugate-gradient iterations. Defaults to the size of
            the unknown system for ``k=1``, and ten times that for ``k>1``, whose
            slow convergence needs many more iterations.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        The solved function values at every vertex, of type
        :class:`warp.float32`. Its shape matches ``boundary_values``: ``(num_points,)``
        for one-dimensional input, ``(num_points, d)`` for two-dimensional input.
        The boundary entries equal ``boundary_values`` exactly.

    Raises:
        ValueError: If ``k`` is less than 1, if ``boundary`` and
            ``boundary_values`` disagree in length, or if ``boundary_values`` is
            neither one- nor two-dimensional.
    """
    if k < 1:
        raise ValueError(f"`k` must be at least 1, but got {k}.")

    device = wp.get_device(device) if device is not None else points.device
    columns_2d = boundary_values.ndim == 2
    if boundary_values.ndim not in (1, 2):
        raise ValueError(f"`boundary_values` must be 1- or 2-dimensional, but got {boundary_values.ndim} dimensions.")
    if boundary_values.shape[0] != boundary.shape[0]:
        raise ValueError(
            f"`boundary_values` has {boundary_values.shape[0]} rows but `boundary` has {boundary.shape[0]} indices."
        )

    # The harmonic (k=1) system is well conditioned, so it is assembled and solved
    # in single precision. The biharmonic and higher operators square the
    # Laplacian, whose condition number grows like h^-4; single-precision conjugate
    # gradient stalls on them, so k>=2 is assembled and solved in double precision.
    solve_dtype = wp.float32 if k == 1 else wp.float64

    flat_indices, resolved_simplex_size = _resolve_simplex_indices(indices, simplex_size)

    laplacian_matrix = laplacian(points, flat_indices, simplex_size=resolved_simplex_size, device=device)
    n = laplacian_matrix.shape[0]
    laplacian_matrix = _as_scalar_type(laplacian_matrix, solve_dtype)

    if k >= 2:
        # Match igl::harmonic, which lumps mass with the Voronoi rule on surfaces.
        mass_kind = MassMatrixType.VORONOI if resolved_simplex_size == 3 else MassMatrixType.BARYCENTRIC
        mass = massmatrix(points, flat_indices, kind=mass_kind, simplex_size=resolved_simplex_size, device=device)
        operator = _harmonic_operator(laplacian_matrix, _as_scalar_type(mass, solve_dtype), k)
    else:
        operator = laplacian_matrix

    # Split the vertices into the constrained boundary and the free interior. The
    # boundary indices are assumed distinct, so the interior count is known from
    # the shapes alone and the indices are gathered on device.
    num_interior = n - boundary.shape[0]
    interior = _interior_indices(boundary, n, num_interior, device)

    num_functions = boundary_values.shape[1] if columns_2d else 1
    result = wp.zeros((n, num_functions) if columns_2d else n, dtype=wp.float32, device=device)

    boundary_selection = _selection_matrix(boundary, n, solve_dtype, device)
    if num_interior == 0:
        # Every vertex is constrained; the solution is just the boundary data.
        for c in range(num_functions):
            column = _column(boundary_values, c, columns_2d, solve_dtype, device)
            solution = bsr_mv(boundary_selection, column, transpose=True)
            _store(result, solution, c, columns_2d, device)
        return result

    interior_selection = _selection_matrix(interior, n, solve_dtype, device)
    reduced = bsr_mm(bsr_mm(interior_selection, operator), bsr_transposed(interior_selection))
    reduced_preconditioner = preconditioner(reduced, "diag")
    if max_iters is not None:
        maxiter = max_iters
    elif k == 1:
        maxiter = num_interior
    else:
        # Biharmonic CG needs many more than n iterations; see the note in the
        # docstring about the cost of k>1.
        maxiter = 10 * num_interior

    for c in range(num_functions):
        column = _column(boundary_values, c, columns_2d, solve_dtype, device)
        # Scatter the boundary data into a full-length vector, apply the operator,
        # and gather (with a sign flip) the interior rows to form the right-hand
        # side of the reduced system ``reduced @ x = -[operator @ z_b]_interior``.
        solution = bsr_mv(boundary_selection, column, transpose=True)
        forcing = bsr_mv(operator, solution)
        rhs = bsr_mv(interior_selection, forcing, alpha=-1.0)

        x = wp.zeros(num_interior, dtype=solve_dtype, device=device)
        cg(reduced, rhs, x, tol=tol, maxiter=maxiter, M=reduced_preconditioner)

        # Scatter the interior solution back on top of the boundary values.
        solution = bsr_mv(interior_selection, x, y=solution, transpose=True, alpha=1.0, beta=1.0)
        _store(result, solution, c, columns_2d, device)

    return result


def _as_scalar_type(matrix: BsrMatrix, dtype: type) -> BsrMatrix:
    return matrix if matrix.scalar_type == dtype else bsr_copy(matrix, scalar_type=dtype)


def _column(boundary_values: wp.array, c: int, columns_2d: bool, dtype: type, device: DeviceLike) -> wp.array:
    # Extract column ``c`` (or the whole vector) as ``float32``, then cast to the
    # solve dtype so single- and double-precision paths share one code path.
    if columns_2d:
        single = wp.empty(boundary_values.shape[0], dtype=wp.float32, device=device)
        wp.launch(
            _get_column, dim=boundary_values.shape[0], inputs=[boundary_values, c], outputs=[single], device=device
        )
    else:
        single = boundary_values
    if dtype == wp.float32:
        return single
    out = wp.empty(single.shape[0], dtype=wp.float64, device=device)
    wp.launch(_to_f64, dim=single.shape[0], inputs=[single], outputs=[out], device=device)
    return out


def _store(result: wp.array, solution: wp.array, c: int, columns_2d: bool, device: DeviceLike) -> None:
    if solution.dtype != wp.float32:
        single = wp.empty(solution.shape[0], dtype=wp.float32, device=device)
        wp.launch(_to_f32, dim=solution.shape[0], inputs=[solution], outputs=[single], device=device)
        solution = single
    if not columns_2d:
        wp.copy(result, solution)
    else:
        wp.launch(_set_column, dim=result.shape[0], inputs=[result, c, solution], device=device)
