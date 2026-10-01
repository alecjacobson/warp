# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagonal lumped mass matrices for triangle and tetrahedral meshes.

The public entry point is :func:`massmatrix`. It shares the simplex-index
resolution and element-measure device functions with
:mod:`warp._src.geometry.laplacian`, so the two operators agree on what an
element is and how large it is.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import warp as wp
from warp._src.geometry.laplacian import (
    _resolve_num_points,
    _resolve_simplex_indices,
    _validate_output_matrix,
    triangle_cotangent_weights,
    triangle_double_area,
    triangle_edge_length_sq,
)
from warp._src.sparse import (
    BsrMatrix,
    bsr_from_triplets,
    bsr_set_from_triplets,
)

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


class MassMatrixType(enum.IntEnum):
    """Kind of diagonal lumped mass matrix to assemble."""

    BARYCENTRIC = 0
    """Each element's measure split equally among its vertices.

    Always positive and defined for triangles and tetrahedra. This is
    ``igl::massmatrix`` with ``MASSMATRIX_TYPE_BARYCENTRIC``.
    """
    VORONOI = 1
    """Mixed Voronoi area of Meyer et al., for triangle meshes.

    Each vertex gets its (circumcentric) Voronoi area in non-obtuse triangles,
    and a safe half/quarter split of the area in obtuse ones, so the entries stay
    positive. This is ``igl::massmatrix`` with ``MASSMATRIX_TYPE_VORONOI`` and the
    default ``igl::harmonic`` uses for surfaces. Implemented for triangles only.
    """


@wp.kernel
def _row_indices(out_indices: wp.array[int]):
    # Fill ``out_indices`` with 0, 1, 2, ..., used as the row and column indices
    # of a diagonal matrix so it can be assembled through the triplet path.
    i = wp.tid()
    out_indices[i] = i


@wp.kernel
def triangle_barycentric_mass(points: wp.array[wp.vec3], indices: wp.array[int], out_diagonal: wp.array[float]):
    # Each triangle contributes a third of its area to each of its three
    # vertices. Summed over incident triangles this is the barycentric lumped
    # mass, the row sum of the consistent P1 mass matrix.
    tri = wp.tid()
    i0 = indices[3 * tri + 0]
    i1 = indices[3 * tri + 1]
    i2 = indices[3 * tri + 2]

    area = 0.5 * triangle_double_area(points[i0], points[i1], points[i2])
    share = area / 3.0

    wp.atomic_add(out_diagonal, i0, share)
    wp.atomic_add(out_diagonal, i1, share)
    wp.atomic_add(out_diagonal, i2, share)


@wp.kernel
def triangle_voronoi_mass(points: wp.array[wp.vec3], indices: wp.array[int], out_diagonal: wp.array[float]):
    # The mixed Voronoi area of Meyer et al. In a non-obtuse triangle each vertex
    # receives its circumcentric Voronoi area; in an obtuse one the circumcenter
    # lies outside, so the obtuse vertex gets half the triangle area and the other
    # two a quarter each, which keeps every entry positive.
    tri = wp.tid()
    v = wp.vec3i(indices[3 * tri + 0], indices[3 * tri + 1], indices[3 * tri + 2])
    p0 = points[v[0]]
    p1 = points[v[1]]
    p2 = points[v[2]]

    area = 0.5 * triangle_double_area(p0, p1, p2)
    # ``triangle_cotangent_weights`` returns half the cotangent of each corner
    # angle, so a negative entry marks the obtuse corner.
    half_cot = triangle_cotangent_weights(p0, p1, p2)
    edge_sq = triangle_edge_length_sq(p0, p1, p2)

    if half_cot[0] < 0.0 or half_cot[1] < 0.0 or half_cot[2] < 0.0:
        for k in range(3):
            wp.atomic_add(out_diagonal, v[k], wp.where(half_cot[k] < 0.0, area * 0.5, area * 0.25))
    else:
        # Voronoi area at vertex ``k`` is an eighth of the sum, over its two
        # incident edges, of squared edge length times the cotangent of the
        # opposite angle. ``edge_sq[m]`` is the edge opposite vertex ``m`` and
        # ``2 * half_cot[m]`` the cotangent at vertex ``m``, giving the quarter
        # factor below.
        for k in range(3):
            a = (k + 1) % 3
            b = (k + 2) % 3
            wp.atomic_add(out_diagonal, v[k], 0.25 * (edge_sq[a] * half_cot[a] + edge_sq[b] * half_cot[b]))


@wp.kernel
def tet_barycentric_mass(points: wp.array[wp.vec3], indices: wp.array[int], out_diagonal: wp.array[float]):
    # Each tetrahedron contributes a quarter of its volume to each of its four
    # vertices, the tetrahedral analogue of the triangle rule above.
    tet = wp.tid()
    i0 = indices[4 * tet + 0]
    i1 = indices[4 * tet + 1]
    i2 = indices[4 * tet + 2]
    i3 = indices[4 * tet + 3]

    e1 = points[i1] - points[i0]
    e2 = points[i2] - points[i0]
    e3 = points[i3] - points[i0]
    m = wp.mat33(
        e1[0],
        e2[0],
        e3[0],
        e1[1],
        e2[1],
        e3[1],
        e1[2],
        e2[2],
        e3[2],
    )
    volume = wp.abs(wp.determinant(m)) / 6.0
    share = volume / 4.0

    wp.atomic_add(out_diagonal, i0, share)
    wp.atomic_add(out_diagonal, i1, share)
    wp.atomic_add(out_diagonal, i2, share)
    wp.atomic_add(out_diagonal, i3, share)


def massmatrix(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    out_mass: BsrMatrix | None = None,
    *,
    kind: MassMatrixType = MassMatrixType.BARYCENTRIC,
    num_points: int | None = None,
    simplex_size: int | None = None,
    device: DeviceLike | None = None,
) -> BsrMatrix:
    """Assemble a diagonal lumped mass matrix of a triangle or tetrahedral mesh.

    The result is a diagonal matrix whose entry for vertex ``i`` is the mass lumped
    onto it. With :attr:`MassMatrixType.BARYCENTRIC` this is the sum over incident
    elements of the element's measure divided by its number of vertices -- a third
    of each incident triangle's area, or a quarter of each incident tetrahedron's
    volume -- matching ``igl::massmatrix`` with ``MASSMATRIX_TYPE_BARYCENTRIC``.
    With :attr:`MassMatrixType.VORONOI` (triangles only) it is the mixed Voronoi
    area of Meyer et al., matching ``MASSMATRIX_TYPE_VORONOI`` and the mass the
    default ``igl::harmonic`` uses for surfaces. Either way the diagonal sums to
    the total area or volume of the mesh and every entry is positive for a
    non-degenerate mesh.

    The operation is differentiable with respect to ``points``: launch it inside a
    :class:`warp.Tape` with ``points.requires_grad`` set to obtain gradients. The
    gradient of an element's contribution is undefined where its measure vanishes,
    so degenerate elements will produce non-finite gradients. Voronoi weighting is
    piecewise (it branches on whether a triangle is obtuse), so its gradient is
    valid within a branch but undefined where a triangle becomes right-angled.

    Args:
        points: Array of vertex positions of type :class:`warp.vec3`.
        indices: Mesh connectivity, given either as a flat :class:`warp.int32`
            array of consecutive per-simplex vertex indices, or as a typed array
            of :class:`warp.vec3i` (triangles) or :class:`warp.vec4i`
            (tetrahedra). A typed array fixes the simplex size from its element
            type; a flat array uses ``simplex_size``.
        out_mass: Optional output matrix of shape ``(num_points, num_points)``
            with scalar (1x1) blocks and :class:`warp.float32` coefficients. Any
            blocks it already holds are discarded. If ``None``, a new matrix is
            allocated.
        kind: Which lumped mass to assemble, as a :class:`MassMatrixType` member.
            :attr:`MassMatrixType.VORONOI` supports triangle meshes only.
        num_points: Number of vertices, which fixes the size of the matrix.
            Passing it alongside ``points`` is an error unless it agrees with the
            length of ``points``.
        simplex_size: Number of vertices per element, either 3 (triangles) or 4
            (tetrahedra). Used only when ``indices`` is a flat ``int32`` array; it
            defaults to 3 and must agree with the element type of a typed
            ``indices`` array.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A diagonal sparse matrix of shape ``(num_points, num_points)`` with scalar
        (1x1) blocks, which is ``out_mass`` when it is provided.

    Raises:
        ValueError: If ``indices`` has an unsupported dtype, if ``simplex_size``
            is neither 3 nor 4 or disagrees with a typed ``indices`` array, if
            ``num_points`` disagrees with ``points``, or if ``out_mass`` is
            provided but its scalar type, block shape, device, or shape does not
            match the expected output.
        NotImplementedError: If :attr:`MassMatrixType.VORONOI` is requested for a
            tetrahedral mesh.
    """
    kind = MassMatrixType(kind)
    device = wp.get_device(device) if device is not None else points.device
    indices, simplex_size = _resolve_simplex_indices(indices, simplex_size)
    num_simplices = indices.shape[0] // simplex_size
    num_points = _resolve_num_points(points, indices, num_points, out_mass, device)

    if out_mass is not None:
        _validate_output_matrix(out_mass, "out_mass", (num_points, num_points), device)

    if kind == MassMatrixType.VORONOI:
        if simplex_size != 3:
            raise NotImplementedError("Voronoi mass is implemented for triangle meshes only.")
        mass_kernel = triangle_voronoi_mass
    else:
        mass_kernel = triangle_barycentric_mass if simplex_size == 3 else tet_barycentric_mass

    diagonal = wp.zeros(num_points, dtype=wp.float32, device=device, requires_grad=points.requires_grad)
    wp.launch(mass_kernel, dim=num_simplices, inputs=[points, indices], outputs=[diagonal], device=device)

    rows = wp.empty(num_points, dtype=wp.int32, device=device)
    wp.launch(_row_indices, dim=num_points, outputs=[rows], device=device)

    if out_mass is None:
        return bsr_from_triplets(num_points, num_points, rows, rows, diagonal, prune_numerical_zeros=False)

    bsr_set_from_triplets(out_mass, rows, rows, diagonal, prune_numerical_zeros=False)
    return out_mass
