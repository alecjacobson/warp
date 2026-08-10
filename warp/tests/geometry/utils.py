# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and reference implementations for :mod:`warp.geometry` tests.

Three kinds of helper live here:

* Mesh builders, which return ``(points, indices)`` as NumPy arrays of shape
  ``(num_vertices, 3)`` and ``(3 * num_triangles,)``. All meshes are generated
  procedurally so the tests stay deterministic and need no asset files.
* Reference implementations in float64 NumPy, written independently of the Warp
  kernels so a comparison against them is a real check rather than a restatement.
* :data:`ARRAY_OPS`, a table describing every array-level entry point, which
  drives the shared-contract and gradient tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import warp as wp
import warp.geometry as geo

##########################################################################
## Mesh builders
##########################################################################


def single_triangle() -> tuple[np.ndarray, np.ndarray]:
    """A single 3-4-5 right triangle in the z=0 plane, area 6."""
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    return points, indices


def equilateral_triangle(side: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A single equilateral triangle: every corner angle is exactly pi/3."""
    h = side * math.sqrt(3.0) / 2.0
    points = np.array([[0.0, 0.0, 0.0], [side, 0.0, 0.0], [side * 0.5, h, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    return points, indices


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    """A closed, outward-oriented tetrahedron with one corner at the origin."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
    return points, indices


def unit_cube(center: tuple[float, float, float] = (0.0, 0.0, 0.0), side: float = 1.0):
    """A closed, outward-oriented axis-aligned cube (8 vertices, 12 triangles)."""
    h = 0.5 * side
    c = np.asarray(center, dtype=np.float64)
    points = (
        np.array(
            [
                [-h, -h, -h],
                [+h, -h, -h],
                [+h, +h, -h],
                [-h, +h, -h],
                [-h, -h, +h],
                [+h, -h, +h],
                [+h, +h, +h],
                [-h, +h, +h],
            ],
            dtype=np.float64,
        )
        + c
    ).astype(np.float32)
    indices = np.array(
        [
            0, 3, 2,  0, 2, 1,  # -z
            4, 5, 6,  4, 6, 7,  # +z
            0, 1, 5,  0, 5, 4,  # -y
            1, 2, 6,  1, 6, 5,  # +x
            2, 3, 7,  2, 7, 6,  # +y
            3, 0, 4,  3, 4, 7,  # -x
        ],
        dtype=np.int32,
    )  # fmt: skip
    return points, indices


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    points = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )  # fmt: skip
    return points, faces


def _subdivide(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four, sharing the new edge midpoints. Preserves winding."""
    verts = list(points)
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = len(verts)
            verts.append(0.5 * (points[a] + points[b]))
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int32)


def icosphere(subdivisions: int = 2, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    """A closed genus-0 sphere approximation, outward-oriented."""
    points, faces = _icosahedron()
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    for _ in range(subdivisions):
        points, faces = _subdivide(points, faces)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points * radius + np.asarray(center, dtype=np.float64)
    return points.astype(np.float32), faces.flatten().astype(np.int32)


def torus(n_major: int = 24, n_minor: int = 12, major_radius: float = 1.0, minor_radius: float = 0.35):
    """A closed genus-1 torus, outward-oriented. Euler characteristic 0."""
    u = np.arange(n_major) * (2.0 * math.pi / n_major)
    v = np.arange(n_minor) * (2.0 * math.pi / n_minor)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    ring = major_radius + minor_radius * np.cos(vv)
    points = np.stack([ring * np.cos(uu), ring * np.sin(uu), minor_radius * np.sin(vv)], axis=-1)
    points = points.reshape(-1, 3)

    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            faces.extend([[a, b, c], [a, c, d]])

    return points.astype(np.float32), np.array(faces, dtype=np.int32).flatten()


def planar_grid(nx: int = 5, ny: int = 5, jitter: float = 0.0, rng=None):
    """An open grid in the z=0 plane, normal +z. Interior vertices have zero curvature.

    A nonzero ``jitter`` displaces interior vertices in-plane, which makes the
    tessellation non-uniform so the vertex-normal weighting schemes disagree.
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)

    if jitter:
        if rng is None:
            rng = np.random.default_rng(0)
        interior = np.ones((nx, ny), dtype=bool)
        interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
        mask = interior.reshape(-1)
        points[mask, :2] += rng.uniform(-jitter, jitter, size=(int(mask.sum()), 2))

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = (i + 1) * ny + (j + 1)
            d = i * ny + (j + 1)
            faces.extend([[a, b, c], [a, c, d]])

    return points.astype(np.float32), np.array(faces, dtype=np.int32).flatten()


def sliver_triangles(heights=(1e-1, 1e-2, 1e-3, 1e-4)):
    """Needle triangles of decreasing height, with closed-form corner angles.

    Each triangle spans ``(0,0,0)``, ``(1,0,0)``, ``(0.5,h,0)``. The two base
    angles are ``atan(2h)`` and the apex angle is ``pi - 2*atan(2h)``, so as ``h``
    shrinks the angles approach 0 and pi -- exactly where ``acos(dot(...))`` loses
    precision and Kahan's half-angle formula does not.
    """
    points = []
    faces = []
    for k, h in enumerate(heights):
        points.extend([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, float(h), 0.0]])
        faces.append([3 * k, 3 * k + 1, 3 * k + 2])

    expected = np.array(
        [[math.atan(2.0 * h), math.atan(2.0 * h), math.pi - 2.0 * math.atan(2.0 * h)] for h in heights],
        dtype=np.float64,
    )
    return np.array(points, dtype=np.float32), np.array(faces, dtype=np.int32).flatten(), expected


def perturbed_icosphere(rng, subdivisions: int = 1, radius: float = 1.0, amplitude: float = 0.08):
    """A closed genus-0 mesh with irregular geometry, for bulk and gradient checks."""
    points, indices = icosphere(subdivisions=subdivisions, radius=radius)
    points = points + rng.uniform(-amplitude, amplitude, size=points.shape).astype(np.float32)
    return points.astype(np.float32), indices


def to_warp(points_np, indices_np, device, requires_grad: bool = False):
    """Upload a ``(points, indices)`` NumPy pair to Warp arrays on ``device``."""
    points = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=requires_grad)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    return points, indices


##########################################################################
## Reference implementations (float64 NumPy)
##########################################################################


def _corners(points, indices):
    f = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    p = np.asarray(points, dtype=np.float64)
    return p[f[:, 0]], p[f[:, 1]], p[f[:, 2]], f


def ref_triangle_normals(points, indices, normalized=False):
    v0, v1, v2, _ = _corners(points, indices)
    n = np.cross(v1 - v0, v2 - v0)
    if normalized:
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return n


def ref_triangle_areas(points, indices):
    v0, v1, v2, _ = _corners(points, indices)
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def ref_triangle_corner_angles(points, indices):
    """Interior angles at (v0, v1, v2) via the stable half-angle formula."""
    v0, v1, v2, _ = _corners(points, indices)

    def angle_at(x, y, z):
        a = (x - y) / np.linalg.norm(x - y, axis=1, keepdims=True)
        b = (z - y) / np.linalg.norm(z - y, axis=1, keepdims=True)
        return 2.0 * np.arctan2(np.linalg.norm(a - b, axis=1), np.linalg.norm(a + b, axis=1))

    return np.stack([angle_at(v2, v0, v1), angle_at(v0, v1, v2), angle_at(v1, v2, v0)], axis=-1)


def ref_vertex_normals(points, indices, weighting=geo.VertexNormalWeighting.AREA, normalized=False):
    """Accumulate incident face normals onto vertices.

    Note that the ANGLE scheme weights by the *half* angle, matching the
    implementation. Since every weight is scaled equally that only affects the
    magnitude of the unnormalized result, never its direction.
    """
    v0, v1, v2, f = _corners(points, indices)
    n = np.cross(v1 - v0, v2 - v0)

    weights = np.ones((f.shape[0], 3))
    if weighting == geo.VertexNormalWeighting.UNIFORM:
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
    elif weighting == geo.VertexNormalWeighting.ANGLE:
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        weights = 0.5 * ref_triangle_corner_angles(points, indices)

    out = np.zeros((np.asarray(points).shape[0], 3))
    for corner in range(3):
        np.add.at(out, f[:, corner], n * weights[:, corner : corner + 1])

    if normalized:
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out


def ref_vertex_gaussian_curvature(points, indices):
    _, _, _, f = _corners(points, indices)
    angles = ref_triangle_corner_angles(points, indices)
    out = np.full(np.asarray(points).shape[0], 2.0 * math.pi)
    for corner in range(3):
        np.add.at(out, f[:, corner], -angles[:, corner])
    return out


def ref_moments(points, indices):
    """Volume, first moment, and centroidal inertia of the enclosed solid (unit density).

    Integrates over the tetrahedra spanned by the origin and each triangle. For a
    tetrahedron with a vertex at the origin and the rest at p0, p1, p2,
    ``int(x_i x_j) = det/120 * (sum_k p_k[i] p_k[j] + s[i] s[j])`` with
    ``s = p0 + p1 + p2``, from the barycentric identity ``int(l_a l_b) = V (1 + d_ab) / 20``.
    """
    v0, v1, v2, _ = _corners(points, indices)

    det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))  # 6 * signed tet volume
    volume = det.sum() / 6.0

    s = v0 + v1 + v2
    first = (det[:, None] * s).sum(axis=0) / 24.0

    corners = np.stack([v0, v1, v2, s], axis=1)  # (num_triangles, 4, 3)
    raw = np.einsum("t,tki,tkj->ij", det, corners, corners) / 120.0

    central = raw - np.outer(first, first) / volume
    inertia = np.trace(central) * np.eye(3) - central
    return volume, first, inertia


##########################################################################
## Operation table
##########################################################################


@dataclass(frozen=True)
class GeometryOp:
    """Describes one array-level entry point for the shared-contract tests.

    Adding a new operation to :mod:`warp.geometry` should mean adding one entry
    here; the contract and gradient tests are generated from this table.
    """

    name: str
    """Unique identifier, used to build test method names."""

    func: Callable
    """The public entry point in :mod:`warp.geometry`."""

    out_names: tuple[str, ...]
    """Names of the output parameters, in positional order, for error matching."""

    out_dtypes: tuple
    """Warp dtype expected for each output."""

    out_lengths: Callable[[int, int], tuple[int, ...]]
    """Maps ``(num_vertices, num_triangles)`` to the required length of each output."""

    options: dict = field(default_factory=dict)
    """Keyword-only options passed on every call."""

    differentiable: bool = True
    """Whether the operation propagates gradients to ``points``."""

    def __call__(self, points, indices, *outs, device=None):
        return self.func(points, indices, *outs, **self.options, device=device)

    def outputs(self, result):
        """Normalize a return value to a tuple of arrays."""
        return result if isinstance(result, tuple) else (result,)


def _wrong_dtype(dtype):
    """A dtype that is never the expected one, for negative validation tests."""
    return wp.vec3 if dtype != wp.vec3 else wp.float32


ARRAY_OPS = [
    GeometryOp(
        name="triangle_areas",
        func=geo.triangle_areas,
        out_names=("out_areas",),
        out_dtypes=(wp.float32,),
        out_lengths=lambda nv, nt: (nt,),
    ),
    GeometryOp(
        name="triangle_corner_angles",
        func=geo.triangle_corner_angles,
        out_names=("out_angles",),
        out_dtypes=(wp.vec3,),
        out_lengths=lambda nv, nt: (nt,),
    ),
    GeometryOp(
        name="triangle_normals",
        func=geo.triangle_normals,
        out_names=("out_normals",),
        out_dtypes=(wp.vec3,),
        out_lengths=lambda nv, nt: (nt,),
    ),
    GeometryOp(
        name="triangle_normals_normalized",
        func=geo.triangle_normals,
        out_names=("out_normals",),
        out_dtypes=(wp.vec3,),
        out_lengths=lambda nv, nt: (nt,),
        options={"normalized": True},
    ),
    GeometryOp(
        name="vertex_normals",
        func=geo.vertex_normals,
        out_names=("out_normals",),
        out_dtypes=(wp.vec3,),
        out_lengths=lambda nv, nt: (nv,),
    ),
    GeometryOp(
        name="vertex_normals_angle_normalized",
        func=geo.vertex_normals,
        out_names=("out_normals",),
        out_dtypes=(wp.vec3,),
        out_lengths=lambda nv, nt: (nv,),
        options={"weighting": geo.VertexNormalWeighting.ANGLE, "normalized": True},
    ),
    GeometryOp(
        name="vertex_gaussian_curvature",
        func=geo.vertex_gaussian_curvature,
        out_names=("out_curvature",),
        out_dtypes=(wp.float32,),
        out_lengths=lambda nv, nt: (nv,),
    ),
    GeometryOp(
        name="moments",
        func=geo.moments,
        out_names=("out_volume", "out_first_moment", "out_inertia"),
        out_dtypes=(wp.float32, wp.vec3, wp.mat33),
        out_lengths=lambda nv, nt: (1, 1, 1),
    ),
]
