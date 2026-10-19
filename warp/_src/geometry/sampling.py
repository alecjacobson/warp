# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.utils import array_scan, radix_sort_pairs

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

##########################################################################
## Device functions and structs (reusable within kernels)
##
## These are **first-class** citizens: they operate on a single sample and
## may be called from within your own ``@wp.kernel`` definitions.
##########################################################################


@wp.struct
class MeshSample:
    """A point sampled on a triangle mesh, expressed as a face and barycentric coordinates.

    The layout mirrors what the ``wp.mesh_query_*`` builtins return: a triangle
    index plus barycentric coordinates. Recover the world-space position with
    ``wp.mesh_eval_position(mesh, sample.face, sample.uv[0], sample.uv[1])``.
    """

    face: wp.int32
    """Index of the sampled triangle (a face of the mesh)."""

    uv: wp.vec2
    """Barycentric coordinates ``(u, v)`` of the sample within :attr:`face`. The
    third coordinate is ``1 - u - v``."""


@wp.struct
class UniformSamplerState:
    """Device-side state of a :class:`UniformSampler`, passed to :func:`draw_sample`.

    It bundles the mesh identifier with the cumulative area distribution needed
    to pick a triangle with probability proportional to its area. Obtain an
    instance from ``UniformSampler.state`` and pass it as a kernel argument.
    """

    mesh: wp.uint64
    """Identifier of the :class:`warp.Mesh` being sampled."""

    cdf: wp.array[wp.float64]
    """Normalized cumulative distribution over triangle areas, ending at ``1``."""


@wp.func
def sample_barycentrics(rng: wp.uint32) -> wp.vec2:
    """Draw barycentric coordinates uniformly over the unit triangle.

    Args:
        rng: Random number generator state. Passed by value, so the caller's
            state is not advanced; use a distinct state for each draw.

    Returns:
        Barycentric coordinates ``(u, v)`` uniformly distributed over a triangle,
        with ``u >= 0``, ``v >= 0``, and ``u + v <= 1``.
    """
    r0 = wp.randf(rng)
    r1 = wp.randf(rng)
    # Map the unit square to the triangle so that area is preserved (Turk, 1990).
    su = wp.sqrt(r0)
    u = 1.0 - su
    v = r1 * su
    return wp.vec2(u, v)


@wp.func
def curvature_corrected_distance(p1: wp.vec3, n1: wp.vec3, p2: wp.vec3, n2: wp.vec3) -> wp.float32:
    """Approximate the geodesic (on-surface) distance between two surface points.

    This is *not* a true geodesic (no mesh connectivity or shortest-path solve);
    it corrects the Euclidean chord by the amount the surface normal turns between
    the two points, using the fast normal-based estimate of Bowers et al.
    (SIGGRAPH Asia 2010). It needs only the two points and their unit surface
    normals, and integrates the differential arc length of a curve whose normal
    turns linearly from ``n1`` to ``n2`` along the connecting direction. The
    estimate is never smaller than the Euclidean distance, equals it on a flat
    region (``n1 == n2``), and is *exact* on a sphere -- but it models the surface
    between the points as a single constant-curvature arc, so it is only reliable
    when the points are close relative to the surface's curvature (the regime the
    Poisson-disk conflict test uses it in).

    Args:
        p1: First surface point.
        n1: Unit surface normal at ``p1``.
        p2: Second surface point.
        n2: Unit surface normal at ``p2``.

    Returns:
        The approximate geodesic distance ``dg >= ||p2 - p1||``.
    """
    d = p2 - p1
    de = wp.length(d)
    if de == 0.0:
        return 0.0
    v = d / de
    c1 = wp.clamp(wp.dot(n1, v), -1.0, 1.0)
    c2 = wp.clamp(wp.dot(n2, v), -1.0, 1.0)
    denom = c1 - c2
    if wp.abs(denom) < 1.0e-6:
        # Limit c2 -> c1: dg = de / sqrt(1 - c1^2). Guard the near-fold case where
        # the normal aligns with the connecting direction (distance -> large).
        s = wp.max(1.0 - c1 * c1, 1.0e-12)
        return de / wp.sqrt(s)
    return de * (wp.asin(c1) - wp.asin(c2)) / denom


# Scale that turns the second ``randf`` draw (a multiple of 2**-24 in [0, 1)) into the low
# 24 bits of a 48-bit random number.
_FACE_RANDOM_LOW_SCALE = wp.constant(wp.float64(5.9604644775390625e-8))  # 2**-24


@wp.func
def draw_sample(state: UniformSamplerState, rng: wp.uint32) -> MeshSample:
    """Draw one point uniformly over the surface of a mesh.

    A triangle is chosen with probability proportional to its area via a binary
    search of the cumulative distribution in ``state``, then a point is drawn
    uniformly within that triangle. Callable from within a :func:`warp.kernel`.

    The triangle is selected with a 48-bit random number against a ``float64``
    cumulative distribution. A triangle holding a fraction ``p`` of the total
    area is therefore selected with a relative error of at most about
    ``2**-48 / p``, which is ``num_triangles * 2**-48`` for equal-area triangles:
    about ``4e-9`` for a million triangles and ``3e-6`` at the 700 million
    triangles that fit in ``int32`` indices. Triangles holding less than about
    ``2**-48`` of the total area may never be selected.

    Args:
        state: Sampler state, typically ``UniformSampler.state`` passed as a
            kernel argument.
        rng: Random number generator state. Warp functions receive it by value,
            so the caller's ``rng`` is *not* advanced: calling ``draw_sample`` twice
            with the same ``rng`` returns the same sample. Use a distinct state
            (e.g. ``wp.rand_init(seed, tid * n + i)``) for each draw.

    Returns:
        The sampled :class:`MeshSample` (a triangle index and barycentrics).
    """
    high = wp.randf(rng)
    low = wp.randf(rng)
    r = wp.float64(high) + wp.float64(low) * _FACE_RANDOM_LOW_SCALE

    num_faces = state.cdf.shape[0]
    face = wp.lower_bound(state.cdf, r)
    # Triangle ``i`` owns the half-open interval ``[cdf[i - 1], cdf[i])``, so a draw
    # exactly on a CDF value belongs to the next triangle. This also skips any
    # zero-area triangles, whose CDF value equals their predecessor's.
    while face < num_faces - 1 and state.cdf[face] <= r:
        face += 1

    sample = MeshSample()
    sample.face = wp.min(face, num_faces - 1)
    sample.uv = sample_barycentrics(rng)
    return sample


# register API functions so they appear in the documentation

wp._src.context.register_api_function(
    sample_barycentrics,
    module="warp.geometry",
    differentiable=False,
)
wp._src.context.register_api_function(
    draw_sample,
    module="warp.geometry",
    differentiable=False,
)


##########################################################################
## Array-level operations and the host-side sampler
##########################################################################


# Codes written to the device-side error flag. The highest code wins when several are
# raised, so root causes (invalid areas) outrank their consequences (a bad total).
_ERROR_NONE = 0
_ERROR_GRID_AXES = 1
_ERROR_GRID_CELLS = 2
_ERROR_ZERO_AREA = 3
_ERROR_INVALID_AREAS = 4


@wp.kernel(enable_backward=False)
def _triangle_areas_kernel(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    out_areas: wp.array[wp.float64],
    out_error: wp.array[wp.int32],
):
    tri = wp.tid()
    v0 = points[indices[tri * 3 + 0]]
    v1 = points[indices[tri * 3 + 1]]
    v2 = points[indices[tri * 3 + 2]]
    area = 0.5 * wp.length(wp.cross(v1 - v0, v2 - v0))
    if not wp.isfinite(area):
        wp.atomic_max(out_error, 0, 4)  # _ERROR_INVALID_AREAS
    out_areas[tri] = wp.float64(area)


@wp.kernel(enable_backward=False)
def _convert_areas_kernel(
    areas: wp.array[wp.float32],
    out_areas: wp.array[wp.float64],
    out_error: wp.array[wp.int32],
):
    tri = wp.tid()
    area = areas[tri]
    if area < 0.0 or not wp.isfinite(area):
        wp.atomic_max(out_error, 0, 4)  # _ERROR_INVALID_AREAS
    out_areas[tri] = wp.float64(area)


@wp.kernel(enable_backward=False)
def _normalize_cdf_kernel(
    cumulative: wp.array[wp.float64],
    out_cdf: wp.array[wp.float64],
    out_error: wp.array[wp.int32],
):
    tri = wp.tid()
    total = cumulative[cumulative.shape[0] - 1]
    if total > wp.float64(0.0) and wp.isfinite(total):
        out_cdf[tri] = cumulative[tri] / total
    else:
        out_cdf[tri] = wp.float64(1.0)
        wp.atomic_max(out_error, 0, 3)  # _ERROR_ZERO_AREA


@wp.kernel(enable_backward=False)
def _store_total_area_kernel(cumulative: wp.array[wp.float64], out_total_area: wp.array[wp.float64]):
    out_total_area[0] = cumulative[cumulative.shape[0] - 1]


@wp.kernel(enable_backward=False)
def _draw_kernel(
    state: UniformSamplerState,
    seed: wp.int32,
    out_faces: wp.array[wp.int32],
    out_uv: wp.array[wp.vec2],
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    sample = draw_sample(state, rng)
    out_faces[tid] = sample.face
    out_uv[tid] = sample.uv


def _as_vec3_array(points, device) -> wp.array:
    if isinstance(points, wp.array):
        return points.to(device) if points.device != device else points
    return wp.array(np.asarray(points, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)


def _as_index_array(faces, device) -> wp.array:
    if isinstance(faces, wp.array):
        arr = faces if faces.dtype == wp.int32 else wp.array(faces.numpy(), dtype=wp.int32, device=faces.device)
        return arr.to(device) if arr.device != device else arr
    return wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)


def _as_float_array(values, device) -> wp.array:
    if isinstance(values, wp.array):
        arr = values if values.dtype == wp.float32 else wp.array(values.numpy(), dtype=wp.float32, device=values.device)
        arr = arr if arr.ndim == 1 else arr.flatten()
        return arr.to(device) if arr.device != device else arr
    return wp.array(np.asarray(values, dtype=np.float32).reshape(-1), dtype=wp.float32, device=device)


def _raise_for_error(code: int, *, supplied_areas: bool = False, radius: float | None = None) -> None:
    """Raise the ``ValueError`` matching a device-side error code (a no-op for ``0``)."""
    if code == _ERROR_NONE:
        return
    if code == _ERROR_INVALID_AREAS:
        if supplied_areas:
            raise ValueError("`face_areas` must be non-negative and finite.")
        raise ValueError("Triangle areas must be finite; check that the vertex positions are finite.")
    if code == _ERROR_ZERO_AREA:
        raise ValueError("The mesh has zero total surface area, so there is nothing to sample.")
    if code == _ERROR_GRID_AXES:
        raise ValueError(
            f"`radius` ({radius}) is too small for the mesh extent: the spatial grid would need more than "
            f"{np.iinfo(np.int32).max} cells along an axis."
        )
    if code == _ERROR_GRID_CELLS:
        raise ValueError(
            f"`radius` ({radius}) is too small for the mesh extent: the flattened cell ids of the spatial grid "
            "would overflow."
        )
    raise ValueError(f"Unknown sampling error code {code}.")


class UniformSampler:
    """Draw points uniformly over the surface of a triangle mesh.

    The sampler weights each triangle by its area so that samples are spread
    evenly across the surface even when the tessellation is non-uniform. It
    precomputes a cumulative area distribution (CDF) once at construction; each
    subsequent draw is a binary search plus a within-triangle sample.

    Two ways to draw are provided:

    * :meth:`sample` (host side) launches a kernel and returns arrays of faces
      and barycentric coordinates -- convenient for one-shot sampling.
    * :func:`draw_sample` (device side) samples a single point from within your own
      kernel, given the sampler's ``state``.

    When the vertex positions move but the triangles do not, :meth:`update`
    rebuilds the area distribution from the new positions without any host
    synchronization, so it can be recorded in a CUDA graph. The vertex positions are
    read from the array passed as ``points`` when it is a :class:`warp.array` on
    ``device``; otherwise the sampler holds its own copy, available as
    ``points``.

    Args:
        points: Vertex positions, either a :class:`warp.array` of
            :class:`warp.vec3` or an array-like of shape ``(num_vertices, 3)``.
        faces: Triangle vertex indices, either a flat :class:`warp.array` of
            :class:`warp.int32` (length ``3 * num_triangles``) or an array-like
            reshapeable to ``(num_triangles, 3)``.
        face_areas: Optional per-triangle areas of length ``num_triangles``,
            either a :class:`warp.array` of :class:`warp.float32` or an array-like.
            Supply them to reuse areas the caller already has; when ``None`` they
            are computed from ``points`` and ``faces``. A :class:`warp.float32`
            array on ``device`` is read in place by :meth:`update`; anything else
            is a snapshot.
        device: Device on which to build the sampler. Defaults to the device of
            ``points`` when it is a :class:`warp.array`, otherwise the current
            device.

    Attributes:
        total_area: Total surface area of the mesh when the sampler was built, as a
            host ``float``. Not refreshed by :meth:`update`, which never reads back to
            the host.
        total_area_device: :class:`warp.array` of :class:`warp.float64` and shape
            ``(1,)`` holding the current total surface area. :meth:`update` refreshes it
            on the device, so device code and CUDA graphs can read it without a sync.
    """

    def __init__(self, points, faces, *, face_areas=None, device: DeviceLike | None = None):
        if device is None:
            device = points.device if isinstance(points, wp.array) else wp.get_device()
        self.device = wp.get_device(device)

        self.points = _as_vec3_array(points, self.device)
        self.indices = _as_index_array(faces, self.device)
        if self.indices.ndim != 1 or self.indices.shape[0] % 3 != 0:
            raise ValueError(
                f"`faces` must contain a multiple of 3 indices (flat or (num_triangles, 3)), got shape {self.indices.shape}."
            )
        self.num_triangles = self.indices.shape[0] // 3
        if self.num_triangles == 0:
            raise ValueError("`faces` must describe at least one triangle.")

        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

        self._face_areas = None
        if face_areas is not None:
            self._face_areas = _as_float_array(face_areas, self.device)
            if self._face_areas.shape[0] != self.num_triangles:
                raise ValueError(
                    f"`face_areas` must have length num_triangles ({self.num_triangles}), "
                    f"got {self._face_areas.shape[0]}."
                )

        # Workspace for the area distribution, allocated once so that ``update`` never
        # allocates. ``_error`` collects device-side error codes (see ``_ERROR_*``).
        self._areas = wp.empty(self.num_triangles, dtype=wp.float64, device=self.device)
        self._cumulative = wp.empty(self.num_triangles, dtype=wp.float64, device=self.device)
        self.cdf = wp.empty(self.num_triangles, dtype=wp.float64, device=self.device)
        self.total_area_device = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._error = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.state = UniformSamplerState()
        self.state.mesh = self.mesh.id
        self.state.cdf = self.cdf

        self.update()
        self.check_error()

        # Handy for sizing candidate pools (e.g. Poisson-disk sampling).
        self.total_area = float(self.total_area_device.numpy()[0])

    def check_error(self) -> None:
        """Raise ``ValueError`` if the last :meth:`update` found invalid input.

        :meth:`update` cannot raise because it never reads back to the host. It
        records a device-side error instead, which this method reads, so calling it
        synchronizes the device.
        """
        _raise_for_error(int(self._error.numpy()[0]), supplied_areas=self._face_areas is not None)

    def update(self) -> None:
        """Rebuild the area distribution from the current vertex positions.

        Call this after the vertex positions (or a ``face_areas`` array supplied at
        construction) have changed in place while the triangles stay the same.
        It launches only device work and allocates nothing, so it can be captured
        in a CUDA graph. Invalid input, such as non-finite areas or a zero total
        area, is recorded on the device and raised by :meth:`check_error`.
        :attr:`total_area_device` holds the new total area; :attr:`total_area`
        keeps the value from construction.
        """
        self._error.zero_()
        if self._face_areas is None:
            wp.launch(
                _triangle_areas_kernel,
                dim=self.num_triangles,
                inputs=[self.points, self.indices],
                outputs=[self._areas, self._error],
                device=self.device,
            )
        else:
            wp.launch(
                _convert_areas_kernel,
                dim=self.num_triangles,
                inputs=[self._face_areas],
                outputs=[self._areas, self._error],
                device=self.device,
            )
        array_scan(self._areas, self._cumulative, inclusive=True)
        wp.launch(
            _normalize_cdf_kernel,
            dim=self.num_triangles,
            inputs=[self._cumulative],
            outputs=[self.cdf, self._error],
            device=self.device,
        )
        wp.launch(
            _store_total_area_kernel,
            dim=1,
            inputs=[self._cumulative],
            outputs=[self.total_area_device],
            device=self.device,
        )

    def sample(
        self,
        num_samples: int,
        *,
        seed: int = 0,
        out_faces: wp.array | None = None,
        out_uv: wp.array | None = None,
    ) -> tuple[wp.array, wp.array]:
        """Draw ``num_samples`` points uniformly over the mesh surface.

        Args:
            num_samples: Number of points to draw.
            seed: Seed for the random number generator. Different seeds give
                independent sample sets.
            out_faces: Optional output array of :class:`warp.int32` and length
                ``num_samples`` for the sampled face indices. Allocated if
                ``None``.
            out_uv: Optional output array of :class:`warp.vec2` and length
                ``num_samples`` for the barycentric coordinates. Allocated if
                ``None``.

        Returns:
            A tuple ``(faces, uv)`` where ``faces`` are triangle indices and
            ``uv`` are barycentric coordinates within each face, matching the
            layout of :class:`MeshSample`.
        """
        if out_faces is None:
            out_faces = wp.empty(num_samples, dtype=wp.int32, device=self.device)
        if out_uv is None:
            out_uv = wp.empty(num_samples, dtype=wp.vec2, device=self.device)

        wp.launch(
            _draw_kernel,
            dim=num_samples,
            inputs=[self.state, seed],
            outputs=[out_faces, out_uv],
            device=self.device,
        )
        return out_faces, out_uv

    def sample_points(self, num_samples: int, *, seed: int = 0) -> wp.array:
        """Draw ``num_samples`` points and return their world-space positions.

        A convenience wrapper around :meth:`sample` that evaluates each sample's
        position on the mesh.

        Args:
            num_samples: Number of points to draw.
            seed: Seed for the random number generator.

        Returns:
            A :class:`warp.array` of :class:`warp.vec3` positions on the surface.
        """
        faces, uv = self.sample(num_samples, seed=seed)
        positions = wp.empty(num_samples, dtype=wp.vec3, device=self.device)
        wp.launch(
            _eval_positions_kernel,
            dim=num_samples,
            inputs=[self.mesh.id, faces, uv],
            outputs=[positions],
            device=self.device,
        )
        return positions


@wp.kernel(enable_backward=False)
def _eval_positions_kernel(
    mesh: wp.uint64,
    faces: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out_positions: wp.array[wp.vec3],
):
    tid = wp.tid()
    p = uv[tid]
    out_positions[tid] = wp.mesh_eval_position(mesh, faces[tid], p[0], p[1])


def uniformly_sample(
    points,
    faces,
    num_samples: int,
    *,
    seed: int = 0,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array]:
    """Sample points uniformly over the surface of a triangle mesh.

    Each triangle is chosen with probability proportional to its area, then a
    point is drawn uniformly within it, so the samples are spread evenly across
    the surface regardless of how the mesh is tessellated. This is a convenience
    wrapper that builds a :class:`UniformSampler` and samples once; construct a
    :class:`UniformSampler` directly to amortize the setup over many draws.

    Args:
        points: Vertex positions, either a :class:`warp.array` of
            :class:`warp.vec3` or an array-like of shape ``(num_vertices, 3)``.
        faces: Triangle vertex indices, either a flat :class:`warp.array` of
            :class:`warp.int32` (length ``3 * num_triangles``) or an array-like
            reshapeable to ``(num_triangles, 3)``.
        num_samples: Number of points to draw.
        seed: Seed for the random number generator.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(faces, uv)`` where ``faces`` is a :class:`warp.array` of
        :class:`warp.int32` triangle indices and ``uv`` is a :class:`warp.array`
        of :class:`warp.vec2` barycentric coordinates, matching the layout of
        :class:`MeshSample`.
    """
    sampler = UniformSampler(points, faces, device=device)
    return sampler.sample(num_samples, seed=seed)


##########################################################################
## Parallel Poisson-disk (blue-noise) sampling on surfaces
##
## Bowers, Wang, Wei and Maletz, "Parallel Poisson Disk Sampling with
## Spectrum Analysis on Surfaces", ACM SIGGRAPH Asia 2010.
##########################################################################

# Candidate status values used by the parallel maximal-independent-set solver.
_POISSON_ACTIVE = wp.constant(wp.int32(0))
_POISSON_ACCEPTED = wp.constant(wp.int32(1))


@wp.kernel(enable_backward=False)
def _poisson_priority_kernel(seed: wp.int32, out_priority: wp.array[wp.float32]):
    # The candidate draw seeds its generator with ``(seed, tid)``. Salting the seed here
    # gives the priority its own stream; otherwise a candidate's priority would equal the
    # random number that selected its face, which tracks the face index.
    tid = wp.tid()
    rng = wp.rand_init(seed ^ 1540483477, tid)
    out_priority[tid] = wp.randf(rng)


@wp.kernel(enable_backward=False)
def _poisson_mask_kernel(status: wp.array[wp.int32], error: wp.array[wp.int32], out_mask: wp.array[wp.int32]):
    tid = wp.tid()
    if status[tid] == _POISSON_ACCEPTED and error[0] == 0:
        out_mask[tid] = 1
    else:
        out_mask[tid] = 0


@wp.kernel(enable_backward=False)
def _poisson_count_kernel(offsets: wp.array[wp.int32], error: wp.array[wp.int32], out_count: wp.array[wp.int32]):
    # ``offsets`` is the inclusive prefix sum of the accepted mask, so its last entry is
    # the number of samples. An invalid input yields no samples.
    if error[0] == 0:
        out_count[0] = offsets[offsets.shape[0] - 1]
    else:
        out_count[0] = 0


@wp.kernel(enable_backward=False)
def _poisson_compact_kernel(
    mask: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    in_faces: wp.array[wp.int32],
    in_uv: wp.array[wp.vec2],
    in_points: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
    out_uv: wp.array[wp.vec2],
    out_points: wp.array[wp.vec3],
):
    # ``offsets`` is an inclusive prefix sum of the accepted mask, so a selected
    # candidate writes to slot ``offsets[tid] - 1``.
    tid = wp.tid()
    if mask[tid] == 1:
        idx = offsets[tid] - 1
        out_faces[idx] = in_faces[tid]
        out_uv[idx] = in_uv[tid]
        out_points[idx] = in_points[tid]


##########################################################################
## Single-entry spatial hash + phase groups (the paper's Euclidean path)
##
## Following Bowers et al., the grid cell edge is ``radius / sqrt(3)`` so a
## cell's diagonal equals ``radius``: two points in one cell are always closer
## than ``radius``, hence a cell holds at most one accepted sample. Only the
## few cells that actually contain candidates are ever stored, in a compact
## spatial hash keyed by the integer cell id (``key % table_size`` with linear
## probing) -- so memory scales with the sampled *surface*, not the 3D bounding
## volume. Conflicts span at most ``ceil(sqrt(3)) = 2`` cells, so a candidate
## checks a 5x5x5 block of the hash. Cells are resolved in 27 phase groups
## (coordinates modulo 3): two cells of the same group are at least 3 cells
## apart, so samples accepted within one group can never conflict, and each
## group is one fully parallel pass.
##########################################################################

_POISSON_SEARCH = wp.constant(wp.int32(2))
_POISSON_PERIOD = wp.constant(wp.int32(3))
_POISSON_EMPTY = wp.constant(wp.int64(-1))  # empty hash slot (cell ids are >= 0)


@wp.struct
class _PoissonGrid:
    """Geometry of the spatial grid, kept on the device so that it can follow moving vertices."""

    lo: wp.vec3d
    inv_mu: wp.float64
    dims: wp.vec3i


@wp.func
def _cell_coord(p: wp.vec3, grid: _PoissonGrid) -> wp.vec3i:
    # Subtract the grid origin in float64 *before* scaling: in float32, a mesh far from
    # the origin (or with widely separated parts) can place nearby points more than the
    # conflict-search distance apart in cell coordinates and hide a conflict.
    zero = wp.int64(0)
    cx = wp.clamp(wp.int64((wp.float64(p[0]) - grid.lo[0]) * grid.inv_mu), zero, wp.int64(grid.dims[0] - 1))
    cy = wp.clamp(wp.int64((wp.float64(p[1]) - grid.lo[1]) * grid.inv_mu), zero, wp.int64(grid.dims[1] - 1))
    cz = wp.clamp(wp.int64((wp.float64(p[2]) - grid.lo[2]) * grid.inv_mu), zero, wp.int64(grid.dims[2] - 1))
    return wp.vec3i(wp.int32(cx), wp.int32(cy), wp.int32(cz))


@wp.func
def _cell_id(cx: wp.int32, cy: wp.int32, cz: wp.int32, grid: _PoissonGrid) -> wp.int64:
    # 64-bit so the id space can exceed 2^31 for fine grids on large meshes.
    return (wp.int64(cz) * wp.int64(grid.dims[1]) + wp.int64(cy)) * wp.int64(grid.dims[0]) + wp.int64(cx)


@wp.func
def _cell_phase(c: wp.vec3i) -> wp.int32:
    return (c[2] % _POISSON_PERIOD) * 9 + (c[1] % _POISSON_PERIOD) * 3 + (c[0] % _POISSON_PERIOD)


@wp.func
def _hash_slot0(key: wp.int64, table_size: wp.int32) -> wp.int32:
    # A multiplicative mix keeps sequential cell ids from clustering under linear
    # probing, then reduce into ``[0, table_size)``.
    m = key * wp.int64(2654435761)
    m = m ^ (m >> wp.int64(21))
    r = m % wp.int64(table_size)
    if r < wp.int64(0):
        r = r + wp.int64(table_size)
    return wp.int32(r)


@wp.func
def _hash_find(key: wp.int64, table_size: wp.int32, slot_key: wp.array[wp.int64]) -> wp.int32:
    # Return the slot holding ``key``, or -1 if the cell is absent (empty slot).
    slot = _hash_slot0(key, table_size)
    for _ in range(table_size):
        cur = slot_key[slot]
        if cur == key:
            return slot
        if cur == _POISSON_EMPTY:
            return -1
        slot = slot + 1
        if slot >= table_size:
            slot = 0
    return -1


@wp.kernel(enable_backward=False)
def _poisson_hash_insert_kernel(
    points: wp.array[wp.vec3],
    grid: wp.array[_PoissonGrid],
    table_size: wp.int32,
    slot_key: wp.array[wp.int64],
):
    # Insert each candidate's cell id once (deduplicated by CAS on an empty slot).
    tid = wp.tid()
    g = grid[0]
    c = _cell_coord(points[tid], g)
    key = _cell_id(c[0], c[1], c[2], g)
    slot = _hash_slot0(key, table_size)
    for _ in range(table_size):
        cur = slot_key[slot]
        if cur == key:
            return
        if cur == _POISSON_EMPTY:
            old = wp.atomic_cas(slot_key, slot, _POISSON_EMPTY, key)
            if old == _POISSON_EMPTY or old == key:
                return
        slot = slot + 1
        if slot >= table_size:
            slot = 0


@wp.kernel(enable_backward=False)
def _poisson_setup_kernel(
    points: wp.array[wp.vec3],
    grid: wp.array[_PoissonGrid],
    table_size: wp.int32,
    slot_key: wp.array[wp.int64],
    out_slot: wp.array[wp.int32],
    out_phase: wp.array[wp.int32],
):
    # Cache each candidate's hash slot and phase group so the phase passes below
    # avoid recomputing them.
    tid = wp.tid()
    g = grid[0]
    c = _cell_coord(points[tid], g)
    out_slot[tid] = _hash_find(_cell_id(c[0], c[1], c[2], g), table_size, slot_key)
    out_phase[tid] = _cell_phase(c)


@wp.func
def _cell_free(
    pos: wp.vec3,
    c: wp.vec3i,
    grid: _PoissonGrid,
    table_size: wp.int32,
    slot_key: wp.array[wp.int64],
    slot_sample: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    r_sq: wp.float32,
) -> bool:
    # True if no accepted sample in the surrounding 5x5x5 block is within radius.
    for dz in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
        nz = c[2] + dz
        if nz >= 0 and nz < grid.dims[2]:
            for dy in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                ny = c[1] + dy
                if ny >= 0 and ny < grid.dims[1]:
                    for dx in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                        nx = c[0] + dx
                        if nx >= 0 and nx < grid.dims[0]:
                            slot = _hash_find(_cell_id(nx, ny, nz, grid), table_size, slot_key)
                            if slot >= 0:
                                s = slot_sample[slot]
                                if s >= 0:
                                    d = points[s] - pos
                                    if wp.dot(d, d) < r_sq:
                                        return False
    return True


@wp.kernel(enable_backward=False)
def _poisson_phase_max_kernel(
    phase: wp.int32,
    points: wp.array[wp.vec3],
    grid: wp.array[_PoissonGrid],
    table_size: wp.int32,
    radius: wp.float32,
    slot_key: wp.array[wp.int64],
    slot_sample: wp.array[wp.int32],
    cand_slot: wp.array[wp.int32],
    cand_phase: wp.array[wp.int32],
    priority: wp.array[wp.float32],
    status: wp.array[wp.int32],
    free: wp.array[wp.int32],
    cell_best: wp.array[wp.float32],
):
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase:
        return
    g = grid[0]
    p = points[tid]
    c = _cell_coord(p, g)
    if _cell_free(p, c, g, table_size, slot_key, slot_sample, points, radius * radius):
        free[tid] = 1
        wp.atomic_max(cell_best, cand_slot[tid], priority[tid])
    else:
        free[tid] = 0


@wp.kernel(enable_backward=False)
def _poisson_phase_pick_kernel(
    phase: wp.int32,
    cand_slot: wp.array[wp.int32],
    cand_phase: wp.array[wp.int32],
    priority: wp.array[wp.float32],
    status: wp.array[wp.int32],
    free: wp.array[wp.int32],
    cell_best: wp.array[wp.float32],
    cell_winner: wp.array[wp.int32],
):
    # Among conflict-free candidates sharing the winning priority in a cell, the
    # lowest index claims it (a deterministic tie-break).
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase or free[tid] == 0:
        return
    slot = cand_slot[tid]
    if priority[tid] == cell_best[slot]:
        wp.atomic_min(cell_winner, slot, tid)


@wp.kernel(enable_backward=False)
def _poisson_phase_accept_kernel(
    phase: wp.int32,
    cand_slot: wp.array[wp.int32],
    cand_phase: wp.array[wp.int32],
    status: wp.array[wp.int32],
    free: wp.array[wp.int32],
    cell_winner: wp.array[wp.int32],
    slot_sample: wp.array[wp.int32],
):
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase or free[tid] == 0:
        return
    slot = cand_slot[tid]
    if cell_winner[slot] == tid:
        slot_sample[slot] = tid
        status[tid] = _POISSON_ACCEPTED


##########################################################################
## Geodesic variant (optional): identical single-entry hash + phase groups,
## but the conflict test uses the approximate geodesic distance instead of the
## Euclidean one. Only the "free" (conflict-check) kernel differs; the pick and
## accept passes are shared. It is a separate kernel with its own arguments, so
## the Euclidean kernels above take no extra arguments and gain no branch (the
## solver picks which kernel to launch).
##########################################################################


@wp.func
def _cell_free_geodesic(
    pos: wp.vec3,
    normal: wp.vec3,
    c: wp.vec3i,
    grid: _PoissonGrid,
    table_size: wp.int32,
    slot_key: wp.array[wp.int64],
    slot_sample: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radius: wp.float32,
    r_sq: wp.float32,
) -> bool:
    # As _cell_free, but a Euclidean-close accepted sample only conflicts when its
    # approximate geodesic distance is also below the radius. dg >= de, so the
    # cheap Euclidean test still prunes everything beyond the radius first.
    for dz in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
        nz = c[2] + dz
        if nz >= 0 and nz < grid.dims[2]:
            for dy in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                ny = c[1] + dy
                if ny >= 0 and ny < grid.dims[1]:
                    for dx in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                        nx = c[0] + dx
                        if nx >= 0 and nx < grid.dims[0]:
                            slot = _hash_find(_cell_id(nx, ny, nz, grid), table_size, slot_key)
                            if slot >= 0:
                                s = slot_sample[slot]
                                if s >= 0:
                                    d = points[s] - pos
                                    if wp.dot(d, d) < r_sq:
                                        if curvature_corrected_distance(pos, normal, points[s], normals[s]) < radius:
                                            return False
    return True


@wp.kernel(enable_backward=False)
def _poisson_phase_max_geodesic_kernel(
    phase: wp.int32,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    grid: wp.array[_PoissonGrid],
    table_size: wp.int32,
    radius: wp.float32,
    slot_key: wp.array[wp.int64],
    slot_sample: wp.array[wp.int32],
    cand_slot: wp.array[wp.int32],
    cand_phase: wp.array[wp.int32],
    priority: wp.array[wp.float32],
    status: wp.array[wp.int32],
    free: wp.array[wp.int32],
    cell_best: wp.array[wp.float32],
):
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase:
        return
    g = grid[0]
    p = points[tid]
    c = _cell_coord(p, g)
    if _cell_free_geodesic(
        p, normals[tid], c, g, table_size, slot_key, slot_sample, points, normals, radius, radius * radius
    ):
        free[tid] = 1
        wp.atomic_max(cell_best, cand_slot[tid], priority[tid])
    else:
        free[tid] = 0


@wp.kernel(enable_backward=False)
def _eval_normals_kernel(
    mesh: wp.uint64,
    faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
):
    tid = wp.tid()
    out_normals[tid] = wp.mesh_eval_face_normal(mesh, faces[tid])


@wp.kernel(enable_backward=False)
def _bounds_kernel(
    points: wp.array[wp.vec3],
    out_lo: wp.array[wp.float32],
    out_hi: wp.array[wp.float32],
):
    tid = wp.tid()
    p = points[tid]
    for k in range(3):
        wp.atomic_min(out_lo, k, p[k])
        wp.atomic_max(out_hi, k, p[k])


def _bounding_box(points: wp.array, device) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of ``points``, reduced on the GPU (only 6 floats are
    read back, rather than the whole array)."""
    lo = wp.full(3, 1.0e30, dtype=wp.float32, device=device)
    hi = wp.full(3, -1.0e30, dtype=wp.float32, device=device)
    wp.launch(_bounds_kernel, dim=points.shape[0], inputs=[points], outputs=[lo, hi], device=device)
    return lo.numpy(), hi.numpy()


@wp.kernel(enable_backward=False)
def _poisson_grid_kernel(
    bounds_lo: wp.array[wp.float32],
    bounds_hi: wp.array[wp.float32],
    inv_mu: wp.float64,
    out_grid: wp.array[_PoissonGrid],
    out_error: wp.array[wp.int32],
):
    # Derive the grid from the vertex bounds on the device, so the whole solve needs no
    # host read-back and can follow moving vertices. An oversized grid is recorded in the
    # error flag (and replaced by a single cell) instead of raising.
    grid = _PoissonGrid()
    grid.inv_mu = inv_mu
    grid.lo = wp.vec3d(wp.float64(bounds_lo[0]), wp.float64(bounds_lo[1]), wp.float64(bounds_lo[2]))
    max_axis = wp.float64(2147483647.0)
    max_cells = wp.float64(9.2e18)  # just below the int64 limit on flattened cell ids
    cells = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    valid = True
    total = wp.float64(1.0)
    for k in range(3):
        count = wp.ceil((wp.float64(bounds_hi[k]) - grid.lo[k]) * inv_mu) + wp.float64(1.0)
        if not (count <= max_axis):  # also rejects NaN
            valid = False
            wp.atomic_max(out_error, 0, 1)  # _ERROR_GRID_AXES
        else:
            cells[k] = wp.max(count, wp.float64(1.0))
        total = total * cells[k]
    if valid and total > max_cells:
        valid = False
        wp.atomic_max(out_error, 0, 2)  # _ERROR_GRID_CELLS
    if valid:
        grid.dims = wp.vec3i(wp.int32(cells[0]), wp.int32(cells[1]), wp.int32(cells[2]))
    else:
        grid.dims = wp.vec3i(1, 1, 1)
    out_grid[0] = grid


@wp.kernel(enable_backward=False)
def _poisson_cellid_kernel(
    points: wp.array[wp.vec3],
    grid: wp.array[_PoissonGrid],
    out_cellid: wp.array[wp.int64],
):
    tid = wp.tid()
    g = grid[0]
    c = _cell_coord(points[tid], g)
    out_cellid[tid] = _cell_id(c[0], c[1], c[2], g)


@wp.kernel(enable_backward=False)
def _poisson_iota_kernel(out: wp.array[wp.int32]):
    tid = wp.tid()
    out[tid] = tid


@wp.kernel(enable_backward=False)
def _poisson_gather_kernel(
    order: wp.array[wp.int32],
    in_points: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    in_uv: wp.array[wp.vec2],
    in_priority: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
    out_uv: wp.array[wp.vec2],
    out_priority: wp.array[wp.float32],
):
    tid = wp.tid()
    j = order[tid]
    out_points[tid] = in_points[j]
    out_faces[tid] = in_faces[j]
    out_uv[tid] = in_uv[j]
    out_priority[tid] = in_priority[j]


# Extension seam: this sampler is radius-driven (count emergent). A future
# count-driven strategy (e.g. weighted sample elimination to an exact count)
# should slot in additively -- keep ``radius`` widen-able to optional and reserve
# the keyword ``num_points`` for the target count, so that it can be added without
# deprecating anything.
class PoissonDiskSampler:
    """Draw a Poisson-disk (blue-noise) point set over a triangle mesh surface.

    Sampling is driven by a target ``radius``: no two returned samples are closer
    than ``radius`` in Euclidean distance, and the set is *maximal* -- no further
    candidate could be added without violating that spacing (so the sample count
    is emergent, not specified). The distribution therefore has the characteristic
    blue-noise spectrum -- suppressed low frequencies and no structured aliasing --
    which makes it well suited to stippling, scattering, remeshing seeds, and
    Monte-Carlo integration.

    The sampler follows Bowers et al., *"Parallel Poisson Disk Sampling with
    Spectrum Analysis on Surfaces"* (SIGGRAPH Asia 2010): it draws a dense pool of
    area-weighted candidates with :class:`UniformSampler`, then resolves conflicts
    entirely in parallel as a priority-based maximal independent set over the
    graph connecting candidates closer than ``radius``.

    By default the minimum distance is Euclidean, the standard approximation to
    geodesic distance, accurate when ``radius`` is small relative to the surface's
    curvature. Set ``geodesic`` to instead measure the approximate on-surface
    distance (``curvature_corrected_distance``): this stops samples on opposite sides of
    a thin feature -- close in 3D but far along the surface -- from over-separating,
    at the cost of a normal per candidate and a slightly heavier conflict test.
    The geodesic path is a strict addition; the Euclidean path is unchanged.

    Args:
        points: Vertex positions, either a :class:`warp.array` of
            :class:`warp.vec3` or an array-like of shape ``(num_vertices, 3)``.
        faces: Triangle vertex indices, either a flat :class:`warp.array` of
            :class:`warp.int32` (length ``3 * num_triangles``) or an array-like
            reshapeable to ``(num_triangles, 3)``.
        radius: Minimum distance between any two samples (Euclidean, or geodesic
            when ``geodesic`` is set).
        num_candidates: Size of the candidate pool. If ``None``, it is set to
            ``candidate_multiplier`` times the theoretical maximal sample count.
        candidate_multiplier: Oversampling factor used when ``num_candidates`` is
            ``None``. Larger values give a denser, more nearly maximal result at
            higher cost.
        seed: Seed for candidate generation and priorities. Fixing it makes the
            result deterministic.
        geodesic: If set, use Bowers et al.'s approximate geodesic metric
            (``curvature_corrected_distance``) for the minimum distance instead of the
            Euclidean one. Keeps a single sample per grid cell: the paper's
            multiple-samples-per-cell extension is intentionally omitted -- it was
            implemented and measured to add nothing under this approximation.
        face_areas: Optional precomputed per-triangle areas forwarded to the
            internal :class:`UniformSampler`; computed from the mesh when ``None``.
        device: Device on which to run. Defaults to the device of ``points``.

    Attributes:
        radius: The minimum-distance radius used.
        total_area: Total surface area of the mesh when the sampler was built, as a host
            ``float``. Not refreshed by :meth:`update`.
        points_buffer: Fixed-capacity :class:`warp.array` of :class:`warp.vec3` backing
            :attr:`points`, with length ``num_candidates``. Only the first
            ``sample_count[0]`` entries are valid.
        faces_buffer: Fixed-capacity :class:`warp.array` backing :attr:`faces`.
        uv_buffer: Fixed-capacity :class:`warp.array` backing :attr:`uv`.
        sample_count: Device-side :class:`warp.array` of :class:`warp.int32` and shape
            ``(1,)`` holding the number of valid samples after the latest solve.

    The host-side properties :attr:`points`, :attr:`faces`, :attr:`uv` and
    :attr:`num_samples` read the result back, so they synchronize the device and
    cannot be used while a CUDA graph is being captured. Device code and CUDA graphs
    use the ``*_buffer`` arrays together with ``sample_count`` instead.

    **Moving vertices and CUDA graphs.** :meth:`update` re-runs the whole pipeline on
    the current vertex positions with no host synchronization and no allocation, so it
    can be captured in a CUDA graph and replayed each frame while the vertices move
    in place. The triangle connectivity, ``radius``, ``num_candidates`` and ``seed``
    are fixed when the graph is captured. Because the candidate count is fixed, pass an
    explicit ``num_candidates`` when the mesh area may change a lot. Pass ``points`` as
    a :class:`warp.array` on ``device`` so that the sampler reads it in place.

    **Memory.** The sampler keeps a workspace of roughly ``150 * num_candidates`` bytes
    (candidates, a sorted copy, and the spatial hash) for as long as it lives, so that
    :meth:`update` never allocates.
    """

    def __init__(
        self,
        points,
        faces,
        radius: float,
        *,
        num_candidates: int | None = None,
        candidate_multiplier: float = 12.0,
        seed: int = 0,
        geodesic: bool = False,
        face_areas=None,
        device: DeviceLike | None = None,
    ):
        if not (0.0 < radius < float("inf")):
            raise ValueError(f"`radius` must be positive and finite, got {radius}.")
        if num_candidates is not None and num_candidates < 1:
            raise ValueError(f"`num_candidates` must be at least 1, got {num_candidates}.")

        self._sampler = UniformSampler(points, faces, face_areas=face_areas, device=device)
        self.device = self._sampler.device
        self.radius = float(radius)
        self.geodesic = bool(geodesic)
        self.seed = int(seed)
        self.total_area = self._sampler.total_area

        # Theoretical maximal count assumes hexagonal packing of disks of radius
        # ``radius / 2``, i.e. one sample per ``sqrt(3)/2 * radius^2`` of area.
        n_est = self.total_area / (0.8660254037844386 * self.radius * self.radius)
        if num_candidates is None:
            num_candidates = int(max(1.0, candidate_multiplier * n_est))
        self.num_candidates = int(num_candidates)
        if 2 * self.num_candidates + 1 > np.iinfo(np.int32).max:
            raise ValueError(
                f"`num_candidates` ({self.num_candidates}) is too large: the spatial hash needs "
                "2 * num_candidates + 1 slots, which must fit in a 32-bit integer."
            )

        # The cell edge is ``radius / sqrt(3)`` (diagonal ``radius``), giving at most one
        # sample per cell.
        self._inv_mu = float(1.7320508075688772 / self.radius)
        self._allocate_workspace()

        self.update(self.seed)
        # Raise now, rather than on first use, if the input was invalid.
        self._read_sample_count()

    def _allocate_workspace(self) -> None:
        """Allocate every array :meth:`update` uses, so that updating never allocates."""
        n = self.num_candidates
        table_size = 2 * n + 1
        device = self.device

        # Candidates in generation order.
        self._cand_points = wp.empty(n, dtype=wp.vec3, device=device)
        self._cand_faces = wp.empty(n, dtype=wp.int32, device=device)
        self._cand_uv = wp.empty(n, dtype=wp.vec2, device=device)
        self._priority = wp.empty(n, dtype=wp.float32, device=device)

        # Grid geometry, derived from the vertex bounds on the device.
        self._bounds_lo = wp.empty(3, dtype=wp.float32, device=device)
        self._bounds_hi = wp.empty(3, dtype=wp.float32, device=device)
        self._grid = wp.empty(1, dtype=_PoissonGrid, device=device)

        # Candidates sorted by grid cell. radix_sort_pairs needs 2*n storage for both
        # keys and values.
        self._sort_keys = wp.empty(2 * n, dtype=wp.int64, device=device)
        self._sort_order = wp.empty(2 * n, dtype=wp.int32, device=device)
        self._sorted_points = wp.empty(n, dtype=wp.vec3, device=device)
        self._sorted_faces = wp.empty(n, dtype=wp.int32, device=device)
        self._sorted_uv = wp.empty(n, dtype=wp.vec2, device=device)
        self._sorted_priority = wp.empty(n, dtype=wp.float32, device=device)
        self._sorted_normals = wp.empty(n, dtype=wp.vec3, device=device) if self.geodesic else None

        # Single-entry spatial hash and per-candidate solver state.
        self._slot_key = wp.empty(table_size, dtype=wp.int64, device=device)
        self._slot_sample = wp.empty(table_size, dtype=wp.int32, device=device)
        self._cell_best = wp.empty(table_size, dtype=wp.float32, device=device)
        self._cell_winner = wp.empty(table_size, dtype=wp.int32, device=device)
        self._cand_slot = wp.empty(n, dtype=wp.int32, device=device)
        self._cand_phase = wp.empty(n, dtype=wp.int32, device=device)
        self._status = wp.empty(n, dtype=wp.int32, device=device)
        self._free = wp.empty(n, dtype=wp.int32, device=device)

        # Compaction.
        self._mask = wp.empty(n, dtype=wp.int32, device=device)
        self._offsets = wp.empty(n, dtype=wp.int32, device=device)
        self.points_buffer = wp.empty(n, dtype=wp.vec3, device=device)
        self.faces_buffer = wp.empty(n, dtype=wp.int32, device=device)
        self.uv_buffer = wp.empty(n, dtype=wp.vec2, device=device)
        self.sample_count = wp.zeros(1, dtype=wp.int32, device=device)

    def update(self, seed: int | None = None) -> None:
        """Re-run the sampling on the current vertex positions.

        Call this after the vertex positions have changed in place while the
        triangles stay the same. It launches only device work, never reads back to the
        host and allocates nothing, so it can be recorded in a CUDA graph. Replaying
        the graph repeats the same ``seed``, which keeps the sample set coherent
        from frame to frame as the mesh moves.

        Invalid input, such as non-finite vertex positions or a ``radius`` too small
        for the mesh extent, cannot be raised from here. It is recorded on the device,
        yields no samples, and is raised as ``ValueError`` the next time
        :attr:`num_samples`, :attr:`points`, :attr:`faces` or :attr:`uv` is read, or by
        :meth:`check_error`.

        Args:
            seed: Seed for candidate generation and priorities. Defaults to the
                ``seed`` given at construction.
        """
        seed = self.seed if seed is None else int(seed)
        n = self.num_candidates
        device = self.device
        sampler = self._sampler

        # Stage 0: rebuild the area distribution from the current vertex positions.
        sampler.update()

        # Stage 1: dense area-weighted candidate pool and its world positions.
        sampler.sample(n, seed=seed, out_faces=self._cand_faces, out_uv=self._cand_uv)
        wp.launch(
            _eval_positions_kernel,
            dim=n,
            inputs=[sampler.mesh.id, self._cand_faces, self._cand_uv],
            outputs=[self._cand_points],
            device=device,
        )
        wp.launch(_poisson_priority_kernel, dim=n, inputs=[seed], outputs=[self._priority], device=device)

        # Stage 2: grid geometry from the vertex bounds, then the cell sort.
        self._bounds_lo.fill_(1.0e30)
        self._bounds_hi.fill_(-1.0e30)
        wp.launch(
            _bounds_kernel,
            dim=sampler.points.shape[0],
            inputs=[sampler.points],
            outputs=[self._bounds_lo, self._bounds_hi],
            device=device,
        )
        wp.launch(
            _poisson_grid_kernel,
            dim=1,
            inputs=[self._bounds_lo, self._bounds_hi, self._inv_mu],
            outputs=[self._grid, sampler._error],
            device=device,
        )
        self._sort_by_cell()

        # Geodesic mode also needs a surface (face) normal per candidate. Evaluate
        # it after the sort, directly from the sorted faces, so it lines up with the
        # sorted candidates without a separate gather.
        if self.geodesic:
            wp.launch(
                _eval_normals_kernel,
                dim=n,
                inputs=[sampler.mesh.id, self._sorted_faces],
                outputs=[self._sorted_normals],
                device=device,
            )

        # Stage 3: parallel conflict resolution.
        self._solve_hash(seed)

        # Stage 4: compact accepted candidates into the output buffers.
        self._compact()

    def _sort_by_cell(self) -> None:
        """Sort the candidates by grid cell so that spatially adjacent candidates are
        processed together. The phase passes are memory-bound on random hash lookups;
        ordering candidates by cell makes neighboring threads read overlapping 5x5x5
        blocks, which turns those lookups into cache hits and roughly halves the solve
        time. This mirrors the paper's step of sorting the point cloud by cell id."""
        n = self.num_candidates
        device = self.device
        wp.launch(
            _poisson_cellid_kernel,
            dim=n,
            inputs=[self._cand_points, self._grid],
            outputs=[self._sort_keys],
            device=device,
        )
        wp.launch(_poisson_iota_kernel, dim=n, outputs=[self._sort_order], device=device)
        radix_sort_pairs(self._sort_keys, self._sort_order, count=n)
        wp.launch(
            _poisson_gather_kernel,
            dim=n,
            inputs=[self._sort_order, self._cand_points, self._cand_faces, self._cand_uv, self._priority],
            outputs=[self._sorted_points, self._sorted_faces, self._sorted_uv, self._sorted_priority],
            device=device,
        )

    def _solve_hash(self, seed: int) -> None:
        """Single-entry spatial hash + 27 phase groups. Uses the Euclidean conflict
        kernel by default, or the geodesic one when ``geodesic`` is set.

        Memory scales with the sampled surface (only non-empty cells are stored),
        and every conflict check reads a constant 5x5x5 block of the hash.
        """
        n = self.num_candidates
        device = self.device
        # The table holds at least twice the candidate count, which bounds the load factor
        # below 1/2 (distinct cells <= candidates), so linear probing stays short.
        table_size = 2 * n + 1

        self._slot_key.fill_(-1)  # -1 = empty
        self._slot_sample.fill_(-1)
        # Each cell's hash slot belongs to exactly one phase group, so a slot is
        # written in only one of the 27 passes. The per-cell election scratch can
        # therefore be initialized once, not reset per phase.
        self._cell_best.fill_(-1.0)
        self._cell_winner.fill_(n)
        self._status.zero_()
        self._free.zero_()

        wp.launch(
            _poisson_hash_insert_kernel,
            dim=n,
            inputs=[self._sorted_points, self._grid, table_size],
            outputs=[self._slot_key],
            device=device,
        )
        wp.launch(
            _poisson_setup_kernel,
            dim=n,
            inputs=[self._sorted_points, self._grid, table_size, self._slot_key],
            outputs=[self._cand_slot, self._cand_phase],
            device=device,
        )

        # Process the 27 phase groups in a seed-dependent random order to avoid
        # directional bias, as recommended in the paper.
        rng = np.random.default_rng(seed)
        phase_order = rng.permutation(27)
        for phase_index in phase_order:
            phase = int(phase_index)
            if self.geodesic:
                wp.launch(
                    _poisson_phase_max_geodesic_kernel,
                    dim=n,
                    inputs=[
                        phase,
                        self._sorted_points,
                        self._sorted_normals,
                        self._grid,
                        table_size,
                        self.radius,
                        self._slot_key,
                        self._slot_sample,
                        self._cand_slot,
                        self._cand_phase,
                        self._sorted_priority,
                        self._status,
                        self._free,
                        self._cell_best,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    _poisson_phase_max_kernel,
                    dim=n,
                    inputs=[
                        phase,
                        self._sorted_points,
                        self._grid,
                        table_size,
                        self.radius,
                        self._slot_key,
                        self._slot_sample,
                        self._cand_slot,
                        self._cand_phase,
                        self._sorted_priority,
                        self._status,
                        self._free,
                        self._cell_best,
                    ],
                    device=device,
                )
            wp.launch(
                _poisson_phase_pick_kernel,
                dim=n,
                inputs=[
                    phase,
                    self._cand_slot,
                    self._cand_phase,
                    self._sorted_priority,
                    self._status,
                    self._free,
                    self._cell_best,
                    self._cell_winner,
                ],
                device=device,
            )
            wp.launch(
                _poisson_phase_accept_kernel,
                dim=n,
                inputs=[
                    phase,
                    self._cand_slot,
                    self._cand_phase,
                    self._status,
                    self._free,
                    self._cell_winner,
                    self._slot_sample,
                ],
                device=device,
            )

    def _compact(self) -> None:
        """Pack the accepted candidates at the front of the output buffers and record their count."""
        n = self.num_candidates
        device = self.device
        error = self._sampler._error
        wp.launch(_poisson_mask_kernel, dim=n, inputs=[self._status, error], outputs=[self._mask], device=device)
        array_scan(self._mask, self._offsets, inclusive=True)
        wp.launch(
            _poisson_count_kernel, dim=1, inputs=[self._offsets, error], outputs=[self.sample_count], device=device
        )
        wp.launch(
            _poisson_compact_kernel,
            dim=n,
            inputs=[self._mask, self._offsets, self._sorted_faces, self._sorted_uv, self._sorted_points],
            outputs=[self.faces_buffer, self.uv_buffer, self.points_buffer],
            device=device,
        )

    def check_error(self) -> None:
        """Raise ``ValueError`` if the latest solve found invalid input.

        :meth:`update` cannot raise because it never reads back to the host, so it
        records invalid input on the device instead. Reading this synchronizes the device.
        """
        _raise_for_error(
            int(self._sampler._error.numpy()[0]),
            supplied_areas=self._sampler._face_areas is not None,
            radius=self.radius,
        )

    def _read_sample_count(self) -> int:
        self.check_error()
        return int(self.sample_count.numpy()[0])

    @property
    def total_area_device(self) -> wp.array:
        """Current total surface area as a device array of shape ``(1,)``, refreshed by :meth:`update`."""
        return self._sampler.total_area_device

    @property
    def num_samples(self) -> int:
        """Number of samples after the latest solve. Reading it synchronizes the device."""
        return self._read_sample_count()

    @property
    def points(self) -> wp.array:
        """Sample positions (:class:`warp.vec3`), sized to :attr:`num_samples`."""
        return self.points_buffer[: self._read_sample_count()]

    @property
    def faces(self) -> wp.array:
        """Face index (:class:`warp.int32`) of each sample, sized to :attr:`num_samples`."""
        return self.faces_buffer[: self._read_sample_count()]

    @property
    def uv(self) -> wp.array:
        """Barycentric coordinates (:class:`warp.vec2`) of each sample, sized to :attr:`num_samples`."""
        return self.uv_buffer[: self._read_sample_count()]

    def pair_correlation(self, *, r_max: float | None = None, num_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Pair-correlation function of this sampler's points (see :func:`pair_correlation`).

        Uses :attr:`total_area`, the area when the sampler was built, for the density
        normalization. After :meth:`update` has changed the mesh area, call the module-level
        :func:`pair_correlation` with the current area instead. For a Poisson-disk set,
        ``g(r)`` is near zero below :attr:`radius`.
        """
        return pair_correlation(self.points, self.total_area, r_max=r_max, num_bins=num_bins, device=self.device)


def poisson_disk_sample(
    points,
    faces,
    radius: float,
    *,
    num_candidates: int | None = None,
    candidate_multiplier: float = 12.0,
    seed: int = 0,
    geodesic: bool = False,
    face_areas=None,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Sample a Poisson-disk (blue-noise) point set over a triangle mesh surface.

    Sampling is driven by a target ``radius``: no two returned samples are closer
    than ``radius`` and the set is maximal (the count is emergent). This is a
    convenience wrapper that builds a :class:`PoissonDiskSampler` and reads back
    its result; construct the class directly to inspect the sampler or reuse the
    candidate pool.

    Args:
        points: Vertex positions, either a :class:`warp.array` of
            :class:`warp.vec3` or an array-like of shape ``(num_vertices, 3)``.
        faces: Triangle vertex indices, either a flat :class:`warp.array` of
            :class:`warp.int32` (length ``3 * num_triangles``) or an array-like
            reshapeable to ``(num_triangles, 3)``.
        radius: Minimum distance between any two samples (Euclidean, or geodesic
            when ``geodesic`` is set).
        num_candidates: Size of the candidate pool. If ``None``, it is set to
            ``candidate_multiplier`` times the theoretical maximal sample count.
        candidate_multiplier: Oversampling factor used when ``num_candidates`` is
            ``None``.
        seed: Seed for candidate generation and priorities.
        geodesic: If set, measure the minimum distance with the approximate
            geodesic (on-surface) metric of ``curvature_corrected_distance`` instead of
            the Euclidean one, which avoids over-separating samples across thin
            features. See :class:`PoissonDiskSampler`.
        face_areas: Optional precomputed per-triangle areas, forwarded to
            :class:`UniformSampler`. See :class:`PoissonDiskSampler`.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(faces, uv, points)`` of :class:`warp.array` giving the face
        index, barycentric coordinates, and world-space position of each sample.
    """
    sampler = PoissonDiskSampler(
        points,
        faces,
        radius,
        geodesic=geodesic,
        face_areas=face_areas,
        num_candidates=num_candidates,
        candidate_multiplier=candidate_multiplier,
        seed=seed,
        device=device,
    )
    return sampler.faces, sampler.uv, sampler.points


##########################################################################
## Spectrum analysis on surfaces: the pair-correlation function (PCF)
##
## The paper measures blue-noise quality with a Fourier power spectrum in a
## spectral mesh basis (mesh-Laplacian eigenfunctions). That basis is expensive
## to build. The differential-domain pair-correlation function is the modern,
## basis-free equivalent (Wei and Wang 2011): it works directly on the surface
## samples via pairwise distances and reveals the same signature -- ``g(r) ~ 0``
## inside the Poisson radius, a peak just past it, and ``g(r) -> 1`` far away.
##########################################################################


@wp.kernel(enable_backward=False)
def _pcf_histogram_kernel(
    grid: wp.uint64,
    points: wp.array[wp.vec3],
    r_max: wp.float32,
    inv_dr: wp.float32,
    num_bins: wp.int32,
    counts: wp.array[wp.int32],
):
    tid = wp.tid()
    pi = points[tid]
    neighbors = wp.hash_grid_query(grid, pi, r_max)
    for j in neighbors:
        if j != tid:
            d = wp.length(points[j] - pi)
            if d < r_max:
                b = wp.int32(d * inv_dr)
                if b < num_bins:
                    wp.atomic_add(counts, b, 1)


def pair_correlation(
    points: wp.array,
    area: float,
    *,
    r_max: float | None = None,
    num_bins: int = 64,
    device: DeviceLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the pair-correlation function of a point set on a surface.

    The pair-correlation function ``g(r)`` is the differential-domain measure of
    blue-noise quality: the density of sample pairs at separation ``r``,
    normalized so that a uniform Poisson (white-noise) process gives ``g(r) = 1``
    everywhere. A Poisson-disk set instead shows ``g(r) ~ 0`` below its minimum
    distance, a peak just beyond it, and mild oscillations that decay to ``1`` --
    the surface analog of the radial power spectrum used in the paper.

    Distances are Euclidean, accumulated over every pair within ``r_max`` using a
    :class:`warp.HashGrid`, then normalized per radial bin by the count expected
    for a uniform process of the same density ``N / area``.

    Args:
        points: Sample positions, a :class:`warp.array` of :class:`warp.vec3`.
        area: Total surface area the samples are drawn from, used to set the
            reference density.
        r_max: Largest separation to measure. Defaults to ``6`` times the mean
            sample spacing ``sqrt(area / N)``.
        num_bins: Number of radial bins in ``[0, r_max]``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(radii, g)`` of NumPy arrays: the bin-center radii and the
        pair-correlation value in each bin.
    """
    device = wp.get_device(device) if device is not None else points.device
    if points.device != device:
        points = points.to(device)
    num_points = int(points.shape[0])
    if num_points < 2:
        raise ValueError("`pair_correlation` needs at least two points.")
    if area <= 0.0:
        raise ValueError(f"`area` must be positive, got {area}.")

    if r_max is None:
        r_max = 6.0 * float(np.sqrt(area / num_points))
    r_max = float(r_max)
    if r_max <= 0.0:
        raise ValueError(f"`r_max` must be positive, got {r_max}.")
    dr = r_max / num_bins

    # Reduce the bounds on the GPU (6 floats back) rather than reading back every point.
    lo, hi = _bounding_box(points, device)
    extent = np.maximum(hi - lo, r_max)
    dims = np.clip(np.ceil(extent / r_max).astype(np.int64), 1, 512)
    grid = wp.HashGrid(int(dims[0]), int(dims[1]), int(dims[2]), device=device)
    grid.build(points, r_max)

    counts = wp.zeros(num_bins, dtype=wp.int32, device=device)
    wp.launch(
        _pcf_histogram_kernel,
        dim=num_points,
        inputs=[grid.id, points, r_max, float(1.0 / dr), num_bins],
        outputs=[counts],
        device=device,
    )

    counts_np = counts.numpy().astype(np.float64)
    radii = (np.arange(num_bins) + 0.5) * dr
    # Expected ordered-pair count for a uniform process: for each of N points,
    # density * area of the annulus [r, r + dr] ~ rho * 2*pi*r*dr neighbors.
    density = num_points / area
    expected = num_points * density * 2.0 * np.pi * radii * dr
    g = np.divide(counts_np, expected, out=np.zeros_like(counts_np), where=expected > 0.0)
    return radii, g
