# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """Device-side state of a :class:`UniformSampler`, passed to :func:`draw`.

    It bundles the mesh identifier with the cumulative area distribution needed
    to pick a triangle with probability proportional to its area. Obtain an
    instance from ``UniformSampler.state`` and pass it as a kernel argument.
    """

    mesh: wp.uint64
    """Identifier of the :class:`warp.Mesh` being sampled."""

    cdf: wp.array(dtype=wp.float32)
    """Normalized cumulative distribution over triangle areas, ending at ``1``."""


@wp.func
def sample_barycentrics(rng: wp.uint32) -> wp.vec2:
    """Draw barycentric coordinates uniformly over the unit triangle.

    Args:
        rng: Random number generator state, updated in place by each draw.

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
def geodesic_distance(p1: wp.vec3, n1: wp.vec3, p2: wp.vec3, n2: wp.vec3) -> wp.float32:
    """Approximate the geodesic (on-surface) distance between two surface points.

    Uses the fast normal-based estimate of Bowers et al. (SIGGRAPH Asia 2010),
    which needs only the two points and their unit surface normals -- no mesh
    connectivity or parametrization. It integrates the differential arc length of
    a curve whose normal turns linearly from ``n1`` to ``n2`` along the connecting
    direction. The estimate is never smaller than the Euclidean distance, equals
    it on a flat region (``n1 == n2``), and is *exact* on a sphere.

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


@wp.func
def draw(state: UniformSamplerState, rng: wp.uint32) -> MeshSample:
    """Draw one point uniformly over the surface of a mesh.

    A triangle is chosen with probability proportional to its area via a binary
    search of the cumulative distribution in ``state``, then a point is drawn
    uniformly within that triangle. Callable from within a :func:`warp.kernel`.

    Args:
        state: Sampler state, typically ``UniformSampler.state`` passed as a
            kernel argument.
        rng: Random number generator state, updated in place by each draw. A
            single ``rng`` can therefore feed repeated ``draw`` calls in a loop.

    Returns:
        The sampled :class:`MeshSample` (a triangle index and barycentrics).
    """
    r = wp.randf(rng)
    sample = MeshSample()
    sample.face = wp.lower_bound(state.cdf, r)
    sample.uv = sample_barycentrics(rng)
    return sample


##########################################################################
## Array-level operations and the host-side sampler
##########################################################################


@wp.kernel(enable_backward=False)
def _triangle_areas_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out_areas: wp.array(dtype=wp.float32),
):
    tri = wp.tid()
    v0 = points[indices[tri * 3 + 0]]
    v1 = points[indices[tri * 3 + 1]]
    v2 = points[indices[tri * 3 + 2]]
    out_areas[tri] = 0.5 * wp.length(wp.cross(v1 - v0, v2 - v0))


@wp.kernel(enable_backward=False)
def _areas_invalid_kernel(areas: wp.array(dtype=wp.float32), out_flag: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    a = areas[tid]
    if a < 0.0 or not wp.isfinite(a):
        out_flag[0] = 1


@wp.kernel(enable_backward=False)
def _normalize_cdf_kernel(
    cumulative: wp.array(dtype=wp.float32),
    out_cdf: wp.array(dtype=wp.float32),
):
    tri = wp.tid()
    total = cumulative[cumulative.shape[0] - 1]
    # A degenerate (zero-area) mesh has no meaningful area weighting; fall back to
    # a uniform-over-triangles CDF so sampling still terminates instead of NaN.
    if total > 0.0:
        out_cdf[tri] = cumulative[tri] / total
    else:
        out_cdf[tri] = float(tri + 1) / float(cumulative.shape[0])


@wp.kernel(enable_backward=False)
def _draw_kernel(
    state: UniformSamplerState,
    seed: wp.int32,
    out_faces: wp.array(dtype=wp.int32),
    out_uv: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    sample = draw(state, rng)
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


class UniformSampler:
    """Draw points uniformly over the surface of a triangle mesh.

    The sampler weights each triangle by its area so that samples are spread
    evenly across the surface even when the tessellation is non-uniform. It
    precomputes a cumulative area distribution (CDF) once at construction; each
    subsequent draw is a binary search plus a within-triangle sample.

    Two ways to draw are provided:

    * :meth:`sample` (host side) launches a kernel and returns arrays of faces
      and barycentric coordinates -- convenient for one-shot sampling.
    * :attr:`draw` (device side) samples a single point from within your own
      kernel. It is the module-level :func:`draw` exposed as a member ``@wp.func``,
      so ``sampler.draw(sampler.state, rng)`` resolves inside a kernel when
      ``sampler`` is captured from the enclosing scope.

    Args:
        points: Vertex positions, either a :class:`warp.array` of
            :class:`warp.vec3` or an array-like of shape ``(num_vertices, 3)``.
        faces: Triangle vertex indices, either a flat :class:`warp.array` of
            :class:`warp.int32` (length ``3 * num_triangles``) or an array-like
            reshapeable to ``(num_triangles, 3)``.
        face_areas: Optional per-triangle areas of length ``num_triangles``,
            either a :class:`warp.array` of :class:`warp.float32` or an array-like.
            Supply them to reuse areas the caller already has; when ``None`` they
            are computed from ``points`` and ``faces``.
        device: Device on which to build the sampler. Defaults to the device of
            ``points`` when it is a :class:`warp.array`, otherwise the current
            device.

    Attributes:
        total_area: Total surface area of the mesh.
    """

    # Expose the device sampling function as a member ``@wp.func`` so that it can
    # be called as ``sampler.draw(sampler.state, rng)`` from inside a kernel that
    # captures the sampler, in addition to the module-level ``warp.geometry.draw``.
    # ``draw`` is a ``warp.Function`` (not a plain function), so attribute access
    # returns it unbound rather than turning it into a Python method.
    draw = draw

    def __init__(self, points, faces, *, face_areas=None, device: DeviceLike | None = None):
        if device is None:
            device = points.device if isinstance(points, wp.array) else wp.get_device()
        self.device = wp.get_device(device)

        self.points = _as_vec3_array(points, self.device)
        self.indices = _as_index_array(faces, self.device)
        self.num_triangles = self.indices.shape[0] // 3
        if self.num_triangles == 0:
            raise ValueError("`faces` must describe at least one triangle.")

        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

        # Build the normalized cumulative area distribution once. Reuse the
        # caller's per-triangle areas when given, otherwise compute them.
        if face_areas is None:
            areas = wp.empty(self.num_triangles, dtype=wp.float32, device=self.device)
            wp.launch(
                _triangle_areas_kernel,
                dim=self.num_triangles,
                inputs=[self.points, self.indices],
                outputs=[areas],
                device=self.device,
            )
        else:
            areas = _as_float_array(face_areas, self.device)
            if areas.shape[0] != self.num_triangles:
                raise ValueError(
                    f"`face_areas` must have length num_triangles ({self.num_triangles}), got {areas.shape[0]}."
                )
            # Negative or non-finite areas would produce a non-monotonic CDF and
            # silently corrupt the area weighting, so reject them up front.
            invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
            wp.launch(
                _areas_invalid_kernel, dim=self.num_triangles, inputs=[areas], outputs=[invalid], device=self.device
            )
            if int(invalid.numpy()[0]) != 0:
                raise ValueError("`face_areas` must be non-negative and finite.")
        cumulative = wp.empty(self.num_triangles, dtype=wp.float32, device=self.device)
        array_scan(areas, cumulative, inclusive=True)

        # The inclusive scan's final entry is the total surface area, handy for
        # sizing candidate pools (e.g. Poisson-disk sampling). Reading that one
        # element avoids a second full reduction over the triangles.
        self.total_area = float(cumulative[self.num_triangles - 1 :].numpy()[0])

        self.cdf = wp.empty(self.num_triangles, dtype=wp.float32, device=self.device)
        wp.launch(
            _normalize_cdf_kernel,
            dim=self.num_triangles,
            inputs=[cumulative],
            outputs=[self.cdf],
            device=self.device,
        )

        self.state = UniformSamplerState()
        self.state.mesh = self.mesh.id
        self.state.cdf = self.cdf

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
    faces: wp.array(dtype=wp.int32),
    uv: wp.array(dtype=wp.vec2),
    out_positions: wp.array(dtype=wp.vec3),
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
def _poisson_priority_kernel(seed: wp.int32, out_priority: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    out_priority[tid] = wp.randf(rng)


@wp.kernel(enable_backward=False)
def _poisson_mask_kernel(status: wp.array(dtype=wp.int32), out_mask: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if status[tid] == _POISSON_ACCEPTED:
        out_mask[tid] = 1
    else:
        out_mask[tid] = 0


@wp.kernel(enable_backward=False)
def _poisson_compact_kernel(
    status: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    in_faces: wp.array(dtype=wp.int32),
    in_uv: wp.array(dtype=wp.vec2),
    in_points: wp.array(dtype=wp.vec3),
    out_faces: wp.array(dtype=wp.int32),
    out_uv: wp.array(dtype=wp.vec2),
    out_points: wp.array(dtype=wp.vec3),
):
    # ``offsets`` is an inclusive prefix sum of the accepted mask, so an accepted
    # candidate writes to slot ``offsets[tid] - 1``.
    tid = wp.tid()
    if status[tid] == _POISSON_ACCEPTED:
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


@wp.func
def _cell_coord(p: wp.vec3, lo: wp.vec3, inv_mu: wp.float32, gx: wp.int32, gy: wp.int32, gz: wp.int32) -> wp.vec3i:
    cx = wp.clamp(wp.int32((p[0] - lo[0]) * inv_mu), 0, gx - 1)
    cy = wp.clamp(wp.int32((p[1] - lo[1]) * inv_mu), 0, gy - 1)
    cz = wp.clamp(wp.int32((p[2] - lo[2]) * inv_mu), 0, gz - 1)
    return wp.vec3i(cx, cy, cz)


@wp.func
def _cell_id(cx: wp.int32, cy: wp.int32, cz: wp.int32, gx: wp.int32, gy: wp.int32) -> wp.int64:
    # 64-bit so the id space can exceed 2^31 for fine grids on large meshes.
    return (wp.int64(cz) * wp.int64(gy) + wp.int64(cy)) * wp.int64(gx) + wp.int64(cx)


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
def _hash_find(key: wp.int64, table_size: wp.int32, slot_key: wp.array(dtype=wp.int64)) -> wp.int32:
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
    points: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_mu: wp.float32,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    slot_key: wp.array(dtype=wp.int64),
):
    # Insert each candidate's cell id once (deduplicated by CAS on an empty slot).
    tid = wp.tid()
    c = _cell_coord(points[tid], lo, inv_mu, gx, gy, gz)
    key = _cell_id(c[0], c[1], c[2], gx, gy)
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
    points: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_mu: wp.float32,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    slot_key: wp.array(dtype=wp.int64),
    out_slot: wp.array(dtype=wp.int32),
    out_phase: wp.array(dtype=wp.int32),
):
    # Cache each candidate's hash slot and phase group so the phase passes below
    # avoid recomputing them.
    tid = wp.tid()
    c = _cell_coord(points[tid], lo, inv_mu, gx, gy, gz)
    out_slot[tid] = _hash_find(_cell_id(c[0], c[1], c[2], gx, gy), table_size, slot_key)
    out_phase[tid] = _cell_phase(c)


@wp.func
def _cell_free(
    pos: wp.vec3,
    c: wp.vec3i,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    slot_key: wp.array(dtype=wp.int64),
    slot_sample: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    r_sq: wp.float32,
) -> bool:
    # True if no accepted sample in the surrounding 5x5x5 block is within radius.
    for dz in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
        nz = c[2] + dz
        if nz >= 0 and nz < gz:
            for dy in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                ny = c[1] + dy
                if ny >= 0 and ny < gy:
                    for dx in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                        nx = c[0] + dx
                        if nx >= 0 and nx < gx:
                            slot = _hash_find(_cell_id(nx, ny, nz, gx, gy), table_size, slot_key)
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
    points: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_mu: wp.float32,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    radius: wp.float32,
    slot_key: wp.array(dtype=wp.int64),
    slot_sample: wp.array(dtype=wp.int32),
    cand_slot: wp.array(dtype=wp.int32),
    cand_phase: wp.array(dtype=wp.int32),
    priority: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
    free: wp.array(dtype=wp.int32),
    cell_best: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase:
        return
    p = points[tid]
    c = _cell_coord(p, lo, inv_mu, gx, gy, gz)
    if _cell_free(p, c, gx, gy, gz, table_size, slot_key, slot_sample, points, radius * radius):
        free[tid] = 1
        wp.atomic_max(cell_best, cand_slot[tid], priority[tid])
    else:
        free[tid] = 0


@wp.kernel(enable_backward=False)
def _poisson_phase_pick_kernel(
    phase: wp.int32,
    cand_slot: wp.array(dtype=wp.int32),
    cand_phase: wp.array(dtype=wp.int32),
    priority: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
    free: wp.array(dtype=wp.int32),
    cell_best: wp.array(dtype=wp.float32),
    cell_winner: wp.array(dtype=wp.int32),
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
    cand_slot: wp.array(dtype=wp.int32),
    cand_phase: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    free: wp.array(dtype=wp.int32),
    cell_winner: wp.array(dtype=wp.int32),
    slot_sample: wp.array(dtype=wp.int32),
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
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    slot_key: wp.array(dtype=wp.int64),
    slot_sample: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    r_sq: wp.float32,
) -> bool:
    # As _cell_free, but a Euclidean-close accepted sample only conflicts when its
    # approximate geodesic distance is also below the radius. dg >= de, so the
    # cheap Euclidean test still prunes everything beyond the radius first.
    for dz in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
        nz = c[2] + dz
        if nz >= 0 and nz < gz:
            for dy in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                ny = c[1] + dy
                if ny >= 0 and ny < gy:
                    for dx in range(-_POISSON_SEARCH, _POISSON_SEARCH + 1):
                        nx = c[0] + dx
                        if nx >= 0 and nx < gx:
                            slot = _hash_find(_cell_id(nx, ny, nz, gx, gy), table_size, slot_key)
                            if slot >= 0:
                                s = slot_sample[slot]
                                if s >= 0:
                                    d = points[s] - pos
                                    if wp.dot(d, d) < r_sq:
                                        if geodesic_distance(pos, normal, points[s], normals[s]) < radius:
                                            return False
    return True


@wp.kernel(enable_backward=False)
def _poisson_phase_max_geodesic_kernel(
    phase: wp.int32,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_mu: wp.float32,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    table_size: wp.int32,
    radius: wp.float32,
    slot_key: wp.array(dtype=wp.int64),
    slot_sample: wp.array(dtype=wp.int32),
    cand_slot: wp.array(dtype=wp.int32),
    cand_phase: wp.array(dtype=wp.int32),
    priority: wp.array(dtype=wp.float32),
    status: wp.array(dtype=wp.int32),
    free: wp.array(dtype=wp.int32),
    cell_best: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if status[tid] != _POISSON_ACTIVE or cand_phase[tid] != phase:
        return
    p = points[tid]
    c = _cell_coord(p, lo, inv_mu, gx, gy, gz)
    if _cell_free_geodesic(
        p, normals[tid], c, gx, gy, gz, table_size, slot_key, slot_sample, points, normals, radius, radius * radius
    ):
        free[tid] = 1
        wp.atomic_max(cell_best, cand_slot[tid], priority[tid])
    else:
        free[tid] = 0


@wp.kernel(enable_backward=False)
def _eval_normals_kernel(
    mesh: wp.uint64,
    faces: wp.array(dtype=wp.int32),
    out_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_normals[tid] = wp.mesh_eval_face_normal(mesh, faces[tid])


@wp.kernel(enable_backward=False)
def _bounds_kernel(
    points: wp.array(dtype=wp.vec3),
    out_lo: wp.array(dtype=wp.float32),
    out_hi: wp.array(dtype=wp.float32),
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
def _poisson_cellid_kernel(
    points: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_mu: wp.float32,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
    out_cellid: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    c = _cell_coord(points[tid], lo, inv_mu, gx, gy, gz)
    out_cellid[tid] = _cell_id(c[0], c[1], c[2], gx, gy)


@wp.kernel(enable_backward=False)
def _poisson_iota_kernel(out: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    out[tid] = tid


@wp.kernel(enable_backward=False)
def _poisson_gather_kernel(
    order: wp.array(dtype=wp.int32),
    in_points: wp.array(dtype=wp.vec3),
    in_faces: wp.array(dtype=wp.int32),
    in_uv: wp.array(dtype=wp.vec2),
    in_priority: wp.array(dtype=wp.float32),
    out_points: wp.array(dtype=wp.vec3),
    out_faces: wp.array(dtype=wp.int32),
    out_uv: wp.array(dtype=wp.vec2),
    out_priority: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    j = order[tid]
    out_points[tid] = in_points[j]
    out_faces[tid] = in_faces[j]
    out_uv[tid] = in_uv[j]
    out_priority[tid] = in_priority[j]


class PoissonDiskSampler:
    """Draw a Poisson-disk (blue-noise) point set over a triangle mesh surface.

    No two returned samples are closer than ``radius`` in Euclidean distance, and
    the set is *maximal*: no further candidate could be added without violating
    that spacing. The distribution therefore has the characteristic blue-noise
    spectrum -- suppressed low frequencies and no structured aliasing -- which
    makes it well suited to stippling, scattering, remeshing seeds, and
    Monte-Carlo integration.

    The sampler follows Bowers et al., *"Parallel Poisson Disk Sampling with
    Spectrum Analysis on Surfaces"* (SIGGRAPH Asia 2010): it draws a dense pool of
    area-weighted candidates with :class:`UniformSampler`, then resolves conflicts
    entirely in parallel as a priority-based maximal independent set over the
    graph connecting candidates closer than ``radius``.

    By default the minimum distance is Euclidean, the standard approximation to
    geodesic distance, accurate when ``radius`` is small relative to the surface's
    curvature. Set ``geodesic`` to instead measure the approximate on-surface
    distance (:func:`geodesic_distance`): this stops samples on opposite sides of
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
            (:func:`geodesic_distance`) for the minimum distance instead of the
            Euclidean one. Keeps a single sample per grid cell: the paper's
            multiple-samples-per-cell extension is intentionally omitted -- it was
            implemented and measured to add nothing under this approximation (see
            ``design/parallel-poisson-disk-sampling.md``).
        face_areas: Optional precomputed per-triangle areas forwarded to the
            internal :class:`UniformSampler`; computed from the mesh when ``None``.
        device: Device on which to run. Defaults to the device of ``points``.

    Attributes:
        points: :class:`warp.array` of :class:`warp.vec3` sample positions.
        faces: :class:`warp.array` of :class:`warp.int32` face indices.
        uv: :class:`warp.array` of :class:`warp.vec2` barycentric coordinates.
        num_samples: Number of samples in the result.
        radius: The minimum-distance radius used.
        total_area: Total surface area of the mesh.
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
        if radius <= 0.0:
            raise ValueError(f"`radius` must be positive, got {radius}.")
        if num_candidates is not None and num_candidates < 1:
            raise ValueError(f"`num_candidates` must be at least 1, got {num_candidates}.")

        self._sampler = UniformSampler(points, faces, face_areas=face_areas, device=device)
        self.device = self._sampler.device
        self.radius = float(radius)
        self.geodesic = bool(geodesic)
        self.total_area = self._sampler.total_area

        # Theoretical maximal count assumes hexagonal packing of disks of radius
        # ``radius / 2``, i.e. one sample per ``sqrt(3)/2 * radius^2`` of area.
        n_est = self.total_area / (0.8660254037844386 * self.radius * self.radius)
        if num_candidates is None:
            num_candidates = int(max(1.0, candidate_multiplier * n_est))
        self.num_candidates = int(num_candidates)

        # The whole pipeline stays on the GPU: candidate generation, the bounding
        # box (reduced on device), the hash and phase passes, and compaction. Only
        # a few scalars ever cross to the host (total area, the bounds, the final
        # sample count).

        # Stage 1: dense area-weighted candidate pool and its world positions.
        cand_faces, cand_uv = self._sampler.sample(self.num_candidates, seed=seed)
        cand_points = wp.empty(self.num_candidates, dtype=wp.vec3, device=self.device)
        wp.launch(
            _eval_positions_kernel,
            dim=self.num_candidates,
            inputs=[self._sampler.mesh.id, cand_faces, cand_uv],
            outputs=[cand_points],
            device=self.device,
        )

        # Stage 2: parallel conflict resolution.
        priority = wp.empty(self.num_candidates, dtype=wp.float32, device=self.device)
        wp.launch(
            _poisson_priority_kernel,
            dim=self.num_candidates,
            inputs=[seed],
            outputs=[priority],
            device=self.device,
        )
        lo, hi = _bounding_box(self._sampler.points, self.device)

        # Sort the candidates by grid cell so that spatially adjacent candidates
        # are processed together. The phase passes are memory-bound on random hash
        # lookups; ordering candidates by cell makes neighboring threads read
        # overlapping 5x5x5 blocks, which turns those lookups into cache hits and
        # roughly halves the solve time. This mirrors the paper's step of sorting
        # the point cloud by cell id.
        cand_points, cand_faces, cand_uv, priority = self._sort_by_cell(
            cand_points, cand_faces, cand_uv, priority, lo, hi
        )

        # Geodesic mode also needs a surface (face) normal per candidate. Evaluate
        # it here, after the sort, directly from the sorted faces -- so it lines up
        # with the sorted candidates without a separate gather.
        cand_normals = None
        if self.geodesic:
            cand_normals = wp.empty(self.num_candidates, dtype=wp.vec3, device=self.device)
            wp.launch(
                _eval_normals_kernel,
                dim=self.num_candidates,
                inputs=[self._sampler.mesh.id, cand_faces],
                outputs=[cand_normals],
                device=self.device,
            )

        status = self._solve_hash(cand_points, priority, lo, hi, seed, cand_normals)

        # Stage 3: compact accepted candidates into tight output arrays.
        self.points, self.faces, self.uv = self._compact(status, cand_points, cand_faces, cand_uv)
        self.num_samples = int(self.points.shape[0])

    def _grid_params(self, lo: np.ndarray, hi: np.ndarray) -> tuple[float, int, int, int, wp.vec3]:
        """Grid geometry shared by the cell sort and the hash solve, so the two
        always agree on cell size and dimensions. The cell edge is
        ``radius / sqrt(3)`` (diagonal ``radius``), giving at most one sample per
        cell."""
        mu = self.radius / 1.7320508075688772  # radius / sqrt(3)
        inv_mu = float(1.0 / mu)
        gx, gy, gz = (int(v) for v in np.maximum(np.ceil((hi - lo) / mu).astype(np.int64) + 1, 1))
        lo_vec = wp.vec3(float(lo[0]), float(lo[1]), float(lo[2]))
        return inv_mu, gx, gy, gz, lo_vec

    def _sort_by_cell(
        self,
        cand_points: wp.array,
        cand_faces: wp.array,
        cand_uv: wp.array,
        priority: wp.array,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        n = self.num_candidates
        inv_mu, gx, gy, gz, lo_vec = self._grid_params(lo, hi)

        # radix_sort_pairs needs 2*n storage for both keys and values.
        cellid = wp.empty(2 * n, dtype=wp.int64, device=self.device)
        order = wp.empty(2 * n, dtype=wp.int32, device=self.device)
        wp.launch(
            _poisson_cellid_kernel, dim=n, inputs=[cand_points, lo_vec, inv_mu, gx, gy, gz, cellid], device=self.device
        )
        wp.launch(_poisson_iota_kernel, dim=n, outputs=[order], device=self.device)
        radix_sort_pairs(cellid, order, count=n)

        out_points = wp.empty(n, dtype=wp.vec3, device=self.device)
        out_faces = wp.empty(n, dtype=wp.int32, device=self.device)
        out_uv = wp.empty(n, dtype=wp.vec2, device=self.device)
        out_priority = wp.empty(n, dtype=wp.float32, device=self.device)
        wp.launch(
            _poisson_gather_kernel,
            dim=n,
            inputs=[order, cand_points, cand_faces, cand_uv, priority],
            outputs=[out_points, out_faces, out_uv, out_priority],
            device=self.device,
        )
        return out_points, out_faces, out_uv, out_priority

    def _solve_hash(
        self,
        cand_points: wp.array,
        priority: wp.array,
        lo: np.ndarray,
        hi: np.ndarray,
        seed: int,
        cand_normals: wp.array | None = None,
    ) -> wp.array:
        """Single-entry spatial hash + 27 phase groups. Uses the Euclidean
        conflict kernel by default, or the geodesic one when ``cand_normals`` is
        given.

        Memory scales with the sampled surface (only non-empty cells are stored),
        and every conflict check reads a constant 5x5x5 block of the hash.
        """
        inv_mu, gx, gy, gz, lo_vec = self._grid_params(lo, hi)

        # Table sized to at least twice the candidate count bounds the load factor
        # below 1/2 (distinct cells <= candidates), so linear probing stays short.
        table_size = 2 * self.num_candidates + 1
        slot_key = wp.full(table_size, -1, dtype=wp.int64, device=self.device)  # -1 = empty

        grid_args = [lo_vec, inv_mu, gx, gy, gz, table_size]
        wp.launch(
            _poisson_hash_insert_kernel,
            dim=self.num_candidates,
            inputs=[cand_points, *grid_args, slot_key],
            device=self.device,
        )

        cand_slot = wp.empty(self.num_candidates, dtype=wp.int32, device=self.device)
        cand_phase = wp.empty(self.num_candidates, dtype=wp.int32, device=self.device)
        wp.launch(
            _poisson_setup_kernel,
            dim=self.num_candidates,
            inputs=[cand_points, *grid_args, slot_key],
            outputs=[cand_slot, cand_phase],
            device=self.device,
        )

        status = wp.zeros(self.num_candidates, dtype=wp.int32, device=self.device)
        free = wp.zeros(self.num_candidates, dtype=wp.int32, device=self.device)
        slot_sample = wp.full(table_size, -1, dtype=wp.int32, device=self.device)
        # Each cell's hash slot belongs to exactly one phase group, so a slot is
        # written in only one of the 27 passes. The per-cell election scratch can
        # therefore be initialized once, not reset per phase.
        cell_best = wp.full(table_size, -1.0, dtype=wp.float32, device=self.device)
        cell_winner = wp.full(table_size, self.num_candidates, dtype=wp.int32, device=self.device)

        # Process the 27 phase groups in a seed-dependent random order to avoid
        # directional bias, as recommended in the paper.
        rng = np.random.default_rng(seed)
        phase_order = rng.permutation(27)
        for phase in phase_order:
            if cand_normals is None:
                wp.launch(
                    _poisson_phase_max_kernel,
                    dim=self.num_candidates,
                    inputs=[
                        int(phase),
                        cand_points,
                        lo_vec,
                        inv_mu,
                        gx,
                        gy,
                        gz,
                        table_size,
                        self.radius,
                        slot_key,
                        slot_sample,
                        cand_slot,
                        cand_phase,
                        priority,
                        status,
                        free,
                        cell_best,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    _poisson_phase_max_geodesic_kernel,
                    dim=self.num_candidates,
                    inputs=[
                        int(phase),
                        cand_points,
                        cand_normals,
                        lo_vec,
                        inv_mu,
                        gx,
                        gy,
                        gz,
                        table_size,
                        self.radius,
                        slot_key,
                        slot_sample,
                        cand_slot,
                        cand_phase,
                        priority,
                        status,
                        free,
                        cell_best,
                    ],
                    device=self.device,
                )
            wp.launch(
                _poisson_phase_pick_kernel,
                dim=self.num_candidates,
                inputs=[int(phase), cand_slot, cand_phase, priority, status, free, cell_best, cell_winner],
                device=self.device,
            )
            wp.launch(
                _poisson_phase_accept_kernel,
                dim=self.num_candidates,
                inputs=[int(phase), cand_slot, cand_phase, status, free, cell_winner, slot_sample],
                device=self.device,
            )
        return status

    def _compact(
        self, status: wp.array, cand_points: wp.array, cand_faces: wp.array, cand_uv: wp.array
    ) -> tuple[wp.array, wp.array, wp.array]:
        mask = wp.empty(self.num_candidates, dtype=wp.int32, device=self.device)
        wp.launch(_poisson_mask_kernel, dim=self.num_candidates, inputs=[status], outputs=[mask], device=self.device)

        offsets = wp.empty(self.num_candidates, dtype=wp.int32, device=self.device)
        array_scan(mask, offsets, inclusive=True)
        # Read back only the final prefix-sum entry (the accepted count), not the
        # whole offsets array.
        num = int(offsets[self.num_candidates - 1 :].numpy()[0])

        out_points = wp.empty(num, dtype=wp.vec3, device=self.device)
        out_faces = wp.empty(num, dtype=wp.int32, device=self.device)
        out_uv = wp.empty(num, dtype=wp.vec2, device=self.device)
        wp.launch(
            _poisson_compact_kernel,
            dim=self.num_candidates,
            inputs=[status, offsets, cand_faces, cand_uv, cand_points],
            outputs=[out_faces, out_uv, out_points],
            device=self.device,
        )
        return out_points, out_faces, out_uv

    def pair_correlation(self, *, r_max: float | None = None, num_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Pair-correlation function of this sampler's points (see :func:`pair_correlation`).

        Uses the sampler's own :attr:`total_area` for the density normalization.
        For a Poisson-disk set, ``g(r)`` is near zero below :attr:`radius`.
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

    No two returned samples are closer than ``radius``, and the set is maximal.
    This is a convenience wrapper that builds a :class:`PoissonDiskSampler` and
    reads back its result; construct the class directly to inspect the sampler or
    reuse the candidate pool.

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
            geodesic (on-surface) metric of :func:`geodesic_distance` instead of
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
    points: wp.array(dtype=wp.vec3),
    r_max: wp.float32,
    inv_dr: wp.float32,
    num_bins: wp.int32,
    counts: wp.array(dtype=wp.int32),
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


##########################################################################
## Rigid registration: iterative closest point (ICP)
##
## Building blocks for ICP -- a closest-point query yielding a target point and
## its normal, and the linearized point-to-plane Gauss-Newton contribution of one
## correspondence -- plus the host-side register_rigid driver.
##########################################################################


@wp.struct
class ClosestPoint:
    """Result of a closest-point query on a target surface.

    Recover the match with ``if cp.valid != 0``.
    """

    point: wp.vec3
    """Closest point on the target."""
    normal: wp.vec3
    """Unit surface normal at :attr:`point`."""
    distance: wp.float32
    """Distance from the query point to :attr:`point`."""
    valid: wp.int32
    """``1`` if a match was found within the distance bound, else ``0``."""


@wp.struct
class GaussNewtonTerm:
    """One correspondence's contribution to the linearized ICP normal equations,
    ``sum w J J^T x = sum w b J`` for the 6-DOF increment ``x = [rotation; translation]``."""

    jacobian: wp.spatial_vector
    """The ``1x6`` Jacobian row ``J``."""
    b: wp.float32
    """The scalar right-hand side ``b``."""


@wp.func
def closest_on_mesh(mesh: wp.uint64, p: wp.vec3, max_dist: wp.float32) -> ClosestPoint:
    """Closest point on a triangle mesh to ``p`` within ``max_dist``.

    Returns the closest surface point and its face normal. The target BVH is
    fixed, so this can be called every ICP iteration on transformed source points
    without any rebuild. Callable from your own :func:`warp.kernel` definitions.

    Args:
        mesh: Identifier of the target :class:`warp.Mesh`.
        p: Query point (a transformed source point).
        max_dist: Reject matches farther than this.

    Returns:
        A :class:`ClosestPoint`; check ``valid``.
    """
    out = ClosestPoint()
    out.valid = wp.int32(0)
    query = wp.mesh_query_point(mesh, p, max_dist)
    if query.result:
        out.point = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out.normal = wp.mesh_eval_face_normal(mesh, query.face)
        out.distance = wp.length(out.point - p)
        out.valid = wp.int32(1)
    return out


@wp.func
def point_plane_term(p: wp.vec3, q: wp.vec3, n: wp.vec3) -> GaussNewtonTerm:
    """Linearized point-to-plane Gauss-Newton term for one correspondence.

    For a source point ``p`` (already in the current frame), its closest target
    point ``q`` and unit normal ``n``, the linearized residual is
    ``r = (p - q).n + a.(p x n) + t.n`` in the 6-DOF increment ``x = [a; t]``.
    This returns the Jacobian row ``J = [p x n, n]`` and ``b = -(p - q).n``.

    Args:
        p: Source point in the current frame.
        q: Closest target point.
        n: Unit target normal at ``q``.

    Returns:
        A :class:`GaussNewtonTerm` with the Jacobian row and right-hand side.
    """
    jr = wp.cross(p, n)
    term = GaussNewtonTerm()
    term.jacobian = wp.spatial_vector(jr[0], jr[1], jr[2], n[0], n[1], n[2])
    term.b = -wp.dot(p - q, n)
    return term


@wp.func
def symmetric_plane_term(p: wp.vec3, q: wp.vec3, n_p: wp.vec3, n_q: wp.vec3) -> GaussNewtonTerm:
    """Linearized symmetric Gauss-Newton term for one correspondence.

    Implements the symmetric objective of Rusinkiewicz, "A Symmetric Objective
    Function for ICP" (2019), which uses *both* surfaces' normals and splits the
    rotation between them. With ``n = n_p + n_q``, the linearized residual is
    ``r = (p - q).n + a.(0.5 (p + q) x n) + t.n``, so the Jacobian row is
    ``J = [0.5 (p + q) x n, n]`` and ``b = -(p - q).n``. It shares the point-to-plane
    6x6 structure but converges over a wider basin. Callable from your own kernels.

    Args:
        p: Source point in the current frame.
        q: Closest target point.
        n_p: Unit source normal at ``p`` (in the current frame).
        n_q: Unit target normal at ``q``.

    Returns:
        A :class:`GaussNewtonTerm` with the Jacobian row and right-hand side.
    """
    n = n_p + n_q
    jr = 0.5 * wp.cross(p + q, n)
    term = GaussNewtonTerm()
    term.jacobian = wp.spatial_vector(jr[0], jr[1], jr[2], n[0], n[1], n[2])
    term.b = -wp.dot(p - q, n)
    return term


@wp.func
def _correspondence_term(p: wp.vec3, q: wp.vec3, n_q: wp.vec3, n_p: wp.vec3, symmetric: wp.int32) -> GaussNewtonTerm:
    # Pick the point-to-plane or symmetric term. In both, the signed residual is
    # recovered as ``-term.b``.
    if symmetric != 0:
        return symmetric_plane_term(p, q, n_p, n_q)
    return point_plane_term(p, q, n_q)


@wp.func
def _accumulate_normal_equations_at(
    term: GaussNewtonTerm,
    weight: wp.float32,
    a_base: wp.int32,
    g_base: wp.int32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
):
    # Scatter ``weight * J J^T`` (upper triangle, 21 entries) and ``weight * b * J``
    # (6 entries) into the normal-equation accumulators, starting at the given
    # offsets (``a_base``, ``g_base``) so several batched problems can share one
    # pair of arrays.
    j = term.jacobian
    wb = weight * term.b
    for i in range(6):
        wp.atomic_add(g, g_base + i, wb * j[i])
        base = i * 6 - (i * (i - 1)) / 2
        for k in range(i, 6):
            wp.atomic_add(a_upper, a_base + base + (k - i), weight * j[i] * j[k])


@wp.func
def _accumulate_normal_equations(
    term: GaussNewtonTerm,
    weight: wp.float32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
):
    # Scatter into a single (unbatched) 21+6 accumulator.
    _accumulate_normal_equations_at(term, weight, 0, 0, a_upper, g)


@wp.func
def _robust_weight(r: wp.float32, inv_scale_sq: wp.float32) -> wp.float32:
    # Welsch (Leclerc) robust weight exp(-(r/s)^2), with ``inv_scale_sq = 1/s^2``.
    # A non-positive ``inv_scale_sq`` disables robust weighting (weight 1), which
    # recovers the plain least-squares point-to-plane step.
    if inv_scale_sq <= 0.0:
        return 1.0
    return wp.exp(-r * r * inv_scale_sq)


@wp.func
def _plane_normal(p: wp.vec3, q: wp.vec3, geometric: wp.vec3, normal_mode: wp.int32) -> wp.vec3:
    # ``normal_mode == 0`` uses the geometric (surface/PCA) normal; ``1`` uses the
    # closest-point direction ``normalize(p - q)`` instead. The latter is exactly
    # the surface normal for a point interior to a triangle, but degrades near
    # convergence (where ``p - q -> 0``), so it falls back to the geometric normal
    # when the offset is tiny.
    if normal_mode == 0:
        return geometric
    d = p - q
    dl = wp.length(d)
    if dl > 1.0e-7:
        return d / dl
    return geometric


@wp.func
def _source_index(tid: wp.int32, num_source: wp.int32, stochastic: wp.int32, seed: wp.int32) -> wp.int32:
    # With stochastic subsampling, thread ``tid`` draws a random source index
    # (sampling with replacement, reseeded each iteration); otherwise it maps to
    # its own source point.
    if stochastic == 0:
        return tid
    state = wp.rand_init(seed, tid)
    return wp.randi(state, 0, num_source)


@wp.kernel(enable_backward=False)
def _icp_accumulate_mesh_kernel(
    source: wp.array(dtype=wp.vec3),
    rots: wp.array(dtype=wp.mat33),
    transs: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    max_dist: wp.float32,
    num_source: wp.int32,
    stochastic: wp.int32,
    seed: wp.int32,
    inv_scale: wp.array(dtype=wp.float32),
    source_normals: wp.array(dtype=wp.vec3),
    symmetric: wp.int32,
    normal_mode: wp.int32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
):
    # Single problem: the transform and robust scale live in length-1 GPU arrays
    # so the driver never reads them back between iterations.
    tid = wp.tid()
    rot = rots[0]
    trans = transs[0]
    idx = _source_index(tid, num_source, stochastic, seed)
    p = rot * source[idx] + trans
    cp = closest_on_mesh(mesh, p, max_dist)
    if cp.valid == 0:
        return
    n_q = _plane_normal(p, cp.point, cp.normal, normal_mode)
    n_p = wp.vec3(0.0, 0.0, 0.0)
    if symmetric != 0:
        n_p = rot * source_normals[idx]
    term = _correspondence_term(p, cp.point, n_q, n_p, symmetric)
    r = -term.b
    w = _robust_weight(r, inv_scale[0])
    _accumulate_normal_equations(term, w, a_upper, g)
    wp.atomic_add(stats, 0, r * r)
    wp.atomic_add(stats, 1, 1.0)


@wp.func
def closest_on_points(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    p: wp.vec3,
    max_dist: wp.float32,
) -> ClosestPoint:
    """Nearest target point (and its normal) to ``p`` within ``max_dist``.

    Searches a fixed :class:`warp.HashGrid` built over the target points, so it
    can be called every ICP iteration on transformed source points without any
    rebuild -- the point-cloud analogue of :func:`closest_on_mesh`. Callable from
    your own :func:`warp.kernel` definitions.

    Args:
        grid: Identifier of the :class:`warp.HashGrid` over ``points``.
        points: Target points.
        normals: Per-point unit normals, one per entry of ``points``.
        p: Query point (a transformed source point).
        max_dist: Reject matches farther than this (also the search radius).

    Returns:
        A :class:`ClosestPoint`; check ``valid``.
    """
    out = ClosestPoint()
    out.valid = wp.int32(0)
    best = max_dist
    best_idx = wp.int32(-1)
    neighbors = wp.hash_grid_query(grid, p, max_dist)
    for j in neighbors:
        d = wp.length(points[j] - p)
        if d < best:
            best = d
            best_idx = j
    if best_idx >= 0:
        out.point = points[best_idx]
        out.normal = normals[best_idx]
        out.distance = best
        out.valid = wp.int32(1)
    return out


@wp.kernel(enable_backward=False)
def _icp_accumulate_points_kernel(
    source: wp.array(dtype=wp.vec3),
    rots: wp.array(dtype=wp.mat33),
    transs: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    num_source: wp.int32,
    stochastic: wp.int32,
    seed: wp.int32,
    inv_scale: wp.array(dtype=wp.float32),
    source_normals: wp.array(dtype=wp.vec3),
    symmetric: wp.int32,
    normal_mode: wp.int32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    rot = rots[0]
    trans = transs[0]
    idx = _source_index(tid, num_source, stochastic, seed)
    p = rot * source[idx] + trans
    cp = closest_on_points(grid, points, normals, p, max_dist)
    if cp.valid == 0:
        return
    n_q = _plane_normal(p, cp.point, cp.normal, normal_mode)
    n_p = wp.vec3(0.0, 0.0, 0.0)
    if symmetric != 0:
        n_p = rot * source_normals[idx]
    term = _correspondence_term(p, cp.point, n_q, n_p, symmetric)
    r = -term.b
    w = _robust_weight(r, inv_scale[0])
    _accumulate_normal_equations(term, w, a_upper, g)
    wp.atomic_add(stats, 0, r * r)
    wp.atomic_add(stats, 1, 1.0)


@wp.kernel(enable_backward=False)
def _icp_accumulate_mesh_batched_kernel(
    source: wp.array(dtype=wp.vec3),
    source_stride: wp.int32,
    rots: wp.array(dtype=wp.mat33),
    transs: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    max_dist: wp.float32,
    num_source: wp.int32,
    stochastic: wp.int32,
    seed: wp.int32,
    inv_scale_sq: wp.array(dtype=wp.float32),
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
):
    # 2D launch (batch b, thread t): each batch b carries its own transform, its
    # own robust scale, and its own 21/6/2 accumulator slice, all against the one
    # shared target mesh (the rigid-motion payoff extends to the whole batch).
    b, t = wp.tid()
    idx = _source_index(t, num_source, stochastic, seed + b)
    p = rots[b] * source[source_stride * b + idx] + transs[b]
    cp = closest_on_mesh(mesh, p, max_dist)
    if cp.valid == 0:
        return
    term = point_plane_term(p, cp.point, cp.normal)
    r = wp.dot(p - cp.point, cp.normal)
    w = _robust_weight(r, inv_scale_sq[b])
    _accumulate_normal_equations_at(term, w, b * 21, b * 6, a_upper, g)
    wp.atomic_add(stats, b * 2 + 0, r * r)
    wp.atomic_add(stats, b * 2 + 1, 1.0)


@wp.kernel(enable_backward=False)
def _icp_accumulate_points_batched_kernel(
    source: wp.array(dtype=wp.vec3),
    source_stride: wp.int32,
    rots: wp.array(dtype=wp.mat33),
    transs: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    num_source: wp.int32,
    stochastic: wp.int32,
    seed: wp.int32,
    inv_scale_sq: wp.array(dtype=wp.float32),
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
):
    b, t = wp.tid()
    idx = _source_index(t, num_source, stochastic, seed + b)
    p = rots[b] * source[source_stride * b + idx] + transs[b]
    cp = closest_on_points(grid, points, normals, p, max_dist)
    if cp.valid == 0:
        return
    term = point_plane_term(p, cp.point, cp.normal)
    r = wp.dot(p - cp.point, cp.normal)
    w = _robust_weight(r, inv_scale_sq[b])
    _accumulate_normal_equations_at(term, w, b * 21, b * 6, a_upper, g)
    wp.atomic_add(stats, b * 2 + 0, r * r)
    wp.atomic_add(stats, b * 2 + 1, 1.0)


@wp.kernel(enable_backward=False)
def _estimate_normals_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    min_neighbors: wp.int32,
    normals: wp.array(dtype=wp.vec3),
):
    # Per-point normal from a local PCA: the eigenvector of the neighborhood
    # covariance with the smallest eigenvalue is the surface normal. Its sign is
    # arbitrary, which is fine for point-to-plane (the normal equations are
    # invariant to flipping ``n``).
    tid = wp.tid()
    pi = points[tid]

    centroid = wp.vec3(0.0, 0.0, 0.0)
    count = wp.int32(0)
    q1 = wp.hash_grid_query(grid, pi, radius)
    for j in q1:
        if wp.length(points[j] - pi) <= radius:
            centroid += points[j]
            count += 1
    if count < min_neighbors:
        normals[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    centroid = centroid / float(count)

    cov = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q2 = wp.hash_grid_query(grid, pi, radius)
    for j in q2:
        if wp.length(points[j] - pi) <= radius:
            e = points[j] - centroid
            cov += wp.outer(e, e)

    vectors, values = wp.eig3(cov)
    smallest = wp.int32(0)
    if values[1] < values[smallest]:
        smallest = wp.int32(1)
    if values[2] < values[smallest]:
        smallest = wp.int32(2)
    normal = wp.vec3(vectors[0, smallest], vectors[1, smallest], vectors[2, smallest])
    normals[tid] = wp.normalize(normal)


##########################################################################
## On-device Gauss-Newton solve and transform update
##
## Keeping the 6x6 solve and the transform update on the GPU lets the whole ICP
## iteration run without a per-iteration host readback, so the loop has no host
## syncs (when early stopping is disabled) and is CUDA-graph capturable.
##########################################################################


@wp.func
def _packed_sym_entry(a_upper: wp.array(dtype=wp.float32), off: wp.int32, r: wp.int32, c: wp.int32) -> wp.float32:
    # Read entry (r, c) of the symmetric 6x6 from its packed upper triangle
    # (21 entries) starting at ``off`` -- the layout written by
    # ``_accumulate_normal_equations_at``.
    i = wp.min(r, c)
    k = wp.max(r, c)
    idx = i * 6 - (i * (i - 1)) / 2 + (k - i)
    return a_upper[off + idx]


@wp.func
def _solve_normal_equations(
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    off_a: wp.int32,
    off_g: wp.int32,
    damping: wp.float32,
) -> wp.spatial_vector:
    # Solve the damped 6x6 system ``(A + damping I) x = g`` for the 6-DOF increment
    # ``x = [rotation; translation]`` using a 3x3 Schur complement over the
    # rotation/translation blocks (Warp only inverts up to 4x4 directly).
    a_rr = wp.mat33()
    a_rt = wp.mat33()
    a_tt = wp.mat33()
    for i in range(3):
        for j in range(3):
            a_rr[i, j] = _packed_sym_entry(a_upper, off_a, i, j)
            a_rt[i, j] = _packed_sym_entry(a_upper, off_a, i, 3 + j)
            a_tt[i, j] = _packed_sym_entry(a_upper, off_a, 3 + i, 3 + j)
    damp = wp.mat33(damping, 0.0, 0.0, 0.0, damping, 0.0, 0.0, 0.0, damping)
    a_rr = a_rr + damp
    a_tt = a_tt + damp
    g_r = wp.vec3(g[off_g + 0], g[off_g + 1], g[off_g + 2])
    g_t = wp.vec3(g[off_g + 3], g[off_g + 4], g[off_g + 5])
    tt_inv = wp.inverse(a_tt)
    rt_tt_inv = a_rt * tt_inv
    schur = a_rr - rt_tt_inv * wp.transpose(a_rt)
    schur_inv = wp.inverse(schur)
    x_r = schur_inv * (g_r - rt_tt_inv * g_t)
    x_t = tt_inv * (g_t - wp.transpose(a_rt) * x_r)
    return wp.spatial_vector(x_r[0], x_r[1], x_r[2], x_t[0], x_t[1], x_t[2])


@wp.func
def _so3_exp_device(a: wp.vec3) -> wp.mat33:
    # Rotation matrix from a rotation vector via Rodrigues' formula.
    theta = wp.length(a)
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if theta < 1.0e-12:
        return identity
    k = a / theta
    kx = wp.mat33(0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0)
    return identity + wp.sin(theta) * kx + (1.0 - wp.cos(theta)) * (kx * kx)


@wp.kernel(enable_backward=False)
def _icp_solve_kernel(
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
    damping: wp.float32,
    robust_k: wp.float32,
    robust_enabled: wp.int32,
    rots: wp.array(dtype=wp.mat33),
    transs: wp.array(dtype=wp.vec3),
    inv_scale: wp.array(dtype=wp.float32),
    update_sq: wp.array(dtype=wp.float32),
):
    # One thread per problem: solve its 6x6 system, compose the increment onto its
    # transform in place, record its squared update norm, and set the robust scale
    # for the next iteration from this iterate's residual RMS -- all on the GPU.
    b = wp.tid()
    x = _solve_normal_equations(a_upper, g, b * 21, b * 6, damping)
    d_rot = _so3_exp_device(wp.vec3(x[0], x[1], x[2]))
    rots[b] = d_rot * rots[b]
    transs[b] = d_rot * transs[b] + wp.vec3(x[3], x[4], x[5])
    update_sq[b] = wp.dot(x, x)

    inv_scale[b] = 0.0
    if robust_enabled != 0:
        count = stats[b * 2 + 1]
        if count > 0.0:
            rmse = wp.sqrt(stats[b * 2 + 0] / count)
            if rmse > 0.0:
                s = robust_k * rmse
                inv_scale[b] = 1.0 / (s * s)


##########################################################################
## Morton (Z-order) reordering of source points
##
## The per-iteration cost is dominated by the closest-point query, whose BVH /
## hash-grid traversal is latency-bound. Reordering the source points along a
## Morton curve once (rigid motion preserves their relative layout) makes
## neighboring threads query neighboring regions, so the traversals stay cache-
## and warp-coherent. This is inert to the result -- the normal equations are a
## sum over correspondences, independent of order -- but speeds up the query at
## high occupancy (large point counts or batched solves).
##########################################################################


@wp.func
def _morton_spread(v: wp.uint32) -> wp.uint32:
    # Spread the low 10 bits of ``v`` to every third bit (Morton "Part1By2").
    x = v & wp.uint32(0x000003FF)
    x = (x ^ (x << wp.uint32(16))) & wp.uint32(0xFF0000FF)
    x = (x ^ (x << wp.uint32(8))) & wp.uint32(0x0300F00F)
    x = (x ^ (x << wp.uint32(4))) & wp.uint32(0x030C30C3)
    x = (x ^ (x << wp.uint32(2))) & wp.uint32(0x09249249)
    return x


@wp.kernel(enable_backward=False)
def _morton_key_kernel(
    points: wp.array(dtype=wp.vec3),
    lo: wp.vec3,
    inv_extent: wp.vec3,
    block_size: wp.int32,
    keys: wp.array(dtype=wp.int64),
):
    # A 30-bit Morton code of the point's cell, prefixed with its batch-block index
    # so a single sort keeps blocks contiguous and Z-orders within each.
    tid = wp.tid()
    p = points[tid]
    fx = wp.clamp((p[0] - lo[0]) * inv_extent[0], 0.0, 1.0) * 1023.0
    fy = wp.clamp((p[1] - lo[1]) * inv_extent[1], 0.0, 1.0) * 1023.0
    fz = wp.clamp((p[2] - lo[2]) * inv_extent[2], 0.0, 1.0) * 1023.0
    code = (
        _morton_spread(wp.uint32(fx))
        | (_morton_spread(wp.uint32(fy)) << wp.uint32(1))
        | (_morton_spread(wp.uint32(fz)) << wp.uint32(2))
    )
    block = wp.int64(tid / block_size)
    keys[tid] = (block << wp.int64(32)) | wp.int64(code)


@wp.kernel(enable_backward=False)
def _gather_vec3_kernel(order: wp.array(dtype=wp.int32), src: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = src[order[tid]]


def _morton_order(points, count, block_size, device):
    """Return a length-``count`` permutation that Z-orders ``points`` within each
    contiguous block of ``block_size`` (radix sort needs 2x storage)."""
    lo, hi = _bounding_box(points, device)
    inv_extent = 1.0 / np.maximum(hi - lo, 1.0e-9)
    keys = wp.empty(2 * count, dtype=wp.int64, device=device)
    order = wp.empty(2 * count, dtype=wp.int32, device=device)
    wp.launch(
        _morton_key_kernel,
        dim=count,
        inputs=[points, wp.vec3(*lo.astype(np.float32)), wp.vec3(*inv_extent.astype(np.float32)), int(block_size)],
        outputs=[keys],
        device=device,
    )
    wp.launch(_poisson_iota_kernel, dim=count, outputs=[order], device=device)
    radix_sort_pairs(keys, order, count=count)
    return order


def _reorder_points(points, order, count, device):
    out = wp.empty(count, dtype=wp.vec3, device=device)
    wp.launch(_gather_vec3_kernel, dim=count, inputs=[order, points], outputs=[out], device=device)
    return out


##########################################################################
## Host-side driver
##########################################################################


class RegistrationResult:
    """Outcome of :func:`register_rigid`.

    Attributes:
        transform: The ``(4, 4)`` rigid transform aligning source to target.
        rmse: Root-mean-square point-to-plane residual at the final iterate.
        iterations: Number of iterations run.
        converged: Whether the update-norm tolerance was met before ``max_iters``.
    """

    def __init__(self, transform, rmse, iterations, converged):
        self.transform = transform
        self.rmse = float(rmse)
        self.iterations = int(iterations)
        self.converged = bool(converged)


class BatchedRegistrationResult:
    """Outcome of :func:`register_rigid_batched` over ``B`` problems.

    Attributes:
        transforms: The ``(B, 4, 4)`` rigid transforms, one per problem.
        rmse: The ``(B,)`` final point-to-plane RMSE of each problem.
        iterations: Number of iterations run (shared across the batch).
        converged: The ``(B,)`` boolean convergence flags.
        best_index: Index of the lowest-RMSE problem (e.g. the winning
            initialization in a multi-start sweep).
        transform: The single ``(4, 4)`` transform of the best problem, for
            convenience.
    """

    def __init__(self, transforms, rmse, iterations, converged):
        self.transforms = np.asarray(transforms, dtype=np.float64)
        self.rmse = np.asarray(rmse, dtype=np.float64)
        self.iterations = int(iterations)
        self.converged = np.asarray(converged, dtype=bool)
        finite = np.where(np.isfinite(self.rmse), self.rmse, np.inf)
        self.best_index = int(np.argmin(finite))
        self.transform = self.transforms[self.best_index]


def _build_target_mesh(target, device) -> wp.Mesh:
    if isinstance(target, wp.Mesh):
        return target
    points, faces = target
    return wp.Mesh(
        points=_as_vec3_array(points, device),
        indices=wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
    )


def _is_mesh_target(target) -> bool:
    """A target is a mesh if it is a :class:`warp.Mesh` or a ``(points, faces)``
    pair whose second element has an integer dtype; otherwise it is a point cloud
    (bare points, or a ``(points, normals)`` pair with a floating-point second
    element)."""
    if isinstance(target, wp.Mesh):
        return True
    if isinstance(target, (tuple, list)) and len(target) == 2 and not isinstance(target[0], wp.array):
        second = np.asarray(target[1])
        return np.issubdtype(second.dtype, np.integer)
    return False


class _PointTarget:
    """A point-cloud target: a hash grid, points, per-point normals, and the
    correspondence search radius. Holds references so the grid outlives the ICP
    loop."""

    def __init__(self, grid, points, normals, search_radius):
        self.grid = grid
        self.points = points
        self.normals = normals
        self.search_radius = search_radius


def _build_target_points(target, device, max_corr_dist, estimate_normals=True) -> _PointTarget:
    if isinstance(target, (tuple, list)) and not isinstance(target[0], wp.array) and len(target) == 2:
        points, normals = target
        points = _as_vec3_array(points, device)
        normals = _as_vec3_array(normals, device)
    else:
        points = _as_vec3_array(target, device)
        normals = None

    num_points = points.shape[0]
    lo, hi = _bounding_box(points, device)
    diagonal = float(np.linalg.norm(hi - lo))
    spacing = diagonal / max(num_points ** (1.0 / 3.0), 1.0) if diagonal > 0.0 else 1.0

    search_radius = float(max_corr_dist) if max_corr_dist is not None else 5.0 * spacing
    normal_radius = 3.0 * spacing
    grid_radius = max(search_radius, normal_radius)

    extent = np.maximum(hi - lo, grid_radius)
    dims = np.clip(np.ceil(extent / grid_radius).astype(np.int64), 1, 512)
    grid = wp.HashGrid(int(dims[0]), int(dims[1]), int(dims[2]), device=device)
    grid.build(points, grid_radius)

    if normals is None:
        # Kept length ``num_points`` so ``closest_on_points`` can index it even
        # when the residual uses the closest-point direction (no PCA needed).
        normals = wp.zeros(num_points, dtype=wp.vec3, device=device)
        if estimate_normals:
            wp.launch(
                _estimate_normals_kernel,
                dim=num_points,
                inputs=[grid.id, points, float(normal_radius), 4],
                outputs=[normals],
                device=device,
            )

    return _PointTarget(grid, points, normals, search_radius)


def _estimate_normals(points, device):
    """Estimate per-point unit normals by local PCA over a hash grid (used for the
    source normals of the symmetric variant)."""
    num_points = points.shape[0]
    lo, hi = _bounding_box(points, device)
    diagonal = float(np.linalg.norm(hi - lo))
    spacing = diagonal / max(num_points ** (1.0 / 3.0), 1.0) if diagonal > 0.0 else 1.0
    radius = 3.0 * spacing

    extent = np.maximum(hi - lo, radius)
    dims = np.clip(np.ceil(extent / radius).astype(np.int64), 1, 512)
    grid = wp.HashGrid(int(dims[0]), int(dims[1]), int(dims[2]), device=device)
    grid.build(points, radius)

    normals = wp.zeros(num_points, dtype=wp.vec3, device=device)
    wp.launch(
        _estimate_normals_kernel,
        dim=num_points,
        inputs=[grid.id, points, float(radius), 4],
        outputs=[normals],
        device=device,
    )
    return normals


_ROBUST_KERNELS = {"welsch", "tukey"}
_VARIANTS = {"point_to_plane", "symmetric"}
_PLANE_NORMALS = {"surface", "closest_point"}


def register_rigid(
    source,
    target,
    *,
    init=None,
    max_iters: int = 50,
    tol: float = 1e-6,
    max_corr_dist: float | None = None,
    variant: str = "point_to_plane",
    plane_normal: str = "surface",
    source_normals=None,
    sample_count: int | None = None,
    robust: str | None = None,
    robust_k: float = 3.0,
    damping: float = 1e-9,
    seed: int = 0,
    device: DeviceLike | None = None,
) -> RegistrationResult:
    """Rigidly align a source point set to a target surface with point-to-plane ICP.

    Each iteration transforms the source by the current estimate, finds the
    closest point (and normal) on the *fixed* target, and takes a Gauss-Newton
    step of the linearized point-to-plane objective. Because the motion is rigid,
    the target's BVH is built once and never rebuilt.

    Optionally, following the practical insight of Bouaziz et al., "Sparse
    Iterative Closest Point" (2013), each iteration can use only a random subset
    of the source points (``sample_count``) and down-weight outlier
    correspondences with a robust kernel (``robust``), keeping the cheap 6x6
    solve while gaining speed and outlier tolerance.

    Args:
        source: Source points, a :class:`warp.array` of :class:`warp.vec3` or an
            array-like of shape ``(num_points, 3)``.
        target: Target surface. Either a mesh -- a :class:`warp.Mesh` or a
            ``(points, faces)`` pair -- or a point cloud -- a ``(num_points, 3)``
            array (normals are estimated by local PCA) or a ``(points, normals)``
            pair.
        init: Optional ``(4, 4)`` initial transform (defaults to identity).
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance on the 6-DOF update norm; iteration stops early
            once the update is smaller than this. The 6x6 solve and transform
            update run on the device, so early stopping costs only a single scalar
            readback per iteration. Set ``tol=0`` (or negative) to disable early
            stopping and run exactly ``max_iters`` iterations, in which case the
            loop performs no host synchronization and is CUDA-graph capturable.
        max_corr_dist: Reject correspondences farther than this. For mesh targets
            it defaults to no bound; for point-cloud targets it also sets the
            nearest-neighbor search radius and defaults to a few times the point
            spacing.
        variant: ``"point_to_plane"`` (default) or ``"symmetric"`` -- the
            symmetric objective of Rusinkiewicz 2019, which uses both surfaces'
            normals for a wider convergence basin. ``"symmetric"`` needs source
            normals: supply them via ``source_normals`` or they are estimated
            once by local PCA.
        plane_normal: Which normal defines the point-to-plane residual:
            ``"surface"`` (default) uses the target's geometric normal (the mesh
            face normal or the point cloud's PCA normal); ``"closest_point"`` uses
            the closest-point direction ``normalize(p - q)`` from the query
            instead, needing no precomputed normals. Only used by the
            ``"point_to_plane"`` variant.
        source_normals: Optional per-source-point unit normals (only used by the
            ``"symmetric"`` variant).
        sample_count: If set, use this many randomly drawn source points per
            iteration (sampling with replacement, reseeded each iteration)
            instead of all of them. Defaults to using every source point.
        robust: Robust weighting kernel for correspondences: ``"welsch"`` (or its
            alias ``"tukey"``) down-weights residuals by ``exp(-(r/s)^2)``, with
            the scale ``s`` adapted from the running residual RMS. ``None``
            (default) uses plain least squares.
        robust_k: Multiplier setting the robust scale ``s = robust_k * rmse``;
            larger keeps more correspondences at full weight. Only used when
            ``robust`` is set.
        damping: Levenberg-style diagonal damping added to the 6x6 system.
        seed: Base random seed for stochastic subsampling.
        device: Device on which to run. Defaults to the device of ``source``.

    Returns:
        A :class:`RegistrationResult`.
    """
    if robust is not None and robust not in _ROBUST_KERNELS:
        raise ValueError(f"`robust` must be one of {sorted(_ROBUST_KERNELS)} or None, got {robust!r}.")
    if variant not in _VARIANTS:
        raise ValueError(f"`variant` must be one of {sorted(_VARIANTS)}, got {variant!r}.")
    if plane_normal not in _PLANE_NORMALS:
        raise ValueError(f"`plane_normal` must be one of {sorted(_PLANE_NORMALS)}, got {plane_normal!r}.")
    device = (
        wp.get_device(device)
        if device is not None
        else (source.device if isinstance(source, wp.array) else wp.get_device())
    )
    device = wp.get_device(device)

    src = _as_vec3_array(source, device)
    n = src.shape[0]

    is_mesh = _is_mesh_target(target)
    if is_mesh:
        mesh = _build_target_mesh(target, device)
        max_dist = float(max_corr_dist) if max_corr_dist is not None else 1.0e30
    else:
        # The closest-point residual needs no target normals; skip PCA unless the
        # surface-normal residual or the symmetric variant will use them.
        need_normals = plane_normal == "surface" or variant == "symmetric"
        point_target = _build_target_points(target, device, max_corr_dist, estimate_normals=need_normals)
        max_dist = point_target.search_radius

    symmetric = 1 if variant == "symmetric" else 0
    normal_mode = 1 if plane_normal == "closest_point" else 0
    if symmetric:
        src_normals = (
            _as_vec3_array(source_normals, device) if source_normals is not None else _estimate_normals(src, device)
        )
    else:
        src_normals = wp.zeros(1, dtype=wp.vec3, device=device)

    # Z-order the source once so the per-iteration closest-point queries stay cache
    # coherent (inert to the result; see _morton_order).
    order = _morton_order(src, n, n, device)
    src = _reorder_points(src, order, n, device)
    if symmetric:
        src_normals = _reorder_points(src_normals, order, n, device)

    init = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)

    stochastic = 1 if (sample_count is not None and sample_count > 0) else 0
    num_threads = int(sample_count) if stochastic else n
    robust_enabled = 1 if robust is not None else 0

    # Transform state and accumulators all live on the device; the iteration reads
    # nothing back, so with ``tol <= 0`` (fixed iterations) the loop has no host
    # syncs and is CUDA-graph capturable.
    rots = wp.array(init[:3, :3][None].astype(np.float32), dtype=wp.mat33, device=device)
    transs = wp.array(init[:3, 3][None].astype(np.float32), dtype=wp.vec3, device=device)
    inv_scale = wp.zeros(1, dtype=wp.float32, device=device)
    update_sq = wp.zeros(1, dtype=wp.float32, device=device)
    a_upper = wp.zeros(21, dtype=wp.float32, device=device)
    g = wp.zeros(6, dtype=wp.float32, device=device)
    stats = wp.zeros(2, dtype=wp.float32, device=device)

    early_stop = tol > 0.0
    converged = False
    iterations = max_iters
    for step in range(max_iters):
        a_upper.zero_()
        g.zero_()
        stats.zero_()
        iter_seed = int(seed) + step
        if is_mesh:
            accum_inputs = [src, rots, transs, mesh.id, max_dist]
        else:
            accum_inputs = [
                src,
                rots,
                transs,
                point_target.grid.id,
                point_target.points,
                point_target.normals,
                max_dist,
            ]
        wp.launch(
            _icp_accumulate_mesh_kernel if is_mesh else _icp_accumulate_points_kernel,
            dim=num_threads,
            inputs=[*accum_inputs, n, stochastic, iter_seed, inv_scale, src_normals, symmetric, normal_mode],
            outputs=[a_upper, g, stats],
            device=device,
        )
        wp.launch(
            _icp_solve_kernel,
            dim=1,
            inputs=[a_upper, g, stats, float(damping), float(robust_k), robust_enabled],
            outputs=[rots, transs, inv_scale, update_sq],
            device=device,
        )
        if early_stop and float(np.sqrt(update_sq.numpy()[0])) < tol:
            converged = True
            iterations = step + 1
            break

    rot = rots.numpy()[0].astype(np.float64)
    trans = transs.numpy()[0].astype(np.float64)
    st = stats.numpy()
    count = float(st[1])
    rmse = float(np.sqrt(st[0] / count)) if count > 0 else float("inf")

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return RegistrationResult(transform=transform, rmse=rmse, iterations=iterations, converged=converged)


def register_rigid_batched(
    source,
    target,
    inits,
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
    max_corr_dist: float | None = None,
    sample_count: int | None = None,
    robust: str | None = None,
    robust_k: float = 3.0,
    damping: float = 1e-9,
    seed: int = 0,
    device: DeviceLike | None = None,
) -> BatchedRegistrationResult:
    """Run ``B`` independent point-to-plane ICP problems in parallel.

    All ``B`` problems share a single target, so its acceleration structure is
    built once and queried by the whole batch. The correspondences of every
    problem accumulate into their own ``6x6`` system in one GPU launch; the small
    per-problem solves are done on the host.

    This exposes multi-initialization (pass several ``inits`` with one shared
    ``source`` to escape the narrow basin of a single start and keep the best
    result via :attr:`BatchedRegistrationResult.best_index`) and multi-source
    batching (pass a ``(B, num_points, 3)`` ``source``). The target is shared
    across the batch.

    Args:
        source: Either a single ``(num_points, 3)`` source shared by all problems,
            or a ``(B, num_points, 3)`` stack of per-problem sources.
        target: Target surface, as in :func:`register_rigid`.
        inits: ``(B, 4, 4)`` initial transforms, one per problem.
        max_iters: Maximum number of iterations (shared across the batch).
        tol: Convergence tolerance on each problem's 6-DOF update norm; iteration
            stops once every problem is within ``tol``. Each problem's 6x6 solve
            runs on the device; set ``tol=0`` to run exactly ``max_iters``
            iterations with no host synchronization (CUDA-graph capturable).
        max_corr_dist: Reject correspondences farther than this (see
            :func:`register_rigid`).
        sample_count: Optional per-iteration stochastic subsample size.
        robust: Optional robust weighting kernel (see :func:`register_rigid`).
        robust_k: Robust-scale multiplier.
        damping: Levenberg-style diagonal damping added to each 6x6 system.
        seed: Base random seed for stochastic subsampling.
        device: Device on which to run.

    Returns:
        A :class:`BatchedRegistrationResult`.
    """
    if robust is not None and robust not in _ROBUST_KERNELS:
        raise ValueError(f"`robust` must be one of {sorted(_ROBUST_KERNELS)} or None, got {robust!r}.")

    device = (
        wp.get_device(device)
        if device is not None
        else (source.device if isinstance(source, wp.array) else wp.get_device())
    )
    device = wp.get_device(device)

    inits = np.asarray(inits, dtype=np.float64)
    if inits.ndim != 3 or inits.shape[1:] != (4, 4):
        raise ValueError(f"`inits` must have shape (B, 4, 4), got {inits.shape}.")
    batch_size = inits.shape[0]

    source_np = np.asarray(source.numpy() if isinstance(source, wp.array) else source, dtype=np.float32)
    if source_np.ndim == 3:
        # Per-problem sources: (B, N, 3).
        if source_np.shape[0] != batch_size:
            raise ValueError(f"multi-source batch {source_np.shape[0]} != inits batch {batch_size}.")
        n = source_np.shape[1]
        source_stride = n
        src = wp.array(source_np.reshape(-1, 3), dtype=wp.vec3, device=device)
    else:
        n = source_np.shape[0]
        source_stride = 0
        src = wp.array(source_np, dtype=wp.vec3, device=device)

    # Z-order the source(s) once so the per-iteration closest-point queries stay
    # cache coherent at batch occupancy (inert to the result; see _morton_order).
    total = src.shape[0]
    src = _reorder_points(src, _morton_order(src, total, n, device), total, device)

    is_mesh = _is_mesh_target(target)
    if is_mesh:
        mesh = _build_target_mesh(target, device)
        max_dist = float(max_corr_dist) if max_corr_dist is not None else 1.0e30
    else:
        point_target = _build_target_points(target, device, max_corr_dist)
        max_dist = point_target.search_radius

    stochastic = 1 if (sample_count is not None and sample_count > 0) else 0
    num_threads = int(sample_count) if stochastic else n
    robust_enabled = 1 if robust is not None else 0

    # Per-problem transforms and robust scales stay on the device; each iteration
    # accumulates every problem's system and solves them all in one solve launch,
    # with no host readback (so ``tol <= 0`` is graph capturable).
    rots = wp.array(inits[:, :3, :3].astype(np.float32), dtype=wp.mat33, device=device)
    transs = wp.array(inits[:, :3, 3].astype(np.float32), dtype=wp.vec3, device=device)
    inv_scale = wp.zeros(batch_size, dtype=wp.float32, device=device)
    update_sq = wp.zeros(batch_size, dtype=wp.float32, device=device)
    a_upper = wp.zeros(batch_size * 21, dtype=wp.float32, device=device)
    g = wp.zeros(batch_size * 6, dtype=wp.float32, device=device)
    stats = wp.zeros(batch_size * 2, dtype=wp.float32, device=device)

    early_stop = tol > 0.0
    iterations = max_iters
    for step in range(max_iters):
        a_upper.zero_()
        g.zero_()
        stats.zero_()
        iter_seed = int(seed) + step * batch_size
        if is_mesh:
            accum = _icp_accumulate_mesh_batched_kernel
            accum_inputs = [src, source_stride, rots, transs, mesh.id, max_dist]
        else:
            accum = _icp_accumulate_points_batched_kernel
            accum_inputs = [
                src, source_stride, rots, transs,
                point_target.grid.id, point_target.points, point_target.normals, max_dist,
            ]  # fmt: skip
        wp.launch(
            accum,
            dim=(batch_size, num_threads),
            inputs=[*accum_inputs, n, stochastic, iter_seed, inv_scale],
            outputs=[a_upper, g, stats],
            device=device,
        )
        wp.launch(
            _icp_solve_kernel,
            dim=batch_size,
            inputs=[a_upper, g, stats, float(damping), float(robust_k), robust_enabled],
            outputs=[rots, transs, inv_scale, update_sq],
            device=device,
        )
        if early_stop and float(np.sqrt(update_sq.numpy().max())) < tol:
            iterations = step + 1
            break

    rots_np = rots.numpy().astype(np.float64)
    transs_np = transs.numpy().astype(np.float64)
    st = stats.numpy().reshape(batch_size, 2)
    counts = st[:, 1]
    rmse = np.where(counts > 0, np.sqrt(np.where(counts > 0, st[:, 0], 0.0) / np.maximum(counts, 1.0)), np.inf)
    converged = np.sqrt(update_sq.numpy()) < max(tol, 1.0e-9)

    transforms = np.tile(np.eye(4), (batch_size, 1, 1))
    transforms[:, :3, :3] = rots_np
    transforms[:, :3, 3] = transs_np
    return BatchedRegistrationResult(transforms=transforms, rmse=rmse, iterations=iterations, converged=converged)
