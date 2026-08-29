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

    def __init__(self, points, faces, *, device: DeviceLike | None = None):
        if device is None:
            device = points.device if isinstance(points, wp.array) else wp.get_device()
        self.device = wp.get_device(device)

        self.points = _as_vec3_array(points, self.device)
        self.indices = _as_index_array(faces, self.device)
        self.num_triangles = self.indices.shape[0] // 3
        if self.num_triangles == 0:
            raise ValueError("`faces` must describe at least one triangle.")

        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

        # Build the normalized cumulative area distribution once.
        areas = wp.empty(self.num_triangles, dtype=wp.float32, device=self.device)
        wp.launch(
            _triangle_areas_kernel,
            dim=self.num_triangles,
            inputs=[self.points, self.indices],
            outputs=[areas],
            device=self.device,
        )
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
## accept passes are shared. Kept as a separate kernel so the Euclidean path
## above is untouched (no extra arguments, no branch).
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
def _gather_vec3_kernel(
    order: wp.array(dtype=wp.int32),
    in_v: wp.array(dtype=wp.vec3),
    out_v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_v[tid] = in_v[order[tid]]


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
        geodesic: If set, use the approximate geodesic metric for the minimum
            distance instead of the Euclidean one. Uses a single sample per grid
            cell (the paper's multiple-samples-per-cell extension for very thin
            features is not implemented).
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
        device: DeviceLike | None = None,
    ):
        if radius <= 0.0:
            raise ValueError(f"`radius` must be positive, got {radius}.")
        if num_candidates is not None and num_candidates < 1:
            raise ValueError(f"`num_candidates` must be at least 1, got {num_candidates}.")

        self._sampler = UniformSampler(points, faces, device=device)
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

        # Geodesic mode also needs a surface normal per candidate (face normal).
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
        cand_points, cand_faces, cand_uv, priority, order = self._sort_by_cell(
            cand_points, cand_faces, cand_uv, priority, lo, hi
        )
        if self.geodesic:
            sorted_normals = wp.empty(self.num_candidates, dtype=wp.vec3, device=self.device)
            wp.launch(
                _gather_vec3_kernel,
                dim=self.num_candidates,
                inputs=[order, cand_normals],
                outputs=[sorted_normals],
                device=self.device,
            )
            cand_normals = sorted_normals

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
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
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
        # ``order`` holds the sort permutation in its first n entries, so callers
        # can gather auxiliary per-candidate arrays (e.g. normals) the same way.
        return out_points, out_faces, out_uv, out_priority, order

    def _solve_hash(
        self,
        cand_points: wp.array,
        priority: wp.array,
        lo: np.ndarray,
        hi: np.ndarray,
        seed: int,
        cand_normals: wp.array | None = None,
    ) -> wp.array:
        """The paper's Euclidean path: single-entry spatial hash + 27 phase groups.

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
