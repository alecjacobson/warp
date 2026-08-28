# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.utils import array_scan

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
    instance from :attr:`UniformSampler.state` and pass it as a kernel argument.
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
def draw(state: UniformSamplerState, rng: wp.uint32) -> MeshSample:
    """Draw one point uniformly over the surface of a mesh.

    A triangle is chosen with probability proportional to its area via a binary
    search of the cumulative distribution in ``state``, then a point is drawn
    uniformly within that triangle. Callable from within a :func:`warp.kernel`.

    Args:
        state: Sampler state, typically :attr:`UniformSampler.state` passed as a
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
