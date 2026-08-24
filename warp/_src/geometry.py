# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.marching_cubes import MarchingCubes
from warp._src.types import type_repr, types_equal

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

##########################################################################
## Swept volume (motion envelope) of animated rigid meshes
##########################################################################


class SweptVolumeSign(enum.IntEnum):
    """Method used to classify inside/outside when evaluating the per-mesh SDF.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel and compared against.
    """

    NORMAL = 0
    """Sign from the closest face's normal, via
    :func:`warp.mesh_query_point_sign_normal`. Fast; assumes each input mesh is
    watertight and consistently oriented."""

    WINDING_NUMBER = 1
    """Sign from the solid angle (generalized winding number), via
    :func:`warp.mesh_query_point_sign_winding_number`. Robust to non-watertight
    or inconsistently oriented meshes, but slower and requires every input mesh
    to be built with ``support_winding_number=True``."""


@wp.func
def swept_volume_sdf(
    p: wp.vec3,
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    max_dist: wp.float32,
    sign_mode: wp.int32,
) -> wp.float32:
    # Signed distance from world-space point ``p`` to the union of every input
    # mesh over every sampled pose, i.e. ``min_m min_s sdf_m(X[m, s]^-1 p)``.
    #
    # The motion is rigid, so instead of transforming the geometry we push the
    # query point back into each mesh's rest frame. One closest-point query per
    # (mesh, sample) therefore evaluates one pose, and a single call here folds
    # the whole space-time union into one scalar. Returns ``+max_dist`` when no
    # face lies within ``max_dist`` of ``p`` at any pose. Callable from within
    # user kernels.
    num_meshes = mesh_ids.shape[0]
    num_samples = transforms.shape[1]

    best = max_dist
    for m in range(num_meshes):
        mesh_id = mesh_ids[m]
        for s in range(num_samples):
            # Map the query point into the mesh's rest frame for this pose. The
            # map is rigid, so distances are preserved between the two frames.
            p_local = wp.transform_point(wp.transform_inverse(transforms[m, s]), p)

            if sign_mode == wp.int32(SweptVolumeSign.WINDING_NUMBER.value):
                query = wp.mesh_query_point_sign_winding_number(mesh_id, p_local, max_dist)
            else:
                query = wp.mesh_query_point_sign_normal(mesh_id, p_local, max_dist)

            if query.result:
                closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                dist = query.sign * wp.length(p_local - closest)
                best = wp.min(best, dist)

    return best


@wp.kernel
def swept_volume_field_kernel(
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    origin: wp.vec3,
    spacing: wp.vec3,
    max_dist: wp.float32,
    sign_mode: wp.int32,
    out_field: wp.array3d[wp.float32],
):
    i, j, k = wp.tid()
    p = origin + wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    out_field[i, j, k] = swept_volume_sdf(p, mesh_ids, transforms, max_dist, sign_mode)


def _swept_volume_transforms(transforms, device):
    """Normalize the ``transforms`` argument into a ``(num_meshes, num_samples)``
    :class:`warp.array2d` of :class:`warp.transform` on ``device``."""
    if isinstance(transforms, wp.array):
        if transforms.ndim != 2:
            raise ValueError(
                f"'transforms' must be a 2D array of shape (num_meshes, num_samples), got ndim {transforms.ndim}."
            )
        if not types_equal(transforms.dtype, wp.transform):
            raise TypeError(f"'transforms' must have dtype wp.transform, got {type_repr(transforms.dtype)}.")
        return transforms.to(device)

    # Accept nested Python sequences / NumPy arrays of shape (M, S, 7).
    arr = np.asarray(transforms, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 7:
        raise ValueError(
            "'transforms' must be a wp.array2d(dtype=wp.transform) or an array of "
            f"shape (num_meshes, num_samples, 7), got shape {arr.shape}."
        )
    return wp.array2d(arr, dtype=wp.transform, device=device)


@wp.kernel(enable_backward=False)
def swept_volume_mesh_aabb_kernel(
    points: wp.array[wp.vec3],
    slot: int,
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
):
    # Rest-pose axis-aligned bounds of one mesh, reduced into slot ``slot``.
    i = wp.tid()
    p = points[i]
    wp.atomic_min(out_lower, slot, p)
    wp.atomic_max(out_upper, slot, p)


@wp.kernel(enable_backward=False)
def swept_volume_bounds_kernel(
    mesh_lower: wp.array[wp.vec3],
    mesh_upper: wp.array[wp.vec3],
    transforms: wp.array2d[wp.transform],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
):
    # Union, over every (mesh, sample), of the pose-transformed rest-pose AABB.
    m, s = wp.tid()
    lo = mesh_lower[m]
    hi = mesh_upper[m]
    if lo[0] > hi[0]:  # empty mesh: nothing was reduced into this slot
        return
    xform = transforms[m, s]
    # A rigid map sends the box to one whose axis-aligned bounds are spanned by
    # its eight transformed corners, so it suffices to reduce those.
    for a in range(2):
        cx = wp.where(a == 0, lo[0], hi[0])
        for b in range(2):
            cy = wp.where(b == 0, lo[1], hi[1])
            for c in range(2):
                cz = wp.where(c == 0, lo[2], hi[2])
                p = wp.transform_point(xform, wp.vec3(cx, cy, cz))
                wp.atomic_min(out_lower, 0, p)
                wp.atomic_max(out_upper, 0, p)


def _swept_volume_bounds(meshes, transforms, margin, device):
    """World-space axis-aligned bounds of every mesh over every sampled pose.

    Reduces each mesh's rest-pose vertices to an axis-aligned box, then transforms
    the eight box corners by every pose and reduces their union, entirely with
    Warp kernels. Pads by ``margin`` on every side and returns ``(lower, upper)``
    as :class:`warp.vec3`.
    """
    num_meshes, num_samples = transforms.shape[0], transforms.shape[1]

    pos_inf = wp.vec3(math.inf, math.inf, math.inf)
    neg_inf = wp.vec3(-math.inf, -math.inf, -math.inf)

    mesh_lower = wp.full(num_meshes, pos_inf, dtype=wp.vec3, device=device)
    mesh_upper = wp.full(num_meshes, neg_inf, dtype=wp.vec3, device=device)
    for m, mesh in enumerate(meshes):
        if mesh.points.shape[0] > 0:
            wp.launch(
                swept_volume_mesh_aabb_kernel,
                dim=mesh.points.shape[0],
                inputs=[mesh.points, m],
                outputs=[mesh_lower, mesh_upper],
                device=device,
            )

    out_lower = wp.full(1, pos_inf, dtype=wp.vec3, device=device)
    out_upper = wp.full(1, neg_inf, dtype=wp.vec3, device=device)
    wp.launch(
        swept_volume_bounds_kernel,
        dim=(num_meshes, num_samples),
        inputs=[mesh_lower, mesh_upper, transforms],
        outputs=[out_lower, out_upper],
        device=device,
    )

    # Read back the six numbers (as the OBB code does) and pad on the host.
    lo = out_lower.numpy()[0]
    hi = out_upper.numpy()[0]
    if not math.isfinite(float(lo[0])):
        raise ValueError("Could not compute swept-volume bounds: all input meshes are empty.")
    lower = wp.vec3(float(lo[0]) - margin, float(lo[1]) - margin, float(lo[2]) - margin)
    upper = wp.vec3(float(hi[0]) + margin, float(hi[1]) + margin, float(hi[2]) + margin)
    return lower, upper


def swept_volume_field(
    meshes,
    transforms,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    margin: float | None = None,
    max_dist: float | None = None,
    sign_mode: SweptVolumeSign = SweptVolumeSign.NORMAL,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.vec3, wp.vec3]:
    """Sample the swept-volume signed-distance field on a dense regular grid.

    Computes, at each grid node ``p``, the signed distance to the union of the
    input meshes over all sampled poses::

        D(p) = min_m min_s  sdf_m( X[m, s]^-1 p )

    by brute force over every (mesh, sample) pair ("dense time stamping"). The
    field is negative inside the swept volume and positive outside, so extracting
    its zero isosurface (see :func:`swept_volume`) yields the motion envelope.

    Because the poses are the *provided* samples, the field only accounts for the
    geometry at those instants; motion between consecutive samples is not
    conservatively bounded. Supply a sufficiently fine time sampling for the
    desired tolerance.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses, either as a
            :class:`warp.array2d` of :class:`warp.transform` with shape
            ``(num_meshes, num_samples)`` or as an array of shape
            ``(num_meshes, num_samples, 7)`` (translation ``xyz`` followed by
            quaternion ``xyzw``).
        voxel_size: Edge length of a grid cell in world units. Required unless
            ``resolution`` is given.
        resolution: Optional explicit node counts ``(nx, ny, nz)``. Overrides
            ``voxel_size`` for choosing the grid dimensions.
        margin: Padding added on every side of the swept bounding box, in world
            units. Defaults to twice ``voxel_size`` so the surface is not
            clipped, or to ``0`` when only ``resolution`` is given (pass an
            explicit ``margin`` to avoid clipping the surface at the boundary).
        max_dist: Maximum search distance for the closest-point queries. Nodes
            farther than this from every posed mesh are left at ``+max_dist``.
            Defaults to the grid's diagonal length so the field is valid
            everywhere.
        sign_mode: Inside/outside classification method (see
            :class:`SweptVolumeSign`).
        device: Device on which to build the field. Defaults to the device of
            the first mesh.

    Returns:
        A tuple ``(field, lower, upper)`` where ``field`` is a
        ``wp.array3d(dtype=wp.float32)`` of signed distances, and ``lower`` and
        ``upper`` are the :class:`warp.vec3` world coordinates that grid nodes
        ``(0, 0, 0)`` and ``(nx-1, ny-1, nz-1)`` map to.
    """
    if len(meshes) == 0:
        raise ValueError("'meshes' must contain at least one mesh.")
    if voxel_size is None and resolution is None:
        raise ValueError("Provide either 'voxel_size' or 'resolution'.")
    if voxel_size is not None and voxel_size <= 0.0:
        raise ValueError(f"'voxel_size' must be positive, got {voxel_size}.")

    dims = None
    if resolution is not None:
        dims = tuple(int(n) for n in resolution)
        if len(dims) != 3:
            raise ValueError(f"'resolution' must have exactly 3 entries, got {dims}.")
        if any(n < 2 for n in dims):
            raise ValueError(f"'resolution' must be at least 2 along each axis, got {dims}.")

    device = wp.get_device(device) if device is not None else meshes[0].device

    # The kernel launches on a single device, so every mesh must already live there.
    for i, mesh in enumerate(meshes):
        if mesh.device != device:
            raise ValueError(f"'meshes[{i}]' is on device '{mesh.device}' but the launch device is '{device}'.")
        # Querying the winding number of a mesh built without it returns garbage
        # signs rather than failing, so reject that combination up front.
        if sign_mode == SweptVolumeSign.WINDING_NUMBER and not mesh.support_winding_number:
            raise ValueError(
                f"'meshes[{i}]' was not built with support_winding_number=True, which "
                "SweptVolumeSign.WINDING_NUMBER requires."
            )

    mesh_ids = wp.array([mesh.id for mesh in meshes], dtype=wp.uint64, device=device)
    transforms_wp = _swept_volume_transforms(transforms, device)
    if transforms_wp.shape[0] != len(meshes):
        raise ValueError(f"'transforms' has {transforms_wp.shape[0]} rows but there are {len(meshes)} meshes.")
    if transforms_wp.shape[1] == 0:
        raise ValueError("'transforms' must contain at least one pose sample.")

    if margin is None:
        margin = 2.0 * voxel_size if voxel_size is not None else 0.0

    lower, upper = _swept_volume_bounds(meshes, transforms_wp, margin, device)
    extent = (upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2])

    if dims is None:
        # Number of nodes = number of cells + 1; guarantee at least 2 nodes.
        dims = tuple(max(2, int(math.ceil(extent[a] / voxel_size)) + 1) for a in range(3))

    # Snap the upper corner so the node spacing is exactly (extent / cells).
    spacing = tuple(extent[a] / (dims[a] - 1) for a in range(3))
    upper = wp.vec3(*(lower[a] + spacing[a] * (dims[a] - 1) for a in range(3)))

    if max_dist is None:
        max_dist = math.sqrt(extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2])

    origin = lower
    spacing_v = wp.vec3(*spacing)

    field = wp.empty(dims, dtype=wp.float32, device=device)
    wp.launch(
        swept_volume_field_kernel,
        dim=dims,
        inputs=[mesh_ids, transforms_wp, origin, spacing_v, float(max_dist), int(sign_mode)],
        outputs=[field],
        device=device,
    )

    lower_v = lower
    upper_v = upper
    return field, lower_v, upper_v


def swept_volume(
    meshes,
    transforms,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    margin: float | None = None,
    max_dist: float | None = None,
    iso: float = 0.0,
    sign_mode: SweptVolumeSign = SweptVolumeSign.NORMAL,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array]:
    """Extract the swept volume (motion envelope) of animated rigid meshes.

    Samples the swept-volume signed-distance field with
    :func:`swept_volume_field` and extracts its ``iso`` isosurface with marching
    cubes, returning a single triangle mesh in world coordinates that encloses
    the union of every input mesh over every sampled pose.

    This is the dense-stamping baseline: the field is evaluated by brute force at
    the provided pose samples, with no root finding or narrow-band acceleration.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses; see
            :func:`swept_volume_field`.
        voxel_size: Edge length of a grid cell in world units. Required unless
            ``resolution`` is given.
        resolution: Optional explicit node counts ``(nx, ny, nz)``.
        margin: Padding added on every side of the swept bounding box, in world
            units. Defaults to twice ``voxel_size`` (or ``0`` when only
            ``resolution`` is given); see :func:`swept_volume_field`.
        max_dist: Maximum closest-point search distance; see
            :func:`swept_volume_field`.
        iso: Field level to extract. ``0.0`` traces the envelope through the
            sampled poses; a positive value dilates it outward. Marching cubes
            reconstructs the 1-Lipschitz field by linear interpolation, which at
            sharp convex features overestimates the field and pulls the surface
            inside the true one, so a stamped pose can poke through the ``0.0``
            isosurface by up to the grid's covering radius. Extracting at
            ``iso = 0.5 * hypot(hx, hy, hz)`` (the covering radius, ``sqrt(3)/2 *
            voxel_size`` for a cubic cell of the actual spacings ``hx, hy, hz``)
            guarantees every stamped pose stays enclosed.
        sign_mode: Inside/outside classification method (see
            :class:`SweptVolumeSign`).
        device: Device on which to run. Defaults to the device of the first mesh.

    Returns:
        A tuple ``(vertices, indices)`` where ``vertices`` is a
        ``wp.array(dtype=wp.vec3)`` in world coordinates and ``indices`` is a
        flat ``wp.array(dtype=wp.int32)`` with three entries per triangle.
    """
    field, lower, upper = swept_volume_field(
        meshes,
        transforms,
        voxel_size,
        resolution=resolution,
        margin=margin,
        max_dist=max_dist,
        sign_mode=sign_mode,
        device=device,
    )

    return MarchingCubes.extract_surface_marching_cubes(
        field,
        threshold=iso,
        domain_bounds_lower_corner=lower,
        domain_bounds_upper_corner=upper,
    )
