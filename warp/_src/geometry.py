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


def _swept_volume_bounds(meshes, transforms_np, margin):
    """World-space axis-aligned bounds of every mesh over every sampled pose.

    Transforms each mesh's rest-pose corner cloud by all of its poses and unions
    the results, then pads by ``margin`` on every side.
    """
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)

    for m, mesh in enumerate(meshes):
        pts = mesh.points.numpy().astype(np.float64)
        if pts.size == 0:
            continue
        # Rest-pose AABB corners are enough: a rigid map sends the AABB into a
        # box whose extent is bounded by transforming the 8 corners.
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        corners = np.array(
            [[[lo[0], hi[0]][a], [lo[1], hi[1]][b], [lo[2], hi[2]][c]] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        )

        for pose in transforms_np[m]:
            t = pose[:3]
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
            # Rotate the corners by the pose quaternion (x, y, z, w) and translate.
            rotated = _quat_rotate_np(np.array([qx, qy, qz, qw]), corners) + t
            lower = np.minimum(lower, rotated.min(axis=0))
            upper = np.maximum(upper, rotated.max(axis=0))

    if not np.all(np.isfinite(lower)):
        raise ValueError("Could not compute swept-volume bounds: all input meshes are empty.")

    lower -= margin
    upper += margin
    return lower, upper


def _quat_rotate_np(q, v):
    """Rotate row vectors ``v`` (..., 3) by quaternion ``q`` = (x, y, z, w)."""
    x, y, z, w = q
    # Rotation matrix from a unit quaternion.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return v @ R.T


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

    device = wp.get_device(device) if device is not None else meshes[0].device

    mesh_ids = wp.array([mesh.id for mesh in meshes], dtype=wp.uint64, device=device)
    transforms_wp = _swept_volume_transforms(transforms, device)
    if transforms_wp.shape[0] != len(meshes):
        raise ValueError(f"'transforms' has {transforms_wp.shape[0]} rows but there are {len(meshes)} meshes.")
    transforms_np = transforms_wp.numpy()

    if margin is None:
        margin = 2.0 * voxel_size if voxel_size is not None else 0.0

    lower, upper = _swept_volume_bounds(meshes, transforms_np, margin)
    extent = upper - lower

    if resolution is not None:
        dims = tuple(int(n) for n in resolution)
        if any(n < 2 for n in dims):
            raise ValueError(f"'resolution' must be at least 2 along each axis, got {dims}.")
    else:
        # Number of nodes = number of cells + 1; guarantee at least 2 nodes.
        dims = tuple(max(2, int(math.ceil(e / voxel_size)) + 1) for e in extent)

    # Snap the upper corner so the node spacing is exactly (extent / cells).
    spacing = np.array([extent[a] / (dims[a] - 1) for a in range(3)], dtype=np.float64)
    upper = lower + spacing * (np.array(dims) - 1)

    if max_dist is None:
        max_dist = float(np.linalg.norm(extent))

    origin = wp.vec3(float(lower[0]), float(lower[1]), float(lower[2]))
    spacing_v = wp.vec3(float(spacing[0]), float(spacing[1]), float(spacing[2]))

    field = wp.empty(dims, dtype=wp.float32, device=device)
    wp.launch(
        swept_volume_field_kernel,
        dim=dims,
        inputs=[mesh_ids, transforms_wp, origin, spacing_v, float(max_dist), int(sign_mode)],
        outputs=[field],
        device=device,
    )

    lower_v = wp.vec3(float(lower[0]), float(lower[1]), float(lower[2]))
    upper_v = wp.vec3(float(upper[0]), float(upper[1]), float(upper[2]))
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
