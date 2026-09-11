# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp

if TYPE_CHECKING:
    from warp._src.context import DeviceLike


_SF_PHI = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
"""First Super-Fibonacci spiral constant, ``1 / sqrt(2)``."""

_SF_PSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))
"""Second Super-Fibonacci spiral constant, the reciprocal of the plastic-like root."""


@wp.func
def super_fibonacci(i: int, n: int) -> wp.quat:
    # Maps sample index ``i`` in ``[0, n)`` to a near-uniformly distributed unit
    # quaternion on SO(3) via the Super-Fibonacci spiral (Alexa, "Super-Fibonacci
    # Spirals: Fast, Low-Discrepancy Sampling of SO(3)", CVPR 2022).
    #
    # Both spiral angles grow linearly with ``i``, and only their fractional turn
    # matters. They are therefore wrapped into ``[0, 1)`` in double precision
    # before reaching sin/cos: in float32 the angle for ``i`` in the tens of
    # thousands is large enough that argument reduction consumes the fractional
    # bits the spiral depends on, and the samples stop being low-discrepancy.
    s = wp.float64(i) + wp.float64(0.5)
    a = s * _SF_PHI
    b = s * _SF_PSI
    alpha = wp.float32(a - wp.floor(a)) * 2.0 * wp.pi
    beta = wp.float32(b - wp.floor(b)) * 2.0 * wp.pi

    t = wp.float32(s / wp.float64(n))
    r = wp.sqrt(t)
    R = wp.sqrt(1.0 - t)
    return wp.quat(r * wp.sin(alpha), r * wp.cos(alpha), R * wp.sin(beta), R * wp.cos(beta))


class OBBMeasureType(enum.IntEnum):
    """Objective minimized when searching for an oriented bounding box.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel launch (Warp kernels cannot take a ``str`` parameter).
    """

    VOLUME = 0
    """Minimize the volume of the bounding box."""
    SURFACE_AREA = 1
    """Minimize the surface area of the bounding box."""


_OBB_POINT_CHUNKS = 256
"""Threads each candidate OBB orientation splits its point loop across."""


@wp.kernel(enable_backward=False)
def oriented_bounding_box_samples_kernel(num_samples: int, rotations: wp.array(dtype=wp.quat)):
    # Fill the spiral portion of the candidate list. Any extra candidates are
    # written into the slots past ``num_samples`` by the kernels below.
    i = wp.tid()
    rotations[i] = super_fibonacci(i, num_samples)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_identity_kernel(slot: int, rotations: wp.array(dtype=wp.quat)):
    # The spiral never contains the identity exactly, so the axis-aligned box is
    # not otherwise among the candidates.
    rotations[slot] = wp.quat_identity()


@wp.kernel(enable_backward=False)
def point_sum_kernel(points: wp.array(dtype=wp.vec3), out_sum: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wp.atomic_add(out_sum, 0, points[i])


@wp.kernel(enable_backward=False)
def point_covariance_kernel(
    points: wp.array(dtype=wp.vec3),
    point_sum: wp.array(dtype=wp.vec3),
    out_covariance: wp.array(dtype=wp.mat33),
):
    # Scatter matrix about the centroid. Left unnormalized: scaling by 1/n does
    # not change the eigenvectors, which is all the caller wants.
    i = wp.tid()
    centroid = point_sum[0] / float(points.shape[0])
    d = points[i] - centroid
    wp.atomic_add(out_covariance, 0, wp.outer(d, d))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_pca_kernel(
    covariance: wp.array(dtype=wp.mat33),
    slot: int,
    rotations: wp.array(dtype=wp.quat),
):
    # The principal axes of the point set are the eigenvectors of its covariance
    # matrix, which is a good starting guess for elongated shapes: the spiral
    # resolves orientation only to its sample spacing, and a few degrees of error
    # costs a lot of volume when one axis is much longer than the others.
    # Eigenvalues are unused: the box is the same whichever order the axes come in.
    Q, _eigenvalues = wp.eig3(covariance[0])

    # ``eig3`` returns orthonormal columns but does not promise a right-handed
    # frame, and ``quat_from_matrix`` is only defined for a pure rotation.
    if wp.determinant(Q) < 0.0:
        Q = wp.mat33(
            -Q[0, 0], Q[0, 1], Q[0, 2],
            -Q[1, 0], Q[1, 1], Q[1, 2],
            -Q[2, 0], Q[2, 1], Q[2, 2],
        )  # fmt: skip

    # Q's columns are the principal axes in world space, so Q maps box-local to
    # world; the search below wants the world-to-box-local direction.
    rotations[slot] = wp.quat_inverse(wp.quat_from_matrix(Q))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_bounds_kernel(
    points: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    num_chunks: int,
    min_bounds: wp.array(dtype=wp.vec3),
    max_bounds: wp.array(dtype=wp.vec3),
):
    # Parallel over (orientation, point chunk). Parallelizing over orientations
    # alone caps the launch at one thread per candidate no matter how many points
    # there are, which leaves most of the GPU idle; splitting the point loop as
    # well makes the available parallelism scale with the point count.
    candidate, chunk = wp.tid()

    rot = rotations[candidate]

    # Axis-aligned bounds of this thread's slice of points in the rotated frame.
    # A grid-stride slice keeps neighboring threads on neighboring points, so the
    # reads coalesce.
    lo = wp.vec3(wp.inf, wp.inf, wp.inf)
    hi = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

    num_points = points.shape[0]
    for j in range(chunk, num_points, num_chunks):
        rotated = wp.quat_rotate(rot, points[j])
        lo = wp.min(lo, rotated)
        hi = wp.max(hi, rotated)

    # Combine the per-chunk bounds. Unlike a sum, min/max are exact in floating
    # point, so this reduction is deterministic regardless of the order in which
    # the atomics land.
    wp.atomic_min(min_bounds, candidate, lo)
    wp.atomic_max(max_bounds, candidate, hi)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_measure_kernel(
    rotations: wp.array(dtype=wp.quat),
    min_bounds: wp.array(dtype=wp.vec3),
    max_bounds: wp.array(dtype=wp.vec3),
    measure_type: int,
    measures: wp.array(dtype=wp.float32),
    transforms: wp.array(dtype=wp.transform),
    extents: wp.array(dtype=wp.vec3),
):
    # One thread per candidate orientation, scoring the box found above.
    i = wp.tid()

    rot = rotations[i]
    lo = min_bounds[i]
    hi = max_bounds[i]

    # Full side lengths of the box and the measure being minimized. Default to the
    # volume; only the surface-area branch overrides it (keeps `measure` defined on
    # every path for codegen).
    dims = hi - lo
    measure = dims[0] * dims[1] * dims[2]
    if measure_type == wp.static(int(OBBMeasureType.SURFACE_AREA)):
        measure = 2.0 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2])

    # Box center in the rotated frame, mapped back into world space. The stored
    # transform takes the box's local axis-aligned frame to world coordinates.
    center = (hi + lo) * 0.5
    world_center = wp.quat_rotate_inv(rot, center)

    measures[i] = measure
    extents[i] = dims
    transforms[i] = wp.transform(world_center, wp.quat_inverse(rot))


def oriented_bounding_box(
    points: wp.array(dtype=wp.vec3),
    measure_type: OBBMeasureType = OBBMeasureType.VOLUME,
    num_samples: int = 4096,
    *,
    include_axis_aligned: bool = True,
    include_pca: bool = True,
    device: DeviceLike | None = None,
) -> tuple[wp.transform, wp.vec3, float]:
    """Approximate an oriented bounding box (OBB) of a point set by sampling orientations.

    Candidate orientations are drawn from a Super-Fibonacci spiral, which spreads
    ``num_samples`` rotations near-uniformly over SO(3) (Alexa, "Super-Fibonacci
    Spirals: Fast, Low-Discrepancy Sampling of SO(3)", CVPR 2022). For each
    orientation the axis-aligned bounding box of the rotated points is evaluated, and
    the orientation whose box minimizes ``measure_type`` is returned. The result is an
    approximation whose quality improves with ``num_samples``; it is not guaranteed to
    be the globally optimal OBB.

    Two further candidates are appended by default. ``include_axis_aligned`` adds the
    identity rotation, which the spiral never contains exactly and which guarantees the
    result is no worse than the axis-aligned bounding box. ``include_pca`` adds the
    principal axes of the point set, obtained from the eigenvectors of its covariance
    matrix. The spiral resolves orientation only to roughly its sample spacing, and on
    a strongly elongated shape a few degrees of error costs a large amount of volume,
    so the principal axes are often a much better answer there than any sampled
    rotation. For a near-isotropic point set the covariance eigenvectors are close to
    degenerate and that candidate carries little information, leaving accuracy governed
    by ``num_samples`` alone. Both extras are cheap relative to ``num_samples``
    candidates.

    Args:
        points: Array of point positions of type :class:`warp.vec3`.
        measure_type: Quantity to minimize, as an :class:`OBBMeasureType` member:
            :attr:`OBBMeasureType.VOLUME` minimizes the box volume;
            :attr:`OBBMeasureType.SURFACE_AREA` minimizes the box surface area.
        num_samples: Number of candidate orientations to sample from the spiral. Larger
            values give a tighter box at higher cost.
        include_axis_aligned: Whether to also evaluate the identity rotation.
        include_pca: Whether to also evaluate the principal axes of ``points``.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(transform, extents, measure)`` where ``transform`` is a
        :class:`warp.transform` mapping the box's local frame (axis-aligned and
        centered at the origin) into world space, ``extents`` is a :class:`warp.vec3`
        of the box's full side lengths, and ``measure`` is the achieved value of
        ``measure_type``.

    Note:
        This function synchronizes with the device to select the winning
        orientation, so it cannot be captured in a CUDA graph.

        With ``include_pca`` enabled the covariance matrix is accumulated with
        floating-point atomics, so the principal axes -- and therefore the result,
        when they win -- can vary in the last bits between runs. Pass
        ``include_pca=False`` for a bitwise reproducible result.

    Raises:
        ValueError: If ``num_samples`` is not a positive integer, or if ``points``
            is empty.
    """
    if num_samples < 1:
        raise ValueError(f"`num_samples` must be a positive integer, but got {num_samples}.")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("`points` must contain at least one point, but got an empty array.")

    measure_code = int(OBBMeasureType(measure_type))
    device = wp.get_device(device) if device is not None else points.device

    # Candidate orientations: the spiral first, then any extras in the trailing slots.
    num_candidates = num_samples + int(include_axis_aligned) + int(include_pca)
    rotations = wp.empty(num_candidates, dtype=wp.quat, device=device)

    wp.launch(
        oriented_bounding_box_samples_kernel,
        dim=num_samples,
        inputs=[num_samples],
        outputs=[rotations],
        device=device,
    )

    slot = num_samples
    if include_axis_aligned:
        wp.launch(oriented_bounding_box_identity_kernel, dim=1, inputs=[slot], outputs=[rotations], device=device)
        slot += 1

    if include_pca:
        point_sum = wp.zeros(1, dtype=wp.vec3, device=device)
        covariance = wp.zeros(1, dtype=wp.mat33, device=device)
        wp.launch(point_sum_kernel, dim=num_points, inputs=[points], outputs=[point_sum], device=device)
        wp.launch(
            point_covariance_kernel, dim=num_points, inputs=[points, point_sum], outputs=[covariance], device=device
        )
        wp.launch(
            oriented_bounding_box_pca_kernel, dim=1, inputs=[covariance, slot], outputs=[rotations], device=device
        )
        slot += 1

    # Threads per orientation. The candidate count alone is far too little parallelism
    # to fill a GPU, so the point loop is split as well; the cap keeps the atomic
    # contention on each orientation's bounds low.
    num_chunks = max(1, min(num_points, _OBB_POINT_CHUNKS))

    # Seeded to an empty interval so the atomic min/max reduction can only shrink it.
    min_bounds = wp.full(num_candidates, wp.vec3(math.inf, math.inf, math.inf), dtype=wp.vec3, device=device)
    max_bounds = wp.full(num_candidates, wp.vec3(-math.inf, -math.inf, -math.inf), dtype=wp.vec3, device=device)

    wp.launch(
        oriented_bounding_box_bounds_kernel,
        dim=(num_candidates, num_chunks),
        inputs=[points, rotations, num_chunks],
        outputs=[min_bounds, max_bounds],
        device=device,
    )

    # Per-orientation results; the search is not differentiable, so no grads are needed.
    measures = wp.empty(num_candidates, dtype=wp.float32, device=device)
    transforms = wp.empty(num_candidates, dtype=wp.transform, device=device)
    extents = wp.empty(num_candidates, dtype=wp.vec3, device=device)

    wp.launch(
        oriented_bounding_box_measure_kernel,
        dim=num_candidates,
        inputs=[rotations, min_bounds, max_bounds, measure_code],
        outputs=[measures, transforms, extents],
        device=device,
    )

    # Select the sampled orientation whose box minimizes the measure. The candidate
    # count is small, so the argmin runs on the host.
    measures_np = measures.numpy()
    best = int(np.argmin(measures_np))

    row = transforms.numpy()[best]
    best_transform = wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )
    ext = extents.numpy()[best]
    best_extents = wp.vec3(float(ext[0]), float(ext[1]), float(ext[2]))
    best_measure = float(measures_np[best])

    return best_transform, best_extents, best_measure
