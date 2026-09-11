# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

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


_SF_PLASTIC = wp.constant(wp.float64(0.7548776662466927))
"""Reciprocal of the plastic number, a low-discrepancy step for the refinement angles."""

_GOLDEN_ANGLE = wp.constant(wp.float32(2.399963229728653))
"""Golden angle in radians, spacing successive refinement axes on a Fibonacci sphere."""


@wp.func
def small_rotation(i: int, n: int, radius: wp.float32) -> wp.quat:
    # A deterministic small rotation used to probe orientations near a current best.
    # The axis is the i-th point of a Fibonacci sphere (near-uniform directions) and the
    # signed angle is a low-discrepancy value in ``[-radius, radius]``; together the batch
    # samples a shrinking spherical neighborhood of rotations without any host randomness.
    u = (wp.float32(i) + 0.5) / wp.float32(n)
    z = 1.0 - 2.0 * u
    r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    phi = _GOLDEN_ANGLE * wp.float32(i)
    axis = wp.vec3(r * wp.cos(phi), r * wp.sin(phi), z)

    t = wp.float64(i) * _SF_PLASTIC
    frac = wp.float32(t - wp.floor(t))
    angle = radius * (2.0 * frac - 1.0)

    half = 0.5 * angle
    s = wp.sin(half)
    return wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(half))


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

_OBB_REFINE_RADIUS0 = 0.35
"""Initial angular radius (radians) of the local search around the best orientation."""

_OBB_REFINE_SHRINK = 0.72
"""Factor by which the refine radius shrinks each round, focusing the local search."""

_OBB_REFINE_BATCH = 128
"""Default number of orientations probed per refinement round."""

_OBB_MAX_SEARCH_POINTS = 100_000
"""Default cap on the points used to *score* orientations; the winner is re-measured exactly."""


@wp.kernel(enable_backward=False)
def oriented_bounding_box_samples_kernel(num_samples: int, rotations: wp.array[wp.quat]):
    # Fill the spiral portion of the candidate list. Any extra candidates are
    # written into the slots past ``num_samples`` by the kernels below.
    i = wp.tid()
    rotations[i] = super_fibonacci(i, num_samples)


@wp.kernel(enable_backward=False)
def oriented_bounding_box_identity_kernel(slot: int, rotations: wp.array[wp.quat]):
    # The spiral never contains the identity exactly, so the axis-aligned box is
    # not otherwise among the candidates.
    rotations[slot] = wp.quat_identity()


@wp.kernel(enable_backward=False)
def point_sum_kernel(points: wp.array[wp.vec3], out_sum: wp.array[wp.vec3]):
    i = wp.tid()
    wp.atomic_add(out_sum, 0, points[i])


@wp.kernel(enable_backward=False)
def point_covariance_kernel(
    points: wp.array[wp.vec3],
    point_sum: wp.array[wp.vec3],
    out_covariance: wp.array[wp.mat33],
):
    # Scatter matrix about the centroid. Left unnormalized: scaling by 1/n does
    # not change the eigenvectors, which is all the caller wants.
    i = wp.tid()
    centroid = point_sum[0] / float(points.shape[0])
    d = points[i] - centroid
    wp.atomic_add(out_covariance, 0, wp.outer(d, d))


@wp.kernel(enable_backward=False)
def oriented_bounding_box_pca_kernel(
    covariance: wp.array[wp.mat33],
    slot: int,
    rotations: wp.array[wp.quat],
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
    points: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    num_chunks: int,
    min_bounds: wp.array[wp.vec3],
    max_bounds: wp.array[wp.vec3],
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
    rotations: wp.array[wp.quat],
    min_bounds: wp.array[wp.vec3],
    max_bounds: wp.array[wp.vec3],
    measure_type: int,
    measures: wp.array[wp.float32],
    transforms: wp.array[wp.transform],
    extents: wp.array[wp.vec3],
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


@wp.kernel(enable_backward=False)
def oriented_bounding_box_subsample_kernel(
    points: wp.array[wp.vec3],
    stride: int,
    out_points: wp.array[wp.vec3],
):
    # Gather every ``stride``-th point. The orientation that minimizes the box is a
    # property of the cloud's shape, which a strided subsample captures; scoring the
    # candidates on it instead of every point makes the search cost independent of the
    # (possibly millions of) input points. The winner's bounds are later recomputed
    # exactly over the full set, so the returned box is unchanged.
    i = wp.tid()
    out_points[i] = points[i * stride]


@wp.kernel(enable_backward=False)
def oriented_bounding_box_refine_generate_kernel(
    champion: wp.array[wp.quat],
    radius: float,
    batch: int,
    rot_batch: wp.array[wp.quat],
):
    # Each thread proposes one orientation in a shrinking neighborhood of the current
    # champion, so the batch performs a parallel local search around it.
    i = wp.tid()
    rot_batch[i] = wp.normalize(small_rotation(i, batch, radius) * champion[0])


@wp.kernel(enable_backward=False)
def oriented_bounding_box_champion_kernel(
    rotations: wp.array[wp.quat],
    measures: wp.array[wp.float32],
    champion: wp.array[wp.quat],
    champion_measure: wp.array[wp.float32],
):
    # Single-threaded argmin over a scored batch, committing its best to the champion if
    # it improves on it. Kept on device (and single-threaded, so the champion quaternion is
    # never torn between competing threads) so the whole search stays capturable in a CUDA
    # graph. A strict ``<`` resolves ties to the lowest index, matching a host ``argmin``;
    # seeding ``champion_measure`` to infinity makes the first call adopt the batch's best.
    best = int(0)
    best_measure = measures[0]
    for i in range(1, measures.shape[0]):
        if measures[i] < best_measure:
            best_measure = measures[i]
            best = i

    if best_measure < champion_measure[0]:
        champion_measure[0] = best_measure
        champion[0] = rotations[best]


def oriented_bounding_box(
    points: wp.array[wp.vec3],
    measure_type: OBBMeasureType = OBBMeasureType.VOLUME,
    num_samples: int = 4096,
    *,
    include_axis_aligned: bool = True,
    include_pca: bool = True,
    max_search_points: int | None = _OBB_MAX_SEARCH_POINTS,
    refine_iters: int = 4,
    refine_batch: int = _OBB_REFINE_BATCH,
    device: DeviceLike | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
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

    After the initial search the best orientation is refined by ``refine_iters`` rounds of
    coarse-to-fine local search: each round scores a batch of orientations in a shrinking
    angular neighborhood of the current best and keeps any improvement. Because the spiral
    resolves orientation only to its sample spacing, this reaches the quality of a much
    larger ``num_samples`` at a fraction of the cost. For a large point set the orientation
    search runs over a strided subsample of at most ``max_search_points`` points -- the best
    orientation is a property of the cloud's shape, which a subsample captures -- while the
    returned box is always re-measured over every point, so it stays exact.

    Args:
        points: Array of point positions of type :class:`warp.vec3`.
        measure_type: Quantity to minimize, as an :class:`OBBMeasureType` member:
            :attr:`OBBMeasureType.VOLUME` minimizes the box volume;
            :attr:`OBBMeasureType.SURFACE_AREA` minimizes the box surface area.
        num_samples: Number of candidate orientations to sample from the spiral. Larger
            values give a tighter box at higher cost.
        include_axis_aligned: Whether to also evaluate the identity rotation.
        include_pca: Whether to also evaluate the principal axes of ``points``.
        max_search_points: Upper bound on how many points are used to *score* candidate
            orientations. When ``points`` is larger, candidates are scored over a
            deterministic strided subsample of about this size; the winning orientation's
            box is then recomputed over the full point set, so the returned box is exact.
            Pass ``None`` to always score over every point.
        refine_iters: Number of coarse-to-fine refinement rounds after the initial search.
            Pass ``0`` to return the best sampled orientation directly.
        refine_batch: Number of orientations probed per refinement round.
        device: Device on which to run. Defaults to the device of ``points``.

    Returns:
        A tuple ``(transform, extents, measure)`` of length-1 device arrays, left on
        the device so the whole search is capturable in a CUDA graph. ``transform`` is
        a :class:`warp.array` of :class:`warp.transform` mapping the box's local frame
        (axis-aligned and centered at the origin) into world space, ``extents`` is a
        :class:`warp.array` of :class:`warp.vec3` of the box's full side lengths, and
        ``measure`` is a :class:`warp.array` of ``float32`` holding the achieved value
        of ``measure_type``. Read an element with ``transform.numpy()[0]`` (which
        synchronizes) when a host-side value is needed.

    Note:
        The search, refinement, and winner selection all run on the device, so this
        function does not synchronize and can be captured in a CUDA graph. Refinement is
        deterministic (its neighborhood is generated without host randomness).

        With ``include_pca`` enabled the covariance matrix is accumulated with
        floating-point atomics, so the principal axes -- and therefore the result,
        when they win -- can vary in the last bits between runs. Pass
        ``include_pca=False`` for a bitwise reproducible result.

    Raises:
        ValueError: If ``num_samples`` or ``refine_batch`` is not a positive integer, if
            ``refine_iters`` is negative, if ``max_search_points`` is not positive, or if
            ``points`` is empty.
    """
    if num_samples < 1:
        raise ValueError(f"`num_samples` must be a positive integer, but got {num_samples}.")
    if refine_iters < 0:
        raise ValueError(f"`refine_iters` must be a non-negative integer, but got {refine_iters}.")
    if refine_batch < 1:
        raise ValueError(f"`refine_batch` must be a positive integer, but got {refine_batch}.")
    if max_search_points is not None and max_search_points < 1:
        raise ValueError(f"`max_search_points` must be a positive integer or None, but got {max_search_points}.")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("`points` must contain at least one point, but got an empty array.")

    measure_code = int(OBBMeasureType(measure_type))
    device = wp.get_device(device) if device is not None else points.device

    # Points used to *score* orientations. For a large cloud a strided subsample is enough
    # to choose the orientation -- it is a property of the shape, not of every point -- and
    # it makes the search cost independent of the point count. The winner's box is recomputed
    # over the full set at the end, so the returned box is exact.
    if max_search_points is not None and num_points > max_search_points:
        stride = (num_points + max_search_points - 1) // max_search_points
        num_search = (num_points + stride - 1) // stride
        search_points = wp.empty(num_search, dtype=wp.vec3, device=device)
        wp.launch(
            oriented_bounding_box_subsample_kernel,
            dim=num_search,
            inputs=[points, stride],
            outputs=[search_points],
            device=device,
        )
    else:
        search_points = points
        num_search = num_points

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
        wp.launch(point_sum_kernel, dim=num_search, inputs=[search_points], outputs=[point_sum], device=device)
        wp.launch(
            point_covariance_kernel,
            dim=num_search,
            inputs=[search_points, point_sum],
            outputs=[covariance],
            device=device,
        )
        wp.launch(
            oriented_bounding_box_pca_kernel, dim=1, inputs=[covariance, slot], outputs=[rotations], device=device
        )
        slot += 1

    # Threads per orientation. The candidate count alone is far too little parallelism
    # to fill a GPU, so the point loop is split as well; the cap keeps the atomic
    # contention on each orientation's bounds low.
    num_chunks = max(1, min(num_search, _OBB_POINT_CHUNKS))

    measures, _, _ = _score_orientations(search_points, rotations, num_candidates, num_chunks, measure_code, device)

    # Champion: the best orientation found so far, tracked on device so the whole search
    # stays capturable in a CUDA graph. Seeded to +inf so the initial candidates are adopted.
    champion = wp.empty(1, dtype=wp.quat, device=device)
    champion_measure = wp.full(1, wp.float32(math.inf), dtype=wp.float32, device=device)
    wp.launch(
        oriented_bounding_box_champion_kernel,
        dim=1,
        inputs=[rotations, measures],
        outputs=[champion, champion_measure],
        device=device,
    )

    # Coarse-to-fine refinement: each round scores a batch of orientations in a shrinking
    # neighborhood of the champion and keeps any improvement. A fixed number of rounds of
    # fixed-size launches keeps the whole thing capturable.
    if refine_iters > 0:
        rot_batch = wp.empty(refine_batch, dtype=wp.quat, device=device)
        batch_chunks = max(1, min(num_search, _OBB_POINT_CHUNKS))
        radius = _OBB_REFINE_RADIUS0
        for _ in range(refine_iters):
            wp.launch(
                oriented_bounding_box_refine_generate_kernel,
                dim=refine_batch,
                inputs=[champion, radius, refine_batch],
                outputs=[rot_batch],
                device=device,
            )
            batch_measures, _, _ = _score_orientations(
                search_points, rot_batch, refine_batch, batch_chunks, measure_code, device
            )
            wp.launch(
                oriented_bounding_box_champion_kernel,
                dim=1,
                inputs=[rot_batch, batch_measures],
                outputs=[champion, champion_measure],
                device=device,
            )
            radius *= _OBB_REFINE_SHRINK

    # Re-measure the winning orientation over the full point set. This makes the returned box
    # exact when the search used a subsample or a refined orientation, and unifies every path.
    out_measure, out_transform, out_extents = _score_orientations(
        points, champion, 1, max(1, min(num_points, _OBB_POINT_CHUNKS)), measure_code, device
    )

    return out_transform, out_extents, out_measure


def _score_orientations(points, rotations, count, num_chunks, measure_code, device):
    """Score ``count`` orientations over ``points``: their box bounds, measure, transform, extents.

    Returns ``(measures, transforms, extents)`` device arrays of length ``count``.
    """
    min_bounds = wp.full(count, wp.vec3(math.inf, math.inf, math.inf), dtype=wp.vec3, device=device)
    max_bounds = wp.full(count, wp.vec3(-math.inf, -math.inf, -math.inf), dtype=wp.vec3, device=device)
    wp.launch(
        oriented_bounding_box_bounds_kernel,
        dim=(count, num_chunks),
        inputs=[points, rotations, num_chunks],
        outputs=[min_bounds, max_bounds],
        device=device,
    )

    measures = wp.empty(count, dtype=wp.float32, device=device)
    transforms = wp.empty(count, dtype=wp.transform, device=device)
    extents = wp.empty(count, dtype=wp.vec3, device=device)
    wp.launch(
        oriented_bounding_box_measure_kernel,
        dim=count,
        inputs=[rotations, min_bounds, max_bounds, measure_code],
        outputs=[measures, transforms, extents],
        device=device,
    )
    return measures, transforms, extents
