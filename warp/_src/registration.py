# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import warp as wp

if TYPE_CHECKING:
    from warp._src.context import DeviceLike

##########################################################################
## Device functions and structs (reusable within kernels)
##
## Building blocks for iterative closest point (ICP): a closest-point query
## that yields a target point and its normal, and the linearized point-to-plane
## Gauss-Newton contribution of one correspondence.
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
def _accumulate_normal_equations(
    term: GaussNewtonTerm,
    weight: wp.float32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
):
    # Scatter ``weight * J J^T`` (upper triangle, 21 entries) and ``weight * b * J``
    # (6 entries) into the shared normal-equation accumulators.
    j = term.jacobian
    wb = weight * term.b
    for i in range(6):
        wp.atomic_add(g, i, wb * j[i])
        base = i * 6 - (i * (i - 1)) / 2
        for k in range(i, 6):
            wp.atomic_add(a_upper, base + (k - i), weight * j[i] * j[k])


@wp.kernel(enable_backward=False)
def _icp_accumulate_mesh_kernel(
    source: wp.array(dtype=wp.vec3),
    rot: wp.mat33,
    trans: wp.vec3,
    mesh: wp.uint64,
    max_dist: wp.float32,
    a_upper: wp.array(dtype=wp.float32),
    g: wp.array(dtype=wp.float32),
    stats: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = rot * source[tid] + trans
    cp = closest_on_mesh(mesh, p, max_dist)
    if cp.valid == 0:
        return
    term = point_plane_term(p, cp.point, cp.normal)
    _accumulate_normal_equations(term, 1.0, a_upper, g)
    r = wp.dot(p - cp.point, cp.normal)
    wp.atomic_add(stats, 0, r * r)
    wp.atomic_add(stats, 1, 1.0)


##########################################################################
## Host-side driver
##########################################################################

_UPPER_INDEX = np.array(
    [[min(i, k) * 6 - (min(i, k) * (min(i, k) - 1)) // 2 + abs(i - k) for k in range(6)] for i in range(6)]
)


def _sym6(a_upper: np.ndarray) -> np.ndarray:
    """Expand 21 packed upper-triangle entries into a symmetric 6x6 matrix."""
    return a_upper[_UPPER_INDEX]


def _so3_exp(a: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector via Rodrigues' formula."""
    theta = float(np.linalg.norm(a))
    if theta < 1e-12:
        return np.eye(3)
    k = a / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


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


def _as_vec3_array(points, device) -> wp.array:
    if isinstance(points, wp.array):
        return points.to(device) if points.device != device else points
    return wp.array(np.asarray(points, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)


def _build_target_mesh(target, device) -> wp.Mesh:
    if isinstance(target, wp.Mesh):
        return target
    points, faces = target
    return wp.Mesh(
        points=_as_vec3_array(points, device),
        indices=wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
    )


def register_rigid(
    source,
    target,
    *,
    init=None,
    max_iters: int = 50,
    tol: float = 1e-6,
    max_corr_dist: float | None = None,
    damping: float = 1e-9,
    device: DeviceLike | None = None,
) -> RegistrationResult:
    """Rigidly align a source point set to a target surface with point-to-plane ICP.

    Each iteration transforms the source by the current estimate, finds the
    closest point (and normal) on the *fixed* target, and takes a Gauss-Newton
    step of the linearized point-to-plane objective. Because the motion is rigid,
    the target's BVH is built once and never rebuilt.

    Args:
        source: Source points, a :class:`warp.array` of :class:`warp.vec3` or an
            array-like of shape ``(num_points, 3)``.
        target: Target mesh, given as a :class:`warp.Mesh` or a ``(points, faces)``
            pair.
        init: Optional ``(4, 4)`` initial transform (defaults to identity).
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance on the 6-DOF update norm.
        max_corr_dist: Reject correspondences farther than this. Defaults to no
            bound.
        damping: Levenberg-style diagonal damping added to the 6x6 system.
        device: Device on which to run. Defaults to the device of ``source``.

    Returns:
        A :class:`RegistrationResult`.
    """
    device = (
        wp.get_device(device)
        if device is not None
        else (source.device if isinstance(source, wp.array) else wp.get_device())
    )
    device = wp.get_device(device)

    src = _as_vec3_array(source, device)
    n = src.shape[0]
    mesh = _build_target_mesh(target, device)
    max_dist = float(max_corr_dist) if max_corr_dist is not None else 1.0e30

    init = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)
    rot = init[:3, :3].copy()
    trans = init[:3, 3].copy()

    a_upper = wp.zeros(21, dtype=wp.float32, device=device)
    g = wp.zeros(6, dtype=wp.float32, device=device)
    stats = wp.zeros(2, dtype=wp.float32, device=device)

    rmse = float("inf")
    converged = False
    iterations = 0
    for step in range(max_iters):
        iterations = step + 1
        a_upper.zero_()
        g.zero_()
        stats.zero_()
        wp.launch(
            _icp_accumulate_mesh_kernel,
            dim=n,
            inputs=[src, wp.mat33(rot.astype(np.float32)), wp.vec3(*trans.astype(np.float32)), mesh.id, max_dist],
            outputs=[a_upper, g, stats],
            device=device,
        )
        a_np = a_upper.numpy().astype(np.float64)
        g_np = g.numpy().astype(np.float64)
        st = stats.numpy()
        count = float(st[1])
        rmse = float(np.sqrt(st[0] / count)) if count > 0 else float("inf")

        system = _sym6(a_np) + damping * np.eye(6)
        x = np.linalg.solve(system, g_np)
        d_rot = _so3_exp(x[:3])
        rot = d_rot @ rot
        trans = d_rot @ trans + x[3:]

        if float(np.linalg.norm(x)) < tol:
            converged = True
            break

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return RegistrationResult(transform=transform, rmse=rmse, iterations=iterations, converged=converged)
