# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Hessian
#
# Computes the value, gradient, and Hessian of the point-mesh signed
# distance at points sampled in a mesh's bounding box, using second-order
# forward-mode jets (wp.JetSpace2).
#
# The mesh query cannot be pushed through jet arithmetic -- BVH traversal and
# closest-feature selection are branchy and non-smooth. So instead of writing
# the signed distance in jet operators, we call the ordinary mesh query in
# plain floats, compute the analytic first and second derivatives of the
# signed distance in closed form (geometry space, vec3/mat33), and inject them
# into a seeded point jet with a small custom chain-rule rule (``lift_seed_vec3``
# below). Reverse mode and tapes are not involved: one forward pass of the
# second-order jet carries value, gradient, and the full 3x3 Hessian.
#
# This is the "oracular" custom-derivative pattern: extract/reassemble handles
# anything expressible in jet arithmetic, but a black-box query like the mesh
# SDF needs its analytic derivatives supplied by hand.
###########################################################################

import argparse
import os

import numpy as np

import warp as wp
import warp.examples

# Width 3: differentiate the scalar signed distance with respect to the three
# coordinates of the query point. A width-3 second-order jet therefore carries a
# length-3 gradient and a 3x3 Hessian.
WIDTH = 3
J2 = wp.JetSpace2(WIDTH)

# Tolerance for calling a returned barycentric coordinate "zero", used to
# classify the closest feature (face/edge/vertex) after the fact from the
# ``(u, v)`` the mesh query returns.
#
# TAU_REPR repairs only the floating-point residue of an *exact* edge hit. Under
# strict IEEE binary32 the native closest-point routine (see
# ``closest_point_to_triangle`` in warp/native/intersect.h) sets one coordinate
# to zero on an edge, but packs the result as two floats and lets the third be
# reconstructed by subtraction; that reconstruction can leave a residue of up to
# 2^-25 = 1/4 FLT_EPSILON. Measured against the actual CPU and CUDA builds, the
# worst-case edge residue is exactly 2^-25, so the comparison must be inclusive
# (``<=``). This is a representation-repair constant, not a geometric tolerance:
# a face point whose true third weight is within 2^-25 of zero is deliberately
# classified as an edge (a face can lie arbitrarily close to an edge, so every
# positive tolerance absorbs a thin face collar into the edge case).
#
# TAU_GEOM is an optional, application-chosen geometric collar. Leave it at 0 for
# the narrowest post-facto convention; raise it to intentionally treat a band of
# near-edge faces as edges (collar behavior like the native sign query).
TAU_REPR = float.fromhex("0x1p-25")  # 2.9802322e-08
TAU_GEOM = 0.0
TAU = max(TAU_REPR, TAU_GEOM)


@wp.func
def feature_tangent_projector(mesh: wp.uint64, face: int, u: float, v: float, n: wp.vec3) -> wp.mat33:
    """Projector ``T`` onto the directions the closest point is free to slide along.

    ==================  ================================  ============================
    Closest feature     Projector ``T``                   Resulting Hessian
    ==================  ================================  ============================
    face interior       ``I - n nᵀ``  (2 dims)            ``0`` (a plane is flat)
    edge                ``t tᵀ``      (1 dim, edge ``t``)  rank 1
    vertex              ``0``         (0 dims)             ``(s/dist)(I - n nᵀ)``, rank 2
    ==================  ================================  ============================

    The closest feature is recovered from the barycentric weights the mesh query
    returns. With ``w = 1 - u - v``, the weights map to the triangle vertices as
    ``u -> vertex 0, v -> vertex 1, w -> vertex 2`` (matching
    :func:`warp.mesh_eval_position`). Counting weights within :data:`TAU` of zero
    gives the feature: two zeros -> vertex, one zero -> edge (spanned by the two
    non-zero-weight vertices), none -> face interior. See :data:`TAU_REPR` for why
    that tolerance is what it is.

    ``.. note::`` **This recovers the triangle feature, not the surface feature.**
    The per-simplex curvature is exact only in the full-dimensional part of a
    feature's normal cone, and discrete curvature measures that cone's size:
    ``pi - dihedral`` for an edge, the angle defect ``2*pi - sum(theta)`` for a
    vertex. A flat feature has a collapsed cone, so the formula is spurious there:

    * a coplanar internal edge (dihedral ~ pi, e.g. a face diagonal) is read as an
      edge and given rank-1 curvature where the surface is flat and the Hessian is
      really zero;
    * a zero-defect vertex (a flat fan, or a subdivision point on a straight
      crease) is read as a vertex and given rank-2 curvature where the true feature
      is a face (zero) or an edge (rank 1).

    Fixing this needs adjacency / discrete curvature to demote flat features, which
    a single query does not return. But it only bites on the measure-zero set of
    points whose closest point lands exactly on such a feature -- probability zero
    for generic sampling -- so this example leaves it unhandled.
    """
    w = 1.0 - u - v
    zu = wp.abs(u) <= wp.static(TAU)
    zv = wp.abs(v) <= wp.static(TAU)
    zw = wp.abs(w) <= wp.static(TAU)

    # Two near-zero weights: closest point is a vertex, fixed as the query moves.
    if (zv and zw) or (zu and zw) or (zu and zv):
        return wp.mat33(0.0)

    # One near-zero weight: closest point slides along the edge of the other two
    # vertices. T removes that tangent direction from the curvature.
    if zu or zv or zw:
        m = wp.mesh_get(mesh)
        p0 = m.points[m.indices[face * 3 + 0]]
        p1 = m.points[m.indices[face * 3 + 1]]
        p2 = m.points[m.indices[face * 3 + 2]]
        if zw:
            t = wp.normalize(p1 - p0)  # weight of vertex 2 vanished -> edge (0, 1)
        elif zv:
            t = wp.normalize(p2 - p0)  # weight of vertex 1 vanished -> edge (0, 2)
        else:
            t = wp.normalize(p2 - p1)  # weight of vertex 0 vanished -> edge (1, 2)
        return wp.outer(t, t)

    # Face interior: the two free tangent directions are the whole tangent plane.
    return wp.identity(n=3, dtype=float) - wp.outer(n, n)


@wp.func
def signed_distance_derivs(mesh: wp.uint64, p: wp.vec3, max_dist: float):
    """Signed distance and its analytic gradient/Hessian at ``p`` (plain floats).

    Returns ``(hit, value, grad, hess)`` where ``hit`` is 0 when no surface lies
    within ``max_dist``. ``grad`` is the outward unit normal; ``hess`` is
    ``(s/dist)(I - n nᵀ - T)`` with ``T`` from :func:`feature_tangent_projector`.
    """
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    if not q.result:
        return 0, 0.0, wp.vec3(), wp.mat33()

    c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    r = p - c
    dist = wp.length(r)
    n = r / dist  # unit direction from the surface toward the query point
    s = q.sign  # +1 outside, -1 inside

    value = s * dist  # signed distance
    grad = s * n  # gradient of the signed distance is the outward unit normal

    tangent = feature_tangent_projector(mesh, q.face, q.u, q.v, n)
    normal_proj = wp.identity(n=3, dtype=float) - wp.outer(n, n)
    hess = (s / dist) * (normal_proj - tangent)
    return 1, value, grad, hess


@wp.func
def lift_seed_vec3(value: float, grad: wp.vec3, hess: wp.mat33, p: J2.vec3) -> J2.scalar:
    """Inject a scalar field's analytic derivatives onto a seeded point jet.

    For a scalar field ``d(x)`` with ``∇d = grad`` and ``∇²d = hess`` evaluated at
    a point jet ``p``, the second-order chain rule is

        value_out = d
        grad_out  = Jᵀ ∇d
        hess_out  = Jᵀ (∇²d) J  +  Σ_c (∇d)_c · (∂²p_c)

    where ``J = p.grad`` is the 3xwidth Jacobian of the point with respect to the
    seed directions. ``p`` here comes from :func:`seed_vec3`, so it is a *seed*:
    ``p.hess`` is zero and the final term drops. (A composed, non-seed ``p`` --
    e.g. a point that is itself a jet function of some parameters -- would need
    that term; it is exactly what a general ``J2.lift`` helper would add.)
    """
    grad_out = wp.transpose(p.grad) @ grad  # Jᵀ ∇d
    hess_out = wp.transpose(p.grad) @ (hess @ p.grad)  # Jᵀ (∇²d) J
    return J2.scalar(value, grad_out, hess_out)


@wp.kernel(enable_backward=False)
def sdf_hessian_kernel(
    points: wp.array[wp.vec3],
    mesh: wp.uint64,
    max_dist: float,
    valid: wp.array[wp.int32],
    value: wp.array[float],
    grad: wp.array[J2.grad],
    hess: wp.array[J2.hess],
):
    i = wp.tid()
    hit, val, g, H = signed_distance_derivs(mesh, points[i], max_dist)
    valid[i] = hit
    if hit == 0:
        return

    # Seed the three coordinates of the query point as directions 0, 1, 2.
    p = J2.seed_vec3(points[i], 0, 1, 2)
    out = lift_seed_vec3(val, g, H, p)
    value[i] = out.value
    grad[i] = out.grad
    hess[i] = out.hess


def load_bunny(scale: float = 10.0):
    """Load the bundled bunny mesh as ``(points, indices)`` NumPy arrays."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32) * scale
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return points, indices


class Example:
    def __init__(self, num_samples: int = 100000, seed: int = 42, max_dist: float = 1.0e6):
        self.max_dist = max_dist

        points_np, indices_np = load_bunny()
        self.mesh = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3),
            indices=wp.array(indices_np, dtype=wp.int32),
        )

        # Uniformly sample the mesh's axis-aligned bounding box.
        lo = points_np.min(axis=0)
        hi = points_np.max(axis=0)
        rng = np.random.default_rng(seed)
        samples = rng.uniform(lo, hi, size=(num_samples, 3)).astype(np.float32)

        self.points = wp.array(samples, dtype=wp.vec3)
        self.valid = wp.zeros(num_samples, dtype=wp.int32)
        self.value = wp.zeros(num_samples, dtype=float)
        self.grad = wp.zeros(num_samples, dtype=J2.grad)
        self.hess = wp.zeros(num_samples, dtype=J2.hess)

    def compute(self):
        wp.launch(
            sdf_hessian_kernel,
            dim=len(self.points),
            inputs=[self.points, self.mesh.id, self.max_dist],
            outputs=[self.valid, self.value, self.grad, self.hess],
        )
        wp.synchronize_device()

    def report(self):
        valid = self.valid.numpy().astype(bool)
        n_valid = int(valid.sum())
        print(f"sampled {len(valid)} points, {n_valid} with a surface within max_dist")
        if n_valid == 0:
            return

        hess = self.hess.numpy()[valid]
        symmetry = float(np.abs(hess - np.transpose(hess, (0, 2, 1))).max())
        print(f"Hessian symmetry residual (max |H - Hᵀ|): {symmetry:.3e}")
        print(f"Hessian magnitude (max |H_ij|):           {float(np.abs(hess).max()):.3e}")


def main(device=None, num_samples: int = 100000, seed: int = 42):
    with wp.ScopedDevice(device):
        example = Example(num_samples=num_samples, seed=seed)
        example.compute()
        example.report()
        return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path", type=str, default=None, help="Unused (no rendering); accepted for harness compatibility."
    )
    parser.add_argument("--num-samples", type=int, default=100000, help="Number of bounding-box samples.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the sampling RNG.")
    args = parser.parse_known_args()[0]

    main(device=args.device, num_samples=args.num_samples, seed=args.seed)
