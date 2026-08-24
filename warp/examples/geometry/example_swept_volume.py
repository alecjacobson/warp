# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Geometry Swept Volume
#
# Computes the swept volume (motion envelope) of an animated rigid assembly
# with warp.geometry.swept_volume: the single closed mesh that encloses the
# union of every input mesh over every sampled pose.
#
# The method samples a dense signed-distance field
#
#     D(p) = min_mesh min_sample  sdf_mesh( X[mesh, sample]^-1 p )
#
# by pushing each grid point back into every mesh's rest frame (the motion is
# rigid, so one closest-point query per pose is enough), then extracts the zero
# isosurface with marching cubes. This is the dense-stamping baseline: no root
# finding or narrow band, so motion *between* the sampled poses is not
# conservatively bounded -- sample finely enough for the tolerance you need.
#
# By default the example animates a procedural two-link arm, so it runs with no
# external assets. Pass --usd to run on an animated USD hierarchy instead, such
# as the UR10 arm from the swept-volume feature request (GH-1824).
#
# Inside/outside is classified with --sign (default 'auto'): the winding number
# for USD assemblies -- CAD parts like the UR10 are open, non-watertight visual
# shells, and the faster closest-face-normal classifier is incoherent on them
# (spurious interior pockets, hundreds of disconnected junk shells) -- and the
# normal classifier for the watertight procedural arm.
#
#   uv run --with usd-core warp/examples/geometry/example_swept_volume.py
#   uv run --with usd-core warp/examples/geometry/example_swept_volume.py --usd ur10_animated.usda
#   uv run --with usd-core --with polyscope warp/examples/geometry/example_swept_volume.py --polyscope
###########################################################################

import numpy as np

import warp as wp
import warp.geometry


def box_mesh(size, center=(0.0, 0.0, 0.0)):
    """Axis-aligned box triangle mesh (outward-oriented), as (points, indices)."""
    sx, sy, sz = (0.5 * s for s in size)
    cx, cy, cz = center
    corners = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # bottom (-z)
            [4, 5, 6],
            [4, 6, 7],  # top (+z)
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 3, 7],
            [2, 7, 6],  # +y
            [1, 2, 6],
            [1, 6, 5],  # +x
            [3, 0, 4],
            [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    )
    return corners, faces.reshape(-1)


def quat_pitch(angle):
    """wp/USD-order (x, y, z, w) quaternion for a rotation about +y (pitch)."""
    return np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)], dtype=np.float32)


def _rotate_np(q, v):
    """Rotate row vectors ``v`` (..., 3) by quaternion ``q`` = (x, y, z, w)."""
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return v @ R.T


def compose(a, b):
    """Compose two transforms given as (7,) arrays: result applies b then a."""
    ta = wp.transform(wp.vec3(*a[:3]), wp.quat(*a[3:]))
    tb = wp.transform(wp.vec3(*b[:3]), wp.quat(*b[3:]))
    out = wp.transform_multiply(ta, tb)
    return np.array([*out.p, *out.q], dtype=np.float32)


def procedural_arm(num_samples=24, device=None):
    """A two-link arm that swings through a pick-and-place-like arc.

    Returns ``(meshes, transforms, times)`` where ``transforms`` has shape
    ``(num_meshes, num_samples, 7)`` (translation xyz + quaternion xyzw).
    """
    # Rest-pose geometry: three boxes forming base, upper arm, forearm. Each link
    # is modeled in its own local frame with its joint at the origin.
    base_pts, base_idx = box_mesh((0.6, 0.6, 0.3), center=(0.0, 0.0, 0.15))
    upper_pts, upper_idx = box_mesh((0.25, 0.25, 1.2), center=(0.0, 0.0, 0.6))
    fore_pts, fore_idx = box_mesh((0.2, 0.2, 1.0), center=(0.0, 0.0, 0.5))

    meshes = [
        wp.Mesh(wp.array(base_pts, dtype=wp.vec3, device=device), wp.array(base_idx, dtype=wp.int32, device=device)),
        wp.Mesh(wp.array(upper_pts, dtype=wp.vec3, device=device), wp.array(upper_idx, dtype=wp.int32, device=device)),
        wp.Mesh(wp.array(fore_pts, dtype=wp.vec3, device=device), wp.array(fore_idx, dtype=wp.int32, device=device)),
    ]

    # Joint frames: the shoulder sits atop the base (z=0.3), the elbow atop the
    # upper arm (z=1.2), both pitching about y in their parent's frame.
    times = np.linspace(0.0, 1.0, num_samples).astype(np.float32)
    transforms = np.zeros((3, num_samples, 7), dtype=np.float32)
    transforms[:, :, 6] = 1.0  # identity quaternions by default

    for s, t in enumerate(times):
        # Two joints swing out of phase to sweep a broad, non-convex region.
        shoulder_angle = 1.2 * np.sin(2.0 * np.pi * t)
        elbow_angle = 1.4 * np.sin(2.0 * np.pi * t + 1.0)

        # Base is static.
        transforms[0, s] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Upper arm = shoulder joint rotation about the shoulder frame.
        upper_world = np.array([0.0, 0.0, 0.3, *quat_pitch(shoulder_angle)], dtype=np.float32)
        transforms[1, s] = upper_world

        # Forearm = upper-arm transform composed with the elbow rotation, so the
        # kinematic chain is respected.
        elbow_local = np.array([0.0, 0.0, 1.2, *quat_pitch(elbow_angle)], dtype=np.float32)
        transforms[2, s] = compose(upper_world, elbow_local)

    return meshes, transforms, times


def _triangulate(counts, idx):
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if counts.size and (counts == 3).all():
        return idx.reshape(-1, 3)
    ntri = counts - 2
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    face = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(int(ntri.sum())) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    s = starts[face]
    return np.stack([idx[s], idx[s + k], idx[s + k + 1]], axis=1)


def load_usd_assembly(path, num_samples=24, device=None):
    """Extract rest-pose meshes and sampled world transforms from an animated USD.

    Every ``UsdGeomMesh`` (including through instance proxies) becomes one
    :class:`warp.Mesh`; its per-sample world transform is read from the stage's
    xform cache. Returns ``(meshes, transforms, times)``.

    The meshes are built with ``support_winding_number=True``. CAD assemblies
    like the UR10 are made of open, non-watertight visual shells, for which
    closest-face-normal sign classification is unreliable and produces an
    incoherent field (spurious interior pockets, hundreds of junk shells). The
    generalized winding number stays robust, so the USD path uses it by default
    (see :class:`warp.geometry.SweptVolumeSign`).
    """
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    prims, meshes = [], []
    for prim in stage.Traverse(pred):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        points = geom.GetPointsAttr().Get()
        if not points:
            continue
        faces = _triangulate(geom.GetFaceVertexCountsAttr().Get(), geom.GetFaceVertexIndicesAttr().Get())
        prims.append(prim)
        meshes.append(
            wp.Mesh(
                wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device),
                wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device),
                support_winding_number=True,
            )
        )

    if not meshes:
        raise SystemExit(f"No UsdGeomMesh found in {path}")

    t0 = stage.GetStartTimeCode()
    t1 = stage.GetEndTimeCode()
    if t1 <= t0:
        t1 = t0 + 1.0
    times = np.linspace(t0, t1, num_samples).astype(np.float32)

    cache = UsdGeom.XformCache()
    transforms = np.zeros((len(meshes), num_samples, 7), dtype=np.float32)
    for s, t in enumerate(times):
        cache.SetTime(Usd.TimeCode(float(t)))
        for m, prim in enumerate(prims):
            mat = cache.GetLocalToWorldTransform(prim)
            xform = Gf.Transform(mat)
            trans = xform.GetTranslation()
            rot = xform.GetRotation().GetQuat()
            imag = rot.GetImaginary()
            transforms[m, s] = [trans[0], trans[1], trans[2], imag[0], imag[1], imag[2], rot.GetReal()]

    return meshes, transforms, times


def write_usd(stage_path, verts, indices):
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.CreateNew(stage_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/swept_volume")
    v = verts.numpy()
    f = indices.numpy()
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * (len(f) // 3))
    mesh.CreateFaceVertexIndicesAttr(f.tolist())
    stage.Save()


def main(
    usd_path=None,
    num_samples=24,
    voxel_size=0.08,
    sign_mode=None,
    stage_path="example_geometry_swept_volume.usd",
    show_polyscope=False,
):
    device = wp.get_device()

    if usd_path is not None:
        meshes, transforms, times = load_usd_assembly(usd_path, num_samples=num_samples, device=device)
        label = usd_path
        # CAD assemblies are open shells; the normal classifier is incoherent on
        # them, so default the USD path to the robust winding number.
        default_sign = warp.geometry.SweptVolumeSign.WINDING_NUMBER
    else:
        meshes, transforms, times = procedural_arm(num_samples=num_samples, device=device)
        label = "procedural two-link arm"
        # The procedural boxes are watertight, so the faster normal classifier is fine.
        default_sign = warp.geometry.SweptVolumeSign.NORMAL

    if sign_mode is None:
        sign_mode = default_sign

    total_tris = sum(len(m.indices.numpy()) // 3 for m in meshes)
    print(
        f"{label}: {len(meshes)} meshes, {total_tris} triangles, "
        f"{num_samples} pose samples over t in [{times[0]:g}, {times[-1]:g}], sign={sign_mode.name}"
    )

    with wp.ScopedTimer("swept_volume"):
        verts, indices = warp.geometry.swept_volume(
            meshes,
            transforms,
            voxel_size=voxel_size,
            sign_mode=sign_mode,
            device=device,
        )
        wp.synchronize_device()

    v = verts.numpy()
    print(f"envelope: {len(v)} vertices, {len(indices.numpy()) // 3} triangles")
    print(f"envelope AABB: min {np.round(v.min(axis=0), 3)}  max {np.round(v.max(axis=0), 3)}")

    if stage_path:
        write_usd(stage_path, verts, indices)
        print(f"wrote {stage_path}")

    if show_polyscope:
        import polyscope as ps  # noqa: PLC0415

        ps.set_up_dir("z_up")
        ps.init()
        ps.register_surface_mesh("swept volume", v, indices.numpy().reshape(-1, 3), color=(0.92, 0.41, 0.20))
        # Overlay the stamped input poses as a translucent point cloud. Transform
        # a capped, random subset of each mesh's vertices with NumPy so large USD
        # assemblies (hundreds of thousands of points) stay responsive.
        rng = np.random.default_rng(0)
        clouds = []
        for m in range(len(meshes)):
            pts = meshes[m].points.numpy().astype(np.float64)
            keep = rng.choice(len(pts), size=min(2000, len(pts)), replace=False)
            pts = pts[keep]
            for s in range(num_samples):
                q = transforms[m, s, 3:]  # (x, y, z, w)
                t = transforms[m, s, :3]
                clouds.append(_rotate_np(q, pts) + t)
        ps.register_point_cloud("stamped poses", np.concatenate(clouds), radius=0.001, color=(0.16, 0.47, 0.84))
        ps.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--usd",
        type=str,
        default=None,
        help="Path to an animated USD assembly (e.g. the UR10). Uses a procedural arm if omitted.",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of pose samples to stamp.")
    parser.add_argument("--voxel-size", type=float, default=0.08, help="Grid cell size in world units.")
    parser.add_argument(
        "--sign",
        type=str,
        default="auto",
        choices=["auto", "normal", "winding"],
        help="Inside/outside classifier. 'auto' picks winding for USD (non-watertight CAD) and normal for the procedural arm.",
    )
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_geometry_swept_volume.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--polyscope", action="store_true", help="Show the result in an interactive polyscope viewer.")
    args = parser.parse_known_args()[0]

    sign_mode = {
        "auto": None,
        "normal": warp.geometry.SweptVolumeSign.NORMAL,
        "winding": warp.geometry.SweptVolumeSign.WINDING_NUMBER,
    }[args.sign]
    with wp.ScopedDevice(args.device):
        main(
            usd_path=args.usd,
            num_samples=args.num_samples,
            voxel_size=args.voxel_size,
            sign_mode=sign_mode,
            stage_path=args.stage_path,
            show_polyscope=args.polyscope,
        )
