# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download verified public sources and prepare immutable 3D kNN workloads.

Data archives and generated arrays stay outside the source repository. No raw
scans are upsampled: the actual reference count is recorded when a scan is small.
"""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SOURCES = {
    "bunny": {
        "filename": "bunny.tar.gz",
        "url": "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        "sha256": "a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84",
        "reference": "https://graphics.stanford.edu/data/3Dscanrep/",
        "terms": "Stanford repository usage/acknowledgment notes; retain source attribution",
    },
    "kitti": {
        "filename": "kitti-drive0002.zip",
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",
        "sha256": "ebba5c778893e0d564c261699bcbf10093722df5a3c7e69ab7b7ddaa7637893d",
        "reference": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        "terms": "KITTI dataset terms: CC BY-NC-SA 3.0; research benchmark, source archive not redistributed",
    },
    "redwood": {
        "filename": "DemoICPPointClouds.zip",
        "url": "https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip",
        "sha256": "b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5",
        "reference": "https://www.open3d.org/docs/release/python_api/open3d.data.DemoICPPointClouds.html",
        "terms": "Open3D/Redwood source attribution; preprocessed RGB-D fragment, not raw sensor depth frames",
    },
}
FAMILIES = ("bunny_scan", "kitti", "redwood", "bunny_surface", "sphere", "volume", "clusters", "thin_plane")


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_sources(directory, download=False):
    directory.mkdir(parents=True, exist_ok=True)
    for info in SOURCES.values():
        path = directory / info["filename"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path}; use --download to fetch the source archive")
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(info["url"], temporary)
            if sha256(temporary) != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {temporary}")
            temporary.replace(path)
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")


def read_bunny(directory, member):
    from plyfile import PlyData  # noqa: PLC0415 - optional dataset preparation dependency

    with tarfile.open(directory / SOURCES["bunny"]["filename"]) as archive:
        data = PlyData.read(io.BytesIO(archive.extractfile(member).read()))
    vertices = np.column_stack([data["vertex"][a] for a in ("x", "y", "z")]).astype(np.float64)
    faces = np.stack(data["face"]["vertex_indices"]) if "face" in data else None
    return vertices, faces


def read_scan(directory, family):
    if family == "bunny_scan":
        member = "bunny/data/bun000.ply"
        points, _ = read_bunny(directory, member)
    elif family == "kitti":
        member = "2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin"
        with zipfile.ZipFile(directory / SOURCES["kitti"]["filename"]) as archive:
            points = np.frombuffer(archive.read(member), dtype="<f4").reshape(-1, 4)[:, :3].astype(np.float64)
    else:
        member = "cloud_bin_0.pcd"
        with zipfile.ZipFile(directory / SOURCES["redwood"]["filename"]) as archive:
            name = next(n for n in archive.namelist() if n.endswith(member))
            header, data = archive.read(name).split(b"DATA binary\n", 1)
        fields = next(line for line in header.decode().splitlines() if line.startswith("FIELDS ")).split()[1:]
        values = np.frombuffer(data, dtype="<f4").reshape(-1, len(fields))
        points = values[:, [fields.index(a) for a in ("x", "y", "z")]].astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], {
        "archive_member": member,
        "source_rows": len(points),
        "removed_nonfinite": int((~finite).sum()),
    }


def prepare(directory, family, n, nq, sigma, seed):
    rng = np.random.default_rng(seed)
    metadata = {"family": family, "requested_N": n, "Q": nq, "sigma": sigma, "seed": seed}
    if family in ("bunny_scan", "kitti", "redwood"):
        source, details = read_scan(directory, family)
        metadata.update(details)
        count = min(n, len(source) - nq)
        order = rng.permutation(len(source))
        p, q = source[order[:count]], source[order[count : count + nq]]
        metadata["query_generation"] = "disjoint held-out source rows; no resampling or upsampling"
        lower, upper = source.min(axis=0), source.max(axis=0)
    elif family == "bunny_surface":
        vertices, faces = read_bunny(directory, "bunny/reconstruction/bun_zipper.ply")
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        chosen = triangles[rng.choice(len(faces), n + nq, p=areas / areas.sum())]
        u, v = rng.random((2, n + nq))
        u = np.sqrt(u)
        samples = (
            (1 - u[:, None]) * chosen[:, 0] + (u * (1 - v))[:, None] * chosen[:, 1] + (u * v)[:, None] * chosen[:, 2]
        )
        p, q = samples[:n], samples[n:]
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        metadata.update(
            archive_member="bunny/reconstruction/bun_zipper.ply",
            query_generation="independent area-uniform triangle samples",
        )
    else:
        if family == "sphere":
            samples = rng.normal(size=(n + nq, 3))
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        elif family == "volume":
            samples = rng.uniform(-1, 1, (n + nq, 3))
        elif family == "thin_plane":
            samples = rng.uniform(-1, 1, (n + nq, 3))
            samples[:, 2] *= 0.001
        else:
            centers = rng.uniform(-0.8, 0.8, (8, 3))
            samples = centers[rng.integers(8, size=n + nq)] + rng.normal(scale=0.025, size=(n + nq, 3))
            # Explicit sparse outliers stress bounding-volume traversal.
            outliers = rng.choice(n + nq, max(1, (n + nq) // 50), replace=False)
            samples[outliers] = rng.uniform(-1, 1, (len(outliers), 3))
        p, q = samples[:n], samples[n:]
        lower, upper = np.full(3, -1.0), np.full(3, 1.0)
        metadata["query_generation"] = "independent samples of the same distribution"
    center, scale = (lower + upper) * 0.5, float(np.max(upper - lower))
    p, q = (p - center) / scale, (q - center) / scale
    if sigma:
        p += rng.normal(scale=sigma, size=p.shape)
        q += rng.normal(scale=sigma, size=q.shape)
    p, q = np.ascontiguousarray(p, dtype=np.float32), np.ascontiguousarray(q, dtype=np.float32)
    metadata.update(
        N=len(p), center=center.tolist(), scale=scale, noise_units="fraction of original longest bounding-box edge"
    )
    metadata["points_sha256"] = hashlib.sha256(p.tobytes()).hexdigest()
    metadata["queries_sha256"] = hashlib.sha256(q.tobytes()).hexdigest()
    return p, q, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--suite", choices=("smoke", "pilot"), default="pilot")
    args = parser.parse_args()
    get_sources(args.source_dir, args.download)
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [
        (f, 4096 if args.suite == "smoke" else 100_000, 128 if args.suite == "smoke" else 1024, (1, 32))
        for f in FAMILIES
    ]
    if args.suite == "pilot":
        specs += [(f, 1_000_000, 8192, (11,)) for f in ("sphere", "volume")]
        specs += [(f, 1000, 32, (32,)) for f in ("sphere", "volume")]
    report = {
        "schema_version": 1,
        "suite": args.suite,
        "numpy_version": np.__version__,
        "sources": SOURCES,
        "cases": [],
    }
    for family, n, nq, ks in specs:
        for sigma in (0.0, 0.01):
            seed = 1201 + FAMILIES.index(family) * 101
            p, q, metadata = prepare(args.source_dir, family, n, nq, sigma, seed)
            filename = f"{family}-n{n}-q{nq}-sigma{sigma:g}.npz"
            np.savez_compressed(args.output / filename, points=p, queries=q)
            for k in ks:
                report["cases"].append(
                    {
                        **metadata,
                        "id": f"{family}-n{n}-q{nq}-sigma{sigma:g}-k{k}",
                        "k": k,
                        "file": filename,
                        "file_sha256": sha256(args.output / filename),
                    }
                )
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(report['cases'])} cases to {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
