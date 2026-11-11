# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional GPU library adapters for the broad 3D point-query comparison.

All adapters return CUDA int64 indices and float32 Euclidean distances in
original query order. Output conversion and query ordering are timed. Selection
is sorted by distance; equal-distance index ordering is library-specific.
"""

import ctypes
import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.benchmark_point_query import load_physicsnemo
from warp.examples.benchmarks.point_query_common import (
    make_api_query,
    make_ordered_query,
    point_query_configuration,
    query_codes,
)
from warp.examples.benchmarks.point_query_cubql import CubqlLibrary

METHODS = (
    "warp_default",
    "warp_configured",
    "physicsnemo",
    "cubql",
    "cupy_kdtree",
    "keops",
    "faiss_flat",
    "cuvs_brute_force",
    "cuvs_unexpanded",
    "cuvs_minkowski",
    "pytorch3d",
    "cuvs_cagra",
)


class Provider:
    def __init__(self, name, cache, cubql_library=None):
        self.name = name
        self.metadata = {"exact_algorithm": name != "cuvs_cagra", "name": name}
        if name.startswith("warp_") or name == "cubql":
            self.metadata["output_allocation"] = "Torch CUDA allocation, wrapped by Warp without copying"
        if name == "physicsnemo":
            self.library, digest = load_physicsnemo(Path(cache) / "physicsnemo")
            self.metadata["source_sha256"] = digest
        elif name == "cubql":
            if cubql_library is None:
                raise ValueError("--cubql-library is required for direct cuBQL")
            self.library = CubqlLibrary(Path(cubql_library))
            self.metadata["library_sha256"] = hashlib.sha256(Path(cubql_library).read_bytes()).hexdigest()
        elif name == "cupy_kdtree":
            self.library = importlib.import_module("cupyx.scipy.spatial").KDTree
            self.metadata["leafsize"] = 16
        elif name == "keops":
            self.library = importlib.import_module("pykeops.torch").LazyTensor
            self.metadata["stream_behavior"] = "KeOps 2.3 NVRTC uses stream 0 and cuCtxSynchronize internally"
        elif name == "faiss_flat":
            self.library = importlib.import_module("faiss")
            importlib.import_module("faiss.contrib.torch_utils")
            self.resources = self.library.StandardGpuResources()
            self.resources.setTempMemory(256 * 1024**2)
            self.resources.setDefaultStream(
                0, self.library.cast_integer_to_cudastream_t(torch.cuda.current_stream().cuda_stream)
            )
            self.metadata.update(
                scratch_bytes=256 * 1024**2,
                float16=False,
                imported_version=self.library.__version__,
                imported_path=self.library.__file__,
            )
        elif name.startswith("cuvs_"):
            self.library = importlib.import_module(
                "cuvs.neighbors." + ("cagra" if name == "cuvs_cagra" else "brute_force")
            )
            from cuvs.common import Resources  # noqa: PLC0415

            self.resources = Resources()
            # The cuVS 26.10 Python integer stream binding crashed in this
            # environment. Use explicit pointer conversion through the public
            # C API so every library really uses the same stream.
            libcuvs = importlib.import_module("libcuvs")
            self.c_api = ctypes.CDLL(str(Path(libcuvs.__file__).parent / "lib64/libcuvs_c.so"))
            self.c_api.cuvsStreamSet.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
            self.c_api.cuvsStreamSet.restype = ctypes.c_int
            if self.c_api.cuvsStreamSet(self.resources.get_c_obj(), torch.cuda.current_stream().cuda_stream) != 1:
                raise RuntimeError("cuvsStreamSet failed")
            self.metadata["stream_binding"] = "public cuvsStreamSet C API"
            self.metadata["output_allocation"] = "Torch CUDA allocation on the query stream"
            if name != "cuvs_cagra":
                self.metric = "minkowski" if name == "cuvs_minkowski" else "sqeuclidean"
                if name == "cuvs_unexpanded":
                    # The public C API exposes L2Unexpanded=4, but cuVS 26.10
                    # omits its name from the Python metric registry. Register
                    # only that binding; the native implementation is unchanged.
                    self.metric = "benchmark_l2_unexpanded"
                    importlib.import_module("cuvs.distance").DISTANCE_TYPES[self.metric] = 4
                    self.metadata["metric_binding"] = "C API L2Unexpanded=4 added to Python DISTANCE_TYPES"
                self.metadata.update(metric=self.metric, metric_arg=2.0)
            if name == "cuvs_cagra":
                self.index_params = self.library.IndexParams(
                    build_algo="nn_descent", intermediate_graph_degree=64, graph_degree=32
                )
                self.search_params = self.library.SearchParams(itopk_size=64)
                self.metadata.update(
                    build_algo="nn_descent",
                    graph_degree=32,
                    intermediate_graph_degree=64,
                    itopk_size=64,
                    zero_padding="3 to 4 dimensions inside timed build/search; preserves L2 distance",
                )
        elif name == "pytorch3d":
            self.library = importlib.import_module("pytorch3d.ops").knn_points
            self.metadata["sorting"] = "return_sorted=False, followed by torch.sort and gather on GPU"

    def bind(self, points, queries, k):
        return Problem(self, points, queries, k)


class Problem:
    def __init__(self, provider, points, queries, k):
        self.provider = provider
        self.name = provider.name
        self.p, self.q, self.k = points, queries, k
        self.nq = len(queries)
        self.leaf = 4
        self.stride = 1
        self.ordered = False
        self.block_dim = 256
        if self.name in ("warp_default", "warp_configured", "cubql"):
            self.wp_points = wp.from_torch(points, dtype=wp.vec3)
            self.wp_queries = wp.from_torch(queries, dtype=wp.vec3)
            self.excludes = wp.full(self.nq, -1, dtype=int, device="cuda:0")
            if self.name == "warp_configured":
                self.leaf, self.stride, self.ordered = point_query_configuration(
                    len(points), self.nq, self.wp_points.device.sm_count
                )
            self.block_dim = 32 if self.stride > 1 else 256
            if self.name != "cubql":
                self.kernel = make_ordered_query(k, True) if self.ordered else make_api_query(k, True, self.stride)
        if self.name.startswith("cuvs_") or self.name == "cupy_kdtree":
            import cupy as cp  # noqa: PLC0415

            self.cp = cp
            self.cp_points, self.cp_queries = cp.from_dlpack(points), cp.from_dlpack(queries)

    @property
    def has_index(self):
        return self.name not in ("physicsnemo", "keops", "pytorch3d")

    def build(self):
        if self.name.startswith("warp_"):
            return wp.Bvh(self.wp_points, self.wp_points, constructor="lbvh", leaf_size=self.leaf)
        if self.name == "cubql":
            return self.provider.library.build(self.wp_points, leaf=4)
        if self.name == "cupy_kdtree":
            return self.provider.library(self.cp_points, leafsize=16)
        if self.name == "faiss_flat":
            config = self.provider.library.GpuIndexFlatConfig()
            config.device, config.useFloat16 = 0, False
            index = self.provider.library.GpuIndexFlatL2(self.provider.resources, 3, config)
            index.add(self.p)
            return index
        if self.name in ("cuvs_brute_force", "cuvs_unexpanded", "cuvs_minkowski"):
            return self.provider.library.build(
                self.cp_points,
                metric=self.provider.metric,
                metric_arg=2.0,
                resources=self.provider.resources,
            )
        if self.name == "cuvs_cagra":
            return self.provider.library.build(
                self.provider.index_params,
                self.cp.from_dlpack(torch.nn.functional.pad(self.p, (0, 1))),
                resources=self.provider.resources,
            )
        return None

    def search(self, index):
        if self.name.startswith("warp_"):
            torch_ids = torch.empty((self.nq, self.k), dtype=torch.int64, device=self.q.device)
            torch_distances = torch.empty((self.nq, self.k), dtype=torch.float32, device=self.q.device)
            ids, distances = wp.from_torch(torch_ids), wp.from_torch(torch_distances)
            counts = wp.empty(self.nq, dtype=int, device="cuda:0")
            extra = []
            if self.ordered:
                keys = wp.empty(2 * self.nq, dtype=int, device="cuda:0")
                order = wp.empty_like(keys)
                wp.launch(query_codes, self.nq, inputs=[index.id, self.wp_queries, keys, order])
                wp.utils.radix_sort_pairs(keys, order, self.nq)
                extra = [order]
            wp.launch(
                self.kernel,
                self.nq * self.stride,
                inputs=[
                    index.id,
                    self.wp_points,
                    self.wp_queries,
                    np.inf,
                    self.excludes,
                    *extra,
                    ids,
                    distances,
                    counts,
                ],
                block_dim=self.block_dim,
            )
            return torch_ids, torch_distances
        if self.name == "cubql":
            torch_ids = torch.empty((self.nq, self.k), dtype=torch.int32, device=self.q.device)
            torch_distances = torch.empty((self.nq, self.k), dtype=torch.float32, device=self.q.device)
            ids, distances = wp.from_torch(torch_ids), wp.from_torch(torch_distances)
            counts = wp.empty(self.nq, dtype=int, device="cuda:0")
            index.query(self.wp_queries, self.k, np.inf, self.excludes, ids, distances, counts)
            return torch_ids.long(), torch_distances.clamp_min(0).sqrt()
        if self.name == "physicsnemo":
            return self.provider.library.knn_impl(self.p, self.q, self.k)
        if self.name == "pytorch3d":
            result = self.provider.library(self.q[None], self.p[None], K=self.k, return_nn=False, return_sorted=False)
            ds, order = result.dists[0].sort(dim=1)
            return result.idx[0].gather(1, order), ds.clamp_min(0).sqrt()
        if self.name == "keops":
            lazy = self.provider.library
            d2 = ((lazy(self.q[:, None, :]) - lazy(self.p[None, :, :])) ** 2).sum(-1)
            ds, ids = d2.Kmin_argKmin(self.k, dim=1)
            return ids.long(), ds.clamp_min(0).sqrt()
        if self.name == "faiss_flat":
            ds, ids = index.search(self.q, self.k)
            return ids.long(), ds.clamp_min(0).sqrt()
        if self.name == "cupy_kdtree":
            ds, ids = index.query(self.cp_queries, k=self.k, eps=0.0)
            return torch.from_dlpack(ids).reshape(self.nq, self.k).long(), torch.from_dlpack(ds).reshape(
                self.nq, self.k
            ).float()
        # Allocate outputs on the measured stream with Torch's graph-aware
        # allocator. cuVS's default Python output allocation uses stream 0.
        ids = torch.empty(
            (self.nq, self.k), device=self.q.device, dtype=torch.uint32 if self.name == "cuvs_cagra" else torch.int64
        )
        ds = torch.empty((self.nq, self.k), device=self.q.device, dtype=torch.float32)
        outputs = {"neighbors": self.cp.from_dlpack(ids), "distances": self.cp.from_dlpack(ds)}
        if self.name in ("cuvs_brute_force", "cuvs_unexpanded", "cuvs_minkowski"):
            self.provider.library.search(index, self.cp_queries, self.k, resources=self.provider.resources, **outputs)
        else:
            self.provider.library.search(
                self.provider.search_params,
                index,
                self.cp.from_dlpack(torch.nn.functional.pad(self.q, (0, 1))),
                self.k,
                resources=self.provider.resources,
                **outputs,
            )
        return ids.long(), ds if self.name == "cuvs_minkowski" else ds.clamp_min(0).sqrt()

    def one_shot(self):
        index = self.build()
        result = self.search(index)
        return index, result
