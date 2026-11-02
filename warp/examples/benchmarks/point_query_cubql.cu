// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Optional direct cuBQL baseline, using the revision vendored in this checkout.
// Compile as a standalone shared library; it does not change Warp's native ABI.
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
// cuBQL requires bvh.h before builder/cuda.h, and CUB before cuBQL.
// clang-format off
#include <cub/cub.cuh>
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"
#include <cmath>
// clang-format on

using Tree = cuBQL::BinaryBVH<float, 3>;
using Point = cuBQL::vec3f;

__global__ void make_boxes(const Point* points, cuBQL::box3f* boxes, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        boxes[i] = cuBQL::box3f(points[i], points[i]);
}

extern "C" void* point_cubql_build(const Point* points, int n, int leaf, cudaStream_t stream)
{
    Tree* tree = new Tree;
    if (!n)
        return tree;
    cuBQL::box3f* boxes = nullptr;
    cudaMallocAsync(&boxes, sizeof(cuBQL::box3f) * n, stream);
    make_boxes<<<(n + 255) / 256, 256, 0, stream>>>(points, boxes, n);
    cuBQL::gpuBuilder(*tree, boxes, n, cuBQL::BuildConfig(leaf), stream);
    cudaFreeAsync(boxes, stream);
    return tree;
}

extern "C" void point_cubql_destroy(Tree* tree, cudaStream_t stream)
{
    cuBQL::free(*tree, stream);
    delete tree;
}

template <int K>
__global__ void query_kernel(
    Tree tree,
    const Point* points,
    const Point* queries,
    int nq,
    float cutoff,
    const int* exclude,
    int* indices,
    float* distances,
    int* counts
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nq)
        return;
    Point q = queries[row];
    float ds[K] = {};
    int ids[K] = {};
    int count = 0;
    float cutoff_sq = cutoff * cutoff;
    if (cutoff_sq == 0.0f)
        cutoff_sq = 0x1p-149f;
    if (cutoff > 0 && isfinite(q.x) && isfinite(q.y) && isfinite(q.z) && tree.numPrims) {
        // Adapt only selection: match Warp's round-to-nearest scores, strict
        // cutoff, index exclusion, and deterministic ties. cuBQL's own distance
        // bounds use matching arithmetic with --fmad=false; nextafter preserves ties in
        // its strict shrinking-radius traversal.
        auto visit = [&](int id) {
            if (id != exclude[row]) {
                Point delta = q - points[id];
                float d = __fadd_rn(
                    __fadd_rn(__fmul_rn(delta.x, delta.x), __fmul_rn(delta.y, delta.y)), __fmul_rn(delta.z, delta.z)
                );
                if (d < cutoff_sq && (count < K || d < ds[K - 1] || (d == ds[K - 1] && id < ids[K - 1]))) {
                    int pos = count < K ? count++ : K - 1;
                    while (pos > 0 && (d < ds[pos - 1] || (d == ds[pos - 1] && id < ids[pos - 1]))) {
                        ds[pos] = ds[pos - 1];
                        ids[pos] = ids[pos - 1];
                        --pos;
                    }
                    ds[pos] = d;
                    ids[pos] = id;
                }
            }
            return count == K ? nextafterf(ds[K - 1], INFINITY) : cutoff_sq;
        };
        cuBQL::shrinkingRadiusQuery::forEachPrim(visit, tree, q, cutoff_sq);
    }
    counts[row] = count;
    for (int r = 0; r < K; ++r) {
        indices[row * K + r] = r < count ? ids[r] : -1;
        distances[row * K + r] = r < count ? ds[r] : INFINITY;
    }
}

extern "C" int point_cubql_query(
    Tree* tree,
    const Point* points,
    const Point* queries,
    int nq,
    int k,
    float cutoff,
    const int* excludes,
    int* indices,
    float* distances,
    int* counts,
    cudaStream_t stream
)
{
#define RUN(K)                                                                                                           \
    case K:                                                                                                              \
        query_kernel<K><<<(nq + 255) / 256, 256, 0, stream>>>(                                                             \
            *tree, points, queries, nq, cutoff, excludes, indices, distances, counts                                      \
        );                                                                                                               \
        break
    switch (k) {
        RUN(1);
        RUN(4);
        RUN(8);
        RUN(11);
        RUN(16);
        RUN(32);
        RUN(64);
    default:
        return -1;
    }
#undef RUN
    return int(cudaGetLastError());
}
