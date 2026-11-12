// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Standalone benchmark of the vendored, unmodified cuBQL findKNN implementation.
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
// clang-format off
#include <cub/cub.cuh>
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "cuBQL/builder/cuda/rebinMortonBuilder.h"
#include "cuBQL/queries/pointData/knn.h"
// clang-format on

using Tree = cuBQL::BinaryBVH<float, 3>;
using Point = cuBQL::vec3f;

__global__ void upstream_boxes(const Point* points, cuBQL::box3f* boxes, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        boxes[i] = cuBQL::box3f(points[i], points[i]);
}

extern "C" void* upstream_build(const Point* points, int n, int builder, int leaf, cudaStream_t stream)
{
    Tree* tree = new Tree;
    cuBQL::box3f* boxes = nullptr;
    cudaMallocAsync(&boxes, sizeof(cuBQL::box3f) * n, stream);
    upstream_boxes<<<(n + 255) / 256, 256, 0, stream>>>(points, boxes, n);
    cuBQL::BuildConfig config(leaf);  // leaf=0 leaves the choice to cuBQL.
    switch (builder) {
    case 0:
        cuBQL::gpuBuilder(*tree, boxes, n, config, stream);
        break;
    case 1:
        cuBQL::cuda::radixBuilder(*tree, boxes, n, config, stream);
        break;
    case 2:
        cuBQL::cuda::rebinRadixBuilder(*tree, boxes, n, config, stream);
        break;
    case 3:
        cuBQL::gpuBuilder(*tree, boxes, n, config.enableSAH(), stream);
        break;
    case 4:
        cuBQL::gpuBuilder(*tree, boxes, n, config.enableELH(), stream);
        break;
    }
    cudaFreeAsync(boxes, stream);
    return tree;
}

extern "C" void upstream_destroy(Tree* tree, cudaStream_t stream)
{
    cuBQL::free(*tree, stream);
    delete tree;
}

template <int K, int Stride>
__global__ void upstream_query_kernel(
    Tree tree,
    const Point* points,
    const Point* queries,
    int nq,
    const int* order,
    int64_t* indices,
    float* distances,
    int* counts
)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t % Stride != 0 || t / Stride >= nq)
        return;
    int row = order ? order[t / Stride] : t / Stride;
    cuBQL::knn::Candidate candidates[K];
    // No replacement distance calculation, selector, tie rule or traversal.
    auto result = cuBQL::points::findKNN(candidates, K, tree, points, queries[row]);
    counts[row] = result.numFound;
    for (int j = 0; j < K; ++j) {
        indices[row * K + j] = j < result.numFound ? candidates[j].primID : -1;
        distances[row * K + j] = j < result.numFound ? sqrtf(candidates[j].sqrDist) : INFINITY;
    }
}

template <int K>
void launch_upstream(
    Tree tree,
    const Point* points,
    const Point* queries,
    int nq,
    int stride,
    int block,
    const int* order,
    int64_t* indices,
    float* distances,
    int* counts,
    cudaStream_t stream
)
{
#define RUN(S) case S: upstream_query_kernel<K, S><<<(nq * S + block - 1) / block, block, 0, stream>>>(tree, points, queries, nq, order, indices, distances, counts); break
    switch (stride) {
        RUN(1);
        RUN(2);
        RUN(4);
        RUN(8);
        RUN(16);
        RUN(32);
    }
#undef RUN
}

extern "C" int upstream_query(
    Tree* tree,
    const Point* points,
    const Point* queries,
    int nq,
    int k,
    int stride,
    int block,
    const int* order,
    int64_t* indices,
    float* distances,
    int* counts,
    cudaStream_t stream
)
{
#define RUN(K) case K: launch_upstream<K>(*tree, points, queries, nq, stride, block, order, indices, distances, counts, stream); break
    switch (k) {
        RUN(1);
        RUN(11);
        RUN(32);
    default:
        return -1;
    }
#undef RUN
    return int(cudaGetLastError());
}
