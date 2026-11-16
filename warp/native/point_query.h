// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"

namespace wp {

// Use the same non-contracting, monotone accumulation for primitive distances
// and box bounds. In particular, do not compare an FMA bound to an unfused score.
CUDA_CALLABLE inline float point_distance_sq(const vec3& a, const vec3& b)
{
    const vec3 d = a - b;
#ifdef __CUDA_ARCH__
    return __fadd_rn(__fadd_rn(__fmul_rn(d[0], d[0]), __fmul_rn(d[1], d[1])), __fmul_rn(d[2], d[2]));
#else
    volatile float x = d[0] * d[0];
    volatile float y = d[1] * d[1];
    volatile float z = d[2] * d[2];
    volatile float xy = x + y;
    return xy + z;
#endif
}

CUDA_CALLABLE inline bool point_pair_less(float da, int ia, float db, int ib)
{
    return da < db || (da == db && ia < ib);
}

// Kernel-local inferred state. No owner, device allocation, or public fields.
template <int K> struct point_query_t {
    static_assert(K > 0, "point queries require a positive capacity");
    float distances[K];
    int indices[K];
    int count;
    int cursor;

    CUDA_CALLABLE point_query_t()
        : count(0)
        , cursor(0)
    {
    }
    CUDA_CALLABLE point_query_t& operator+=(const point_query_t&) { return *this; }

    CUDA_CALLABLE inline float bound() const { return count == K ? distances[0] : INFINITY; }

    // Max heap; the root is the worst pair, including its original-index tie.
    CUDA_CALLABLE inline void insert(float d, int id)
    {
        if (count < K) {
            int j = count++;
            while (j > 0) {
                int p = (j - 1) / 2;
                if (!point_pair_less(distances[p], indices[p], d, id))
                    break;
                distances[j] = distances[p];
                indices[j] = indices[p];
                j = p;
            }
            distances[j] = d;
            indices[j] = id;
        } else if (point_pair_less(d, id, distances[0], indices[0])) {
            sift(d, id, K);
        }
    }

    CUDA_CALLABLE inline void sift(float d, int id, int size)
    {
        int j = 0;
        for (;;) {
            int c = j * 2 + 1;
            if (c >= size)
                break;
            if (c + 1 < size && point_pair_less(distances[c], indices[c], distances[c + 1], indices[c + 1]))
                ++c;
            if (!point_pair_less(d, id, distances[c], indices[c]))
                break;
            distances[j] = distances[c];
            indices[j] = indices[c];
            j = c;
        }
        distances[j] = d;
        indices[j] = id;
    }

    CUDA_CALLABLE inline void finish()
    {
        for (int end = count - 1; end > 0; --end) {
            const float d = distances[end];
            const int id = indices[end];
            distances[end] = distances[0];
            indices[end] = indices[0];
            sift(d, id, end);
        }
    }
};

// Dedicated scalar selector: avoid dynamically indexed heap operations for 1-NN.
template <> struct point_query_t<1> {
    float distances[1];
    int indices[1];
    int count;
    int cursor;
    CUDA_CALLABLE point_query_t()
        : count(0)
        , cursor(0)
    {
    }
    CUDA_CALLABLE point_query_t& operator+=(const point_query_t&) { return *this; }
    CUDA_CALLABLE inline float bound() const { return count ? distances[0] : INFINITY; }
    CUDA_CALLABLE inline void insert(float d, int id)
    {
        if (!count || point_pair_less(d, id, distances[0], indices[0])) {
            distances[0] = d;
            indices[0] = id;
            count = 1;
        }
    }
    CUDA_CALLABLE inline void finish() { }
};

CUDA_CALLABLE inline bool point_query_valid_input(const vec3& q, float radius)
{
    return radius > 0.0f && isfinite(q[0]) && isfinite(q[1]) && isfinite(q[2]);
}

CUDA_CALLABLE inline float point_query_cutoff_sq(float radius)
{
    // Overflow to infinity is intentional within the supported finite-distance domain.
    const float r2 = radius * radius;
    return r2 == 0.0f ? 0x1p-149f : r2;
}

template <typename Selector>
CUDA_CALLABLE inline void
point_query_scan(const array_t<vec3>& points, const vec3& q, float cutoff_sq, int exclude, Selector& selected)
{
    float threshold = selected.bound();
    for (int i = 0; i < points.shape[0]; ++i) {
        if (i == exclude)
            continue;
        const float d = point_distance_sq(q, points.data[i]);
        if (d < cutoff_sq && d <= threshold) {
            selected.insert(d, i);
            threshold = selected.bound();
        }
    }
}

CUDA_CALLABLE inline float point_box_distance_sq(const BVH& bvh, int node, const vec3& q)
{
    const BVHPackedNodeHalf lo = bvh_load_node(bvh.node_lowers, node);
    const BVHPackedNodeHalf hi = bvh_load_node(bvh.node_uppers, node);
    const vec3 closest(
        std_min(std_max(q[0], lo.x), hi.x), std_min(std_max(q[1], lo.y), hi.y), std_min(std_max(q[2], lo.z), hi.z)
    );
    return point_distance_sq(q, closest);
}

// Stack capacity is checked, not assumed from a particular constructor. Tests
// instantiate STACK_SIZE=1 to exercise restart without requiring a malformed BVH.
template <int STACK_SIZE = 32, typename Selector>
CUDA_CALLABLE inline void point_query_traverse(
    const BVH& bvh, const array_t<vec3>& points, const vec3& q, float cutoff_sq, int exclude, Selector& selected
)
{
    if (!bvh.num_items)
        return;
    int stack[STACK_SIZE];
    float bounds[STACK_SIZE];
    int size = 0;
    int node = *bvh.root;
    float lower_bound = point_box_distance_sq(bvh, node, q);
    float threshold = selected.bound();
    for (;;) {
        if (lower_bound < cutoff_sq && lower_bound <= threshold) {
            const BVHPackedNodeHalf lo = bvh_load_node(bvh.node_lowers, node);
            const BVHPackedNodeHalf hi = bvh_load_node(bvh.node_uppers, node);
            if (lo.b) {
                for (int p = lo.i; p < int(hi.i); ++p) {
                    const int i = bvh_load_int(bvh.primitive_indices, p);
                    if (i == exclude)
                        continue;
                    const float d = point_distance_sq(q, bvh_load_vec3(points.data, i));
                    if (d < cutoff_sq && d <= threshold) {
                        selected.insert(d, i);
                        threshold = selected.bound();
                    }
                }
            } else {
                int near_node = lo.i;
                int far_node = hi.i;
                float near_bound = point_box_distance_sq(bvh, near_node, q);
                float far_bound = point_box_distance_sq(bvh, far_node, q);
                if (far_bound < near_bound) {
                    const int n = near_node;
                    near_node = far_node;
                    far_node = n;
                    const float d = near_bound;
                    near_bound = far_bound;
                    far_bound = d;
                }
                if (far_bound < cutoff_sq && far_bound <= threshold) {
                    if (size == STACK_SIZE) {
                        selected = Selector();
                        point_query_scan(points, q, cutoff_sq, exclude, selected);
                        return;
                    }
                    stack[size] = far_node;
                    bounds[size++] = far_bound;
                }
                node = near_node;
                lower_bound = near_bound;
                continue;
            }
        }
        if (!size)
            break;
        node = stack[--size];
        lower_bound = bounds[size];  // recheck against the now smaller threshold
    }
}

template <int K>
CUDA_CALLABLE inline point_query_t<K>
point_query_knn(uint64_t id, const array_t<vec3>& points, const vec3& q, float max_dist, int exclude_index)
{
    point_query_t<K> result;
    if (!point_query_valid_input(q, max_dist))
        return result;
    const BVH bvh = bvh_get(id);
#if !defined(NDEBUG)
    assert(points.ndim == 1 && points.strides[0] == sizeof(vec3));
    assert(points.shape[0] == bvh.num_items);
    assert(exclude_index >= -1 && exclude_index < points.shape[0]);
#endif
    point_query_traverse(bvh, points, q, point_query_cutoff_sq(max_dist), exclude_index, result);
    result.finish();
    return result;
}

CUDA_CALLABLE inline int
point_query_nearest(uint64_t id, const array_t<vec3>& points, const vec3& q, float max_dist, int exclude_index)
{
    const point_query_t<1> result = point_query_knn<1>(id, points, q, max_dist, exclude_index);
    return result.count ? result.indices[0] : -1;
}

template <int K> CUDA_CALLABLE inline bool point_query_next(point_query_t<K>& query, int& index)
{
    if (query.cursor == query.count)
        return false;
    index = query.indices[query.cursor++];
    return true;
}

}  // namespace wp
