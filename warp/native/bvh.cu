// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "bvh.h"
#include "cuda_util.h"
#include "error.h"
#include "sort.h"

#include <algorithm>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define THRUST_IGNORE_CUB_VERSION_CHECK
#define REORDER_HOST_TREE

#include <cub/cub.cuh>

extern CUcontext get_current_context();

namespace wp {
void bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int constructor_type, BVH& bvh, int leaf_size);
void bvh_destroy_host(BVH& bvh);

__global__ void memset_kernel(int* dest, int value, size_t n)
{
    const size_t tid
        = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);

    if (tid < n) {
        dest[tid] = value;
    }
}

// for LBVH: this will start with some muted leaf nodes, but that is okay, we can still trace up because there parents
// information is still valid the only thing worth mentioning is that when the parent leaf node is also a leaf node, we
// need to recompute its bounds, since their child information are lost for a compact tree such as those from SAH or
// Median constructor, there is no muted leaf nodes
__global__ void bvh_refit_kernel(
    int n,
    const int* __restrict__ parents,
    int* __restrict__ child_count,
    const int* __restrict__ primitive_indices,
    BVHPackedNodeHalf* __restrict__ node_lowers,
    BVHPackedNodeHalf* __restrict__ node_uppers,
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = node_lowers[index].b;
        int parent = parents[index];

        if (leaf) {
            BVHPackedNodeHalf& lower = node_lowers[index];
            BVHPackedNodeHalf& upper = node_uppers[index];
            // update the leaf node

            // Single-node trees store the root leaf without a parent.
            if (parent == -1 || !node_lowers[parent].b) {
                const int start = lower.i;
                const int end = upper.i;

                bounds3 bound;
                for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                    const int primitive = primitive_indices[primitive_counter];
                    bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                }
                (vec3&)lower = bound.lower;
                (vec3&)upper = bound.upper;
            }
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            parent = parents[index];
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (finished == 1) {
                BVHPackedNodeHalf& parent_lower = node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = node_uppers[parent];
                if (parent_lower.b)
                // a packed leaf node can still be a parent in LBVH, we need to recompute its bounds
                // since we've lost its left and right child node index in the muting process
                {
                    // update the leaf node
                    int parent_parent = parents[parent];

                    // only need to compute bound when this is a valid leaf node
                    if (parent_parent == -1 || !node_lowers[parent_parent].b) {
                        const int start = parent_lower.i;
                        const int end = parent_upper.i;
                        bounds3 bound;
                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                            const int primitive = primitive_indices[primitive_counter];
                            bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                        }

                        (vec3&)parent_lower = bound.lower;
                        (vec3&)parent_upper = bound.upper;
                    }
                } else {
                    const int left_child = parent_lower.i;
                    const int right_child = parent_upper.i;

                    vec3 left_lower = (vec3&)(node_lowers[left_child]);
                    vec3 left_upper = (vec3&)(node_uppers[left_child]);
                    vec3 right_lower = (vec3&)(node_lowers[right_child]);
                    vec3 right_upper = (vec3&)(node_uppers[right_child]);

                    // union of child bounds
                    vec3 lower = min(left_lower, right_lower);
                    vec3 upper = max(left_upper, right_upper);

                    // write new BVH nodes
                    (vec3&)parent_lower = lower;
                    (vec3&)parent_upper = upper;
                }
                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//
class LinearBVHBuilderGPU {
public:
    LinearBVHBuilderGPU();
    ~LinearBVHBuilderGPU();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(
        BVH& bvh,
        const vec3* item_lowers,
        const vec3* item_uppers,
        int num_items,
        bounds3* total_bounds,
        int* item_groups
    );

private:
    void* stream;

    // temporary data used during building
    int* indices;
    uint64_t* keys;
    int* deltas;
    int* range_lefts;
    int* range_rights;
    int* num_children;

    // bounds data when total item bounds built on GPU
    vec3* total_lower;
    vec3* total_upper;
    vec3* total_inv_edges;
};

////////////////////////////////////////////////////////


template <typename Key>
__global__ void compute_morton_codes(
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers,
    int n,
    const vec3* grid_lower,
    const vec3* grid_inv_edges,
    int* __restrict__ indices,
    Key* __restrict__ keys,
    const int* __restrict__ item_groups
)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        vec3 lower = item_lowers[index];
        vec3 upper = item_uppers[index];

        vec3 center = 0.5f * (lower + upper);

        vec3 local = cw_mul((center - grid_lower[0]), grid_inv_edges[0]);

        // 10-bit Morton codes stored in lower 30bits (1024^3 effective resolution)
        // Group stored in upper 32 bits
        uint64_t morton_code = static_cast<uint64_t>(morton3<1024>(local[0], local[1], local[2]));
        const uint64_t group = item_groups ? static_cast<uint32_t>(item_groups[index]) : 0u;
        const uint64_t key = (group << 32) | morton_code;

        indices[index] = index;
        keys[index] = key;
    }
}

// compute a distance metric between adjacent keys; larger across groups
// Using raw XOR magnitude preserves the original builder's assumptions
__global__ void compute_key_deltas(const uint64_t* __restrict__ keys, int* __restrict__ deltas, int n)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const uint64_t diff = keys[index] ^ keys[index + 1];
        deltas[index] = (diff == 0) ? 64 : __clzll(diff);
    }
}

__global__ void build_leaves(
    const vec3* __restrict__ item_lowers,
    const vec3* __restrict__ item_uppers,
    int n,
    bool compute_bounds,
    const int* __restrict__ indices,
    int* __restrict__ range_lefts,
    int* __restrict__ range_rights,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers
)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bounds3 bounds;
        if (compute_bounds) {
            const int item = indices[index];
            bounds.add_bounds(item_lowers[item], item_uppers[item]);
        }
        lowers[index] = make_node(bounds.lower, index, true);
        uppers[index] = make_node(bounds.upper, index + 1, false);
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
__global__ void build_hierarchy(
    int n,
    int* root,
    const int* __restrict__ deltas,
    const uint64_t* __restrict__ keys,
    int* __restrict__ num_children,
    const int* __restrict__ primitive_indices,
    volatile int* __restrict__ range_lefts,
    volatile int* __restrict__ range_rights,
    volatile int* __restrict__ parents,
    volatile BVHPackedNodeHalf* __restrict__ lowers,
    volatile BVHPackedNodeHalf* __restrict__ uppers
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const int internal_offset = n;

        for (;;) {
            int left = range_lefts[index];
            int right = range_rights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n - 1) {
                *root = index;
                parents[index] = -1;

                break;
            }

            int childCount = 0;

            int parent;

            // group away selection of parent. We need to make sure that:
            // 1) If node spans a single group and exactly one neighbor keeps us in-group, choose that side
            // 2) Otherwise, use the original delta rule with parity tie-break to avoid ladders
            // This is important to make sure that our tree does not mix groups until a single group root is formed.
            bool parent_right = false;
            if (left == 0) {
                parent_right = true;
            } else {
                bool decided = false;
                const uint32_t group_left = (uint32_t)(keys[left] >> 32);
                const uint32_t group_right = (uint32_t)(keys[right] >> 32);

                // Check if either of the neighbors keep us in the same group
                if (group_left == group_right) {
                    const uint32_t group = group_left;
                    const bool can_go_right_same = (right < n - 1) && ((uint32_t)(keys[right + 1] >> 32) == group);
                    const bool can_go_left_same = ((uint32_t)(keys[left - 1] >> 32) == group);

                    // If exactly one neighbor keeps us in the same group, choose that side
                    if (can_go_right_same ^ can_go_left_same) {
                        parent_right = can_go_right_same;
                        decided = true;
                    }

                    // If both or neither keep us in the same group, fall through to delta rule
                }

                if (!decided) {
                    // prefer side with greater common prefix
                    if (right != n - 1 && deltas[right] >= deltas[left - 1]) {
                        if (deltas[right] == deltas[left - 1])
                            parent_right = (primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2);
                        else
                            parent_right = true;
                    } else {
                        parent_right = false;
                    }
                }
            }

            if (parent_right) {
                parent = right + internal_offset;

                // set parent left child
                parents[index] = parent;
                lowers[parent].i = index;
                range_lefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            } else {
                parent = left + internal_offset - 1;

                // set parent right child
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            }

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (childCount == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x, lowers[left_child].y, lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x, uppers[left_child].y, uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x, lowers[right_child].y, lowers[right_child].z);


                vec3 right_upper = vec3(uppers[right_child].x, uppers[right_child].y, uppers[right_child].z);

                // bounds_union of child bounds; ensure min on lowers and max on uppers
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);

                // write new BVH nodes
                make_node(lowers + parent, lower, left_child, false);
                make_node(uppers + parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

// A sorted-position suffix gives coincident Morton keys a unique, deterministic
// ordering without modifying the caller's primitive permutation.
template <typename Key> __device__ inline int radix_prefix(const Key* keys, int n, int a, int b)
{
    if (b < 0 || b >= n)
        return -1;
    const uint64_t diff = keys[a] ^ keys[b];
    return diff ? __clzll(diff) : 64 + __clz(a ^ b);
}

// Determine every internal node independently. Unlike the agglomerative path,
// topology construction does not serialize upward through atomic counters.
// Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (2012).
template <typename Key>
__global__ void build_radix_topology(
    int n,
    const Key* keys,
    int* root,
    int* parents,
    int* range_lefts,
    int* range_rights,
    BVHPackedNodeHalf* lowers,
    BVHPackedNodeHalf* uppers
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1)
        return;
    const int direction = radix_prefix(keys, n, i, i + 1) > radix_prefix(keys, n, i, i - 1) ? 1 : -1;
    const int minimum = radix_prefix(keys, n, i, i - direction);
    int limit = 2;
    while (radix_prefix(keys, n, i, i + limit * direction) > minimum)
        limit *= 2;
    int length = 0;
    for (int step = limit / 2; step; step /= 2) {
        if (radix_prefix(keys, n, i, i + (length + step) * direction) > minimum)
            length += step;
    }
    const int j = i + length * direction;
    const int first = min(i, j);
    const int last = max(i, j);
    const int prefix = radix_prefix(keys, n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) / 2;
        const int candidate = split + step;
        if (candidate < last && radix_prefix(keys, n, first, candidate) > prefix)
            split = candidate;
    } while (step > 1);
    const int left = split == first ? split : n + split;
    const int right = split + 1 == last ? split + 1 : n + split + 1;
    const int node = n + i;
    lowers[node] = make_node(vec3(0.0f), left, false);
    uppers[node] = make_node(vec3(0.0f), right, false);
    parents[left] = node;
    parents[right] = node;
    range_lefts[node] = first;
    range_rights[node] = last;
    if (first == 0 && last == n - 1) {
        parents[node] = -1;
        *root = node;
    }
}

__global__ void set_single_leaf_root(int* root, int* parents)
{
    if (threadIdx.x == 0) {
        *root = 0;
        parents[0] = -1;
    }
}

/*
 * LBVH uses a bottom-up constructor which makes variable-sized leaf nodes more challenging to achieve.
 * Simply splitting the ordered primitives into uniform groups of size leaf_size will result in poor
 * quality. Instead, after the hierarchy is built, we convert any intermediate node whose size is
 * <= leaf_size into a new leaf node. This process is done using the new kernel function called
 * mark_packed_leaf_nodes .
 */
template <typename Key>
__global__ void mark_packed_leaf_nodes(
    int n,
    int num_items,
    bool radix_tree,
    const int* __restrict__ range_lefts,
    const int* __restrict__ range_rights,
    const int* __restrict__ parents,
    const Key* __restrict__ keys,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers,
    const int leaf_size
)
{
    // Original primitive leaves already have final [start, end) metadata.
    int node_index = num_items + blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index < n) {
        // mark the node as leaf if its range is less than leaf_size or it is deeper than BVH_QUERY_STACK_SIZE
        // this will forever mute its child nodes so that they will never be accessed

        int left = range_lefts[node_index];
        // the LBVH constructor's range is defined as left <= i <= right
        // we need to convert it to our convention: left <= i < right
        int right = range_rights[node_index] + 1;

        // avoid creating packed leaves that straddle group boundaries
        bool single_group = true;
        const uint64_t group_left = uint64_t(keys[left]) >> 32;
        const uint64_t group_right = uint64_t(keys[right - 1]) >> 32;
        single_group = (group_left == group_right);

        // Small subtrees will be packed regardless of depth. Avoid chasing their
        // parent chains (most nodes in the tree) just to rediscover that fact.
        int depth = 1;
        // A compressed radix path gains at least one prefix bit per level.
        // Most nodes can therefore prove they are below the stack limit without
        // walking any parents. Equal Morton keys use sorted-position suffixes.
        const bool needs_depth = !radix_tree
            || radix_prefix(keys, num_items, left, right - 1) - radix_prefix(keys, num_items, 0, num_items - 1) + 1
                >= BVH_QUERY_STACK_SIZE;
        if (single_group && right - left > leaf_size && needs_depth) {
            int parent = parents[node_index];
            while (parent != -1 && depth < BVH_QUERY_STACK_SIZE) {
                parent = parents[parent];
                ++depth;
            }
        }
        if (single_group && (right - left <= leaf_size || depth >= BVH_QUERY_STACK_SIZE)) {
            lowers[node_index].b = 1;
            lowers[node_index].i = left;
            uppers[node_index].i = right;
        }
    }
}


CUDA_CALLABLE inline vec3 Vec3Max(const vec3& a, const vec3& b) { return wp::max(a, b); }
CUDA_CALLABLE inline vec3 Vec3Min(const vec3& a, const vec3& b) { return wp::min(a, b); }

__device__ inline void propagate_active_bounds(
    int node, const int* parents, int* child_count, BVHPackedNodeHalf* lowers, BVHPackedNodeHalf* uppers
)
{
    int parent = parents[node];
    const volatile BVHPackedNodeHalf* visible_lowers = lowers;
    const volatile BVHPackedNodeHalf* visible_uppers = uppers;
    while (parent != -1) {
        __threadfence();
        if (atomicAdd(child_count + parent, 1) != 1)
            return;
        const int left = lowers[parent].i;
        const int right = uppers[parent].i;
        // Volatile reads bypass stale per-SM cache lines after the other child
        // publishes its bounds through the fence/counter protocol.
        const vec3 left_lower(visible_lowers[left].x, visible_lowers[left].y, visible_lowers[left].z);
        const vec3 right_lower(visible_lowers[right].x, visible_lowers[right].y, visible_lowers[right].z);
        const vec3 left_upper(visible_uppers[left].x, visible_uppers[left].y, visible_uppers[left].z);
        const vec3 right_upper(visible_uppers[right].x, visible_uppers[right].y, visible_uppers[right].z);
        (vec3&)lowers[parent] = min(left_lower, right_lower);
        (vec3&)uppers[parent] = max(left_upper, right_upper);
        node = parent;
        parent = parents[node];
    }
}

// Start one bounds reduction per maximal packed leaf, rather than per original
// primitive. Muted descendants need no bounds until a later rebuild changes the
// topology; ordinary refit already handles these packed leaves.
__global__ void refit_active_leaves(
    int n,
    const int* parents,
    int* child_count,
    const int* primitive_indices,
    BVHPackedNodeHalf* lowers,
    BVHPackedNodeHalf* uppers,
    const vec3* item_lowers,
    const vec3* item_uppers
)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n || !lowers[node].b)
        return;
    int parent = parents[node];
    if (parent != -1 && lowers[parent].b)
        return;
    bounds3 bounds;
    for (int p = lowers[node].i; p < int(uppers[node].i); ++p) {
        const int item = primitive_indices[p];
        bounds.add_bounds(item_lowers[item], item_uppers[item]);
    }
    (vec3&)lowers[node] = bounds.lower;
    (vec3&)uppers[node] = bounds.upper;
    propagate_active_bounds(node, parents, child_count, lowers, uppers);
}

__global__ void compute_total_bounds(
    const vec3* item_lowers, const vec3* item_uppers, vec3* total_lower, vec3* total_upper, int num_items
)
{
    typedef cub::BlockReduce<vec3, 256> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int blockStart = blockDim.x * blockIdx.x;
    const int numValid = ::min(num_items - blockStart, blockDim.x);

    const int tid = blockStart + threadIdx.x;

    if (tid < num_items) {
        vec3 lower = item_lowers[tid];
        vec3 upper = item_uppers[tid];

        vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);

        // sync threads because second reduce uses same temp storage as first
        __syncthreads();

        vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

        if (threadIdx.x == 0) {
            // write out block results, expanded by the radius
            atomic_max(total_upper, block_upper);
            atomic_min(total_lower, block_lower);
        }
    }
}

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
__global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* total_upper, vec3* total_inv_edges)
{
    vec3 edges = (total_upper[0] - total_lower[0]);
    edges += vec3(0.0001f);

    total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
}


LinearBVHBuilderGPU::LinearBVHBuilderGPU()
    : stream(wp_cuda_stream_get_current())
    , indices(NULL)
    , keys(NULL)
    , deltas(NULL)
    , range_lefts(NULL)
    , range_rights(NULL)
    , num_children(NULL)
    , total_lower(NULL)
    , total_upper(NULL)
    , total_inv_edges(NULL)
{
    total_lower = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_upper = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
    total_inv_edges = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    free_device_on_stream(WP_CURRENT_CONTEXT, total_lower, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, total_upper, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, total_inv_edges, stream);
}


void LinearBVHBuilderGPU::build(
    BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds, int* item_groups
)
{
    // allocate temporary memory used during  building
    indices
        = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items * 2, "(native:bvh)");  // *2 for radix sort
    keys = (uint64_t*)wp_alloc_device(
        WP_CURRENT_CONTEXT, (item_groups ? sizeof(uint64_t) : sizeof(uint32_t)) * num_items * 2, "(native:bvh)"
    );  // *2 for radix sort
    uint32_t* point_keys = reinterpret_cast<uint32_t*>(keys);
    if (item_groups) {
        deltas = (int*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)"
        );  // adjacent key prefixes for the group-aware agglomerative constructor
    }
    range_lefts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
    range_rights = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
    num_children = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");

    // if total bounds supplied by the host then we just
    // compute our edge length and upload it to the GPU directly
    if (total_bounds) {
        // calculate Morton codes
        vec3 edges = (*total_bounds).edges();
        edges += vec3(0.0001f);

        vec3 inv_edges = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);

        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
        wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
    } else {
        // IEEE-754 bit patterns for ± FLT_MAX
        constexpr int FLT_MAX_BITS = 0x7f7fffff;
        constexpr int NEG_FLT_MAX_BITS = 0xff7fffff;

        // total_lower := ( +FLT_MAX, +FLT_MAX, +FLT_MAX )
        wp_launch_device(
            WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_lower, FLT_MAX_BITS, sizeof(vec3) / 4)
        );

        // total_upper := ( -FLT_MAX, -FLT_MAX, -FLT_MAX )
        wp_launch_device(
            WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_upper, NEG_FLT_MAX_BITS, sizeof(vec3) / 4)
        );

        // compute the total bounds on the GPU
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_total_bounds, num_items,
            (item_lowers, item_uppers, total_lower, total_upper, num_items)
        );

        // compute the total edge length
        wp_launch_device(WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges));
    }

    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
    if (item_groups) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
            (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
        );
        radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
    } else {
        wp_launch_device(
            WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
            (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, point_keys, item_groups)
        );
        // Ungrouped Morton codes contain only 30 meaningful bits. Narrow keys
        // halve key-buffer traffic as well as eliminating unused radix passes.
        radix_sort_pairs_device(WP_CURRENT_CONTEXT, point_keys, indices, num_items, 0, 30);
    }

    // Preserve the original row mapping independently of the temporary sort keys.
    wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);

    // calculate deltas between adjacent keys
    if (item_groups) {
        wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items - 1));
    }

    // initialize leaf nodes
    wp_launch_device(
        WP_CURRENT_CONTEXT, build_leaves, num_items,
        (item_lowers, item_uppers, num_items, item_groups != nullptr, indices, range_lefts, range_rights,
         bvh.node_lowers, bvh.node_uppers)
    );

    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during
    // building
    wp_memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int) * bvh.max_nodes);

    if (num_items == 1) {
        wp_launch_device(WP_CURRENT_CONTEXT, set_single_leaf_root, 1, (bvh.root, bvh.node_parents));
    } else if (!item_groups) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, build_radix_topology, num_items - 1,
            (num_items, point_keys, bvh.root, bvh.node_parents, range_lefts, range_rights, bvh.node_lowers,
             bvh.node_uppers)
        );
    } else {
        // Preserve the group-aware topology and the single-leaf root handling.
        wp_launch_device(
            WP_CURRENT_CONTEXT, build_hierarchy, num_items,
            (num_items, bvh.root, deltas, keys, num_children, bvh.primitive_indices, range_lefts, range_rights,
             bvh.node_parents, bvh.node_lowers, bvh.node_uppers)
        );
    }
    if (item_groups) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes - num_items,
            (bvh.max_nodes, num_items, false, range_lefts, range_rights, bvh.node_parents, keys, bvh.node_lowers,
             bvh.node_uppers, bvh.leaf_size)
        );
    } else {
        wp_launch_device(
            WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes - num_items,
            (bvh.max_nodes, num_items, true, range_lefts, range_rights, bvh.node_parents, point_keys, bvh.node_lowers,
             bvh.node_uppers, bvh.leaf_size)
        );
    }

    if (!item_groups) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, refit_active_leaves, bvh.max_nodes,
            (bvh.max_nodes, bvh.node_parents, num_children, bvh.primitive_indices, bvh.node_lowers, bvh.node_uppers,
             item_lowers, item_uppers)
        );
    }

    // free temporary memory
    free_device_on_stream(WP_CURRENT_CONTEXT, indices, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, keys, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, deltas, stream);

    free_device_on_stream(WP_CURRENT_CONTEXT, range_lefts, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, range_rights, stream);
    free_device_on_stream(WP_CURRENT_CONTEXT, num_children, stream);
}

// buffer_size is the number of T, not the number of bytes
template <typename T> T* make_device_buffer_of(void* context, T* host_buffer, size_t buffer_size)
{
    T* device_buffer = (T*)wp_alloc_device(context, sizeof(T) * buffer_size, "(native:bvh)");
    ;
    wp_memcpy_h2d(context, device_buffer, host_buffer, sizeof(T) * buffer_size);

    return device_buffer;
}

void copy_host_tree_to_device(void* context, BVH& bvh_host, BVH& bvh_device_on_host)
{
    // Skip the reorder for cuBQL trees: it scrambles cuBQL's natural
    // contiguous child-pair layout (root at 0, children at offset/offset+1)
    // and hurts ray traversal cache behaviour. Grouped BVHs are already
    // reordered by the builder so they don't need it either.
    if (!bvh_host.item_groups && bvh_host.constructor_type != BVH_CONSTRUCTOR_CUBQL) {
        reorder_top_down_bvh(bvh_host);
    }

    bvh_device_on_host.num_nodes = bvh_host.num_nodes;
    bvh_device_on_host.num_leaf_nodes = bvh_host.num_leaf_nodes;
    bvh_device_on_host.max_nodes = bvh_host.max_nodes;
    bvh_device_on_host.num_items = bvh_host.num_items;
    bvh_device_on_host.max_depth = bvh_host.max_depth;
    bvh_device_on_host.leaf_size = bvh_host.leaf_size;
    bvh_device_on_host.constructor_type = bvh_host.constructor_type;

    bvh_device_on_host.root = (int*)wp_alloc_device(context, sizeof(int), "(native:bvh)");
    wp_memcpy_h2d(context, bvh_device_on_host.root, bvh_host.root, sizeof(int));
    bvh_device_on_host.context = context;

    bvh_device_on_host.node_lowers = make_device_buffer_of(context, bvh_host.node_lowers, bvh_host.max_nodes);
    bvh_device_on_host.node_uppers = make_device_buffer_of(context, bvh_host.node_uppers, bvh_host.max_nodes);
    bvh_device_on_host.node_parents = make_device_buffer_of(context, bvh_host.node_parents, bvh_host.max_nodes);
    bvh_device_on_host.primitive_indices
        = make_device_buffer_of(context, bvh_host.primitive_indices, bvh_host.num_items);
}

// create in-place given existing descriptor
void bvh_create_device(
    void* context,
    vec3* lowers,
    vec3* uppers,
    int num_items,
    int constructor_type,
    int* groups,
    int leaf_size,
    BVH& bvh_device_on_host
)
{
    ContextGuard guard(context);
    if (num_items == 0 && constructor_type != BVH_CONSTRUCTOR_CUBQL) {
        // No nodes or scratch buffers are needed for an empty point cloud.
        // In particular, 2*num_items-1 must not become an allocation size.
        bvh_device_on_host = {};
        bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        bvh_device_on_host.item_groups = groups;
        bvh_device_on_host.leaf_size = leaf_size;
        bvh_device_on_host.constructor_type = constructor_type;
        return;
    }
    if (constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        if (groups) {
            wp::set_error_string("Warp error: grouped BVHs are not supported with cuBQL construction");
            memset(&bvh_device_on_host, 0, sizeof(BVH));
            return;
        }
        cubql_bvh_create_device(context, lowers, uppers, num_items, leaf_size, bvh_device_on_host);
        bvh_device_on_host.constructor_type = constructor_type;
        return;
    }

    if (constructor_type == BVH_CONSTRUCTOR_SAH || constructor_type == BVH_CONSTRUCTOR_MEDIAN)
    // CPU based constructors
    {
        // copy bounds back to CPU
        std::vector<vec3> lowers_host(num_items);
        std::vector<vec3> uppers_host(num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, lowers_host.data(), lowers, sizeof(vec3) * num_items);
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, uppers_host.data(), uppers, sizeof(vec3) * num_items);

        // copy groups back to CPU (if exists)
        std::vector<int> groups_host(num_items);
        int* groups_host_ptr = nullptr;
        if (groups) {
            wp_memcpy_d2h(WP_CURRENT_CONTEXT, groups_host.data(), groups, sizeof(int) * num_items);
            groups_host_ptr = groups_host.data();
        }

        // run CPU based constructor
        wp::BVH bvh_host;
        wp::bvh_create_host(
            lowers_host.data(), uppers_host.data(), num_items, constructor_type, groups_host_ptr, leaf_size, bvh_host
        );

        // copy host tree to device
        wp::copy_host_tree_to_device(WP_CURRENT_CONTEXT, bvh_host, bvh_device_on_host);
        // replace host bounds with device bounds
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        bvh_device_on_host.item_groups = groups;
        bvh_device_on_host.constructor_type = constructor_type;
        // node_counts is not allocated for host tree
        bvh_device_on_host.node_counts
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        wp::bvh_destroy_host(bvh_host);
    } else if (constructor_type == BVH_CONSTRUCTOR_LBVH) {
        bvh_device_on_host.leaf_size = leaf_size;
        bvh_device_on_host.num_items = num_items;
        bvh_device_on_host.num_leaf_nodes = num_items;
        bvh_device_on_host.max_nodes = 2 * num_items - 1;
        bvh_device_on_host.node_lowers = (BVHPackedNodeHalf*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes, "(native:bvh)"
        );
        if (groups) {
            wp_memset_device(
                WP_CURRENT_CONTEXT, bvh_device_on_host.node_lowers, 0,
                sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes
            );
        }
        bvh_device_on_host.node_uppers = (BVHPackedNodeHalf*)wp_alloc_device(
            WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes, "(native:bvh)"
        );
        if (groups) {
            wp_memset_device(
                WP_CURRENT_CONTEXT, bvh_device_on_host.node_uppers, 0,
                sizeof(BVHPackedNodeHalf) * bvh_device_on_host.max_nodes
            );
        }
        bvh_device_on_host.node_parents
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        bvh_device_on_host.node_counts
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh_device_on_host.max_nodes, "(native:bvh)");
        bvh_device_on_host.root = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int), "(native:bvh)");
        bvh_device_on_host.primitive_indices
            = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)");
        bvh_device_on_host.item_lowers = lowers;
        bvh_device_on_host.item_uppers = uppers;
        bvh_device_on_host.item_groups = groups;
        bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();
        bvh_device_on_host.constructor_type = constructor_type;

        LinearBVHBuilderGPU builder;
        builder.build(bvh_device_on_host, lowers, uppers, num_items, NULL, groups);
    } else {
        printf(
            "Unrecognized Constructor type: %d! For GPU constructor it should be SAH (0), Median (1), or LBVH (2)!\n",
            constructor_type
        );
    }
}

void bvh_destroy_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_lowers);
    bvh.node_lowers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_uppers);
    bvh.node_uppers = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_parents);
    bvh.node_parents = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.node_counts);
    bvh.node_counts = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.primitive_indices);
    bvh.primitive_indices = NULL;
    wp_free_device(WP_CURRENT_CONTEXT, bvh.root);
    bvh.root = NULL;
}

void bvh_refit_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.node_counts || !bvh.primitive_indices
        || bvh.num_leaf_nodes <= 0) {
        return;
    }

    // clear child counters
    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);
    wp_launch_device(
        WP_CURRENT_CONTEXT, bvh_refit_kernel, bvh.num_leaf_nodes,
        (bvh.num_leaf_nodes, bvh.node_parents, bvh.node_counts, bvh.primitive_indices, bvh.node_lowers, bvh.node_uppers,
         bvh.item_lowers, bvh.item_uppers)
    );
}

void bvh_rebuild_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);
    if (bvh.num_items == 0)
        return;

    if (bvh.constructor_type == BVH_CONSTRUCTOR_CUBQL) {
        cubql_bvh_rebuild_device(bvh);
        bvh.constructor_type = BVH_CONSTRUCTOR_CUBQL;
        return;
    }

    LinearBVHBuilderGPU builder;
    builder.build(bvh, bvh.item_lowers, bvh.item_uppers, bvh.num_items, NULL, bvh.item_groups);
    bvh.constructor_type = BVH_CONSTRUCTOR_LBVH;
}


// Graph kernel arguments own this descriptor value. A captured HtoD copy from
// the constructor's stack would instead retain a dangling host source address.
__global__ void initialize_bvh_descriptor(BVH* target, BVH value)
{
    if (threadIdx.x == 0)
        *target = value;
}

}  // namespace wp


void wp_bvh_refit_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);

        wp::bvh_refit_device(bvh);
    }
}

void wp_bvh_rebuild_device(uint64_t id)
{
    wp::BVH bvh;
    if (bvh_get_descriptor(id, bvh)) {
        ContextGuard guard(bvh.context);

        // A cuBQL rebuild reallocates the device buffers, so the BVH struct (pointers
        // and sizes) changes; an in-place LBVH rebuild reuses its buffers, leaving the
        // struct unchanged.
        const bool was_cubql = (bvh.constructor_type == BVH_CONSTRUCTOR_CUBQL);

        wp::bvh_rebuild_device(bvh);

        // Refresh the host-side descriptor in case any fields changed.
        wp::bvh_add_descriptor(id, bvh);

        // Only the cuBQL path needs the device-side descriptor refreshed. The host->device
        // copy of pageable memory is not CUDA-graph-capture safe, so we must skip it for the
        // in-place LBVH rebuild to keep grouped/LBVH rebuilds capture-safe (its device struct
        // is unchanged anyway).
        if (was_cubql) {
            wp_memcpy_h2d(WP_CURRENT_CONTEXT, (void*)id, &bvh, sizeof(wp::BVH));
        }
    }
}

/*
 * Since we don't even know the number of true leaf nodes, never mention where they are, we will launch
 * the num_items threads, which are identical to the number of leaf nodes in the original tree. The
 * refitting threads will start from the nodes corresponding to the original leaf nodes, which might be
 * muted. However, the muted leaf nodes will still have the pointer to their parents, thus the up-tracing
 * can still work. We will only compute the bounding box of a leaf node if its parent is not a leaf node.
 */
uint64_t wp_bvh_create_device(
    void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size
)
{
    ContextGuard guard(context);
    wp::BVH bvh_device_on_host;
    wp::BVH* bvh_device_ptr = nullptr;

    wp::bvh_create_device(
        WP_CURRENT_CONTEXT, lowers, uppers, num_items, constructor_type, groups, leaf_size, bvh_device_on_host
    );

    if (!bvh_device_on_host.node_lowers && num_items > 0) {
        return 0;
    }

    // create device-side BVH descriptor
    bvh_device_ptr = (wp::BVH*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::BVH), "(native:bvh)");
    wp_launch_device(WP_CURRENT_CONTEXT, wp::initialize_bvh_descriptor, 1, (bvh_device_ptr, bvh_device_on_host));

    uint64_t bvh_id = (uint64_t)bvh_device_ptr;
    wp::bvh_add_descriptor(bvh_id, bvh_device_on_host);
    return bvh_id;
}


void wp_bvh_destroy_device(uint64_t id)
{
    wp::BVH bvh;
    if (wp::bvh_get_descriptor(id, bvh)) {
        wp::bvh_destroy_device(bvh);
        wp::bvh_rem_descriptor(id);

        // free descriptor
        wp_free_device(WP_CURRENT_CONTEXT, (void*)id);
    }
}
