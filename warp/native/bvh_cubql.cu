// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device-side cuBQL integration: build a cuBQL tree on the GPU using cuBQL's
// gpuBuilder, then convert the result into Warp's native BVH layout. The
// matching host-side code lives in bvh_cubql.cpp.

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "warp.h"

#include "bvh.h"
#include "cuda_util.h"
#include "error.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define THRUST_IGNORE_CUB_VERSION_CHECK

// CUB must be included before cuBQL. cuBQL's math/common.h includes <stdexcept>,
// which causes CCCL's _CCCL_HAS_EXCEPTIONS() to be true when typeid.h is later
// pulled in by CUB. This makes __throw_out_of_range non-constexpr, breaking a
// static_assert in typeid.h on GCC < 12 (which lacks P2448R2 relaxed constexpr).
#include <cub/cub.cuh>

#ifndef WP_DISABLE_CUBQL
#include "cuBQL/bvh.h"

// Builder implementation headers depend on definitions emitted by bvh.h.
#include "cuBQL/builder/cuda/radix.h"
#include "cuBQL/builder/cuda/rebinMortonBuilder.h"
#include "cuBQL/queries/pointData/knn.h"
#endif

extern CUcontext get_current_context();

namespace wp {

#ifndef WP_DISABLE_CUBQL

struct alignas(16) CubqlNode {
    vec3 lower;
    vec3 upper;
    uint64_t admin = 0;
};

__global__ void cubql_make_boxes(const vec3* lowers, const vec3* uppers, cuBQL::box3f* boxes, int n)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        const vec3 lower = lowers[tid];
        const vec3 upper = uppers[tid];
        boxes[tid]
            = cuBQL::box3f(cuBQL::vec3f(lower[0], lower[1], lower[2]), cuBQL::vec3f(upper[0], upper[1], upper[2]));
    }
}

// Fast device-side copy of cuBQL nodes into the warp packed-node layout.
// Preserves cuBQL's natural ordering so child pairs remain contiguous.
__global__ void cubql_copy_nodes_to_native(
    const CubqlNode* __restrict__ cubql_nodes,
    int num_nodes,
    BVHPackedNodeHalf* __restrict__ node_lowers,
    BVHPackedNodeHalf* __restrict__ node_uppers,
    int* __restrict__ node_parents
)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_nodes) {
        return;
    }

    const CubqlNode node = cubql_nodes[tid];
    const uint16_t leaf_count = uint16_t(node.admin >> 48);
    const uint64_t offset = node.admin & 0x0000FFFFFFFFFFFFull;
    const bool is_leaf = leaf_count != 0;
    const uint64_t upper_offset = is_leaf ? offset + leaf_count : offset + 1;

    node_lowers[tid] = make_node(node.lower, int(offset), is_leaf);
    node_uppers[tid] = make_node(node.upper, int(upper_offset), false);

    if (!is_leaf) {
        node_parents[offset] = tid;
        node_parents[offset + 1] = tid;
    }
}

struct CubqlNativeBuild {
    std::vector<BVHPackedNodeHalf> node_lowers;
    std::vector<BVHPackedNodeHalf> node_uppers;
    std::vector<int> node_parents;
    std::vector<int> primitive_indices;
    int num_leaf_nodes = 0;
    int max_depth = 0;
};

static uint16_t cubql_leaf_count(uint64_t admin) { return uint16_t(admin >> 48); }

static uint64_t cubql_admin_offset(uint64_t admin) { return admin & 0x0000FFFFFFFFFFFFull; }

static bool cubql_decode_node(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    uint32_t num_prims,
    uint32_t node_index,
    bool& is_leaf,
    uint64_t& offset,
    uint64_t& upper_offset
)
{
    if (node_index >= num_nodes) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid node index");
        return false;
    }

    const CubqlNode& node = nodes[node_index];
    const uint16_t leaf_count = cubql_leaf_count(node.admin);
    offset = cubql_admin_offset(node.admin);
    is_leaf = leaf_count != 0;
    upper_offset = is_leaf ? offset + leaf_count : offset + 1;

    if (offset > uint64_t(INT_MAX) || upper_offset > uint64_t(INT_MAX)) {
        wp::set_error_string("Warp error: cuBQL BVH node offset is too large to convert to the native BVH layout");
        return false;
    }

    if (is_leaf && upper_offset > num_prims) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid primitive range");
        return false;
    }

    if (!is_leaf && offset + 1 >= num_nodes) {
        wp::set_error_string("Warp error: cuBQL BVH contains an invalid child node offset");
        return false;
    }

    return true;
}

static bool cubql_append_subtree_primitives(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    const uint32_t* prim_ids,
    uint32_t num_prims,
    uint32_t node_index,
    std::vector<int>& primitive_indices
)
{
    std::vector<uint32_t> stack;
    stack.push_back(node_index);

    while (!stack.empty()) {
        const uint32_t current_node = stack.back();
        stack.pop_back();

        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, current_node, is_leaf, offset, upper_offset)) {
            return false;
        }

        if (is_leaf) {
            for (uint64_t i = offset; i < upper_offset; ++i) {
                if (prim_ids[i] > uint32_t(INT_MAX)) {
                    wp::set_error_string("Warp error: cuBQL BVH primitive index is too large to convert");
                    return false;
                }
                primitive_indices.push_back(int(prim_ids[i]));
            }
        } else {
            stack.push_back(uint32_t(offset + 1));
            stack.push_back(uint32_t(offset));
        }
    }

    return true;
}

// Walk the cuBQL tree once to determine whether its depth fits within the
// native traversal stack. If it does, we can copy the tree verbatim and keep
// cuBQL's contiguous child-pair layout — which is much friendlier to ray
// traversal cache behaviour than the warp builder's reordered layout.
static bool
cubql_tree_exceeds_native_stack(const CubqlNode* nodes, uint32_t num_nodes, uint32_t num_prims, bool& exceeds_stack)
{
    struct StackEntry {
        uint32_t node_index;
        int depth;
    };

    exceeds_stack = false;

    std::vector<StackEntry> stack;
    stack.push_back({ 0, 0 });

    while (!stack.empty()) {
        const StackEntry entry = stack.back();
        stack.pop_back();

        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, entry.node_index, is_leaf, offset, upper_offset)) {
            return false;
        }

        if (!is_leaf) {
            if (entry.depth >= BVH_QUERY_STACK_SIZE - 1) {
                exceeds_stack = true;
                return true;
            }
            stack.push_back({ uint32_t(offset + 1), entry.depth + 1 });
            stack.push_back({ uint32_t(offset), entry.depth + 1 });
        }
    }

    return true;
}

// Fast-path: copy a cuBQL tree (decoded from device nodes already memcpy'd to
// host) into a warp BVH host descriptor while preserving the cuBQL layout.
static bool cubql_copy_native_order_host_bvh(
    BVH& bvh, const CubqlNode* nodes, const uint32_t* prim_ids, uint32_t num_nodes, uint32_t num_prims
)
{
    bvh.num_nodes = int(num_nodes);
    bvh.num_leaf_nodes = 0;
    bvh.max_depth = 0;
    bvh.max_nodes = int(num_nodes);

    bvh.node_lowers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * num_nodes, "(native:bvh)"));
    bvh.node_uppers
        = static_cast<BVHPackedNodeHalf*>(wp_alloc_host(sizeof(BVHPackedNodeHalf) * num_nodes, "(native:bvh)"));
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * num_nodes, "(native:bvh)"));
    bvh.node_counts = nullptr;
    bvh.primitive_indices = static_cast<int*>(wp_alloc_host(sizeof(int) * num_prims, "(native:bvh)"));
    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.primitive_indices || !bvh.root) {
        wp::set_error_string("Warp error: failed to allocate native BVH storage for cuBQL conversion");
        bvh_destroy_host(bvh);
        return false;
    }

    std::fill(bvh.node_parents, bvh.node_parents + num_nodes, -1);
    bvh.root[0] = 0;

    for (uint32_t i = 0; i < num_prims; ++i) {
        if (prim_ids[i] > uint32_t(INT_MAX)) {
            wp::set_error_string("Warp error: cuBQL BVH primitive index is too large to convert");
            bvh_destroy_host(bvh);
            return false;
        }
        bvh.primitive_indices[i] = int(prim_ids[i]);
    }

    for (uint32_t i = 0; i < num_nodes; ++i) {
        bool is_leaf;
        uint64_t offset;
        uint64_t upper_offset;
        if (!cubql_decode_node(nodes, num_nodes, num_prims, i, is_leaf, offset, upper_offset)) {
            bvh_destroy_host(bvh);
            return false;
        }

        bvh.node_lowers[i] = make_node(nodes[i].lower, int(offset), is_leaf);
        bvh.node_uppers[i] = make_node(nodes[i].upper, int(upper_offset), false);

        if (is_leaf) {
            bvh.num_leaf_nodes++;
        } else {
            bvh.node_parents[offset] = int(i);
            bvh.node_parents[offset + 1] = int(i);
        }
    }

    std::vector<uint32_t> depth_stack;
    std::vector<int> depths;
    depth_stack.push_back(0);
    depths.push_back(0);

    while (!depth_stack.empty()) {
        const uint32_t node_index = depth_stack.back();
        const int depth = depths.back();
        depth_stack.pop_back();
        depths.pop_back();

        bvh.max_depth = std_max(bvh.max_depth, depth);

        if (!bvh.node_lowers[node_index].b) {
            depth_stack.push_back(uint32_t(bvh.node_lowers[node_index].i));
            depths.push_back(depth + 1);
            depth_stack.push_back(uint32_t(bvh.node_uppers[node_index].i));
            depths.push_back(depth + 1);
        }
    }

    return true;
}

static bool cubql_build_native_node(
    const CubqlNode* nodes,
    uint32_t num_nodes,
    const uint32_t* prim_ids,
    uint32_t num_prims,
    uint32_t cubql_node_index,
    int depth,
    int parent,
    CubqlNativeBuild& build,
    int& native_node_index
)
{
    bool is_leaf;
    uint64_t offset;
    uint64_t upper_offset;
    if (!cubql_decode_node(nodes, num_nodes, num_prims, cubql_node_index, is_leaf, offset, upper_offset)) {
        return false;
    }

    const CubqlNode& node = nodes[cubql_node_index];
    native_node_index = int(build.node_lowers.size());
    build.node_lowers.push_back({});
    build.node_uppers.push_back({});
    build.node_parents.push_back(parent);
    build.max_depth = std_max(build.max_depth, depth);

    // Native traversal uses a fixed-size stack.  If cuBQL creates a deeper
    // subtree, pack all primitives below this point into one native leaf.
    // The generic two-child-push traversal needs depth+1 stack entries, so
    // anything at depth >= BVH_QUERY_STACK_SIZE - 1 would overflow.
    if (is_leaf || depth >= BVH_QUERY_STACK_SIZE - 1) {
        const int primitive_start = int(build.primitive_indices.size());
        if (!cubql_append_subtree_primitives(
                nodes, num_nodes, prim_ids, num_prims, cubql_node_index, build.primitive_indices
            )) {
            return false;
        }
        const int primitive_end = int(build.primitive_indices.size());

        build.node_lowers[native_node_index] = make_node(node.lower, primitive_start, true);
        build.node_uppers[native_node_index] = make_node(node.upper, primitive_end, false);
        build.num_leaf_nodes++;
        return true;
    }

    int left_child;
    int right_child;
    if (!cubql_build_native_node(
            nodes, num_nodes, prim_ids, num_prims, uint32_t(offset), depth + 1, native_node_index, build, left_child
        )
        || !cubql_build_native_node(
            nodes, num_nodes, prim_ids, num_prims, uint32_t(offset + 1), depth + 1, native_node_index, build,
            right_child
        )) {
        return false;
    }

    build.node_lowers[native_node_index] = make_node(node.lower, left_child, false);
    build.node_uppers[native_node_index] = make_node(node.upper, right_child, false);
    return true;
}

static bool cubql_copy_to_native_host_bvh(
    BVH& bvh, const CubqlNode* nodes, const uint32_t* prim_ids, uint32_t num_nodes, uint32_t num_prims
)
{
    bvh.num_nodes = 0;
    bvh.num_leaf_nodes = 0;
    bvh.max_depth = 0;
    bvh.max_nodes = 0;

    if (num_nodes == 0 || num_prims == 0) {
        return true;
    }

    bool exceeds_stack = false;
    if (!cubql_tree_exceeds_native_stack(nodes, num_nodes, num_prims, exceeds_stack)) {
        return false;
    }
    if (!exceeds_stack) {
        return cubql_copy_native_order_host_bvh(bvh, nodes, prim_ids, num_nodes, num_prims);
    }

    CubqlNativeBuild build;
    build.node_lowers.reserve(num_nodes);
    build.node_uppers.reserve(num_nodes);
    build.node_parents.reserve(num_nodes);
    build.primitive_indices.reserve(num_prims);

    int root = -1;
    if (!cubql_build_native_node(nodes, num_nodes, prim_ids, num_prims, 0, 0, -1, build, root)) {
        return false;
    }

    if (root != 0 || build.primitive_indices.size() != num_prims) {
        wp::set_error_string("Warp error: cuBQL BVH conversion produced an invalid native BVH");
        return false;
    }

    bvh.num_nodes = int(build.node_lowers.size());
    bvh.num_leaf_nodes = build.num_leaf_nodes;
    bvh.max_depth = build.max_depth;
    bvh.max_nodes = bvh.num_nodes;

    bvh.node_lowers = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * build.node_lowers.size(), "(native:bvh)")
    );
    bvh.node_uppers = static_cast<BVHPackedNodeHalf*>(
        wp_alloc_host(sizeof(BVHPackedNodeHalf) * build.node_uppers.size(), "(native:bvh)")
    );
    bvh.node_parents = static_cast<int*>(wp_alloc_host(sizeof(int) * build.node_parents.size(), "(native:bvh)"));
    bvh.node_counts = nullptr;
    bvh.primitive_indices
        = static_cast<int*>(wp_alloc_host(sizeof(int) * build.primitive_indices.size(), "(native:bvh)"));
    bvh.root = static_cast<int*>(wp_alloc_host(sizeof(int), "(native:bvh)"));

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.primitive_indices || !bvh.root) {
        wp::set_error_string("Warp error: failed to allocate native BVH storage for cuBQL conversion");
        bvh_destroy_host(bvh);
        return false;
    }

    bvh.root[0] = root;

    std::copy(build.node_lowers.begin(), build.node_lowers.end(), bvh.node_lowers);
    std::copy(build.node_uppers.begin(), build.node_uppers.end(), bvh.node_uppers);
    std::copy(build.node_parents.begin(), build.node_parents.end(), bvh.node_parents);
    std::copy(build.primitive_indices.begin(), build.primitive_indices.end(), bvh.primitive_indices);

    return true;
}

// Fast-path device copy: allocate warp-layout device buffers and translate the
// cuBQL nodes directly with a CUDA kernel, keeping cuBQL's child-pair layout.
// Used when the tree fits within BVH_QUERY_STACK_SIZE.
static inline bool cubql_copy_native_order_device(BVH& bvh, const cuBQL::bvh3f& native)
{
    bvh.num_nodes = int(native.numNodes);
    bvh.num_leaf_nodes = int(native.numNodes);  // launch refit over every node; non-leaves filter themselves
    bvh.max_nodes = int(native.numNodes);
    bvh.max_depth = 0;

    bvh.node_lowers = (BVHPackedNodeHalf*)wp_alloc_device(
        WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * native.numNodes, "(native:bvh)"
    );
    bvh.node_uppers = (BVHPackedNodeHalf*)wp_alloc_device(
        WP_CURRENT_CONTEXT, sizeof(BVHPackedNodeHalf) * native.numNodes, "(native:bvh)"
    );
    bvh.node_parents = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * native.numNodes, "(native:bvh)");
    bvh.node_counts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * native.numNodes, "(native:bvh)");
    bvh.primitive_indices = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * native.numPrims, "(native:bvh)");
    bvh.root = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int), "(native:bvh)");

    if (!bvh.node_lowers || !bvh.node_uppers || !bvh.node_parents || !bvh.node_counts || !bvh.primitive_indices
        || !bvh.root) {
        wp::set_error_string("Warp error: failed to allocate native BVH storage for cuBQL conversion");
        bvh_destroy_device(bvh);
        return false;
    }

    int root_index = 0;
    wp_memcpy_h2d(WP_CURRENT_CONTEXT, bvh.root, &root_index, sizeof(int));
    wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, native.primIDs, sizeof(int) * native.numPrims);
    // Fill node_parents with -1 (all 0xFF bytes) — used as a sentinel for "no parent assigned"
    // before the copy kernel writes the real parent indices.
    wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_parents, 0xFF, sizeof(int) * native.numNodes);
    wp_launch_device(
        WP_CURRENT_CONTEXT, cubql_copy_nodes_to_native, native.numNodes,
        (reinterpret_cast<const CubqlNode*>(native.nodes), int(native.numNodes), bvh.node_lowers, bvh.node_uppers,
         bvh.node_parents)
    );
    return true;
}

static inline bool cubql_copy_to_native_device(BVH& bvh, const cuBQL::bvh3f& native)
{
    static_assert(
        sizeof(CubqlNode) == sizeof(cuBQL::bvh3f::node_t), "CubqlNode must match cuBQL::BinaryBVH<float, 3>::Node"
    );

    if (native.numNodes > uint32_t(INT_MAX) || native.numPrims > uint32_t(INT_MAX)) {
        wp::set_error_string("Warp error: cuBQL BVH is too large to convert to the native BVH layout");
        return false;
    }

    if (native.numNodes == 0 || native.numPrims == 0) {
        bvh.num_nodes = 0;
        bvh.num_leaf_nodes = 0;
        bvh.max_nodes = 0;
        bvh.max_depth = 0;
        return true;
    }

    if (native.numPrims != uint32_t(bvh.num_items)) {
        wp::set_error_string("Warp error: cuBQL BVH primitive count does not match the native BVH item count");
        return false;
    }

    std::vector<CubqlNode> nodes_host(native.numNodes);
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, nodes_host.data(), native.nodes, sizeof(CubqlNode) * native.numNodes);

    bool exceeds_stack = false;
    if (!cubql_tree_exceeds_native_stack(nodes_host.data(), native.numNodes, native.numPrims, exceeds_stack)) {
        return false;
    }

    if (!exceeds_stack) {
        return cubql_copy_native_order_device(bvh, native);
    }

    std::vector<uint32_t> prim_ids_host(native.numPrims);
    wp_memcpy_d2h(WP_CURRENT_CONTEXT, prim_ids_host.data(), native.primIDs, sizeof(uint32_t) * native.numPrims);

    BVH bvh_host;
    memset(&bvh_host, 0, sizeof(BVH));
    bvh_host.num_items = bvh.num_items;
    bvh_host.leaf_size = bvh.leaf_size;
    bvh_host.constructor_type = bvh.constructor_type;
    if (!cubql_copy_to_native_host_bvh(
            bvh_host, nodes_host.data(), prim_ids_host.data(), native.numNodes, native.numPrims
        )) {
        bvh_destroy_host(bvh_host);
        return false;
    }

    void* context = bvh.context;
    vec3* item_lowers = bvh.item_lowers;
    vec3* item_uppers = bvh.item_uppers;
    int* item_groups = bvh.item_groups;
    const int num_items = bvh.num_items;
    const int leaf_size = bvh.leaf_size;
    const int constructor_type = bvh.constructor_type;

    copy_host_tree_to_device(context, bvh_host, bvh);
    bvh.item_lowers = item_lowers;
    bvh.item_uppers = item_uppers;
    bvh.item_groups = item_groups;
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;
    bvh.constructor_type = constructor_type;
    bvh.context = context;
    // Without the reorder, leaves are scattered through the array; refit must
    // launch over all nodes (non-leaf threads filter themselves out).
    bvh.num_leaf_nodes = bvh.num_nodes;
    bvh.node_counts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");

    bvh_destroy_host(bvh_host);
    return true;
}

static cuBQL::GpuMemoryResource& cubql_get_mem_resource()
{
    int ordinal = wp_cuda_context_get_device_ordinal(wp_cuda_context_get_current());
    if (wp_cuda_device_is_mempool_supported(ordinal))
        return cuBQL::defaultGpuMemResource();
    static cuBQL::DeviceMemoryResource sync_resource;
    return sync_resource;
}

void cubql_bvh_create_device(
    void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh_device_on_host
)
{
    ContextGuard guard(context);
    memset(&bvh_device_on_host, 0, sizeof(BVH));

    bvh_device_on_host.context = context ? context : wp_cuda_context_get_current();
    bvh_device_on_host.item_lowers = lowers;
    bvh_device_on_host.item_uppers = uppers;
    bvh_device_on_host.num_items = num_items;
    bvh_device_on_host.leaf_size = leaf_size;
    bvh_device_on_host.constructor_type = BVH_CONSTRUCTOR_CUBQL;

    if (num_items <= 0) {
        return;
    }

    cuBQL::box3f* boxes = reinterpret_cast<cuBQL::box3f*>(
        wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(cuBQL::box3f) * num_items, "(native:bvh)")
    );
    cudaStream_t current_stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    wp_launch_device(WP_CURRENT_CONTEXT, cubql_make_boxes, num_items, (lowers, uppers, boxes, num_items));

    auto free_partial_bvh = [&]() { wp::bvh_destroy_device(bvh_device_on_host); };

    cuBQL::bvh3f native {};
    try {
        cuBQL::BuildConfig build_config;
        build_config.enableSAH();
        build_config.makeLeafThreshold = leaf_size;
        cuBQL::gpuBuilder(native, boxes, uint32_t(num_items), build_config, current_stream, cubql_get_mem_resource());
        if (!cubql_copy_to_native_device(bvh_device_on_host, native)) {
            free_partial_bvh();
        }
    } catch (const std::exception& e) {
        wp::set_error_string("Warp error: cuBQL BVH build failed: %s", e.what());
        free_partial_bvh();
    } catch (...) {
        wp::set_error_string("Warp error: cuBQL BVH build failed: unknown exception");
        free_partial_bvh();
    }

    if (native.nodes || native.primIDs) {
        try {
            cuBQL::cuda::free(native, current_stream, cubql_get_mem_resource());
        } catch (const std::exception& e) {
            wp::set_error_string("Warp error: cuBQL BVH free failed: %s", e.what());
            free_partial_bvh();
        } catch (...) {
            wp::set_error_string("Warp error: cuBQL BVH free failed: unknown exception");
            free_partial_bvh();
        }
    }
    free_device_on_stream(WP_CURRENT_CONTEXT, boxes, current_stream);
}

void cubql_bvh_destroy_device(BVH& bvh) { bvh_destroy_device(bvh); }

bool cubql_bvh_refit_device(BVH& bvh)
{
    if (!bvh.root || bvh.num_nodes == 0) {
        return true;
    }

    bvh_refit_device(bvh);
    return true;
}

void cubql_bvh_rebuild_device(BVH& bvh)
{
    ContextGuard guard(bvh.context);

    void* context = bvh.context;
    vec3* lowers = bvh.item_lowers;
    vec3* uppers = bvh.item_uppers;
    const int num_items = bvh.num_items;
    const int leaf_size = bvh.leaf_size;

    bvh_destroy_device(bvh);
    cubql_bvh_create_device(context, lowers, uppers, num_items, leaf_size, bvh);
}

#else  // WP_DISABLE_CUBQL

void cubql_bvh_create_device(
    void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh_device_on_host
)
{
    wp::set_error_string("Warp error: cuBQL support disabled (WP_DISABLE_CUBQL)");
    memset(&bvh_device_on_host, 0, sizeof(BVH));
    bvh_device_on_host.constructor_type = BVH_CONSTRUCTOR_CUBQL;
}

void cubql_bvh_destroy_device(BVH&) { }
bool cubql_bvh_refit_device(BVH&) { return true; }
void cubql_bvh_rebuild_device(BVH&) { }

#endif  // WP_DISABLE_CUBQL

}  // namespace wp

#ifndef WP_DISABLE_CUBQL
namespace {

// Use the caller's pool policy; cuBQL's default resource changes its release threshold.
struct PointQueryMemoryResource final : cuBQL::GpuMemoryResource {
    void malloc(void** pointer, size_t size, cudaStream_t stream) override
    {
        const cudaError_t status = cudaMallocAsync(pointer, size, stream);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }
    void free(void* pointer, cudaStream_t stream) override
    {
        const cudaError_t status = cudaFreeAsync(pointer, stream);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }
};

struct CubqlPointIndex {
    void* context;
    cuBQL::bvh3f tree {};
    PointQueryMemoryResource memory;
};

template <int Capacity, int Stride>
__global__ void cubql_point_query_kernel(
    cuBQL::bvh3f tree,
    const cuBQL::vec3f* points,
    const cuBQL::vec3f* queries,
    int count,
    int k,
    void* indices,
    float* distances,
    int* counts,
    bool indices_int64,
    bool squared
)
{
    const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread % Stride || thread / Stride >= size_t(count))
        return;
    const int row = int(thread / Stride);
    const cuBQL::vec3f query = queries[row];
    cuBQL::knn::Candidate candidates[Capacity];
    int found = 0;
    // Keep exact-capacity specializations for the measured k=1/11/32 regimes.
    const int requested = Capacity == 64 ? k : Capacity;
    if (tree.numPrims && isfinite(query.x) && isfinite(query.y) && isfinite(query.z))
        found = cuBQL::points::findKNN(candidates, requested, tree, points, query).numFound;
    counts[row] = found;
    for (int neighbor = 0; neighbor < requested; ++neighbor) {
        const size_t output = size_t(row) * requested + neighbor;
        const int index = neighbor < found ? candidates[neighbor].primID : -1;
        if (indices_int64)
            static_cast<int64_t*>(indices)[output] = index;
        else
            static_cast<int*>(indices)[output] = index;
        float distance = INFINITY;
        if (index >= 0) {
            // Differentiate the same Euclidean distance that is returned to callers.
            const cuBQL::vec3f delta = query - points[index];
            distance = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            if (!squared)
                distance = sqrtf(distance);
        }
        distances[output] = distance;
    }
}

template <int Capacity>
void cubql_launch_point_query(
    const CubqlPointIndex& index,
    const wp::vec3* points,
    const wp::vec3* queries,
    int count,
    int k,
    int stride,
    int block_dim,
    void* indices,
    float* distances,
    int* counts,
    bool indices_int64,
    bool squared,
    cudaStream_t stream
)
{
    const unsigned blocks = unsigned((size_t(count) * stride + block_dim - 1) / block_dim);
#define WP_CUBQL_QUERY(Stride) \
    case Stride: \
        cubql_point_query_kernel<Capacity, Stride><<<blocks, block_dim, 0, stream>>>( \
            index.tree, reinterpret_cast<const cuBQL::vec3f*>(points), reinterpret_cast<const cuBQL::vec3f*>(queries), \
            count, k, indices, distances, counts, indices_int64, squared); \
        break
    switch (stride) {
        WP_CUBQL_QUERY(1);
        WP_CUBQL_QUERY(2);
        WP_CUBQL_QUERY(4);
        WP_CUBQL_QUERY(8);
        WP_CUBQL_QUERY(16);
        WP_CUBQL_QUERY(32);
    }
#undef WP_CUBQL_QUERY
}

}  // namespace
#endif

uint64_t
wp_point_query_cubql_create_device(void* context, const wp::vec3* points, int count, int builder, int leaf_size)
{
#ifndef WP_DISABLE_CUBQL
    ContextGuard guard(context);
    const cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_context_get_stream(context));
    cudaStreamCaptureStatus capture_status;
    if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess
        || capture_status != cudaStreamCaptureStatusNone) {
        wp::set_error_string("Direct cuBQL construction cannot be captured; build the index before capture");
        return 0;
    }
    if (count < 0 || (count && !points) || builder < 0 || builder > 2 || leaf_size < 0) {
        wp::set_error_string("Invalid direct cuBQL construction arguments");
        return 0;
    }
    auto* index = new CubqlPointIndex { context };
    cuBQL::box3f* boxes = nullptr;
    try {
        if (count) {
            index->memory.malloc(reinterpret_cast<void**>(&boxes), sizeof(cuBQL::box3f) * count, stream);
            wp::cubql_make_boxes<<<(count + 255) / 256, 256, 0, stream>>>(points, points, boxes, count);
            cuBQL::BuildConfig config(leaf_size);
            if (builder == 0)
                cuBQL::gpuBuilder(index->tree, boxes, count, config.enableELH(), stream, index->memory);
            else if (builder == 1)
                cuBQL::cuda::radixBuilder(index->tree, boxes, count, config, stream, index->memory);
            else
                cuBQL::cuda::rebinRadixBuilder(index->tree, boxes, count, config, stream, index->memory);
            index->memory.free(boxes, stream);
            boxes = nullptr;
        }
        return reinterpret_cast<uint64_t>(index);
    } catch (const std::exception& error) {
        wp::set_error_string("Direct cuBQL construction failed: %s", error.what());
    } catch (...) {
        wp::set_error_string("Direct cuBQL construction failed: unknown exception");
    }
    // Preserve the original diagnostic if cleanup encounters another CUDA error.
    cudaFreeAsync(boxes, stream);
    cudaFreeAsync(index->tree.nodes, stream);
    cudaFreeAsync(index->tree.primIDs, stream);
    delete index;
#else
    wp::set_error_string("Direct cuBQL point queries are disabled in this build");
#endif
    return 0;
}

void wp_point_query_cubql_destroy_device(uint64_t id)
{
#ifndef WP_DISABLE_CUBQL
    if (!id)
        return;
    auto* index = reinterpret_cast<CubqlPointIndex*>(id);
    ContextGuard guard(index->context);
    // Owners must outlive launches and graph replay, as with ordinary Warp BVHs.
    // Async allocations require an explicit wait before cudaFree. Destruction
    // is outside the reusable query path and cannot occur during capture.
    const cudaError_t wait_status = cudaDeviceSynchronize();
    const cudaError_t nodes_status = cudaFree(index->tree.nodes);
    const cudaError_t indices_status = cudaFree(index->tree.primIDs);
    if (wait_status != cudaSuccess || nodes_status != cudaSuccess || indices_status != cudaSuccess)
        wp::set_error_string("Failed to retire direct cuBQL point index storage");
    delete index;
#endif
}

bool wp_point_query_cubql_query_device(
    uint64_t id,
    const wp::vec3* points,
    const wp::vec3* queries,
    int count,
    int k,
    int stride,
    int block_dim,
    void* indices,
    float* distances,
    int* counts,
    bool indices_int64,
    bool squared
)
{
#ifndef WP_DISABLE_CUBQL
    if (!id || count < 0 || k < 1 || k > 64 || stride < 1 || stride > 32 || (stride & (stride - 1)) || block_dim < 32
        || block_dim > 256 || block_dim % 32 || (count && (!queries || !indices || !distances || !counts))) {
        wp::set_error_string("Invalid direct cuBQL query arguments");
        return false;
    }
    if (!count)
        return true;
    const auto& index = *reinterpret_cast<const CubqlPointIndex*>(id);
    ContextGuard guard(index.context);
    const cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_context_get_stream(index.context));
#define WP_CUBQL_LAUNCH(Capacity) \
    cubql_launch_point_query<Capacity>(index, points, queries, count, k, stride, block_dim, indices, distances, counts, \
        indices_int64, squared, stream)
    if (k == 1) {
        WP_CUBQL_LAUNCH(1);
    } else if (k == 11) {
        WP_CUBQL_LAUNCH(11);
    } else if (k == 32) {
        WP_CUBQL_LAUNCH(32);
    } else {
        WP_CUBQL_LAUNCH(64);
    }
#undef WP_CUBQL_LAUNCH
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        return true;
    wp::set_error_string("Direct cuBQL query launch failed: %s", cudaGetErrorString(status));
#else
    wp::set_error_string("Direct cuBQL point queries are disabled in this build");
#endif
    return false;
}
