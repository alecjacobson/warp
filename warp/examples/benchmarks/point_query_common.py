# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact exhaustive baselines and kernel factories for point-query experiments."""

import functools

import warp as wp


@functools.cache
def make_api_query(k, euclidean64=False):
    index_type = wp.int64 if euclidean64 else wp.int32

    @wp.kernel(module="unique", enable_backward=False)
    def api(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        cutoff: float,
        excludes: wp.array[int],
        indices: wp.array2d[index_type],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ):
        t = wp.tid()
        search = wp.point_query_knn(bvh, points, queries[t], cutoff, wp.static(k), excludes[t])
        i = int(-1)
        count = int(0)
        while wp.point_query_next(search, i):
            indices[t, count] = index_type(i)
            d = wp.length_sq(queries[t] - points[i])
            if wp.static(euclidean64):
                d = wp.sqrt(d)
            distances[t, count] = d
            count += 1
        counts[t] = count
        for r in range(count, wp.static(k)):
            indices[t, r] = index_type(-1)
            distances[t, r] = wp.inf

    return api


@functools.cache
def make_query(k, method="bvh", stack_size=32):
    # Native snippets let all baselines use exactly the production distance and
    # pair ordering. These helpers are benchmark-only, not public scalar APIs.
    selector = f"wp::point_query_t<{k}>"
    setup = ""
    if method.endswith("sorted"):
        setup = f"""
        struct Sorted : wp::point_query_t<{k}> {{
            CUDA_CALLABLE float bound() const {{ return this->count == {k} ? this->distances[{k - 1}] : INFINITY; }}
            CUDA_CALLABLE void insert(float d, int id) {{
                int pos = this->count;
                if (pos == {k}) {{
                    if (!wp::point_pair_less(d, id, this->distances[pos-1], this->indices[pos-1])) return;
                    --pos;
                }} else ++this->count;
                while (pos > 0 && wp::point_pair_less(d, id, this->distances[pos-1], this->indices[pos-1])) {{
                    this->distances[pos] = this->distances[pos-1];
                    this->indices[pos] = this->indices[pos-1];
                    --pos;
                }}
                this->distances[pos] = d; this->indices[pos] = id;
            }}
            CUDA_CALLABLE void finish() {{}}
        }};
        """
        selector = "Sorted"
    if method.startswith("bvh"):
        search = f"wp::point_query_traverse<{stack_size}>(wp::bvh_get(bvh), points, q, r2, exclude, selected);"
    elif method.startswith("brute"):
        search = "wp::point_query_scan(points, q, r2, exclude, selected);"
    elif method == "sphere":
        search = f"""
        const wp::BVH tree = wp::bvh_get(bvh);
        if (tree.num_items) {{
            const auto lo = wp::bvh_load_node(tree.node_lowers, *tree.root);
            const auto hi = wp::bvh_load_node(tree.node_uppers, *tree.root);
            const wp::vec3 low(lo.x, lo.y, lo.z), high(hi.x, hi.y, hi.z);
            const wp::vec3 far = wp::max(wp::abs(q-low), wp::abs(q-high));
            float cover = sqrtf(wp::point_distance_sq(far, wp::vec3(0.0f))) * 1.00001f + 1.e-6f;
            float radius = wp::max(sqrtf(wp::point_distance_sq(low, high)) * 0.01f, 1.e-6f);
            radius = wp::min(radius, max_dist);
            for (;;) {{
                selected = {selector}();
                auto region = wp::bvh_query_sphere(bvh, q, radius, -1);
                int id;
                while (wp::bvh_query_sphere_next(region, id, INFINITY)) {{
                    float d = wp::point_distance_sq(q, points.data[id]);
                    if (id != exclude && d < r2) selected.insert(d, id);
                }}
                if (selected.count == {k} || radius >= max_dist || radius >= cover) break;
                radius = wp::min(wp::min(radius * 2.0f, cover), max_dist);
            }}
        }}
        """
    else:
        raise ValueError(method)
    snippet = (
        setup
        + f"""
    {selector} selected;
    const float r2 = wp::point_query_cutoff_sq(max_dist);
    if (wp::point_query_valid_input(q, max_dist)) {{ {search} }}
    selected.finish();
    counts.data[row] = selected.count;
    for (int r=0; r<{k}; ++r) {{
        wp::index(indices, row, r) = r < selected.count ? selected.indices[r] : -1;
        wp::index(distances, row, r) = r < selected.count ? selected.distances[r] : INFINITY;
    }}
    """
    )

    @wp.func_native(snippet)
    def native_query(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        q: wp.vec3,
        max_dist: float,
        exclude: int,
        row: int,
        indices: wp.array2d[int],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def query(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        max_dist: float,
        excludes: wp.array[int],
        indices: wp.array2d[int],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ):
        t = wp.tid()
        native_query(bvh, points, queries[t], max_dist, excludes[t], t, indices, distances, counts)

    return query


@wp.func_native("return wp::point_distance_sq(a,b);")
def distance_sq(a: wp.vec3, b: wp.vec3) -> float: ...


@wp.func
def masked_distance(p: wp.vec3, id: int, q: wp.vec3, n: int, exclude: int, cutoff: float):
    d = distance_sq(q, p)
    if id >= n or id == exclude or not cutoff > 0.0 or not wp.isfinite(q) or not d < cutoff * cutoff:
        d = wp.inf
    return d


@wp.func
def tied_id(d: float, id: int, best: float):
    if d == best and wp.isfinite(d):
        return id
    return 2147483647


@wp.func
def remove_selected(d: float, id: int, best_id: int):
    if id == best_id:
        return wp.inf
    return d


@functools.cache
def make_tiled(k, bp=128):
    vf = wp.types.vector(length=k, dtype=float)
    vi = wp.types.vector(length=k, dtype=int)

    @wp.kernel(module="unique", enable_backward=False)
    def tiled(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        cutoff: float,
        excludes: wp.array[int],
        partitions: int,
        partial_i: wp.array3d[int],
        partial_d: wp.array3d[float],
    ):
        qi, partition, lane = wp.tid()
        n = points.shape[0]
        chunks = (n + wp.static(bp) - 1) // wp.static(bp)
        first = chunks * partition // partitions
        last = chunks * (partition + 1) // partitions
        best_d = vf(wp.inf)
        best_i = vi(2147483647)
        for chunk in range(first, last):
            positions = wp.tile_load(points, shape=wp.static(bp), offset=chunk * wp.static(bp))
            ids = wp.tile_map(wp.add, wp.tile_arange(wp.static(bp), dtype=int), chunk * wp.static(bp))
            ds = wp.tile_map(masked_distance, positions, ids, queries[qi], n, excludes[qi], cutoff)
            for _r in range(wp.static(k)):
                d = wp.tile_extract(wp.tile_min(ds), 0)
                i = wp.tile_extract(wp.tile_min(wp.tile_map(tied_id, ds, ids, d)), 0)
                ds = wp.tile_map(remove_selected, ds, ids, i)
                if lane == 0 and i != 2147483647:
                    j = int(wp.static(k - 1))
                    if d < best_d[j] or (d == best_d[j] and i < best_i[j]):
                        while j > 0 and (d < best_d[j - 1] or (d == best_d[j - 1] and i < best_i[j - 1])):
                            best_d[j] = best_d[j - 1]
                            best_i[j] = best_i[j - 1]
                            j -= 1
                        best_d[j] = d
                        best_i[j] = i
        if lane == 0:
            for r in range(wp.static(k)):
                i = best_i[r]
                if i == 2147483647:
                    i = -1
                partial_i[qi, partition, r] = i
                partial_d[qi, partition, r] = best_d[r]

    snippet = f"""
    wp::point_query_t<{k}> selected;
    for (int p=0; p<partial_i.shape[1]; ++p) {{
        for (int r=0; r<{k}; ++r) {{
            const int id = wp::index(partial_i, row, p, r);
            if (id >= 0) selected.insert(wp::index(partial_d, row, p, r), id);
        }}
    }}
    selected.finish();
    counts.data[row] = selected.count;
    for (int r=0; r<{k}; ++r) {{
        wp::index(indices,row,r) = r < selected.count ? selected.indices[r] : -1;
        wp::index(distances,row,r) = r < selected.count ? selected.distances[r] : INFINITY;
    }}
    """

    @wp.func_native(snippet)
    def native_merge(
        row: int,
        partial_i: wp.array3d[int],
        partial_d: wp.array3d[float],
        indices: wp.array2d[int],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def merge(
        partial_i: wp.array3d[int],
        partial_d: wp.array3d[float],
        indices: wp.array2d[int],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ):
        native_merge(wp.tid(), partial_i, partial_d, indices, distances, counts)

    return tiled, merge


@wp.func_native("""
    const wp::BVH tree = wp::bvh_get(bvh);
    if (!tree.num_items || !wp::isfinite(q)) return 0;
    const auto lo = wp::bvh_load_node(tree.node_lowers, *tree.root);
    const auto hi = wp::bvh_load_node(tree.node_uppers, *tree.root);
    const wp::vec3 lower(lo.x,lo.y,lo.z), upper(hi.x,hi.y,hi.z);
    wp::vec3 extent = upper-lower;
    for (int a=0;a<3;++a) if (!(extent[a]>0.0f)) extent[a]=1.0f;
    const wp::vec3 unit = wp::cw_div(q-lower,extent);
    return static_cast<int>(wp::morton3<1024>(unit[0],unit[1],unit[2]));
""")
def query_morton_code(bvh: wp.uint64, q: wp.vec3) -> int: ...


@wp.kernel(enable_backward=False)
def query_codes(bvh: wp.uint64, queries: wp.array[wp.vec3], keys: wp.array[int], order: wp.array[int]):
    i = wp.tid()
    keys[i] = query_morton_code(bvh, queries[i])
    order[i] = i


@functools.cache
def make_ordered_query(k, euclidean64=False):
    index_type = wp.int64 if euclidean64 else wp.int32

    @wp.kernel(module="unique", enable_backward=False)
    def ordered(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        cutoff: float,
        excludes: wp.array[int],
        order: wp.array[int],
        indices: wp.array2d[index_type],
        distances: wp.array2d[float],
        counts: wp.array[int],
    ):
        t = order[wp.tid()]
        search = wp.point_query_knn(bvh, points, queries[t], cutoff, wp.static(k), excludes[t])
        i = int(-1)
        count = int(0)
        while wp.point_query_next(search, i):
            indices[t, count] = index_type(i)
            d = wp.length_sq(queries[t] - points[i])
            if wp.static(euclidean64):
                d = wp.sqrt(d)
            distances[t, count] = d
            count += 1
        counts[t] = count
        for r in range(count, wp.static(k)):
            indices[t, r] = index_type(-1)
            distances[t, r] = wp.inf

    return ordered


@functools.cache
def make_tiled_batched(k, bp=128, bq=4):
    """Reuse each reference tile across Bq queries; only lane zero retains top-k."""
    mf = wp.types.matrix(shape=(bq, k), dtype=float)
    mi = wp.types.matrix(shape=(bq, k), dtype=int)

    @wp.kernel(module="unique", enable_backward=False)
    def tiled_batched(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        cutoff: float,
        excludes: wp.array[int],
        partitions: int,
        partial_i: wp.array3d[int],
        partial_d: wp.array3d[float],
    ):
        group, partition, lane = wp.tid()
        n = points.shape[0]
        chunks = (n + wp.static(bp) - 1) // wp.static(bp)
        first = chunks * partition // partitions
        last = chunks * (partition + 1) // partitions
        best_d = mf(wp.inf)
        best_i = mi(2147483647)
        for chunk in range(first, last):
            positions = wp.tile_load(points, shape=wp.static(bp), offset=chunk * wp.static(bp))
            ids = wp.tile_map(wp.add, wp.tile_arange(wp.static(bp), dtype=int), chunk * wp.static(bp))
            for slot in range(wp.static(bq)):
                qi = group * wp.static(bq) + slot
                query = wp.vec3(0.0)
                excluded = int(-1)
                radius = float(0.0)
                if qi < queries.shape[0]:
                    query = queries[qi]
                    excluded = excludes[qi]
                    radius = cutoff
                ds = wp.tile_map(masked_distance, positions, ids, query, n, excluded, radius)
                for _r in range(wp.static(k)):
                    d = wp.tile_extract(wp.tile_min(ds), 0)
                    i = wp.tile_extract(wp.tile_min(wp.tile_map(tied_id, ds, ids, d)), 0)
                    ds = wp.tile_map(remove_selected, ds, ids, i)
                    if lane == 0 and i != 2147483647:
                        j = int(wp.static(k - 1))
                        if d < best_d[slot, j] or (d == best_d[slot, j] and i < best_i[slot, j]):
                            while j > 0 and (
                                d < best_d[slot, j - 1] or (d == best_d[slot, j - 1] and i < best_i[slot, j - 1])
                            ):
                                best_d[slot, j] = best_d[slot, j - 1]
                                best_i[slot, j] = best_i[slot, j - 1]
                                j -= 1
                            best_d[slot, j] = d
                            best_i[slot, j] = i
        if lane == 0:
            for slot in range(wp.static(bq)):
                qi = group * wp.static(bq) + slot
                if qi < queries.shape[0]:
                    for r in range(wp.static(k)):
                        i = best_i[slot, r]
                        if i == 2147483647:
                            i = -1
                        partial_i[qi, partition, r] = i
                        partial_d[qi, partition, r] = best_d[slot, r]

    return tiled_batched, make_tiled(k, bp)[1]
