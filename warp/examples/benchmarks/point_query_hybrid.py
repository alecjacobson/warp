# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental combinations of Morton layout, seeding, and candidate selection."""

import functools
from pathlib import Path

import torch

import warp as wp
from warp.examples.benchmarks.point_query_common import query_codes


@wp.func_native("""
    const wp::BVH tree = wp::bvh_get(bvh);
    reordered.data[p] = points.data[tree.primitive_indices[p]];
""")
def gather_point(bvh: wp.uint64, points: wp.array[wp.vec3], reordered: wp.array[wp.vec3], p: int): ...


@wp.kernel(enable_backward=False)
def gather_points(bvh: wp.uint64, points: wp.array[wp.vec3], reordered: wp.array[wp.vec3]):
    gather_point(bvh, points, reordered, wp.tid())


@wp.func_native("""
    auto tree=wp::bvh_get(bvh);
    auto lo=tree.node_lowers[*tree.root],hi=tree.node_uppers[*tree.root];
    wp::vec3 lower(lo.x,lo.y,lo.z),upper(hi.x,hi.y,hi.z);
    wp::vec3 edges=upper-lower;
    float inv=1.0f/(wp::max(wp::max(edges[0],edges[1]),edges[2])+0.0001f);
    wp::vec3 u=(points.data[tree.primitive_indices[p]]-lower)*inv;
    codes.data[p]=wp::morton3<1024>(u[0],u[1],u[2]);
""")
def gather_code(bvh: wp.uint64, points: wp.array[wp.vec3], codes: wp.array[int], p: int): ...


@wp.kernel(enable_backward=False)
def gather_codes(bvh: wp.uint64, points: wp.array[wp.vec3], codes: wp.array[int]):
    gather_code(bvh, points, codes, wp.tid())


@functools.cache
def make_hybrid(k, packed=False, seed=False, selector="heap", diagnostics=False):
    if diagnostics and k < 3:
        raise ValueError("Diagnostic output requires k >= 3")
    if selector not in ("heap", "sorted") or seed not in (False, True, "morton", "carry", "cached"):
        raise ValueError("Unknown experimental selector or seeding strategy")
    # Reuse the production traversal so experiments differ only in named factors.
    source = (Path(wp.__file__).parent / "native/point_query.h").read_text()
    traversal = source.split("    if (!bvh.num_items)\n", 1)[1].split("\ntemplate <int K>", 1)[0]
    traversal = "    if (!bvh.num_items)\n" + traversal
    traversal = traversal.rsplit("}", 1)[0]
    traversal = traversal.replace("STACK_SIZE", "32")
    traversal = traversal.replace("selected = Selector();", "selected = Selected();")
    traversal = traversal.replace("float threshold = selected.bound();", "float threshold = seed_bound;")
    traversal = traversal.replace("threshold = selected.bound();", "threshold = wp::min(seed_bound, selected.bound());")
    traversal = traversal.replace(
        "bvh_load_vec3(points.data, i)",
        "bvh_load_vec3(packed_points.data, p)" if packed else "bvh_load_vec3(points.data, i)",
    )
    traversal = traversal.replace("                        return;", "                        return;")
    traversal = traversal.replace("i == exclude", "i == exclude || i >= points.shape[0]")
    if diagnostics:
        traversal = traversal.replace("for (;;) {", "for (;;) { ++visits;")
        traversal = traversal.replace("const float d = point_distance_sq", "++tests; const float d = point_distance_sq")
        traversal = traversal.replace("selected.insert(d, i);", "++adds; selected.insert(d, i);")
    # Place traversal in a local callable to preserve early-return behavior.
    setup = f"using Selected = wp::point_query_t<{k}>;"
    if selector == "sorted":
        setup = f"""
        struct Selected : wp::point_query_t<{k}> {{
            CUDA_CALLABLE float bound() const {{ return this->count == {k} ? this->distances[{k - 1}] : INFINITY; }}
            CUDA_CALLABLE void insert(float d, int id) {{
                int pos = this->count;
                if (pos == {k}) {{
                    if (!wp::point_pair_less(d,id,this->distances[pos-1],this->indices[pos-1])) return;
                    --pos;
                }} else ++this->count;
                while (pos > 0 && wp::point_pair_less(d,id,this->distances[pos-1],this->indices[pos-1])) {{
                    this->distances[pos] = this->distances[pos-1];
                    this->indices[pos] = this->indices[pos-1];
                    --pos;
                }}
                this->distances[pos] = d; this->indices[pos] = id;
            }}
            CUDA_CALLABLE void finish() {{}}
        }};
        """
    seed_code = ""
    if seed:
        seed_code = f"""
        int node = *bvh.root;
        auto lo = wp::bvh_load_node(bvh.node_lowers,node);
        auto hi = wp::bvh_load_node(bvh.node_uppers,node);
        while (!lo.b) {{
            const float dl = wp::point_box_distance_sq(bvh,lo.i,q);
            const float dr = wp::point_box_distance_sq(bvh,hi.i,q);
            node = dr < dl ? hi.i : lo.i;
            lo = wp::bvh_load_node(bvh.node_lowers,node);
            hi = wp::bvh_load_node(bvh.node_uppers,node);
        }}
        Selected nearby;
        const int width = wp::min({2 * k},bvh.num_items);
        const int start = wp::clamp(int(lo.i)-{k},0,bvh.num_items-width);
        for (int p=start;p<start+width;++p) {{
            int i = bvh.primitive_indices[p];
            if (i == exclude || i >= points.shape[0]) continue;
            const float d = wp::point_distance_sq(q,{"packed_points.data[p]" if packed else "points.data[i]"});
            if (d < cutoff_sq) nearby.insert(d,i);
        }}
        seed_bound = nearby.bound();
        """
    if seed in ("morton", "carry", "cached"):
        a = seed_code.index("        int node = *bvh.root;")
        b = seed_code.index("        Selected nearby;")
        seed_code = (
            seed_code[:a]
            + r"""
        auto lo = wp::bvh_load_node(bvh.node_lowers,*bvh.root);
        auto hi = wp::bvh_load_node(bvh.node_uppers,*bvh.root);
        const wp::vec3 lower(lo.x,lo.y,lo.z), upper(hi.x,hi.y,hi.z);
        wp::vec3 edges=upper-lower;
        const float inv=1.0f/(wp::max(wp::max(edges[0],edges[1]),edges[2])+0.0001f);
        auto code = [&](wp::vec3 p) {
            wp::vec3 u=(p-lower)*inv;
            return wp::morton3<1024>(u[0],u[1],u[2]);
        };
        const auto key=code(q);
        int first=0,last=bvh.num_items;
        while(first<last) {
            int middle=first+(last-first)/2;
            auto midkey=code(points.data[bvh.primitive_indices[middle]]);
            if(midkey < key) first=middle+1;
            else last=middle;
        }
        """
            + seed_code[b:]
        )
        seed_code = seed_code.replace("int(lo.i)", "first")
    if seed in ("carry", "cached"):
        seed_code = seed_code.replace("Selected nearby;", "")
        seed_code = seed_code.replace("nearby.insert", "selected.insert").replace("nearby.bound()", "selected.bound()")
        seed_code = seed_code.replace("for (int p=start;", "seed_first=start; seed_last=start+width; for (int p=start;")
        traversal = traversal.replace("if (i == exclude", "if ((p >= seed_first && p < seed_last) || i == exclude")
    if seed == "cached":
        seed_code = seed_code.replace(
            "code(points.data[bvh.primitive_indices[middle]])", "wp::bvh_load_int(sorted_codes.data,middle)"
        )
    snippet = f"""
    using namespace wp;
    {setup}
    int visits=0, tests=0, adds=0, seed_first=0, seed_last=0;
    Selected selected;
    const BVH bvh = bvh_get(bvh_id);
    const float cutoff_sq = point_query_cutoff_sq(cutoff);
    const int exclude = -1;
    const vec3 q = queries.data[row];
    float seed_bound = INFINITY;
    if (bvh.num_items) {{ {seed_code} }}
    auto traverse = [&]() {{ {traversal} }};
    traverse();
    selected.finish();
    for (int r=0;r<{k};++r) {{
        wp::index(ids,row,r) = r < selected.count ? selected.indices[r] : -1;
        wp::index(distances,row,r) = r < selected.count ? sqrtf(selected.distances[r]) : INFINITY;
    }}
    """

    if diagnostics:
        snippet += "wp::index(ids,row,0)=visits; wp::index(ids,row,1)=tests; wp::index(ids,row,2)=adds;"

    @wp.func_native(snippet)
    def native(
        bvh_id: wp.uint64,
        points: wp.array[wp.vec3],
        packed_points: wp.array[wp.vec3],
        sorted_codes: wp.array[int],
        queries: wp.array[wp.vec3],
        cutoff: float,
        row: int,
        ids: wp.array2d[wp.int64],
        distances: wp.array2d[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def query(
        bvh: wp.uint64,
        points: wp.array[wp.vec3],
        packed_points: wp.array[wp.vec3],
        sorted_codes: wp.array[int],
        queries: wp.array[wp.vec3],
        order: wp.array[int],
        ids: wp.array2d[wp.int64],
        distances: wp.array2d[float],
    ):
        native(bvh, points, packed_points, sorted_codes, queries, wp.inf, order[wp.tid()], ids, distances)

    return query


class Hybrid:
    """Experimental finite, nonempty clouds without cutoff/exclusion/gradient support."""

    def __init__(
        self, points, queries, k, leaf=8, ordered=True, packed=False, seed=False, selector="heap", isotropic=False
    ):
        if isotropic and isinstance(seed, str):
            raise ValueError("Sentinel padding supports only nearest-leaf seeding in this experiment")
        self.cached = seed == "cached"
        self.isotropic = isotropic
        self.p, self.q, self.k = points, queries, k
        self.points, self.queries = wp.from_torch(points, dtype=wp.vec3), wp.from_torch(queries, dtype=wp.vec3)
        self.leaf, self.ordered, self.packed = leaf, ordered, packed
        self.kernel = make_hybrid(k, packed, seed, selector)

    def build(self):
        build_points = self.points
        if self.isotropic:
            lower, upper = self.p.amin(dim=0), self.p.amax(dim=0)
            center, extent = (lower + upper) * 0.5, (upper - lower).max()
            padded = torch.cat((self.p, (center - extent * 0.5)[None], (center + extent * 0.5)[None]))
            build_points = wp.from_torch(padded, dtype=wp.vec3)
        tree = wp.Bvh(build_points, build_points, constructor="lbvh", leaf_size=self.leaf)
        packed = self.points
        if self.packed:
            packed = wp.empty_like(build_points)
            wp.launch(gather_points, len(build_points), inputs=[tree.id, build_points, packed])
        codes = wp.empty(len(build_points) if self.cached else 1, dtype=int)
        if self.cached:
            wp.launch(gather_codes, len(build_points), inputs=[tree.id, build_points, codes])
        return tree, packed, codes

    def search(self, index):
        tree, packed, codes = index
        n = len(self.q)
        ids = torch.empty((n, self.k), dtype=torch.int64, device=self.q.device)
        distances = torch.empty((n, self.k), dtype=torch.float32, device=self.q.device)
        if self.ordered:
            keys, order = wp.empty(2 * n, dtype=int), wp.empty(2 * n, dtype=int)
            wp.launch(query_codes, n, inputs=[tree.id, self.queries, keys, order])
            wp.utils.radix_sort_pairs(keys, order, n)
        else:
            order = wp.from_torch(torch.arange(n, dtype=torch.int32, device=self.q.device))
        wp.launch(
            self.kernel,
            n,
            inputs=[
                tree.id,
                self.points,
                packed,
                codes,
                self.queries,
                order,
                wp.from_torch(ids),
                wp.from_torch(distances),
            ],
        )
        return ids, distances

    def one_shot(self):
        index = self.build()
        return index, self.search(index)
