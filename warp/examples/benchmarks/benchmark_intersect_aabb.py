# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pure Warp box predicates with existing native float32 routines.

cuBQL's full headers require host standard-library headers unavailable to NVRTC.
The cuBQL intersection routines are copied below, with their includes/pragma
removed, and use minimal vector/box adapters. No intersection arithmetic is
rewritten. Keeping the source here makes tests work from installed wheels, which
contain the compiled cuBQL BVH builder but do not ship these source headers.
Warp's AABB and robust ray adapters call intersect.h directly.
Graph timings exclude compilation, transfers, and broad phase.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import warp as wp
import warp.geometry
from warp.examples.benchmarks.benchmark_intersect_seg_tri import benchmark


@wp.func_native("return wp::intersect_aabb_aabb(a, b, lower, upper);")
def native_aabb(a: wp.vec3, b: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


@wp.func_native("""
wp::vec3 d = p1-p0;
wp::vec3 rcp(1.0f/d[0], 1.0f/d[1], 1.0f/d[2]);
float t;
return wp::intersect_ray_aabb_robust(p0, d, rcp, lower, upper, t) && t <= 1.0f;
""")
def native_ray(p0: wp.vec3, p1: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


@wp.func_native("""
return cuBQL::triangles::lineSegmentIntersectsBox(
    cuBQL::box3f(lower, upper), cuBQL::vec3f(p0), cuBQL::vec3f(p1));
""")
def native_seg(p0: wp.vec3, p1: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


@wp.func_native("""
return cuBQL::triangles::triangleIntersectsBox(
    cuBQL::box3f(lower, upper), cuBQL::vec3f(a), cuBQL::vec3f(b), cuBQL::vec3f(c));
""")
def native_tri(a: wp.vec3, b: wp.vec3, c: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool: ...


CUBQL_TYPES = r"""
#define __cubql_both CUDA_CALLABLE
namespace cuBQL {
struct vec3f {
    float x,y,z;
    CUDA_CALLABLE vec3f(float a):x(a),y(a),z(a){}
    CUDA_CALLABLE vec3f(float a,float b,float c):x(a),y(b),z(c){}
    CUDA_CALLABLE vec3f(wp::vec3 v):x(v[0]),y(v[1]),z(v[2]){}
    CUDA_CALLABLE operator wp::vec3() const {return wp::vec3(x,y,z);}
};
CUDA_CALLABLE inline vec3f operator+(vec3f a,vec3f b) {return vec3f(a.x+b.x,a.y+b.y,a.z+b.z);}
CUDA_CALLABLE inline vec3f operator-(vec3f a,vec3f b) {return vec3f(a.x-b.x,a.y-b.y,a.z-b.z);}
CUDA_CALLABLE inline bool operator==(vec3f a,vec3f b) {return a.x==b.x && a.y==b.y && a.z==b.z;}
CUDA_CALLABLE inline float dot(vec3f a,vec3f b) {float r=0.f; r+=a.x*b.x; r+=a.y*b.y; r+=a.z*b.z; return r;}
CUDA_CALLABLE inline vec3f cross(vec3f a,vec3f b) {return vec3f(a.y*b.z-b.y*a.z,a.z*b.x-b.z*a.x,a.x*b.y-b.x*a.y);}
CUDA_CALLABLE inline float min(float a,float b) {return wp::min(a,b);}
CUDA_CALLABLE inline float max(float a,float b) {return wp::max(a,b);}
CUDA_CALLABLE inline float PosInfTy() {return INFINITY;}
CUDA_CALLABLE inline float NegInfTy() {return -INFINITY;}
struct box3f {vec3f lower,upper; CUDA_CALLABLE box3f(wp::vec3 l,wp::vec3 u):lower(l),upper(u){}};
namespace triangles {struct Triangle {vec3f a,b,c;};}
}
"""


# Copied from warp/native/cuBQL/queries/triangleData/math/boxTriangleIntersections.h.
# Original header SHA-256: 82fc7184f5b4e17c508d29ccb2531b4bd4083c8a4512ccab8d65088e4b8ad0c5
# Preserve the upstream license and routine bodies; only includes/pragma removed.
CUBQL_INTERSECTIONS = r"""// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0



namespace cuBQL {
  namespace triangles {
    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! checks if a given triangle A-B-C *intersects* (ie, _not_ only
        "overlaps its bounding box) a given box. Only returns
        true/false, does not produce any actual intersection shape */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               Triangle triangle);
    inline __cubql_both
    bool triangleIntersectsBox(/*! the four vertices of the triangle
                                 to test */
                               Triangle triangle,
                               /*! the query box to check against */
                               box3f queryBox);

    /*! checks if a given triangle A-B-C *intersects* (ie, _not_ only
        "overlaps its bounding box) a given box. Only returns
        true/false, does not produce any actual intersection shape */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               vec3f a, vec3f b, vec3f c);

    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    /*! given a line segment A-B, and a parameter interval [t0,t1]
      along that line (t=0 being at A, and t=1 being at B), clip
      that interval against the given DIM'th dimensional "slab" of
      the given query box. This modifies the [t0,t1] interval in
      place such that after returning from his function [t0,t1] is
      the intersection of the input interval and this dimension's
      slab interval */
    template</*! dimension to perform this clip in */int DIM>
    inline __cubql_both
    void clipLineSegmentToSlabOfBox(/*! the query box to clip
                                      against */
                                    box3f queryBox,
                                    /*! the line segment to
                                      consider */
                                    vec3f a, vec3f b,
                                    /*! the input (and output)
                                      interval along a-b */
                                    float &t0, float &t1)
    {
      float lower_k, upper_k, a_k, b_k;
      if (DIM==0) {
        lower_k = queryBox.lower.x;
        upper_k = queryBox.upper.x;
        a_k = a.x;
        b_k = b.x;
      } else if (DIM==1) {
        lower_k = queryBox.lower.y;
        upper_k = queryBox.upper.y;
        a_k = a.y;
        b_k = b.y;
      } else {
        lower_k = queryBox.lower.z;
        upper_k = queryBox.upper.z;
        a_k = a.z;
        b_k = b.z;
      }

      if (a_k == b_k) {
        /* if a_k == b_k, then the line segment is exactly parallel to
           the dimension we're testing; in this case it's either
           entirely within that slab (iff lower_k <= a_k <= upper_k),
           or entirely outside */
        if (a_k > upper_k || a_k < lower_k) {
          t0 = PosInfTy(); t1 = NegInfTy();
        }
      } else {
        /* line segment is not parallel to the slab planes, so we can
           compute (t_lower,t_upper) distances to the two planes
           bounding this slab ... */
        float t_lower = (lower_k - a_k) / (b_k - a_k);
        float t_upper = (upper_k - a_k) / (b_k - a_k);
        /* ... and then clip this slab-segment against [t0,t1] segment */
        t0 = max(t0,min(t_lower,t_upper));
        t1 = min(t1,max(t_lower,t_upper));
      }
    }


    /*! checks if a given line segment A-B intersects with a box */
    inline __cubql_both
    bool lineSegmentIntersectsBox(/*! the query box to check against */
                                  box3f queryBox,
                                  /*! the end points of the line
                                    segment to test against */
                                  vec3f a,
                                  vec3f b)
    {
      /* initial t-range of segment to clip to box, staring with
         [0,1] */
      float t0 = 0.f;
      float t1 = 1.f;

      /* clip that segment to X-slab of box */
      clipLineSegmentToSlabOfBox<0>(queryBox,a,b,t0,t1);

      /* clip that segment to Y-slab of box */
      clipLineSegmentToSlabOfBox<1>(queryBox,a,b,t0,t1);

      /* clip that segment to Z-slab of box */
      clipLineSegmentToSlabOfBox<2>(queryBox,a,b,t0,t1);

      /* segment intersects if and only if the clipped range is still
         a non-inverted interval */
      return t0 <= t1;
    }



    template<int X, int Y>
    inline __cubql_both
    bool diagonalIntersectsTriangle(/*! the query box being tested */
                                    box3f box,
                                    /* the triangle we're testing against */
                                    vec3f a, vec3f b, vec3f c)
    {
      // compute corners of given diagonal; one on front side and
      // one on back side
      vec3f front(X?box.lower.x:box.upper.x,
                  Y?box.lower.y:box.upper.y,
                  box.lower.z);
      vec3f back(X?box.upper.x:box.lower.x,
                 Y?box.upper.y:box.lower.y,
                 box.upper.z);

      // iw - MIGHT want to subtract 'front' from all of
      // a,b,c,front,and back, to get better numerical accuracy

      // compute plane equation of triangle, and check if front and
      // back points lie on opposing signs of that plane (because if
      // not, it can't intersect)
      vec3f n = cross(b-a,c-a);
      if (n == vec3f(0.f)) {
        return false;
      }

      float p_front = dot(front-a,n);
      float p_back  = dot(back-a,n);
      if (p_front * p_back > 0.f) {
        /* both on same side */
        return false;
      }

      // finally, pluecker test of winding order
      vec3f org = front;
      vec3f dir = back - front;

      auto pluecker=[](vec3f a0, vec3f a1, vec3f b0, vec3f b1)
      { return dot(a1-a0,cross(b1,b0))+dot(b1-b0,cross(a1,a0)); };

      // compute pluecker coordinates dot product of all edges wrt x
      // axis ray. since the ray is mostly 0es and 1es, this should all
      // evaluate to some fairly simple expressions
      float sx = pluecker(org,org+dir,a,b);
      float sy = pluecker(org,org+dir,b,c);
      float sz = pluecker(org,org+dir,c,a);

      // for ray to be inside edges it must have all positive or all
      // negative pluecker winding order
      auto min3=[](float x, float y, float z)
      { return min(min(x,y),z); };
      auto max3=[](float x, float y, float z)
      { return max(max(x,y),z); };

      float lo = min3(sx,sy,sz);
      float hi = max3(sx,sy,sz);
      bool result = (min3(sx,sy,sz) >= 0.f || max3(sx,sy,sz) <= 0.f);
      return result;
    }

    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               vec3f a, vec3f b, vec3f c)
    {
      /*! check if any of the triangle lineSegments is at least partially
        inside the box; if any one is - even partially - the
        triangle is inside the box */
      if (lineSegmentIntersectsBox(queryBox,a,b)) return true;
      if (lineSegmentIntersectsBox(queryBox,b,c)) return true;
      if (lineSegmentIntersectsBox(queryBox,c,a)) return true;

      /*! check the four possible box diagonals on whether they
        intersect the triangle - this catches the case where a
        triangles' edges "surround" the box, but some of the
        triangle's inside is within the box */
      if (diagonalIntersectsTriangle<0,0>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<0,1>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<1,0>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<1,1>(queryBox,a,b,c)) return true;

      /*! if neither of the above two cases occurred, the triangle
        must be outside the box */
      return false;
    }

    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               Triangle triangle)
    { return triangleIntersectsBox(queryBox,triangle.a,triangle.b,triangle.c); }

    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the four vertices of the triangle
                                 to test */
                               Triangle triangle,
                               /*! the query box to check against */
                               box3f queryBox)
    { return triangleIntersectsBox(queryBox,triangle.a,triangle.b,triangle.c); }

  } // ::cuBQL::triangles
} // ::cuBQL

  """


def make_kernel(predicate, arity, cubql=False):
    if arity == 5:

        @wp.kernel(module="unique", enable_backward=False)
        def evaluate(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
            i = wp.tid()
            hits[i] = predicate(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4])
    else:

        @wp.kernel(module="unique", enable_backward=False)
        def evaluate(points: wp.array2d[wp.vec3], hits: wp.array[bool]):
            i = wp.tid()
            hits[i] = predicate(points[i, 0], points[i, 1], points[i, 2], points[i, 3])

    if cubql:
        preamble = CUBQL_TYPES + CUBQL_INTERSECTIONS + "\n#undef __cubql_both\n"
        wp.set_module_options(
            {"extra_build_options": wp.ModuleBuildOptions(extra_cuda_preamble=preamble, extra_cpu_preamble=preamble)},
            module=evaluate.module,
        )
    return evaluate


def make_workloads(count, kind):
    rng = np.random.default_rng(20261029)
    arity = 5 if kind == "tri" else 4
    random = rng.uniform(-1, 1, (count, arity, 3)).astype(np.float32)
    random[:, -2:] = np.sort(random[:, -2:], axis=1)
    if kind == "aabb":
        random[:, :2] = np.sort(random[:, :2], axis=1)
    separated = random.copy()
    separated[:, -2:, 0] += 4
    contained = random.copy()
    contained[:, -2] = -2
    contained[:, -1] = 2
    small = random.copy()
    small[:, -1] = small[:, -2] + 0.05
    contact = random.copy()
    contact[:, :, 2] = 0
    return {"random": random, "separated": separated, "contained": contained, "small_boxes": small, "coplanar": contact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=262144)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("outputs/intersect_aabb_benchmark.json"))
    args = parser.parse_args()
    if min(args.pairs, args.repeats, args.samples) < 1:
        parser.error("--pairs, --repeats, and --samples must be positive")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("CUDA is required for the event/graph benchmark")
    predicates = {
        "aabb": {"pure": wp.geometry.intersect_aabb_aabb, "native": native_aabb},
        "seg": {"pure": wp.geometry.intersect_seg_aabb, "cubql": native_seg, "robust_ray": native_ray},
        "tri": {"pure": wp.geometry.intersect_tri_aabb, "cubql": native_tri},
    }
    results = {}
    for kind, routines in predicates.items():
        kernels = {name: make_kernel(f, 5 if kind == "tri" else 4, name == "cubql") for name, f in routines.items()}
        for workload, points in make_workloads(args.pairs, kind).items():
            result = benchmark(points, kernels, device, args.repeats, args.samples)
            results[f"{kind}/{workload}"] = result
            for name, stats in result.items():
                print(
                    f"{kind}/{workload}/{name}: {stats['median_ms'] * 1000:.3f} us; "
                    f"{stats['disagreements_with_pure']} disagreements; pure speedup {stats['speedup_of_pure']:.3f}x"
                )
    report = {
        "device": device.name,
        "warp_version": wp.__version__,
        "precision": "float32",
        "pairs": args.pairs,
        "repeats": args.repeats,
        "samples": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
