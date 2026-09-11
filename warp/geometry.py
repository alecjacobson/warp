# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations.

This module provides functions to process 2D and 3D geometry and their
associated topological data structures (e.g., meshes), GPU-accelerated
isosurface extraction, and surface sampling.

Dense-grid isosurface backends take a 3-D ``wp.float32`` field sampled at grid
nodes and share the :class:`IsoSurfaceBase` interface, so they can be swapped
without changing calling code: :class:`IsoSurfaceMarchingCubes` produces
triangles.

Sparse extraction skips the dense grid entirely. :func:`sparse_marching_cubes`
takes an implicit function and builds a Lipschitz octree around the level set,
so cost scales with surface area rather than volume. :func:`lipschitz_octree`
and :func:`sparse_marching_cubes_from_cells` expose its two stages separately.

Array-level functions such as :func:`swept_volume_mesh` launch kernels over a whole
mesh or grid. Device functions such as :func:`swept_volume_sdf` evaluate a
single point and may be called from within your own :func:`warp.kernel`
definitions.

Surface sampling draws points on triangle meshes:

* **Uniform sampling** (:func:`uniformly_sample`, :class:`UniformSampler`) draws
  points spread evenly across a mesh regardless of how finely it is tessellated,
  and the device function :func:`draw` samples a single point from within your
  own :func:`warp.kernel` definitions.
* **Poisson-disk sampling** (:func:`poisson_disk_sample`,
  :class:`PoissonDiskSampler`) draws blue-noise point sets in which no two
  samples are closer than a given radius, with an optional on-surface distance
  metric (``geodesic=True``). :func:`pair_correlation` measures the resulting
  blue-noise spectrum on the surface.

The Poisson-disk sampler and its spectrum analysis implement Bowers, Wang, Wei
and Maletz, "Parallel Poisson Disk Sampling with Spectrum Analysis on Surfaces"
(ACM Transactions on Graphics 29(6), SIGGRAPH Asia 2010; doi:10.1145/1882261.1866188).

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry.iso_surface import IsoSurfaceBase as IsoSurfaceBase
from warp._src.geometry.marching_cubes import IsoSurfaceMarchingCubes as IsoSurfaceMarchingCubes
from warp._src.geometry.sparse_marching_cubes import (
    lipschitz_octree as lipschitz_octree,
    sparse_marching_cubes as sparse_marching_cubes,
    sparse_marching_cubes_from_cells as sparse_marching_cubes_from_cells,
)
from warp._src.geometry import SweptVolumeSignMode as SweptVolumeSignMode
from warp._src.geometry import delaunay_edge_flip as delaunay_edge_flip
from warp._src.geometry import find_triangle_neighbor_edge_index as find_triangle_neighbor_edge_index
from warp._src.geometry import swept_volume_bounds as swept_volume_bounds
from warp._src.geometry import swept_volume_field as swept_volume_field
from warp._src.geometry import swept_volume_mesh as swept_volume_mesh
from warp._src.geometry import swept_volume_sdf as swept_volume_sdf

# Don't expose these quite yet in case we want to change the naming conventions.
# from warp._src.geometry import in_circle as in_circle
# from warp._src.geometry import signed_area as signed_area
from warp._src.geometry import tri_tri_adjacency as tri_tri_adjacency

from warp._src.geometry import MeshSample as MeshSample
from warp._src.geometry import PoissonDiskSampler as PoissonDiskSampler
from warp._src.geometry import UniformSampler as UniformSampler
from warp._src.geometry import UniformSamplerState as UniformSamplerState
from warp._src.geometry import draw as draw
from warp._src.geometry import pair_correlation as pair_correlation
from warp._src.geometry import poisson_disk_sample as poisson_disk_sample
from warp._src.geometry import sample_barycentrics as sample_barycentrics
from warp._src.geometry import uniformly_sample as uniformly_sample
