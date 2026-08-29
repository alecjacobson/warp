# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations for triangle meshes.

This module provides GPU-accelerated surface sampling on triangle meshes:

* **Uniform sampling** (:func:`uniformly_sample`, :class:`UniformSampler`) draws
  points spread evenly across a mesh regardless of how finely it is tessellated,
  and the device function :func:`draw` samples a single point from within your
  own :func:`warp.kernel` definitions.
* **Poisson-disk sampling** (:func:`poisson_disk_sample`,
  :class:`PoissonDiskSampler`) draws blue-noise point sets in which no two
  samples are closer than a given radius, using the parallel algorithm of
  Bowers et al. (SIGGRAPH Asia 2010). :func:`pair_correlation` measures the
  resulting spectrum on the surface.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import MeshSample as MeshSample
from warp._src.geometry import PoissonDiskSampler as PoissonDiskSampler
from warp._src.geometry import UniformSampler as UniformSampler
from warp._src.geometry import UniformSamplerState as UniformSamplerState
from warp._src.geometry import draw as draw
from warp._src.geometry import pair_correlation as pair_correlation
from warp._src.geometry import poisson_disk_sample as poisson_disk_sample
from warp._src.geometry import sample_barycentrics as sample_barycentrics
from warp._src.geometry import uniformly_sample as uniformly_sample
