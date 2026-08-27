# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

This module provides GPU-accelerated topology analysis of triangle meshes,
computed without vertex positions: combinatorial topology statistics (edge
incidence and orientation, vertex manifoldness, and degeneracies) and
edge-connected component labeling.

.. figure:: /img/warp_geometry_connected_components.gif
   :align: center
   :width: 80%

   The 50 edge-connected components of the libigl *truck* mesh (2,956 vertices,
   4,770 triangles) labeled by :func:`connected_components` -- the body is a
   single component while each wheel, tire, and the spare are separate. Rendered
   with `polyscope <https://polyscope.run>`_ using one color per component,
   orbiting the up axis.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import TriangleMeshTopologyStatistics as TriangleMeshTopologyStatistics
from warp._src.geometry import triangle_mesh_topology_statistics as triangle_mesh_topology_statistics
from warp._src.geometry import connected_components as connected_components
