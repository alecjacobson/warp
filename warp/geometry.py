# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

This module provides GPU-accelerated geometry operations on triangle meshes,
currently in-place Delaunay edge flipping for 2D meshes and the
triangle-triangle adjacency structure it builds on.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import delaunay_edge_flip as delaunay_edge_flip
from warp._src.geometry import find_adjacent_triangle as find_adjacent_triangle

# Don't expose these quite yet in case we want to change the naming conventions.
# from warp._src.geometry import in_circle as in_circle
# from warp._src.geometry import signed_area as signed_area
from warp._src.geometry import triangle_triangle_adjacency as triangle_triangle_adjacency
