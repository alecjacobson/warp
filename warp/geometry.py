# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations for triangle meshes.

This module provides GPU-accelerated, differentiable operations on triangle
mesh geometry, such as per-triangle area computation.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import VertexNormalWeighting as VertexNormalWeighting
from warp._src.geometry import triangle_areas as triangle_areas
from warp._src.geometry import triangle_corner_angles as triangle_corner_angles
from warp._src.geometry import moments as moments
from warp._src.geometry import triangle_normals as triangle_normals
from warp._src.geometry import vertex_gaussian_curvature as vertex_gaussian_curvature
from warp._src.geometry import vertex_normals as vertex_normals
