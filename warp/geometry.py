# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations for triangle meshes.

This module provides GPU-accelerated, differentiable operations on triangle
mesh geometry, such as per-triangle area computation.

Two tiers of API are exposed. Array-level functions such as
:func:`triangle_areas` and :func:`vertex_normals` launch kernels over a whole
mesh. Device functions such as :func:`triangle_normal` and
:func:`corner_half_angle` operate on a single element and may be called from
within your own :func:`warp.kernel` definitions.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import OBBMeasureType as OBBMeasureType
from warp._src.geometry import VertexNormalWeighting as VertexNormalWeighting
from warp._src.geometry import corner_half_angle as corner_half_angle
from warp._src.geometry import moments as moments
from warp._src.geometry import oriented_bounding_box as oriented_bounding_box
from warp._src.geometry import super_fibonacci as super_fibonacci
from warp._src.geometry import triangle_areas as triangle_areas
from warp._src.geometry import triangle_corner_angles as triangle_corner_angles
from warp._src.geometry import triangle_corner_half_angles as triangle_corner_half_angles
from warp._src.geometry import triangle_double_area as triangle_double_area
from warp._src.geometry import triangle_normal as triangle_normal
from warp._src.geometry import triangle_normals as triangle_normals
from warp._src.geometry import vertex_gaussian_curvature as vertex_gaussian_curvature
from warp._src.geometry import vertex_normals as vertex_normals
