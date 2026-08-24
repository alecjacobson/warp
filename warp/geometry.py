# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations for triangle meshes.

This module provides GPU-accelerated operations on triangle mesh geometry, such
as the swept volume (motion envelope) of animated rigid meshes.

Two tiers of API are exposed. Array-level functions such as
:func:`swept_volume` and :func:`swept_volume_field` launch kernels over a whole
grid. Device functions such as :func:`swept_volume_sdf` evaluate a single point
and may be called from within your own :func:`warp.kernel` definitions.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import SweptVolumeSign as SweptVolumeSign
from warp._src.geometry import swept_volume as swept_volume
from warp._src.geometry import swept_volume_field as swept_volume_field
from warp._src.geometry import swept_volume_sdf as swept_volume_sdf
