# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid registration (iterative closest point) for point clouds and meshes.

GPU-accelerated ICP that aligns a source point set to a target surface. The
default is the point-to-plane Gauss-Newton formulation, with normals taken from
the closest-point query on the target. Because the motion is rigid, the target's
acceleration structure is built once and never rebuilt across iterations.

The host driver :func:`register_rigid` runs the full loop; the device functions
:func:`closest_on_mesh` and :func:`point_plane_term` are exposed for use in your
own :func:`warp.kernel` definitions.

Usage:
    This module must be explicitly imported::

        import warp.registration
"""

# isort: skip_file

from warp._src.registration import ClosestPoint as ClosestPoint
from warp._src.registration import GaussNewtonTerm as GaussNewtonTerm
from warp._src.registration import RegistrationResult as RegistrationResult
from warp._src.registration import closest_on_mesh as closest_on_mesh
from warp._src.registration import point_plane_term as point_plane_term
from warp._src.registration import register_rigid as register_rigid
