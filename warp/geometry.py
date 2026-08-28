# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations for triangle meshes.

This module provides GPU-accelerated operations on triangle mesh geometry. It
begins with uniform surface sampling: drawing points spread evenly across a
mesh regardless of how finely it is tessellated.

Two tiers of API are exposed. The host-level function :func:`uniformly_sample`
and the :class:`UniformSampler` class launch kernels over a whole mesh. The
device function :func:`draw` operates on a single sample and may be called from
within your own :func:`warp.kernel` definitions.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import MeshSample as MeshSample
from warp._src.geometry import UniformSampler as UniformSampler
from warp._src.geometry import UniformSamplerState as UniformSamplerState
from warp._src.geometry import draw as draw
from warp._src.geometry import sample_barycentrics as sample_barycentrics
from warp._src.geometry import uniformly_sample as uniformly_sample
