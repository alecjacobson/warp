# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve the deprecated top-level point-query array API."""

import functools

from warp._src.geometry.point_query import CubqlPointIndex as CubqlPointIndex
from warp._src.geometry.point_query import PointQueryBackend as PointQueryBackend
from warp._src.geometry.point_query import PointQueryConfig as PointQueryConfig
from warp._src.geometry.point_query import PointQueryObjective as PointQueryObjective
from warp._src.geometry.point_query import _build_point_index, _knn
from warp._src.geometry.point_query import point_query_config as _point_query_config
from warp._src.logger import log_warning


@functools.wraps(_knn, assigned=("__annotations__",))
def knn(*args, **kwargs):
    """Preserve the legacy array query and its raw index return value."""
    log_warning(
        "wp.knn() is deprecated; use warp.geometry.knn() with a warp.geometry.PointIndex or target array. "
        "The geometry API returns a PointIndex owner as its first result.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _knn(*args, **kwargs)


@functools.wraps(_build_point_index, assigned=("__annotations__",))
def build_point_index(*args, **kwargs):
    """Preserve the legacy builder and its BVH, cuBQL index, or tuple result."""
    log_warning(
        "wp.build_point_index() is deprecated; construct warp.geometry.PointIndex(targets, ...) instead. "
        "Use wp.Bvh(points, points) directly for scalar kernel queries.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _build_point_index(*args, **kwargs)


@functools.wraps(_point_query_config, assigned=("__annotations__",))
def point_query_config(*args, **kwargs):
    """Preserve the legacy point-query tuning helper."""
    log_warning(
        "wp.point_query_config() is deprecated; use warp.geometry.point_query_config() instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _point_query_config(*args, **kwargs)


@functools.wraps(_knn, assigned=("__module__", "__annotations__"))
def point_query_knn_batch(*args, **kwargs):
    """Preserve the original batched-query signature and raw index return value."""
    log_warning(
        "wp.point_query_knn_batch() is deprecated; use warp.geometry.knn() instead. "
        "wp.knn() also remains available for compatibility with the original return values.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _knn(*args, **kwargs)
