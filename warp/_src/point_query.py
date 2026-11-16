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
    """Preserve the deprecated array query and its raw index return value.

    New code should use :func:`warp.geometry.knn` with a target array or
    :class:`warp.geometry.PointIndex`. This compatibility call still accepts
    ``bvh=`` and returns the original ``(bvh, indices, distances, counts)`` tuple;
    batched inputs return a tuple of raw indexes as the first value.
    """
    log_warning(
        "wp.knn() is deprecated; use warp.geometry.knn() with a warp.geometry.PointIndex or target array. "
        "The geometry API returns a PointIndex owner as its first result.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _knn(*args, **kwargs)


@functools.wraps(_build_point_index, assigned=("__annotations__",))
def build_point_index(*args, **kwargs):
    """Preserve the deprecated builder and its BVH, cuBQL index, or tuple result.

    New code should construct :class:`warp.geometry.PointIndex`. This
    compatibility call retains the earlier raw-index return contract; its
    result is accepted by ``wp.knn(..., bvh=...)``, not as a geometry owner.
    """
    log_warning(
        "wp.build_point_index() is deprecated; construct warp.geometry.PointIndex(targets, ...) instead. "
        "Use wp.Bvh(points, points) directly for scalar kernel queries.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _build_point_index(*args, **kwargs)


@functools.wraps(_point_query_config, assigned=("__annotations__",))
def point_query_config(*args, **kwargs):
    """Preserve the deprecated point-query tuning helper.

    Use :func:`warp.geometry.point_query_config` for the same Warp-backend
    settings without a deprecation warning.
    """
    log_warning(
        "wp.point_query_config() is deprecated; use warp.geometry.point_query_config() instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _point_query_config(*args, **kwargs)


@functools.wraps(_knn, assigned=("__module__", "__annotations__"))
def point_query_knn_batch(*args, **kwargs):
    """Preserve the original batched-query signature and raw index return value.

    Use :func:`warp.geometry.knn` for new code. This deprecated call retains
    the original four outputs, with a raw BVH/index or tuple of indexes first.
    The geometry API instead returns a :class:`warp.geometry.PointIndex` owner.
    """
    log_warning(
        "wp.point_query_knn_batch() is deprecated; use warp.geometry.knn() instead. "
        "wp.knn() also remains available for compatibility with the original return values.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _knn(*args, **kwargs)
