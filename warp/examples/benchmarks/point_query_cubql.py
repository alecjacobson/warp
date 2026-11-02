# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional ctypes adapter for the direct cuBQL benchmark library."""

import ctypes

import warp as wp


class CubqlLibrary:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path.resolve()))
        ptr = ctypes.c_void_p
        self.lib.point_cubql_build.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr]
        self.lib.point_cubql_build.restype = ptr
        self.lib.point_cubql_destroy.argtypes = [ptr, ptr]
        self.lib.point_cubql_destroy.restype = None
        self.lib.point_cubql_query.argtypes = [
            ptr,
            ptr,
            ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
        ]
        self.lib.point_cubql_query.restype = ctypes.c_int

    def build(self, points, leaf=4):
        stream = wp.get_stream(points.device).cuda_stream
        return _Tree(self, points, self.lib.point_cubql_build(points.ptr, len(points), leaf, stream))


class _Tree:
    def __init__(self, library, points, handle):
        self.library = library
        self.points = points
        self.handle = handle

    def query(self, queries, k, cutoff, excludes, indices, distances, counts):
        error = self.library.lib.point_cubql_query(
            self.handle,
            self.points.ptr,
            queries.ptr,
            len(queries),
            k,
            cutoff,
            excludes.ptr,
            indices.ptr,
            distances.ptr,
            counts.ptr,
            wp.get_stream(self.points.device).cuda_stream,
        )
        if error:
            raise RuntimeError(f"cuBQL benchmark query returned CUDA status {error}")

    def __del__(self):
        if self.handle:
            with self.points.device.context_guard:
                self.library.lib.point_cubql_destroy(self.handle, wp.get_stream(self.points.device).cuda_stream)
            self.handle = None
