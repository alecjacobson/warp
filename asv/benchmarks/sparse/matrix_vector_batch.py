# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep of batched ``bsr_mv`` strategies over matrix structure, batch width and precision.

Each sample replays a graph of ``PRODUCTS`` products ``Y = A @ X`` on the GPU timeline.
Unsupported strategy and matrix combinations are skipped, so the table compares only
strategies that can run. ``LOOP`` is the baseline of separate single-vector products.
"""

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sparse as wps

from ..benchmarks_utils import CudaEventTimer, capture_launches, setup_once

PRODUCTS = 16


@fem.integrand
def _scalar_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@fem.integrand
def _vector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.D(u, s), fem.D(v, s)) + wp.dot(u(s), v(s))


def _fem_matrix(geo, dtype, vector):
    space = fem.make_polynomial_space(geo, dtype=wp.vec3 if vector else float)
    form = _vector_form if vector else _scalar_form
    matrix = fem.integrate(form, fields={"u": fem.make_trial(space), "v": fem.make_test(space)}, output_dtype=dtype)
    wps.bsr_compress(matrix)
    return matrix


def _irregular_matrix(dtype, rows=32768, seed=11):
    """CSR matrix whose row lengths vary from 1 to 96 entries."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 97, size=rows)
    row_index = np.repeat(np.arange(rows), lengths)
    columns = rng.integers(0, rows, size=row_index.size)
    values = rng.normal(size=row_index.size)
    return wps.bsr_from_triplets(
        rows,
        rows,
        wp.array(row_index, dtype=int),
        wp.array(columns, dtype=int),
        wp.array(values, dtype=dtype),
    )


def build(problem, dtype):
    """Return the benchmark matrix for ``problem``.

    ``poisson2d`` and ``poisson3d`` are scalar P1 grids with 9 and 27 entries per row,
    ``elastic3d`` is a 3x3-block P1 grid with 27 blocks per row, and ``irregular`` has
    widely varying row lengths. The ``warpack`` case matches the multiple-RHS solver
    benchmarks in ``asv/benchmarks/optim``.
    """
    if problem == "poisson2d":
        return _fem_matrix(fem.Grid2D(res=(256, 256)), dtype, False)
    if problem == "poisson3d":
        return _fem_matrix(fem.Grid3D(res=(40, 40, 40)), dtype, False)
    if problem == "elastic3d":
        return _fem_matrix(fem.Grid3D(res=(24, 24, 24)), dtype, True)
    if problem == "warpack":
        return _fem_matrix(fem.Grid2D(res=(32, 32)), dtype, False)
    return _irregular_matrix(dtype)


class BsrMvBatch:
    params = (
        ["poisson2d", "poisson3d", "elastic3d", "irregular", "warpack"],
        [1, 2, 4, 8, 16, 33, 64],
        ["float32", "float64"],
        ["LOOP", "ROWS", "COLUMNS"],
    )
    param_names = ["problem", "nrhs", "dtype", "algorithm"]
    timer = CudaEventTimer()
    number = 1
    repeat = 10
    rounds = 2

    @setup_once
    def setup(self, problem, nrhs, dtype, algorithm):
        self.device = wp.get_device("cuda:0")
        algorithm = wps.BsrMvAlgorithm[algorithm]
        with wp.ScopedDevice(self.device):
            self.a = build(problem, getattr(wp, dtype))
            self.x = wp.ones((nrhs, self.a.shape[1]), dtype=self.a.scalar_type)
            self.y = wp.zeros((nrhs, self.a.shape[0]), dtype=self.a.scalar_type)
            self.work = wp.empty(nrhs * (self.a.shape[0] + self.a.shape[1]), dtype=self.a.scalar_type)

            def product():
                wps.bsr_mv(self.a, self.x, self.y, algorithm=algorithm, work_buffer=self.work)

            try:
                product()
            except ValueError as error:
                raise NotImplementedError(str(error)) from error
            self.graph = capture_launches(product, PRODUCTS)
            wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)

    def time_cuda(self, problem, nrhs, dtype, algorithm):
        wp.capture_launch(self.graph)
