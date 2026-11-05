# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Products of one BSR matrix with a batch of vectors stored as rows of a 2D array."""

import enum
import functools
from typing import Any

import warp as wp
from warp._src.types import type_scalar_type


class BsrMvAlgorithm(enum.IntEnum):
    """Kernel strategy for :func:`warp.sparse.bsr_mv` with a batch of vectors.

    A batch is a 2D array whose rows each hold one vector. Explicit choices always run
    the named strategy and raise :class:`ValueError` for matrices or devices it does not
    support.
    """

    AUTO = 0
    """Select a strategy from the matrix block shape and size, the batch size and the device.

    :func:`warp.sparse.bsr_mv` decides on every call without synchronizing, so it cannot
    see how many entries each row actually stores. On CUDA it uses ``ROWS`` for 3x3 BSR
    up to four vectors and ``COLUMNS`` beyond, ``ROWS`` for CSR matrices of at most 2048
    rows, and ``LOOP`` for larger CSR matrices, which forgoes ``COLUMNS`` speedups on
    dense or irregular rows. Prepared operators that decide once at construction can count
    stored entries and choose ``COLUMNS`` for such rows. The CPU uses ``ROWS``. The rule was fitted on one GPU model and may change
    between releases; pin a strategy when the best choice for a workload is known.
    """

    LOOP = 1
    """Apply the single-vector product to each row in turn. Supports every matrix, device and ``transpose``."""

    ROWS = 2
    """Read each matrix row once for up to four vectors stored as rows. Supports CSR and 3x3 BSR
    matrices, compact or padded CSR on CUDA, and compact storage on CPU."""

    COLUMNS = 3
    """Transpose the batch so that the vectors' entries are adjacent, read each matrix row once for
    up to eight vectors, and transpose back. Supports compact or padded CSR and compact 3x3 BSR
    matrices on CUDA. Uses scratch storage for both transposed operands."""


@wp.func
def copy_value(value: Any):
    return value


@wp.kernel(module="unique")
def transpose_rows(source: wp.array2d[Any], target: wp.array2d[Any]):
    i, j = wp.tid()
    # A padded shared stride avoids the worst transpose bank conflicts.
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    wp.tile_store(target, result, offset=(j * 32, i * 32))


@wp.kernel(module="unique")
def csr_eight_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row + 1], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.func
def multiply(a: Any, b: Any):
    return a * b


@wp.kernel(module="unique")
def csr_eight_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 8 + lane % 8, lane // 8
    value = x.dtype(0.0)
    if rhs < x.shape[1]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 4):
            value += x.dtype(values[e]) * x[columns[e], rhs]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=0)
    if rhs < x.shape[1] and slot == 0:
        y[row, rhs] = wp.tile_extract(totals, lane % 8)


@wp.kernel(module="unique")
def transpose_axpby(
    source: wp.array2d[Any],
    y: wp.array2d[Any],
    z: wp.array2d[Any],
    alpha: wp.float64,
    beta: wp.float64,
):
    i, j = wp.tid()
    tile = wp.tile_load(source, shape=(32, 33), offset=(i * 32, j * 32), storage="shared")
    square = wp.tile_view(tile, shape=(32, 32), offset=(0, 0))
    result = wp.tile_map(copy_value, wp.tile_transpose(square))
    result = wp.tile_map(
        multiply,
        result,
        wp.tile_full(shape=(32, 32), value=source.dtype(alpha), dtype=source.dtype),
    )
    if beta != wp.float64(0.0):
        previous = wp.tile_load(y, shape=(32, 32), offset=(j * 32, i * 32))
        result += wp.tile_map(
            multiply,
            previous,
            wp.tile_full(shape=(32, 32), value=source.dtype(beta), dtype=source.dtype),
        )
    wp.tile_store(z, result, offset=(j * 32, i * 32))


@wp.kernel(module="unique")
def csr_four_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row + 1], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


@wp.kernel(module="unique")
def csr_four_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[Any],
    x: wp.array2d[Any],
    y: wp.array2d[Any],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


def csr_block_product(matrix, x, y):
    """Launch the four-RHS row product ``y[r] = matrix @ x[r]`` for a CSR matrix."""
    if matrix.row_counts is not None:
        wp.launch_tiled(
            csr_four_rhs_padded,
            dim=((x.shape[0] + 3) // 4, matrix.nrow),
            inputs=[matrix.offsets, matrix.row_counts, matrix.columns, matrix.values, x, y],
            block_dim=32,
            device=matrix.device,
        )
        return
    wp.launch_tiled(
        csr_four_rhs,
        dim=((x.shape[0] + 3) // 4, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )


@functools.lru_cache(None)
def bsr_eight_rhs(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[matrix],
        x: wp.array2d[dtype],
        y: wp.array2d[dtype],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 8 + lane % 8, lane // 8
        value = vector(dtype(0.0))
        if rhs < x.shape[1]:
            for e in range(offsets[row] + slot, offsets[row + 1], 4):
                col = columns[e] * 3
                value += values[e] * vector(x[col, rhs], x[col + 1, rhs], x[col + 2, rhs])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=0)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=0)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=0)
        if rhs < x.shape[1] and slot == 0:
            y[row * 3, rhs] = wp.tile_extract(a, lane % 8)
            y[row * 3 + 1, rhs] = wp.tile_extract(b, lane % 8)
            y[row * 3 + 2, rhs] = wp.tile_extract(c, lane % 8)

    return compute


@functools.lru_cache(None)
def bsr_four_rhs(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[matrix],
        x: wp.array2d[dtype],
        y: wp.array2d[dtype],
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 4 + lane // 8, lane % 8
        value = vector(dtype(0.0))
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, offsets[row + 1], 8):
                col = columns[e] * 3
                value += values[e] * vector(x[rhs, col], x[rhs, col + 1], x[rhs, col + 2])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=1)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=1)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=1)
        if rhs < x.shape[0] and slot == 0:
            y[rhs, row * 3] = wp.tile_extract(a, lane // 8)
            y[rhs, row * 3 + 1] = wp.tile_extract(b, lane // 8)
            y[rhs, row * 3 + 2] = wp.tile_extract(c, lane // 8)

    return compute


@wp.kernel(enable_backward=False)
def _combine(x: wp.array[Any], y: wp.array[Any], z: wp.array[Any], alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    v = z.dtype(0)
    if alpha != wp.float64(0):
        v = z.dtype(alpha) * x[i]
    if beta != wp.float64(0):
        v += z.dtype(beta) * y[i]
    z[i] = v


@functools.lru_cache(None)
def _cpu_rows(block_type, padded):
    scalar = type_scalar_type(block_type)
    block = 1 if block_type == scalar else block_type._shape_[0]

    @wp.kernel(enable_backward=False, module="unique")
    def product(
        offsets: wp.array[int],
        counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array[block_type],
        x: wp.array2d[scalar],
        y: wp.array2d[scalar],
    ):
        rhs, row = wp.tid()
        block_row = row // wp.static(block)
        end = offsets[block_row + 1]
        if wp.static(padded):
            end = offsets[block_row] + counts[block_row]
        value = scalar(0)
        for e in range(offsets[block_row], end):
            if wp.static(block == 1):
                value += values[e] * x[rhs, columns[e]]
            else:
                for j in range(wp.static(block)):
                    value += values[e][row % wp.static(block), j] * x[rhs, columns[e] * wp.static(block) + j]
        y[rhs, row] = value

    return product


def supports(algorithm, A, device):
    """Return whether a row or column strategy can run ``A`` on ``device``."""
    if A.block_shape not in ((1, 1), (3, 3)):
        return False
    # The CSR kernels read scalar values; a 1x1 matrix block type is not one.
    if A.block_shape == (1, 1) and A.values.dtype != A.scalar_type:
        return False
    if algorithm == BsrMvAlgorithm.COLUMNS and not device.is_cuda:
        return False
    # Only CSR has padded-row CUDA kernels.
    return A.row_counts is None or A.block_shape == (1, 1) or not device.is_cuda


def auto_strategy(block_shape, rows, count, is_cuda, density=None):
    """Return the strategy ``AUTO`` selects for a supported matrix.

    Args:
        block_shape: ``(1, 1)`` or ``(3, 3)``.
        rows: Scalar row count.
        count: Batch size.
        is_cuda: Whether the product runs on CUDA.
        density: Mean stored blocks per block row, or ``None`` when unknown, which
            selects the rule that needs no stored-entry count.
    """
    if not is_cuda:
        return BsrMvAlgorithm.ROWS
    # Thresholds come from a sweep over FEM and irregular matrices on an L40 GPU
    # (asv/benchmarks/sparse/matrix_vector_batch.py).
    if block_shape == (3, 3):
        return BsrMvAlgorithm.ROWS if count <= 4 else BsrMvAlgorithm.COLUMNS
    if count < 2:
        return BsrMvAlgorithm.LOOP
    if rows <= 2048:
        return BsrMvAlgorithm.ROWS
    if density is None or density <= 16:
        return BsrMvAlgorithm.LOOP
    if density <= 32:
        if count <= 2:
            return BsrMvAlgorithm.LOOP
        return BsrMvAlgorithm.ROWS if count <= 4 else BsrMvAlgorithm.COLUMNS
    return BsrMvAlgorithm.COLUMNS


def stored_density(A):
    """Return the mean stored blocks per block row, reading the count back from the device.

    This synchronizes, so call it once while preparing, outside graph capture.
    """
    if A.nrow == 0:
        return 0.0
    if A.row_counts is None:
        stored = int(A.offsets.numpy()[A.nrow])
    else:
        stored = int(A.row_counts.numpy()[: A.nrow].sum())
    return stored / A.nrow


def resolve(algorithm, A, count, device, transpose, density=None):
    """Return the concrete strategy for a batch of ``count`` vectors, validating explicit choices.

    ``density`` is the mean stored blocks per block row when the caller knows it; see
    :func:`auto_strategy`.
    """
    if not isinstance(algorithm, BsrMvAlgorithm):
        raise TypeError("algorithm must be a BsrMvAlgorithm")
    if algorithm == BsrMvAlgorithm.AUTO:
        if transpose or not supports(BsrMvAlgorithm.ROWS, A, device):
            return BsrMvAlgorithm.LOOP
        if count < 2 and density is None:
            return BsrMvAlgorithm.LOOP
        return auto_strategy(A.block_shape, A.shape[0], count, device.is_cuda, density)
    if algorithm == BsrMvAlgorithm.LOOP:
        return algorithm
    if transpose:
        raise ValueError(f"{algorithm.name} does not support transpose; use BsrMvAlgorithm.LOOP")
    if not supports(algorithm, A, device):
        raise ValueError(
            f"{algorithm.name} supports compact or padded CSR and compact 3x3 BSR matrices"
            + (" on CUDA" if algorithm == BsrMvAlgorithm.COLUMNS else "")
            + f"; got block shape {A.block_shape}"
            + (" with padded rows" if A.row_counts is not None else "")
            + f" on {device}"
        )
    return algorithm


def scratch_size(algorithm, A, count):
    """Return the scalar scratch entries a strategy needs for ``count`` vectors."""
    if algorithm == BsrMvAlgorithm.COLUMNS:
        return count * (A.shape[0] + A.shape[1])
    if algorithm == BsrMvAlgorithm.ROWS:
        return count * A.shape[0]
    return 0


def product(algorithm, A, x, y, z, alpha, beta, scratch):
    """Compute ``z[r] = alpha * A @ x[r] + beta * y[r]`` for each row with a row or column strategy.

    ``x``, ``y`` and ``z`` are contiguous 2D scalar arrays that may alias exactly; ``scratch`` is
    a flat scalar array of at least :func:`scratch_size` entries.
    """
    device = A.values.device
    count, rows, cols = x.shape[0], A.shape[0], A.shape[1]
    if alpha == 0:
        # Leave x unread, so uninitialized or nonfinite inputs cannot reach z.
        wp.launch(
            _combine, z.size, [y.flatten(), y.flatten(), z.flatten(), wp.float64(0), wp.float64(beta)], device=device
        )
        return
    if algorithm == BsrMvAlgorithm.COLUMNS:
        first = scratch[: cols * count].reshape((cols, count))
        second = scratch[cols * count : (cols + rows) * count].reshape((rows, count))
        wp.launch_tiled(
            transpose_rows,
            dim=((count + 31) // 32, (cols + 31) // 32),
            inputs=[x, first],
            block_dim=128,
            device=device,
        )
        if A.block_shape == (1, 1):
            kernel = csr_eight_rhs if A.row_counts is None else csr_eight_rhs_padded
            inputs = [A.offsets, A.columns, A.values, first, second]
            if A.row_counts is not None:
                inputs.insert(1, A.row_counts)
        else:
            kernel = bsr_eight_rhs(A.scalar_type)
            inputs = [A.offsets, A.columns, A.values, first, second]
        wp.launch_tiled(kernel, dim=((count + 7) // 8, A.nrow), inputs=inputs, block_dim=32, device=device)
        wp.launch_tiled(
            transpose_axpby,
            dim=((rows + 31) // 32, (count + 31) // 32),
            inputs=[second, y, z, wp.float64(alpha), wp.float64(beta)],
            block_dim=128,
            device=device,
        )
        return
    direct = alpha == 1 and beta == 0 and x.ptr != z.ptr
    output = z if direct else scratch[: count * rows].reshape((count, rows))
    if not device.is_cuda:
        padded = A.row_counts is not None
        wp.launch(
            _cpu_rows(A.values.dtype, padded),
            (count, rows),
            [A.offsets, A.row_counts if padded else A.offsets, A.columns, A.values, x, output],
            device=device,
        )
    elif A.block_shape == (1, 1):
        csr_block_product(A, x, output)
    else:
        wp.launch_tiled(
            bsr_four_rhs(A.scalar_type),
            dim=((count + 3) // 4, A.nrow),
            inputs=[A.offsets, A.columns, A.values, x, output],
            block_dim=32,
            device=device,
        )
    if not direct:
        wp.launch(
            _combine,
            y.size,
            [output.flatten(), y.flatten(), z.flatten(), wp.float64(alpha), wp.float64(beta)],
            device=device,
        )
