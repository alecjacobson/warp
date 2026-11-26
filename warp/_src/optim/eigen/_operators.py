# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt upstream linear actions to private eigenvector rows."""

import warp as wp
from warp._src.optim._multi_rhs import MultiRHSOperator, _scalar_shape
from warp._src.optim.linear import aslinearoperator
from warp._src.sparse import BsrMatrix


@wp.kernel(enable_backward=False)
def merge_statistics(source: wp.array[int], target: wp.array[int]):
    i = wp.tid()
    if i == 1:
        wp.atomic_max(target, i, source[i])
    else:
        wp.atomic_add(target, i, source[i])


@wp.kernel(enable_backward=False)
def _record_inner_failure_kernel(converged: wp.array[int], failure: wp.array[int]):
    i = wp.tid()
    if converged[i] == 0:
        wp.atomic_max(failure, 0, 1)


class RowsOperator:
    """Prepare fixed row capacities for one upstream operator."""

    def __init__(self, source, capacity):
        self.source = source
        self.single = aslinearoperator(source)
        shape = _scalar_shape(self.single)
        if len(shape) == 1:
            shape = (shape[0], shape[0])
        if shape[0] < 1 or shape[0] != shape[1]:
            raise ValueError("Eigen operators must be nonempty and square")
        if self.single.scalar_type != wp.float64:
            raise TypeError("Eigen operators must use float64")
        if self.single.batch_offsets is not None:
            raise ValueError("Eigen operators must describe one system, not partitioned systems")
        self.n, self.device = shape[0], self.single.device
        self._root = MultiRHSOperator(source, capacity)
        self._capacities = {count: self._root.with_nrhs(count) for count in range(1, capacity)}
        self._capacities[capacity] = self._root

    def apply(self, x, y):
        width = x.shape[0]
        action = self._capacities.get(width)
        if action is None:
            if self.device.is_cuda and self.device.is_capturing:
                raise RuntimeError("Prepare the operator row capacity before capture")
            action = self._root.with_nrhs(width)
            self._capacities[width] = action
        action.matvec(x.flatten(), y.flatten(), y.flatten(), 1.0, 0.0)


class SparseOperator(RowsOperator):
    """Retain sparse metadata for qualified eigen optimizations."""

    def __init__(self, matrix, capacity):
        super().__init__(matrix, capacity)
        self.matrix = matrix


def rows_operator(source, capacity):
    return SparseOperator(source, capacity) if isinstance(source, BsrMatrix) else RowsOperator(source, capacity)


class SolveAction:
    """Borrow a prepared solve without requiring its concrete backend."""

    def __init__(self, action, strict=False):
        if not all(
            hasattr(action, name)
            for name in ("n", "nrhs", "device", "dtype", "solve", "converged", "stats", "reset_stats")
        ):
            raise TypeError("Supply a prepared linear solve action with device diagnostics")
        if action.dtype != wp.float64:
            raise TypeError("Eigen solve actions require float64 inputs and outputs")
        self.action = action
        self.n, self.width, self.device = action.n, action.nrhs, action.device
        self.stats = action.stats
        self.requires_static_outer_capture = getattr(action, "requires_static_outer_capture", False)
        self.failure = wp.zeros(1, dtype=int, device=self.device) if strict else None

    def apply(self, x, y):
        self.action.solve(x, y)
        if self.failure is not None:
            wp.launch(
                _record_inner_failure_kernel, self.width, [self.action.converged, self.failure], device=self.device
            )

    def reset_history(self):
        if self.failure is not None:
            self.failure.zero_()

    def reset_stats(self):
        self.action.reset_stats()

    def _for_active_width(self, width):
        return SolveAction(self.action.with_nrhs(width))

    def for_width(self, width):
        return self._for_active_width(width)
