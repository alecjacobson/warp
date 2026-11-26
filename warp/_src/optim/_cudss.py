# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide explicitly optional, prepared cuDSS linear solves."""

import ctypes
import weakref
from contextlib import suppress
from typing import Any, Literal

import warp as wp
from warp._src import context
from warp._src.optim._multi_rhs import MultiRHSOperator
from warp._src.optim._solve_action import _positive_int, _resolve_tolerances, _SolveAction
from warp._src.sparse import BsrMatrix


@wp.kernel(enable_backward=False)
def _correction_rhs_kernel(b: wp.array[Any], ax: wp.array[Any], residual: wp.array[Any]):
    i = wp.tid()
    residual[i] = b[i] - ax[i]


@wp.kernel(enable_backward=False)
def _add_correction_kernel(x: wp.array[Any], correction: wp.array[Any]):
    i = wp.tid()
    x[i] += correction[i]


@wp.kernel(enable_backward=False)
def _direct_tolerance_kernel(
    rhs_sq: wp.array[wp.float64], threshold: wp.array[wp.float64], tol: wp.float64, atol: wp.float64
):
    i = wp.tid()
    threshold[i] = wp.max(atol * atol, tol * tol * rhs_sq[i])


class CuDSS(_SolveAction):
    """Factor an assembled CUDA system for repeated direct solves.

    Opt in by installing ``warp-cudss`` and a supported NVIDIA cuDSS library.
    Importing this class does not load either dependency. Construction performs
    analysis and numerical factorization outside capture. ``solve`` uses fixed
    internal buffers, accepts changing right-hand sides, and reports fresh
    original-system residuals on device. Matrix assembly belongs to the caller.

    Create, update and record solves on the construction stream. Order replay
    on other streams explicitly; mutable native state is not reentrant. Captured
    graphs retain the solver and prevent explicit close until destroyed. Pure
    solve capture is supported after an eager warmup; cuDSS solves cannot be
    nested in a CUDA conditional graph body on the qualified provider stack.
    Eigen routines therefore use a fixed outer budget for this action.

    Args:
        A: Nonempty square float32 or float64 Warp BSR matrix on CUDA, with full
            storage. Keep topology and buffer identities fixed for reuse.
        nrhs: Positive number of right-hand sides prepared together.
        matrix_type: ``"spd"``, ``"symmetric"`` or ``"general"``; the caller must
            ensure the matrix satisfies the selected mathematical category.
        tol: Relative fresh linear-residual tolerance; defaults to 1e-10 for
            float64 and 1e-5 for float32. It reports accuracy, not an iterative budget.
        atol: Nonnegative absolute fresh linear-residual tolerance.
        refinement_steps: Fixed number of Warp residual-correction solves. None
            chooses zero for SPD and one for the other categories.

    Attributes:
        converged: Device acceptance flags for the latest right-hand sides.
        absolute_residual: Fresh original-system residual norms.
        relative_residual: Residual norms divided by the right-hand-side norms.
        tolerance: Effective absolute acceptance tolerances.
        iterations: Number of native solve phases per right-hand side.
        stats: Cumulative solve phases, maximum phases, failed and solved RHS counts.
        residual_maxima: Cumulative maximum absolute and relative residuals.
    """

    requires_static_outer_capture = True

    def __init__(
        self,
        A: BsrMatrix,
        nrhs: int = 1,
        *,
        matrix_type: Literal["spd", "symmetric", "general"] = "spd",
        tol: float | None = None,
        atol: float = 0.0,
        refinement_steps: int | None = None,
    ):
        if not isinstance(A, BsrMatrix):
            raise TypeError("CuDSS requires an assembled Warp BSR matrix")
        if A.shape[0] < 1 or A.shape[0] != A.shape[1] or A.block_shape[0] != A.block_shape[1]:
            raise ValueError("CuDSS requires a nonempty square matrix with square blocks")
        if A.scalar_type not in (wp.float32, wp.float64):
            raise TypeError("CuDSS supports float32 or float64 matrices")
        if not A.device.is_cuda:
            raise ValueError("CuDSS requires a CUDA device; use KrylovSolve on CPU")
        if A.device.is_capturing:
            raise RuntimeError("Construct CuDSS outside CUDA capture")
        if context.runtime.tape is not None:
            raise RuntimeError("CuDSS does not support reverse-mode differentiation")
        if matrix_type not in ("spd", "symmetric", "general"):
            raise ValueError("matrix_type must be spd, symmetric or general")
        nrhs = _positive_int("nrhs", nrhs)
        if refinement_steps is None:
            refinement_steps = int(matrix_type != "spd")
        if isinstance(refinement_steps, bool) or not isinstance(refinement_steps, int):
            raise TypeError("refinement_steps must be an integer")
        if refinement_steps < 0:
            raise ValueError("refinement_steps must be nonnegative")
        if tol is None:
            tol = 1e-10 if A.scalar_type == wp.float64 else 1e-5
        self.tol, self.atol = _resolve_tolerances(tol, atol, A.scalar_type)
        try:
            from warp_cudss import CudssSolver  # noqa: PLC0415 - optional native provider
        except ImportError as error:
            raise ImportError("Install warp-cudss and a supported NVIDIA cuDSS library to use CuDSS") from error
        from warp._src.optim._cudss_memory import Workspace  # noqa: PLC0415 - optional provider workspace

        self.source, self.matrix_type = A, matrix_type
        self.refinement_steps = refinement_steps
        self._init_action(A.shape[0], nrhs, A.scalar_type, A.device, "fresh", wp.float64)
        self._stream = A.device.stream
        self._graphs = weakref.WeakSet()
        self._closed = False
        self._invalidated = False
        self._warmed = False
        self._provider = None
        self._workspace = None
        self._storage = self._storage_identity()
        self._rhs = wp.zeros((nrhs, self.n), dtype=self.dtype, device=self.device)
        # Refinement overwrites the native RHS with a correction residual and
        # therefore needs a separate preserved original. Plain solves do not.
        self._native_rhs = wp.empty_like(self._rhs) if refinement_steps else self._rhs
        self._native_solution = wp.empty_like(self._rhs)
        self._solution = wp.empty_like(self._rhs) if refinement_steps else self._native_solution
        self._ax = wp.empty_like(self._rhs)
        self._threshold_sq = wp.zeros(nrhs, dtype=wp.float64, device=self.device)
        self._phases = wp.full(1, 1 + refinement_steps, dtype=int, device=self.device)
        self._operator = MultiRHSOperator(A, nrhs, algorithm="rows" if nrhs == 1 else "auto")
        try:
            self._provider = CudssSolver(mtype=matrix_type, device=self.device)
            self._workspace = Workspace(self._provider)
            from warp_cudss._bindings import CUDSS_CONFIG_MATCHING_ALG, _check  # noqa: PLC0415

            # Disable optional matching explicitly across supported cuDSS versions.
            matching = ctypes.c_int(0)
            _check(
                self._provider._lib.cudssConfigSet(
                    self._provider._config, CUDSS_CONFIG_MATCHING_ALG, ctypes.byref(matching), ctypes.sizeof(matching)
                ),
                "cudssConfigSet(MATCHING_ALG)",
            )
            A.nnz_sync()
            self._provider.setup(A, self._native_solution, self._native_rhs, nrhs=nrhs)
            self.solve(self._rhs, self._solution)
            wp.synchronize_stream(self._stream)
            self._check_factorization()
            self.reset_stats()
            self.converged.zero_()
        except BaseException:
            if self._provider is not None:
                with suppress(Exception):
                    self._provider.release()
            self._closed = True
            raise

    def _storage_identity(self):
        a = self.source
        counts = None if a.row_counts is None else a.row_counts.ptr
        return (a.shape, a.dtype, a.offsets.ptr, a.columns.ptr, counts, a.values.ptr, a.values.size)

    def _check_factorization(self):
        # CUDSS_DATA_INFO is host metadata, queried only after setup/update has
        # synchronized. Never add this query to the replayable solve path.
        query = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        )(("cudssDataGet", self._provider._lib))
        info = ctypes.c_int(-1)
        written = ctypes.c_size_t()
        status = query(
            self._provider._handle,
            self._provider._data,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(written),
        )
        if status != 0 or written.value != ctypes.sizeof(info) or info.value != 0:
            self._invalidated = True
            raise RuntimeError(f"cuDSS factorization failed: query status={status}, factor info={info.value}")

    def _validate_use(self):
        if self._closed:
            raise RuntimeError("CuDSS is closed")
        if self._invalidated or self._storage_identity() != self._storage:
            raise RuntimeError("CuDSS storage changed; rebuild the solver and dependent graphs")
        if self.device.stream is not self._stream:
            raise ValueError("Record CuDSS solves and updates on the construction stream")
        if context.runtime.tape is not None:
            raise RuntimeError("CuDSS does not support reverse-mode differentiation")

    def _retain_capture(self):
        capture_id = context.runtime.core.wp_cuda_stream_get_capture_id(self._stream.cuda_stream)
        graph = context.runtime.captures.get(capture_id)
        if graph is None:
            raise RuntimeError("CuDSS capture requires a Warp-managed graph for resource ownership")
        if graph not in self._graphs:
            if not hasattr(graph, "_linear_solve_owners"):
                graph._linear_solve_owners = []
            graph._linear_solve_owners.append(self)
            self._graphs.add(graph)

    def solve(self, rhs: wp.array, out: wp.array, x0: wp.array | None = None, *, diagnostics: bool = True):
        """Solve into caller-owned storage, supporting exact input/output aliasing.

        Args:
            rhs: Contiguous ``(nrhs,n)`` rows, or ``(n,)`` for one RHS.
            out: Matching output storage; shifted partial overlaps are rejected.
            x0: Must be None; a direct solve has no initial-guess policy.
            diagnostics: Refresh per-RHS accuracy and cumulative statistics.
                False leaves diagnostics unchanged and must not be used when
                a consumer relies on fresh acceptance flags.

        Returns:
            Device arrays containing solve phase counts, squared residual norms
            and squared tolerances, matching the prepared solve-action convention.
        """
        self._validate_use()
        if x0 is not None:
            raise ValueError("CuDSS does not accept an initial guess")
        rows, output = self._prepare_solve(rhs, out)
        if self._stream.is_capturing:
            if not self._warmed:
                raise RuntimeError("Warm the CuDSS solve before capture")
            self._retain_capture()
        wp.copy(self._rhs, rows)
        if self.refinement_steps:
            wp.copy(self._native_rhs, self._rhs)
        self._provider.solve()
        if self.refinement_steps:
            wp.copy(self._solution, self._native_solution)
        for _ in range(self.refinement_steps):
            self._operator.matvec(self._solution.flatten(), self._ax.flatten(), self._ax.flatten(), 1.0, 0.0)
            wp.launch(
                _correction_rhs_kernel,
                self._rhs.size,
                [self._rhs.flatten(), self._ax.flatten(), self._native_rhs.flatten()],
                device=self.device,
            )
            self._provider.solve()
            wp.launch(
                _add_correction_kernel,
                self._solution.size,
                [self._solution.flatten(), self._native_solution.flatten()],
                device=self.device,
            )
        if self._workspace.error is not None:
            raise self._workspace.error
        if diagnostics:
            self._operator.matvec(self._solution.flatten(), self._ax.flatten(), self._ax.flatten(), 1.0, 0.0)
            self._residual_sq.zero_()
            self._rhs_sq.zero_()
            self._squared_norms(self._ax, self._rhs, 0)
            wp.launch(
                _direct_tolerance_kernel,
                self.nrhs,
                [self._rhs_sq, self._threshold_sq, self.tol, self.atol],
                device=self.device,
            )
            self._record(self._phases, self._residual_sq, self._rhs_sq, self._threshold_sq, 0, self.nrhs)
        wp.copy(output, self._solution)
        self._warmed = True
        return self.iterations, self._residual_sq, self._threshold_sq

    def update(self):
        """Refactor changed values in the original sparse buffers outside capture.

        Preserve topology and array identities. This refreshes native factors;
        it does not assemble or infer changes to the matrix. Existing captured
        solves may be reused only when factor workspace addresses remain stable.

        Returns:
            This solver after refactorization and eager workspace validation.
        """
        self._validate_use()
        if self.device.is_capturing:
            raise RuntimeError("CuDSS numerical refactorization must run outside capture")
        previous = set(self._workspace.buffers)
        try:
            self._provider.refactor(self.source)
            self.solve(self._rhs, self._solution, diagnostics=False)
            wp.synchronize_stream(self._stream)
            self._check_factorization()
        except Exception:
            self._invalidated = True
            raise
        if self._graphs and set(self._workspace.buffers) != previous:
            self._invalidated = True
            raise RuntimeError("Refactorization grew workspace; destroy dependent graphs and rebuild CuDSS")
        return self

    def close(self):
        """Release native resources after all dependent graphs have been destroyed."""
        if self._closed:
            return
        if self.device.is_capturing:
            raise RuntimeError("Close CuDSS outside capture")
        if self._graphs:
            raise RuntimeError("Destroy captured graphs referencing CuDSS before closing it")
        wp.synchronize_stream(self._stream)
        try:
            self._provider.release()
        finally:
            self._closed = True
            self._workspace.buffers.clear()
            self._workspace.available.clear()

    def __del__(self):
        # Keep callbacks and workspace alive until native destruction completes,
        # including when construction failed partway through native setup.
        with suppress(Exception):
            self.close()
