# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose reusable symmetric eigensolvers with device results."""

import ctypes
import math
from enum import IntEnum
from operator import index

import warp as wp
from warp._src import context
from warp._src.optim._multi_rhs import _scalar_shape, _StreamOrder
from warp._src.optim._solve_action import KrylovSolve
from warp._src.optim.linear import aslinearoperator

from ._capture import capture_scope
from ._operators import SolveAction, rows_operator
from .block import BlockLOBPCG, Identity
from .generalized import generalized_residual
from .generalized_krylov import GeneralizedKrylovSchur
from .linalg_tiles import gram_tiled
from .solver import KrylovSchur as _KrylovSchur


class EigenStatus(IntEnum):
    """Describe the outcome of a partial eigenproblem solve."""

    NOT_RUN = 0
    CONVERGED = 1
    ITERATION_LIMIT = 2
    BREAKDOWN = 3
    INVALID_METRIC = 4
    INNER_SOLVE_FAILED = 5
    NONFINITE = 6


@wp.kernel(enable_backward=False)
def _result_order_kernel(values: wp.array[wp.float64], order: wp.array[int]):
    for i in range(order.shape[0]):
        order[i] = i
    for i in range(order.shape[0]):
        best = i
        for j in range(i + 1, order.shape[0]):
            if values[order[j]] < values[order[best]]:
                best = j
        previous = order[i]
        order[i] = order[best]
        order[best] = previous


@wp.kernel(enable_backward=False)
def _result_rows_kernel(source: wp.array2d[wp.float64], order: wp.array[int], target: wp.array2d[wp.float64]):
    i, j = wp.tid()
    target[i, j] = source[order[i], j]


@wp.kernel(enable_backward=False)
def _result_values_kernel(
    values: wp.array[wp.float64],
    order: wp.array[int],
    norms: wp.array2d[wp.float64],
    mass_norms: wp.array[wp.float64],
    mass_gram: wp.array2d[wp.float64],
    core_status: wp.array[int],
    inner_failure: wp.array[int],
    tol: wp.float64,
    atol: wp.float64,
    out_values: wp.array[wp.float64],
    absolute: wp.array[wp.float64],
    relative: wp.array[wp.float64],
    valid: wp.array[bool],
    count: wp.array[int],
    status: wp.array[int],
):
    accepted = int(0)
    failure = int(0)
    for row in range(order.shape[0]):
        for column in range(order.shape[0]):
            entry = mass_gram[row, column]
            if not wp.isfinite(entry) or wp.abs(entry - wp.float64(row == column)) > wp.max(
                wp.float64(1e-8), wp.float64(10.0) * tol
            ):
                failure = 4
    for i in range(order.shape[0]):
        j = order[i]
        value = values[j]
        residual = wp.sqrt(norms[j, 0])
        scale = wp.sqrt(norms[j, 1]) + wp.abs(value) * wp.sqrt(norms[j, 2])
        ratio = wp.float64(wp.inf)
        if scale > wp.float64(0.0):
            ratio = residual / scale
        elif residual == wp.float64(0.0):
            ratio = wp.float64(0.0)
        finite = wp.isfinite(value) and wp.isfinite(residual) and wp.isfinite(scale)
        if not finite:
            failure = 6
        if not wp.isfinite(mass_norms[j]) or mass_norms[j] <= wp.float64(0.0):
            failure = 4
        valid[i] = finite and mass_norms[j] > wp.float64(0.0) and residual <= atol + tol * scale
        out_values[i] = value
        absolute[i] = residual
        relative[i] = ratio
        accepted += int(valid[i])
    if inner_failure[0] != 0:
        failure = 5
    elif core_status[0] != 0 and failure == 0:
        failure = 3
    if failure != 0:
        for i in range(order.shape[0]):
            valid[i] = False
        accepted = 0
    count[0] = accepted
    status[0] = failure
    if failure == 0:
        status[0] = 2
        if accepted == order.shape[0]:
            status[0] = 1


@wp.kernel(enable_backward=False)
def _mass_norms_kernel(x: wp.array2d[wp.float64], mx: wp.array2d[wp.float64], norms: wp.array[wp.float64]):
    i, part = wp.tid()
    a = wp.tile_load(x[i], shape=256, offset=part * 256)
    b = wp.tile_load(mx[i], shape=256, offset=part * 256)
    wp.tile_atomic_add(norms, wp.tile_sum(wp.tile_map(wp.mul, a, b)), offset=i)


@wp.kernel(enable_backward=False)
def _loop_kernel(
    count: wp.array[int], status: wp.array[int], iterations: wp.array[int], running: wp.array[int], k: int, budget: int
):
    iterations[0] += 1
    running[0] = int(count[0] < k and status[0] == 0 and iterations[0] < budget)


class EigenResult:
    """View state-owned eigenpairs and diagnostics on the problem device.

    Eigenvectors have shape ``(k, n)`` with one vector per row, ordered by
    ascending eigenvalue. Subsequent solves overwrite these arrays. Retain the
    result's owning state until dependent device work completes.

    Attributes:
        eigenvalues: Selected original-problem eigenvalues.
        eigenvectors: Selected vectors normalized in the mass inner product.
        residual_norms: Fresh original-equation residual norms.
        relative_residuals: Residual divided by ``||A u|| + |lambda| ||M u||``.
        valid: Per-pair residual acceptance flags, cleared on numerical failure.
        converged: Number of accepted pairs, in a device integer array of length one.
        iterations: Executed outer iterations, in a device integer array of length one.
        status: An :class:`EigenStatus` value in a device integer array of length one.
    """

    def __init__(self, owner):
        self._owner = owner
        k, n, device = owner.k, owner.n, owner.device
        self.eigenvalues = wp.zeros(k, dtype=wp.float64, device=device)
        self.eigenvectors = wp.zeros((k, n), dtype=wp.float64, device=device)
        self.residual_norms = wp.zeros(k, dtype=wp.float64, device=device)
        self.relative_residuals = wp.zeros_like(self.residual_norms)
        self.valid = wp.zeros(k, dtype=bool, device=device)
        self.converged = wp.zeros(1, dtype=int, device=device)
        self.status = wp.zeros(1, dtype=int, device=device)
        self.iterations = owner._core.iterations

    def check(self):
        """Synchronize and raise if the requested eigenpairs were not accepted.

        Returns:
            This result after confirming that all requested pairs converged.
        """
        status = EigenStatus(int(self.status.numpy()[0]))
        if status != EigenStatus.CONVERGED:
            raise RuntimeError(f"Eigen solve did not converge: {status.name}")
        return self


class EigenSolverState:
    """Share prepared execution and result ownership between eigen algorithms.

    Construct :class:`LOBPCG` or :class:`KrylovSchur` instead. Preparation may
    allocate and synchronize; calling a prepared CUDA state can be captured in
    a caller's graph. Scratch is not reentrant. Order graph executions that
    share a state, operand scratch or a borrowed inverse/preconditioner.
    Reverse-mode differentiation is not supported.
    """

    def __init__(self):
        if type(self) is EigenSolverState:
            raise TypeError("Construct LOBPCG or KrylovSchur")

    def _finish_construction(self, operator, mass, k, tol, atol, maxiter):
        self._core.atol = atol
        if isinstance(self._core, BlockLOBPCG) and self._core._search is not None:
            for branch in self._core._search.buckets:
                branch.atol = atol
        self.n, self.device, self.k = operator.n, operator.device, k
        self.tol, self.atol, self.maxiter = tol, atol, maxiter
        self._operator, self._mass = operator, mass
        self._identity_metric = isinstance(mass, Identity)
        self._prepared = False
        self._graph = None
        self._stream_order = _StreamOrder(self.device)
        self.result = EigenResult(self)
        self._order = wp.empty(k, dtype=int, device=self.device)
        self._identity_order = wp.array(list(range(k)), dtype=int, device=self.device)
        # Fresh validation overwrites the previous operator images only after
        # iteration has finished. Initialization rebuilds them on the next call.
        self._ax = self._core.ax[:k]
        self._mx = self._core.eigenvectors if self._identity_metric else self._core.bx[:k]
        self._norms = wp.zeros((k, 3), dtype=wp.float64, device=self.device)
        self._mass_norms = wp.zeros(k, dtype=wp.float64, device=self.device)
        self._mass_gram = wp.zeros((k, k), dtype=wp.float64, device=self.device)
        inverse = getattr(self, "_inverse", None)
        self._inner_failure = inverse.failure if inverse is not None else wp.zeros(1, dtype=int, device=self.device)

    def _finalize(self):
        core, result = self._core, self.result
        self._operator.apply(core.eigenvectors, self._ax)
        if not self._identity_metric:
            self._mass.apply(core.eigenvectors, self._mx)
        self._norms.zero_()
        self._mass_norms.zero_()
        self._mass_gram.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[core.eigenvectors, self._ax, self._mx, core.eigenvalues, self._identity_order, self._norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            _mass_norms_kernel,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[core.eigenvectors, self._mx, self._mass_norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            gram_tiled,
            dim=((self.k + 7) // 8, (self.k + 7) // 8, (self.n + 255) // 256),
            inputs=[core.eigenvectors, self._mx, self._mass_gram],
            block_dim=128,
            device=self.device,
        )
        wp.launch(_result_order_kernel, 1, [core.eigenvalues, self._order], device=self.device)
        wp.launch(
            _result_rows_kernel,
            (self.k, self.n),
            [core.eigenvectors, self._order, result.eigenvectors],
            device=self.device,
        )
        wp.launch(
            _result_values_kernel,
            1,
            [
                core.eigenvalues,
                self._order,
                self._norms,
                self._mass_norms,
                self._mass_gram,
                core.status,
                self._inner_failure,
                self.tol,
                self.atol,
                result.eigenvalues,
                result.residual_norms,
                result.relative_residuals,
                result.valid,
                result.converged,
                result.status,
            ],
            device=self.device,
        )

    def prepare(self):
        """Warm kernels and prepare fixed workspace outside CUDA capture.

        Returns:
            This state, ready for eager calls or caller-owned CUDA capture.
        """
        self._reject_tape()
        if self.device.is_capturing:
            raise RuntimeError("Prepare eigen state outside capture")
        if self._prepared:
            return self
        core = self._core
        if self.device.is_cuda:
            core.solve(1)
            if isinstance(core, BlockLOBPCG) and core._search is not None:
                core._search.warm()
            self._finalize()
            with capture_scope(self.device) as capture:
                self._record()
            self._graph = capture.graph
            executable = ctypes.c_void_p()
            if not context.runtime.core.wp_cuda_graph_create_exec(
                self.device.context,
                self.device.stream.cuda_stream,
                self._graph.graph,
                ctypes.byref(executable),
            ):
                raise RuntimeError(f"Eigen graph preparation failed: {context.runtime.get_error_string()}")
            self._graph.graph_exec = executable
        else:
            core.initialize()
            core.step()
            if not isinstance(core, BlockLOBPCG):
                core.finalize()
            self._finalize()
        self.result.status.zero_()
        self.result.valid.zero_()
        self.result.converged.zero_()
        self.result.iterations.zero_()
        if self.device.is_cuda:
            # Finish graph upload and status initialization without executing a
            # full eigen solve as part of preparation.
            wp.synchronize_device(self.device)
        self._prepared = True
        return self

    def _record(self):
        core = self._core
        core.initialize()
        core.iterations.zero_()
        core.running.fill_(1)

        def iteration():
            core.step()
            if isinstance(core, BlockLOBPCG):
                wp.launch(
                    _loop_kernel,
                    1,
                    [core.converged, core.status, core.iterations, core.running, self.k, self.maxiter],
                    device=self.device,
                )
            else:
                core.finalize()
                core._update_loop(self.maxiter)

        operands = (core.op, getattr(core, "mass", None), getattr(core, "preconditioner", None))
        conditional = wp.is_conditional_graph_supported() and not any(
            getattr(operand, "requires_static_outer_capture", False) for operand in operands
        )
        if conditional:
            wp.capture_while(core.running, iteration)
        else:
            for _ in range(self.maxiter):
                iteration()
        self._finalize()

    def _reject_tape(self):
        if context.runtime is not None and context.runtime.tape is not None:
            raise RuntimeError("Eigen solvers do not support reverse-mode differentiation")

    def __call__(self):
        """Solve from current inputs and return device-resident eigenpairs.

        Returns:
            The state-owned :class:`EigenResult`, overwritten on the next call.
        """
        self._reject_tape()
        if not self._prepared:
            raise RuntimeError("Call prepare() before solving")
        if not self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("CPU eigen execution is eager; CPU graph capture is not supported")
        self._stream_order.acquire(self.device)
        if self.device.is_cuda:
            if self.device.is_capturing:
                self._record()
            else:
                wp.capture_launch(self._graph)
        else:
            core = self._core
            core.initialize()
            core.iterations.zero_()
            for _ in range(self.maxiter):
                core.step()
                if not isinstance(core, BlockLOBPCG):
                    core.finalize()
                wp.launch(
                    _loop_kernel,
                    1,
                    [core.converged, core.status, core.iterations, core.running, self.k, self.maxiter],
                    device=self.device,
                )
                # CPU eager execution reads only CPU storage; CUDA replay uses graph control.
                if not core.running.numpy()[0]:
                    break
            self._finalize()
        return self.result


def _validate_problem(A, k, tol, atol, maxiter):
    if context.runtime is not None and context.runtime.tape is not None:
        raise RuntimeError("Eigen solvers do not support reverse-mode differentiation")
    operator = aslinearoperator(A)
    shape = _scalar_shape(operator)
    n = shape[0]
    if (len(shape) == 2 and shape[1] != n) or n < 2:
        raise ValueError("A must be square with at least two scalar degrees of freedom")
    if operator.scalar_type != wp.float64:
        raise TypeError("Eigen solvers currently require float64 operators")
    if operator.batch_offsets is not None:
        raise ValueError("A must represent one system")
    if isinstance(k, bool) or isinstance(maxiter, bool):
        raise TypeError("k and maxiter must be integers, not bool")
    k, maxiter = index(k), index(maxiter)
    if not 0 < k < n or maxiter < 1:
        raise ValueError("Require 0 < k < n and maxiter > 0")
    tol, atol = float(1e-8 if tol is None else tol), float(atol)
    if not math.isfinite(tol) or tol <= 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("Require finite tol > 0 and atol >= 0")
    if operator.device.is_capturing:
        raise RuntimeError("Construct eigen state outside capture")
    return n, k, tol, atol, maxiter


def _validate_rows(array, n, device, name, max_rows):
    if array is None:
        return
    if array.dtype != wp.float64:
        raise TypeError(f"{name} must use float64")
    if array.device != device or array.ndim != 2 or array.shape[1] != n or not 1 <= array.shape[0] <= max_rows:
        raise ValueError(f"{name} must have shape (1..{max_rows}, {n}) on {device}")
    if not array.is_contiguous:
        raise ValueError(f"{name} must be contiguous")


def _validate_operand(source, n, device, name):
    if source is None:
        return
    if hasattr(source, "solve"):
        if source.n != n or source.device != device:
            raise ValueError(f"{name} must match A's dimension and device")
        if source.dtype != wp.float64:
            raise TypeError(f"{name} must accept float64 rows")
        return
    operator = aslinearoperator(source)
    if _scalar_shape(operator) not in ((n,), (n, n)) or operator.device != device:
        raise ValueError(f"{name} must match A's shape and device")
    if operator.scalar_type != wp.float64:
        raise TypeError(f"{name} must use float64")
    if operator.batch_offsets is not None:
        raise ValueError(f"{name} must describe one system")


class LOBPCG(EigenSolverState):
    """Compute the smallest algebraic eigenpairs of a symmetric problem.

    Solve ``A u = lambda mass u`` with a symmetric A and SPD mass, or identity
    mass when omitted. Inputs and outputs use float64. Optional KrylovSolve
    preconditioning can use float32 internally. Matrix assembly belongs to the
    caller. Call ``prepare()`` once before executing the state.

    Args:
        A: Symmetric Warp sparse matrix, array or LinearOperator.
        k: Number of requested modes, strictly between zero and system size.
        mass: SPD metric with the same dimensions, dtype and device as A.
        preconditioner: SPD correction operator or prepared multi-RHS solve.
        initial_vectors: Optional independent contiguous starting rows.
        deflation: Known invariant subspace rows to mass-orthonormalize and exclude.
        block_size: Working width, at least k and at most one third of the dimension.
        tol: Positive relative original-equation residual tolerance.
        atol: Nonnegative absolute original-equation residual tolerance.
        maxiter: Positive maximum number of block updates per invocation.
    """

    def __init__(
        self,
        A,
        k,
        *,
        mass=None,
        preconditioner=None,
        initial_vectors=None,
        deflation=None,
        block_size=None,
        tol=None,
        atol=0.0,
        maxiter=100,
    ):
        n, k, tol, atol, maxiter = _validate_problem(A, k, tol, atol, maxiter)
        if isinstance(block_size, bool):
            raise TypeError("block_size must be an integer, not bool")
        width = k if block_size is None else index(block_size)
        if not k <= width or 3 * width > n:
            raise ValueError("Require k <= block_size and 3 * block_size <= n")
        device = aslinearoperator(A).device
        _validate_operand(mass, n, device, "mass")
        _validate_operand(preconditioner, n, device, "preconditioner")
        if hasattr(preconditioner, "solve") and preconditioner.nrhs != width:
            raise ValueError("The preconditioner RHS count must equal block_size")
        _validate_rows(initial_vectors, n, device, "initial_vectors", width)
        _validate_rows(deflation, n, device, "deflation", n - width)
        capacity = max(3 * width, 0 if deflation is None else deflation.shape[0])
        operator = rows_operator(A, capacity)
        metric = Identity(n, device) if mass is None else rows_operator(mass, capacity)
        correction = None
        if preconditioner is not None:
            correction = (
                SolveAction(preconditioner)
                if hasattr(preconditioner, "solve")
                else rows_operator(preconditioner, width)
            )
        use_buckets = device.is_cuda and wp.is_conditional_graph_supported()
        self._core = BlockLOBPCG(
            operator,
            k,
            width,
            mass=metric,
            preconditioner=correction,
            initial_vectors=initial_vectors,
            constraints=deflation,
            normalize_constraints=True,
            tol=tol,
            gram_chunk=4096,
            symmetric_products=True,
            fused_updates=True,
            paired_rotation=True,
            reuse_operator_images=True,
            search_bucket_size=4 if use_buckets else 0,
            preconditioner_bucket_size=4 if use_buckets and isinstance(preconditioner, KrylovSolve) else 0,
        )
        self._finish_construction(operator, metric, k, tol, atol, maxiter)


class KrylovSchur(EigenSolverState):
    """Compute selected symmetric eigenpairs using restarted Krylov iteration.

    Supply a prepared solve of ``A - sigma mass`` for shift-invert, or a mass
    solve for an SPD generalized regular problem. The caller assembles shifted
    matrices and owns inverse resources. Residuals always use the original A
    and mass. Call ``prepare()`` before execution.

    Args:
        A: Symmetric float64 sparse matrix, array or LinearOperator.
        k: Number of requested eigenpairs, strictly between zero and system size.
        mass: Optional SPD metric on the same device.
        which: Selection: LM, SM, LA, SA or BE for standard problems; LM, LA or SA
            for generalized regular problems; LM for shift-invert.
        sigma: Finite real shift, supplied together with inverse.
        inverse: Prepared one-RHS solve of the caller's shifted system.
        mass_inverse: Prepared one-RHS mass solve for generalized regular mode.
        initial_vector: Optional contiguous float64 starting vector of length n.
        ncv: Search dimension, greater than k and no greater than n.
        tol: Positive relative original-equation residual tolerance.
        atol: Nonnegative absolute original-equation residual tolerance.
        maxiter: Positive maximum number of restart cycles per invocation.
    """

    def __init__(
        self,
        A,
        k,
        *,
        mass=None,
        which="LM",
        sigma=None,
        inverse=None,
        mass_inverse=None,
        initial_vector=None,
        ncv=None,
        tol=None,
        atol=0.0,
        maxiter=100,
    ):
        n, k, tol, atol, maxiter = _validate_problem(A, k, tol, atol, maxiter)
        if isinstance(ncv, bool):
            raise TypeError("ncv must be an integer, not bool")
        dimension = min(n, max(2 * k, k + 8)) if ncv is None else index(ncv)
        if not k < dimension <= n:
            raise ValueError("Require k < ncv <= n")
        if which not in ("LM", "SM", "LA", "SA", "BE"):
            raise ValueError("which must be LM, SM, LA, SA or BE")
        if (sigma is None) != (inverse is None):
            raise ValueError("Supply sigma and inverse together")
        if sigma is not None and (not math.isfinite(float(sigma)) or which != "LM" or mass_inverse is not None):
            raise ValueError("Shift-invert requires a finite sigma, which='LM', and no mass_inverse")
        if sigma is None and ((mass is not None) != (mass_inverse is not None)):
            raise ValueError("Generalized regular mode requires mass and mass_inverse together")
        if mass is not None and sigma is None and which not in ("LM", "LA", "SA"):
            raise ValueError("Generalized regular mode supports LM, LA or SA")
        device = aslinearoperator(A).device
        for name, operand in (("mass", mass), ("inverse", inverse), ("mass_inverse", mass_inverse)):
            _validate_operand(operand, n, device, name)
        for operand in (inverse, mass_inverse):
            if operand is not None and (not hasattr(operand, "solve") or operand.nrhs != 1):
                raise ValueError("Inverse actions must be prepared one-RHS solves")
        operator = rows_operator(A, dimension)
        metric = Identity(n, operator.device) if mass is None else rows_operator(mass, dimension)
        if initial_vector is not None:
            if initial_vector.ndim != 1:
                raise ValueError("initial_vector must be one dimensional")
            _validate_rows(initial_vector.reshape((1, n)), n, operator.device, "initial_vector", 1)
        if sigma is not None or mass is not None:
            action = SolveAction(inverse if sigma is not None else mass_inverse, strict=True)
            self._inverse = action
            self._core = GeneralizedKrylovSchur(
                operator,
                metric,
                action,
                k,
                ncv=dimension,
                tol=tol,
                target=0.0 if sigma is None else float(sigma),
                mode="regular" if sigma is None else "shift-invert",
                which=which,
                initial_vector=initial_vector,
                identity_metric=mass is None,
            )
        else:
            self._core = _KrylovSchur(operator, k, ncv=dimension, which=which, tol=tol, initial_vector=initial_vector)
        self._finish_construction(operator, metric, k, tol, atol, maxiter)


def lobpcg(
    A,
    k,
    *,
    mass=None,
    preconditioner=None,
    initial_vectors=None,
    deflation=None,
    block_size=None,
    tol=None,
    atol=0.0,
    maxiter=100,
    run=True,
):
    """Construct LOBPCG and return a prepared result, or the unrun state.

    Args:
        A: Symmetric operator; see :class:`LOBPCG` for the problem contract.
        k: Number of requested eigenpairs.
        mass: Optional SPD metric.
        preconditioner: Optional correction operator or prepared solve.
        initial_vectors: Optional starting rows.
        deflation: Optional known invariant subspace to exclude.
        block_size: Working block width.
        tol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        maxiter: Maximum block updates.
        run: Execute after preparation; otherwise return the constructed state.

    Returns:
        An EigenResult when run is true, otherwise an unprepared LOBPCG state.
    """
    state = LOBPCG(
        A,
        k,
        mass=mass,
        preconditioner=preconditioner,
        initial_vectors=initial_vectors,
        deflation=deflation,
        block_size=block_size,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return state.prepare()() if run else state


def krylov_schur(
    A,
    k,
    *,
    mass=None,
    which="LM",
    sigma=None,
    inverse=None,
    mass_inverse=None,
    initial_vector=None,
    ncv=None,
    tol=None,
    atol=0.0,
    maxiter=100,
    run=True,
):
    """Construct KrylovSchur and return a prepared result, or the unrun state.

    Args:
        A: Symmetric operator; see :class:`KrylovSchur` for the problem contract.
        k: Number of requested eigenpairs.
        mass: Optional SPD metric.
        which: Requested spectral selection.
        sigma: Optional real shift, paired with inverse.
        inverse: Optional prepared shifted solve.
        mass_inverse: Prepared mass solve for generalized regular mode.
        initial_vector: Optional starting vector.
        ncv: Search dimension.
        tol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        maxiter: Maximum restart cycles.
        run: Execute after preparation; otherwise return the constructed state.

    Returns:
        An EigenResult when run is true, otherwise an unprepared KrylovSchur state.
    """
    state = KrylovSchur(
        A,
        k,
        mass=mass,
        which=which,
        sigma=sigma,
        inverse=inverse,
        mass_inverse=mass_inverse,
        initial_vector=initial_vector,
        ncv=ncv,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return state.prepare()() if run else state
