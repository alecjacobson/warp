# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive factorized sparse approximate inverse preconditioning in Warp.

Rows are built independently by growing lower-triangular supports on the graph
of the diagonally equilibrated matrix. For each support S, solve A[S,S] z = e_i
and set G[i,S] = z / sqrt(z[i]). Application uses two sparse products G.T G x;
there is no global sparse factorization or triangular solve.

See https://hypre.readthedocs.io/en/latest/solvers-fsai.html for the adaptive
FSAI algorithm. This implementation uses a row-size cap rather than a step cap.
"""

from functools import lru_cache

import warp as wp
from warp._src import sparse as sp
from warp._src.optim.linear import LinearOperator
from warp._src.utils import array_scan

__all__ = ["FSAI"]


@lru_cache(None)
def _build_kernels(dtype, width, step_size=1):
    best_indices = wp.types.vector(length=step_size, dtype=wp.int32)
    best_scores = wp.types.vector(length=step_size, dtype=dtype)
    indices = wp.types.vector(length=width, dtype=wp.int32)
    vector = wp.types.vector(length=width, dtype=dtype)
    matrix = wp.types.matrix(shape=(width, width), dtype=dtype)

    @wp.func
    def entry(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        i: int,
        j: int,
    ):
        slot = sp.bsr_block_index(i, j, offsets, columns)
        result = dtype(0)
        if slot >= 0:
            result = values[slot]
        return result

    @wp.kernel(enable_backward=False, module="unique")
    def diagonal(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        scale: wp.array[dtype],
        status: wp.array[int],
    ):
        i = wp.tid()
        d = entry(offsets, columns, values, i, i)
        if d > dtype(0) and wp.isfinite(d):
            scale[i] = dtype(1) / wp.sqrt(d)
        else:
            scale[i] = dtype(0)
            wp.atomic_add(status, 0, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def build(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        scale: wp.array[dtype],
        tolerance: dtype,
        pivot_floor: dtype,
        cols_out: wp.array[int],
        vals_out: wp.array[dtype],
        status: wp.array[int],
    ):
        i = wp.tid()
        pattern = indices(-1)
        chol = matrix(dtype(0))
        z = vector(dtype(0))
        v = vector(dtype(0))
        pattern[0] = i
        chol[0, 0] = dtype(1)
        z[0] = dtype(1)
        v[0] = dtype(1)
        size = int(1)
        # Search the frontier of the current support. Duplicates are harmless:
        # candidate scores are evaluated exactly, without a truncated hash table.
        while size < wp.static(width):
            candidates = best_indices(-1)
            scores = best_scores(dtype(0))
            for p in range(size):
                row = pattern[p]
                for e in range(offsets[row], offsets[row + 1]):
                    c = columns[e]
                    selected = bool(c >= i)
                    for q in range(size):
                        if pattern[q] == c:
                            selected = True
                    for q in range(wp.static(step_size)):
                        if candidates[q] == c:
                            selected = True
                    if not selected:
                        residual = dtype(0)
                        for q in range(size):
                            j = pattern[q]
                            residual += entry(offsets, columns, values, c, j) * scale[c] * scale[j] * z[q]
                        score = wp.abs(residual)
                        for rank in range(wp.static(step_size)):
                            if score > scores[rank] or (score == scores[rank] and c < candidates[rank]):
                                for rev in range(wp.static(step_size - 1)):
                                    dest = wp.static(step_size - 1) - rev
                                    if dest > rank:
                                        candidates[dest] = candidates[dest - 1]
                                        scores[dest] = scores[dest - 1]
                                candidates[rank] = c
                                scores[rank] = score
                                break
            if candidates[0] < 0:
                break
            old_z0 = z[0]
            stopped = bool(False)
            for rank in range(wp.static(step_size)):
                best = candidates[rank]
                if best < 0 or size == wp.static(width):
                    break
                # Border the Cholesky factor of the selected principal submatrix.
                w = vector(dtype(0))
                pivot = dtype(1)
                for p in range(size):
                    a = entry(offsets, columns, values, best, pattern[p]) * scale[best] * scale[pattern[p]]
                    for q in range(p):
                        a -= chol[p, q] * w[q]
                    w[p] = a / chol[p, p]
                    pivot -= w[p] * w[p]
                if pivot <= pivot_floor or not wp.isfinite(pivot):
                    wp.atomic_add(status, 1, 1)
                    stopped = True
                    break
                pattern[size] = best
                for p in range(size):
                    chol[size, p] = w[p]
                chol[size, size] = wp.sqrt(pivot)
                # Existing forward-solve entries are unchanged by bordering L.
                rhs = dtype(0)
                for p in range(size):
                    rhs -= w[p] * v[p]
                v[size] = rhs / chol[size, size]
                size += 1
            # A[S,S] z = e_0; no global factorization or triangular dependency.
            for rev in range(size):
                p = size - 1 - rev
                a = v[p]
                for q in range(p + 1, size):
                    a -= chol[q, p] * z[q]
                z[p] = a / chol[p, p]
            # psi=1/z[0]. Relative reduction is 1-old_z0/new_z0.
            if stopped or (tolerance > dtype(0) and dtype(1) - old_z0 / z[0] <= tolerance):
                break
        norm = wp.sqrt(z[0])
        valid = bool(True)
        for p in range(wp.static(width)):
            e = i * wp.static(width) + p
            cols_out[e] = -1
            vals_out[e] = dtype(0)
            if p < size:
                cols_out[e] = pattern[p]
                vals_out[e] = z[p] / norm * scale[pattern[p]]
                if not wp.isfinite(vals_out[e]):
                    valid = False
                if p == 0 and vals_out[e] <= dtype(0):
                    valid = False
        if not valid:
            wp.atomic_add(status, 2, 1)

    return diagonal, build


@wp.kernel(enable_backward=False, module="unique")
def _validate_float32_factor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.float32],
    status: wp.array[int],
):
    row = wp.tid()
    invalid = bool(False)
    for e in range(offsets[row], offsets[row + 1]):
        if not wp.isfinite(values[e]):
            invalid = True
        if columns[e] == row and values[e] <= wp.float32(0):
            invalid = True
    if invalid:
        wp.atomic_add(status, 0, 1)


class FSAI(LinearOperator):
    """Build an adaptive factorized sparse approximate inverse of an SPD BSR matrix.

    This :class:`warp.optim.linear.LinearOperator` applies ``G.T @ G`` and can be
    passed as ``M`` to :func:`warp.optim.linear.cg` or :func:`warp.optim.linear.cr`.
    Rows of the lower-triangular factor ``G`` are built independently on the matrix
    device. Square blocks are expanded into scalar entries, so this is scalar FSAI
    even when the input uses block storage. No host numerical backend is used.

    Args:
        A: Square :class:`warp.sparse.BsrMatrix` with square blocks and ``float16``,
            ``float32`` or ``float64`` scalar entries. Both triangles must be stored. Symmetry
            and positive definiteness are caller preconditions, not checked globally.
            Stored columns must be sorted and unique within each row, as produced
            by the standard BSR construction functions.
        max_row_size: Maximum scalar entries per factor row, including its diagonal,
            in [1, 64]. One entry gives Jacobi preconditioning. Larger supports use
            quadratic per-thread setup workspace and increase setup and apply costs.
        kap_tolerance: Relative local energy improvement below which support growth
            stops, in [0, 1). Zero disables this stopping test. It is checked after
            each batch, so changing ``max_step_size`` may require retuning it.
        pivot_floor: Smallest accepted pivot of the diagonally equilibrated local
            Cholesky factorization, in (0, 1). ``None`` selects 1e-3 for ``float16``,
            1e-6 for ``float32`` and 1e-12 for ``float64``. Setup truncates a row
            before a rejected pivot.
        apply_lanes: CUDA threads cooperating on each scalar row during application;
            one of 1, 2, 4, 8, 16, or 32. One uses the ordinary sparse product.
            CPU uses the ordinary product regardless of this setting.
        factor_dtype: Factor storage type: the matrix scalar type, or ``wp.float32``
            to compress a ``float64`` factor. ``None`` uses the matrix scalar type.
            Products accumulate in the matrix scalar type; the transpose is formed
            from the same rounded factor, preserving the Gram form.
        reuse_pattern: Retain a plan for :meth:`update` on the original factor
            supports. Requires compact input storage; use :func:`warp.sparse.bsr_copy`
            to compact a padded matrix first. Ordinary construction accepts either.
        max_step_size: Maximum distinct largest-residual frontier entries selected
            per growth step, in [1, 64]. Batching reduces searches but can change the
            selected supports. A short batch does not exhaust the row-size budget.

    Attributes:
        G: Canonical scalar BSR factor with finite entries and a positive diagonal.
        GT: Stored transpose of ``G``, including any storage rounding.
        truncated_rows: Number of setup rows stopped by the pivot safeguard.
        source: Matrix used by the most recent successful construction or update.

    Note:
        Construction and updates synchronize and cannot be captured. Application is
        CUDA graph-capturable after warming up the kernels. Instances own scratch:
        do not apply the same instance concurrently on independent streams, or
        overlap an update with application. Treat ``G`` and ``GT`` as read-only.
        Rebuild or call :meth:`update` after changing the input values. This
        preconditioner does not provide automatic differentiation.

    Raises:
        TypeError: The matrix scalar type is unsupported.
        ValueError: The matrix shape or an option is invalid, a diagonal is missing,
            nonpositive or nonfinite, or the factor cannot be represented by the
            selected storage type with finite entries and a positive diagonal.

    Example:
        .. code-block:: python

            from warp.optim.linear import FSAI, cg

            M = FSAI(A, max_row_size=16, reuse_pattern=True)
            cg(A, b, x, M=M)
            # After updating A.values without changing A's sparsity:
            M.update()
            cg(A, b, x, M=M)
    """

    def __init__(
        self,
        A: sp.BsrMatrix,
        max_row_size: int = 8,
        kap_tolerance: float = 1.0e-3,
        pivot_floor: float | None = None,
        apply_lanes: int = 1,
        factor_dtype: type | None = None,
        reuse_pattern: bool = False,
        max_step_size: int = 1,
    ):
        if not isinstance(A, sp.BsrMatrix) or A.shape[0] != A.shape[1] or A.block_shape[0] != A.block_shape[1]:
            raise ValueError("FSAI requires a square BSR matrix with square blocks")
        if A.scalar_type not in (wp.float16, wp.float32, wp.float64):
            raise TypeError("FSAI supports float16, float32 and float64")
        if not isinstance(max_row_size, int) or not 1 <= max_row_size <= 64:
            raise ValueError("max_row_size must be an integer in [1, 64]")
        if not isinstance(max_step_size, int) or not 1 <= max_step_size <= 64:
            raise ValueError("max_step_size must be an integer in [1, 64]")
        if not 0 <= kap_tolerance < 1:
            raise ValueError("kap_tolerance must be in [0, 1)")
        if pivot_floor is None:
            pivot_floor = {wp.float16: 1.0e-3, wp.float32: 1.0e-6, wp.float64: 1.0e-12}[A.scalar_type]
        if not 0 < pivot_floor < 1:
            raise ValueError("pivot_floor must be in (0, 1)")
        if not isinstance(apply_lanes, int) or apply_lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError("apply_lanes must be one of 1, 2, 4, 8, 16, 32")
        if factor_dtype is None:
            factor_dtype = A.scalar_type
        if factor_dtype != A.scalar_type and not (A.scalar_type == wp.float64 and factor_dtype == wp.float32):
            raise ValueError("factor_dtype must be the matrix scalar type, or wp.float32 for a float64 matrix")
        if reuse_pattern and A.row_counts is not None:
            raise ValueError("reuse_pattern requires compact BSR storage; canonicalize with bsr_copy")
        self.max_row_size = max_row_size
        self.max_step_size = max_step_size
        self.factor_dtype = factor_dtype
        self.apply_lanes = apply_lanes
        self.source = A
        # Settle the source count before scalarization: nnz can otherwise be a
        # very loose assembly upper bound, causing oversized temporary buffers.
        A.nnz_sync()
        # Also canonicalizes padded storage. No host matrix staging.

        if A.row_counts is not None:
            scalar = sp.bsr_copy(A, block_shape=(1, 1))
        elif A.dtype == A.scalar_type and not reuse_pattern:
            scalar = A
        else:
            scalar = _scalarize_compact(A)
        n = scalar.nrow
        dtype, device = scalar.scalar_type, scalar.device
        scale = wp.empty(n, dtype=dtype, device=device)
        status = wp.zeros(3, dtype=int, device=device)
        diagonal, build = _build_kernels(dtype, max_row_size, max_step_size)
        wp.launch(
            diagonal,
            n,
            [scalar.offsets, scalar.columns, scalar.values, scale, status],
            device=device,
        )
        cols = wp.empty(n * max_row_size, dtype=int, device=device)
        vals = wp.empty(n * max_row_size, dtype=dtype, device=device)
        wp.launch(
            build,
            n,
            [
                scalar.offsets,
                scalar.columns,
                scalar.values,
                scale,
                dtype(kap_tolerance),
                dtype(pivot_floor),
                cols,
                vals,
                status,
            ],
            device=device,
        )
        diagnostics = status.numpy()
        if diagnostics[0]:
            raise ValueError(f"FSAI requires finite positive diagonals ({diagnostics[0]} invalid rows)")
        self.truncated_rows = int(diagnostics[1])
        if diagnostics[2]:
            raise ValueError("FSAI factor lost finite entries or a positive diagonal")

        self.G = _pack_factor(n, max_row_size, cols, vals)
        if factor_dtype != dtype:
            self.G = sp.bsr_copy(self.G, scalar_type=factor_dtype)
            status.zero_()
            wp.launch(
                _validate_float32_factor,
                n,
                [self.G.offsets, self.G.columns, self.G.values, status],
                device=device,
            )
            if status.numpy()[0]:
                raise ValueError("factor_dtype conversion lost finite entries or a positive diagonal")
        self.GT = sp.bsr_transposed(self.G)
        self._tmp = wp.empty(n, dtype=dtype, device=device)
        super().__init__(A.shape, A.dtype, device, self._apply)
        self._refit = None
        if reuse_pattern:
            self._refit = _RefitPlan(self, A, scalar, max_row_size, pivot_floor)

    def update(self, A: sp.BsrMatrix | None = None) -> "FSAI":
        """Refit on the original factor supports, preserving captured apply buffers.

        Requires ``reuse_pattern=True`` and identical compact BSR topology/storage.
        This is not a new adaptive pattern search. Rebuild when convergence worsens.
        Failed validation leaves the previous factors intact. Updates synchronize.

        Args:
            A: Matrix with new numerical values. ``None`` reuses :attr:`source`.

        Returns:
            This preconditioner with updated numerical factors.

        Raises:
            ValueError: No reuse plan exists, matrix metadata or topology changed,
                or a local factorization or storage conversion failed. Unlike
                initial construction, refits reject a bad pivot without truncating
                the fixed support. Rebuild to select new supports.
        """
        if self._refit is None:
            raise ValueError("Construct FSAI with reuse_pattern=True before updating")
        A = self.source if A is None else A
        if not isinstance(A, sp.BsrMatrix):
            raise ValueError("update requires a Warp BSR matrix")
        self._refit.update(self, A, _build_kernels(self.scalar_type, self.max_row_size, self.max_step_size)[0])
        self.source = A
        return self

    def _apply(self, x, y, z, alpha, beta):
        x = x.view(self.scalar_type).flatten()
        z = z.view(self.scalar_type).flatten()
        y = y.view(self.scalar_type).flatten()
        if alpha != 0.0:
            _matvec(self.G, x, self._tmp, self._tmp, 1.0, 0.0, self.apply_lanes)
        # x is fully consumed before writing z, including when x, y, z alias.
        _matvec(self.GT, self._tmp, y, z, alpha, beta, self.apply_lanes)


@lru_cache(None)
def _pack_kernels(dtype, width):
    ivec = wp.types.vector(length=width, dtype=int)
    vec = wp.types.vector(length=width, dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def sort_rows(columns: wp.array[int], values: wp.array[dtype], counts: wp.array[int]):
        i = wp.tid()
        cols = ivec(-1)
        vals = vec(dtype(0))
        size = int(0)
        for p in range(wp.static(width)):
            c = columns[i * wp.static(width) + p]
            if c >= 0:
                v = values[i * wp.static(width) + p]
                q = size
                while q > 0:
                    if cols[q - 1] <= c:
                        break
                    cols[q] = cols[q - 1]
                    vals[q] = vals[q - 1]
                    q -= 1
                cols[q] = c
                vals[q] = v
                size += 1
        counts[i + 1] = size
        for p in range(size):
            columns[i * wp.static(width) + p] = cols[p]
            values[i * wp.static(width) + p] = vals[p]

    @wp.kernel(enable_backward=False, module="unique")
    def pack(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        outcols: wp.array[int],
        outvals: wp.array[dtype],
    ):
        i = wp.tid()
        for e in range(offsets[i], offsets[i + 1]):
            p = i * wp.static(width) + e - offsets[i]
            outcols[e] = columns[p]
            outvals[e] = values[p]

    return sort_rows, pack


def _pack_factor(n, width, columns, values):
    result = sp.bsr_zeros(n, n, values.dtype, device=values.device)
    counts = wp.zeros(n + 1, dtype=int, device=values.device)
    sort, pack = _pack_kernels(values.dtype, width)
    wp.launch(sort, n, [columns, values, counts], device=values.device)
    array_scan(counts, result.offsets)
    result.notify_nnz_changed()
    wp.launch(
        pack,
        n,
        [result.offsets, columns, values, result.columns, result.values],
        device=values.device,
    )
    return result


@lru_cache(None)
def _scalarize_kernel(dtype, block_size):
    @wp.kernel(enable_backward=False, module="unique")
    def scalarize(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        outoff: wp.array[int],
        outcol: wp.array[int],
        outval: wp.array[dtype],
    ):
        i = wp.tid()
        r = i // wp.static(block_size)
        component = i % wp.static(block_size)
        start = offsets[r]
        count = offsets[r + 1] - start
        base = start * wp.static(block_size * block_size) + component * count * wp.static(block_size)
        outoff[i] = base
        for p in range(count):
            for c in range(wp.static(block_size)):
                dest = base + p * wp.static(block_size) + c
                outcol[dest] = columns[start + p] * wp.static(block_size) + c
                outval[dest] = values[
                    (start + p) * wp.static(block_size * block_size) + component * wp.static(block_size) + c
                ]
        if i == outoff.shape[0] - 2:
            outoff[i + 1] = base + count * wp.static(block_size)

    return scalarize


def _scalarize_compact(A):
    """Expand canonical compact square BSR directly, retaining numerical zeros."""
    b = A.block_shape[0]
    result = sp.bsr_zeros(A.shape[0], A.shape[1], A.scalar_type, device=A.device)
    count = A.nnz_sync() * b * b
    result.columns = wp.empty(count, dtype=int, device=A.device)
    result.values = wp.empty(count, dtype=A.scalar_type, device=A.device)
    wp.launch(
        _scalarize_kernel(A.scalar_type, b),
        result.nrow,
        [
            A.offsets,
            A.columns,
            A.values.view(A.scalar_type).flatten(),
            result.offsets,
            result.columns,
            result.values,
        ],
        device=A.device,
    )
    result.notify_nnz_changed(nnz=count)
    return result


@wp.kernel(enable_backward=False, module="unique")
def _compare_topology(a: wp.array[int], b: wp.array[int], bad: wp.array[int]):
    i = wp.tid()
    if a[i] != b[i]:
        wp.atomic_add(bad, 0, 1)


@wp.kernel(enable_backward=False, module="unique")
def _source_map(
    offsets: wp.array[int],
    columns: wp.array[int],
    source_offsets: wp.array[int],
    source_columns: wp.array[int],
    block_size: int,
    mapping: wp.array[int],
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        slot = sp.bsr_block_index(i // block_size, j // block_size, source_offsets, source_columns)
        mapping[e] = slot * block_size * block_size + (i % block_size) * block_size + j % block_size


@wp.kernel(enable_backward=False, module="unique")
def _transpose_map(
    offsets: wp.array[int],
    columns: wp.array[int],
    toff: wp.array[int],
    tcol: wp.array[int],
    mapping: wp.array[int],
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        mapping[e] = sp.bsr_block_index(columns[e], i, toff, tcol)


@lru_cache(None)
def _refit_kernels(dtype, storage, width):
    vec = wp.types.vector(length=width, dtype=dtype)
    mat = wp.types.matrix(shape=(width, width), dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def gather(source: wp.array[dtype], mapping: wp.array[int], out: wp.array[dtype]):
        e = wp.tid()
        out[e] = source[mapping[e]]

    @wp.kernel(enable_backward=False, module="unique")
    def refit(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        goff: wp.array[int],
        gcol: wp.array[int],
        scale: wp.array[dtype],
        floor: dtype,
        out: wp.array[storage],
        bad: wp.array[int],
    ):
        i = wp.tid()
        start = goff[i]
        size = goff[i + 1] - start
        chol = mat(dtype(0))
        valid = bool(True)
        for p in range(size):
            r = gcol[start + p]
            for q in range(p + 1):
                c = gcol[start + q]
                slot = sp.bsr_block_index(r, c, offsets, columns)
                a = dtype(0)
                if slot >= 0:
                    a = values[slot] * scale[r] * scale[c]
                for k in range(q):
                    a -= chol[p, k] * chol[q, k]
                if p == q:
                    if a <= floor or not wp.isfinite(a):
                        valid = False
                        a = dtype(1)
                    chol[p, q] = wp.sqrt(a)
                else:
                    chol[p, q] = a / chol[q, q]
        z = vec(dtype(0))
        for p in range(size):
            a = dtype(0)
            if gcol[start + p] == i:
                a = dtype(1)
            for q in range(p):
                a -= chol[p, q] * z[q]
            z[p] = a / chol[p, p]
        for rev in range(size):
            p = size - 1 - rev
            a = z[p]
            for q in range(p + 1, size):
                a -= chol[q, p] * z[q]
            z[p] = a / chol[p, p]
        # Canonical lower-triangular support: the diagonal is last.
        norm = wp.sqrt(z[size - 1])
        for p in range(size):
            value = storage(z[p] / norm * scale[gcol[start + p]])
            out[start + p] = value
            if not wp.isfinite(value):
                valid = False
            if p == size - 1 and value <= storage(0):
                valid = False
        if not valid:
            wp.atomic_add(bad, 0, 1)

    @wp.kernel(enable_backward=False, module="unique")
    def transpose(values: wp.array[storage], mapping: wp.array[int], out: wp.array[storage]):
        e = wp.tid()
        out[mapping[e]] = values[e]

    return gather, refit, transpose


class _RefitPlan:
    def __init__(self, owner, A, scalar, width, floor):
        self.signature = (A.shape, A.dtype, A.device, A.row_counts is not None)
        self.topology = [wp.clone(a) for a in self.arrays(A)]
        self.scalar = scalar
        self.floor = floor
        self.gather, self.refit, self.scatter_transpose = _refit_kernels(A.scalar_type, owner.factor_dtype, width)
        self.mapping = wp.empty(scalar.nnz_sync(), dtype=int, device=A.device)
        # Padded source rows need compact search endpoints, not unused slots.
        # _source_map's search is safe for compact input; padded input is rejected
        # explicitly for opt-in reuse, while ordinary construction supports it.
        wp.launch(
            _source_map,
            scalar.nrow,
            [scalar.offsets, scalar.columns, A.offsets, A.columns, A.block_shape[0], self.mapping],
            device=A.device,
        )
        self.transpose = wp.empty(owner.G.nnz_sync(), dtype=int, device=A.device)
        wp.launch(
            _transpose_map,
            owner.G.nrow,
            [owner.G.offsets, owner.G.columns, owner.GT.offsets, owner.GT.columns, self.transpose],
            device=A.device,
        )
        self.values = wp.empty_like(scalar.values)
        self.candidate = wp.empty_like(owner.G.values)
        self.scale = wp.empty(scalar.nrow, dtype=A.scalar_type, device=A.device)
        self.bad = wp.zeros(2, dtype=int, device=A.device)

    @staticmethod
    def arrays(A):
        nnz = A.nnz_sync()
        # Warp rejects zero-length slices, including [:0] on an empty array.
        columns = A.columns[:nnz] if nnz else wp.empty(0, dtype=int, device=A.device)
        return [A.offsets[: A.nrow + 1], columns] + ([A.row_counts] if A.row_counts is not None else [])

    def update(self, owner, A, diagonal):
        if (A.shape, A.dtype, A.device, A.row_counts is not None) != self.signature:
            raise ValueError("FSAI update requires the original shape, dtype, device and topology")
        incoming = self.arrays(A)
        if any(a.shape != b.shape for a, b in zip(incoming, self.topology, strict=True)):
            raise ValueError("FSAI update requires identical topology storage")
        self.bad.zero_()
        for a, b in zip(incoming, self.topology, strict=True):
            wp.launch(_compare_topology, a.size, [a, b, self.bad], device=A.device)
        if self.bad.numpy()[0]:
            raise ValueError("FSAI update requires identical sparsity; rebuild to change it")
        if A.values.size < A.nnz_sync():
            raise ValueError("FSAI update requires a values array covering every stored block")
        wp.launch(
            self.gather,
            self.mapping.size,
            [A.values.view(A.scalar_type).flatten(), self.mapping, self.values],
            device=A.device,
        )
        s = self.scalar
        wp.launch(
            diagonal,
            s.nrow,
            [s.offsets, s.columns, self.values, self.scale, self.bad],
            device=A.device,
        )
        wp.launch(
            self.refit,
            s.nrow,
            [
                s.offsets,
                s.columns,
                self.values,
                owner.G.offsets,
                owner.G.columns,
                self.scale,
                A.scalar_type(self.floor),
                self.candidate,
                self.bad,
            ],
            device=A.device,
        )
        if self.bad.numpy()[0]:
            raise ValueError("FSAI refit requires finite positive diagonals and SPD local supports")
        wp.copy(owner.G.values, self.candidate)
        wp.launch(
            self.scatter_transpose,
            self.transpose.size,
            [owner.G.values, self.transpose, owner.GT.values],
            device=A.device,
        )


@lru_cache(None)
def _grouped_kernel(dtype, lanes, storage_type):
    rows = 128 // lanes

    @wp.kernel(enable_backward=False, module="unique")
    def mv(
        n: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[storage_type],
        x: wp.array[dtype],
        y: wp.array[dtype],
        z: wp.array[dtype],
        alpha: dtype,
        beta: dtype,
    ):
        row, lane = wp.tid()
        result = dtype(0)
        if row < n and alpha != dtype(0):
            for e in range(offsets[row] + lane, offsets[row + 1], wp.static(lanes)):
                result += dtype(values[e]) * x[columns[e]]
        t = wp.tile_reshape(wp.tile(result), shape=(wp.static(rows), wp.static(lanes)))
        sums = wp.tile_sum(t, axis=1)
        result = wp.tile_extract(sums, row % wp.static(rows))
        if lane == 0 and row < n:
            result *= alpha
            if beta != dtype(0):
                result += beta * y[row]
            z[row] = result

    return mv


def _grouped_mv(a, x, y, z, alpha, beta, lanes):
    rows = 128 // lanes
    wp.launch(
        _grouped_kernel(x.dtype, lanes, a.scalar_type),
        dim=(((a.nrow + rows - 1) // rows) * rows, lanes),
        inputs=[
            a.nrow,
            a.offsets,
            a.columns,
            a.values,
            x,
            y,
            z,
            x.dtype(alpha),
            x.dtype(beta),
        ],
        device=a.device,
        block_dim=128,
    )


@lru_cache(None)
def _mixed_kernel(storage_type, dtype):
    @wp.kernel(enable_backward=False, module="unique")
    def mv(
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[storage_type],
        x: wp.array[dtype],
        y: wp.array[dtype],
        z: wp.array[dtype],
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        result = dtype(0)
        if alpha != dtype(0):
            for e in range(offsets[row], offsets[row + 1]):
                result += dtype(values[e]) * x[columns[e]]
            result *= alpha
        if beta != dtype(0):
            result += beta * y[row]
        z[row] = result

    return mv


def _matvec(a, x, y, z, alpha, beta, lanes):
    """Canonical scalar CSR; x and z must not alias for the tiled path."""
    if a.device.is_cuda and lanes > 1:
        _grouped_mv(a, x, y, z, alpha, beta, lanes)
    elif a.scalar_type != x.dtype:
        wp.launch(
            _mixed_kernel(a.scalar_type, x.dtype),
            a.nrow,
            [a.offsets, a.columns, a.values, x, y, z, x.dtype(alpha), x.dtype(beta)],
            device=a.device,
        )
    else:
        if beta != 0.0 and z.ptr != y.ptr:
            wp.copy(z, y)
        sp.bsr_mv(a, x, z, alpha=alpha, beta=beta)
