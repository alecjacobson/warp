# SPDX-FileCopyrightText: Copyright (c) 2026 warpack contributors
# SPDX-License-Identifier: MIT

"""Reusable, graph-capturable symmetric block eigensolver."""

import math
from operator import index as integer_index

import warp as wp

from . import kernels as K
from . import krylov_kernels as KR
from . import linalg_tiles as KT
from ._operators import SparseOperator
from .generalized import generalized_outputs, generalized_residual
from .projected import GlobalJacobi, jacobi_kernel


class ProjectedState:
    """Own projected algebra and workspace shared by the eigen algorithms."""

    def __init__(
        self,
        operator,
        k,
        ncv=None,
        which="SA",
        iteration_operator=None,
        tol=1.0e-9,
        target=None,
        initial_vectors=None,
    ):
        self.op = operator
        self.iter_op = iteration_operator or operator
        self.n, self.device = operator.n, operator.device
        self.k = k
        self.m = min(self.n, max(2 * k, k + 8) if ncv is None else ncv)
        if not 1 <= k <= self.m <= self.n:
            raise ValueError("require 1 <= k <= ncv <= n")
        if which not in ("SA", "LA", "LM", "SM", "BE"):
            raise ValueError("which must be SA, LA, LM, SM, or BE")
        self.which = ("SA", "LA", "LM", "SM", "BE").index(which)
        self.target = target
        if target is not None:
            self.which = 5
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if iteration_operator is None and self.m < self.n and which != "LM":
            if type(self) is ProjectedState:
                raise ValueError(
                    "Block iteration needs an explicit iteration_operator for this selection; use KrylovSchur otherwise"
                )
        self.tol = tol
        self.initial_vectors = initial_vectors
        if initial_vectors is not None:
            if (
                initial_vectors.dtype != wp.float64
                or initial_vectors.ndim != 2
                or initial_vectors.shape[1] != self.n
                or not 1 <= initial_vectors.shape[0] <= self.m
            ):
                raise ValueError("initial_vectors must have shape (1..ncv, n) and dtype FP64")
            if initial_vectors.device != self.device:
                raise ValueError("initial_vectors must be on the operator device")

        def empty(shape, dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.q, self.z = empty((self.m, self.n)), empty((self.m, self.n))
        self._initial_rows = self.q[: initial_vectors.shape[0]] if initial_vectors is not None else None
        self.g, self.l, self.v = [empty((self.m, self.m)) for _ in range(3)]
        self.coeff = empty((self.m, self.m))
        self.jtmp, self.vtmp = empty((self.m, self.m)), empty((self.m, self.m))
        self.cs, self.ss = empty(self.m), empty(self.m)
        self.partners = empty(self.m, wp.int32)
        self.ritz = empty(self.m)
        self.order = empty(self.m, wp.int32)
        self.sort_work = empty(self.m, wp.int32)
        self.status = empty(1, wp.int32)
        self.eigenvalues, self.residuals = empty(k), empty(k)
        self.eigenvectors, self.ax = empty((k, self.n)), empty((k, self.n))
        self.norms = empty((k, 1))
        self.converged = empty(1, wp.int32)
        self.iterations = empty(1, wp.int32)
        self.running = empty(1, wp.int32)
        # Measured Ada crossover: the fused global rounds outperform the
        # shared kernel from 48 columns. Preserve other architectures' policy.
        self._prefer_global_projected = self.m >= 48 and self.device.is_cuda and self.device.arch == 89
        self._global_jacobi = None
        if (
            (self.m > 64 or self._prefer_global_projected)
            and self.m % 2 == 0
            and self.device.is_cuda
            and wp.is_conditional_graph_supported()
        ):
            self._global_jacobi = GlobalJacobi(self.g, self.v, self.cs, self.ss, self.partners, self.jtmp, self.vtmp)

    def rotate(self, q, out):
        wp.launch(
            K.rotation_coefficients,
            (out.shape[0], self.m),
            [self.v, self.order, self.coeff],
            device=self.device,
        )
        wp.launch_tiled(
            KT.rotate_tiled,
            dim=((out.shape[0] + 3) // 4, (self.n + 31) // 32),
            inputs=[q, self.coeff, out],
            block_dim=128,
            device=self.device,
        )

    def diagonalize(self):
        self._diagonalize_projected()
        if self.which >= 4:
            wp.launch(
                K.sort_extra,
                1,
                [
                    self.ritz,
                    self.sort_work,
                    self.order,
                    self.which,
                    float(self.target or 0.0),
                ],
                device=self.device,
            )

    def _diagonalize_projected(self):
        if not self.device.is_cuda or self.m % 2:
            wp.launch(
                K.jacobi,
                1,
                [self.g, self.v, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        if self._global_jacobi is not None:
            self._global_jacobi.apply()
            wp.launch(
                K.sort_diagonal,
                self.m,
                [self.g, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        if self.m <= 64:
            wp.launch_tiled(
                jacobi_kernel(self.m),
                dim=1,
                inputs=[self.g, self.v],
                block_dim=256,
                device=self.device,
            )
            wp.launch(
                K.sort_diagonal,
                self.m,
                [self.g, self.ritz, self.order, self.which],
                device=self.device,
            )
            return
        wp.launch(K.jacobi_init, (self.m, self.m), [self.g, self.v], device=self.device)
        for _ in range(30):
            for r in range(self.m - 1):
                wp.launch(
                    K.jacobi_pairs,
                    self.m // 2,
                    [self.g, self.cs, self.ss, self.partners, r],
                    device=self.device,
                )
                wp.launch(
                    K.jacobi_right,
                    (self.m, self.m),
                    [
                        self.g,
                        self.v,
                        self.jtmp,
                        self.vtmp,
                        self.cs,
                        self.ss,
                        self.partners,
                    ],
                    device=self.device,
                )
                wp.launch(
                    K.jacobi_left,
                    (self.m, self.m),
                    [
                        self.g,
                        self.v,
                        self.jtmp,
                        self.vtmp,
                        self.cs,
                        self.ss,
                        self.partners,
                    ],
                    device=self.device,
                )
        wp.launch(
            K.sort_diagonal,
            self.m,
            [self.g, self.ritz, self.order, self.which],
            device=self.device,
        )

    def solve(self, iterations=50, seed=42):
        """Initialize and execute a fixed iteration budget, updating device outputs.

        Args:
            iterations: Fixed number of outer iterations/restart cycles, not inner sparse-solver
                iterations.
            seed: Integer seed used for generated start directions.

        Returns:
            This solver instance. This is not a host tuple of eigenvalues and eigenvectors;
            outputs remain device arrays.
        """
        iterations = integer_index(iterations)
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        self.initialize(seed)
        for _ in range(iterations):
            self.step()
        self.finalize()
        self.iterations.fill_(iterations)
        return self


class KrylovSchur(ProjectedState):
    """Thick-restarted symmetric Krylov / Rayleigh-Ritz iteration.

    Expands the retained Ritz space by operator applications, with two-pass
    classical Gram-Schmidt. Unlike orthogonal iteration, this targets algebraic
    spectrum ends without requiring a spectral shift bound.

    An optional nonzero FP64 ``initial_vector`` supplies the starting direction.
    Initialization normalizes its current device contents, so captured replays
    can use updated starts without recapturing. Otherwise ``seed`` controls the
    random start.

    Adaptive replay also waits for independent-chain exploration (unless the
    basis spans the full problem). Thus it can continue after all ``k`` returned
    residuals pass. ``converged`` counts residual-passing pairs; neither that
    count nor a fixed iteration budget proves global spectrum completeness.

    Unless sparse products are explicitly reapplied after restart, residuals
    use rotated cached operator images (including inverse and generic callback
    operators). At tight tolerances these are estimates, not fresh applications of
    the operator. ``EigenpairEvaluation(..., eigenvalues=solver.eigenvalues)``
    checks the returned pairs with fresh products; use ``rhs_width=1`` for a
    one-RHS inverse. This optional final check does not alter iteration stopping.

    Args:
        operator: Real symmetric operator exposing n, device and apply(x, y) on FP64 row arrays.
        k: Number of requested eigenpairs (positive integer).
        ncv: Maximum basis dimension. None uses min(n, max(2*k, k+8)); require k < ncv <= n
            after clipping.
        which: Selection: LM/SM (largest/smallest magnitude), LA/SA (largest/smallest
            algebraic), or BE (both ends).
        tol: Positive tolerance for ||A*x-lambda*x|| / max(1, abs(lambda)); cached-product
            residuals are estimates.
        initial_vector: Optional finite nonzero FP64 device array of shape (n,). Read on each
            initialization; not modified.

    Attributes:
        eigenvalues: FP64 device array (k,) of original-problem eigenvalues.
        eigenvectors: FP64 device array (k, n); one eigenvector per row.
        residuals: FP64 device array (k,) of normalized eigenpair residuals.
        converged: int32 device array (1,); number of residual-passing requested pairs.
        status: int32 device array (1,); zero means no recorded numerical failure, not
            convergence.
        iterations: int32 device array (1,); executed outer iterations/restart cycles.
    """

    def __init__(self, operator, k, ncv=None, which="LM", tol=1.0e-9, initial_vector=None):
        super().__init__(operator, k, ncv=ncv, which=which, tol=tol)
        self.initial_vector = initial_vector
        if initial_vector is not None:
            if initial_vector.dtype != wp.float64 or initial_vector.size != self.n:
                raise ValueError("initial_vector must contain n FP64 entries")
            if initial_vector.device != self.device:
                raise ValueError("initial_vector must be on the operator device")
            self.initial_vector = initial_vector.reshape((1, self.n))
        # Extremal targets need expansion room to resolve clusters. Interior SM
        # Ritz selection instead needs the larger retained space: reducing it
        # loses accuracy on the transformed-SM SymEigsShift_17 regression.
        retain_limit = self.m - 1 if which == "SM" else max(k, self.m // 2)
        self.keep = min(k + 4, retain_limit)
        # Sparse products can be cheaper and more accurate than a dense basis
        # rotation. Keep cached products for dense operators and inverses.
        self._reapply_after_restart = (
            isinstance(operator, SparseOperator)
            and operator.matrix.nnz / operator.matrix.nrow * operator.matrix.block_shape[1] <= self.m // 2
        )
        if self.m <= k:
            raise ValueError("Krylov restart requires ncv > k")
        self.dots = wp.zeros((self.m, 1), dtype=wp.float64, device=self.device)
        self.norm = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        self.source_norm = wp.zeros_like(self.norm)
        self.breakdown = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.recoveries = wp.zeros_like(self.breakdown)
        self.dot_parts = wp.empty((self.m, (self.n + 1023) // 1024), dtype=wp.float64, device=self.device)
        self.norm_parts = wp.empty((self.n + 1023) // 1024, dtype=wp.float64, device=self.device)
        self.retained = wp.empty((self.keep, self.n), dtype=wp.float64, device=self.device)
        self.retained_ax = wp.empty_like(self.retained)
        self.restart_scores = wp.zeros((self.k, 2), dtype=wp.float64, device=self.device)
        self.restart_seed = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.physical_norms = wp.zeros((self.k, 3), dtype=wp.float64, device=self.device)
        self.views = [self.q[j : j + 1] for j in range(self.m)]
        self.zviews = [self.z[j : j + 1] for j in range(self.m)]

    def solve(self, iterations=50, seed=42):
        """Initialize and execute a fixed iteration budget, updating device outputs.

        Args:
            iterations: Fixed number of outer iterations/restart cycles, not inner sparse-solver
                iterations.
            seed: Integer seed used for generated start directions.

        Returns:
            This solver instance. This is not a host tuple of eigenvalues and eigenvectors;
            outputs remain device arrays.
        """
        iterations = integer_index(iterations)
        if iterations < 1:
            raise ValueError("Krylov iteration needs at least one restart cycle")
        return super().solve(iterations, seed)

    def _update_loop(self, max_iterations):
        wp.launch(
            KR.loop_update,
            1,
            [
                self.converged,
                self.status,
                self.iterations,
                self.running,
                self.recoveries,
                self.k,
                max_iterations,
                self.m == self.n,
            ],
            device=self.device,
        )

    def _orthogonalize_column(self, j, check):
        for pass_index in range(2):
            rows = j + int(pass_index == 0)
            wp.launch_tiled(
                KR.dot_partials,
                dim=(rows, self.dot_parts.shape[1]),
                inputs=[self.q, self.dot_parts, j],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.finish_dots,
                dim=rows,
                inputs=[
                    self.dot_parts,
                    self.dots,
                    self.source_norm,
                    self.g,
                    j,
                    check and pass_index == 0,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.subtract,
                dim=self.norm_parts.size,
                inputs=[self.q, self.dots, self.norm_parts, j, pass_index == 1],
                block_dim=128,
                device=self.device,
            )
        wp.launch_tiled(
            KR.finish_norm,
            dim=1,
            inputs=[
                self.norm_parts,
                self.norm,
                self.source_norm,
                self.breakdown,
                self.recoveries,
                check,
            ],
            block_dim=128,
            device=self.device,
        )

    def expand(self, j):
        self.op.apply(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self._orthogonalize_column(j, True)

        def recover():
            wp.launch(
                K.fallback_seed,
                self.n,
                [self.q, self.breakdown, self.recoveries, j],
                device=self.device,
            )
            self._orthogonalize_column(j, False)

        if self.device.is_cuda and wp.get_stream(self.device).is_capturing:
            wp.capture_if(self.breakdown, recover)
        else:
            # Eager execution also keeps the decision on device.
            self._recover_eager(j)
        wp.launch(K.basis_scale, self.n, [self.q, self.norm, j], device=self.device)

    def _recover_eager(self, j):
        # Gated kernels are only used outside captured replay.
        wp.launch(
            K.fallback_seed,
            self.n,
            [self.q, self.breakdown, self.recoveries, j],
            device=self.device,
        )
        for _ in range(2):
            self.dots.zero_()
            wp.launch_tiled(
                K.fallback_dots,
                dim=(j, (self.n + 255) // 256),
                inputs=[self.q, self.dots, self.breakdown, j],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                K.fallback_subtract,
                self.n,
                [self.q, self.dots, self.breakdown, j],
                device=self.device,
            )
        self.norm.zero_()
        wp.launch_tiled(
            K.basis_norm,
            dim=(self.n + 255) // 256,
            inputs=[self.q, self.norm, j],
            block_dim=128,
            device=self.device,
        )

    def initialize(self, seed=42):
        """Initialize the iteration from current input values and supplied or seeded starts.

        Args:
            seed: Integer seed used for generated start directions.

        Returns:
            None. Results are written to the documented device arrays.
        """
        self.recoveries.zero_()
        self.g.zero_()
        self.status.zero_()
        if self.initial_vector is None:
            wp.launch(K.random_block, (1, self.n), [self.q, seed], device=self.device)
        else:
            wp.copy(self.views[0], self.initial_vector)
        self.norm.zero_()
        wp.launch_tiled(
            K.basis_norm,
            dim=(self.n + 255) // 256,
            inputs=[self.q, self.norm, 0],
            block_dim=128,
            device=self.device,
        )
        wp.launch(K.basis_scale, self.n, [self.q, self.norm, 0], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)

    def step(self):
        """Advance one outer iteration or Krylov restart cycle using existing device state.

        Returns:
            None. Results are written to the documented device arrays.
        """
        for j in range(self.keep, self.m):
            self.expand(j)
        self.op.apply(self.views[-1], self.zviews[-1])
        # Orthogonalization supplied every new projected column except the last.
        self.complete_projection()
        self.diagonalize()
        self.rotate(self.q, self.retained)
        self.update_retained_products()
        wp.copy(self.q, self.retained, count=self.keep * self.n)
        wp.copy(self.z, self.retained_ax, count=self.keep * self.n)
        self.reset_projection()

    def update_retained_products(self):
        if self._reapply_after_restart:
            self.op.apply(self.retained, self.retained_ax)
        else:
            self.rotate(self.z, self.retained_ax)

    def reset_projection(self):
        # After recovery, retained Ritz residuals need not be collinear. Expand
        # the least converged requested pair instead of an arbitrary last row.
        # Only q/z are permuted: retained outputs stay in requested spectrum order.
        self.restart_scores.zero_()
        wp.launch_tiled(
            KR.restart_residuals,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.retained, self.retained_ax, self.ritz, self.order, self.restart_scores],
            block_dim=128,
            device=self.device,
        )
        wp.launch(KR.restart_seed, 1, [self.restart_scores, self.restart_seed], device=self.device)
        wp.launch(
            KR.restart_swap,
            self.n,
            [self.q, self.z, self.restart_seed, self.keep],
            device=self.device,
        )
        # The retained Ritz block is diagonal; future columns start at zero.
        wp.launch(
            KR.restart_projection,
            self.g.shape,
            [self.g, self.ritz, self.order, self.keep, self.restart_seed],
            device=self.device,
        )

    def complete_projection(self):
        wp.launch_tiled(
            KR.last_projection,
            dim=(self.m, self.dot_parts.shape[1]),
            inputs=[self.q, self.z, self.dot_parts],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            KR.finish_projection,
            dim=self.m,
            inputs=[self.dot_parts, self.g],
            block_dim=128,
            device=self.device,
        )

    def finalize(self):
        """Evaluate retained pairs using the original symmetric residual scale."""

        wp.copy(self.eigenvectors, self.retained, count=self.k * self.n)
        wp.copy(self.ax, self.retained_ax, count=self.k * self.n)
        self.physical_norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.eigenvectors, self.ritz, self.order, self.physical_norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.ritz,
                self.order,
                self.physical_norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                getattr(self, "atol", 0.0),
                self.converged,
            ],
            device=self.device,
        )
