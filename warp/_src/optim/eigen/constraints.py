# SPDX-FileCopyrightText: Copyright (c) 2026 warpack contributors
# SPDX-License-Identifier: MIT

"""Constraint row reduction and device-resident subspace projections."""

import math

import warp as wp

from . import kernels as K


@wp.kernel
def subtract_projection(
    basis: wp.array2d[wp.float64],
    coefficients: wp.array2d[wp.float64],
    vector: wp.array2d[wp.float64],
):
    i = wp.tid()
    value = vector[0, i]
    for j in range(basis.shape[0]):
        value -= basis[j, i] * coefficients[j, 0]
    vector[0, i] = value


@wp.kernel
def _validate_metric_basis(gram: wp.array2d[wp.float64], status: wp.array[wp.int32], tolerance: wp.float64):
    i, j = wp.tid()
    expected = wp.float64(0.0)
    if i == j:
        expected = wp.float64(1.0)
    if not wp.isfinite(gram[i, j]) or wp.abs(gram[i, j] - expected) > tolerance:
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _validate_basis_factor(gram: wp.array2d[wp.float64], lower: wp.array2d[wp.float64], status: wp.array[wp.int32]):
    i = wp.tid()
    diagonal = gram[i, i]
    pivot = lower[i, i]
    if (
        not wp.isfinite(diagonal)
        or diagonal <= wp.float64(0.0)
        or not wp.isfinite(pivot)
        or pivot * pivot <= wp.float64(1e-12) * diagonal
    ):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _weighted_subtract(
    rows: wp.array2d[wp.float64],
    coefficients: wp.array2d[wp.float64],
    vector: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    poison: bool,
):
    i = wp.tid()
    value = vector[0, i]
    if status[0] != 0:
        value = wp.float64(0.0)
        if poison:
            value = wp.float64(wp.nan)
    else:
        for j in range(rows.shape[0]):
            value -= rows[j, i] * coefficients[j, 0]
    vector[0, i] = value


class MassOrthogonalComplementInverse:
    """Apply ``P inverse P.T`` with ``P = I - Q Q.T M`` entirely on device.

    ``basis`` stores FP64 basis rows; ``mass`` is an SPD operator. By default
    the caller supplies ``Q.T M Q = I``. With ``normalize=True``, two passes
    of mass CholeskyQR normalize an owned copy entirely in Warp. The input
    remains unchanged; numerically dependent rows are rejected (relative
    squared Cholesky pivot <= 1e-12). For a known nullspace of symmetric K,
    wrapping the one-RHS inverse of K-sigma*M excludes that nullspace from
    generalized shift-invert iteration without replacing the mass metric.
    This is not an arbitrary-constraint eigensolver: original-pencil residuals
    are justified when the supplied space is invariant, e.g. rigid modes of K.

    Construction and ``refresh()`` cache M Q and check mass orthonormality using
    Warp kernels. ``basis_status`` is a device int: 0 valid, 1 invalid. Invalid
    bases cause NaN outputs, not silently incorrect projection. No numerical
    host readback occurs. Call ``refresh()`` after changing basis or mass values;
    it is graph capturable. Normalization, when enabled, starts from the input
    rows each time. Capture binds array storage; replacing buffers requires
    recapture. Updating the wrapped inverse/factors is separately
    the caller's responsibility. The wrapper does not own or close the inverse.

    Application supports multiple RHSs and exact in-place input/output aliasing;
    other overlapping views are unsupported. Buffers are preallocated and must
    not be used concurrently on different streams. The basis must remain live.

    Args:
        inverse: Borrowed one-RHS real inverse. The wrapper serializes multiple RHSs through it.
        mass: SPD mass/metric operator on the same device and dimension as operator.
        basis: Borrowed independent FP64 device rows (q, n). Must be mass-orthonormal unless
            normalize=True.
        basis_tolerance: Positive finite tolerance for entrywise mass-Gram error.
        normalize: Normalize an owned copy using device CholeskyQR at construction and
            refresh().

    Attributes:
        basis_status: int32 device array (1,): 0 valid, 1 invalid basis. Invalid application
            produces NaNs.
    """

    def __init__(self, inverse, mass, basis, basis_tolerance=1e-10, normalize=False):

        self.n, self.device = inverse.n, inverse.device
        if mass.n != self.n or mass.device != self.device:
            raise ValueError("mass and inverse must have matching dimensions and devices")
        if getattr(inverse, "width", 1) != 1:
            raise ValueError("mass projection requires a one-RHS inverse")
        if basis.ndim != 2 or basis.shape[1] != self.n or not 1 <= basis.shape[0] <= self.n:
            raise ValueError("basis must have shape (rank, n), with 1 <= rank <= n")
        if basis.dtype != wp.float64 or basis.device != self.device:
            raise ValueError("basis must be FP64 on the inverse device")
        if not math.isfinite(basis_tolerance) or basis_tolerance <= 0:
            raise ValueError("basis_tolerance must be finite and positive")
        self.inverse, self.mass, self.input_basis = inverse, mass, basis
        self.normalize = bool(normalize)
        self.basis = wp.empty_like(basis) if self.normalize else basis
        self.width = 1
        self.basis_tolerance = float(basis_tolerance)
        self.requires_static_outer_capture = getattr(inverse, "requires_static_outer_capture", False)
        rank = basis.shape[0]
        self.mass_basis = wp.empty_like(basis)
        self.rhs = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.dots = wp.zeros((rank, 1), dtype=wp.float64, device=self.device)
        self.gram = wp.zeros((rank, rank), dtype=wp.float64, device=self.device)
        self.lower = wp.empty_like(self.gram) if self.normalize else None
        self.basis_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.refresh()

    def _metric_gram(self):
        rank = self.basis.shape[0]
        for j in range(rank):
            self.mass.apply(self.basis[j : j + 1], self.mass_basis[j : j + 1])
        self.gram.zero_()
        wp.launch_tiled(
            K.gram_partial,
            dim=(rank, rank, (self.n + 255) // 256),
            inputs=[self.basis, self.mass_basis, self.gram],
            block_dim=128,
            device=self.device,
        )

    def refresh(self):
        """Refresh the metric basis and its status, including optional normalization.

        Returns:
            None. Results are written to the documented device arrays.
        """
        self.basis_status.zero_()
        rank = self.basis.shape[0]
        if self.normalize:
            wp.copy(self.basis, self.input_basis)
            for _ in range(2):
                self._metric_gram()
                wp.launch(K.cholesky, 1, [self.gram, self.lower, self.basis_status], device=self.device)
                wp.launch(
                    _validate_basis_factor,
                    rank,
                    [self.gram, self.lower, self.basis_status],
                    device=self.device,
                )
                wp.launch(K.triangular, self.n, [self.basis, self.lower], device=self.device)
        self._metric_gram()
        wp.launch(
            _validate_metric_basis,
            (rank, rank),
            [self.gram, self.basis_status, self.basis_tolerance],
            device=self.device,
        )

    def reset_history(self):
        """Reset any wrapped inverse history on device before a new solve.

        Returns:
            None. Results are written to the documented device arrays.
        """
        if hasattr(self.inverse, "reset_history"):
            self.inverse.reset_history()

    def _project(self, dot_rows, subtract_rows, vector, poison):
        self.dots.zero_()
        wp.launch_tiled(
            K.gram_partial,
            dim=(self.basis.shape[0], 1, (self.n + 255) // 256),
            inputs=[dot_rows, vector, self.dots],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            _weighted_subtract,
            self.n,
            [subtract_rows, self.dots, vector, self.basis_status, poison],
            device=self.device,
        )

    def apply(self, x, y):
        """Apply the operator to device rows, writing into the supplied output.

        Args:
            x: Input device rows of the dtype and shape required by this operator.
            y: Output device rows, matching x. Follow the class-specific aliasing rules; do not
                assume overlapping views are supported.

        Returns:
            None. Results are written to the documented device arrays.
        """
        if x.ndim != 2 or x.shape[1] != self.n or x.shape != y.shape:
            raise ValueError("RHS/output must have matching shape (count, n)")
        if x.dtype != wp.float64 or y.dtype != wp.float64:
            raise ValueError("RHS/output must be FP64")
        if x.device != self.device or y.device != self.device:
            raise ValueError("RHS/output must be on the inverse device")
        for j in range(x.shape[0]):
            wp.copy(self.rhs, x[j : j + 1])
            # Dual projection P.T before the inverse; primal P after it.
            self._project(self.basis, self.mass_basis, self.rhs, False)
            self.inverse.apply(self.rhs, y[j : j + 1])
            self._project(self.mass_basis, self.basis, y[j : j + 1], True)
