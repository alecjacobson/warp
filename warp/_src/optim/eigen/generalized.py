# SPDX-FileCopyrightText: Copyright (c) 2026 warpack contributors
# SPDX-License-Identifier: MIT


import warp as wp


@wp.kernel
def generalized_residual(
    x: wp.array2d[wp.float64],
    ax: wp.array2d[wp.float64],
    bx: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(bx[j], shape=256, offset=chunk * 256)
    r = a - b * vals[order[j]]
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=0)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, a, a)), offset=1)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, b, b)), offset=2)


@wp.kernel
def generalized_outputs(
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    out: wp.array[wp.float64],
    errors: wp.array[wp.float64],
    tol: wp.float64,
    atol: wp.float64,
    converged: wp.array[wp.int32],
):
    j = wp.tid()
    lam = vals[order[j]]
    denominator = wp.sqrt(norms[j, 1]) + wp.abs(lam) * wp.sqrt(norms[j, 2]) + atol / tol
    residual = wp.sqrt(norms[j, 0])
    err = wp.float64(wp.inf)
    if denominator > wp.float64(0.0):
        err = residual / denominator
    elif residual == wp.float64(0.0):
        err = wp.float64(0.0)
    out[j] = lam
    errors[j] = err
    if err <= wp.float64(tol):
        wp.atomic_add(converged, 0, 1)
