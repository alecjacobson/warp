# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for sparse gradient/Hessian assembly via ``wp.indexed_sum`` (GH-1767).

* Phase 1 -- NumPy reference oracles: analytic local Hessians agree with a
  finite-difference oracle (:mod:`warp.tests.summand_references`).
* Phase 3 -- the ``wp.summand`` / ``wp.summand_grad`` / ``wp.summand_hessian``
  dict API assembles gradients and Hessians that match the references, and the
  Hessian ``BsrMatrix`` gives correct HVPs.
"""

import unittest

import numpy as np

import warp as wp
from warp.sparse import bsr_mv
from warp.tests import summand_references as ref
from warp.tests.unittest_utils import *

# Rest length baked as a constant so the summand needs no extra per-element
# parameter array (MVP scope). The sample geometry straddles it (stretch and
# compression), exercising the indefinite Hessian regime.
_REST_LENGTH = wp.constant(0.7)


# ---------------------------------------------------------------------------
# Summands with manual derivatives written in the argument-indexed dict form.
# ---------------------------------------------------------------------------


@wp.summand
def spring_energy(p0: wp.vec3, p1: wp.vec3) -> float:
    return 0.5 * (wp.length(p0 - p1) - _REST_LENGTH) ** 2.0


@wp.summand_grad(spring_energy)
def _(p0: wp.vec3, p1: wp.vec3):
    d = p0 - p1
    l = wp.length(d)
    n = d / l
    g = (l - _REST_LENGTH) * n
    return {0: g, 1: -g}


@wp.summand_hessian(spring_energy)
def _(p0: wp.vec3, p1: wp.vec3):
    d = p0 - p1
    l = wp.length(d)
    n = d / l
    r = _REST_LENGTH
    ident = wp.identity(n=3, dtype=float)
    h = (1.0 - r / l) * ident + (r / l) * wp.outer(n, n)
    return {(0, 0): h, (0, 1): -h, (1, 1): h}  # upper triangle only


@wp.summand
def inertia_energy(p0: wp.vec3) -> float:
    return 0.5 * wp.length_sq(p0)


@wp.summand_grad(inertia_energy)
def _(p0: wp.vec3):
    return {0: p0}


@wp.summand_hessian(inertia_energy)
def _(p0: wp.vec3):
    return {(0, 0): wp.identity(n=3, dtype=float)}


@wp.summand
def vertex_midpoint_energy(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3) -> float:
    d = p0 - 0.5 * (p1 + p2)
    return 0.5 * wp.length_sq(d)


@wp.summand_grad(vertex_midpoint_energy)
def _(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    d = p0 - 0.5 * (p1 + p2)
    return {0: d, 1: -0.5 * d, 2: -0.5 * d}


@wp.summand_hessian(vertex_midpoint_energy)
def _(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    ident = wp.identity(n=3, dtype=float)
    return {
        (0, 0): ident,
        (0, 1): -0.5 * ident,
        (0, 2): -0.5 * ident,
        (1, 1): 0.25 * ident,
        (1, 2): 0.25 * ident,
        (2, 2): 0.25 * ident,
    }


# ---------------------------------------------------------------------------
# Shared sample geometry and helpers.
# ---------------------------------------------------------------------------


def _sample_positions():
    # A little zig-zag chain; edge lengths straddle the rest length 0.7.
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.3, 0.0],
            [1.4, -0.2, 0.1],
            [1.9, 0.6, -0.3],
        ],
        dtype=np.float32,
    )


def _edges():
    return np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)


def _bsr_to_dense(bsr):
    off = bsr.offsets.numpy()
    row_counts = None if bsr.row_counts is None else bsr.row_counts.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy()
    bs = bsr.block_shape
    dense = np.zeros(bsr.shape)
    for row in range(bsr.nrow):
        beg = off[row]
        end = off[row + 1] if row_counts is None else beg + row_counts[row]
        for blk in range(beg, end):
            col = columns[blk]
            dense[row * bs[0] : (row + 1) * bs[0], col * bs[1] : (col + 1) * bs[1]] += values[blk]
    return dense


# ===========================================================================
# Phase 1: NumPy reference oracles (analytic == finite difference).
# ===========================================================================


class TestSummandReferences(unittest.TestCase):
    def _check(self, f, hess, x, tol=1e-5):
        h_analytic = hess(x)
        h_fd = ref.fd_hessian(f, x)
        np.testing.assert_allclose(h_analytic, h_fd, atol=tol, rtol=1e-4)
        np.testing.assert_allclose(h_analytic, h_analytic.T, atol=1e-12)

    def test_zero_rest_spring(self):
        rng = np.random.default_rng(0)
        self._check(ref.zero_rest_spring_value, ref.zero_rest_spring_hessian, rng.standard_normal(6))

    def test_rest_spring_stretched(self):
        r = 0.7
        x = np.array([0, 0, 0, 1.5, 0, 0.0])
        self._check(lambda z: ref.rest_spring_value(z, r), lambda z: ref.rest_spring_hessian(z, r), x)

    def test_rest_spring_compressed_is_indefinite(self):
        r = 0.7
        x = np.array([0, 0, 0, 0.3, 0, 0.0])
        self._check(lambda z: ref.rest_spring_value(z, r), lambda z: ref.rest_spring_hessian(z, r), x)
        evals = np.linalg.eigvalsh(ref.rest_spring_hessian(x, r))
        self.assertLess(evals.min(), -1e-6)

    def test_vertex_midpoint(self):
        rng = np.random.default_rng(2)
        self._check(ref.vertex_midpoint_value, ref.vertex_midpoint_hessian, rng.standard_normal(9))


# ===========================================================================
# Phase 3: the wp.summand dict API for gradients and Hessians.
# ===========================================================================


def test_spring_hessian(test, device):
    pos_np = _sample_positions()
    edges_np = _edges()
    num_verts = pos_np.shape[0]

    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    edges = wp.array(edges_np, dtype=wp.vec2i, device=device)

    total = wp.indexed_sum(spring_energy, edges)
    dense = _bsr_to_dense(total(pos).hessian[pos, pos])

    r = float(_REST_LENGTH)
    stencils = [tuple(e) for e in edges_np]
    expected = ref.assemble_global_hessian(num_verts, stencils, lambda z: ref.rest_spring_hessian(z, r), pos_np)
    np.testing.assert_allclose(dense, expected, atol=1e-4, rtol=1e-3)


def test_spring_gradient(test, device):
    # Assembled gradient vs. finite-difference gradient of the total energy.
    pos_np = _sample_positions()
    edges_np = _edges()
    num_verts = pos_np.shape[0]

    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    edges = wp.array(edges_np, dtype=wp.vec2i, device=device)

    total = wp.indexed_sum(spring_energy, edges)
    grad = total(pos).gradient[pos]
    wp.synchronize_device()

    r = float(_REST_LENGTH)
    stencils = [tuple(e) for e in edges_np]

    def energy(flat):
        return ref.total_energy(stencils, lambda z: ref.rest_spring_value(z, r), flat.reshape(num_verts, 3))

    g_fd = ref.fd_gradient(energy, pos_np.reshape(-1).astype(np.float64)).reshape(num_verts, 3)
    np.testing.assert_allclose(grad.numpy(), g_fd, atol=1e-3, rtol=1e-3)


def test_spring_value(test, device):
    # Forward energy sum vs. the NumPy reference total energy.
    pos_np = _sample_positions()
    edges_np = _edges()
    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    edges = wp.array(edges_np, dtype=wp.vec2i, device=device)

    val = wp.indexed_sum(spring_energy, edges)(pos).value

    r = float(_REST_LENGTH)
    stencils = [tuple(e) for e in edges_np]
    expected = ref.total_energy(stencils, lambda z: ref.rest_spring_value(z, r), pos_np)
    np.testing.assert_allclose(val, expected, atol=1e-5, rtol=1e-4)


def test_inertia_per_vertex(test, device):
    # 1-node stencil: per-vertex identity blocks -> global Hessian is identity.
    pos_np = _sample_positions()
    num_verts = pos_np.shape[0]
    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    vertex_ids = wp.array(np.arange(num_verts, dtype=np.int32), dtype=int, device=device)

    total = wp.indexed_sum(inertia_energy, vertex_ids)
    value = total(pos)

    dense = _bsr_to_dense(value.hessian[pos, pos])
    np.testing.assert_allclose(dense, np.eye(3 * num_verts), atol=1e-5)

    # Gradient of 0.5|p|^2 is p.
    np.testing.assert_allclose(value.gradient[pos].numpy(), pos_np, atol=1e-5)


def test_hvp_matches_dense(test, device):
    pos_np = _sample_positions()
    edges_np = _edges()
    num_verts = pos_np.shape[0]

    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    edges = wp.array(edges_np, dtype=wp.vec2i, device=device)

    H = wp.indexed_sum(spring_energy, edges)(pos).hessian[pos, pos]

    rng = np.random.default_rng(3)
    v_np = rng.standard_normal((num_verts, 3)).astype(np.float32)
    v = wp.array(v_np, dtype=wp.vec3, device=device)
    y = bsr_mv(H, v)
    wp.synchronize_device()

    dense = _bsr_to_dense(H)
    expected = (dense @ v_np.reshape(-1)).reshape(num_verts, 3)
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-4, rtol=1e-3)


def test_vertex_midpoint_k3(test, device):
    # 3-node stencil exercises the k=3 assembly path.
    pos_np = _sample_positions()
    num_verts = pos_np.shape[0]
    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    tris_np = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    tris = wp.array(tris_np, dtype=wp.vec3i, device=device)

    dense = _bsr_to_dense(wp.indexed_sum(vertex_midpoint_energy, tris)(pos).hessian[pos, pos])
    stencils = [tuple(t) for t in tris_np]
    expected = ref.assemble_global_hessian(num_verts, stencils, ref.vertex_midpoint_hessian, pos_np)
    np.testing.assert_allclose(dense, expected, atol=1e-5)


devices = get_test_devices()


class TestSummand(unittest.TestCase):
    pass


add_function_test(TestSummand, "test_spring_hessian", test_spring_hessian, devices=devices)
add_function_test(TestSummand, "test_spring_gradient", test_spring_gradient, devices=devices)
add_function_test(TestSummand, "test_spring_value", test_spring_value, devices=devices)
add_function_test(TestSummand, "test_inertia_per_vertex", test_inertia_per_vertex, devices=devices)
add_function_test(TestSummand, "test_hvp_matches_dense", test_hvp_matches_dense, devices=devices)
add_function_test(TestSummand, "test_vertex_midpoint_k3", test_vertex_midpoint_k3, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
