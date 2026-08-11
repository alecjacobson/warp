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

"""Sparse Hessian/gradient assembly from indexed sums of local summands (MVP).

Early implementation of the design in ``design/sparse-hessians.md`` (GH-1767).

A summand is a scalar local energy written as a Warp function and tagged with
``@wp.summand``. Manual first and second derivatives are supplied with
``@wp.summand_grad`` / ``@wp.summand_hessian``, using an argument-indexed sparse
dict convention::

    @wp.summand
    def spring_energy(p0: wp.vec3, p1: wp.vec3) -> float:
        return 0.5 * wp.length_sq(p0 - p1)


    @wp.summand_grad(spring_energy)
    def _(p0: wp.vec3, p1: wp.vec3):
        d = p0 - p1
        return {0: d, 1: -d}  # {arg_index: dE/d(arg)}


    @wp.summand_hessian(spring_energy)
    def _(p0: wp.vec3, p1: wp.vec3):
        ident = wp.identity(n=3, dtype=float)
        return {(0, 0): ident, (0, 1): -ident, (1, 1): ident}  # upper triangle only

The derivative bodies never actually return a Python ``dict`` at runtime: the
decorators read the literal-keyed dict from the function's AST and *synthesize* a
strict-typed ``wp.func`` for the specific derivative request (the "factory"
model). Off-diagonal Hessian blocks are given for the upper triangle only; the
lower triangle is the transpose.

Current MVP scope: a single differentiable ``wp.array(dtype=wp.vec3)`` variable
over homogeneous 1-, 2-, or 3-node stencils. ``value.value`` is the summed scalar
energy; ``value.hessian[positions, positions]`` returns a ``mat33``
:class:`~warp.sparse.BsrMatrix`; ``value.gradient[positions]`` returns a ``vec3``
array; ``value.vjp(positions, seed)`` returns ``seed`` times that gradient (with
``gradient[x] == vjp(x, 1.0)``). Extra per-element
parameters, mixed dtypes, multi-variable off-diagonal blocks, automatic
backends, and PSD projection are tracked as next steps.
"""

import ast
import copy
import types

import warp as wp
from warp._src.codegen import make_full_qualified_name
from warp._src.context import Function, get_module
from warp._src.sparse import bsr_from_triplets

__all__ = ["IndexedSum", "Summand", "indexed_sum", "summand", "summand_grad", "summand_hessian"]


# ---------------------------------------------------------------------------
# Factory: turn a literal-keyed dict-returning authoring function into a
# strict-typed wp.func that returns the requested blocks as a tuple.
# ---------------------------------------------------------------------------


class _DictReturnToTuple(ast.NodeTransformer):
    """Rewrite ``return {literal_key: expr, ...}`` into ``return (e0, e1, ...)``.

    ``ordered_keys`` fixes the tuple order; keys absent from the dict are filled
    with ``zero_src`` (a Warp zero-constructor like ``wp.mat33()``). The function's
    return annotation is stripped so Warp infers the concrete tuple type.
    """

    def __init__(self, ordered_keys, zero_src):
        self._ordered_keys = ordered_keys
        self._zero_src = zero_src

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Return(self, node):
        if not isinstance(node.value, ast.Dict):
            return node
        present = {ast.literal_eval(k): v for k, v in zip(node.value.keys, node.value.values, strict=True)}
        elts = []
        for key in self._ordered_keys:
            if key in present:
                elts.append(present[key])
            else:
                elts.append(copy.deepcopy(ast.parse(self._zero_src, mode="eval").body))
        new = ast.Return(value=ast.Tuple(elts=elts, ctx=ast.Load()))
        return ast.fix_missing_locations(ast.copy_location(new, node))


_synth_counter = [0]


def _synthesize_blocks_func(authoring_fn, ordered_keys, zero_src):
    """Build a wp.func returning ``ordered_keys`` blocks from a dict-returning spec."""
    _synth_counter[0] += 1
    suffix = f"__blocks_{_synth_counter[0]}"

    # Copy the authoring function with a private globals dict so the synthesized
    # function keeps the user's module scope (wp.length, constants, ...).
    g = dict(authoring_fn.__globals__)
    fn = types.FunctionType(
        authoring_fn.__code__, g, name=authoring_fn.__name__ + suffix, argdefs=authoring_fn.__defaults__
    )
    fn.__annotations__ = {k: v for k, v in getattr(authoring_fn, "__annotations__", {}).items() if k != "return"}

    key = make_full_qualified_name(authoring_fn) + suffix
    module = get_module(authoring_fn.__module__)
    Function(
        func=fn,
        key=key,
        namespace="",
        module=module,
        value_func=None,
        scope_locals={},
        code_transformers=[_DictReturnToTuple(ordered_keys, zero_src)],
    )
    return module.functions[key]


def _hessian_block_keys(num_nodes):
    """Upper-triangle (including diagonal) block keys for a ``k``-node stencil."""
    return [(i, j) for i in range(num_nodes) for j in range(i, num_nodes)]


def _gradient_block_keys(num_nodes):
    return list(range(num_nodes))


# ---------------------------------------------------------------------------
# Assembly kernels. Each references an injected local-derivative func by name;
# _build_assembly_kernel binds the synthesized func into a private globals copy.
# ---------------------------------------------------------------------------


@wp.func
def _emit_block(
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array3d(dtype=float),
    idx: int,
    ri: int,
    ci: int,
    block: wp.mat33,
):
    rows[idx] = ri
    cols[idx] = ci
    for r in range(3):
        for c in range(3):
            vals[idx, r, c] = block[r, c]


def _hess_k1_template(
    elements: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array3d(dtype=float),
):
    tid = wp.tid()
    i0 = elements[tid]
    b00 = _local_blocks(x[i0])  # noqa: F821
    _emit_block(rows, cols, vals, tid, i0, i0, b00)


def _hess_k2_template(
    elements: wp.array(dtype=wp.vec2i),
    x: wp.array(dtype=wp.vec3),
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array3d(dtype=float),
):
    tid = wp.tid()
    e = elements[tid]
    i0 = e[0]
    i1 = e[1]
    b00, b01, b11 = _local_blocks(x[i0], x[i1])  # noqa: F821
    base = tid * 4
    _emit_block(rows, cols, vals, base + 0, i0, i0, b00)
    _emit_block(rows, cols, vals, base + 1, i0, i1, b01)
    _emit_block(rows, cols, vals, base + 2, i1, i0, wp.transpose(b01))
    _emit_block(rows, cols, vals, base + 3, i1, i1, b11)


def _hess_k3_template(
    elements: wp.array(dtype=wp.vec3i),
    x: wp.array(dtype=wp.vec3),
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    vals: wp.array3d(dtype=float),
):
    tid = wp.tid()
    e = elements[tid]
    i0 = e[0]
    i1 = e[1]
    i2 = e[2]
    b00, b01, b02, b11, b12, b22 = _local_blocks(x[i0], x[i1], x[i2])  # noqa: F821
    base = tid * 9
    _emit_block(rows, cols, vals, base + 0, i0, i0, b00)
    _emit_block(rows, cols, vals, base + 1, i0, i1, b01)
    _emit_block(rows, cols, vals, base + 2, i0, i2, b02)
    _emit_block(rows, cols, vals, base + 3, i1, i0, wp.transpose(b01))
    _emit_block(rows, cols, vals, base + 4, i1, i1, b11)
    _emit_block(rows, cols, vals, base + 5, i1, i2, b12)
    _emit_block(rows, cols, vals, base + 6, i2, i0, wp.transpose(b02))
    _emit_block(rows, cols, vals, base + 7, i2, i1, wp.transpose(b12))
    _emit_block(rows, cols, vals, base + 8, i2, i2, b22)


# The gradient assembly scales each per-node contribution by ``seed`` -- the
# (scalar) cotangent of the summed energy -- so it is the vector-Jacobian
# product seed * dE/dx. seed=1 gives the plain gradient.


def _grad_k1_template(
    elements: wp.array(dtype=int), x: wp.array(dtype=wp.vec3), seed: float, grad: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    i0 = elements[tid]
    g0 = _local_blocks(x[i0])  # noqa: F821
    wp.atomic_add(grad, i0, seed * g0)


def _grad_k2_template(
    elements: wp.array(dtype=wp.vec2i), x: wp.array(dtype=wp.vec3), seed: float, grad: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    e = elements[tid]
    i0 = e[0]
    i1 = e[1]
    g0, g1 = _local_blocks(x[i0], x[i1])  # noqa: F821
    wp.atomic_add(grad, i0, seed * g0)
    wp.atomic_add(grad, i1, seed * g1)


def _grad_k3_template(
    elements: wp.array(dtype=wp.vec3i), x: wp.array(dtype=wp.vec3), seed: float, grad: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    e = elements[tid]
    i0 = e[0]
    i1 = e[1]
    i2 = e[2]
    g0, g1, g2 = _local_blocks(x[i0], x[i1], x[i2])  # noqa: F821
    wp.atomic_add(grad, i0, seed * g0)
    wp.atomic_add(grad, i1, seed * g1)
    wp.atomic_add(grad, i2, seed * g2)


_HESS_TEMPLATES = {1: _hess_k1_template, 2: _hess_k2_template, 3: _hess_k3_template}
_GRAD_TEMPLATES = {1: _grad_k1_template, 2: _grad_k2_template, 3: _grad_k3_template}

# Number of triplet blocks written per element by the Hessian assembly (k*k).
_HESS_BLOCKS_PER_ELEM = {1: 1, 2: 4, 3: 9}

_kernel_counter = [0]


def _build_assembly_kernel(template, func, inject_name="_local_blocks"):
    """Specialize an assembly template by injecting a local func under ``inject_name``."""
    _kernel_counter[0] += 1
    key = f"{template.__name__}_{_kernel_counter[0]}"
    g = dict(template.__globals__)
    g[inject_name] = func
    fn = types.FunctionType(template.__code__, g, name=key, argdefs=template.__defaults__)
    fn.__annotations__ = dict(template.__annotations__)
    return wp.Kernel(func=fn, key=key)


# Forward energy assembly: sum the summand energy over all stencils.


def _energy_k1_template(elements: wp.array(dtype=int), x: wp.array(dtype=wp.vec3), acc: wp.array(dtype=float)):
    tid = wp.tid()
    i0 = elements[tid]
    wp.atomic_add(acc, 0, _local_energy(x[i0]))  # noqa: F821


def _energy_k2_template(elements: wp.array(dtype=wp.vec2i), x: wp.array(dtype=wp.vec3), acc: wp.array(dtype=float)):
    tid = wp.tid()
    e = elements[tid]
    wp.atomic_add(acc, 0, _local_energy(x[e[0]], x[e[1]]))  # noqa: F821


def _energy_k3_template(elements: wp.array(dtype=wp.vec3i), x: wp.array(dtype=wp.vec3), acc: wp.array(dtype=float)):
    tid = wp.tid()
    e = elements[tid]
    wp.atomic_add(acc, 0, _local_energy(x[e[0]], x[e[1]], x[e[2]]))  # noqa: F821


_ENERGY_TEMPLATES = {1: _energy_k1_template, 2: _energy_k2_template, 3: _energy_k3_template}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Summand:
    """A local summand energy tagged for use with :func:`indexed_sum`.

    Manual derivatives registered via :func:`summand_grad` / :func:`summand_hessian`
    are stored as authoring functions and compiled on demand by the factory.
    """

    def __init__(self, func):
        self.func = func if isinstance(func, wp.Function) else wp.func(func)
        self.grad_authoring_fn = None
        self.hessian_authoring_fn = None
        # Assembly kernels are pure and are cached per (kind, num_nodes) so
        # repeated calls (e.g. every optimizer step) reuse the compiled kernel
        # instead of synthesizing and recompiling a new one each time.
        self._kernel_cache = {}
        self.__doc__ = getattr(func, "__doc__", None)
        self.__name__ = getattr(func, "__name__", "summand")

    def _assembly_kernel(self, kind, num_nodes):
        cached = self._kernel_cache.get((kind, num_nodes))
        if cached is not None:
            return cached

        if kind == "energy":
            kernel = _build_assembly_kernel(_ENERGY_TEMPLATES[num_nodes], self.func, inject_name="_local_energy")
        elif kind == "gradient":
            if self.grad_authoring_fn is None:
                raise ValueError("no gradient registered; use @wp.summand_grad to provide one.")
            local = _synthesize_blocks_func(self.grad_authoring_fn, _gradient_block_keys(num_nodes), "wp.vec3()")
            kernel = _build_assembly_kernel(_GRAD_TEMPLATES[num_nodes], local)
        elif kind == "hessian":
            if self.hessian_authoring_fn is None:
                raise ValueError("no Hessian registered; use @wp.summand_hessian to provide one.")
            local = _synthesize_blocks_func(self.hessian_authoring_fn, _hessian_block_keys(num_nodes), "wp.mat33()")
            kernel = _build_assembly_kernel(_HESS_TEMPLATES[num_nodes], local)
        else:
            raise ValueError(f"unknown assembly kind {kind!r}")

        self._kernel_cache[(kind, num_nodes)] = kernel
        return kernel


def summand(func):
    """Decorator marking a Warp function as a local summand energy.

    Args:
        func: A function computing a scalar energy for one stencil. Registered as
            a ``wp.func`` if it is not one already.
    """
    return Summand(func)


def summand_grad(energy):
    """Register a manual gradient for a summand (companion to :func:`summand`).

    The decorated function returns a dict ``{arg_index: dE/d(arg)}`` listing the
    per-node gradient (only nonzero entries are required). It mirrors
    :func:`warp.func_grad` in spirit, but returns the gradient by value.

    Args:
        energy: The :class:`Summand` whose gradient this defines.
    """
    if not isinstance(energy, Summand):
        raise TypeError("@wp.summand_grad must decorate a function registered with @wp.summand.")

    def wrapper(grad_fn):
        energy.grad_authoring_fn = grad_fn
        return grad_fn

    return wrapper


def summand_hessian(energy):
    """Register a manual Hessian for a summand (companion to :func:`summand`).

    The decorated function returns a dict ``{(arg_i, arg_j): block}`` giving the
    dense local Hessian blocks. Only the upper triangle (``i <= j``) is required;
    the lower triangle is taken as the transpose.

    Args:
        energy: The :class:`Summand` whose Hessian this defines.
    """
    if not isinstance(energy, Summand):
        raise TypeError("@wp.summand_hessian must decorate a function registered with @wp.summand.")

    def wrapper(hessian_fn):
        energy.hessian_authoring_fn = hessian_fn
        return hessian_fn

    return wrapper


def _num_nodes_from_indices(indices):
    dtype = indices.dtype
    if dtype in (int, wp.int32):
        return 1
    if dtype is wp.vec2i:
        return 2
    if dtype is wp.vec3i:
        return 3
    raise TypeError(
        "indexed_sum indices must be an array of int (1-node), wp.vec2i (2-node), "
        f"or wp.vec3i (3-node) stencils; got dtype {dtype!r}."
    )


class IndexedSum:
    """A summand paired with the stencil indices it is summed over.

    Args:
        energy: A :class:`Summand` (or raw Warp function) describing the term.
        indices: Array of stencil indices; its dtype selects the stencil arity
            (``int`` -> 1 node, ``wp.vec2i`` -> 2, ``wp.vec3i`` -> 3).
    """

    def __init__(self, energy, indices):
        self.energy = energy if isinstance(energy, Summand) else Summand(energy)
        self.indices = indices
        self.num_nodes = _num_nodes_from_indices(indices)
        self.num_elements = indices.shape[0]

    def __call__(self, positions):
        """Bind variable data and return a value object exposing derivatives."""
        return IndexedSumValue(self, positions)


class _HessianView:
    """Indexable accessor: ``value.hessian[positions, positions]`` -> BsrMatrix."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise KeyError("hessian must be indexed as hessian[var, var].")
        row_var, col_var = key
        positions = self._value.positions
        if row_var is not positions or col_var is not positions:
            raise NotImplementedError("this MVP only supports the diagonal block hessian[positions, positions].")
        return self._value._assemble_hessian()


class _GradientView:
    """Indexable accessor: ``value.gradient[positions]`` -> wp.array(vec3)."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, var):
        # gradient[x] is the vector-Jacobian product with unit seed.
        return self._value.vjp(var, 1.0)


class IndexedSumValue:
    """Result of evaluating an :class:`IndexedSum` on variable data."""

    def __init__(self, indexed_sum, positions):
        if positions.dtype is not wp.vec3:
            raise TypeError("this MVP requires the differentiable variable to be wp.array(dtype=wp.vec3).")
        self.indexed_sum = indexed_sum
        self.positions = positions

    @property
    def value(self):
        """The scalar total energy ``sum_i g(stencil_i)``.

        Returns a host ``float`` and therefore synchronizes the device.
        """
        isum = self.indexed_sum
        kernel = isum.energy._assembly_kernel("energy", isum.num_nodes)
        acc = wp.zeros(1, dtype=float, device=self.positions.device)
        wp.launch(
            kernel, dim=isum.num_elements, inputs=[isum.indices, self.positions, acc], device=self.positions.device
        )
        return float(acc.numpy()[0])

    @property
    def hessian(self):
        return _HessianView(self)

    @property
    def gradient(self):
        return _GradientView(self)

    def vjp(self, var, seed=1.0):
        """Vector-Jacobian product of the summed energy: ``seed * dE/dvar``.

        For a scalar summand energy the value is a scalar, so ``seed`` is its
        (scalar) cotangent and the result is the gradient scaled by ``seed``:
        ``seed=1`` gives the gradient, ``seed=-1`` gives its negation (handy as a
        Newton right-hand side), ``seed=alpha`` scales it. ``gradient[var]`` is
        exactly ``vjp(var, 1.0)``.

        This mirrors reverse-mode autodiff in JAX/PyTorch, where the gradient of a
        scalar output is the VJP seeded with ``1.0``. When the vector-valued
        (Jacobian) track lands, ``seed`` becomes the output-space cotangent
        vector; see ``design/sparse-hessians.md``.

        Args:
            var: The differentiable variable to differentiate against.
            seed: Scalar cotangent of the summed energy.

        Returns:
            A ``wp.array(dtype=wp.vec3)`` holding ``seed * dE/dvar``.
        """
        if var is not self.positions:
            raise NotImplementedError("this MVP only supports vjp against positions.")
        return self._assemble_vjp(seed)

    def _assemble_hessian(self):
        isum = self.indexed_sum
        k = isum.num_nodes
        kernel = isum.energy._assembly_kernel("hessian", k)

        positions = self.positions
        device = positions.device
        ne = isum.num_elements
        n_triplets = ne * _HESS_BLOCKS_PER_ELEM[k]
        rows = wp.zeros(n_triplets, dtype=int, device=device)
        cols = wp.zeros(n_triplets, dtype=int, device=device)
        vals = wp.zeros((n_triplets, 3, 3), dtype=float, device=device)
        wp.launch(kernel, dim=ne, inputs=[isum.indices, positions, rows, cols, vals], device=device)

        num_verts = positions.shape[0]
        return bsr_from_triplets(num_verts, num_verts, rows, cols, vals)

    def _assemble_vjp(self, seed):
        isum = self.indexed_sum
        kernel = isum.energy._assembly_kernel("gradient", isum.num_nodes)

        positions = self.positions
        device = positions.device
        grad = wp.zeros(positions.shape[0], dtype=wp.vec3, device=device)
        wp.launch(kernel, dim=isum.num_elements, inputs=[isum.indices, positions, float(seed), grad], device=device)
        return grad


def indexed_sum(energy, indices):
    """Create a sparse-derivative assembler for a summed local energy.

    Returns an :class:`IndexedSum`; call it with the position array to obtain a
    value whose ``hessian[positions, positions]`` is the assembled
    :class:`~warp.sparse.BsrMatrix` and whose ``gradient[positions]`` is the
    assembled ``vec3`` gradient array.
    """
    return IndexedSum(energy, indices)
