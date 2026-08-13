# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-order forward-mode jets as ordinary differentiable Warp types.

A jet carries a value together with its derivatives along ``width`` directions
at once::

    x = value + sum_q coeff[q] * eps_q

Evaluating a function over jets instead of floats produces its derivatives as a
side effect of evaluating it, with no hand-differentiation. Because the jet
arithmetic is itself ordinary Warp code, Warp's reverse-mode autodiff can
differentiate through it, which makes reverse-over-forward -- and therefore
Hessians -- available without a second-order tape.

:func:`JetSpace` specializes the types at code-generation time:

    ========== ================== ====================
    Name       ``value``          ``coeff``
    ========== ================== ====================
    ``scalar`` ``dtype``          ``vector(width)``
    ``vec2``   ``vector(2)``      ``matrix(2, width)``
    ``vec3``   ``vector(3)``      ``matrix(3, width)``
    ========== ================== ====================

The derivative width is independent of the geometric vector dimension.

The generated arithmetic is registered as overloads of Warp's own builtins, so
jets are written with ordinary operator and builtin syntax::

    J = wp.JetSpace(6)


    @wp.func
    def spring_energy(x0: J.vec3, x1: J.vec3) -> J.scalar:
        d = x1 - x0
        r = wp.length(d) - 1.0
        return 0.5 * r * r


    @wp.kernel
    def local_gradient(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        grad: wp.array[J.coeff],
    ):
        i = wp.tid()

        # Six local scalar variables:
        # x0.x/y/z -> directions 0/1/2
        # x1.x/y/z -> directions 3/4/5
        a = J.seed_vec3(x0[i], 0, 1, 2)
        b = J.seed_vec3(x1[i], 3, 4, 5)

        grad[i] = spring_energy(a, b).coeff

Note that ``spring_energy`` is a plain :func:`warp.func`. Operators resolve
through Warp's builtin overload table, so nothing has to be imported or bound
into the calling module beyond the space itself.

Component access follows the same route: ``v[0]`` on a ``J.vec3`` resolves
through the ``extract`` builtin and yields a ``J.scalar``.

Seeding and construction have no builtin counterpart and stay on the returned
namespace, as do the 2D-only helpers :attr:`perp` and :attr:`cross2`, which
Warp does not define for ``vec2``.
"""

# NOTE: deliberately no `from __future__ import annotations` here. The structs
# and functions below are annotated with types created inside _make_jet_space,
# and Warp resolves struct annotations with inspect.get_annotations(eval_str=True),
# which evaluates them against module globals. Deferred annotations would turn
# them into strings that no longer resolve.

from types import SimpleNamespace

import warp as wp
import warp._src.context
from warp._src.types import matrix, vector

_CACHE = {}


def _register(name: str, fn) -> None:
    """Add every signature of ``fn`` as an overload of the builtin ``name``.

    A Warp function defined several times under one name accumulates its
    signatures in ``user_overloads``. Builtin overload resolution matches
    against a function's own ``input_types`` and does not recurse into that
    dict, so each signature has to be appended individually.
    """
    builtin = warp._src.context.builtin_functions[name]

    for overload in tuple(fn.user_overloads.values()) or (fn,):
        builtin.add_overload(overload)


def _make_jet_space(width: int, dtype):
    if width <= 0:
        raise ValueError("Jet width must be positive")

    # Concrete Warp types generated at Python scope.
    Coeff = vector(width, dtype)

    NativeVec2 = vector(2, dtype)
    NativeVec3 = vector(3, dtype)

    CoeffMat2 = matrix((2, width), dtype)
    CoeffMat3 = matrix((3, width), dtype)

    # ------------------------------------------------------------------
    # Core data types
    # ------------------------------------------------------------------

    @wp.struct
    class JetScalar:
        value: dtype
        coeff: Coeff

    @wp.struct
    class JetVec2:
        value: NativeVec2
        coeff: CoeffMat2

    @wp.struct
    class JetVec3:
        value: NativeVec3
        coeff: CoeffMat3

    # ------------------------------------------------------------------
    # Scalar construction / seeding
    # ------------------------------------------------------------------

    @wp.func
    def scalar_constant(x: dtype) -> JetScalar:
        return JetScalar(x, Coeff())

    @wp.func
    def scalar_with_coeff(x: dtype, coeff: Coeff) -> JetScalar:
        return JetScalar(x, coeff)

    @wp.func
    def scalar_seed(x: dtype, direction: int) -> JetScalar:
        c = Coeff()
        c[direction] = dtype(1.0)
        return JetScalar(x, c)

    # ------------------------------------------------------------------
    # Arithmetic
    #
    # Warp maps the AST operators to the builtin names add/sub/mul/div/pow/
    # pos/neg, so registering these below makes ordinary operator syntax work
    # on jets in any module.
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(a.value + b.value, a.coeff + b.coeff)

    @wp.func
    def jet_add(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value + b, a.coeff)

    @wp.func
    def jet_add(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a + b.value, b.coeff)

    @wp.func
    def jet_sub(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(a.value - b.value, a.coeff - b.coeff)

    @wp.func
    def jet_sub(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value - b, a.coeff)

    @wp.func
    def jet_sub(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a - b.value, -b.coeff)

    @wp.func
    def jet_pos(a: JetScalar) -> JetScalar:
        return a

    @wp.func
    def jet_neg(a: JetScalar) -> JetScalar:
        return JetScalar(-a.value, -a.coeff)

    @wp.func
    def jet_mul(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(
            a.value * b.value,
            a.coeff * b.value + b.coeff * a.value,
        )

    @wp.func
    def jet_mul(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value * b, a.coeff * b)

    @wp.func
    def jet_mul(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a * b.value, a * b.coeff)

    @wp.func
    def jet_div(a: JetScalar, b: JetScalar) -> JetScalar:
        inv_b = dtype(1.0) / b.value
        value = a.value * inv_b
        coeff = (a.coeff - b.coeff * value) * inv_b
        return JetScalar(value, coeff)

    @wp.func
    def jet_div(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value / b, a.coeff / b)

    @wp.func
    def jet_div(a: dtype, b: JetScalar) -> JetScalar:
        inv_b = dtype(1.0) / b.value
        return JetScalar(
            a * inv_b,
            (-a * inv_b * inv_b) * b.coeff,
        )

    # ------------------------------------------------------------------
    # Scalar elementary functions
    #
    # Each unary function is the chain rule f(a) = f(value) with derivative
    # f'(value) scaling the coefficients, so they all route through _lift1.
    # ------------------------------------------------------------------

    @wp.func
    def _lift1(value: dtype, fp: dtype, a: JetScalar) -> JetScalar:
        # Single-argument chain rule: value = f(a.value), fp = f'(a.value).
        return JetScalar(value, fp * a.coeff)

    @wp.func
    def jet_pow(a: JetScalar, p: dtype) -> JetScalar:
        return _lift1(wp.pow(a.value, p), p * wp.pow(a.value, p - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, p: int) -> JetScalar:
        return _lift1(wp.pow(a.value, dtype(p)), dtype(p) * wp.pow(a.value, dtype(p) - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, b: JetScalar) -> JetScalar:
        # a**b = exp(b log a); d = a**b (b/a da + log(a) db).
        value = wp.pow(a.value, b.value)
        coeff = value * ((b.value / a.value) * a.coeff + wp.log(a.value) * b.coeff)
        return JetScalar(value, coeff)

    @wp.func
    def jet_pow(a: dtype, b: JetScalar) -> JetScalar:
        value = wp.pow(a, b.value)
        return JetScalar(value, (value * wp.log(a)) * b.coeff)

    @wp.func
    def jet_sin(a: JetScalar) -> JetScalar:
        return _lift1(wp.sin(a.value), wp.cos(a.value), a)

    @wp.func
    def jet_cos(a: JetScalar) -> JetScalar:
        return _lift1(wp.cos(a.value), -wp.sin(a.value), a)

    @wp.func
    def jet_tan(a: JetScalar) -> JetScalar:
        c = wp.cos(a.value)
        return _lift1(wp.tan(a.value), dtype(1.0) / (c * c), a)

    @wp.func
    def jet_asin(a: JetScalar) -> JetScalar:
        return _lift1(wp.asin(a.value), dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value), a)

    @wp.func
    def jet_acos(a: JetScalar) -> JetScalar:
        return _lift1(wp.acos(a.value), -dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value), a)

    @wp.func
    def jet_atan(a: JetScalar) -> JetScalar:
        return _lift1(wp.atan(a.value), dtype(1.0) / (dtype(1.0) + a.value * a.value), a)

    @wp.func
    def jet_exp(a: JetScalar) -> JetScalar:
        e = wp.exp(a.value)
        return _lift1(e, e, a)

    @wp.func
    def jet_log(a: JetScalar) -> JetScalar:
        return _lift1(wp.log(a.value), dtype(1.0) / a.value, a)

    @wp.func
    def jet_sqrt(a: JetScalar) -> JetScalar:
        s = wp.sqrt(a.value)
        return _lift1(s, dtype(0.5) / s, a)

    @wp.func
    def jet_atan2(y: JetScalar, x: JetScalar) -> JetScalar:
        d = x.value * x.value + y.value * y.value
        return JetScalar(wp.atan2(y.value, x.value), (x.value * y.coeff - y.value * x.coeff) / d)

    @wp.func
    def jet_atan2(y: JetScalar, x: dtype) -> JetScalar:
        d = x * x + y.value * y.value
        return JetScalar(wp.atan2(y.value, x), (x * y.coeff) / d)

    @wp.func
    def jet_atan2(y: dtype, x: JetScalar) -> JetScalar:
        d = x.value * x.value + y * y
        return JetScalar(wp.atan2(y, x.value), (-y * x.coeff) / d)

    @wp.func
    def jet_abs(a: JetScalar) -> JetScalar:
        # Same nondifferentiability at zero as ordinary abs().
        if a.value > dtype(0.0):
            return a
        if a.value < dtype(0.0):
            return -a
        return JetScalar(dtype(0.0), Coeff())

    @wp.func
    def jet_sign(a: JetScalar) -> JetScalar:
        return JetScalar(wp.sign(a.value), Coeff())

    # ------------------------------------------------------------------
    # Selection / branching. Comparisons are value-only (a jet has no order),
    # so branch on ``.value`` and carry the chosen jet's derivatives through.
    # ------------------------------------------------------------------

    @wp.func
    def jet_min(a: JetScalar, b: JetScalar) -> JetScalar:
        if a.value <= b.value:
            return a
        return b

    @wp.func
    def jet_max(a: JetScalar, b: JetScalar) -> JetScalar:
        if a.value >= b.value:
            return a
        return b

    @wp.func
    def jet_clamp(a: JetScalar, lo: dtype, hi: dtype) -> JetScalar:
        if a.value < lo:
            return JetScalar(lo, Coeff())
        if a.value > hi:
            return JetScalar(hi, Coeff())
        return a

    @wp.func
    def jet_where(cond: bool, a: JetScalar, b: JetScalar) -> JetScalar:
        if cond:
            return a
        return b

    # ------------------------------------------------------------------
    # Vec2 / Vec3 construction
    # ------------------------------------------------------------------

    @wp.func
    def vec2_constant(v: NativeVec2) -> JetVec2:
        return JetVec2(v, CoeffMat2())

    @wp.func
    def vec3_constant(v: NativeVec3) -> JetVec3:
        return JetVec3(v, CoeffMat3())

    @wp.func
    def vec2_from_scalars(x: JetScalar, y: JetScalar) -> JetVec2:
        c = CoeffMat2()
        for q in range(wp.static(width)):
            c[0, q] = x.coeff[q]
            c[1, q] = y.coeff[q]
        return JetVec2(NativeVec2(x.value, y.value), c)

    @wp.func
    def vec3_from_scalars(x: JetScalar, y: JetScalar, z: JetScalar) -> JetVec3:
        c = CoeffMat3()
        for q in range(wp.static(width)):
            c[0, q] = x.coeff[q]
            c[1, q] = y.coeff[q]
            c[2, q] = z.coeff[q]
        return JetVec3(NativeVec3(x.value, y.value, z.value), c)

    @wp.func
    def seed_vec2(v: NativeVec2, i0: int, i1: int) -> JetVec2:
        c = CoeffMat2()
        c[0, i0] = dtype(1.0)
        c[1, i1] = dtype(1.0)
        return JetVec2(v, c)

    @wp.func
    def seed_vec3(v: NativeVec3, i0: int, i1: int, i2: int) -> JetVec3:
        c = CoeffMat3()
        c[0, i0] = dtype(1.0)
        c[1, i1] = dtype(1.0)
        c[2, i2] = dtype(1.0)
        return JetVec3(v, c)

    @wp.func
    def directional_vec2(v: NativeVec2, dv: NativeVec2) -> JetVec2:
        # Convenience for HVPs with width=1. If width>1, only direction 0
        # is populated and the remaining directions stay zero.
        c = CoeffMat2()
        c[0, 0] = dv[0]
        c[1, 0] = dv[1]
        return JetVec2(v, c)

    @wp.func
    def directional_vec3(v: NativeVec3, dv: NativeVec3) -> JetVec3:
        c = CoeffMat3()
        c[0, 0] = dv[0]
        c[1, 0] = dv[1]
        c[2, 0] = dv[2]
        return JetVec3(v, c)

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @wp.func
    def jet_extract(v: JetVec2, i: int) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = v.coeff[i, q]
        return JetScalar(v.value[i], c)

    @wp.func
    def jet_extract(v: JetVec3, i: int) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = v.coeff[i, q]
        return JetScalar(v.value[i], c)

    # ------------------------------------------------------------------
    # Vec2 arithmetic
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a.value + b.value, a.coeff + b.coeff)

    @wp.func
    def jet_add(a: JetVec2, b: NativeVec2) -> JetVec2:
        return JetVec2(a.value + b, a.coeff)

    @wp.func
    def jet_add(a: NativeVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a + b.value, b.coeff)

    @wp.func
    def jet_sub(a: JetVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a.value - b.value, a.coeff - b.coeff)

    @wp.func
    def jet_sub(a: JetVec2, b: NativeVec2) -> JetVec2:
        return JetVec2(a.value - b, a.coeff)

    @wp.func
    def jet_sub(a: NativeVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a - b.value, -b.coeff)

    @wp.func
    def jet_pos(a: JetVec2) -> JetVec2:
        return a

    @wp.func
    def jet_neg(a: JetVec2) -> JetVec2:
        return JetVec2(-a.value, -a.coeff)

    @wp.func
    def jet_mul(a: JetVec2, s: dtype) -> JetVec2:
        return JetVec2(a.value * s, a.coeff * s)

    @wp.func
    def jet_mul(s: dtype, a: JetVec2) -> JetVec2:
        return JetVec2(s * a.value, s * a.coeff)

    @wp.func
    def jet_mul(a: JetVec2, s: JetScalar) -> JetVec2:
        c = CoeffMat2()
        for q in range(wp.static(width)):
            c[0, q] = a.coeff[0, q] * s.value + a.value[0] * s.coeff[q]
            c[1, q] = a.coeff[1, q] * s.value + a.value[1] * s.coeff[q]
        return JetVec2(a.value * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: JetVec2) -> JetVec2:
        c = CoeffMat2()
        for q in range(wp.static(width)):
            c[0, q] = s.value * a.coeff[0, q] + s.coeff[q] * a.value[0]
            c[1, q] = s.value * a.coeff[1, q] + s.coeff[q] * a.value[1]
        return JetVec2(s.value * a.value, c)

    @wp.func
    def jet_div(a: JetVec2, s: dtype) -> JetVec2:
        return JetVec2(a.value / s, a.coeff / s)

    @wp.func
    def jet_div(a: JetVec2, s: JetScalar) -> JetVec2:
        inv_s = dtype(1.0) / s.value
        value = a.value * inv_s
        c = CoeffMat2()
        for q in range(wp.static(width)):
            c[0, q] = (a.coeff[0, q] - value[0] * s.coeff[q]) * inv_s
            c[1, q] = (a.coeff[1, q] - value[1] * s.coeff[q]) * inv_s
        return JetVec2(value, c)

    # ------------------------------------------------------------------
    # Vec3 arithmetic
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a.value + b.value, a.coeff + b.coeff)

    @wp.func
    def jet_add(a: JetVec3, b: NativeVec3) -> JetVec3:
        return JetVec3(a.value + b, a.coeff)

    @wp.func
    def jet_add(a: NativeVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a + b.value, b.coeff)

    @wp.func
    def jet_sub(a: JetVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a.value - b.value, a.coeff - b.coeff)

    @wp.func
    def jet_sub(a: JetVec3, b: NativeVec3) -> JetVec3:
        return JetVec3(a.value - b, a.coeff)

    @wp.func
    def jet_sub(a: NativeVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a - b.value, -b.coeff)

    @wp.func
    def jet_pos(a: JetVec3) -> JetVec3:
        return a

    @wp.func
    def jet_neg(a: JetVec3) -> JetVec3:
        return JetVec3(-a.value, -a.coeff)

    @wp.func
    def jet_mul(a: JetVec3, s: dtype) -> JetVec3:
        return JetVec3(a.value * s, a.coeff * s)

    @wp.func
    def jet_mul(s: dtype, a: JetVec3) -> JetVec3:
        return JetVec3(s * a.value, s * a.coeff)

    @wp.func
    def jet_mul(a: JetVec3, s: JetScalar) -> JetVec3:
        c = CoeffMat3()
        for q in range(wp.static(width)):
            c[0, q] = a.coeff[0, q] * s.value + a.value[0] * s.coeff[q]
            c[1, q] = a.coeff[1, q] * s.value + a.value[1] * s.coeff[q]
            c[2, q] = a.coeff[2, q] * s.value + a.value[2] * s.coeff[q]
        return JetVec3(a.value * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: JetVec3) -> JetVec3:
        c = CoeffMat3()
        for q in range(wp.static(width)):
            c[0, q] = s.value * a.coeff[0, q] + s.coeff[q] * a.value[0]
            c[1, q] = s.value * a.coeff[1, q] + s.coeff[q] * a.value[1]
            c[2, q] = s.value * a.coeff[2, q] + s.coeff[q] * a.value[2]
        return JetVec3(s.value * a.value, c)

    @wp.func
    def jet_div(a: JetVec3, s: dtype) -> JetVec3:
        return JetVec3(a.value / s, a.coeff / s)

    @wp.func
    def jet_div(a: JetVec3, s: JetScalar) -> JetVec3:
        inv_s = dtype(1.0) / s.value
        value = a.value * inv_s
        c = CoeffMat3()
        for q in range(wp.static(width)):
            c[0, q] = (a.coeff[0, q] - value[0] * s.coeff[q]) * inv_s
            c[1, q] = (a.coeff[1, q] - value[1] * s.coeff[q]) * inv_s
            c[2, q] = (a.coeff[2, q] - value[2] * s.coeff[q]) * inv_s
        return JetVec3(value, c)

    # ------------------------------------------------------------------
    # Dot products
    # ------------------------------------------------------------------

    @wp.func
    def jet_dot(a: JetVec2, b: JetVec2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = (
                a.coeff[0, q] * b.value[0]
                + a.coeff[1, q] * b.value[1]
                + a.value[0] * b.coeff[0, q]
                + a.value[1] * b.coeff[1, q]
            )
        return JetScalar(wp.dot(a.value, b.value), c)

    @wp.func
    def jet_dot(a: JetVec2, b: NativeVec2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a.coeff[0, q] * b[0] + a.coeff[1, q] * b[1]
        return JetScalar(wp.dot(a.value, b), c)

    @wp.func
    def jet_dot(a: NativeVec2, b: JetVec2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.coeff[0, q] + a[1] * b.coeff[1, q]
        return JetScalar(wp.dot(a, b.value), c)

    @wp.func
    def jet_dot(a: JetVec3, b: JetVec3) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = (
                a.coeff[0, q] * b.value[0]
                + a.coeff[1, q] * b.value[1]
                + a.coeff[2, q] * b.value[2]
                + a.value[0] * b.coeff[0, q]
                + a.value[1] * b.coeff[1, q]
                + a.value[2] * b.coeff[2, q]
            )
        return JetScalar(wp.dot(a.value, b.value), c)

    @wp.func
    def jet_dot(a: JetVec3, b: NativeVec3) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a.coeff[0, q] * b[0] + a.coeff[1, q] * b[1] + a.coeff[2, q] * b[2]
        return JetScalar(wp.dot(a.value, b), c)

    @wp.func
    def jet_dot(a: NativeVec3, b: JetVec3) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.coeff[0, q] + a[1] * b.coeff[1, q] + a[2] * b.coeff[2, q]
        return JetScalar(wp.dot(a, b.value), c)

    # ------------------------------------------------------------------
    # Length / normalization
    # ------------------------------------------------------------------

    @wp.func
    def jet_length_sq(v: JetVec2) -> JetScalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length_sq(v: JetVec3) -> JetScalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length(v: JetVec2) -> JetScalar:
        return jet_sqrt(jet_dot(v, v))

    @wp.func
    def jet_length(v: JetVec3) -> JetScalar:
        return jet_sqrt(jet_dot(v, v))

    @wp.func
    def jet_normalize(v: JetVec2) -> JetVec2:
        return jet_div(v, jet_length(v))

    @wp.func
    def jet_normalize(v: JetVec3) -> JetVec3:
        return jet_div(v, jet_length(v))

    # ------------------------------------------------------------------
    # 2D geometry
    #
    # Warp defines neither of these for vec2, so they stay on the namespace
    # instead of becoming builtin overloads.
    # ------------------------------------------------------------------

    @wp.func
    def jet_perp(v: JetVec2) -> JetVec2:
        c = CoeffMat2()
        for q in range(wp.static(width)):
            c[0, q] = -v.coeff[1, q]
            c[1, q] = v.coeff[0, q]
        return JetVec2(NativeVec2(-v.value[1], v.value[0]), c)

    @wp.func
    def jet_cross2(a: JetVec2, b: JetVec2) -> JetScalar:
        # Scalar 2D cross product: ax*by - ay*bx
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = (
                a.coeff[0, q] * b.value[1]
                + a.value[0] * b.coeff[1, q]
                - a.coeff[1, q] * b.value[0]
                - a.value[1] * b.coeff[0, q]
            )
        return JetScalar(
            a.value[0] * b.value[1] - a.value[1] * b.value[0],
            c,
        )

    @wp.func
    def jet_cross2(a: JetVec2, b: NativeVec2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a.coeff[0, q] * b[1] - a.coeff[1, q] * b[0]
        return JetScalar(a.value[0] * b[1] - a.value[1] * b[0], c)

    @wp.func
    def jet_cross2(a: NativeVec2, b: JetVec2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.coeff[1, q] - a[1] * b.coeff[0, q]
        return JetScalar(a[0] * b.value[1] - a[1] * b.value[0], c)

    # ------------------------------------------------------------------
    # 3D cross product
    # ------------------------------------------------------------------

    @wp.func
    def jet_cross(a: JetVec3, b: JetVec3) -> JetVec3:
        c = CoeffMat3()

        ax = a.value[0]
        ay = a.value[1]
        az = a.value[2]

        bx = b.value[0]
        by = b.value[1]
        bz = b.value[2]

        for q in range(wp.static(width)):
            dax = a.coeff[0, q]
            day = a.coeff[1, q]
            daz = a.coeff[2, q]

            dbx = b.coeff[0, q]
            dby = b.coeff[1, q]
            dbz = b.coeff[2, q]

            # d(a x b) = da x b + a x db
            c[0, q] = day * bz - daz * by + ay * dbz - az * dby
            c[1, q] = daz * bx - dax * bz + az * dbx - ax * dbz
            c[2, q] = dax * by - day * bx + ax * dby - ay * dbx

        return JetVec3(wp.cross(a.value, b.value), c)

    @wp.func
    def jet_cross(a: JetVec3, b: NativeVec3) -> JetVec3:
        c = CoeffMat3()

        bx = b[0]
        by = b[1]
        bz = b[2]

        for q in range(wp.static(width)):
            dax = a.coeff[0, q]
            day = a.coeff[1, q]
            daz = a.coeff[2, q]

            c[0, q] = day * bz - daz * by
            c[1, q] = daz * bx - dax * bz
            c[2, q] = dax * by - day * bx

        return JetVec3(wp.cross(a.value, b), c)

    @wp.func
    def jet_cross(a: NativeVec3, b: JetVec3) -> JetVec3:
        c = CoeffMat3()

        ax = a[0]
        ay = a[1]
        az = a[2]

        for q in range(wp.static(width)):
            dbx = b.coeff[0, q]
            dby = b.coeff[1, q]
            dbz = b.coeff[2, q]

            c[0, q] = ay * dbz - az * dby
            c[1, q] = az * dbx - ax * dbz
            c[2, q] = ax * dby - ay * dbx

        return JetVec3(wp.cross(a, b.value), c)

    # ------------------------------------------------------------------
    # Publish into Warp's builtin overload table.
    #
    # From here on, ordinary Warp syntax resolves on jets by argument type,
    # in any module, with nothing bound into the caller's namespace.
    # ------------------------------------------------------------------

    _register("add", jet_add)
    _register("sub", jet_sub)
    _register("mul", jet_mul)
    _register("div", jet_div)
    _register("pow", jet_pow)
    _register("neg", jet_neg)
    _register("pos", jet_pos)

    _register("sin", jet_sin)
    _register("cos", jet_cos)
    _register("tan", jet_tan)
    _register("asin", jet_asin)
    _register("acos", jet_acos)
    _register("atan", jet_atan)
    _register("atan2", jet_atan2)
    _register("exp", jet_exp)
    _register("log", jet_log)
    _register("sqrt", jet_sqrt)
    _register("abs", jet_abs)
    _register("sign", jet_sign)

    _register("min", jet_min)
    _register("max", jet_max)
    _register("clamp", jet_clamp)
    _register("where", jet_where)

    _register("extract", jet_extract)

    _register("dot", jet_dot)
    _register("length", jet_length)
    _register("length_sq", jet_length_sq)
    _register("normalize", jet_normalize)
    _register("cross", jet_cross)

    return SimpleNamespace(
        width=width,
        dtype=dtype,
        scalar=JetScalar,
        vec2=JetVec2,
        vec3=JetVec3,
        coeff=Coeff,
        native_vec2=NativeVec2,
        native_vec3=NativeVec3,
        coeff_mat2=CoeffMat2,
        coeff_mat3=CoeffMat3,
        constant=scalar_constant,
        with_coeff=scalar_with_coeff,
        seed=scalar_seed,
        make_vec2=vec2_from_scalars,
        make_vec3=vec3_from_scalars,
        constant_vec2=vec2_constant,
        constant_vec3=vec3_constant,
        seed_vec2=seed_vec2,
        seed_vec3=seed_vec3,
        directional_vec2=directional_vec2,
        directional_vec3=directional_vec3,
        perp=jet_perp,
        cross2=jet_cross2,
    )


def JetSpace(width: int, dtype=wp.float32):
    """Return the jet types and helpers for a given derivative width.

    The first call for a ``(width, dtype)`` pair generates the types and
    registers their arithmetic as overloads of Warp's builtins; later calls
    return the same cached namespace.

    Args:
        width: Number of simultaneous forward-mode directions. This is the
            number of scalar variables differentiated with respect to, not the
            dimension of ``vec2``/``vec3``.
        dtype: Scalar type the jets are built on.

    Returns:
        A namespace holding the generated types (``scalar``, ``vec2``,
        ``vec3``, ``coeff``) and the seeding and construction helpers that have
        no builtin counterpart.

    Registering the arithmetic mutates Warp's global builtin overload table,
    which is what lets ``a * b`` resolve on jets from any module. The effect is
    additive and lasts for the lifetime of the process.
    """
    width = int(width)
    key = (width, dtype)

    J = _CACHE.get(key)
    if J is None:
        J = _make_jet_space(width, dtype)
        _CACHE[key] = J

    return J


# ==========================================================================
# Second-order (forward-over-forward) jets.
#
# A width-k second-order jet carries value, gradient, and the full k x k
# Hessian of one scalar with respect to k variables:
#
#     x = value + grad . eps + 0.5 eps^T hess eps
#
# Propagating it through a scalar function in a single forward pass yields the
# local Hessian directly -- no reverse pass, no tape. The per-intermediate
# state is O(k^2), so this is a small-k strategy: register pressure grows
# quadratically where the first-order jet's grows linearly.
#
# Scalars only (enough to differentiate a scalar energy); no vec2/vec3.
# ==========================================================================

_CACHE2 = {}


def _make_jet_space2(width: int, dtype):
    if width <= 0:
        raise ValueError("Jet width must be positive")

    Grad = vector(width, dtype)
    Hess = matrix((width, width), dtype)

    @wp.struct
    class Jet2Scalar:
        value: dtype
        grad: Grad
        hess: Hess

    # ---- construction / seeding ----

    @wp.func
    def scalar_constant(x: dtype) -> Jet2Scalar:
        return Jet2Scalar(x, Grad(), Hess())

    @wp.func
    def scalar_seed(x: dtype, direction: int) -> Jet2Scalar:
        g = Grad()
        g[direction] = dtype(1.0)
        return Jet2Scalar(x, g, Hess())

    # ---- linear ops ----

    @wp.func
    def jet_add(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a.value + b.value, a.grad + b.grad, a.hess + b.hess)

    @wp.func
    def jet_add(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value + b, a.grad, a.hess)

    @wp.func
    def jet_add(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a + b.value, b.grad, b.hess)

    @wp.func
    def jet_sub(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a.value - b.value, a.grad - b.grad, a.hess - b.hess)

    @wp.func
    def jet_sub(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value - b, a.grad, a.hess)

    @wp.func
    def jet_sub(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a - b.value, -b.grad, -b.hess)

    @wp.func
    def jet_pos(a: Jet2Scalar) -> Jet2Scalar:
        return a

    @wp.func
    def jet_neg(a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(-a.value, -a.grad, -a.hess)

    @wp.func
    def jet_mul(a: Jet2Scalar, s: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value * s, a.grad * s, a.hess * s)

    @wp.func
    def jet_mul(s: dtype, a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(s * a.value, s * a.grad, s * a.hess)

    @wp.func
    def jet_div(a: Jet2Scalar, s: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value / s, a.grad / s, a.hess / s)

    # ---- product: hess picks up the grad outer products ----

    @wp.func
    def jet_mul(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        outer_ab = wp.outer(a.grad, b.grad)
        return Jet2Scalar(
            a.value * b.value,
            a.value * b.grad + b.value * a.grad,
            a.value * b.hess + b.value * a.hess + outer_ab + wp.transpose(outer_ab),
        )

    @wp.func
    def jet_div(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / b.value
        value = a.value * inv
        grad = (a.grad - value * b.grad) * inv
        outer_gb = wp.outer(grad, b.grad)
        hess = (a.hess - value * b.hess - outer_gb - wp.transpose(outer_gb)) * inv
        return Jet2Scalar(value, grad, hess)

    # ---- elementary functions: chain rule with f' and f'' ----

    @wp.func
    def _lift(value: dtype, fp: dtype, fpp: dtype, a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(value, fp * a.grad, fp * a.hess + fpp * wp.outer(a.grad, a.grad))

    @wp.func
    def jet_sin(a: Jet2Scalar) -> Jet2Scalar:
        return _lift(wp.sin(a.value), wp.cos(a.value), -wp.sin(a.value), a)

    @wp.func
    def jet_cos(a: Jet2Scalar) -> Jet2Scalar:
        return _lift(wp.cos(a.value), -wp.sin(a.value), -wp.cos(a.value), a)

    @wp.func
    def jet_exp(a: Jet2Scalar) -> Jet2Scalar:
        e = wp.exp(a.value)
        return _lift(e, e, e, a)

    @wp.func
    def jet_log(a: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / a.value
        return _lift(wp.log(a.value), inv, -inv * inv, a)

    @wp.func
    def jet_sqrt(a: Jet2Scalar) -> Jet2Scalar:
        s = wp.sqrt(a.value)
        fp = dtype(0.5) / s
        return _lift(s, fp, -dtype(0.5) * fp / a.value, a)

    @wp.func
    def jet_tan(a: Jet2Scalar) -> Jet2Scalar:
        c = wp.cos(a.value)
        sec2 = dtype(1.0) / (c * c)
        return _lift(wp.tan(a.value), sec2, dtype(2.0) * wp.tan(a.value) * sec2, a)

    @wp.func
    def jet_asin(a: Jet2Scalar) -> Jet2Scalar:
        r = dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value)
        return _lift(wp.asin(a.value), r, a.value * r * r * r, a)

    @wp.func
    def jet_acos(a: Jet2Scalar) -> Jet2Scalar:
        r = dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value)
        return _lift(wp.acos(a.value), -r, -a.value * r * r * r, a)

    @wp.func
    def jet_atan(a: Jet2Scalar) -> Jet2Scalar:
        u = dtype(1.0) / (dtype(1.0) + a.value * a.value)
        return _lift(wp.atan(a.value), u, -dtype(2.0) * a.value * u * u, a)

    @wp.func
    def jet_pow(a: Jet2Scalar, p: int) -> Jet2Scalar:
        pf = dtype(p)
        v = wp.pow(a.value, pf)
        fp = pf * wp.pow(a.value, pf - dtype(1.0))
        fpp = pf * (pf - dtype(1.0)) * wp.pow(a.value, pf - dtype(2.0))
        return _lift(v, fp, fpp, a)

    @wp.func
    def jet_pow(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        v = wp.pow(a, b.value)
        la = wp.log(a)
        return _lift(v, v * la, v * la * la, b)

    @wp.func
    def jet_div(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / b.value
        return _lift(a * inv, -a * inv * inv, dtype(2.0) * a * inv * inv * inv, b)

    # ---- two-argument chain rule: value = F(u, v), F's partials as scalars ----

    @wp.func
    def _lift2(
        value: dtype,
        fu: dtype,
        fv: dtype,
        fuu: dtype,
        fvv: dtype,
        fuv: dtype,
        u: Jet2Scalar,
        v: Jet2Scalar,
    ) -> Jet2Scalar:
        ouv = wp.outer(u.grad, v.grad)
        hess = (
            fu * u.hess
            + fv * v.hess
            + fuu * wp.outer(u.grad, u.grad)
            + fvv * wp.outer(v.grad, v.grad)
            + fuv * (ouv + wp.transpose(ouv))
        )
        return Jet2Scalar(value, fu * u.grad + fv * v.grad, hess)

    @wp.func
    def jet_atan2(y: Jet2Scalar, x: Jet2Scalar) -> Jet2Scalar:
        d = x.value * x.value + y.value * y.value
        d2 = d * d
        return _lift2(
            wp.atan2(y.value, x.value),
            x.value / d,  # d/dy
            -y.value / d,  # d/dx
            -dtype(2.0) * x.value * y.value / d2,  # d2/dy2
            dtype(2.0) * x.value * y.value / d2,  # d2/dx2
            (y.value * y.value - x.value * x.value) / d2,  # d2/dydx
            y,
            x,
        )

    @wp.func
    def jet_pow(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        g = wp.pow(a.value, b.value)
        la = wp.log(a.value)
        inva = dtype(1.0) / a.value
        return _lift2(
            g,
            g * b.value * inva,  # d/da
            g * la,  # d/db
            g * b.value * (b.value - dtype(1.0)) * inva * inva,  # d2/da2
            g * la * la,  # d2/db2
            g * (dtype(1.0) + b.value * la) * inva,  # d2/dadb
            a,
            b,
        )

    # ---- branching: value-only comparisons carry the chosen jet's derivatives ----

    @wp.func
    def jet_abs(a: Jet2Scalar) -> Jet2Scalar:
        if a.value > dtype(0.0):
            return a
        if a.value < dtype(0.0):
            return -a
        return Jet2Scalar(dtype(0.0), Grad(), Hess())

    @wp.func
    def jet_sign(a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(wp.sign(a.value), Grad(), Hess())

    @wp.func
    def jet_min(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if a.value <= b.value:
            return a
        return b

    @wp.func
    def jet_max(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if a.value >= b.value:
            return a
        return b

    @wp.func
    def jet_clamp(a: Jet2Scalar, lo: dtype, hi: dtype) -> Jet2Scalar:
        if a.value < lo:
            return Jet2Scalar(lo, Grad(), Hess())
        if a.value > hi:
            return Jet2Scalar(hi, Grad(), Hess())
        return a

    @wp.func
    def jet_where(cond: bool, a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if cond:
            return a
        return b

    _register("add", jet_add)
    _register("sub", jet_sub)
    _register("mul", jet_mul)
    _register("div", jet_div)
    _register("pow", jet_pow)
    _register("neg", jet_neg)
    _register("pos", jet_pos)

    _register("sin", jet_sin)
    _register("cos", jet_cos)
    _register("tan", jet_tan)
    _register("asin", jet_asin)
    _register("acos", jet_acos)
    _register("atan", jet_atan)
    _register("atan2", jet_atan2)
    _register("exp", jet_exp)
    _register("log", jet_log)
    _register("sqrt", jet_sqrt)
    _register("abs", jet_abs)
    _register("sign", jet_sign)

    _register("min", jet_min)
    _register("max", jet_max)
    _register("clamp", jet_clamp)
    _register("where", jet_where)

    return SimpleNamespace(
        width=width,
        dtype=dtype,
        scalar=Jet2Scalar,
        grad=Grad,
        hess=Hess,
        constant=scalar_constant,
        seed=scalar_seed,
    )


def JetSpace2(width: int, dtype=wp.float32):
    """Return second-order (forward-over-forward) jet types for a given width.

    A width-k second-order jet tracks value, gradient, and the full k x k
    Hessian; propagating it through a scalar function in one forward pass yields
    the local Hessian with no reverse pass. Per-intermediate state is O(k^2), so
    this suits small k. Scalars only.

    Args:
        width: Number of variables differentiated with respect to.
        dtype: Scalar type the jets are built on.
    """
    width = int(width)
    key = (width, dtype)

    J = _CACHE2.get(key)
    if J is None:
        J = _make_jet_space2(width, dtype)
        _CACHE2[key] = J

    return J
