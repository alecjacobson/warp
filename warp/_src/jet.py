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

:func:`JetSpace` specializes the types at code-generation time. It is described
by the inputs being differentiated, called slots:

    ============================ ======= ===================================
    ``JetSpace(slots)``          width   meaning
    ============================ ======= ===================================
    ``(wp.vec3, wp.vec3)``       6       two 3D points
    ``(wp.vec2, wp.mat22)``      6       a 2D point and a 2x2 matrix
    ``4``                        4       four scalars
    ============================ ======= ===================================

and generates:

    ========== ================== ====================
    Name       ``value``          ``coeff``
    ========== ================== ====================
    ``scalar`` ``dtype``          ``vector(width)``
    ``vec2``   ``vector(2)``      ``matrix(2, width)``
    ``vec3``   ``vector(3)``      ``matrix(3, width)``
    ========== ================== ====================

(plus ``mat2``, ``mat3``, ``mat32`` and ``mat23``). The derivative width is
independent of the geometric vector dimension.

The generated arithmetic is registered as overloads of Warp's own builtins, so
jets are written with ordinary operator and builtin syntax::

    J = wp.JetSpace((wp.vec3, wp.vec3))


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

        # Six local scalar variables: x0.x/y/z -> directions 0/1/2 and
        # x1.x/y/z -> directions 3/4/5.
        a, b = J.seed(x0[i], x1[i])

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

import itertools
import linecache
import numbers
import operator
from types import SimpleNamespace

import warp as wp
import warp._src.context
from warp._src.types import float_types, matrix, vector

# Specialized namespaces, keyed by (width, dtype): first order and second order.
_CACHE = {}
_CACHE2 = {}


def _check_space_args(width: int, dtype) -> None:
    """Validate a jet space's parameters before any type is generated.

    Checked up front because generating a space mutates Warp's global builtin
    overload table, which cannot be undone for the life of the process.
    """
    if width <= 0:
        raise ValueError("Jet width must be positive")
    if dtype not in float_types:
        raise TypeError(
            f"Jet dtype must be a Warp floating-point type, got {getattr(dtype, '__name__', dtype)}. "
            "Derivatives are not representable in an integer type: division in the chain rule "
            "would truncate."
        )


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
    _check_space_args(width, dtype)

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
    def scalar_variable(x: dtype, direction: int) -> JetScalar:
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
        # p == 0 makes this the constant 1, whose derivative is 0. The general
        # p * a**(p-1) form would evaluate it as 0 * inf at a == 0 and give NaN.
        # Non-integer p < 1 is left alone: its derivative really is infinite
        # there, and inf is the honest answer.
        if p == dtype(0.0):
            return JetScalar(dtype(1.0), Coeff())
        return _lift1(wp.pow(a.value, p), p * wp.pow(a.value, p - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, p: int) -> JetScalar:
        if p == 0:
            return JetScalar(dtype(1.0), Coeff())
        return _lift1(wp.pow(a.value, dtype(p)), dtype(p) * wp.pow(a.value, dtype(p) - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, b: JetScalar) -> JetScalar:
        # a**b: d = b a**(b-1) da + a**b log(a) db.
        #
        # Both partials are formed to stay finite at a.value == 0. Factoring
        # them out of the value instead -- value * (b / a) and value * log(a) --
        # each evaluates as 0 * inf there and gives NaN, even though the limits
        # are finite for b > 0. The da partial below is the same expression the
        # constant-exponent overload uses; the db partial takes its limit, which
        # is 0 as a -> 0 from above.
        value = wp.pow(a.value, b.value)
        da = b.value * wp.pow(a.value, b.value - dtype(1.0))
        if a.value > dtype(0.0):
            return JetScalar(value, da * a.coeff + (value * wp.log(a.value)) * b.coeff)
        return JetScalar(value, da * a.coeff)

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
    # Square matrix jets (mat2 / mat3)
    #
    # ``coeff`` flattens the matrix row-major: entry (r, c) is row r*n + c of a
    # (n*n, width) coefficient block. Enough to write FEM energies with a native
    # deformation gradient -- construction, transpose, matmul, matrix-vector
    # product, determinant, and trace.
    # ------------------------------------------------------------------

    NativeMat2 = matrix((2, 2), dtype)
    NativeMat3 = matrix((3, 3), dtype)
    NativeMat32 = matrix((3, 2), dtype)
    NativeMat23 = matrix((2, 3), dtype)
    CoeffMat4 = matrix((4, width), dtype)
    CoeffMat9 = matrix((9, width), dtype)
    CoeffMat6 = matrix((6, width), dtype)  # a 3x2 or 2x3 block, row-major

    @wp.struct
    class JetMat2:
        value: NativeMat2
        coeff: CoeffMat4

    @wp.struct
    class JetMat3:
        value: NativeMat3
        coeff: CoeffMat9

    @wp.struct
    class JetMat32:  # 3x2, e.g. a triangle deformation gradient
        value: NativeMat32
        coeff: CoeffMat6

    @wp.struct
    class JetMat23:  # 2x3
        value: NativeMat23
        coeff: CoeffMat6

    @wp.func
    def mat2_from_cols(c0: JetVec2, c1: JetVec2) -> JetMat2:
        v = NativeMat2()
        m = CoeffMat4()
        for r in range(2):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            for q in range(wp.static(width)):
                m[r * 2 + 0, q] = c0.coeff[r, q]
                m[r * 2 + 1, q] = c1.coeff[r, q]
        return JetMat2(v, m)

    @wp.func
    def mat3_from_cols(c0: JetVec3, c1: JetVec3, c2: JetVec3) -> JetMat3:
        v = NativeMat3()
        m = CoeffMat9()
        for r in range(3):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            v[r, 2] = c2.value[r]
            for q in range(wp.static(width)):
                m[r * 3 + 0, q] = c0.coeff[r, q]
                m[r * 3 + 1, q] = c1.coeff[r, q]
                m[r * 3 + 2, q] = c2.coeff[r, q]
        return JetMat3(v, m)

    @wp.func
    def mat32_from_cols(c0: JetVec3, c1: JetVec3) -> JetMat32:
        v = NativeMat32()
        m = CoeffMat6()
        for r in range(3):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            for q in range(wp.static(width)):
                m[r * 2 + 0, q] = c0.coeff[r, q]
                m[r * 2 + 1, q] = c1.coeff[r, q]
        return JetMat32(v, m)

    @wp.func
    def jet_transpose(a: JetMat32) -> JetMat23:
        m = CoeffMat6()
        for r in range(3):
            for c in range(2):
                for q in range(wp.static(width)):
                    m[c * 3 + r, q] = a.coeff[r * 2 + c, q]
        return JetMat23(wp.transpose(a.value), m)

    @wp.func
    def jet_transpose(a: JetMat23) -> JetMat32:
        m = CoeffMat6()
        for r in range(2):
            for c in range(3):
                for q in range(wp.static(width)):
                    m[c * 2 + r, q] = a.coeff[r * 3 + c, q]
        return JetMat32(wp.transpose(a.value), m)

    @wp.func
    def jet_mul(a: JetMat23, b: JetMat32) -> JetMat2:
        # (2x3)(3x2) = 2x2, product rule over the shared dimension.
        m = CoeffMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.coeff[i * 3 + k, q] * b.value[k, j] + a.value[i, k] * b.coeff[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat2(a.value * b.value, m)

    @wp.func
    def jet_add(a: JetMat2, b: JetMat2) -> JetMat2:
        return JetMat2(a.value + b.value, a.coeff + b.coeff)

    @wp.func
    def jet_add(a: JetMat3, b: JetMat3) -> JetMat3:
        return JetMat3(a.value + b.value, a.coeff + b.coeff)

    @wp.func
    def jet_sub(a: JetMat2, b: JetMat2) -> JetMat2:
        return JetMat2(a.value - b.value, a.coeff - b.coeff)

    @wp.func
    def jet_sub(a: JetMat3, b: JetMat3) -> JetMat3:
        return JetMat3(a.value - b.value, a.coeff - b.coeff)

    @wp.func
    def jet_mul(a: JetMat2, s: dtype) -> JetMat2:
        return JetMat2(a.value * s, a.coeff * s)

    @wp.func
    def jet_mul(s: dtype, a: JetMat2) -> JetMat2:
        return JetMat2(s * a.value, s * a.coeff)

    @wp.func
    def jet_mul(a: JetMat3, s: dtype) -> JetMat3:
        return JetMat3(a.value * s, a.coeff * s)

    @wp.func
    def jet_mul(s: dtype, a: JetMat3) -> JetMat3:
        return JetMat3(s * a.value, s * a.coeff)

    @wp.func
    def jet_transpose(a: JetMat3) -> JetMat3:
        m = CoeffMat9()
        for r in range(3):
            for c in range(3):
                for q in range(wp.static(width)):
                    m[c * 3 + r, q] = a.coeff[r * 3 + c, q]
        return JetMat3(wp.transpose(a.value), m)

    @wp.func
    def jet_transpose(a: JetMat2) -> JetMat2:
        m = CoeffMat4()
        for r in range(2):
            for c in range(2):
                for q in range(wp.static(width)):
                    m[c * 2 + r, q] = a.coeff[r * 2 + c, q]
        return JetMat2(wp.transpose(a.value), m)

    @wp.func
    def jet_mul(a: JetMat3, b: JetMat3) -> JetMat3:
        m = CoeffMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.coeff[i * 3 + k, q] * b.value[k, j] + a.value[i, k] * b.coeff[k * 3 + j, q]
                    m[i * 3 + j, q] = acc
        return JetMat3(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat2, b: JetMat2) -> JetMat2:
        m = CoeffMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a.coeff[i * 2 + k, q] * b.value[k, j] + a.value[i, k] * b.coeff[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat2(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat2, v: JetVec2) -> JetVec2:
        c = CoeffMat2()
        for i in range(2):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(2):
                    acc += a.coeff[i * 2 + k, q] * v.value[k] + a.value[i, k] * v.coeff[k, q]
                c[i, q] = acc
        return JetVec2(a.value * v.value, c)

    @wp.func
    def jet_mul(a: JetMat3, v: JetVec3) -> JetVec3:
        c = CoeffMat3()
        for i in range(3):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(3):
                    acc += a.coeff[i * 3 + k, q] * v.value[k] + a.value[i, k] * v.coeff[k, q]
                c[i, q] = acc
        return JetVec3(a.value * v.value, c)

    @wp.func
    def jet_trace(a: JetMat3) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a.coeff[0, q] + a.coeff[4, q] + a.coeff[8, q]
        return JetScalar(a.value[0, 0] + a.value[1, 1] + a.value[2, 2], c)

    @wp.func
    def jet_trace(a: JetMat2) -> JetScalar:
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = a.coeff[0, q] + a.coeff[3, q]
        return JetScalar(a.value[0, 0] + a.value[1, 1], c)

    @wp.func
    def jet_determinant(a: JetMat3) -> JetScalar:
        # d(det A) = sum_ij cofactor(i,j) * dA[i,j]  (Jacobi's formula).
        m = a.value
        cof0 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
        cof1 = -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        cof2 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
        cof3 = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
        cof4 = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
        cof5 = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])
        cof6 = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
        cof7 = -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0])
        cof8 = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = (
                cof0 * a.coeff[0, q]
                + cof1 * a.coeff[1, q]
                + cof2 * a.coeff[2, q]
                + cof3 * a.coeff[3, q]
                + cof4 * a.coeff[4, q]
                + cof5 * a.coeff[5, q]
                + cof6 * a.coeff[6, q]
                + cof7 * a.coeff[7, q]
                + cof8 * a.coeff[8, q]
            )
        return JetScalar(wp.determinant(m), c)

    @wp.func
    def jet_determinant(a: JetMat2) -> JetScalar:
        m = a.value
        c = Coeff()
        for q in range(wp.static(width)):
            c[q] = m[1, 1] * a.coeff[0, q] - m[1, 0] * a.coeff[1, q] - m[0, 1] * a.coeff[2, q] + m[0, 0] * a.coeff[3, q]
        return JetScalar(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0], c)

    @wp.func
    def jet_inverse(a: JetMat3) -> JetMat3:
        # d(A^-1) = -A^-1 (dA) A^-1.
        ainv = wp.inverse(a.value)
        m = CoeffMat9()
        for q in range(wp.static(width)):
            da = NativeMat3()
            for r in range(3):
                for c in range(3):
                    da[r, c] = a.coeff[r * 3 + c, q]
            dinv = -(ainv * da * ainv)
            for r in range(3):
                for c in range(3):
                    m[r * 3 + c, q] = dinv[r, c]
        return JetMat3(ainv, m)

    @wp.func
    def jet_inverse(a: JetMat2) -> JetMat2:
        ainv = wp.inverse(a.value)
        m = CoeffMat4()
        for q in range(wp.static(width)):
            da = NativeMat2()
            for r in range(2):
                for c in range(2):
                    da[r, c] = a.coeff[r * 2 + c, q]
            dinv = -(ainv * da * ainv)
            for r in range(2):
                for c in range(2):
                    m[r * 2 + c, q] = dinv[r, c]
        return JetMat2(ainv, m)

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

    _register("transpose", jet_transpose)
    _register("determinant", jet_determinant)
    _register("trace", jet_trace)
    _register("inverse", jet_inverse)

    return SimpleNamespace(
        width=width,
        dtype=dtype,
        scalar=JetScalar,
        vec2=JetVec2,
        vec3=JetVec3,
        mat2=JetMat2,
        mat3=JetMat3,
        mat32=JetMat32,
        mat23=JetMat23,
        coeff=Coeff,
        native_vec2=NativeVec2,
        native_vec3=NativeVec3,
        native_mat2=NativeMat2,
        native_mat3=NativeMat3,
        coeff_mat2=CoeffMat2,
        coeff_mat3=CoeffMat3,
        constant=scalar_constant,
        with_coeff=scalar_with_coeff,
        variable=scalar_variable,
        make_vec2=vec2_from_scalars,
        make_vec3=vec3_from_scalars,
        constant_vec2=vec2_constant,
        constant_vec3=vec3_constant,
        make_mat2=mat2_from_cols,
        make_mat3=mat3_from_cols,
        make_mat32=mat32_from_cols,
        perp=jet_perp,
        cross2=jet_cross2,
        _slot_jet_types={
            "scalar": (JetScalar, Coeff),
            "vec2": (JetVec2, CoeffMat2),
            "vec3": (JetVec3, CoeffMat3),
            "mat2": (JetMat2, CoeffMat4),
            "mat3": (JetMat3, CoeffMat9),
            "mat32": (JetMat32, CoeffMat6),
            "mat23": (JetMat23, CoeffMat6),
        },
    )


# Shape of each supported slot kind, and the order of the components it
# contributes to the derivative directions: row-major for matrices.
_SLOT_SHAPES = {
    "scalar": (),
    "vec2": (2,),
    "vec3": (3,),
    "mat2": (2, 2),
    "mat3": (3, 3),
    "mat32": (3, 2),
    "mat23": (2, 3),
}
_SHAPE_TO_KIND = {shape: kind for kind, shape in _SLOT_SHAPES.items()}

_SLOT_SUMMARY = "scalar, vec2, vec3, mat22, mat33, mat32 and mat23"

# Layout-specific namespaces (what JetSpace and JetSpace2 return), keyed
# by (order, width, dtype, slot kinds). Types and arithmetic live in _CACHE and
# _CACHE2 and are shared by every layout of the same width and dtype.
_CACHE_LAYOUT = {}

_GENERATED_IDS = itertools.count()


class _JetNamespace(SimpleNamespace):
    """Namespace of a jet space that explains the helpers it deliberately lacks."""

    def __getattr__(self, name):
        # Only reached when normal lookup fails.
        if name == "seed":
            raise AttributeError(
                f"seed() needs one derivative direction per component of the slots, "
                f"but this space has width {self.width} for {self.slot_components} components. "
                "Use directional() to supply the directions explicitly."
            )
        raise AttributeError(name)


def _slot_components(kind: str):
    """Return the index tuples of the components a slot of ``kind`` contributes."""
    shape = _SLOT_SHAPES[kind]
    if len(shape) == 0:
        return [()]
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _classify_slot(slot):
    """Return ``(kind, scalar_dtype)`` for a slot type.

    ``scalar_dtype`` is ``None`` for Python ``float``, which does not commit to a precision.
    """
    if slot is float:
        return "scalar", None
    if slot in float_types:
        return "scalar", slot

    shape = getattr(slot, "_shape_", None)
    scalar_dtype = getattr(slot, "_wp_scalar_type_", None)
    if shape is not None and scalar_dtype is not None and tuple(shape) in _SHAPE_TO_KIND:
        if scalar_dtype not in float_types:
            raise TypeError(f"Jet slot {slot.__name__} must have a floating-point scalar type")
        return _SHAPE_TO_KIND[tuple(shape)], scalar_dtype

    raise TypeError(
        f"Unsupported jet slot type {getattr(slot, '__name__', slot)!r}. Supported slot types are {_SLOT_SUMMARY}."
    )


def _normalize_slots(slots, dtype):
    """Resolve the ``slots`` argument of :func:`JetSpace` into ``(kinds, dtype)``."""
    if isinstance(slots, numbers.Integral) and not isinstance(slots, bool):
        count = int(slots)
        if count <= 0:
            raise ValueError("Jet width must be positive")
        return ("scalar",) * count, wp.float32 if dtype is None else dtype

    if not isinstance(slots, (tuple, list)):
        slots = (slots,)
    if len(slots) == 0:
        raise ValueError("A jet space needs at least one slot")

    kinds = []
    inferred = set()
    for slot in slots:
        kind, scalar_dtype = _classify_slot(slot)
        kinds.append(kind)
        if scalar_dtype is not None:
            inferred.add(scalar_dtype)

    if len(inferred) > 1:
        names = ", ".join(sorted(t.__name__ for t in inferred))
        raise TypeError(f"Jet slots mix scalar types ({names}); all slots must share one precision")
    if dtype is not None and inferred and dtype not in inferred:
        raise TypeError(
            f"dtype={dtype.__name__} conflicts with the slot types, which are {next(iter(inferred)).__name__}"
        )

    if dtype is None:
        dtype = next(iter(inferred)) if inferred else wp.float32
    return tuple(kinds), dtype


def _native_slot_type(kind: str, dtype):
    shape = _SLOT_SHAPES[kind]
    if len(shape) == 0:
        return dtype
    if len(shape) == 1:
        return vector(shape[0], dtype)
    return matrix(shape, dtype)


def _define_generated(name: str, source: str, namespace: dict):
    """Compile ``source`` as a Warp function named ``name``.

    The number of slots varies per space, so the signature and body of the
    helpers below cannot be written out once. The source is registered with
    :mod:`linecache` so Warp's codegen can read it back like any other function.
    """
    filename = f"<warp-jet-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": __name__, "wp": wp, **namespace}
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def _coeff_assignment(kind: str, component: int, column: int, value: str, var: str) -> str:
    # A scalar slot's coefficient is a vector over directions; every other slot's is
    # a (components, directions) matrix, with matrices flattened row-major.
    if kind == "scalar":
        return f"{var}[{column}] = {value}"
    return f"{var}[{component}, {column}] = {value}"


def _build_slot_helpers(base, kinds, width: int, dtype, second_order: bool, include_directional: bool):
    """Generate ``seed`` (and ``directional``) for one slot layout.

    ``seed`` places a unit direction on every component of every slot, in order.
    ``directional`` takes the directions explicitly, so ``width`` may be smaller
    than the number of components -- one direction gives a Hessian-vector product.

    Both take their operands as separate positional arguments rather than as
    tuples. Warp does not propagate adjoints into a tuple argument, which would
    silently zero the reverse-over-forward derivatives these helpers exist to
    support.
    """
    uid = next(_GENERATED_IDS)
    nslots = len(kinds)

    natives = tuple(_native_slot_type(kind, dtype) for kind in kinds)

    namespace = {"dtype": dtype}
    for kind, (jet_type, coeff_type) in base._slot_jet_types.items():
        namespace[f"Jet_{kind}"] = jet_type
        namespace[f"Coeff_{kind}"] = coeff_type
    if second_order:
        namespace["Hess"] = base.hess

    # Annotation types are passed by name so the source stays readable.
    for s_, native in enumerate(natives):
        namespace[f"Native_{s_}"] = native

    value_params = [f"v{s}: Native_{s}" for s in range(nslots)]

    def construct(slot: int) -> str:
        kind = kinds[slot]
        extra = ", Hess()" if second_order else ""
        return f"    j{slot} = Jet_{kind}(v{slot}, c{slot}{extra})"

    def returns():
        return "    return " + ", ".join(f"j{s}" for s in range(nslots))

    # seed: component k of every slot gets its own unit direction.
    lines = ["@wp.func", f"def jet_seed_{uid}({', '.join(value_params)}):"]
    offset = 0
    for s, kind in enumerate(kinds):
        lines.append(f"    c{s} = Coeff_{kind}()")
        for k, _ in enumerate(_slot_components(kind)):
            lines.append("    " + _coeff_assignment(kind, k, offset + k, "dtype(1.0)", f"c{s}"))
        lines.append(construct(s))
        offset += len(_slot_components(kind))
    lines.append(returns())

    total = offset
    seed = _define_generated(f"jet_seed_{uid}", "\n".join(lines) + "\n", namespace) if total == width else None

    directional = None
    if include_directional:
        # One direction per column. After the values come the directions, flat and
        # direction-major: argument d{q * nslots + s} is slot s's value along
        # direction q.
        direction_params = [f"d{q * nslots + s}: Native_{s}" for q in range(width) for s in range(nslots)]

        lines = ["@wp.func", f"def jet_directional_{uid}({', '.join(value_params + direction_params)}):"]
        for s, kind in enumerate(kinds):
            lines.append(f"    c{s} = Coeff_{kind}()")
            for q in range(width):
                for k, component in enumerate(_slot_components(kind)):
                    access = f"d{q * nslots + s}" + (f"[{', '.join(map(str, component))}]" if component else "")
                    lines.append("    " + _coeff_assignment(kind, k, q, access, f"c{s}"))
            lines.append(construct(s))
        lines.append(returns())
        directional = _define_generated(f"jet_directional_{uid}", "\n".join(lines) + "\n", namespace)

    return seed, directional, natives, total


def _layout_namespace(base, kinds, width: int, dtype, second_order: bool):
    key = (second_order, width, dtype, kinds)
    J = _CACHE_LAYOUT.get(key)
    if J is None:
        seed, directional, natives, total = _build_slot_helpers(
            base, kinds, width, dtype, second_order, include_directional=not second_order
        )
        members = {k: v for k, v in vars(base).items() if not k.startswith("_")}
        members.update(slots=natives, slot_components=total)
        if directional is not None:
            members["directional"] = directional
        if seed is not None:
            members["seed"] = seed
        J = _JetNamespace(**members)
        _CACHE_LAYOUT[key] = J
    return J


def JetSpace(slots, dtype=None, *, width=None):
    """Return the jet types and helpers for a set of differentiated inputs.

    The first call for a given width and scalar type generates the types and
    registers their arithmetic as overloads of Warp's builtins; later calls
    return a namespace over the same types.

    Args:
        slots: The inputs differentiated with respect to. Either an integer
            ``n``, meaning ``n`` scalar inputs, or a sequence of types, one per
            input: ``wp.float32``, ``wp.vec2``, ``wp.vec3``, ``wp.mat22``,
            ``wp.mat33``, ``wp.mat32`` or ``wp.mat23``. A spring over two nodes
            is ``(wp.vec3, wp.vec3)``.
        dtype: Scalar type the jets are built on. Inferred from ``slots`` when
            they are Warp types; ``wp.float32`` when ``slots`` is an integer
            or contains only Python ``float``.
        width: Number of derivative directions each jet carries. Defaults to the
            total number of scalar components in ``slots``, so that a jet carries
            the full gradient. A smaller value suits directional derivatives,
            such as a Hessian-vector product with ``width=1``.

    Returns:
        A namespace holding the generated types (``scalar``, ``vec2``,
        ``vec3``, ``mat2``, ``mat3``, ``mat32``, ``mat23``, ``coeff``), the
        helpers that construct jets, and ``slots``, the native type of each slot.

    Jets are created inside a kernel with ``seed`` or ``directional``::

        J = wp.JetSpace((wp.vec3, wp.vec3))
        a, b = J.seed(x0, x1)

    ``seed(*values)`` takes one value per slot and returns one jet per slot,
    with the derivative directions numbered slot by slot in the order the
    components appear (row-major for matrices). With a single slot it takes and
    returns the bare value. It is available only when ``width`` equals the
    number of components; otherwise use ``directional``.

    ``directional(*values, *directions)`` takes the same values, followed by
    ``width * len(slots)`` direction vectors ordered direction-major: direction
    ``q`` of slot ``s`` is argument ``q * len(slots) + s`` after the values. For
    a width-1 space the directions are one value per slot, and the jets carry the
    derivative along the combined direction::

        J = wp.JetSpace((wp.vec3, wp.vec3), width=1)
        a, b = J.directional(x0, x1, v0, v1)

    Operands are positional arguments because Warp does not propagate adjoints
    into a tuple argument, which would silently zero the reverse-over-forward
    derivatives these helpers exist to support.

    ``variable(x, direction)`` is the underlying scalar primitive: a scalar jet
    with a unit derivative along ``direction``. Use it to seed in a loop, where
    the number of scalars is not a fixed argument list.

    ``width`` is fixed when the types are specialized, not when a kernel runs:
    it becomes the length of ``coeff`` and is unrolled into the generated code.
    Every intermediate value carries all ``width`` derivative components, so
    cost in registers and compile time scales with the width requested,
    regardless of how many components are read back. Different widths are
    independent specializations that may coexist in one process.

    This makes jets suited to *local* derivatives of fixed arity -- a spring
    over two ``vec3`` nodes has width 6, a triangle 9, a tetrahedron 12 -- where
    the layout is known where the kernel is written and the dense local gradient
    or Hessian block is scattered into a sparse global matrix. They are not
    suited to differentiating with respect to an unbounded or runtime number of
    variables: a problem-sized width cannot specialize a type, and even when it
    could, forward mode would cost one direction per variable against reverse
    mode's single backward pass. Use :class:`warp.Tape` or :func:`warp.grad` for
    that outer derivative, and jets for the fixed-arity term inside it.

    Registering the arithmetic mutates Warp's global builtin overload table,
    which is what lets ``a * b`` resolve on jets from any module. The effect is
    additive and lasts for the lifetime of the process.
    """
    kinds, dtype = _normalize_slots(slots, dtype)
    total = sum(len(_slot_components(kind)) for kind in kinds)

    if width is None:
        width = total
    else:
        width = operator.index(width)

    key = (width, dtype)
    base = _CACHE.get(key)
    if base is None:
        base = _make_jet_space(width, dtype)
        _CACHE[key] = base

    return _layout_namespace(base, kinds, width, dtype, second_order=False)


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


def _make_jet_space2(width: int, dtype):
    _check_space_args(width, dtype)

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
    def scalar_variable(x: dtype, direction: int) -> Jet2Scalar:
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
    def _pow_const(a: Jet2Scalar, p: dtype) -> Jet2Scalar:
        """a**p for a constant exponent, with the degenerate exponents guarded.

        The general term p (p-1) a**(p-2) evaluates as 0 * inf at a == 0 for
        p == 1, and both derivative terms do so for p == 0, where the true
        derivatives are zero. Non-integer p below those is left alone: its
        derivatives really are infinite at zero, so inf is the honest answer.
        """
        if p == dtype(0.0):
            return Jet2Scalar(dtype(1.0), Grad(), Hess())

        v = wp.pow(a.value, p)
        fp = p * wp.pow(a.value, p - dtype(1.0))

        if p == dtype(1.0):
            return _lift(v, fp, dtype(0.0), a)

        return _lift(v, fp, p * (p - dtype(1.0)) * wp.pow(a.value, p - dtype(2.0)), a)

    @wp.func
    def jet_pow(a: Jet2Scalar, p: int) -> Jet2Scalar:
        return _pow_const(a, dtype(p))

    @wp.func
    def jet_pow(a: Jet2Scalar, p: dtype) -> Jet2Scalar:
        return _pow_const(a, p)

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
    def jet_atan2(y: Jet2Scalar, x: dtype) -> Jet2Scalar:
        d = x * x + y.value * y.value
        return _lift(wp.atan2(y.value, x), x / d, -dtype(2.0) * x * y.value / (d * d), y)

    @wp.func
    def jet_atan2(y: dtype, x: Jet2Scalar) -> Jet2Scalar:
        d = x.value * x.value + y * y
        return _lift(wp.atan2(y, x.value), -y / d, dtype(2.0) * x.value * y / (d * d), x)

    @wp.func
    def jet_pow(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        # Partials written against wp.pow rather than as g / a, for the same
        # reason as the first-order overload: at a.value == 0 the 1 / a form
        # gives 0 * inf = NaN where the limit is finite. The three partials
        # carrying log(a) are zero in that limit for b > 0.
        g = wp.pow(a.value, b.value)
        da = b.value * wp.pow(a.value, b.value - dtype(1.0))
        data = b.value * (b.value - dtype(1.0)) * wp.pow(a.value, b.value - dtype(2.0))
        if a.value > dtype(0.0):
            la = wp.log(a.value)
            return _lift2(
                g,
                da,  # d/da
                g * la,  # d/db
                data,  # d2/da2
                g * la * la,  # d2/db2
                wp.pow(a.value, b.value - dtype(1.0)) * (dtype(1.0) + b.value * la),  # d2/dadb
                a,
                b,
            )
        return _lift2(g, da, dtype(0.0), data, dtype(0.0), dtype(0.0), a, b)

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
        variable=scalar_variable,
        _slot_jet_types={"scalar": (Jet2Scalar, Grad)},
    )


def JetSpace2(slots, dtype=None):
    """Return second-order (forward-over-forward) jet types for a set of scalars.

    A width-k second-order jet tracks value, gradient, and the full k x k
    Hessian; propagating it through a scalar function in one forward pass yields
    the local Hessian with no reverse pass. Per-intermediate state is O(k^2), so
    this suits small k. Scalars only.

    Args:
        slots: Number of scalar variables differentiated with respect to, or a
            sequence of scalar types, one per variable. Fixed when the types are
            specialized, not when a kernel runs.
        dtype: Scalar type the jets are built on; ``wp.float32`` by default.

    Returns:
        A namespace holding the generated type (``scalar``) and the helpers that
        construct jets.

    ``seed(*values)`` takes one scalar per variable and returns one jet per
    variable, each with a unit derivative along its own direction.
    ``variable(x, direction)`` seeds a single scalar, for use in a loop.

    As with :func:`warp.JetSpace`, the width is a compile-time constant and the
    types are for local derivatives of fixed, statically known arity. The
    quadratic state makes that limit tighter here: register pressure and compile
    time both grow with ``width**2``, so for wider energies prefer a first-order
    jet with a reverse sweep over it (via :class:`warp.Tape`, or in-kernel with
    :func:`warp.grad`), which reaches the same Hessian with linear state.
    """
    kinds, dtype = _normalize_slots(slots, dtype)
    if any(kind != "scalar" for kind in kinds):
        raise TypeError("Second-order jets support scalar slots only")
    width = len(kinds)

    key = (width, dtype)
    base = _CACHE2.get(key)
    if base is None:
        base = _make_jet_space2(width, dtype)
        _CACHE2[key] = base

    return _layout_namespace(base, kinds, width, dtype, second_order=True)
