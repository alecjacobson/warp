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

Matrix payloads (``mat2``/``mat3``/``mat32``/``mat23``) carry a deformation
gradient as a jet, and ``quat``/``vec3`` payloads with an ``exp_map``
rotation-vector chart optimize over 3D rotations; the same payloads and chart
exist for second-order jets (:func:`JetSpace2`), giving the tangent Hessian of a
rotation energy in one forward pass.
"""

# NOTE: deliberately no `from __future__ import annotations` here. The structs
# and functions below are annotated with types created inside _make_jet_space,
# and Warp resolves struct annotations with inspect.get_annotations(eval_str=True),
# which evaluates them against module globals. Deferred annotations would turn
# them into strings that no longer resolve.

from types import SimpleNamespace

import warp as wp
import warp._src.context
from warp._src.types import float_types, matrix, quaternion, vector

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
    # Quaternion jets ([x, y, z, w] storage, matching wp.quat).
    #
    # Same scalar-decomposition strategy as the geometry overloads above: the
    # Hamilton product, quat_rotate, and the rotation-vector exp map are written
    # by extracting components to scalar jets, composing, and reassembling, so no
    # derivative is written by hand. quat jets are NOT auto-normalized: a
    # unit-rotation parametrization comes from seeding a vec3 tangent and calling
    # exp_map; seeding the four components directly gives full 4-DOF derivatives.
    # ------------------------------------------------------------------

    NativeQuat = quaternion(dtype)
    QuatCoeff = matrix((4, width), dtype)

    @wp.struct
    class JetQuat:
        value: NativeQuat
        coeff: QuatCoeff

    @wp.func
    def quat_extract(q: JetQuat, c: int) -> JetScalar:
        s = Coeff()
        for i in range(wp.static(width)):
            s[i] = q.coeff[c, i]
        return JetScalar(q.value[c], s)

    @wp.func
    def quat_from_scalars(x: JetScalar, y: JetScalar, z: JetScalar, w: JetScalar) -> JetQuat:
        c = QuatCoeff()
        for i in range(wp.static(width)):
            c[0, i] = x.coeff[i]
            c[1, i] = y.coeff[i]
            c[2, i] = z.coeff[i]
            c[3, i] = w.coeff[i]
        return JetQuat(NativeQuat(x.value, y.value, z.value, w.value), c)

    @wp.func
    def quat_constant(q: NativeQuat) -> JetQuat:
        return JetQuat(q, QuatCoeff())

    @wp.func
    def quat_seed(q: NativeQuat, i0: int, i1: int, i2: int, i3: int) -> JetQuat:
        c = QuatCoeff()
        c[0, i0] = dtype(1.0)
        c[1, i1] = dtype(1.0)
        c[2, i2] = dtype(1.0)
        c[3, i3] = dtype(1.0)
        return JetQuat(q, c)

    @wp.func
    def _quat_hamilton(
        ax: JetScalar,
        ay: JetScalar,
        az: JetScalar,
        aw: JetScalar,
        bx: JetScalar,
        by: JetScalar,
        bz: JetScalar,
        bw: JetScalar,
    ) -> JetQuat:
        rx = jet_sub(jet_add(jet_add(jet_mul(aw, bx), jet_mul(bw, ax)), jet_mul(ay, bz)), jet_mul(az, by))
        ry = jet_sub(jet_add(jet_add(jet_mul(aw, by), jet_mul(bw, ay)), jet_mul(az, bx)), jet_mul(ax, bz))
        rz = jet_sub(jet_add(jet_add(jet_mul(aw, bz), jet_mul(bw, az)), jet_mul(ax, by)), jet_mul(ay, bx))
        rw = jet_sub(jet_sub(jet_sub(jet_mul(aw, bw), jet_mul(ax, bx)), jet_mul(ay, by)), jet_mul(az, bz))
        return quat_from_scalars(rx, ry, rz, rw)

    @wp.func
    def jet_qmul(a: JetQuat, b: JetQuat) -> JetQuat:
        return _quat_hamilton(
            quat_extract(a, 0),
            quat_extract(a, 1),
            quat_extract(a, 2),
            quat_extract(a, 3),
            quat_extract(b, 0),
            quat_extract(b, 1),
            quat_extract(b, 2),
            quat_extract(b, 3),
        )

    @wp.func
    def jet_qmul(a: JetQuat, b: NativeQuat) -> JetQuat:
        return _quat_hamilton(
            quat_extract(a, 0),
            quat_extract(a, 1),
            quat_extract(a, 2),
            quat_extract(a, 3),
            scalar_constant(b[0]),
            scalar_constant(b[1]),
            scalar_constant(b[2]),
            scalar_constant(b[3]),
        )

    @wp.func
    def jet_qmul(a: NativeQuat, b: JetQuat) -> JetQuat:
        return _quat_hamilton(
            scalar_constant(a[0]),
            scalar_constant(a[1]),
            scalar_constant(a[2]),
            scalar_constant(a[3]),
            quat_extract(b, 0),
            quat_extract(b, 1),
            quat_extract(b, 2),
            quat_extract(b, 3),
        )

    @wp.func
    def jet_quat_rotate(q: JetQuat, x: NativeVec3) -> JetVec3:
        # Native quat_rotate expansion; presumes a unit q.
        two = dtype(2.0)
        qx = quat_extract(q, 0)
        qy = quat_extract(q, 1)
        qz = quat_extract(q, 2)
        qw = quat_extract(q, 3)
        xx = x[0]
        xy = x[1]
        xz = x[2]

        c = jet_sub(jet_mul(two, jet_mul(qw, qw)), dtype(1.0))
        d = jet_mul(two, jet_add(jet_add(jet_mul(qx, xx), jet_mul(qy, xy)), jet_mul(qz, xz)))

        rx = jet_add(
            jet_add(jet_mul(c, xx), jet_mul(qx, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qy, xz), jet_mul(qz, xy)), qw), two),
        )
        ry = jet_add(
            jet_add(jet_mul(c, xy), jet_mul(qy, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qz, xx), jet_mul(qx, xz)), qw), two),
        )
        rz = jet_add(
            jet_add(jet_mul(c, xz), jet_mul(qz, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qx, xy), jet_mul(qy, xx)), qw), two),
        )
        return vec3_from_scalars(rx, ry, rz)

    @wp.func
    def jet_quat_rotate_inv(q: JetQuat, x: NativeVec3) -> JetVec3:
        qc = quat_from_scalars(
            jet_neg(quat_extract(q, 0)),
            jet_neg(quat_extract(q, 1)),
            jet_neg(quat_extract(q, 2)),
            quat_extract(q, 3),
        )
        return jet_quat_rotate(qc, x)

    @wp.func
    def jet_qdot(a: JetQuat, b: JetQuat) -> JetScalar:
        return jet_add(
            jet_add(
                jet_add(
                    jet_mul(quat_extract(a, 0), quat_extract(b, 0)),
                    jet_mul(quat_extract(a, 1), quat_extract(b, 1)),
                ),
                jet_mul(quat_extract(a, 2), quat_extract(b, 2)),
            ),
            jet_mul(quat_extract(a, 3), quat_extract(b, 3)),
        )

    @wp.func
    def jet_qlength(q: JetQuat) -> JetScalar:
        return jet_sqrt(jet_qdot(q, q))

    @wp.func
    def jet_qnormalize(q: JetQuat) -> JetQuat:
        inv = jet_div(dtype(1.0), jet_qlength(q))
        return quat_from_scalars(
            jet_mul(quat_extract(q, 0), inv),
            jet_mul(quat_extract(q, 1), inv),
            jet_mul(quat_extract(q, 2), inv),
            jet_mul(quat_extract(q, 3), inv),
        )

    @wp.func
    def quat_exp_map(v: JetVec3) -> JetQuat:
        # q(v) = [ v * sinc_half(s), cos_half(s) ], s = |v|^2, both smooth in v.
        # Series in s near 0 (exact via jet arithmetic), closed form away from 0.
        s = jet_dot(v, v)
        s2 = jet_mul(s, s)
        s3 = jet_mul(s2, s)

        cw_series = jet_add(
            jet_add(
                jet_add(scalar_constant(dtype(1.0)), jet_mul(s, dtype(-1.0 / 8.0))),
                jet_mul(s2, dtype(1.0 / 384.0)),
            ),
            jet_mul(s3, dtype(-1.0 / 46080.0)),
        )
        g_series = jet_add(
            jet_add(
                jet_add(scalar_constant(dtype(0.5)), jet_mul(s, dtype(-1.0 / 48.0))),
                jet_mul(s2, dtype(1.0 / 3840.0)),
            ),
            jet_mul(s3, dtype(-1.0 / 645120.0)),
        )

        a = jet_sqrt(s)
        h = jet_mul(a, dtype(0.5))
        cw_closed = jet_cos(h)
        g_closed = jet_div(jet_sin(h), a)

        small = s.value < dtype(1.0e-3)
        cw = jet_where(small, cw_series, cw_closed)
        g = jet_where(small, g_series, g_closed)

        return quat_from_scalars(
            jet_mul(jet_extract(v, 0), g),
            jet_mul(jet_extract(v, 1), g),
            jet_mul(jet_extract(v, 2), g),
            cw,
        )

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

    _register("mul", jet_qmul)
    _register("quat_rotate", jet_quat_rotate)
    _register("quat_rotate_inv", jet_quat_rotate_inv)
    _register("dot", jet_qdot)
    _register("length", jet_qlength)
    _register("normalize", jet_qnormalize)
    _register("extract", quat_extract)

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
        seed=scalar_seed,
        make_vec2=vec2_from_scalars,
        make_vec3=vec3_from_scalars,
        constant_vec2=vec2_constant,
        constant_vec3=vec3_constant,
        seed_vec2=seed_vec2,
        seed_vec3=seed_vec3,
        directional_vec2=directional_vec2,
        directional_vec3=directional_vec3,
        make_mat2=mat2_from_cols,
        make_mat3=mat3_from_cols,
        make_mat32=mat32_from_cols,
        perp=jet_perp,
        cross2=jet_cross2,
        quat=JetQuat,
        native_quat=NativeQuat,
        quat_coeff=QuatCoeff,
        make_quat=quat_from_scalars,
        constant_quat=quat_constant,
        seed_quat=quat_seed,
        exp_map=quat_exp_map,
        quat_from_rotvec=quat_exp_map,
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

    ``width`` is fixed when the types are specialized, not when a kernel runs:
    it becomes the length of ``coeff`` and is unrolled into the generated code.
    Every intermediate value carries all ``width`` derivative components, so
    cost in registers and compile time scales with the width requested,
    regardless of how many components are read back. Different widths are
    independent specializations that may coexist in one process.

    This makes jets suited to *local* derivatives of fixed arity -- a spring
    over two ``vec3`` nodes is ``width=6``, a triangle ``width=9``, a
    tetrahedron ``width=12`` -- where the width is known where the kernel is
    written and the dense local gradient or Hessian block is scattered into a
    sparse global matrix. They are not suited to differentiating with respect to
    an unbounded or runtime number of variables: a problem-sized width cannot
    specialize a type, and even when it could, forward mode would cost one
    direction per variable against reverse mode's single backward pass. Use
    :class:`warp.Tape` or :func:`warp.grad` for that outer derivative, and jets
    for the fixed-arity term inside it.

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
# The scalar payload is enough to differentiate a scalar energy; vec3 and quat
# payloads are also provided (see below) so a scalar objective written through
# 3D rotations differentiates without hand-derivatives.
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

    # ==================================================================
    # Second-order vec3 / quat payloads.
    #
    # These carry a vec3 or quaternion value alongside the per-component
    # gradient and Hessian, so a scalar energy written through a rotation
    # (``exp_map`` -> ``mul`` -> ``quat_rotate``) yields a dense local Hessian
    # in one forward pass. ``grad`` stores component c's gradient in row c;
    # ``hess`` stacks the component Hessians row-wise, so entry (i, j) of
    # component c's width x width Hessian is ``hess[c * width + i, j]``.
    #
    # Every nonlinear op below is written by extracting each component to a
    # Jet2Scalar, composing with the scalar chain rule already defined above,
    # and reassembling. The O(k^2) Hessian bookkeeping therefore rides entirely
    # on the tested scalar overloads; nothing here derives a second derivative
    # by hand.
    # ==================================================================

    NativeVec3 = vector(3, dtype)
    NativeQuat = quaternion(dtype)

    GradVec3 = matrix((3, width), dtype)
    HessVec3 = matrix((3 * width, width), dtype)
    GradQuat = matrix((4, width), dtype)
    HessQuat = matrix((4 * width, width), dtype)

    @wp.struct
    class Jet2Vec3:
        value: NativeVec3
        grad: GradVec3
        hess: HessVec3

    @wp.struct
    class Jet2Quat:
        value: NativeQuat
        grad: GradQuat
        hess: HessQuat

    # ---- component access: pack/unpack against Jet2Scalar ----

    @wp.func
    def vec3_extract(v: Jet2Vec3, c: int) -> Jet2Scalar:
        g = Grad()
        h = Hess()
        for i in range(wp.static(width)):
            g[i] = v.grad[c, i]
            for j in range(wp.static(width)):
                h[i, j] = v.hess[c * width + i, j]
        return Jet2Scalar(v.value[c], g, h)

    @wp.func
    def quat_extract(q: Jet2Quat, c: int) -> Jet2Scalar:
        g = Grad()
        h = Hess()
        for i in range(wp.static(width)):
            g[i] = q.grad[c, i]
            for j in range(wp.static(width)):
                h[i, j] = q.hess[c * width + i, j]
        return Jet2Scalar(q.value[c], g, h)

    @wp.func
    def vec3_from_scalars(x: Jet2Scalar, y: Jet2Scalar, z: Jet2Scalar) -> Jet2Vec3:
        g = GradVec3()
        h = HessVec3()
        for i in range(wp.static(width)):
            g[0, i] = x.grad[i]
            g[1, i] = y.grad[i]
            g[2, i] = z.grad[i]
            for j in range(wp.static(width)):
                h[i, j] = x.hess[i, j]
                h[width + i, j] = y.hess[i, j]
                h[2 * width + i, j] = z.hess[i, j]
        return Jet2Vec3(NativeVec3(x.value, y.value, z.value), g, h)

    @wp.func
    def quat_from_scalars(x: Jet2Scalar, y: Jet2Scalar, z: Jet2Scalar, w: Jet2Scalar) -> Jet2Quat:
        g = GradQuat()
        h = HessQuat()
        for i in range(wp.static(width)):
            g[0, i] = x.grad[i]
            g[1, i] = y.grad[i]
            g[2, i] = z.grad[i]
            g[3, i] = w.grad[i]
            for j in range(wp.static(width)):
                h[i, j] = x.hess[i, j]
                h[width + i, j] = y.hess[i, j]
                h[2 * width + i, j] = z.hess[i, j]
                h[3 * width + i, j] = w.hess[i, j]
        return Jet2Quat(NativeQuat(x.value, y.value, z.value, w.value), g, h)

    # ---- construction / seeding ----

    @wp.func
    def vec3_constant(v: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(v, GradVec3(), HessVec3())

    @wp.func
    def vec3_seed(v: NativeVec3, i0: int, i1: int, i2: int) -> Jet2Vec3:
        g = GradVec3()
        g[0, i0] = dtype(1.0)
        g[1, i1] = dtype(1.0)
        g[2, i2] = dtype(1.0)
        return Jet2Vec3(v, g, HessVec3())

    @wp.func
    def quat_constant(q: NativeQuat) -> Jet2Quat:
        return Jet2Quat(q, GradQuat(), HessQuat())

    @wp.func
    def quat_seed(q: NativeQuat, i0: int, i1: int, i2: int, i3: int) -> Jet2Quat:
        g = GradQuat()
        g[0, i0] = dtype(1.0)
        g[1, i1] = dtype(1.0)
        g[2, i2] = dtype(1.0)
        g[3, i3] = dtype(1.0)
        return Jet2Quat(q, g, HessQuat())

    # ---- vec3 linear algebra ----

    @wp.func
    def jet2_add(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.value + b.value, a.grad + b.grad, a.hess + b.hess)

    @wp.func
    def jet2_add(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.value + b, a.grad, a.hess)

    @wp.func
    def jet2_add(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a + b.value, b.grad, b.hess)

    @wp.func
    def jet2_sub(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.value - b.value, a.grad - b.grad, a.hess - b.hess)

    @wp.func
    def jet2_sub(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.value - b, a.grad, a.hess)

    @wp.func
    def jet2_sub(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a - b.value, -b.grad, -b.hess)

    @wp.func
    def jet2_dot(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Scalar:
        return jet_add(
            jet_add(
                jet_mul(vec3_extract(a, 0), vec3_extract(b, 0)),
                jet_mul(vec3_extract(a, 1), vec3_extract(b, 1)),
            ),
            jet_mul(vec3_extract(a, 2), vec3_extract(b, 2)),
        )

    @wp.func
    def jet2_length_sq(a: Jet2Vec3) -> Jet2Scalar:
        return jet2_dot(a, a)

    # ---- quaternion products (Hamilton, [x, y, z, w] storage) ----

    @wp.func
    def _hamilton(
        ax: Jet2Scalar,
        ay: Jet2Scalar,
        az: Jet2Scalar,
        aw: Jet2Scalar,
        bx: Jet2Scalar,
        by: Jet2Scalar,
        bz: Jet2Scalar,
        bw: Jet2Scalar,
    ) -> Jet2Quat:
        # a * b, matching native quat.h mul(): the vector part is
        # aw*bv + bw*av + av x bv and the scalar part is aw*bw - av.bv.
        rx = jet_sub(jet_add(jet_add(jet_mul(aw, bx), jet_mul(bw, ax)), jet_mul(ay, bz)), jet_mul(az, by))
        ry = jet_sub(jet_add(jet_add(jet_mul(aw, by), jet_mul(bw, ay)), jet_mul(az, bx)), jet_mul(ax, bz))
        rz = jet_sub(jet_add(jet_add(jet_mul(aw, bz), jet_mul(bw, az)), jet_mul(ax, by)), jet_mul(ay, bx))
        rw = jet_sub(jet_sub(jet_sub(jet_mul(aw, bw), jet_mul(ax, bx)), jet_mul(ay, by)), jet_mul(az, bz))
        return quat_from_scalars(rx, ry, rz, rw)

    @wp.func
    def jet2_mul(a: Jet2Quat, b: Jet2Quat) -> Jet2Quat:
        return _hamilton(
            quat_extract(a, 0),
            quat_extract(a, 1),
            quat_extract(a, 2),
            quat_extract(a, 3),
            quat_extract(b, 0),
            quat_extract(b, 1),
            quat_extract(b, 2),
            quat_extract(b, 3),
        )

    @wp.func
    def jet2_mul(a: Jet2Quat, b: NativeQuat) -> Jet2Quat:
        return _hamilton(
            quat_extract(a, 0),
            quat_extract(a, 1),
            quat_extract(a, 2),
            quat_extract(a, 3),
            scalar_constant(b[0]),
            scalar_constant(b[1]),
            scalar_constant(b[2]),
            scalar_constant(b[3]),
        )

    @wp.func
    def jet2_mul(a: NativeQuat, b: Jet2Quat) -> Jet2Quat:
        return _hamilton(
            scalar_constant(a[0]),
            scalar_constant(a[1]),
            scalar_constant(a[2]),
            scalar_constant(a[3]),
            quat_extract(b, 0),
            quat_extract(b, 1),
            quat_extract(b, 2),
            quat_extract(b, 3),
        )

    # ---- rotate a constant point by a quaternion jet ----

    @wp.func
    def jet2_quat_rotate(q: Jet2Quat, x: NativeVec3) -> Jet2Vec3:
        # Same expansion as native quat_rotate(); presumes a unit q, which the
        # exp_map -> mul(unit) chain maintains to second order.
        two = dtype(2.0)
        qx = quat_extract(q, 0)
        qy = quat_extract(q, 1)
        qz = quat_extract(q, 2)
        qw = quat_extract(q, 3)
        xx = x[0]
        xy = x[1]
        xz = x[2]

        c = jet_sub(jet_mul(two, jet_mul(qw, qw)), dtype(1.0))
        d = jet_mul(two, jet_add(jet_add(jet_mul(qx, xx), jet_mul(qy, xy)), jet_mul(qz, xz)))

        rx = jet_add(
            jet_add(jet_mul(c, xx), jet_mul(qx, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qy, xz), jet_mul(qz, xy)), qw), two),
        )
        ry = jet_add(
            jet_add(jet_mul(c, xy), jet_mul(qy, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qz, xx), jet_mul(qx, xz)), qw), two),
        )
        rz = jet_add(
            jet_add(jet_mul(c, xz), jet_mul(qz, d)),
            jet_mul(jet_mul(jet_sub(jet_mul(qx, xy), jet_mul(qy, xx)), qw), two),
        )
        return vec3_from_scalars(rx, ry, rz)

    # ---- rotation-vector exp map: R^3 tangent -> unit quaternion ----

    @wp.func
    def quat_exp_map(v: Jet2Vec3) -> Jet2Quat:
        # q(v) = [ v * sinc_half(s), cos_half(s) ], with s = |v|^2, where
        # cos_half(s) = cos(sqrt(s)/2) and sinc_half(s) = sin(sqrt(s)/2)/sqrt(s).
        # Both are smooth (even) functions of v: the |v| kink never appears.
        #
        # Near s = 0 the closed form divides by sqrt(s), so a truncated series
        # in s -- itself differentiated exactly by the scalar chain rule -- is
        # used; away from 0 the exact closed form runs. The series matches the
        # true value, first, and second s-derivatives at 0, which is all the
        # second-order jet reads.
        s = jet2_length_sq(v)
        s2 = jet_mul(s, s)
        s3 = jet_mul(s2, s)

        cw_series = jet_add(
            jet_add(
                jet_add(scalar_constant(dtype(1.0)), jet_mul(s, dtype(-1.0 / 8.0))),
                jet_mul(s2, dtype(1.0 / 384.0)),
            ),
            jet_mul(s3, dtype(-1.0 / 46080.0)),
        )
        g_series = jet_add(
            jet_add(
                jet_add(scalar_constant(dtype(0.5)), jet_mul(s, dtype(-1.0 / 48.0))),
                jet_mul(s2, dtype(1.0 / 3840.0)),
            ),
            jet_mul(s3, dtype(-1.0 / 645120.0)),
        )

        a = jet_sqrt(s)
        h = jet_mul(a, dtype(0.5))
        cw_closed = jet_cos(h)
        g_closed = jet_div(jet_sin(h), a)

        small = s.value < dtype(1.0e-3)
        cw = jet_where(small, cw_series, cw_closed)
        g = jet_where(small, g_series, g_closed)

        return quat_from_scalars(
            jet_mul(vec3_extract(v, 0), g),
            jet_mul(vec3_extract(v, 1), g),
            jet_mul(vec3_extract(v, 2), g),
            cw,
        )

    _register("add", jet_add)
    _register("sub", jet_sub)
    _register("mul", jet_mul)
    _register("div", jet_div)
    _register("pow", jet_pow)
    _register("neg", jet_neg)
    _register("pos", jet_pos)

    _register("add", jet2_add)
    _register("sub", jet2_sub)
    _register("mul", jet2_mul)
    _register("dot", jet2_dot)
    _register("quat_rotate", jet2_quat_rotate)
    _register("extract", vec3_extract)
    _register("extract", quat_extract)

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
        vec3=Jet2Vec3,
        quat=Jet2Quat,
        grad=Grad,
        hess=Hess,
        native_vec3=NativeVec3,
        native_quat=NativeQuat,
        constant=scalar_constant,
        seed=scalar_seed,
        constant_vec3=vec3_constant,
        seed_vec3=vec3_seed,
        make_vec3=vec3_from_scalars,
        constant_quat=quat_constant,
        seed_quat=quat_seed,
        make_quat=quat_from_scalars,
        exp_map=quat_exp_map,
        quat_from_rotvec=quat_exp_map,
    )


def JetSpace2(width: int, dtype=wp.float32):
    """Return second-order (forward-over-forward) jet types for a given width.

    A width-k second-order jet tracks value, gradient, and the full k x k
    Hessian; propagating it through a scalar function in one forward pass yields
    the local Hessian with no reverse pass. Per-intermediate state is O(k^2), so
    this suits small k. The namespace provides ``scalar`` payloads, plus ``vec3``
    and ``quat`` payloads with an ``exp_map`` rotation-vector chart for optimizing
    over 3D rotations (the tangent Hessian of a quaternion energy in one pass).

    Args:
        width: Number of variables differentiated with respect to. Fixed when
            the types are specialized, not when a kernel runs.
        dtype: Scalar type the jets are built on.

    As with :func:`warp.JetSpace`, the width is a compile-time constant and the
    types are for local derivatives of fixed, statically known arity. The
    quadratic state makes that limit tighter here: register pressure and compile
    time both grow with ``width**2``, so for wider energies prefer a first-order
    jet with a reverse sweep over it (via :class:`warp.Tape`, or in-kernel with
    :func:`warp.grad`), which reaches the same Hessian with linear state.
    """
    width = int(width)
    key = (width, dtype)

    J = _CACHE2.get(key)
    if J is None:
        J = _make_jet_space2(width, dtype)
        _CACHE2[key] = J

    return J
