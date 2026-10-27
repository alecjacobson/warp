.. _forward_mode_jets:

Forward-Mode Jets
=================

.. currentmodule:: warp

Warp's :class:`wp.Tape <Tape>` and :func:`wp.grad <grad>` differentiate in reverse mode: they replay a computation
backwards and deliver the gradient of one output with respect to every input in a single sweep (see
:ref:`differentiability`). That is the right tool for a loss over a large state, but it has limitations. For one,
reverse mode in Warp currently cannot be applied to its own output, so on its own it is limited to first derivatives.
For another, reverse mode needs a backward sweep for each output it differentiates, so it scales poorly for Jacobian
matrices of many-output functions and for Jacobian-vector products.

Forward-mode *jets* fill these gaps. A jet carries a value together with its derivatives along a fixed number of
directions. A :func:`@wp.func <func>` written generically over its argument types can be evaluated over jets instead
of floats, returning its derivatives along with its value, with no hand-written gradient or Hessian:

.. math::

    x = \text{value} + \sum_{q=1}^{w} \text{coeff}_q \, \varepsilon_q

where :math:`w` is the jet's *width*. :func:`wp.JetSpace <JetSpace>` creates first-order jets, and
:func:`wp.JetSpace2 <JetSpace2>` creates second-order scalar jets that also carry a dense Hessian.

Jets do not replace reverse mode but complement it, with different trade-offs in memory access, operation count and
compile time. See :ref:`jets_when_to_use` below.

A First Example
---------------

Consider a two-link arm in 2D with four degrees of freedom: the position of its base ``base_pos``, the relative
shoulder angle ``shoulder`` and the relative elbow angle ``elbow``. A jet space for these inputs tracks derivatives
along each of them, with a width equal to their four scalar components. For inverse kinematics, the goal is to bring the
end of the arm to a target position ``target_pos``. The residual of that least-squares problem is an ordinary function,
written once and generic over its argument types, so it accepts both plain values and jets. A kernel or another
:func:`@wp.func <func>` can then seed the inputs, call ``residual``, and read off the coefficients to construct the
residual's Jacobian:

.. testcode::

    J = wp.JetSpace((wp.vec2, wp.float32, wp.float32))  # width 4

    @wp.func
    def residual(base_pos: Any, shoulder: Any, elbow: Any, target_pos: wp.vec2):
        # Two links of unit length; the elbow angle is relative to the upper arm.
        elbow_pos = base_pos + wp.cos(shoulder) * wp.vec2(1.0, 0.0) + wp.sin(shoulder) * wp.vec2(0.0, 1.0)
        forearm = shoulder + elbow
        end_pos = elbow_pos + wp.cos(forearm) * wp.vec2(1.0, 0.0) + wp.sin(forearm) * wp.vec2(0.0, 1.0)
        return end_pos - target_pos

    @wp.kernel
    def residual_jacobian_kernel(
        base_pos: wp.array[wp.vec2],
        shoulder: wp.array[float],
        elbow: wp.array[float],
        target_pos: wp.array[wp.vec2],
        jacobian: wp.array[J.coeff_mat2],
    ):
        tid = wp.tid()
        jet_base_pos, jet_shoulder, jet_elbow = J.seed(base_pos[tid], shoulder[tid], elbow[tid])
        jet_residual = residual(jet_base_pos, jet_shoulder, jet_elbow, target_pos[tid])
        jacobian[tid] = jet_residual.coeff  # 2x4: d(residual) / d(base_pos, shoulder, elbow)

    base_pos = wp.array([[0.0, 0.0]], dtype=wp.vec2)
    shoulder = wp.array([np.radians(45.0)], dtype=float)
    elbow = wp.array([np.radians(-45.0)], dtype=float)
    target_pos = wp.array([[-0.5, 0.5]], dtype=wp.vec2)
    jacobian = wp.zeros(1, dtype=J.coeff_mat2)

    wp.launch(residual_jacobian_kernel, dim=1, inputs=[base_pos, shoulder, elbow, target_pos], outputs=[jacobian])

    print(np.round(jacobian.numpy()[0], 4))

.. testoutput::

    [[ 1.      0.     -0.7071  0.    ]
     [ 0.      1.      1.7071  1.    ]]

Nothing is bound into the module beyond the space ``J``. Operators and builtins such as ``end_pos - target_pos`` and
``wp.sin()`` resolve on jets because the jet arithmetic is registered as overloads of Warp's own builtins.

Slots and Seeding
-----------------

The default use case for jets is constructing the Jacobian of a function with respect to a predetermined set of
inputs. The :func:`wp.JetSpace <JetSpace>` constructor takes a tuple of types, called *slots*, such as
``(wp.vec2, wp.float32, wp.float32)`` in the example above. Each slot corresponds to one of the function's inputs. By
default, the width of the jets is the total number of scalar components (``width=4`` in the example above), so the jets
carry the full Jacobian with respect to these inputs.

The slot types are scalars, ``vec2``, ``vec3``, and the ``mat22``, ``mat33``, ``mat32`` and ``mat23`` matrices.
The jets for each are available on the returned namespace as ``J.scalar``, ``J.vec2``, ``J.vec3``, ``J.mat2``,
``J.mat3``, ``J.mat32`` and ``J.mat23``, and ``J.coeff`` is the type of a scalar jet's derivative vector.

Inside a kernel, ``J.seed()`` takes one value per slot and returns one jet per slot. It copies each input's value into
the jet's ``value`` field and sets the ``coeff`` field to the unit vector along the corresponding direction of the
derivative space, ordered by slot and then by component within the slot (row-major for matrices). In effect, the jets
are seeded with the identity matrix, so the coefficients of any downstream output contain its Jacobian.

Width and Directional
---------------------

Jets can also compute directional derivatives, also known as Jacobian-vector products, without first constructing the
full Jacobian. To do this, construct :func:`wp.JetSpace <JetSpace>` with a ``width`` equal to the number of directions
to differentiate along, which does not need to match the total number of slot components. When the width differs from
the number of slot components, ``J.seed()`` is not available, and the derivative directions are provided explicitly
instead.

``J.directional()`` takes the same values as ``J.seed()``, followed by ``width * len(slots)`` direction values, ordered
direction-major: for each direction in turn, one value per slot, of the same type as that slot. The returned jets
carry these directions in their ``coeff`` fields instead of the identity matrix, so the coefficients of any downstream
output contain its Jacobian times the direction vectors.

In the example above, suppose we want the derivative of the residual as the shoulder rotates one way and the elbow
the other way at the same rate, so the forearm keeps its orientation. A space with scalar slots for ``shoulder`` and
``elbow`` and ``width=1`` (instead of the default ``2``) does this, given the direction ``(1, -1)``. The result is the
Jacobian times that direction, the shoulder column printed above minus the elbow column. Because the forearm only
translates, the end of the arm moves exactly as the elbow does:

.. testcode::

    Jd = wp.JetSpace((wp.float32, wp.float32), width=1)

    @wp.kernel
    def directional_derivative_kernel(
        base_pos: wp.array[wp.vec2],
        shoulder: wp.array[float],
        elbow: wp.array[float],
        target_pos: wp.array[wp.vec2],
        directional_derivative: wp.array[Jd.coeff_mat2],
    ):
        tid = wp.tid()
        # Rotate the shoulder and the elbow in opposite directions at the same rate.
        jet_shoulder, jet_elbow = Jd.directional(shoulder[tid], elbow[tid], 1.0, -1.0)
        jet_residual = residual(base_pos[tid], jet_shoulder, jet_elbow, target_pos[tid])
        directional_derivative[tid] = jet_residual.coeff

    directional_derivative = wp.zeros(1, dtype=Jd.coeff_mat2)

    wp.launch(
        directional_derivative_kernel,
        dim=1,
        inputs=[base_pos, shoulder, elbow, target_pos],
        outputs=[directional_derivative],
    )

    print(np.round(directional_derivative.numpy()[0], 4))

.. testoutput::

    [[-0.7071]
     [ 0.7071]]


Component-wise seeding
^^^^^^^^^^^^^^^^^^^^^^

Callers may also access the low-level interface of jets directly, without the
slot types. Passing a number of scalars instead of slot types, as in ``wp.JetSpace(4)``,
creates a space whose width defaults to that number, or is the ``width``
provided. Each scalar variable is then seeded manually with ``J.variable(x,
direction)``, which returns a scalar jet with value ``x`` and a unit derivative
along ``direction``. 

.. testcode::

    Jn = wp.JetSpace(4)  # four scalars, width 4

    @wp.kernel
    def sum_of_cubes_gradient_kernel(x: wp.array2d[float], gradient: wp.array[Jn.coeff]):
        tid = wp.tid()
        total = Jn.constant(0.0)
        for k in range(4):
            xk = Jn.variable(x[tid, k], k)
            total = total + xk * xk * xk
        gradient[tid] = total.coeff

    x = wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    gradient = wp.zeros(1, dtype=Jn.coeff)

    wp.launch(sum_of_cubes_gradient_kernel, dim=1, inputs=[x], outputs=[gradient])

    print(gradient.numpy()[0])

.. testoutput::

    [ 3. 12. 27. 48.]

Hessians
--------

To compute second derivatives, we can either combine warp's reverse mode
(:class:`wp.Tape <Tape>` or :func:`wp.grad <grad>`) with
first-order jets (:func:`wp.JetSpace <JetSpace>`), or use second-order jets
(:func:`wp.JetSpace2 <JetSpace2>`).

:func:`wp.JetSpace2 <JetSpace2>` is constructed and seeded similarly to
:func:`wp.JetSpace <JetSpace>`. 

.. testcode::
   
    J2 = wp.JetSpace2((wp.vec3, wp.vec3))  # width 6

    @wp.func
    def spring_energy(a: Any, b: Any):
        d = a - b
        return 0.5 * (wp.length(d) - 1.0) ** 2

    @wp.kernel
    def hessian_kernel(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        val: wp.array2[float],
        grad: wp.array2[wp.vec6],
        hess: wp.array[wp.mat6]
    ):
        i = wp.tid()
        jet_x0, jet_x1 = J2.seed(x0[i], x1[i])
        f = spring_energy(jet_x0, jet_x1)  
        val[i] = f.value
        grad[i] = f.grad
        hess[i] = f.hess

.. testoutput::


---------- UP TO HERE -------------- 


There are three ways to a local Hessian, and which one is best depends on the energy:

1. **Reverse mode over a first-order jet with** :class:`wp.Tape <Tape>`. A width-``k`` jet gives the gradient in one
   forward launch, then ``k`` backward sweeps give the rows of the Hessian. It reuses Warp's tape and needs a global
   array with ``requires_grad=True``.
2. **Reverse mode over a first-order jet inside the kernel with** :func:`wp.grad <grad>`. A width-1 jet gives the
   directional derivative :math:`\nabla E \cdot v`, and the in-kernel adjoint sweep of that scalar gives the
   Hessian-vector product :math:`H v`. It runs in registers in a single launch, with no tape and no global ``.grad``
   array.
3. **A second-order jet,** :func:`wp.JetSpace2 <JetSpace2>`. One forward pass returns the value, gradient and the dense
   Hessian, with no reverse mode at all.

Route 2, with ``J1`` the width-1 space from above:

.. testcode::

    J1 = wp.JetSpace((wp.vec3, wp.vec3), width=1)

    @wp.func
    def directional_derivative(x0: wp.vec3, x1: wp.vec3, v0: wp.vec3, v1: wp.vec3):
        # First-order jet along the direction (v0, v1): the scalar dE . v.
        a, b = J1.directional(x0, x1, v0, v1)
        return spring_energy(a, b).coeff[0]

    @wp.kernel
    def hessian_vector_product(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        v0: wp.array[wp.vec3],
        v1: wp.array[wp.vec3],
        hv: wp.array2d[wp.vec3],
    ):
        i = wp.tid()

        # The gradient of (dE . v) with respect to the inputs is H v.
        hv0, hv1, _a, _b = wp.grad(directional_derivative)(x0[i], x1[i], v0[i], v1[i])
        hv[i, 0] = hv0
        hv[i, 1] = hv1

    v0 = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    v1 = wp.array([[1.0, 1.0, 0.0]], dtype=wp.vec3)
    hv = wp.zeros((1, 2), dtype=wp.vec3)

    wp.launch(hessian_vector_product, dim=1, inputs=[x0, x1, v0, v1], outputs=[hv])

    print("H v:", hv.numpy().reshape(1, 6))

.. testoutput::

    H v: [[-1.  -0.5  0.   1.   0.5  0. ]]

Looping this over the six coordinate directions gives the full :math:`6 \times 6` block.

Route 3 on a two-variable energy. Second-order spaces take scalar slots, so the energy is written over scalars:

.. testcode::

    J2 = wp.JetSpace2(2)

    @wp.func
    def energy(a: Any, b: Any):
        return a * a * b + wp.sin(b)

    @wp.kernel
    def second_order(out: wp.array2d[float]):
        a, b = J2.seed(1.0, 0.5)
        e = energy(a, b)  # e.value, e.grad (length 2) and e.hess (2 x 2)
        for p in range(2):
            for q in range(2):
                out[p, q] = e.hess[p, q]

    out = wp.zeros((2, 2), dtype=float)
    wp.launch(second_order, dim=1, inputs=[], outputs=[out])

    print(out.numpy())

.. testoutput::

    [[ 1.          2.        ]
     [ 2.         -0.47942555]]

Reverse mode through jets is what makes routes 1 and 2 possible: the jet arithmetic is ordinary Warp code, so
Warp's autodiff differentiates it like any other function. That includes the seeding helpers.

Forward and Reverse Mode Side by Side
-------------------------------------

Both modes compute the same derivatives. They differ in what they cost and in what they can express.

.. list-table::
    :header-rows: 1
    :widths: 24 38 38

    * -
      - Reverse mode (:class:`Tape`, :func:`grad`)
      - Forward-mode jets (:func:`JetSpace`, :func:`JetSpace2`)
    * - Propagates
      - A sensitivity of one output backwards, from the output to every input.
      - Derivatives along ``width`` directions forwards, alongside the value.
    * - One evaluation gives
      - The gradient of a scalar output with respect to *all* inputs.
      - The derivatives of *every* intermediate along ``width`` directions.
    * - Cost scales with
      - The number of outputs (one sweep per output row).
      - The number of directions, ``width``, carried through every intermediate.
    * - Second derivatives
      - Not directly. Reverse mode is applied to a function that already contains a derivative.
      - A second-order jet carries the Hessian; first-order jets also give :math:`H v` through reverse mode.
    * - Number of inputs
      - Unbounded and set at run time, such as every degree of freedom of a simulation.
      - Fixed when the types are created, because ``width`` is a compile-time constant.
    * - Needs
      - A recorded tape, or an in-kernel ``wp.grad`` call over a ``wp.func``.
      - Nothing recorded. The function is called on jet values like any other.
    * - Best for
      - A loss over a large state, in training or optimization.
      - A dense *local* gradient or Hessian of one element, scattered into a global system.

In code, the two read as follows for the same energy:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Reverse mode
      - Forward-mode jets
    * - .. code-block:: python

            @wp.kernel
            def gradient(x0: wp.array[wp.vec3], x1: wp.array[wp.vec3], g: wp.array2d[wp.vec3]):
                i = wp.tid()
                g0, g1 = wp.grad(spring_energy)(x0[i], x1[i])
                g[i, 0] = g0
                g[i, 1] = g1
      - .. code-block:: python

            @wp.kernel
            def gradient(x0: wp.array[wp.vec3], x1: wp.array[wp.vec3], g: wp.array[J.coeff]):
                i = wp.tid()
                a, b = J.seed(x0[i], x1[i])
                g[i] = spring_energy(a, b).coeff

.. _jets_when_to_use:

When to Use Jets
----------------

``width`` is a compile-time constant, not a runtime length. It becomes the length of ``coeff`` and is unrolled into the
generated code, and every intermediate value carries all ``width`` components whether or not they are read back. A
second-order jet carries :math:`w^2` Hessian entries. Register pressure and compile time grow with the width.

That makes jets a good fit for a *local* derivative of statically known arity. A spring over two ``vec3`` nodes has
width 6, a triangle width 9 and a tetrahedron width 12, and the dense local block is what a Newton solver scatters into a
sparse global matrix. A good test is whether the width can be written as a literal next to the function that uses it.

Jets also win when a function has *many outputs* of a few inputs, i.e. a tall Jacobian :math:`r : \mathbb{R}^k \to
\mathbb{R}^m` with :math:`m \gg k`. Reverse mode needs one backward sweep per output, while one width-``k`` jet carries all
``k`` directions through a single forward pass, and any work shared by the outputs runs once (see
:ref:`the Jacobian benchmark below <jets_jacobian_performance>`).

They are the wrong tool for differentiating with respect to a problem-sized or runtime number of variables. A width of
:math:`n` cannot specialize a type, and even where it could, forward mode would cost :math:`n` directions against reverse
mode's single backward pass. Use :class:`Tape` or :func:`grad` for the outer derivative, and jets for the fixed-arity
term inside it.

Performance
-----------

The numbers below come from the benchmarks in ``warp/examples/benchmarks``, run on an NVIDIA RTX PRO 5000 Blackwell
(laptop) GPU in float32 with about one million elements per run. Each time is the best of 20 launches for the gradient
and of 10 for the Hessian, with launch overhead included and no CUDA graph. The tetrahedron Hessian row uses 200,000 elements. Timings on this
laptop GPU vary by tens of percent between runs, so read the table for ratios and orders of magnitude. Every strategy is first checked against the others and against finite
differences. They show the *relative* cost of the strategies, not absolute figures to expect on other hardware or for
other energies.

**Gradient** (``benchmark_jet_gradient_mesh.py``). Evaluating the energy as a width-``k`` jet is about as fast as
reverse mode, and ``wp.grad`` in the same kernel is about as fast again. First-order jets are not a speed-up over reverse
mode here; their value is that the same code also yields second derivatives.

.. list-table::
    :header-rows: 1

    * - Element
      - ``k``
      - Forward-mode jet (ms)
      - Reverse mode, ``Tape`` (ms)
      - ``wp.grad`` in kernel (ms)
    * - Spring
      - 6
      - 0.051
      - 0.066
      - 0.051
    * - Triangle
      - 9
      - 0.074
      - 0.089
      - 0.072
    * - Tetrahedron
      - 12
      - 0.122
      - 0.134
      - 0.119

**Hessian** (``benchmark_jet_hessian_mesh.py``). The in-kernel ``wp.grad`` route and the second-order jet are both
roughly an order of magnitude faster than running a tape, because the per-element Hessian stays in registers and needs a
single launch.

.. list-table::
    :header-rows: 1

    * - Element
      - ``k``
      - ``k`` tape sweeps over a width-``k`` jet (ms)
      - ``k`` tape sweeps over a width-1 jet (ms)
      - In-kernel ``wp.grad`` over a width-1 jet (ms)
      - Second-order jet (ms)
    * - Spring
      - 6
      - 4.87
      - 4.79
      - 0.584
      - 0.561
    * - Triangle
      - 9
      - 72.6
      - 10.8
      - 3.2
      - 2.0
    * - Tetrahedron
      - 12
      - 61.9
      - 2.58
      - 1.00
      - 0.51

The second-order jet is fastest at run time for these small energies, but it pays for that at compile time, because its
state is :math:`O(k^2)` and NVCC compile time is super-linear in ``k``. For the short elasticity energies above a
tetrahedron compiles in seconds, but a longer scalar-form energy at ``k = 12`` can take minutes on a cold cache
(``benchmark_element_hessian.py`` measures this). When compile time matters, in-kernel ``wp.grad`` over a first-order jet
gives the same Hessian with state that grows linearly in ``k``.

.. _jets_jacobian_performance:

**Jacobian of a many-output function** (``benchmark_jet_jacobian.py``). Take a forward model with ``k = 3`` parameters
that integrates a damped pendulum for 16 steps and then reads out ``m`` values of the final state, at 200,000 points. The
integration is shared by all ``m`` readouts. The jet runs it once on a width-3 jet and evaluates every readout on that
jet. The in-kernel ``wp.grad`` route calls ``wp.grad(residual)(x, j)`` once per output, which re-runs the integration for
every row. The tape route is the per-row ``tape.backward`` loop described in :doc:`differentiability`, timed
on its backward sweeps alone.

.. list-table::
    :header-rows: 1

    * - Outputs ``m``
      - Width-3 jet (ms)
      - ``wp.grad`` in kernel (ms)
      - ``m`` tape sweeps (ms)
    * - 4
      - 0.025
      - 0.074
      - 0.41
    * - 16
      - 0.037
      - 0.284
      - 6.1
    * - 32
      - 0.152
      - 0.575
      - 28.7
    * - 64
      - 0.211
      - 1.15
      - not run
    * - 128
      - 0.412
      - 2.28
      - not run

The jet is 3 to 8 times faster than in-kernel ``wp.grad`` for ``m >= 4`` and hundreds of times faster than the tape,
which launches ``m`` backward passes and rewrites an :math:`n \times m` seed each time. The tape was not run beyond
``m = 32`` because its seeds need memory proportional to :math:`n m^2`. Small ``m`` times are close to the launch
floor. The advantage over ``wp.grad`` comes from the shared integration and shrinks as the readouts become the larger
share of the work, because each readout costs ``k + 1`` times more on a jet. It also shrinks as ``k`` grows, since the
jet's cost scales with ``k`` and reverse mode's does not.

Store the Jacobian output-major, ``jac[j * n + i]``, so that neighbouring threads write neighbouring elements. Storing it
point-major, ``jac[i * m + j]``, made the jet about five times slower at ``m = 64`` (1.32 ms against 0.25 ms) and left
the ``wp.grad`` route nearly unchanged, because the jet is fast enough to be bound by those strided stores.

Limitations
-----------

* The width is fixed when the types are created. Spaces of different widths and element types may coexist in a process.
* Creating a space registers its arithmetic on Warp's builtins for the lifetime of the process. The effect is
  additive and happens once per width and scalar type.
* Branching builtins (``min``, ``max``, ``clamp``, ``where``, ``abs``, ``sign``) carry the derivative of the branch they
  take. At a switch point that is a one-sided derivative, and jets do not smooth the kink. Comparisons such as ``a < b``
  do not overload on jets, so branch on ``.value``.
* Only builtins with jet overloads accept jets. Others, such as spatial queries like
  :func:`wp.mesh_query_point() <warp._src.lang.mesh_query_point>`, operate on floats only and need manual integration.
* Second-order jets carry a scalar payload only. Vector and matrix second derivatives decompose into per-component
  scalar jets.
* Jet dtypes must be floating point. Quaternion slots are not supported.
* A plain scalar, vector or matrix of the jet's dtype can be either operand of a jet operation, and is treated as a
  constant. The exception is ``wp.where``, whose two branches must both be jets; wrap a plain branch with
  ``J.constant``.
