.. _forward_mode_jets:

Forward-Mode Jets
=================

.. currentmodule:: warp

Warp differentiates in reverse mode: :class:`wp.Tape <Tape>` and :func:`wp.grad <grad>` record or replay a computation
backwards and deliver the gradient of one output with respect to every input in a single sweep (see
:ref:`differentiability`). That is the right tool for a loss over a large state, but it has no second-order counterpart.
Simulation code often needs a *local Hessian*: the second derivative of a small energy, such as a spring, a triangle or a
tetrahedron, with respect to the handful of values it reads.

Forward-mode *jets* fill that gap. A jet carries a value together with its derivatives along a fixed number of
directions. Evaluating an ordinary :func:`@wp.func <func>` over jets, instead of over floats, returns its derivatives as
a side effect, with no hand-written gradient or Hessian:

.. math::

    x = \text{value} + \sum_{q=1}^{w} \text{coeff}_q \, \varepsilon_q

where :math:`w` is the jet's *width*. :func:`wp.JetSpace <JetSpace>` creates first-order jets, and
:func:`wp.JetSpace2 <JetSpace2>` creates second-order scalar jets that also carry a dense Hessian.

Jets do not replace reverse mode. They handle the fixed-arity term inside a sum; reverse mode handles the sum. See
:ref:`jets_when_to_use` below.

A First Example
---------------

The energy of a spring of rest length one between two 3D nodes is an ordinary function. It is written once, generic
over its argument types, and used both on plain ``vec3`` values and on jets:

.. testcode::

    J = wp.JetSpace((wp.vec3, wp.vec3))  # a spring: two 3D nodes, width 6

    @wp.func
    def spring_energy(x0: Any, x1: Any):
        r = wp.length(x1 - x0) - 1.0
        return 0.5 * r * r

    @wp.kernel
    def gradients(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        forward: wp.array[J.coeff],
        reverse: wp.array2d[wp.vec3],
    ):
        i = wp.tid()

        # Forward mode: seed the six inputs and read the gradient off the jet.
        a, b = J.seed(x0[i], x1[i])
        forward[i] = spring_energy(a, b).coeff

        # Reverse mode: one backward pass through the same function.
        g0, g1 = wp.grad(spring_energy)(x0[i], x1[i])
        reverse[i, 0] = g0
        reverse[i, 1] = g1

    x0 = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    x1 = wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3)
    forward = wp.zeros(1, dtype=J.coeff)
    reverse = wp.zeros((1, 2), dtype=wp.vec3)

    wp.launch(gradients, dim=1, inputs=[x0, x1], outputs=[forward, reverse])

    print("forward:", forward.numpy())
    print("reverse:", reverse.numpy().reshape(1, 6))

.. testoutput::

    forward: [[-1.  0.  0.  1.  0.  0.]]
    reverse: [[-1.  0.  0.  1.  0.  0.]]

Nothing is bound into the module beyond the space ``J``. Operators and builtins such as ``x1 - x0`` and
``wp.length()`` resolve on jets because the jet arithmetic is registered as overloads of Warp's own
builtins; component access (``v[0]``) works the same way.

Slots and Seeding
-----------------

A jet's derivative directions are numbered, so a jet space has to say which input component each direction belongs to.
It does that by the *types* of the inputs being differentiated, called *slots*:

.. list-table::
    :header-rows: 1
    :widths: 35 10 55

    * - ``wp.JetSpace(slots)``
      - Width
      - Meaning
    * - ``(wp.vec3, wp.vec3)``
      - 6
      - Two 3D points, for example a spring.
    * - ``(wp.vec3,) * 4``
      - 12
      - Four 3D points, for example a tetrahedron.
    * - ``(wp.vec2, wp.float32, wp.mat22)``
      - 7
      - A 2D point, a scalar and a 2x2 matrix.
    * - ``4``
      - 4
      - Four scalars. An integer ``n`` means ``n`` scalar slots.

The slot types are scalars, ``vec2``, ``vec3``, and the ``mat22``, ``mat33``, ``mat32`` and ``mat23`` matrices.
The jets for each are available on the returned namespace as ``J.scalar``, ``J.vec2``, ``J.vec3``, ``J.mat2``,
``J.mat3``, ``J.mat32`` and ``J.mat23``, and ``J.coeff`` is the type of a scalar jet's derivative vector.

Inside a kernel, ``J.seed()`` takes one value per slot and returns one jet per slot. It numbers the directions slot by
slot, in the order the components appear (row-major within a matrix):

.. code-block:: python

    a, b = J.seed(x0, x1)  # a, b are J.vec3; a carries directions 0-2, b directions 3-5

Because the arguments are checked against the declared slots by ordinary overload resolution, passing a ``vec2`` to a
``vec3`` slot fails to compile instead of numbering the wrong directions. A single-slot space takes and returns the bare
value, as in ``p = J.seed(x)``.

Directional derivatives
^^^^^^^^^^^^^^^^^^^^^^^

``seed()`` uses the coordinate directions, which gives the full gradient. A derivative along a *chosen* direction needs
only one jet component. Passing ``width=1`` creates a space with a single direction, and ``J.directional()`` takes the
values followed by the direction to differentiate along:

.. code-block:: python

    J1 = wp.JetSpace((wp.vec3, wp.vec3), width=1)

    a, b = J1.directional(x0, x1, v0, v1)  # derivative along the combined direction (v0, v1)

With ``width > 1``, ``directional()`` takes ``width * len(slots)`` directions, ordered direction-major: direction ``q`` of
slot ``s`` is argument ``q * len(slots) + s`` after the values. ``seed()`` exists only when ``width`` equals the number of
components; otherwise use ``directional()``.

.. note::

    Operands are positional arguments rather than tuples because Warp does not propagate adjoints into a tuple
    argument. The call would compile and reverse mode would silently return zeros, which defeats the
    reverse-over-forward Hessians below.

Seeding inside a loop
^^^^^^^^^^^^^^^^^^^^^

When the number of scalars is not a fixed argument list, ``J.variable(x, direction)`` returns a scalar jet with a unit
derivative along one direction.

Hessians
--------

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

Limitations
-----------

* The width is fixed when the types are created. Spaces of different widths and element types may coexist in a process.
* Creating a space registers its arithmetic on Warp's builtins for the lifetime of the process. The effect is
  additive and happens once per width and scalar type.
* Branching builtins (``min``, ``max``, ``clamp``, ``where``, ``abs``, ``sign``) carry the derivative of the branch they
  take. At a switch point that is a one-sided derivative, and jets do not smooth the kink. Comparisons such as ``a < b``
  do not overload on jets, so branch on ``.value``.
* Second-order jets carry a scalar payload only. Vector and matrix second derivatives decompose into per-component
  scalar jets.
* Jet dtypes must be floating point. Quaternion slots are not supported.
