# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative linear solvers for sparse linear systems.

This module provides GPU-accelerated iterative methods for solving linear systems,
including conjugate gradient (CG), biconjugate gradient stabilized (BiCGSTAB), conjugate
residual (CR), and generalized minimal residual (GMRES) methods.

Prepared system solves
----------------------

Use :class:`KrylovSolve` for repeated pure-Warp solves or explicitly opt into
:class:`CuDSS` for a CUDA sparse direct factorization. Both expose
``solve(rhs, out)``, ``update()``, ``reset_stats()`` and device diagnostics.
They accept rows of shape ``(nrhs, n)`` and write to caller-owned storage.
Eigen routines borrow either action; they do not assemble or factor matrices.
Only KrylovSolve supports CPU execution and an initial guess.

cuDSS is optional. Importing Warp and running pure-Warp solvers never loads it.
The initial adapter is qualified with ``warp-cudss`` at commit
``e1d1be41decb8d6a4305acdd40e26a5a86c0f797`` and NVIDIA cuDSS 0.8.0.10.
Install the provider explicitly into an existing project environment::

    uv pip install nvidia-cudss-cu12==0.8.0.10 \\
        'warp-cudss @ git+https://github.com/alecjacobson/warp-cuDSS.git@e1d1be41decb8d6a4305acdd40e26a5a86c0f797'

The provider's own dependencies and NVIDIA cuDSS license apply. No native
binding or vendor binary is included in Warp. Unsupported CPU use and missing
dependencies raise explicit errors, without selecting another solver.

Construct CuDSS outside capture to analyze and factor an assembled matrix::

    from warp.optim import linear

    inverse = linear.CuDSS(A, nrhs=1, matrix_type="spd")
    inverse.solve(b, x)
    with wp.ScopedCapture(device=A.device) as capture:
        inverse.solve(b, x)
    wp.capture_launch(capture.graph)

After changing values in existing sparse buffers, call ``inverse.update()``
outside capture. Keep the sparsity pattern fixed. Changes to dimensions,
topology or array addresses require rebuilding the action and its graphs.
Graph objects retain native resources; destroy dependent graphs before
``inverse.close()``. Record on the construction stream and explicitly order
replays that share the mutable solver. Fresh residual diagnostics describe
the original linear system; inspect ``converged`` before accepting the answer.
The direct action does not provide reverse-mode differentiation.

.. automethod:: warp.optim.linear.CuDSS.reset_stats
"""

# isort: skip_file

from warp._src.optim.fsai import FSAI as FSAI
from warp._src.optim.linear import BiCGSTAB as BiCGSTAB
from warp._src.optim.linear import CG as CG
from warp._src.optim.linear import CR as CR
from warp._src.optim.linear import GMRES as GMRES
from warp._src.optim.linear import LinearOperator as LinearOperator
from warp._src.optim.linear import LinearSolverState as LinearSolverState
from warp._src.optim.linear import aslinearoperator as aslinearoperator
from warp._src.optim.linear import bicgstab as bicgstab
from warp._src.optim.linear import cg as cg
from warp._src.optim.linear import cr as cr
from warp._src.optim.linear import gmres as gmres
from warp._src.optim.linear import preconditioner as preconditioner

from warp._src.optim._multi_rhs import MultiRHSOperator as MultiRHSOperator
from warp._src.optim._solve_action import KrylovSolve as KrylovSolve

from warp._src.optim._cudss import CuDSS as CuDSS
