# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jet Hessian
#
# Builds the local Hessian of a scalar energy with wp.JetSpace, using
# reverse-over-forward autodiff and no hand-written derivatives.
#
#     g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)
#
# g is written once, as a plain @wp.func over jets. Evaluating it over jets
# instead of floats produces its derivatives as a side effect, and because the
# jet arithmetic is ordinary Warp code, wp.Tape differentiates through it --
# which is what turns a first-order tool into a Hessian.
#
# Two strategies are shown, both materializing the same m x k x k result:
#
#   width-k   One k-wide forward pass yields all of grad g at once. Then k
#             reverse sweeps through that widened program extract its k x k
#             Jacobian, one row at a time.
#
#                 forward ~ O(kC), k reverses ~ O(kC) each  ->  O(k^2 C)
#
#   width-1   For each basis direction e_j, a constant-width forward pass
#             yields the scalar Dg[e_j] = grad g . e_j, and one reverse sweep
#             over that scalar yields
#
#                 grad_z (grad g . e_j) = H e_j,
#
#             i.e. column j of H, in one shot. This is the usual "a Hessian is
#             k Hessian-vector products" identity.
#
#                 forward ~ O(C), reverse ~ O(C), k of them  ->  O(kC)
#
# Only the width-1 path keeps derivative state per intermediate at O(1), so its
# register pressure does not grow with k. At k=2 the two are close; the gap
# opens up quickly with k.
#
# All m local terms are handled in parallel: one launch and one reverse sweep
# per direction, not per element.
#
# Both results are checked against float64 finite differences of g, which
# derive nothing by hand. See design/forward-mode-jets.md.
#
###########################################################################

import argparse

import numpy as np

import warp as wp

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Two local variables per term.
K = 2

# One space per strategy: k tangents at once, or a single tangent.
JK = wp.JetSpace(K)
J1 = wp.JetSpace(1)


def make_energy(J):
    """Instantiate g for a given jet space.

    The body is written once. Only the type it is specialized for changes, so
    the two strategies cannot drift apart.
    """

    @wp.func
    def energy(a: J.scalar, b: J.scalar) -> J.scalar:
        return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)

    return energy


energy_k = make_energy(JK)
energy_1 = make_energy(J1)


@wp.kernel
def local_gradient(z: wp.array2d[float], grad_g: wp.array[JK.coeff]):
    """Width-k: seed the identity, read the whole local gradient off .coeff."""
    i = wp.tid()

    # dz0 = [1,0], dz1 = [0,1]
    z0 = JK.seed(z[i, 0], 0)
    z1 = JK.seed(z[i, 1], 1)

    grad_g[i] = energy_k(z0, z1).coeff


@wp.kernel
def local_directional(z: wp.array2d[float], v: wp.vec2, dv: wp.array[float]):
    """Width-1: one directional derivative, as a scalar per element."""
    i = wp.tid()

    z0 = J1.with_coeff(z[i, 0], J1.coeff(v[0]))
    z1 = J1.with_coeff(z[i, 1], J1.coeff(v[1]))

    dv[i] = energy_1(z0, z1).coeff[0]


def hessian_width_k(z_np, device):
    """k reverse sweeps through the k-wide forward program."""
    m = z_np.shape[0]

    z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
    grad_g = wp.zeros(m, dtype=JK.coeff, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(local_gradient, dim=m, inputs=[z], outputs=[grad_g], device=device)

    hessian = np.empty((m, K, K), dtype=np.float32)

    for row in range(K):
        seed_np = np.zeros((m, K), dtype=np.float32)

        # The same gradient component for every local term.
        seed_np[:, row] = 1.0

        tape.backward(grads={grad_g: wp.array(seed_np, dtype=JK.coeff, device=device)})

        # z.grad[i,b] = d grad_g[i,row] / d z[i,b] = H_i[row,b]
        hessian[:, row, :] = z.grad.numpy()
        tape.zero()

    return grad_g.numpy(), hessian


def hessian_width_1(z_np, device):
    """k forward+reverse pairs, each of constant width."""
    m = z_np.shape[0]

    z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
    ones = wp.ones(m, dtype=float, device=device)

    hessian = np.empty((m, K, K), dtype=np.float32)

    for j in range(K):
        e_j = np.zeros(K, dtype=np.float32)
        e_j[j] = 1.0

        dv = wp.zeros(m, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(local_directional, dim=m, inputs=[z, wp.vec2(e_j)], outputs=[dv], device=device)

        # grad_z (grad g . e_j) = H e_j = column j
        tape.backward(grads={dv: ones})

        hessian[:, :, j] = z.grad.numpy()
        tape.zero()

    return hessian


def g_np(z):
    """g in NumPy, for a reference that shares no code with the jets."""
    a = z[:, 0]
    b = z[:, 1]
    return np.sin(a * b) + 0.1 * a**3 + np.exp(b)


def hessian_reference(z, h=1.0e-4):
    """Second differences of g in float64."""
    out = np.empty((z.shape[0], K, K))

    for p in range(K):
        for q in range(K):
            ep = np.zeros_like(z)
            ep[:, p] = h

            eq = np.zeros_like(z)
            eq[:, q] = h

            out[:, p, q] = (g_np(z + ep + eq) - g_np(z + ep - eq) - g_np(z - ep + eq) + g_np(z - ep - eq)) / (
                4.0 * h * h
            )

    return out


def sample_points(n):
    """An n x n grid over [-1,1]^2."""
    t = np.linspace(-0.9, 0.9, n)
    a, b = np.meshgrid(t, t)
    return np.stack((a.ravel(), b.ravel()), axis=1).astype(np.float32)


def plot(z, grad, hessian):
    """Show g, and at each sample the gradient and the Hessian eigenvectors.

    Eigenvectors are drawn rather than a curvature ellipse because this energy
    is indefinite over much of the domain, and an ellipse only exists where the
    Hessian is positive definite.
    """
    t = np.linspace(-1.0, 1.0, 200)
    A, B = np.meshgrid(t, t)
    G = np.sin(A * B) + 0.1 * A**3 + np.exp(B)

    plt.figure(figsize=(7, 6))
    plt.contourf(A, B, G, levels=30, cmap="Greys", alpha=0.6)
    plt.colorbar(label="g(a,b)")

    values, vectors = np.linalg.eigh(hessian)

    for i in range(z.shape[0]):
        for c in range(K):
            lam = values[i, c]
            d = vectors[i, :, c] * np.sqrt(abs(lam)) * 0.12

            # Blue where the energy curves up, red where it curves down.
            color = "tab:blue" if lam > 0.0 else "tab:red"
            plt.plot(
                [z[i, 0] - d[0], z[i, 0] + d[0]],
                [z[i, 1] - d[1], z[i, 1] + d[1]],
                color=color,
                linewidth=2,
            )

    plt.quiver(
        z[:, 0],
        z[:, 1],
        grad[:, 0],
        grad[:, 1],
        color="black",
        alpha=0.5,
        width=0.003,
        scale=30.0,
    )

    plt.plot([], [], color="tab:blue", linewidth=2, label="positive curvature")
    plt.plot([], [], color="tab:red", linewidth=2, label="negative curvature")
    plt.plot([], [], color="black", alpha=0.5, label="gradient")

    plt.xlabel("a")
    plt.ylabel("b")
    plt.title("Local Hessian eigenvectors from reverse-over-forward jets")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.show()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--grid", type=int, default=5, help="Sample points per axis.")
    parser.add_argument("--headless", action="store_true", help="Do not display a plot.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for harness compatibility.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        z = sample_points(args.grid)

        grad, h_wide = hessian_width_k(z, args.device)
        h_hvp = hessian_width_1(z, args.device)

        reference = hessian_reference(z.astype(np.float64))

        print(f"{z.shape[0]} local terms, k={K}")
        print(f"  max |width-k - width-1|  = {np.abs(h_wide - h_hvp).max():.3e}")
        print(f"  max |width-k - fd|       = {np.abs(h_wide - reference).max():.3e}")
        print(f"  max |width-1 - fd|       = {np.abs(h_hvp - reference).max():.3e}")

        # The Hessian of a scalar energy is symmetric, but the two off-diagonals
        # come from separate reverse sweeps, so agreement is not automatic.
        print(f"  max |H - H^T|            = {np.abs(h_wide - np.transpose(h_wide, (0, 2, 1))).max():.3e}")

        np.testing.assert_allclose(h_wide, reference, rtol=1.0e-3, atol=1.0e-4)
        np.testing.assert_allclose(h_hvp, reference, rtol=1.0e-3, atol=1.0e-4)

        if not args.headless:
            if not MATPLOTLIB_AVAILABLE:
                print("matplotlib not available; skipping plot.")
                return

            plot(z, grad, h_wide)


if __name__ == "__main__":
    main()
