# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimize independent decoded clouds using symmetric Chamfer loss and fixed targets."""

import argparse

import numpy as np

import warp as wp


@wp.kernel
def mean_distance_kernel(distances: wp.array3d[float], loss: wp.array[float], weight: float):
    batch, query = wp.tid()
    wp.atomic_add(loss, 0, distances[batch, query, 0] * weight)


@wp.kernel
def update_decoded_kernel(decoded: wp.array2d[wp.vec3], gradients: wp.array2d[wp.vec3], rate: float):
    batch, point = wp.tid()
    decoded[batch, point] -= rate * gradients[batch, point]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=("warp", "cubql"), default="warp", help="Fixed-target query backend")
    parser.add_argument(
        "--reverse-backend", choices=("warp", "cubql"), default="warp", help="Changing decoded index backend"
    )
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least two to demonstrate optimization")
    backend = wp.PointQueryBackend.CUBQL if args.backend == "cubql" else wp.PointQueryBackend.WARP
    reverse_backend = wp.PointQueryBackend.CUBQL if args.reverse_backend == "cubql" else wp.PointQueryBackend.WARP
    rng = np.random.default_rng(42)
    targets_host = rng.normal(size=(2, 256, 3)).astype(np.float32)
    decoded_host = targets_host[:, :128] + np.float32(0.2)
    with wp.ScopedDevice(args.device):
        targets = wp.array(targets_host, dtype=wp.vec3)
        decoded = wp.array(decoded_host, dtype=wp.vec3, requires_grad=True)
        if not targets.shape[1] or not decoded.shape[1]:
            raise ValueError("Chamfer loss requires nonempty clouds in both directions")
        # The target coordinates stay fixed for the entire optimization.
        fixed_indexes = wp.point_query_knn_batch(targets, decoded, 1, backend=backend)[0]
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        values = []
        for _ in range(args.steps):
            loss.zero_()
            with wp.Tape() as tape:
                to_target = wp.point_query_knn_batch(targets, decoded, 1, bvh=fixed_indexes)
                # This direction indexes decoded coordinates, which change every step.
                to_decoded = wp.point_query_knn_batch(
                    decoded, targets, 1, backend=reverse_backend, objective=wp.PointQueryObjective.BUILD_AND_QUERY
                )
                wp.launch(mean_distance_kernel, decoded.shape, [to_target[2], loss, 1.0 / decoded.size])
                wp.launch(mean_distance_kernel, targets.shape, [to_decoded[2], loss, 1.0 / targets.size])
            tape.backward(loss)
            values.append(float(loss.numpy()[0]))
            wp.launch(update_decoded_kernel, decoded.shape, [decoded, decoded.grad, 5.0])
            tape.zero()
        print(f"{backend.name}/{reverse_backend.name}: symmetric Chamfer loss {values[0]:.6f} -> {values[-1]:.6f}")
        if not np.isfinite(values).all() or values[-1] >= values[0]:
            raise RuntimeError("Chamfer optimization did not reduce the loss")


if __name__ == "__main__":
    main()
