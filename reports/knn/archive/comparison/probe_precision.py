"""Small equal-norm diagnostic for cuVS's expanded-L2 and Minkowski paths."""

import json

import cupy as cp
import numpy as np
import torch

import warp as wp
from warp.examples.benchmarks.point_query_backends import Provider


def main():
    wp.init()
    torch.set_grad_enabled(False)
    stream = torch.cuda.Stream()
    # Equal norms, distinct coordinates, and exactly representable float32 inputs.
    points = np.array([[0.5, -1 / 1024, 0], [0.5, 0, 1 / 1024]], dtype=np.float32)
    queries = np.array([[0.5, 1 / 1024, 0]], dtype=np.float32)
    expected = np.linalg.norm(points.astype(np.float64) - queries.astype(np.float64), axis=1)
    rows = []
    with (
        torch.cuda.stream(stream),
        wp.ScopedStream(wp.stream_from_torch(stream)),
        cp.cuda.ExternalStream(stream.cuda_stream),
    ):
        pt, qt = torch.tensor(points, device="cuda"), torch.tensor(queries, device="cuda")
        torch.cuda.synchronize()
        for name in ("cuvs_brute_force", "cuvs_unexpanded", "cuvs_minkowski"):
            provider = Provider(name, ".")
            for k in (1, 2):
                problem = provider.bind(pt, qt, k)
                _index, result = problem.one_shot()
                torch.cuda.synchronize()
                rows.append(
                    {"method": name, "k": k, "indices": result[0].cpu().tolist(), "distances": result[1].cpu().tolist()}
                )
    print(
        json.dumps(
            {
                "points": points.tolist(),
                "queries": queries.tolist(),
                "norms_squared": np.sum(points.astype(np.float64) ** 2, axis=1).tolist(),
                "query_norm_squared": np.sum(queries.astype(np.float64) ** 2, axis=1).tolist(),
                "true_distances": expected.tolist(),
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
