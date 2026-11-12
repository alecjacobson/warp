from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_point_query_cubql import DEFAULT, Library, Problem, passed

wp.init()
library = Library(Path("../artifacts/point_query_cubql_upstream.so"))
rng = np.random.default_rng(201)
p = rng.normal(size=(513, 3)).astype(np.float32)
q = rng.normal(size=(65, 3)).astype(np.float32)
with wp.ScopedStream(wp.Stream("cuda:0")):
    for k in (1, 11, 32):
        problem = Problem(library, p, q, k)
        for builder in range(5):
            for stride in (1, 8, 32):
                config = dict(DEFAULT, builder=builder, stride=stride, ordered=stride == 8)
                tree = problem.build(config)
                output = problem.query(tree, config)
                valid = problem.validate(output)
                if not passed(valid):
                    raise RuntimeError((k, config, valid))
                with wp.ScopedCapture() as capture:
                    problem.query(tree, config, output)
                wp.capture_launch(capture.graph)
                if not passed(problem.validate(output)):
                    raise RuntimeError("Graph validation failed")
                del capture, tree, output
    print("45 builder/k/stride cases pass query correctness and graph replay")
