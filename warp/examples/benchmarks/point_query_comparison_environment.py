# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch an optional comparison environment with consistent CUDA wheel libraries.

Useful when a comparison venv reuses PyTorch from another environment: preload
that venv's matching cuBLAS pair before PyTorch can load an older copy. The
explicit toolkit also supplies the unversioned NVRTC linker name KeOps needs.
Only the child process environment is changed.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--module", default="warp.examples.benchmarks.benchmark_point_query_comparison")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python = args.python.absolute()
    cuda = args.cuda.absolute()
    sites = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json,sys; print(json.dumps([p for p in sys.path if p.endswith('site-packages')]))",
            ],
            text=True,
        )
    )
    libraries = []
    cublas = None
    for site in map(Path, sites):
        libraries.extend(site.glob("nvidia/*/lib"))
        libraries.extend(site.glob("lib*/lib64"))
        candidate = site / "nvidia/cublas/lib"
        if cublas is None and (candidate / "libcublasLt.so.12").exists():
            cublas = candidate
    libraries.append(cuda / "lib64")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries)) + ":" + env.get("LD_LIBRARY_PATH", "")
    if cublas is not None:
        preload = [str(cublas / "libcublasLt.so.12"), str(cublas / "libcublas.so.12")]
        if env.get("LD_PRELOAD"):
            preload.append(env["LD_PRELOAD"])
        env["LD_PRELOAD"] = ":".join(preload)
    env["LIBRARY_PATH"] = str(cuda / "lib64") + ":" + env.get("LIBRARY_PATH", "")
    env["CUDA_HOME"] = env["CUDA_PATH"] = str(cuda)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    os.execve(str(python), [str(python), "-m", args.module, *arguments], env)


if __name__ == "__main__":
    main()
