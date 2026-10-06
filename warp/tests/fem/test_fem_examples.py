# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test warp.fem examples with unittest.

Registers FEM examples as subprocess tests on CUDA devices and runs focused
regression checks on CPU and CUDA devices.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import warp as wp
import warp.sparse as sparse
import warp.tests.unittest_utils
from warp._src.utils import check_p2p
from warp.examples.fem import example_magnetostatics
from warp.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_selected_cuda_test_devices,
    get_selected_cuda_test_devices_with_mempool,
    get_test_devices,
    sanitize_identifier,
)


def _build_command_line_options(test_options: dict[str, Any]) -> list:
    """Build command-line arguments from a test options dictionary."""
    additional_options = []
    for key, value in test_options.items():
        if key == "headless" and value:
            additional_options.extend(["--headless"])
        else:
            additional_options.extend(["--" + key.replace("_", "-"), str(value)])
    return additional_options


def add_fem_example_test(
    cls: type,
    name: str,
    devices: list | None = None,
    test_options: dict[str, Any] | None = None,
    test_options_cpu: dict[str, Any] | None = None,
    test_options_cuda: dict[str, Any] | None = None,
):
    """Register a FEM example that runs as a subprocess test on ``devices``."""
    if test_options is None:
        test_options = {}
    if test_options_cpu is None:
        test_options_cpu = {}
    if test_options_cuda is None:
        test_options_cuda = {}

    def run(test, device):
        if wp.get_device(device).is_cuda:
            options = test_options | test_options_cuda
        else:
            options = test_options | test_options_cpu

        # Mark the test as skipped if NLopt is not installed but required
        nlopt_required = options.pop("nlopt_required", False)
        if nlopt_required:
            try:
                import nlopt  # noqa: PLC0415,F401
            except ImportError:
                test.skipTest("Requires NLopt")

        # Default any USD output into the gitignored warp/tests/outputs/ directory
        # (mirroring the core example harness) and remove it after a passing run,
        # so examples run via the suite never litter the working directory (e.g.
        # the repo root). A test can still pass stage_path=None to disable output.
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        short_name = name.rsplit(".", maxsplit=1)[-1]
        if USD_AVAILABLE:
            stage_path = options.pop(
                "stage_path",
                os.path.join(outputs_dir, f"{short_name}_{sanitize_identifier(device)}.usd"),
            )
        else:
            options.pop("stage_path", None)
            stage_path = "None"

        if stage_path is None:
            # Forward stage_path=None so the example disables USD output entirely
            # (the example maps the literal "None" back to None).
            options["stage_path"] = None
        else:
            options["stage_path"] = stage_path
            if stage_path != "None":
                os.makedirs(outputs_dir, exist_ok=True)
                try:
                    os.remove(stage_path)
                except OSError:
                    pass

        env_vars = os.environ.copy()

        # Propagate the original WARP_CACHE_PATH (not the resolved cache dir)
        # because init_kernel_cache() appends a version subdirectory and we
        # don't want the subprocess to double-append it.
        if "WARP_CACHE_PATH" in os.environ:
            env_vars["WARP_CACHE_PATH"] = os.environ["WARP_CACHE_PATH"]

        if warp.tests.unittest_utils.coverage_enabled:
            with tempfile.NamedTemporaryFile(
                dir=warp.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = ["coverage", "run", f"--data-file={coverage_file.name}"]

            if warp.tests.unittest_utils.coverage_branch:
                command.append("--branch")

            command.extend(["--source", "warp"])
        else:
            command = [sys.executable]

        command.extend(["-m", f"warp.examples.{name}", "--device", str(device)])
        command.extend(_build_command_line_options(options))

        # Pop after building the command line; no example uses test_timeout
        # as an actual CLI flag, so the ordering is harmless in practice.
        test_timeout = options.pop("test_timeout", 600)

        result = subprocess.run(
            command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
        )

        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # Clean up the output stage on success (mirrors the core example harness).
        if stage_path not in (None, "None") and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    add_function_test(cls, f"test_{name}", run, devices=devices, check_output=False)


cuda_devices = get_selected_cuda_test_devices(mode="basic")
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool(mode="basic")


class TestFemExamples(unittest.TestCase):
    pass


class TestFemDiffusionExamples(unittest.TestCase):
    pass


def test_magnetostatics_residual(test, device):
    solve = example_magnetostatics.fem_example_utils.bsr_cg

    def checked_solve(A, b, x, **kwargs):
        result = solve(A, b=b, x=x, **kwargs)
        # Recompute b - A*x in float64 rather than trusting the recursive residual.
        A64 = sparse.bsr_copy(A, scalar_type=wp.float64)
        b64 = wp.array(b.numpy(), dtype=wp.float64, device=device)
        x64 = wp.array(x.numpy(), dtype=wp.float64, device=device)
        residual = wp.clone(b64)
        sparse.bsr_mv(A64, x64, residual, alpha=-1.0, beta=1.0)
        b_norm_sq = wp.utils.array_inner(b64, b64)
        residual_norm_sq = wp.utils.array_inner(residual, residual)
        test.assertGreater(b_norm_sq, 0.0)
        test.assertLessEqual(residual_norm_sq, 1.0e-8 * b_norm_sq)
        return result

    with wp.ScopedDevice(device):
        for fp64 in (False, True):
            with test.subTest(fp64=fp64):
                example = example_magnetostatics.Example(resolution=16, fp64=fp64)
                with patch.object(
                    example_magnetostatics.fem_example_utils, "bsr_cg", side_effect=checked_solve
                ) as solver:
                    example.step()
                solver.assert_called_once()


add_function_test(
    TestFemExamples,
    "test_magnetostatics_residual",
    test_magnetostatics_residual,
    devices=get_test_devices(mode="basic"),
    check_output=False,
)


# MGPU tests may fail on systems where P2P transfers are misconfigured
if check_p2p():
    add_fem_example_test(
        TestFemDiffusionExamples,
        name="fem.example_diffusion_mgpu",
        devices=cuda_devices_with_mempool,
        test_options={"headless": True},
    )

add_fem_example_test(
    TestFemExamples,
    name="fem.example_apic_fluid",
    devices=cuda_devices_with_mempool,
    test_options={"num_frames": 5, "voxel_size": 2.0, "stage_path": None},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_apic_fluid_multi_env",
    devices=cuda_devices,
    test_options={"num_frames": 1, "env_count": 2, "voxel_size": 4.0, "stage_path": None},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_adaptive_grid",
    devices=cuda_devices,
    test_options={"headless": True, "div_conforming": True},
)
add_fem_example_test(
    TestFemDiffusionExamples,
    name="fem.example_diffusion",
    devices=cuda_devices,
    test_options={"resolution": 10, "mesh": "tri", "headless": True},
)
add_fem_example_test(
    TestFemDiffusionExamples,
    name="fem.example_diffusion_3d",
    devices=cuda_devices,
    test_options={"headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_deformed_geometry",
    devices=cuda_devices,
    test_options={"resolution": 10, "mesh": "tri", "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_convection_diffusion",
    devices=cuda_devices,
    test_options={"resolution": 20, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_burgers",
    devices=cuda_devices,
    test_options={"resolution": 20, "num_frames": 25, "degree": 1, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_convection_diffusion_dg",
    devices=cuda_devices,
    test_options={"resolution": 20, "num_frames": 25, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_mixed_elasticity",
    devices=cuda_devices,
    test_options={"nonconforming_stresses": True, "mesh": "quad", "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_stokes_transfer",
    devices=cuda_devices,
    test_options={"headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_stokes",
    devices=cuda_devices,
    test_options={"resolution": 10, "nonconforming_pressures": True, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_navier_stokes",
    devices=cuda_devices,
    test_options={"num_frames": 101, "resolution": 10, "tri_mesh": True, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_streamlines",
    devices=cuda_devices,
    test_options={"headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_distortion_energy",
    devices=cuda_devices,
    test_options={"headless": True, "resolution": 16},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_magnetostatics",
    devices=cuda_devices,
    test_options={"headless": True, "resolution": 16},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_nonconforming_contact",
    devices=cuda_devices,
    test_options={"headless": True, "resolution": 16, "num_steps": 2},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_elastic_shape_optimization",
    devices=cuda_devices,
    test_options={"num_iters": 5, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_darcy_ls_optimization",
    devices=cuda_devices,
    test_options={"num_iters": 5, "resolution": 25, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_cantilever_topology_optimization",
    devices=cuda_devices,
    test_options={
        "nx": 16,
        "ny": 6,
        "nz": 1,
        "max_evals": 3,
        "headless": True,
        "nlopt_required": True,
    },
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_taylor_green",
    devices=cuda_devices_with_mempool,
    test_options={"num_frames": 10, "resolution": 10, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_shallow_water",
    devices=cuda_devices_with_mempool,
    test_options={"num_frames": 10, "resolution": 10, "headless": True},
)
add_fem_example_test(
    TestFemExamples,
    name="fem.example_kelvin_helmholtz",
    devices=cuda_devices_with_mempool,
    test_options={"num_frames": 25, "resolution": 20, "headless": True},
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
