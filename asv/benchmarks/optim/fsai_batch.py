# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from warp.optim import linear

from ..benchmarks_utils import setup_once
from ._rhs_problem import problem


class FSAIBatch:
    params = [(4, 8, 24, 33), ("cpu", "cuda:0")]
    param_names = ["nrhs", "device"]
    number = 1
    repeat = 10
    rounds = 2

    @setup_once
    def setup(self, nrhs, device):
        if not hasattr(getattr(linear, "FSAI", None), "reserve_batch"):
            raise NotImplementedError("Batched FSAI application is not available")
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            # A captured graph addresses these arrays, so they must stay alive.
            self.a, self.b = problem(nrhs)
            self.out = wp.empty_like(self.b)
            self.factor = linear.FSAI(self.a, max_row_size=16, factor_dtype=wp.float32).reserve_batch(nrhs)
            self._run()
            wp.synchronize_device(self.device)
            self.graph = None
            if self.device.is_cuda:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    self._run()
                self.graph = capture.graph
                wp.capture_launch(self.graph)
                wp.synchronize_device(self.device)

    def _run(self):
        self.factor.matvec(self.b, self.out, self.out, 1.0, 0.0)

    def time_batch(self, nrhs, device):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._run()
        wp.synchronize_device(self.device)
