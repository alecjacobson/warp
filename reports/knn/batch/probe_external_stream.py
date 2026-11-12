"""Linux CUDA-driver stress probe for the external-stream batch path."""

import ctypes
import json

import numpy as np

import warp as wp

wp.init()
wp.set_mempool_release_threshold("cuda:0", 512 * 1024**2)
lib = ctypes.CDLL("libcuda.so.1")
lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
handle = ctypes.c_void_p()
with wp.ScopedDevice("cuda:0"):
    if lib.cuStreamCreate(ctypes.byref(handle), 1) != 0:
        raise RuntimeError("cuStreamCreate failed")
    rng = np.random.default_rng(92)
    p = wp.array(rng.normal(size=(100000, 3)).astype(np.float32), dtype=wp.vec3)
    q = wp.array(rng.normal(size=(1024, 3)).astype(np.float32), dtype=wp.vec3)
    warm = wp.point_query_knn_batch(p, q, 8, ordered=True)
    expected = warm[1].numpy()
    stream = wp.Stream("cuda:0", cuda_stream=handle.value)
    with wp.ScopedStream(stream):
        for ordered in (False, True):
            for _ in range(20):
                result = wp.point_query_knn_batch(p, q, 8, ordered=ordered)
            np.testing.assert_array_equal(result[1].numpy(), expected)
    wp.synchronize_device("cuda:0")
    # Unregister the borrowed handle while the external CUDA stream is alive.
    del stream
    if lib.cuStreamDestroy_v2(handle) != 0:
        raise RuntimeError("cuStreamDestroy failed")
print(json.dumps({"external_nonblocking_burst": True}))
