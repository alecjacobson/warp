import sys

import igl
import numpy as np
import polyscope as ps

import warp as wp

# read in a mesh from argv or if not provided, use icosahedron
if len(sys.argv) > 1:
    mesh_path = sys.argv[1]
    V, F = igl.read_triangle_mesh(mesh_path)
else:
    V, F = igl.icosahedron()


@wp.func
def signed_volume_with_origin(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3) -> float:
    return wp.dot(p0, wp.cross(p1, p2)) / 6.0


@wp.kernel(enable_backward=True)
def total_volume_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    total_volume: wp.array(dtype=float),
):
    tid = wp.tid()
    v = signed_volume_with_origin(
        points[indices[3 * tid + 0]], points[indices[3 * tid + 1]], points[indices[3 * tid + 2]]
    )
    wp.atomic_add(total_volume, 0, v)


wp.init()
points = wp.array(V, dtype=wp.vec3, requires_grad=True)
indices = wp.array(F.flatten(), dtype=int)
total_volume = wp.zeros(1, dtype=float, requires_grad=True)

tape = wp.Tape()
with tape:
    wp.launch(total_volume_kernel, dim=indices.shape[0] // 3, inputs=[points, indices], outputs=[total_volume])

tape.backward(loss=total_volume)

# ∂volume/∂points gives area-weighted vertex normals (pointing outward)
vertex_normals = points.grad.numpy().copy()
vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

ps.init("openGL3_egl")
# ps.set_allow_headless_backends(True)
ps_mesh = ps.register_surface_mesh("mesh", V, F)
ps_mesh.add_vector_quantity("vertex normals", vertex_normals, enabled=True)
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode("shadow_only")
# ps.show()
ps.screenshot("example_vertex_normals_via_autodiff.jpg")
