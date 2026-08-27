warp.geometry
=============

.. automodule:: warp.geometry
   :no-members:

.. currentmodule:: warp.geometry

.. figure:: /img/warp_geometry_connected_components.png
   :align: center
   :width: 80%

   The 50 edge-connected components of the libigl *truck* mesh (2,956 vertices,
   4,770 triangles) labeled by :func:`connected_components` -- the body is a
   single component while each wheel, tire, and the spare are separate. Rendered
   with `polyscope <https://polyscope.run>`_ using one color per component.

API
---

.. autosummary::
   :nosignatures:
   :toctree: _generated

   TriangleMeshTopologyStatistics
   connected_components
   triangle_mesh_topology_statistics
