Curvilinear Grids
=================

Curvilinear grids extend the base grid infrastructure to non-Cartesian
coordinate systems.  They provide vector calculus operators -- gradient,
divergence, curl, and Laplacian -- that account for the metric factors of the
underlying coordinate system.  Concrete implementations are supplied for
spherical, cylindrical, and polar coordinates.

.. autoclass:: numgrids.CurvilinearGrid
   :members:

.. autoclass:: numgrids.SphericalGrid
   :members:

.. autoclass:: numgrids.CylindricalGrid
   :members:

.. autoclass:: numgrids.PolarGrid
   :members:
