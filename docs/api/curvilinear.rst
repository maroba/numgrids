Curvilinear Grids
=================

Curvilinear grids extend the base grid infrastructure to non-Cartesian
coordinate systems.  They provide vector calculus operators -- gradient,
divergence, curl, and Laplacian -- that account for the metric factors of the
underlying coordinate system.  Concrete implementations are supplied for
spherical, cylindrical, polar, elliptic cylindrical, prolate spheroidal and
oblate spheroidal coordinates.

.. autoclass:: numgrids.CurvilinearGrid
   :members:

.. autoclass:: numgrids.SphericalGrid
   :members:

.. autoclass:: numgrids.CylindricalGrid
   :members:

.. autoclass:: numgrids.PolarGrid
   :members:

.. autoclass:: numgrids.EllipticCylindricalGrid
   :members:

.. autoclass:: numgrids.ProlateSpheroidalGrid
   :members:

.. autoclass:: numgrids.OblateSpheroidalGrid
   :members:
