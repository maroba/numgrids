API
===


Working with Axes and Grids
---------------------------

.. automodule:: numgrids
    :members: create_axis

.. autoclass:: numgrids.AxisType
    :members:

.. autoclass:: numgrids.Grid
    :members:

.. autoclass:: numgrids.SphericalGrid
    :members:

.. autoclass:: numgrids.CylindricalGrid
    :members:

.. autoclass:: numgrids.PolarGrid
    :members:


Differentiation, Integration and Interpolation
----------------------------------------------

.. autoclass:: numgrids.Diff
    :members:

.. autoclass:: numgrids.Integral
    :members:

.. autoclass:: numgrids.Interpolator
    :members:


Boundary Conditions
-------------------

.. autoclass:: numgrids.BoundaryFace
    :members:

.. autoclass:: numgrids.DirichletBC
    :members:

.. autoclass:: numgrids.NeumannBC
    :members:

.. autoclass:: numgrids.RobinBC
    :members:

.. autofunction:: numgrids.apply_bcs


Save / Load
-----------

.. autofunction:: numgrids.save_grid

.. autofunction:: numgrids.load_grid

