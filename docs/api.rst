API
===


Working with Axes and Grids
---------------------------

.. automodule:: numgrids
    :members: create_axis

.. autoclass:: numgrids.AxisType
    :members:

.. autoclass:: numgrids.axes.EquidistantAxis
    :members:

.. autoclass:: numgrids.axes.ChebyshevAxis
    :members:

.. autoclass:: numgrids.axes.LogAxis
    :members:

.. autoclass:: numgrids.Grid
    :members:

.. autoclass:: numgrids.MultiGrid
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

.. autofunction:: numgrids.diff

.. autoclass:: numgrids.Integral
    :members:

.. autofunction:: numgrids.integrate

.. autoclass:: numgrids.Interpolator
    :members:

.. autofunction:: numgrids.interpolate


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


Adaptive Mesh Refinement
------------------------

.. autoclass:: numgrids.ErrorEstimator
    :members:

.. autoclass:: numgrids.AdaptationResult
    :members:

.. autofunction:: numgrids.adapt

.. autofunction:: numgrids.estimate_error


Save / Load
-----------

.. autofunction:: numgrids.save_grid

.. autofunction:: numgrids.load_grid
