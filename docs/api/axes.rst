Axes
====

Axes are the one-dimensional building blocks from which grids are constructed.
Each axis defines a set of collocation points and the associated
differentiation and integration rules along a single coordinate direction.
Use the short-hand factories (:func:`~numgrids.chebyshev`,
:func:`~numgrids.equidistant`, :func:`~numgrids.periodic`,
:func:`~numgrids.logarithmic`), the generic :func:`~numgrids.create_axis`,
or instantiate one of the concrete classes directly.

.. autofunction:: numgrids.chebyshev

.. autofunction:: numgrids.equidistant

.. autofunction:: numgrids.periodic

.. autofunction:: numgrids.logarithmic

.. autofunction:: numgrids.radial

.. autofunction:: numgrids.colatitude

.. autofunction:: numgrids.create_axis

.. autoclass:: numgrids.AxisType
   :members:

.. autoclass:: numgrids.axes.EquidistantAxis
   :members:

.. autoclass:: numgrids.axes.ChebyshevAxis
   :members:

.. autoclass:: numgrids.axes.LogAxis
   :members:

.. autoclass:: numgrids.axes.RadialAxis
   :members: radial_quadrature_weights, diameter_coords

.. autoclass:: numgrids.axes.ColatitudeAxis
   :members: sine_quadrature_weights, great_circle_coords
