Axes
====

Axes are the one-dimensional building blocks from which grids are constructed.
Each axis defines a set of collocation points and the associated
differentiation and integration rules along a single coordinate direction.
Use :func:`~numgrids.create_axis` to build an axis of the desired type, or
instantiate one of the concrete classes directly.

.. autofunction:: numgrids.create_axis

.. autoclass:: numgrids.AxisType
   :members:

.. autoclass:: numgrids.axes.EquidistantAxis
   :members:

.. autoclass:: numgrids.axes.ChebyshevAxis
   :members:

.. autoclass:: numgrids.axes.LogAxis
   :members:
