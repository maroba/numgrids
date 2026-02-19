Grids
=====

Grids are formed as tensor products of one or more axes.  A
:class:`~numgrids.Grid` combines the collocation points, differentiation
matrices, and integration weights of its constituent axes into a
multi-dimensional structure.  :class:`~numgrids.MultiGrid` extends this
concept to composite domains consisting of several sub-grids.

.. autoclass:: numgrids.Grid
   :members:

.. autoclass:: numgrids.MultiGrid
   :members:
