Grids
=====

Grids are formed as tensor products of one or more axes.  A
:class:`~numgrids.Grid` combines the collocation points, differentiation
matrices, and integration weights of its constituent axes into a
multi-dimensional structure.  :class:`~numgrids.MultiGrid` holds a
hierarchy of grids of different resolution for the same domain.

.. autoclass:: numgrids.Grid
   :members:

.. autoclass:: numgrids.MultiGrid
   :members:
