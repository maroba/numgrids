Grids
=====

Grids are formed as tensor products of one or more axes.  A
:class:`~numgrids.Grid` combines the collocation points, differentiation
matrices, and integration weights of its constituent axes into a
multi-dimensional structure.  :class:`~numgrids.MultiGrid` holds a
hierarchy of grids of different resolution for the same domain and solves
linear boundary value problems on it with the multigrid method.

.. autoclass:: numgrids.Grid
   :members:

.. autoclass:: numgrids.MultiGrid
   :members:

Multigrid solver
----------------

:meth:`MultiGrid.solve <numgrids.MultiGrid.solve>` and
:meth:`MultiGrid.solver <numgrids.MultiGrid.solver>` are front ends of the
solver in :mod:`numgrids.multigrid`.

.. autoclass:: numgrids.multigrid.MultiGridSolver
   :members:

.. autoclass:: numgrids.multigrid.MultiGridInfo
   :members:

.. autoexception:: numgrids.multigrid.MultiGridError
