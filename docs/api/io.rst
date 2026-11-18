Save & Load
===========

Grids and their associated data can be persisted to disk and restored later.
These functions serialise a grid's structure and, optionally, any solution
arrays defined on it, so that expensive grid constructions do not need to be
repeated between sessions.

.. autofunction:: numgrids.save_grid

.. autofunction:: numgrids.load_grid

xarray interoperability
-----------------------

Requires the optional ``xarray`` dependency (``pip install numgrids[xarray]``).

.. autofunction:: numgrids.to_xarray

.. autofunction:: numgrids.from_xarray

VTK export
----------

Files for `ParaView <https://www.paraview.org>`_, VisIt and other VTK-based
tools.  They are written without additional dependencies.  See also
:meth:`Grid.to_vtk <numgrids.Grid.to_vtk>`,
:meth:`Grid.to_vtk_series <numgrids.Grid.to_vtk_series>` and
:meth:`EvolutionResult.to_vtk <numgrids.EvolutionResult.to_vtk>`.

.. autofunction:: numgrids.vtk.write_vtk

.. autofunction:: numgrids.vtk.write_vtk_series

.. autofunction:: numgrids.vtk.cartesian_components
