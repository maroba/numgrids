Save & Load
===========

Grids and their associated data can be persisted to disk and restored later.
These functions serialise a grid's structure and, optionally, any solution
arrays defined on it, so that expensive grid constructions do not need to be
repeated between sessions.

.. autofunction:: numgrids.save_grid

.. autofunction:: numgrids.load_grid
