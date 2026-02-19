Adaptive Mesh Refinement
========================

The adaptive mesh refinement (AMR) utilities help determine the resolution
required for a given accuracy target.  An :class:`~numgrids.ErrorEstimator`
evaluates the local discretisation error on a grid, and the
:func:`~numgrids.adapt` function iteratively refines axes until the estimated
error falls below a prescribed tolerance.

.. autoclass:: numgrids.ErrorEstimator
   :members:

.. autoclass:: numgrids.AdaptationResult
   :members:

.. autofunction:: numgrids.adapt

.. autofunction:: numgrids.estimate_error
