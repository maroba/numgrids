Differentiation, Integration & Interpolation
=============================================

Numgrids provides numerical operators for differentiation, integration, and
interpolation on structured grids.  Each operator is available both as a
class that can be reused across multiple evaluations and as a convenience
function for one-off computations.

.. autoclass:: numgrids.Diff
   :members:

.. autofunction:: numgrids.diff

.. autoclass:: numgrids.Integral
   :members:

.. autofunction:: numgrids.integrate

.. autoclass:: numgrids.Interpolator
   :members:

.. autofunction:: numgrids.interpolate

Operator algebra and solvers
----------------------------

.. automodule:: numgrids.operators
   :no-members:

.. autoclass:: numgrids.Operator
   :members:

Functions of operators
----------------------

.. automodule:: numgrids.funm
   :no-members:

.. autoclass:: numgrids.funm.FourierOperator
   :members: matrix

Linear solvers
--------------

.. automodule:: numgrids.linalg
   :no-members:

.. autoclass:: numgrids.LinearSolver
   :members:

Nonlinear problems
------------------

.. automodule:: numgrids.nonlinear
   :no-members:

.. autofunction:: numgrids.solve_nonlinear

.. autoexception:: numgrids.NonlinearSolverError

.. autofunction:: numgrids.continuation

.. autoclass:: numgrids.ContinuationResult
   :members:

.. automodule:: numgrids.stability
   :no-members:

.. autofunction:: numgrids.linear_stability

.. autoclass:: numgrids.StabilityResult

.. autoclass:: numgrids.Bifurcation

Spectral coefficients
---------------------

.. automodule:: numgrids.spectral
   :no-members:

.. autofunction:: numgrids.spectral.coefficients

.. autofunction:: numgrids.spectral.from_coefficients

.. autofunction:: numgrids.spectral.mode_numbers

.. autofunction:: numgrids.spectral.resample

.. autofunction:: numgrids.spectral.resolution

.. autofunction:: numgrids.spectral.is_resolved

.. autoclass:: numgrids.spectral.Resolution
   :members:

.. autoclass:: numgrids.spectral.AxisResolution
   :members:

.. autofunction:: numgrids.spectral.spectral_filter
