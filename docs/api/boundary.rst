Boundary Conditions
===================

Boundary conditions can be applied to solution arrays and to the sparse
matrix systems that arise from discretised differential equations.  Numgrids
supports Dirichlet, Neumann, and Robin boundary conditions.
:class:`~numgrids.BoundaryFace` identifies which face of the domain a
condition is imposed on, and :func:`~numgrids.apply_bcs` applies a collection
of conditions in a single call.

.. autoclass:: numgrids.BoundaryFace
   :members:

.. autoclass:: numgrids.DirichletBC
   :members:

.. autoclass:: numgrids.NeumannBC
   :members:

.. autoclass:: numgrids.RobinBC
   :members:

.. autofunction:: numgrids.apply_bcs
