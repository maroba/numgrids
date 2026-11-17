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

Compact specifications
----------------------

Unbound specifications are attached to faces when used as dictionary values
in :meth:`Operator.solve <numgrids.Operator.solve>`,
:meth:`Operator.eig <numgrids.Operator.eig>`, :func:`~numgrids.evolve` or
:func:`~numgrids.resolve_bcs`.

.. autoclass:: numgrids.Dirichlet

.. autoclass:: numgrids.Neumann

.. autoclass:: numgrids.Robin

.. autoclass:: numgrids.TimeDependent

.. autofunction:: numgrids.resolve_bcs
