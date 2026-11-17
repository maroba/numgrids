Time Evolution & Analysis
=========================

Method-of-lines time integration and convergence studies.

.. autofunction:: numgrids.evolve

.. autoclass:: numgrids.EvolutionResult
   :members:

.. autofunction:: numgrids.convergence_study

.. autoclass:: numgrids.ConvergenceResult
   :members:

Plotting
--------

Requires the optional ``matplotlib`` dependency (``pip install numgrids[plot]``).

.. autofunction:: numgrids.plots.plot_field

.. autofunction:: numgrids.plots.animate_field

.. autofunction:: numgrids.plots.save_animation
