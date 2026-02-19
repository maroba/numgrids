numgrids
========

**Working with numerical grids made easy.**

.. image:: assets/three_grids.png
   :align: center

*numgrids* gives you a high-level, NumPy-friendly API for numerical grids,
differentiation matrices, and coordinate transformations — so you can focus
on the physics or mathematics of your problem instead of bookkeeping grid
indices and scale factors.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Grids & Axes
      :link: guide/axes
      :link-type: doc

      Equidistant, Chebyshev, logarithmic, and periodic axes.
      Tensor-product grids in any dimension.

   .. grid-item-card:: Spectral Differentiation
      :link: guide/differentiation
      :link-type: doc

      FFT, Chebyshev, and finite-difference methods — selected
      automatically. Sparse matrix export for PDE solves.

   .. grid-item-card:: Curvilinear Coordinates
      :link: guide/curvilinear
      :link-type: doc

      Built-in spherical, cylindrical, and polar grids with
      gradient, divergence, curl, and Laplacian.

   .. grid-item-card:: Boundary Conditions
      :link: guide/boundary-conditions
      :link-type: doc

      Dirichlet, Neumann, and Robin BCs at the array level or
      inside sparse linear systems.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   guide/axes
   guide/grids
   guide/differentiation
   guide/integration
   guide/interpolation
   guide/curvilinear
   guide/boundary-conditions
   guide/io
   guide/multigrid
   guide/amr

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/heat-equation-rod
   examples/poisson-equation-2d
   examples/spherical-harmonics
   examples/quantum-harmonic-oscillator
   examples/wave-equation-drum
   examples/adaptive-refinement-demo
   examples/convergence-study

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/axes
   api/grids
   api/operators
   api/curvilinear
   api/boundary
   api/amr
   api/io

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   dev/contributing
   dev/setup
   dev/testing
   dev/changelog
