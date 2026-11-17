<h1 align="center">numgrids</h1>
<p align="center"> Working with numerical grids made easy.</p>

Setting up numerical grids, differentiation matrices, and coordinate
transformations by hand is tedious and error-prone.
*numgrids* gives you a high-level, NumPy-friendly API that handles all of
this — so you can focus on the physics or mathematics of your problem instead
of bookkeeping grid indices and scale factors.

**Main Features**

- Numerical grids in any dimension from **equidistant**, **Chebyshev**, **logarithmic**, and **periodic** axes
- High-precision **spectral methods** (FFT + Chebyshev) selected automatically where possible
- Differentiation, integration (Clenshaw–Curtis, trapezoidal), and interpolation (spline or spectral)
- **Curvilinear coordinates**: spherical (volume and surface), cylindrical, polar, elliptic cylindrical, prolate and oblate spheroidal — or your own via scale factors
- **Full disks, cylinders, spheres and balls**: axes through the origin and the poles handle these points without coordinate singularities
- **Vector calculus**: gradient, divergence, curl, and Laplacian; volume and surface integrals
- **Operator algebra** with built-in `solve()` for boundary value problems and `eig()` for eigenvalue problems
- **Boundary conditions**: Dirichlet, Neumann, and Robin, specified per face in one line
- **Time evolution** (method of lines) for heat, wave, and Schrödinger-type equations
- **Adaptive mesh refinement**, convergence studies, multigrid hierarchies
- **Plotting**, **save / load** to `.npz`, and **xarray / NetCDF** interoperability
- Fully compatible with *NumPy* and *SciPy*

## Installation

```shell
pip install --upgrade numgrids
```

Optional extras: `numgrids[plot]` (matplotlib), `numgrids[xarray]` (xarray), or `numgrids[all]`.

## Quick Start

```python
from numgrids import *
import numpy as np

# A 2D grid: Chebyshev points in x, periodic in y
grid = Grid(x=chebyshev(30, 0, 1), y=periodic(40))
f = grid.evaluate(lambda x, y: x**2 * np.sin(y))

df_dx = grid.diff(f, axis="x")      # spectral derivative
total = grid.integrate(f**2)        # integral over the domain

# Solve the Poisson equation with u = 0 on the boundary
grid = Grid(x=chebyshev(30, 0, 1), y=chebyshev(30, 0, 1))
u = grid.laplacian_operator().solve(1.0, bcs=0)

# Quantum harmonic oscillator: lowest energies 0.5, 1.5, 2.5, ...
grid = Grid(x=chebyshev(80, -10, 10))
H = -0.5 * grid.diff_operator(2) + 0.5 * grid.coords**2
energies, states = H.eig(k=4, bcs=0)
```

Full documentation including API reference and example notebooks is available at
**[maroba.github.io/numgrids](https://maroba.github.io/numgrids/)**.
