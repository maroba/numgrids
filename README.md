<h1 align="center">numgrids</h1>
<p align="center"> Working with numerical grids made easy.</p>

<p align="center"><a href="https://badge.fury.io/py/numgrids"> <img src="https://badge.fury.io/py/numgrids.svg?branch=main&kill_cache=1" alt="PyPI version"></a> <a href=""> <img src="https://github.com/maroba/numgrids/actions/workflows/checks.yml/badge.svg" alt="build"></a><a href="https://codecov.io/gh/maroba/numgrids"> <img src="https://codecov.io/gh/maroba/numgrids/branch/main/graph/badge.svg?token=JNH9SP7BRG" alt=""></a></p>

  <div align="center"><img src="docs/assets/torus.png" width="25%">  <img src="docs/assets/disk320.png" width="25%"> <img src="docs/assets/cubic.png" width="25%"></div>

**Main Features**

- Quickly define numerical grids for any rectangular or curvilinear coordinate system
- Differentiation and integration
- Interpolation
- Easy manipulation of meshed functions
- Using high precision spectral methods (FFT + Chebyshev) wherever possible
- Includes multigrid functionality
- Fully compatible with *numpy*

## Installation

```shell
pip install --upgrade numgrids
```

## Quick Start

As a quick example, here is how you define a grid on the unit disk using polar coordinates.
Along the azimuthal (angular) direction, choose an equidistant spacing with periodic boundary conditions:

```python
from numgrids import *
from numpy import pi

axis_phi = Axis(AxisType.EQUIDISTANT, 50, 0, 2*pi, periodic=True)
```

<img src="docs/assets/equi_periodic.png" height="326">

Along the radial axis, let's choose a non-equidistant spacing:

```python
axis_radial = Axis(AxisType.CHEBYSHEV, 20, 0, 1)
```

<img src="docs/assets/cheby.png" height="91">

Now combine the axes to a **grid**:

```python
grid = Grid(axis_radial, axis_phi)
```
<img src="docs/assets/disk320.png">

**Sample** a meshed function on this grid:

```python
from numpy import exp, sin

R, Phi = grid.meshed_coords
f = R**2 * sin(Phi)**2
```
Define **partial derivatives** $\partial/\partial r$ and $\partial/\partial \varphi$ and apply them:

```python
# second argument means derivative order, third argument means axis index:
d_dr = Diff(grid, 1, 0) 
d_dphi = Diff(grid, 1, 1)

df_dr = d_dr(f)
df_dphi = d_dphi(f)
```

Obtain the **matrix representation** of the differential operators:

```python
d_dr.as_matrix()

Out: <1000x1000 sparse matrix of type '<class 'numpy.float64'>'
	with 20000 stored elements in COOrdinate format>
```


Define **integration operator**

$$
\int \dots dr d\varphi
$$

```python
I = Integral(grid)
```

Calculate the area integral

$$
\int f(r, \varphi) r dr d\varphi
$$

(taking into account the appropriate integration measure  $r$  for polar coordinates):

```python
I(f * R)
```

Setting **boundary** values to zero

```python
f[grid.boundary] = 0  # grid.boundary is boolean mask selecting boundary grid points
```

or to something more complicated:

```python
f[grid.boundary] = exp(-R[grid.boundary])
```

Create an **interpolation function**

```python
inter = Interpolator(grid, f)
```

Interpolate for a single point

```python
point = (0.1, 0.5)
inter(point)
```

or for many points at once, like for a parametrized curve:

```python
t = np.linspace(0, 1, 100)
points = zip(2*t, t**2)
inter(points)
```


## Usage / Example Notebooks

To get an idea how *numgrids* can be used, have a look at the following example notebooks:

- [How to define grids](examples/how-to-define-grids.ipynb)
- [Partial derivatives in any dimension](examples/partial-derivatives.ipynb)
- [Polar coordinates on unit disk](examples/polar-cooordinates-on-unit-disk.ipynb)
- [Spherical Grid and the Spherical Laplacian](examples/spherical-grid.ipynb)
- [Solving the Schrödinger equation for the quantum harmonic oscillator](examples/quantum-harmonic-oscillator.ipynb)

## Development

### Setting up the project

Clone the repository

```
git clone https://github.com/maroba/numgrids.git
```

In the project root directory, submit

```
python setup.py develop
```

to install the package in development mode.

Run the tests:

```
python -m unittest discover tests
```

### Contributing

1. Fork the repository
2. Develop
3. Write tests!
4. Create an [issue](https://github.com/maroba/numgrids/issues)
5. Create a pull request, when done
