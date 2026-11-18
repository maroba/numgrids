# Axes

An **axis** is the fundamental building block in numgrids. It represents a
single coordinate dimension -- a 1D set of grid points with a specific
spacing strategy. One or more axes are combined into a
[Grid](grids.md) via tensor product.

```python
from numgrids import *
import numpy as np
```

## Creating axes

The quickest way is one of the four short-hand factories, one per axis type:

```python
x = chebyshev(30, 0.0, 1.0)          # Chebyshev nodes on [0, 1]
y = equidistant(50, 0.0, 1.0)        # uniform spacing on [0, 1]
phi = periodic(64)                   # uniform, periodic on [0, 2*pi)
r = logarithmic(40, 1e-3, 10.0)      # log-spaced on [1e-3, 10]
```

All of them take `(num_points, low, high, name=None)`; sensible defaults
exist for the bounds (`[-1, 1]` for Chebyshev, `[0, 1]` for equidistant,
`[0, 2*pi)` for periodic).

Further factories cover special situations: `radial` and `colatitude` for
curvilinear grids that include the origin or the poles, and `semi_infinite`
and `infinite` for [unbounded domains](#unbounded-domains):

```python
r = semi_infinite(40, scale=5.0)     # the half line [0, inf)
x = infinite(60, scale=4.0)          # the whole real line
```

Alternatively, use the generic `create_axis` factory:

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
ax = create_axis("chebyshev", 50, 0.0, 1.0)   # strings work, too
```

The arguments are:

1. **axis type** -- an `AxisType` member or its string value
   (`"equidistant"`, `"equidistant_periodic"` / `"periodic"`, `"chebyshev"`,
   `"log"`, `"radial"`, `"colatitude"`, `"semi_infinite"`, `"infinite"`)
2. **num_points** -- number of grid points (at least 2)
3. **low** -- lower coordinate bound
4. **high** -- upper coordinate bound

## Axis types

### Equidistant (non-periodic)

```python
ax = create_axis(AxisType.EQUIDISTANT, 50, 0.0, 1.0)
```

Uniformly spaced points from `low` to `high` (both endpoints included).
Differentiation uses classical **finite differences** via the findiff library.
Best for problems where uniform resolution is sufficient and the solution does
not have sharp boundary layers.

### Equidistant periodic

```python
ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, 64, 0.0, 2 * np.pi)
```

Uniformly spaced points on a periodic domain. The last point is *not*
included because it would coincide with the first after wrapping.
Differentiation uses **FFT spectral** methods, giving spectral accuracy for
smooth periodic functions.

### Chebyshev

```python
ax = create_axis(AxisType.CHEBYSHEV, 30, -1.0, 1.0)
```

Grid points are placed at Chebyshev nodes, which cluster near the domain
boundaries:

$$
x_k = \cos\!\left(\frac{k\pi}{N-1}\right), \quad k = 0, \ldots, N-1
$$

mapped linearly to $[\text{low}, \text{high}]$. Differentiation uses the
**Chebyshev spectral** differentiation matrix, achieving spectral accuracy
for smooth non-periodic functions. Excellent for resolving boundary layers.

### Logarithmic

```python
ax = create_axis(AxisType.LOGARITHMIC, 40, 0.01, 10.0)
```

Points are logarithmically spaced, clustering near the lower boundary. The
internal coordinate is $\xi = \ln(x)$, and differentiation applies the
chain rule:

$$
\frac{df}{dx} = \frac{1}{x}\,\frac{df}{d(\ln x)}
$$

```{warning}
The lower bound must be strictly positive (`low > 0`). A `ValueError` is
raised otherwise.
```

### Radial (through the origin)

```python
ax = radial(24, 1.0)            # or create_axis("radial", 24, 0, 1.0)
```

The radial coordinate of a *full* disk, cylinder or ball. The points are the
positive half of Chebyshev nodes on the diameter $[-R, R]$, so no point lies
on the origin. Derivatives use the whole diameter: the value at $-r$ is the
value at the antipodal point $(r, \varphi + \pi)$. This gives spectral
accuracy without a coordinate singularity, and only the outer end $r = R$ is
a boundary. A radial axis can only be used as the first axis of a
`PolarGrid`, `CylindricalGrid` or `SphericalGrid`; see
[Curvilinear grids](curvilinear.md#full-disks-cylinders-spheres-and-balls).

### Colatitude (through the poles)

```python
ax = colatitude(16)             # or create_axis("colatitude", 16, 0, np.pi)
```

The polar angle $\theta \in (0, \pi)$ of a full sphere or ball. The points are
$\theta_j = (j + \frac12)\pi/n$, so no point lies on a pole. Derivatives use
Fourier differentiation on the full great circle (a meridian continues across
the pole at $\varphi + \pi$), integrals use Fejér's quadrature. The axis has no
boundary. It can only be used as the polar axis of a `SphericalGrid` or
`SphericalSurfaceGrid`.

### Semi-infinite (half line)

```python
ax = semi_infinite(40, scale=5.0, low=0.0)
ax = create_axis("semi_infinite", 40, 0.0, np.inf, scale=5.0)   # the same
```

The half line $[\text{low}, \infty)$. The points are mapped Chebyshev points:
the first one is `low`, half of them lie in `[low, low + scale]`, and they
thin out towards infinity without ever reaching it. Differentiation is
spectrally accurate for smooth functions that decay (or level off) at
infinity. Only `low` is a boundary. See
[Unbounded domains](#unbounded-domains).

### Infinite (whole line)

```python
ax = infinite(60, scale=4.0, center=0.0)
ax = create_axis("infinite", 60, -np.inf, np.inf, scale=4.0)    # the same
```

The whole real line. Half of the (mapped Chebyshev) points lie in
`[center - scale, center + scale]`. Spectrally accurate for smooth functions
that decay at infinity. The axis has no boundary at all. See
[Unbounded domains](#unbounded-domains).

## Choosing the right axis type

| Scenario | Recommended type | Why |
|----------|-----------------|-----|
| Smooth, periodic function (e.g. Fourier modes) | `EQUIDISTANT_PERIODIC` | FFT spectral accuracy |
| Smooth, non-periodic function | `CHEBYSHEV` | Spectral accuracy without Runge phenomenon |
| Uniform resolution, moderate accuracy needs | `EQUIDISTANT` | Simple, controllable via `acc` parameter |
| Fine resolution near one boundary (e.g. $r \to 0$) | `LOGARITHMIC` | Concentrates points near `low` |
| Radial coordinate of a full disk, cylinder or ball | `RADIAL` | Includes the center without a singularity |
| Polar angle of a full sphere or ball | `COLATITUDE` | Includes the poles without a singularity |
| Radial coordinate away from origin (annulus, shell) | `CHEBYSHEV` or `LOGARITHMIC` | Depends on whether the lower bound is positive |
| Half line $[a, \infty)$, solution decays at infinity | `SEMI_INFINITE` | No artificial outer boundary, spectral accuracy |
| Whole real line, solution decays at infinity | `INFINITE` | No boundary at all, spectral accuracy |

```{tip}
When in doubt, start with `CHEBYSHEV`. It gives spectral accuracy and works
well for most smooth, non-periodic problems.
```

## Unbounded domains

Bound states, boundary layers, solitary waves and wave packets live on a half
line or on the whole real line. Cutting the domain off at some large value
means inventing a boundary condition there and checking that the cut-off does
not matter. The axes `semi_infinite` and `infinite` discretize the unbounded
domain itself, with spectral accuracy:

```python
r = semi_infinite(40, scale=5.0)              # [0, inf)
s = semi_infinite(40, scale=5.0, low=1.0)     # [1, inf)
x = infinite(60, scale=4.0)                   # (-inf, inf), points cluster around 0
y = infinite(60, scale=4.0, center=2.0)       # (-inf, inf), points cluster around 2
```

### How it works

The unbounded coordinate $x$ is the image of a computational variable
$\xi \in [-1, 1]$ under an algebraic map with a **scale length** $L$
(the `scale` argument), and the grid points are the images of Chebyshev
points in $\xi$:

| | `semi_infinite` | `infinite` |
|---|---|---|
| Domain | $[a, \infty)$ | $(-\infty, \infty)$ |
| Map | $x = a + L\,\dfrac{1 + \xi}{1 - \xi}$ | $x = x_c + \dfrac{L\,\xi}{\sqrt{1 - \xi^2}}$ |
| Points in $\xi$ | Chebyshev--Gauss--Radau: $\xi = -1$ included, $\xi = 1$ not | Chebyshev--Gauss: no endpoints |
| Half of the points lie in | $[a, a + L]$ | $[x_c - L, x_c + L]$ |
| Outermost point | $\approx a + 1.6\,n^2 L$ | $\approx x_c \pm 0.64\,n L$ |
| Boundary faces | `"low"` | none |

Infinity corresponds to $\xi = 1$ (and $\xi = -1$ on the whole line), where
there is no grid point. Functions are represented by polynomials in $\xi$ --
these are the *rational Chebyshev functions* of $x$ -- and derivatives follow
from the chain rule $d/dx = (d\xi/dx)\, d/d\xi$. This converges faster than
any power of the number of points for smooth functions that decay, or level
off, at infinity:

```python
grid = Grid(x=infinite(80, scale=4.0))
x = grid.coords
f = np.exp(-x ** 2)

np.abs(grid.diff(f) + 2 * x * f).max()        # 6e-15
grid.integrate(f) - np.sqrt(np.pi)            # 4e-16
```

Everything else works as on any other axis: operators, boundary conditions at
the finite end, eigenproblems, time evolution, integration, interpolation,
refinement, saving and loading. The axes can be combined freely with other
axes in a grid, e.g. `Grid(x=infinite(48, scale=3.0), y=chebyshev(16, 0, 1))`
for an infinitely long channel.

`ax.low` and `ax.high` are the ends of the domain, so they can be
`-np.inf` or `np.inf`; the grid points themselves (`ax.coords`) are all
finite. `ax.to_internal(x)` and `ax.from_internal(xi)` convert between $x$ and
$\xi$.

### What happens at infinity

There is no grid point at infinity, and no boundary condition is imposed
there. None is needed: a solution that grows or keeps oscillating at infinity
is not a polynomial in $\xi$, so the discretization does not find it. What
remains is the solution that is regular at infinity -- for example the
decaying one of $u'' - u = 0$:

```python
grid = Grid(x=semi_infinite(40, scale=4.0))
x = grid.coords
L = grid.diff_operator(2) - 1                 # u'' - u

u = L.solve(0, bcs={"x_low": 1.0})            # u(0) = 1 and nothing else
np.abs(u - np.exp(-x)).max()                  # 2e-13
```

The finite end of a half line is an ordinary boundary face, so Neumann and
Robin conditions work there as well:

```python
u = L.solve(0, bcs={"x_low": Neumann(1.0)})   # -u'(0) = 1, again exp(-x)
```

```{warning}
This only determines the solution if the equation has *one* solution that is
regular at infinity. The Laplace equation outside a sphere has two, $1$ and
$1/r$, and the discrete problem is then (nearly) singular. Solve such
problems for a decaying unknown instead, or add a term that selects one
solution (e.g. $\nabla^2 u - \kappa^2 u$).
```

What cannot be represented: functions that grow at infinity, and waves that
travel to infinity without decaying, such as $\sin x$ or $e^{ikx}$. A function
that oscillates many times while it decays needs many more points than one
that decays monotonically.

### Choosing `scale`

The scale length decides where the points are: half of them within one scale
length, the other half spread out to infinity. It should match the function,
and it matters more than on a bounded axis:

```python
f = lambda x: np.exp(-x ** 2 / 2)             # a Gaussian of width 1
d2f = lambda x: (x ** 2 - 1) * np.exp(-x ** 2 / 2)

for scale in (0.5, 2.0, 5.0, 20.0):
    grid = Grid(infinite(48, scale=scale))
    x = grid.coords
    print(scale, np.abs(grid.diff(f(x), 2) - d2f(x)).max())
# 0.5   5e-03      too small: the tails are not resolved
# 2.0   8e-07
# 5.0   4e-10
# 20.0  3e-02      too large: too few points where the function lives
```

Rules of thumb:

* **`infinite`**: take `scale` about as large as the half-width of the region
  in which the function is appreciably different from its limits -- 3 to 5
  standard deviations for a Gaussian, 2 to 4 widths for `sech` or a
  Lorentzian.
* **`semi_infinite`**: take `scale` several times the decay length -- 5 to 20
  for an exponential decay. The points cluster strongly at the finite end, so
  a generous value is cheap, whereas too small a value is not.
* The best value grows slowly with the number of points.
* When in doubt, repeat the computation with another `scale` or more points:
  what does not change is converged.

### Example: the quantum harmonic oscillator

The eigenvalues of $-\tfrac12 u'' + \tfrac12 x^2 u = E u$ on the real line
are $E_n = n + \tfrac12$:

```python
grid = Grid(x=infinite(64, scale=5.0))
x = grid.coords
H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2

E, psi = H.eig(k=8, which="SR")
E                                             # [0.5, 1.5, ..., 7.5], error 2e-14
```

No boundary conditions are passed: the grid has no faces, and the solutions
that grow like $e^{x^2/2}$ are excluded automatically. The eigenfunctions can
be normalized with the grid's quadrature:

```python
psi0 = psi[0] / np.sqrt(grid.integrate(psi[0] ** 2))
np.abs(psi0 - np.pi ** -0.25 * np.exp(-x ** 2 / 2)).max()    # 2e-14
```

### Example: the hydrogen atom

The radial functions $u(r) = r R(r)$ of the s states satisfy
$-\tfrac12 u'' - u/r = E u$ with $u(0) = 0$; the levels are
$E_n = -1/(2n^2)$:

```python
grid = Grid(r=semi_infinite(48, scale=10.0))
r = grid.coords
V = np.zeros(grid.shape)
V[1:] = -1 / r[1:]                            # r[0] = 0 is the Dirichlet point
H = -0.5 * grid.diff_operator(2) + V

E, u = H.eig(k=3, bcs=0, which="SR")
E                                             # [-0.5, -0.125, -0.0555...], error 8e-13
```

The condition $u(0) = 0$ is imposed at the face `"r_low"` (here simply
`bcs=0`, since it is the only face). The potential is singular at the first
grid point $r = 0$; its value there does not enter the problem, but it must be
finite.

### Integration and interpolation

`grid.integrate(f)` integrates over the whole unbounded domain. The integrand
has to decay: faster than $1/x$ on the half line, like $1/x^2$ or faster on
the whole line.

```python
grid = Grid(r=semi_infinite(48, scale=8.0))
r = grid.coords
grid.integrate(r ** 2 * np.exp(-r))           # 2.0

grid = Grid(x=infinite(48, scale=2.0))
grid.integrate(1 / (1 + grid.coords ** 2))    # 3.141592653589793
```

Interpolation works everywhere in the domain, also far beyond the outermost
grid point and at infinity itself, where it returns the limit of the function:

```python
grid = Grid(x=semi_infinite(40, scale=4.0))
x = grid.coords
grid.interpolate(np.exp(-x), 2.5, method="spectral")      # exp(-2.5), error 3e-13
grid.interpolate(np.tanh(x), np.inf, method="spectral")   # 1.0
```

The spline methods (`"linear"`, `"cubic"`, ...) work in the variable $\xi$ and
cover the whole domain, too.

### Curvilinear grids

A half line can serve as the radial axis of a `PolarGrid`, `CylindricalGrid`
or `SphericalGrid` to describe the *exterior* of a disk, cylinder or ball;
the only boundary is the inner surface:

```python
from scipy.special import kv                  # modified Bessel function

grid = PolarGrid(semi_infinite(32, scale=2.0, low=1.0), periodic(16))
R, Phi = grid.meshed_coords
L = grid.laplacian_operator() - 1             # lap u - u

u = L.solve(0, bcs={"r_low": lambda c: np.cos(2 * c[1])})
np.abs(u - kv(2, R) / kv(2, 1.0) * np.cos(2 * Phi)).max()   # 5e-11
```

With `low=0` a grid point lies on the coordinate singularity $r = 0$, exactly
as for a Chebyshev axis that starts at the origin. An `infinite` axis is a
natural choice for the $z$ axis of a `CylindricalGrid`.

### Plotting

`grid.plot(f)` and `ax.plot()` show all grid points, and the outermost ones
are far away. Restrict the view to the region of interest, e.g. with
`grid.plot(f).set_xlim(-5, 5)`.

## Axis properties

Every axis exposes the following properties:

### `coords`

The 1D array of coordinate values in the user-specified domain:

```python
ax = create_axis(AxisType.CHEBYSHEV, 5, 0.0, 1.0)
print(ax.coords)
# array([0.  , 0.14644661, 0.5, 0.85355339, 1.  ])
```

### `coords_internal`

The raw coordinates in the axis's internal (canonical) coordinate system.
For Chebyshev axes, these are the nodes on $[-1, 1]$; for logarithmic axes,
the uniformly spaced values in $[\ln(\text{low}), \ln(\text{high})]$:

```python
ax = create_axis(AxisType.CHEBYSHEV, 5, 0.0, 1.0)
print(ax.coords_internal)
# array([ 1. ,  0.70710678,  0. , -0.70710678, -1. ])
```

### `boundary`

A `slice` that selects interior points (excluding boundary). For periodic
axes the full range is returned because there are no boundary points:

```python
ax = create_axis(AxisType.EQUIDISTANT, 10, 0.0, 1.0)
print(ax.boundary)   # slice(1, -1, None)

ax_p = create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0.0, 2 * np.pi)
print(ax_p.boundary) # slice(None, None, None)
```

### `periodic`

Boolean flag indicating whether the axis uses periodic boundary conditions:

```python
print(ax.periodic)    # False
print(ax_p.periodic)  # True
```

## Resizing an axis

The `resized()` method returns a new axis of the same type and domain with a
different number of points. All metadata (periodicity, name) is preserved:

```python
ax = create_axis(AxisType.CHEBYSHEV, 20, 0.0, 1.0)
ax_fine = ax.resized(80)
print(len(ax_fine))  # 80
print(type(ax_fine))  # <class 'numgrids.axes.ChebyshevAxis'>
```

This is used internally by `Grid.refine()`, `Grid.coarsen()`, and the
adaptive mesh refinement module.

## Named axes

Pass the `name` keyword to give an axis a label:

```python
ax_r = chebyshev(30, 0.1, 5.0, name="r")
print(ax_r.name)  # "r"
```

Names can be used instead of axis indices everywhere on a grid
(`grid.diff(f, axis="r")`, `grid.faces["r_high"]`, ...), and they produce
nicer plot labels. `ax.renamed("rho")` returns a renamed copy. The easiest
way to name axes is to pass them as keyword arguments to
[`Grid`](grids.md).

## Quadrature weights

Every axis knows a quadrature rule matching its type, available as
`ax.quadrature_weights` (Clenshaw--Curtis for Chebyshev axes, the
trapezoidal rule for periodic axes, Chebyshev-type rules for the mapped
points of unbounded axes, and a rule consistent with the finite-difference
stencils otherwise):

```python
x = chebyshev(20, 0, np.pi)
print(x.quadrature_weights @ np.sin(x.coords))   # 2.0
```

## Visualization

In Jupyter notebooks, call `plot()` to see where the grid points are placed:

```python
ax = create_axis(AxisType.CHEBYSHEV, 20, 0.0, 1.0)
ax.plot()
```

For periodic axes, `plot()` renders the points on a circle to emphasize the
wrap-around topology.

## Indexing

Axes support integer indexing. For periodic axes, the index wraps around
automatically:

```python
ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, 8, 0.0, 2 * np.pi)
print(ax[0])   # 0.0
print(ax[8])   # 0.0  (wraps)
print(ax[-1])  # same as ax[7]
```

## Length

`len(ax)` returns the number of grid points:

```python
ax = create_axis(AxisType.EQUIDISTANT, 100, 0.0, 1.0)
print(len(ax))  # 100
```
