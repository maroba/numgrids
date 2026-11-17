# Spectral coefficients and resolution

On a spectral axis a function is more than its values at the grid points:
it is a sum of basis functions, complex exponentials on periodic axes and
Chebyshev polynomials on Chebyshev axes. The coefficients of this sum are
the most direct answer to a question that comes up in every computation:
*is my function resolved on this grid?* For a smooth function the
coefficients decay rapidly. If the highest ones are negligible, the grid is
fine enough. If they are not, it is too coarse, and their decay tells how
many points would be needed.

```python
from numgrids import *
import numpy as np
```

## Coefficients

`grid.coefficients(f)` transforms a meshed function to its expansion
coefficients, `grid.from_coefficients(c)` transforms back.

On a **Chebyshev axis** the function is a series of Chebyshev polynomials
in the variable $\xi \in [-1, 1]$ that the axis is mapped to,

$$
f(x) = \sum_{k=0}^{n-1} a_k\, T_k(\xi), \qquad \xi = \frac{2x - a - b}{b - a} .
$$

```python
grid = Grid(x=chebyshev(8, -1, 1))
x = grid.coords

grid.coefficients(2 * x**2 - 1 + 0.5 * x)    # T_2 + T_1 / 2  ->  [0, 0.5, 1, 0, 0, 0, 0, 0]
grid.coefficients(np.exp(x))                 # [1.2661, 1.1303, 0.2715, 0.0443, 0.0055, ...]
```

On a **periodic axis** the function is a Fourier series,

$$
f(x) = \sum_k c_k\, e^{i \kappa_k x}, \qquad \kappa_k = \frac{2\pi k}{P},
$$

with the period $P$. The coefficients are complex and stored in the order
of `numpy.fft.fftfreq`: the mode numbers $k = 0, 1, 2, \ldots$ first, then
the negative ones. `mode_numbers` returns the mode number of each
coefficient. The normalization makes $|c_k|$ the amplitude of a mode:
$c_0$ is the mean value of the function, and a real oscillation
$A \cos(\kappa_k x)$ has $c_k = c_{-k} = A/2$.

```python
from numgrids.spectral import mode_numbers

grid = Grid(x=periodic(8))
x = grid.coords

mode_numbers(grid)                             # [ 0,  1,  2,  3, -4, -3, -2, -1]
c = grid.coefficients(1 + 3 * np.cos(2 * x))
c.real.round(12)                               # [1, 0, 1.5, 0, 0, 0, 1.5, 0]
```

The phases refer to the coordinate $x$ itself, so the coefficients do not
depend on where the period starts. For a real function
$c_{-k} = \bar c_k$; `from_coefficients` returns a complex array on
periodic axes, of which you then take the real part.

```{note}
With an even number of points $n$, the mode $k = n/2$ (the Nyquist mode)
cannot be told from $k = -n/2$ on the grid. Its coefficient is stored at
index $n/2$ and is the amplitude of $\cos(\kappa_{n/2} (x - x_0))$, which
is $\pm 1$ at the grid points.
```

The other axis types:

| Axis | Expansion |
|------|-----------|
| `chebyshev` | Chebyshev series in $\xi$ |
| `periodic` | Fourier series |
| `semi_infinite`, `infinite` | Chebyshev series in the computational variable $\xi(x)$ of the axis (the "rational Chebyshev functions" of $x$) |
| `radial` | Chebyshev series on the whole diameter $-R \le s \le R$ |
| `colatitude` | Fourier series on the whole great circle $0 \le \theta < 2\pi$ |
| `equidistant` (not periodic), `logarithmic` | none |

Finite differences do not belong to an expansion in global basis
functions. Asking for coefficients along such an axis raises an error.

Along a radial or colatitude axis with $n$ points there are $2n$
coefficients. The field is continued across the origin or the poles to the
antipodal points, just as for [differentiation](curvilinear.md), and the
coefficients are those of the function on the whole diameter or great
circle.

### Grids with several axes

By default all axes are transformed, which gives the coefficients of the
tensor-product expansion. The `axis` argument selects one axis (or
several); the other axes keep the values at their grid points.

```python
grid = Grid(x=chebyshev(12, 0, 1), y=periodic(16))
X, Y = grid.meshed_coords
f = np.exp(X) * np.sin(Y)

c = grid.coefficients(f)               # shape (12, 16): Chebyshev in x, Fourier in y
cx = grid.coefficients(f, axis="x")    # Chebyshev coefficients for each value of y

np.allclose(grid.from_coefficients(c).real, f)        # True
np.allclose(grid.from_coefficients(cx, axis="x"), f)  # True
```

Operations that are diagonal in coefficient space become one line. The
derivative with respect to $y$, for example, multiplies $c_k$ by
$i\kappa_k$ (here $P = 2\pi$, so $\kappa_k = k$):

```python
k = mode_numbers(grid, "y")
df_dy = grid.from_coefficients(1j * k * c).real

np.allclose(df_dy, np.exp(X) * np.cos(Y))             # True
```

## Changing the resolution

The coefficients do not depend on the number of grid points, as long as
the function is resolved. `grid.resample` uses this to transfer a function
to a grid with other numbers of points: it pads the coefficients with
zeros or truncates the series.

```python
coarse = Grid(x=chebyshev(16, 0, 1), y=periodic(16))
fine = Grid(x=chebyshev(40, 0, 1), y=periodic(64))

f = coarse.evaluate(lambda x, y: np.exp(x) * np.sin(y))
f_fine = coarse.resample(f, fine)                 # or: coarse.resample(f, (40, 64))

np.abs(f_fine - fine.evaluate(lambda x, y: np.exp(x) * np.sin(y))).max()     # 2e-15
```

Going to more points is the same as
[spectral interpolation](interpolation.md#spectral-interpolation) onto the
new grid. Going to fewer points drops the highest modes; a function that
is resolved on the coarser grid loses nothing. The target must be a grid
of the same kind, with the same domain; axes that are discretized with
finite differences must keep their number of points.

## Is my function resolved?

`grid.resolution(f)` examines the coefficients along each axis and reports
whether the function is resolved, and how many points it needs. As an
example, take the Runge function $1/(1 + 25x^2)$ on too few points, on
enough points, and on far too many:

```python
runge = lambda x: 1 / (1 + 25 * x**2)

for n in (40, 120, 400):
    grid = Grid(x=chebyshev(n, -1, 1))
    print(grid.resolution(runge(grid.coords)))
```

```text
Resolution at tol=1e-08: NOT resolved
  axis x: NOT resolved, tail at 1e-03; about 107 points needed
Resolution at tol=1e-08: resolved
  axis x: resolved, 113 of 120 points needed
Resolution at tol=1e-08: resolved
  axis x: resolved, 113 of 400 points needed
```

All three grids agree on the answer: about 110 points. On the coarse grid
it is an extrapolation of the decay seen so far; on the other two it is
read off the spectrum. The tolerance `tol` (default `1e-8`) is relative to
the largest coefficient.

The result is a small object with the verdict and the details for each
axis:

```python
grid = Grid(x=chebyshev(40, -1, 1))
result = grid.resolution(runge(grid.coords), tol=1e-8)

result.resolved            # False
result["x"].needed         # 107
result["x"].tail           # 1.1e-03: sum of the highest modes, relative to the largest one
result["x"].spectrum       # magnitude of each mode, here |a_k|

grid.is_resolved(runge(grid.coords), tol=1e-2)      # True: enough for two digits
```

Is the estimate right? Interpolating from the suggested number of points
indeed gives the tolerance, and half as many are clearly too few:

```python
x_test = np.linspace(-1, 1, 1000)
study = convergence_study(
    lambda n: Grid(x=chebyshev(n, -1, 1)),
    lambda grid: grid.interpolate(runge(grid.coords), x_test, method="spectral"),
    ns=[56, 113],
    exact=lambda grid: runge(x_test),
)
study.errors               # [3.5e-05, 2.1e-10]
```

### How the check works

For each axis, the magnitude of each mode is maximized over the other
axes and divided by the largest one. The function counts as resolved if
the highest modes (a tenth of all modes, at least two) add up to less than
`tol`. The modes that a truncated series lacks are its error, so this
means: more points would not change the function by more than about `tol`
times its size.

The number of points needed follows from the same rule. If the function is
resolved, the series can be chopped where the sum of all dropped modes
falls below `tol`; `needed` is the number of modes kept, plus the
negligible ones that the check wants to see at the end. If it is not
resolved, the decay of the spectrum is extrapolated. Three cases:

- **Geometric decay**, $|a_k| \sim e^{-ck}$, is the mark of a smooth
  (analytic) function. The estimate is usually right within ten to twenty
  percent.
- **Algebraic decay**, $|a_k| \sim k^{-p}$, is the mark of a function with
  a kink or another singularity. The number of points needed becomes very
  large, and the message says so. A spectral axis is then the wrong tool,
  or the domain should be split at the kink.
- **No decay**: the highest modes are about as large as those in the
  middle of the spectrum. Either the number of points is far too small to
  see any decay, or the data contains noise, or the function has a jump or
  a kink whose spectrum falls too slowly to extrapolate. No estimate is
  possible.

```python
grid = Grid(x=chebyshev(100, -1, 1))
x = grid.coords
rng = np.random.default_rng(0)

print(grid.resolution(np.abs(x)**3)["x"])
print(grid.resolution(np.exp(x) + 1e-5 * rng.standard_normal(100))["x"])
```

```text
axis x: NOT resolved, tail at 3e-07; about 604 points needed (algebraic decay of order 4.2: the function is not smooth)
axis x: NOT resolved, tail at 1e-05 and not decaying (100 points: too few, or the data is noisy or not smooth)
```

The fitted `rate` $c$ and `order` $p$ are attributes of the result of each
axis, together with `decay` (`"geometric"`, `"algebraic"` or `"none"`).

```{warning}
The check only sees the values on the grid. A function that oscillates so
fast that the grid takes it for a slow one (aliasing) passes as resolved:
$\sin(20x)$ on 16 periodic points looks exactly like $\sin(4x)$. And an
estimate for an unresolved function is an extrapolation. Repeat the check
on the finer grid.
```

### Each axis on its own

Each axis gets its own verdict. A function can be over-resolved in one
direction and under-resolved in the other:

```python
func = lambda x, y: np.tanh(10 * x) * np.exp(np.sin(y))

grid = Grid(x=chebyshev(40, -1, 1), y=periodic(64))
result = grid.resolution(grid.evaluate(func))
print(result)
```

```text
Resolution at tol=1e-08: NOT resolved
  axis x: NOT resolved, tail at 2e-03; about 159 points needed
  axis y: resolved, 20 of 64 points needed
```

`suggested_grid` is a grid of the same type with the suggested numbers of
points (`suggested_shape`):

```python
better = result.suggested_grid
better.shape                                       # (159, 20)
print(better.resolution(better.evaluate(func)))
```

```text
Resolution at tol=1e-08: resolved
  axis x: resolved, 132 of 159 points needed
  axis y: resolved, 20 of 20 points needed
```

The suggestion follows `needed`, with some caution: an axis is never
enlarged by more than a factor of four in one step, and it is doubled if
the spectrum shows no decay. Axes that are discretized with finite
differences cannot be assessed and keep their number of points; the
report says so. A grid without any spectral axis raises an error.

The check works on curvilinear grids as well, including axes through the
origin and the poles:

```python
ball = SphericalGrid(radial(10), colatitude(14), periodic(24))
X, Y, Z = ball.cartesian_coords
print(ball.resolution(np.exp(X + 0.5 * Z) * np.cos(Y), tol=1e-6))
```

```text
Resolution at tol=1e-06: resolved
  axis r: resolved, 5 of 10 points needed
  axis theta: resolved, 10 of 14 points needed
  axis phi: resolved, 22 of 24 points needed
```

### Plotting the spectra

`result.plot()` draws the spectrum of each axis on a logarithmic scale,
with the tolerance as a dashed line (requires matplotlib):

```python
ax = result.plot()
```

## Filtering

`grid.filter(f)` multiplies the coefficient of each mode $k$ by a factor
$\sigma(k)$ and transforms back. Two filters are available.

**Low-pass truncation** keeps the lowest modes and removes the others.
`keep` is the fraction of the modes that is kept on each axis, or their
number:

```python
grid = Grid(x=periodic(64))
x = grid.coords
f = np.cos(3 * x) + 0.2 * np.cos(25 * x)

np.allclose(grid.filter(f), np.cos(3 * x))             # True: default keep=2/3, modes 0 to 21
np.allclose(grid.filter(f, keep=0.9), f)               # True: modes 0 to 28
np.allclose(grid.filter(f, keep=3), 0)                 # True: only modes 0, 1 and 2
```

The **exponential filter** damps the modes smoothly,

$$
\sigma(k) = \exp\bigl(-\alpha\, (k / k_{\max})^p\bigr),
$$

by default with $\alpha = 36$ and $p = 8$: the lowest third of the modes
is practically untouched ($\sigma > 0.99$ for $k < 0.36\, k_{\max}$), and
the highest mode is damped to round-off level.

```python
smooth = grid.filter(f, "exponential")                     # alpha=36, order=8
gentle = grid.filter(f, "exponential", alpha=4, order=4)

abs(grid.coefficients(smooth))[[3, 25]]                    # [0.5, 0.0007]: 0.5 and 0.1 before
```

`axis` restricts a filter to one axis or several; by default all spectral
axes are filtered, and finite-difference axes are left alone.

```{warning}
Filtering changes the function everywhere, in general also at the
boundary points of non-periodic axes. Apply boundary conditions after
filtering, not before.
```

### Aliasing and the 2/3 rule

The product of two functions contains modes up to the sum of their
highest mode numbers. Modes that the grid cannot represent do not vanish:
they reappear as lower modes and spoil them (aliasing). In a time
stepping scheme for a nonlinear equation this error accumulates and can
make the computation unstable.

The 2/3 rule removes it. If both factors only contain the lowest two
thirds of the modes, the aliased part of their product lands in the upper
third, where a second truncation removes it. `grid.filter` with its
default `keep=2/3` does exactly this:

```python
grid = Grid(x=periodic(32))
fine = Grid(x=periodic(128))            # fine enough to hold the product
rng = np.random.default_rng(1)

u = grid.filter(rng.standard_normal(32))
v = grid.filter(rng.standard_normal(32))

# the exact product, truncated to the modes that are kept
exact = grid.filter(fine.resample(grid.resample(u, fine) * grid.resample(v, fine), grid))

np.abs(u * v - exact).max()                 # 0.9: the product on the grid is aliased
np.abs(grid.filter(u * v) - exact).max()    # 4e-16: filtering it removes the aliasing error
```

The same holds on Chebyshev axes. In practice a smooth filter is often
enough: the exponential filter does not remove the aliasing error
completely, but it keeps more of the spectrum.

## Relation to convergence studies and adaptive refinement

[`convergence_study`](convergence.md) and [`adapt`](amr.md) measure the
error by computing the same quantity on several grids and comparing. This
works for every axis type and for every quantity, but it needs the
function (or the solver) again and again.

The resolution check needs one array on one grid and costs a few fast
transforms. It answers a narrower question, namely whether the *given
function* is represented accurately on its grid, not whether a
differential equation was solved accurately, and it needs spectral axes.
Where it applies, it finds the resolution of each axis in a few steps:

```python
def func(grid):
    X, Y = grid.meshed_coords
    return np.exp(-40 * X**2) / (1.5 - np.cos(Y))

grid = Grid(x=chebyshev(16, -1, 1), y=periodic(8))
for step in range(10):
    result = grid.resolution(func(grid), tol=1e-6)
    print(grid.shape, "resolved" if result.resolved else "not resolved")
    if result.resolved:
        break
    grid = result.suggested_grid
```

```text
(16, 8) not resolved
(32, 16) not resolved
(70, 38) resolved
```

A typical use is a check at the end of a computation: solve the problem,
then ask whether the solution is resolved on the grid it was computed on.

```python
grid = Grid(x=chebyshev(40, 0, 1))
u = grid.laplacian_operator().solve(lambda x: np.exp(4 * x), bcs=0)     # u'' = exp(4x), u(0) = u(1) = 0

grid.is_resolved(u, tol=1e-10)             # True
grid.resolution(u, tol=1e-10)["x"].needed  # 16
```
