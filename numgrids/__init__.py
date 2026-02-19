__version__ = "0.3.0"

from .grids import Grid, MultiGrid
from .api import Diff, AxisType, create_axis, SphericalGrid, CylindricalGrid, PolarGrid, diff, interpolate, integrate
from .interpol import Interpolator
from .integration import Integral
from .boundary import BoundaryFace, DirichletBC, NeumannBC, RobinBC, apply_bcs
from .io import save_grid, load_grid
from .amr import ErrorEstimator, AdaptationResult, adapt, estimate_error

# Backward compatibility alias
Axis = create_axis
