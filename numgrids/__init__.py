__version__ = "0.3.0"

from .grids import Grid, MultiGrid
from .api import Diff, AxisType, create_axis, SphericalGrid, diff, interpolate, integrate
from .interpol import Interpolator
from .integration import Integral

# Backward compatibility alias
Axis = create_axis
