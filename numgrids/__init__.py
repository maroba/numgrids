"""numgrids -- working with numerical grids made easy."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("numgrids")
except _PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

# Import the ``numgrids.diff`` submodule eagerly: importing it lazily later
# would rebind the package attribute ``numgrids.diff`` from the function
# (imported below) to the module.
from . import diff as _diff_module  # noqa: F401, I001
from .amr import AdaptationResult, ErrorEstimator, adapt, estimate_error
from .api import (
    AxisType,
    Diff,
    chebyshev,
    colatitude,
    create_axis,
    diff,
    equidistant,
    infinite,
    integrate,
    interpolate,
    logarithmic,
    periodic,
    radial,
    semi_infinite,
)
from .axes import (
    ChebyshevAxis,
    ColatitudeAxis,
    EquidistantAxis,
    InfiniteAxis,
    LogAxis,
    RadialAxis,
    SemiInfiniteAxis,
)
from .boundary import (
    BoundaryFace,
    Dirichlet,
    DirichletBC,
    Neumann,
    NeumannBC,
    Robin,
    RobinBC,
    TimeDependent,
    apply_bcs,
    resolve_bcs,
)
from .convergence import ConvergenceResult, convergence_study
from .curvilinear import (
    CurvilinearGrid,
    CylindricalGrid,
    EllipticCylindricalGrid,
    OblateSpheroidalGrid,
    PolarGrid,
    ProlateSpheroidalGrid,
    SphericalGrid,
    SphericalSurfaceGrid,
)
from .evolution import EvolutionResult, evolve
from .grids import Grid, MultiGrid
from .integration import Integral
from .interpol import Interpolator
from .io import from_xarray, load_grid, save_grid, to_xarray
from .linalg import LinearSolver
from .multigrid import MultiGridError, MultiGridInfo, MultiGridSolver
from .nonlinear import ContinuationResult, NonlinearSolverError, continuation, solve_nonlinear
from .operators import Operator


def Axis(*args, **kwargs):
    """Deprecated alias of :func:`create_axis`.

    .. deprecated:: 0.5.0
       Use :func:`create_axis` or the short-hand factories
       :func:`chebyshev`, :func:`equidistant`, :func:`periodic` and
       :func:`logarithmic` instead.  The axis base class lives in
       :mod:`numgrids.axes`.
    """
    import warnings
    warnings.warn(
        "numgrids.Axis is deprecated; use numgrids.create_axis (or chebyshev, "
        "equidistant, periodic, logarithmic) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_axis(*args, **kwargs)


__all__ = [
    "AdaptationResult",
    "Axis",
    "AxisType",
    "BoundaryFace",
    "ChebyshevAxis",
    "ColatitudeAxis",
    "ContinuationResult",
    "ConvergenceResult",
    "CurvilinearGrid",
    "CylindricalGrid",
    "Diff",
    "Dirichlet",
    "DirichletBC",
    "EllipticCylindricalGrid",
    "EquidistantAxis",
    "ErrorEstimator",
    "EvolutionResult",
    "Grid",
    "InfiniteAxis",
    "Integral",
    "Interpolator",
    "LinearSolver",
    "LogAxis",
    "MultiGrid",
    "MultiGridError",
    "MultiGridInfo",
    "MultiGridSolver",
    "Neumann",
    "NonlinearSolverError",
    "NeumannBC",
    "OblateSpheroidalGrid",
    "Operator",
    "PolarGrid",
    "ProlateSpheroidalGrid",
    "RadialAxis",
    "Robin",
    "RobinBC",
    "SemiInfiniteAxis",
    "SphericalGrid",
    "SphericalSurfaceGrid",
    "TimeDependent",
    "adapt",
    "apply_bcs",
    "chebyshev",
    "colatitude",
    "continuation",
    "convergence_study",
    "create_axis",
    "diff",
    "equidistant",
    "estimate_error",
    "evolve",
    "from_xarray",
    "infinite",
    "integrate",
    "interpolate",
    "load_grid",
    "logarithmic",
    "periodic",
    "radial",
    "resolve_bcs",
    "save_grid",
    "semi_infinite",
    "solve_nonlinear",
    "to_xarray",
]
