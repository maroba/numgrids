"""Tests for the geometric multigrid solver (MultiGrid.solve, numgrids.multigrid)."""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse.linalg

from numgrids import Diff, Grid, Neumann, Robin, chebyshev, equidistant, logarithmic, periodic
from numgrids.boundary import DirichletBC
from numgrids.grids import MultiGrid
from numgrids.multigrid import MultiGridError, MultiGridInfo, MultiGridSolver


def laplacian(acc=4):
    return lambda grid: grid.laplacian_operator(acc=acc)


def square(n, d=2):
    return MultiGrid(*[equidistant(n, 0, 1)] * d)


class TestExactSolutions:

    def test_1d(self):
        mg = MultiGrid(x=equidistant(257, 0, 1))
        x = mg.levels[0].coords
        exact = np.sin(np.pi * x) + x
        u = mg.solve(laplacian(), lambda x: -np.pi ** 2 * np.sin(np.pi * x), bcs={"x_low": 0, "x_high": 1})
        npt.assert_allclose(u, exact, atol=1e-8)

    @pytest.mark.parametrize("acc, atol", [(2, 1e-3), (4, 1e-5)])
    def test_2d(self, acc, atol):
        mg = square(65)
        X, Y = mg.levels[0].meshed_coords
        exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
        u = mg.solve(laplacian(acc), -5 * np.pi ** 2 * exact, bcs=0)
        npt.assert_allclose(u, exact, atol=atol)

    def test_3d(self):
        mg = square(33, d=3)
        X, Y, Z = mg.levels[0].meshed_coords
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
        u, info = mg.solve(laplacian(), -3 * np.pi ** 2 * exact, bcs=0, return_info=True)
        npt.assert_allclose(u, exact, atol=1e-5)
        assert info.iterations < 40

    def test_polynomial_is_reproduced_exactly(self):
        # Finite differences are exact for u = x^2 + y^2, so only the solver error remains.
        mg = square(33)
        exact = lambda x, y: x ** 2 + y ** 2
        u = mg.solve(laplacian(2), 4.0, bcs=lambda coords: exact(*coords), tol=1e-12)
        npt.assert_allclose(u, mg.levels[0].evaluate(exact), atol=1e-10)


class TestAgreementWithDirectSolver:

    def check(self, mg, operator, rhs, bcs, max_iterations, **options):
        u, info = mg.solve(operator, rhs, bcs, tol=1e-10, return_info=True, **options)
        direct = operator(mg.levels[0]).solve(rhs, bcs)
        npt.assert_allclose(u, direct, atol=1e-7 * max(1.0, np.abs(direct).max()))
        assert info.iterations <= max_iterations
        return info

    @pytest.mark.parametrize("acc, max_iterations", [(2, 12), (4, 32)])
    def test_poisson_2d(self, acc, max_iterations):
        self.check(square(65), laplacian(acc), lambda x, y: np.exp(x) * np.cos(3 * y), 0, max_iterations)

    def test_poisson_3d(self):
        self.check(square(17, d=3), laplacian(), lambda x, y, z: x * np.sin(3 * y) + z, 0, 55)

    def test_variable_coefficients(self):
        # -div(k grad u) + c u with k = 1 + x y / 2, c = 1 + x^2
        def operator(grid):
            x, y = grid.meshed_coords
            k = 1 + 0.5 * x * y
            return -(k * grid.laplacian_operator() + 0.5 * y * grid.diff_operator(1, 0)
                     + 0.5 * x * grid.diff_operator(1, 1)) + (1 + x ** 2)

        mg = MultiGrid(x=equidistant(65, 0, 1), y=equidistant(65, 0, 1))
        self.check(mg, operator, lambda x, y: np.sin(4 * x) + y, {"all": 0, "x_low": 1.0}, 45)

    def test_non_nested_sizes(self):
        mg = MultiGrid(equidistant(64, 0, 1), equidistant(48, 0, 0.75))
        info = self.check(mg, laplacian(), 1.0, 0, 35)
        assert info.shapes == ((64, 48), (32, 24), (16, 12), (8, 6))

    def test_anisotropic_box(self):
        # Same number of points on a 2:1 box: the grid spacings differ by a factor of two.
        mg = MultiGrid(equidistant(65, 0, 2), equidistant(65, 0, 1))
        self.check(mg, laplacian(), lambda x, y: np.sin(x + y), 0, 45)

    def test_dirichlet_values_as_callable(self):
        g = lambda coords: np.sin(coords[0]) * np.exp(coords[1])
        info = self.check(square(65), laplacian(), 0.0, g, 35)
        assert info.iterations > 0

    def test_dirichlet_values_as_array_on_the_finest_face(self):
        mg = square(33)
        y = mg.levels[0].get_axis(1).coords
        bcs = {"all": 0, "0_low": np.sin(np.pi * y)}
        self.check(mg, laplacian(), 0.0, bcs, 35)

    @pytest.mark.parametrize("acc", [2, 4])
    def test_neumann_on_one_face(self, acc):
        bcs = {"all": 0, "0_high": Neumann(lambda coords: np.cos(coords[1]))}
        self.check(square(65), laplacian(acc), lambda x, y: x - y, bcs, 40)

    def test_neumann_on_most_faces_3d(self):
        self.check(square(17, d=3), laplacian(), 1.0, {"all": Neumann(0.5), "0_low": 1.0}, 55)

    def test_robin(self):
        self.check(square(65), laplacian(), 1.0, Robin(1.0, 1.0, 0.3), 35)

    def test_bound_boundary_conditions(self):
        mg = square(33)
        bcs = [DirichletBC(face, 0.5) for face in mg.levels[0].faces.values()]
        self.check(mg, laplacian(), 1.0, bcs, 32)

    def test_logarithmic_axis(self):
        mg = MultiGrid(r=logarithmic(129, 1, 10))
        self.check(mg, lambda grid: grid.diff_operator(2), lambda r: 1 / r, {"r_low": 0, "r_high": 1}, 25)

    def test_operator_built_from_diff_objects(self):
        self.check(square(33), lambda grid: Diff(grid, 2, 0) + Diff(grid, 2, 1), 1.0, 0, 32)

    def test_w_cycle_and_smoothing_steps(self):
        self.check(square(65), laplacian(), 1.0, 0, 30, cycle="W", pre=1, post=3)

    def test_without_low_order_cycle(self):
        info = self.check(square(65), laplacian(2), 1.0, 0, 12, low_order=False)
        assert info.shapes[-1] == (9, 9)

    def test_complex_right_hand_side(self):
        mg = square(33)
        f = mg.levels[0].evaluate(lambda x, y: np.sin(3 * x) + 1j * np.cos(2 * y))
        u = mg.solve(laplacian(), f, bcs=1j, tol=1e-10)
        assert np.iscomplexobj(u)
        npt.assert_allclose(u, laplacian()(mg.levels[0]).solve(f, 1j), atol=1e-7)

    def test_grid_too_small_to_coarsen_is_solved_directly(self):
        mg = MultiGrid(equidistant(9, 0, 1), equidistant(8, 0, 1))
        info = self.check(mg, laplacian(), 1.0, 0, 1)
        assert info.shapes == ((9, 8),)


class TestConvergence:

    @pytest.mark.parametrize("acc, bound", [(2, 12), (4, 30)])
    def test_iterations_do_not_grow_with_the_grid(self, acc, bound):
        iterations = []
        for n in (33, 65, 129):
            _, info = square(n).solve(laplacian(acc), 1.0, bcs=0, tol=1e-10, return_info=True)
            iterations.append(info.iterations)
        assert max(iterations) <= bound
        assert max(iterations) - min(iterations) <= 4

    def test_plain_cycles_have_grid_independent_convergence_factor(self):
        factors = []
        for n in (33, 65, 129):
            _, info = square(n).solve(laplacian(2), 1.0, bcs=0, method="plain", tol=1e-10, return_info=True)
            factors.append(info.convergence_factor)
        assert max(factors) < 0.25
        assert max(factors) - min(factors) < 0.05

    def test_info(self):
        u, info = square(33).solve(laplacian(), 1.0, bcs=0, return_info=True)
        assert isinstance(info, MultiGridInfo)
        assert info.converged
        assert len(info.residuals) == info.iterations + 1
        assert info.residuals[0] == pytest.approx(1.0)
        assert info.residuals[-1] <= 1e-8
        assert all(b <= a * (1 + 1e-12) for a, b in zip(info.residuals, info.residuals[1:]))
        assert 0 < info.convergence_factor < 0.6
        assert info.shapes == ((33, 33), (17, 17), (9, 9), (5, 5))

    def test_residual_is_that_of_the_linear_system(self):
        mg = square(33)
        f = lambda x, y: np.sin(5 * x * y)
        u, info = mg.solve(laplacian(), f, bcs=1.0, tol=1e-6, return_info=True)
        L, b = laplacian()(mg.levels[0]).with_bcs(f, 1.0)
        # Reference: the right-hand side once the boundary values are moved there.
        lifted = np.where(mg.levels[0].boundary, 1.0, 0.0).ravel()
        residual = np.linalg.norm(b - L @ u.ravel()) / np.linalg.norm(b - L @ lifted)
        assert residual <= 1e-6
        assert info.residuals[-1] == pytest.approx(residual, rel=1e-6)

    def test_tolerance_means_the_same_for_boundary_driven_problems(self):
        # Laplace's equation: only the boundary values drive the solution.
        iterations = []
        for n in (33, 129):
            _, info = square(n).solve(laplacian(2), 0.0, bcs=lambda coords: coords[0] ** 2, return_info=True)
            assert info.residuals[0] == pytest.approx(1.0)
            iterations.append(info.iterations)
        assert max(iterations) <= 10

    def test_initial_guess(self):
        mg = square(33)
        direct = laplacian()(mg.levels[0]).solve(1.0, 0)
        u, info = mg.solve(laplacian(), 1.0, bcs=0, u0=direct, return_info=True)
        assert info.iterations == 0
        npt.assert_allclose(u, direct, rtol=0, atol=1e-12)
        _, info_guess = mg.solve(laplacian(), 1.0, bcs=0, u0=direct * 1.001, return_info=True)
        _, info_zero = mg.solve(laplacian(), 1.0, bcs=0, return_info=True)
        assert info_guess.iterations < info_zero.iterations

    def test_min_size_limits_the_coarsest_level(self):
        mg = MultiGrid(equidistant(65, 0, 1), equidistant(65, 0, 1), min_size=17)
        _, info = mg.solve(laplacian(), 1.0, bcs=0, return_info=True)
        assert info.shapes == ((65, 65), (33, 33), (17, 17))

    def test_homogeneous_problem(self):
        u, info = square(33).solve(laplacian(), 0.0, bcs=0, return_info=True)
        npt.assert_array_equal(u, 0.0)
        assert info.iterations == 0


class TestSolverObject:

    def test_reuse_for_several_right_hand_sides(self):
        mg = square(65)
        solver = mg.solver(laplacian(), bcs=0)
        assert isinstance(solver, MultiGridSolver)
        f1 = mg.levels[0].evaluate(lambda x, y: np.sin(3 * x))
        f2 = mg.levels[0].evaluate(lambda x, y: y ** 2)
        u1, u2, u12 = (solver.solve(f, tol=1e-11) for f in (f1, f2, f1 + 2 * f2))
        npt.assert_allclose(u12, u1 + 2 * u2, atol=1e-9)
        assert "(65, 65)" in repr(solver)

    def test_right_hand_side_forms(self):
        mg = square(33)
        solver = MultiGridSolver(mg, laplacian(), bcs=0)
        npt.assert_array_equal(solver.solve(2.0), solver.solve(np.full((33, 33), 2.0)))
        npt.assert_array_equal(solver.solve(2.0), solver.solve(lambda x, y: 2.0))

    def test_named_axes(self):
        mg = MultiGrid(x=equidistant(33, 0, 1), y=equidistant(33, 0, 1))
        assert [ax.name for ax in mg.levels[-1].axes] == ["x", "y"]
        u = mg.solve(lambda grid: grid.diff_operator(2, "x") + grid.diff_operator(2, "y"), 1.0,
                     bcs={"all": 0, "y_high": 1.0})
        npt.assert_allclose(u[:, -1], 1.0)
        npt.assert_allclose(u[0, :-1], 0.0)

    def test_as_preconditioner_for_scipy(self):
        mg = square(65)
        op = laplacian()
        L, b = op(mg.levels[0]).with_bcs(lambda x, y: x + y, 0)
        M = mg.solver(op, bcs=0).as_preconditioner()
        count = []
        x, flag = scipy.sparse.linalg.bicgstab(L, b, M=M, callback=lambda xk: count.append(1))
        assert flag == 0
        assert len(count) < 30
        npt.assert_allclose(x, scipy.sparse.linalg.spsolve(L, b), atol=1e-4)


class TestErrors:

    def test_no_convergence_raises_with_last_iterate(self):
        mg = square(65)
        with pytest.raises(MultiGridError, match="did not converge in 3 iterations") as excinfo:
            mg.solve(laplacian(), 1.0, bcs=0, maxiter=3)
        err = excinfo.value
        assert err.u.shape == (65, 65)
        assert len(err.residuals) == 4
        assert err.residuals[-1] < err.residuals[0]

    def test_plain_cycles_no_convergence(self):
        with pytest.raises(MultiGridError) as excinfo:
            square(33).solve(laplacian(), 1.0, bcs=0, method="plain", maxiter=5)
        assert len(excinfo.value.residuals) == 6

    def test_unattainable_tolerance_stops_early(self):
        with pytest.raises(MultiGridError, match="rounding errors") as excinfo:
            square(65).solve(laplacian(), 1.0, bcs=0, tol=1e-17, maxiter=1000)
        assert len(excinfo.value.residuals) < 300
        assert excinfo.value.residuals[-1] < 1e-11

    @pytest.mark.parametrize("axis", [chebyshev(33, 0, 1), periodic(32, 0, 1)])
    def test_spectral_axes_are_rejected(self, axis):
        mg = MultiGrid(equidistant(33, 0, 1), axis)
        with pytest.raises(TypeError, match="Operator.solve"):
            mg.solve(laplacian(), 1.0, bcs=0)

    def test_operator_must_be_callable(self):
        mg = square(33)
        with pytest.raises(TypeError, match="function that builds the operator"):
            mg.solve(mg.levels[0].laplacian_operator(), 1.0, bcs=0)

    def test_operator_must_use_the_grid_passed_in(self):
        mg = square(33)
        fixed = mg.levels[0].laplacian_operator()
        with pytest.raises(ValueError, match="Build the operator from the grid passed in"):
            mg.solve(lambda grid: fixed, 1.0, bcs=0, low_order=False)

    def test_operator_must_return_an_operator(self):
        with pytest.raises(TypeError, match="must return an Operator"):
            square(33).solve(lambda grid: "laplacian", 1.0, bcs=0)

    def test_every_face_needs_a_condition(self):
        mg = MultiGrid(x=equidistant(33, 0, 1), y=equidistant(33, 0, 1))
        with pytest.raises(ValueError, match="Every boundary face needs a boundary condition"):
            mg.solve(laplacian(), 1.0, bcs={"x_low": 0, "x_high": 0})
        with pytest.raises(ValueError, match="Every boundary face"):
            mg.solve(laplacian(), 1.0)

    def test_pure_neumann_problem_is_singular(self):
        with pytest.raises((ValueError, MultiGridError)):
            square(33).solve(laplacian(), 1.0, bcs=Neumann(0))

    def test_operator_without_diagonal(self):
        with pytest.raises(ValueError, match="zero diagonal"):
            square(33).solve(lambda grid: grid.diff_operator(1, 0), 1.0, bcs=0)

    def test_level_too_coarse_for_the_operator(self):
        biharmonic = lambda grid: grid.diff_operator(4, 0) + grid.laplacian_operator()
        with pytest.raises(ValueError, match="larger `min_size`"):
            MultiGrid(equidistant(65, 0, 1)).solve(biharmonic, 1.0, bcs=0)

    def test_invalid_options(self):
        mg = square(33)
        with pytest.raises(ValueError, match="Unknown method"):
            mg.solve(laplacian(), 1.0, bcs=0, method="cg")
        with pytest.raises(ValueError, match="Unknown cycle"):
            mg.solve(laplacian(), 1.0, bcs=0, cycle="F")
        with pytest.raises(ValueError, match="`pre` and `post`"):
            mg.solve(laplacian(), 1.0, bcs=0, pre=0, post=0)

    def test_wrong_shapes(self):
        mg = square(33)
        with pytest.raises(ValueError, match="'rhs' has shape"):
            mg.solve(laplacian(), np.ones((17, 17)), bcs=0)
        with pytest.raises(ValueError, match="'u0' has shape"):
            mg.solve(laplacian(), 1.0, bcs=0, u0=np.ones(33))
        other = Grid(equidistant(17, 0, 1), equidistant(17, 0, 1))
        with pytest.raises(ValueError, match="bound to a grid of shape"):
            mg.solve(laplacian(), 1.0, bcs=[DirichletBC(face) for face in other.faces.values()])
