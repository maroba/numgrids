import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse
from scipy.optimize import brentq
from scipy.special import ellipk, jn_zeros

import numgrids.nonlinear
import numgrids.stability
from numgrids import Grid, Neumann, PolarGrid, Robin, chebyshev, equidistant, periodic, radial, solve_nonlinear
from numgrids.nonlinear import (
    Bifurcation,
    ContinuationResult,
    StabilityResult,
    continuation,
    linear_stability,
)

# Fold of the Bratu problem u'' + lam e^u = 0, u(0) = u(1) = 0:
# lam_c = 8 s^2 / cosh(s)^2 with s tanh(s) = 1, and max u = 2 ln cosh(s) there.
BRATU_FOLD = 3.513830719125161
BRATU_FOLD_S = brentq(lambda s: s * np.tanh(s) - 1, 1, 2)


@pytest.fixture
def rod():
    return Grid(x=chebyshev(31, 0, 1))


def bratu(grid):
    """Residual and exact Jacobian of the Bratu problem."""
    D2 = grid.diff_operator(2)
    return (lambda u, lam: D2(u) + lam * np.exp(u)), (lambda u, lam: D2 + lam * np.exp(u))


def same_up_to_sign(v, w, atol):
    """Compare two normalized eigenfunctions, which are determined up to their sign."""
    npt.assert_allclose(v * np.sign(np.vdot(w, v).real), w, atol=atol)


class TestLinearProblems:
    """u_t = u'' + a u with known spectra."""

    def test_dirichlet(self, rod):
        # Eigenvalues a - (n pi)^2, eigenfunctions sin(n pi x).
        x = rod.coords
        n = np.arange(1, 6)
        for a, n_unstable in [(9.0, 0), (10.5, 1), (40.0, 2)]:
            result = linear_stability(rod, lambda u, a=a: rod.diff(u, 2) + a * u, 0, bcs=0, k=5)
            assert isinstance(result, StabilityResult)
            assert result.eigenvalues.dtype == float
            npt.assert_allclose(result.eigenvalues, a - (n * np.pi) ** 2, atol=1e-8)
            assert result.stable == (n_unstable == 0)
            assert result.n_unstable == n_unstable
            npt.assert_allclose(result.growth_rate, a - np.pi ** 2, atol=1e-8)
            assert result.eigenfunctions.shape == (5, 31)
            for j in range(5):
                mode = np.sin(n[j] * np.pi * x)
                same_up_to_sign(result.eigenfunctions[j], mode / np.linalg.norm(mode), atol=1e-9)

    def test_stability_flips_at_the_first_eigenvalue(self, rod):
        D2 = rod.diff_operator(2)
        for a in (np.pi ** 2 - 1e-4, np.pi ** 2 + 1e-4):
            result = linear_stability(rod, lambda u, a=a: D2(u) + a * u, 0, bcs=0, jacobian=lambda u, a=a: D2 + a)
            assert result.stable == (a < np.pi ** 2)
            npt.assert_allclose(result.growth_rate, a - np.pi ** 2, atol=1e-8)

    def test_neumann(self, rod):
        # Eigenvalues a - (n pi)^2 with n = 0, 1, ...: the uniform mode decides.
        x = rod.coords
        n = np.arange(4)
        for a in (-1.0, 0.0, 1.0):
            result = linear_stability(rod, lambda u, a=a: rod.diff(u, 2) + a * u, 0, bcs=Neumann(0), k=4)
            npt.assert_allclose(result.eigenvalues, a - (n * np.pi) ** 2, atol=1e-8)
            assert result.stable == (a < 0)
            assert result.n_unstable == (1 if a > 0 else 0)
            npt.assert_allclose(result.eigenfunctions[0], 1 / np.sqrt(31), atol=1e-9)
            mode = np.cos(np.pi * x)
            same_up_to_sign(result.eigenfunctions[1], mode / np.linalg.norm(mode), atol=1e-9)

    def test_marginal_state(self, rod):
        result = linear_stability(rod, lambda u: rod.diff(u, 2), 0, bcs=Neumann(0))
        assert not result.stable
        assert result.n_unstable == 0
        assert abs(result.growth_rate) < 1e-8
        assert repr(result).startswith("StabilityResult(marginal")

    def test_robin(self, rod):
        # u(0) = 0, u'(1) + u(1) = 0: eigenvalues a - mu^2 with mu cos(mu) + sin(mu) = 0.
        mu = np.array([brentq(lambda m: m * np.cos(m) + np.sin(m), (n - 0.5) * np.pi, n * np.pi)
                       for n in (1, 2, 3)])
        bcs = {"x_low": 0, "x_high": Robin(1, 1, 0)}
        for a in (4.0, 4.2):
            result = linear_stability(rod, lambda u, a=a: rod.diff(u, 2) + a * u, 0, bcs=bcs, k=3)
            npt.assert_allclose(result.eigenvalues, a - mu ** 2, atol=1e-8)
            assert result.stable == (a < mu[0] ** 2)
        # The eigenfunctions satisfy the homogeneous boundary conditions.
        for v in result.eigenfunctions:
            npt.assert_allclose(v[0], 0, atol=1e-10)
            npt.assert_allclose(rod.diff(v, 1)[-1] + v[-1], 0, atol=1e-9)

    def test_inhomogeneous_conditions_do_not_matter(self, rod):
        # Perturbations satisfy the homogeneous conditions, whatever the values of the solution.
        x = rod.coords
        F = lambda u: rod.diff(u, 2) + 5 * u  # noqa: E731
        bcs = {"x_low": 1, "x_high": 2}
        u = solve_nonlinear(rod, F, 1 + x, bcs=bcs)
        result = linear_stability(rod, F, u, bcs=bcs, k=3)  # finite-difference Jacobian
        npt.assert_allclose(result.eigenvalues, 5 - (np.arange(1, 4) * np.pi) ** 2, atol=1e-5)
        npt.assert_allclose(result.eigenfunctions[:, [0, -1]], 0, atol=1e-12)

    def test_sign(self, rod):
        # A residual written as -F belongs to u_t = -F(u) with sign=-1.
        plus = linear_stability(rod, lambda u: rod.diff(u, 2) + 12 * u, 0, bcs=0)
        minus = linear_stability(rod, lambda u: -rod.diff(u, 2) - 12 * u, 0, bcs=0, sign=-1)
        npt.assert_allclose(minus.eigenvalues, plus.eigenvalues, atol=1e-9)
        assert not minus.stable and minus.n_unstable == 1
        # Without the switch, the stability is that of the reversed evolution.
        wrong = linear_stability(rod, lambda u: -rod.diff(u, 2) - 12 * u, 0, bcs=0)
        assert wrong.n_unstable == 6

    def test_two_dimensions_with_degenerate_eigenvalues(self):
        grid = Grid(x=chebyshev(13, 0, 1), y=chebyshev(13, 0, 1))
        L = grid.laplacian_operator()
        result = linear_stability(grid, lambda u: L(u) + 30 * u, 0, bcs=0, jacobian=lambda u: L + 30, k=4)
        npt.assert_allclose(result.eigenvalues, 30 - np.pi ** 2 * np.array([2, 5, 5, 8]), atol=1e-5)
        assert result.eigenvalues.dtype == float
        assert result.n_unstable == 1
        v = result.eigenfunctions
        assert v.shape == (4, 13, 13) and v.dtype == float
        # The two modes of the double eigenvalue form a real orthonormal basis.
        npt.assert_allclose(np.sum(v[1] * v[2]), 0, atol=1e-8)
        for j in range(4):
            residual = (L(v[j]) + 30 * v[j] - result.eigenvalues[j] * v[j])[1:-1, 1:-1]
            npt.assert_allclose(residual, 0, atol=1e-6)

    def test_disk(self):
        # On the unit disk with u = 0 on the rim: a - j^2 with the zeros j of the Bessel functions.
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        L = disk.laplacian_operator()
        result = linear_stability(disk, lambda u: L(u) + 10 * u, 0, bcs=0, jacobian=lambda u: L + 10, k=6)
        j0, j1, j2 = jn_zeros(0, 2), jn_zeros(1, 1), jn_zeros(2, 1)
        exact = 10 - np.array([j0[0], j1[0], j1[0], j2[0], j2[0], j0[1]]) ** 2
        npt.assert_allclose(result.eigenvalues, exact, atol=1e-6)
        assert result.n_unstable == 1
        assert result.eigenfunctions.shape == (6, 12, 16)
        npt.assert_allclose(result.eigenfunctions[:, -1, :], 0, atol=1e-12)


class TestBratu:

    def test_lower_solution_is_stable_and_upper_unstable(self, rod):
        F, J = bratu(rod)
        x = rod.coords
        lower = solve_nonlinear(rod, lambda u: F(u, 1.0), 0, bcs=0)
        upper = solve_nonlinear(rod, lambda u: F(u, 1.0), 4 * np.sin(np.pi * x), bcs=0)
        result = linear_stability(rod, F, lower, bcs=0, jacobian=J, param=1.0)
        assert result.stable and result.n_unstable == 0
        assert repr(result).startswith("StabilityResult(stable, growth rate -8.7")
        result = linear_stability(rod, F, upper, bcs=0, jacobian=J, param=1.0)
        assert not result.stable and result.n_unstable == 1
        assert np.all(result.eigenvalues[1:] < 0)
        assert "1 unstable mode," in repr(result)
        # The growing mode has no node.
        assert np.all(result.eigenfunctions[0][1:-1] > 0)
        # The same with a residual of one argument and a finite-difference Jacobian.
        fd = linear_stability(rod, lambda u: F(u, 1.0), upper, bcs=0)
        npt.assert_allclose(fd.eigenvalues, result.eigenvalues, atol=1e-6)

    def test_stability_changes_at_the_fold(self, rod):
        F, J = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=0.05, max_steps=150, stability=4)
        k, = branch.folds
        assert branch.eigenvalues.shape == (151, 4) and branch.eigenvalues.dtype == float
        assert branch.stable.shape == (151,) and branch.stable.dtype == bool
        # Stable up to the fold, exactly one unstable mode behind it.
        fold, = branch.bifurcations
        assert fold.index in (k - 1, k)
        npt.assert_array_equal(branch.stable, np.arange(151) <= fold.index)
        npt.assert_array_equal(branch.n_unstable, np.where(np.arange(151) <= fold.index, 0, 1))
        assert np.all(np.diff(branch.eigenvalues[:, 0]) > 0)
        assert np.all(branch.eigenvalues[:, 1:] < 0)
        # The first point is u = 0 at lam = 0.
        npt.assert_allclose(branch.eigenvalues[0], -(np.arange(1, 5) * np.pi) ** 2, atol=1e-8)

        # The fold is located on the branch, where the leading eigenvalue is zero.
        assert isinstance(fold, Bifurcation)
        assert fold.kind == "fold" and fold.refined and fold.multiplicity == 1
        npt.assert_allclose(fold.param, BRATU_FOLD, atol=1e-9)
        npt.assert_allclose(branch.fold_params, [BRATU_FOLD], atol=1e-9)
        assert branch.fold_u.shape == (1, 31)
        npt.assert_array_equal(branch.fold_u[0], fold.u)
        npt.assert_allclose(fold.u.max(), 2 * np.log(np.cosh(BRATU_FOLD_S)), atol=1e-7)
        at_fold = linear_stability(rod, F, fold.u, bcs=0, jacobian=J, param=fold.param)
        npt.assert_allclose(at_fold.growth_rate, 0, atol=1e-6)
        # The tangent at the fold is the critical mode.
        same_up_to_sign(fold.mode, at_fold.eigenfunctions[0], atol=1e-6)

        assert repr(branch).startswith("ContinuationResult(151 points, parameter from 0 to 0.4")
        assert repr(branch).endswith(f", 1 fold(s), {fold.index + 1} stable point(s), success=True)")
        assert repr(fold) == (f"Bifurcation(fold at param = 3.513830719, between the points {fold.index} "
                              f"and {fold.index + 1})")

    @pytest.mark.parametrize("ds", [0.05, 0.3, 1.0])
    def test_fold_accuracy_does_not_depend_on_the_step(self, rod, ds):
        F, J = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=ds, max_steps=int(6 / ds))
        assert branch.eigenvalues is None and branch.stable is None and branch.n_unstable is None
        assert len(branch.folds) == 1
        npt.assert_allclose(branch.fold_params, [BRATU_FOLD], atol=1e-9)
        fold, = branch.bifurcations
        assert fold.kind == "fold" and fold.refined
        npt.assert_allclose(fold.u.max(), 2 * np.log(np.cosh(BRATU_FOLD_S)), atol=1e-7)

    def test_fold_with_finite_difference_jacobian(self, rod):
        F, _ = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, ds=0.2, max_steps=25, stability=2)
        npt.assert_allclose(branch.fold_params, [BRATU_FOLD], atol=1e-8)
        fold, = branch.bifurcations
        npt.assert_array_equal(branch.n_unstable, np.where(np.arange(26) <= fold.index, 0, 1))

    def test_estimate_without_refinement(self, rod):
        F, J = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=0.05, max_steps=100, refine=False)
        fold, = branch.bifurcations
        assert not fold.refined and "interpolated" in repr(fold)
        npt.assert_allclose(branch.fold_params, [BRATU_FOLD], atol=1e-3)
        assert abs(branch.fold_params[0] - BRATU_FOLD) > 1e-9
        npt.assert_array_equal(branch.fold_u[0], branch.u[branch.folds[0]])
        # A result without bifurcation records (e.g. assembled by hand) gives the same estimate.
        plain = ContinuationResult(branch.params, branch.u, True, "")
        assert plain.bifurcations is None and plain.eigenvalues is None
        npt.assert_allclose(plain.fold_params, branch.fold_params, atol=1e-12)
        npt.assert_array_equal(plain.fold_u, branch.fold_u)
        assert repr(plain) == repr(branch)

    def test_param_range_and_callback_are_not_affected(self, rod):
        F, J = bratu(rod)
        seen = []
        branch = continuation(rod, F, 0, 2, bcs=0, jacobian=J, param_range=(1, 4), max_steps=200, stability=3,
                              callback=lambda u, lam: seen.append(lam))
        assert branch.params[-1] == 1.0
        npt.assert_array_equal(seen, branch.params)
        npt.assert_allclose(branch.fold_params, [BRATU_FOLD], atol=1e-9)
        assert branch.stable[0] and not branch.stable[-1]
        assert branch.eigenvalues.shape == (len(branch), 3)

    def test_fold_next_to_the_end_of_param_range(self, rod):
        # Three points: the start on the lower branch, one near the fold, and the end of
        # the range on the upper branch (which is not the result of a continuation step).
        F, J = bratu(rod)
        branch = continuation(rod, F, 1, 3.51, bcs=0, jacobian=J, ds=0.05, param_range=(3.51, 4), stability=2)
        npt.assert_array_equal(branch.params[[0, -1]], [3.51, 3.51])
        assert len(branch) == 3
        assert branch.stable[0] and not branch.stable[-1]
        fold, = branch.bifurcations
        assert fold.refined
        npt.assert_allclose(fold.param, BRATU_FOLD, atol=1e-9)

    def test_two_dimensions(self):
        # On the unit square, the fold lies at lam_c = 6.808124423.
        grid = Grid(x=chebyshev(11, 0, 1), y=chebyshev(11, 0, 1))
        L = grid.laplacian_operator()
        branch = continuation(grid, lambda u, lam: L(u) + lam * np.exp(u), 0, 0, bcs=0,
                              jacobian=lambda u, lam: L + lam * np.exp(u), ds=0.2, max_steps=40, stability=3)
        fold, = branch.bifurcations
        npt.assert_allclose(fold.param, 6.808124423, atol=1e-5)
        assert branch.fold_u.shape == (1, 11, 11)
        assert branch.eigenvalues.dtype == float
        npt.assert_array_equal(branch.n_unstable, np.where(np.arange(41) <= fold.index, 0, 1))

    def test_disk(self):
        # On the unit disk, the fold of the radially symmetric solutions lies at lam_c = 2.
        disk = PolarGrid(radial(10, 1.0), periodic(8))
        L = disk.laplacian_operator()
        branch = continuation(disk, lambda u, lam: L(u) + lam * np.exp(u), 0, 0, bcs=0,
                              jacobian=lambda u, lam: L + lam * np.exp(u), ds=0.2, max_steps=25, stability=3)
        fold, = branch.bifurcations
        npt.assert_allclose(fold.param, 2.0, atol=1e-6)
        assert branch.stable[0] and not branch.stable[-1]
        assert np.all(branch.n_unstable <= 1)


class TestBranchPoints:

    def test_trivial_branch_of_a_pitchfork(self, rod):
        # u'' + lam (u - u^3) = 0, u(0) = u(1) = 0: u = 0 loses stability at lam = pi^2,
        # further modes become unstable at (n pi)^2.
        D2 = rod.diff_operator(2)
        x = rod.coords
        F = lambda u, lam: D2(u) + lam * (u - u ** 3)  # noqa: E731
        J = lambda u, lam: D2 + lam * (1 - 3 * u ** 2)  # noqa: E731
        for jacobian, atol in [(J, 1e-8), (None, 1e-6)]:
            branch = continuation(rod, F, 0, 0, bcs=0, jacobian=jacobian, ds=1.0, param_range=(0, 100),
                                  stability=5)
            assert len(branch.folds) == 0 and len(branch.fold_params) == 0
            assert branch.fold_u.shape == (0, 31)
            assert [bif.kind for bif in branch.bifurcations] == ["branch_point"] * 3
            npt.assert_allclose([bif.param for bif in branch.bifurcations], np.pi ** 2 * np.array([1, 4, 9]),
                                atol=atol)
            for n, bif in enumerate(branch.bifurcations, start=1):
                assert bif.refined and bif.multiplicity == 1 and bif.frequency == 0
                assert branch.params[bif.index] < bif.param < branch.params[bif.index + 1]
                npt.assert_allclose(bif.u, 0, atol=1e-12)
                mode = np.sin(n * np.pi * x)
                same_up_to_sign(bif.mode, mode / np.linalg.norm(mode), atol=1e-6)
            npt.assert_array_equal(branch.stable, branch.params < np.pi ** 2)
            npt.assert_array_equal(branch.n_unstable, np.floor(np.sqrt(branch.params) / np.pi))
        assert repr(branch) == ("ContinuationResult(101 points, parameter from 0 to 100, 0 fold(s), "
                                "10 stable point(s), 3 branch point(s), success=True)")

    def test_window_of_eigenvalues(self, rod):
        # With a single eigenvalue, only the first crossing can be seen, and no other is invented.
        D2 = rod.diff_operator(2)
        branch = continuation(rod, lambda u, lam: D2(u) + lam * (u - u ** 3), 0, 0, bcs=0,
                              jacobian=lambda u, lam: D2 + lam * (1 - 3 * u ** 2), ds=1.0, param_range=(0, 100),
                              stability=1)
        bif, = branch.bifurcations
        npt.assert_allclose(bif.param, np.pi ** 2, atol=1e-8)
        assert branch.eigenvalues.shape == (101, 1)
        assert branch.n_unstable.max() == 1

    def test_change_of_the_computed_eigenvalues_is_no_bifurcation(self, rod, monkeypatch):
        # For large problems, the computed eigenvalues are those closest to a shift.  An
        # eigenvalue that enters this set looks like a crossing, but does not pass through zero.
        original = numgrids.stability._Linearization.leading

        def leading(self, y, args=(), vectors=False):
            vals, vecs = original(self, y, args, vectors)
            if args[0] > 5.5 and not vectors:
                vals = np.concatenate([[80.0], vals[:-1]])
            return vals, vecs

        monkeypatch.setattr(numgrids.stability._Linearization, "leading", leading)
        D2 = rod.diff_operator(2)
        branch = continuation(rod, lambda u, lam: D2(u) + lam * (u - u ** 3), 0, 0, bcs=0,
                              jacobian=lambda u, lam: D2 + lam * (1 - 3 * u ** 2), ds=1.0, param_range=(0, 9),
                              stability=3)
        npt.assert_array_equal(branch.n_unstable, np.where(branch.params > 5.5, 1, 0))
        assert branch.bifurcations == []

    def test_branch_switching_euler_buckling(self, rod):
        # u'' + lam sin(u) = 0, u(0) = u(1) = 0: the rod buckles at lam = pi^2.  On the
        # buckled branch, sqrt(lam) = 2 K(k) with k = sin(a / 2) and the amplitude a.
        D2 = rod.diff_operator(2)
        F = lambda u, lam: D2(u) + lam * np.sin(u)  # noqa: E731
        J = lambda u, lam: D2 + lam * np.cos(u)  # noqa: E731
        straight = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=1.0, param_range=(0, 20), stability=3)
        bp, = straight.bifurcations
        assert bp.kind == "branch_point"
        npt.assert_allclose(bp.param, np.pi ** 2, atol=1e-8)

        for direction in (1, -1):
            buckled = continuation(rod, F, bp.u, bp.param, bcs=0, jacobian=J, tangent=bp.mode,
                                   direction=direction, ds=0.2, max_steps=30, stability=3)
            assert buckled.success and len(buckled) == 31
            assert buckled.params[0] == bp.param
            amplitude = buckled.u[:, 15]
            assert np.all(direction * amplitude[1:] > 0)
            assert np.all(np.diff(np.abs(amplitude)) > 0) and abs(amplitude[-1]) > 1.5
            npt.assert_allclose(buckled.params[1:], 4 * ellipk(np.sin(amplitude[1:] / 2) ** 2) ** 2, atol=1e-7)
            # The buckled states are stable; the branch point at the start is not reported again.
            assert np.all(buckled.stable[1:]) and not buckled.stable[0]
            assert buckled.bifurcations == []

    def test_branch_switching_in_a_system(self, rod):
        # The buckling problem as a system with the constraint v = u.
        D2 = rod.diff_operator(2)
        F = lambda u, v, lam: (D2(u) + lam * np.sin(v), v - u)  # noqa: E731
        J = lambda u, v, lam: [[D2, lam * np.cos(v)], [-1, 1]]  # noqa: E731
        straight = continuation(rod, F, (0, 0), 0, bcs=(0, None), jacobian=J, ds=1.0, param_range=(0, 20),
                                stability=3, mass=(1, 0))
        bp, = straight.bifurcations
        npt.assert_allclose(bp.param, np.pi ** 2, atol=1e-8)
        assert isinstance(bp.mode, tuple) and len(bp.mode) == 2
        buckled = continuation(rod, F, bp.u, bp.param, bcs=(0, None), jacobian=J, tangent=bp.mode, ds=0.2,
                               max_steps=10, stability=3, mass=(1, 0))
        u, v = buckled.u
        npt.assert_allclose(v, u, atol=1e-12)
        amplitude = u[1:, 15]
        npt.assert_allclose(buckled.params[1:], 4 * ellipk(np.sin(amplitude / 2) ** 2) ** 2, atol=1e-7)
        assert np.all(buckled.stable[1:])
        # A Hopf mode is no direction to follow steady states.
        with pytest.raises(ValueError, match="real"):
            continuation(rod, F, bp.u, bp.param, bcs=(0, None), jacobian=J, tangent=(1j * bp.mode[0], bp.mode[1]))

    def test_branch_point_on_a_curved_branch(self):
        # u'' - sinh(lam) u + u^3 = 0 with u' = 0 at the ends has the uniform solution
        # u = sqrt(sinh(lam)).  Its eigenvalues are 2 sinh(lam) - (n pi)^2.
        grid = Grid(x=chebyshev(21, 0, 1))
        D2 = grid.diff_operator(2)
        F = lambda u, lam: D2(u) - np.sinh(lam) * u + u ** 3  # noqa: E731
        J = lambda u, lam: D2 - np.sinh(lam) + 3 * u ** 2  # noqa: E731
        branch = continuation(grid, F, np.sqrt(np.sinh(1)), 1, bcs=Neumann(0), jacobian=J, ds=0.2,
                              param_range=(1, 3), stability=4)
        npt.assert_allclose(branch.u, np.sqrt(np.sinh(branch.params))[:, None] * np.ones(21), atol=1e-9)
        bif, = branch.bifurcations
        assert bif.kind == "branch_point" and bif.refined
        npt.assert_allclose(bif.param, np.arcsinh(np.pi ** 2 / 2), atol=1e-7)
        npt.assert_allclose(bif.u, np.sqrt(np.sinh(bif.param)), atol=1e-5)
        npt.assert_array_equal(branch.n_unstable, np.where(branch.params < bif.param, 1, 2))
        assert not np.any(branch.stable)
        npt.assert_allclose(branch.eigenvalues[:, :2],
                            2 * np.sinh(branch.params)[:, None] - np.array([0, np.pi ** 2]), atol=1e-7)

        # Without refinement: interpolated between the two neighbouring points.
        coarse = continuation(grid, F, np.sqrt(np.sinh(1)), 1, bcs=Neumann(0), jacobian=J, ds=0.2,
                              param_range=(1, 3), stability=4, refine=False)
        bif, = coarse.bifurcations
        assert not bif.refined and bif.mode is None
        npt.assert_allclose(bif.param, np.arcsinh(np.pi ** 2 / 2), atol=2e-2)

    def test_degenerate_eigenvalue(self):
        # On the square, the eigenvalue lam - 5 pi^2 of u = 0 is double.
        grid = Grid(x=chebyshev(11, 0, 1), y=chebyshev(11, 0, 1))
        L = grid.laplacian_operator()
        branch = continuation(grid, lambda u, lam: L(u) + lam * (u - u ** 3), 0, 0, bcs=0,
                              jacobian=lambda u, lam: L + lam * (1 - 3 * u ** 2), ds=2.0, param_range=(0, 60),
                              stability=4)
        first, second = branch.bifurcations
        assert first.kind == second.kind == "branch_point"
        npt.assert_allclose([first.param, second.param], np.pi ** 2 * np.array([2, 5]), atol=1e-3)
        assert (first.multiplicity, second.multiplicity) == (1, 2)
        assert "multiplicity 2" in repr(second)
        assert branch.n_unstable[-1] == 3

    def test_degenerate_fold(self):
        # u^2 + lam^2 = 1 on every grid point: all eigenvalues 2 u cross at the folds lam = +-1.
        grid = Grid(x=equidistant(5, 0, 1))
        branch = continuation(grid, lambda u, lam: u ** 2 + lam ** 2 - 1, 1, 0, jacobian=lambda u, lam: 2 * u,
                              ds=0.1, max_steps=80, stability=3)
        assert [bif.kind for bif in branch.bifurcations] == ["fold"] * 3
        assert [bif.multiplicity for bif in branch.bifurcations] == [3] * 3
        npt.assert_allclose(branch.fold_params, [1, -1, 1], atol=1e-10)
        npt.assert_allclose(branch.fold_u, 0, atol=1e-8)
        npt.assert_allclose(branch.eigenvalues, 2 * branch.u[:, :3], atol=1e-9)
        npt.assert_array_equal(branch.stable, branch.u[:, 0] < 0)


class TestHopf:
    """The Brusselator with diffusion and no-flux boundaries.

    The uniform state (a, b / a) loses stability to oscillations of the
    mode cos(n pi x) at b = 1 + a^2 + (Du + Dv) (n pi)^2.
    """

    a, Du, Dv = 1.2, 0.05, 0.03

    def problem(self, grid):
        a, Du, Dv = self.a, self.Du, self.Dv
        D2 = grid.diff_operator(2)

        def residual(u, v, b):
            return Du * D2(u) + a - (b + 1) * u + u ** 2 * v, Dv * D2(v) + b * u - u ** 2 * v

        def jacobian(u, v, b):
            return [[Du * D2 - (b + 1) + 2 * u * v, u ** 2],
                    [b - 2 * u * v, Dv * D2 - u ** 2]]

        return residual, jacobian

    def threshold(self, n):
        """Parameter and frequency of the Hopf bifurcation of the mode *n*."""
        a, Du, Dv = self.a, self.Du, self.Dv
        q = (n * np.pi) ** 2
        b = 1 + a ** 2 + (Du + Dv) * q
        return b, np.sqrt(a ** 2 * b - (b - 1 - Du * q) * (a ** 2 + Dv * q))

    def spectrum(self, b, modes):
        a, Du, Dv = self.a, self.Du, self.Dv
        vals = []
        for n in range(modes):
            q = (n * np.pi) ** 2
            vals.extend(np.linalg.eigvals(np.array([[b - 1 - Du * q, a ** 2], [-b, -a ** 2 - Dv * q]])))
        vals = np.array(vals)
        return vals[np.lexsort((-vals.imag, -vals.real))]

    @pytest.mark.parametrize("exact_jacobian", [True, False])
    def test_hopf_bifurcations_of_the_uniform_state(self, exact_jacobian):
        grid = Grid(x=chebyshev(17, 0, 1))
        a = self.a
        residual, jacobian = self.problem(grid)
        branch = continuation(grid, residual, (a, 1.5 / a), 1.5, bcs=Neumann(0),
                              jacobian=jacobian if exact_jacobian else None, ds=0.2, param_range=(1.5, 4),
                              stability=6)
        u, v = branch.u
        npt.assert_allclose(u, a, atol=1e-9)
        npt.assert_allclose(v, branch.params[:, None] / a * np.ones(17), atol=1e-9)

        atol = 1e-8 if exact_jacobian else 1e-6
        first, second = branch.bifurcations
        for n, bif in enumerate((first, second)):
            b, omega = self.threshold(n)
            assert bif.kind == "hopf" and bif.refined and bif.multiplicity == 1
            npt.assert_allclose(bif.param, b, atol=atol)
            npt.assert_allclose(bif.frequency, omega, atol=atol)
            assert branch.params[bif.index] < bif.param < branch.params[bif.index + 1]
            # The critical mode is complex; its shape is cos(n pi x) in both fields.
            mode_u, mode_v = bif.mode
            assert np.iscomplexobj(mode_u)
            shape = np.cos(n * np.pi * grid.coords)
            npt.assert_allclose(np.abs(mode_u) / np.abs(mode_u).max(), np.abs(shape), atol=1e-5)
            npt.assert_allclose(np.abs(mode_v) / np.abs(mode_v).max(), np.abs(shape), atol=1e-5)
        assert "frequency 1.2" in repr(first)

        assert np.iscomplexobj(branch.eigenvalues)
        npt.assert_array_equal(branch.stable, branch.params < first.param)
        expected = np.where(branch.params < first.param, 0, np.where(branch.params < second.param, 2, 4))
        npt.assert_array_equal(branch.n_unstable, expected)
        for i in (0, len(branch) - 1):
            npt.assert_allclose(branch.eigenvalues[i], self.spectrum(branch.params[i], 5)[:6], atol=atol)
        assert repr(branch).endswith("2 Hopf bifurcation(s), success=True)")
        assert len(repr(branch)) < 160

    def test_eigenfunctions_of_a_system(self):
        grid = Grid(x=chebyshev(17, 0, 1))
        a, b = self.a, 3.5
        residual, jacobian = self.problem(grid)
        state = (np.full(17, a), np.full(17, b / a))
        exact = linear_stability(grid, residual, state, bcs=Neumann(0), jacobian=jacobian, param=b, k=4)
        npt.assert_allclose(exact.eigenvalues, self.spectrum(b, 4)[:4], atol=1e-9)
        assert exact.n_unstable == 4 and not exact.stable  # the modes 0 and 1 oscillate and grow
        assert isinstance(exact.eigenfunctions, tuple) and len(exact.eigenfunctions) == 2
        vu, vv = exact.eigenfunctions
        assert vu.shape == vv.shape == (4, 17) and np.iscomplexobj(vu)
        # Normalized over both fields; conjugate pairs have conjugate eigenfunctions.
        npt.assert_allclose(np.sum(np.abs(vu) ** 2 + np.abs(vv) ** 2, axis=1), 1, atol=1e-12)
        npt.assert_allclose(vu[1], vu[0].conj(), atol=1e-9)
        # The eigenfunctions solve the linearized equations: J v = sigma v.
        Du, Dv = self.Du, self.Dv
        D2 = grid.diff_operator(2)
        for j in range(4):
            sigma = exact.eigenvalues[j]
            ru = Du * D2(vu[j]) + (b - 1) * vu[j] + a ** 2 * vv[j] - sigma * vu[j]
            rv = Dv * D2(vv[j]) - b * vu[j] - a ** 2 * vv[j] - sigma * vv[j]
            npt.assert_allclose(ru[1:-1], 0, atol=1e-8)
            npt.assert_allclose(rv[1:-1], 0, atol=1e-8)
        # A finite-difference Jacobian gives the same eigenvalues.
        fd = linear_stability(grid, lambda u, v: residual(u, v, b), state, bcs=Neumann(0), k=4)
        npt.assert_allclose(fd.eigenvalues, exact.eigenvalues, atol=1e-6)


class TestMass:

    def test_scalar_and_density(self, rod):
        D2 = rod.diff_operator(2)
        x = rod.coords
        F = lambda u: D2(u) + 5 * u  # noqa: E731
        J = lambda u: D2 + 5  # noqa: E731
        plain = linear_stability(rod, F, 0, bcs=0, jacobian=J, k=4)
        double = linear_stability(rod, F, 0, bcs=0, jacobian=J, k=4, mass=2.0)
        npt.assert_allclose(double.eigenvalues, plain.eigenvalues / 2, atol=1e-10)
        # rho u_t = u'' + 5 u: the generalized eigenproblem of Operator.eig.
        rho = 1 + x ** 2
        reference, _ = (D2 + 5).eig(k=4, bcs=0, M=rho, which="LR")
        for mass in (rho, scipy.sparse.diags(rho), np.diag(rho)):
            result = linear_stability(rod, F, 0, bcs=0, jacobian=J, k=4, mass=mass)
            npt.assert_allclose(result.eigenvalues, reference, atol=1e-8)

    def test_algebraic_constraint(self, rod):
        # u_t = u'' + e^v, 0 = v - u: the Bratu problem with a field without time derivative.
        D2 = rod.diff_operator(2)
        u = solve_nonlinear(rod, lambda u: D2(u) + np.exp(u), 0, bcs=0)
        scalar = linear_stability(rod, lambda u: D2(u) + np.exp(u), u, bcs=0, jacobian=lambda u: D2 + np.exp(u),
                                  k=4)
        residual = lambda u, v: (D2(u) + np.exp(v), v - u)  # noqa: E731
        blocks = lambda u, v: [[D2, np.exp(v)], [-1, 1]]  # noqa: E731
        for jacobian, atol in [(blocks, 1e-8), (None, 1e-6)]:
            for mass in ((1, 0), (1, None), [[1, None], [None, 0]]):
                system = linear_stability(rod, residual, (u, u), bcs=(0, None), jacobian=jacobian, mass=mass,
                                          k=4)
                npt.assert_allclose(system.eigenvalues, scalar.eigenvalues, atol=atol)
                assert system.stable
                vu, vv = system.eigenfunctions
                npt.assert_allclose(vv, vu, atol=1e-6)
                same_up_to_sign(vu[0] * np.sqrt(2), scalar.eigenfunctions[0], atol=1e-6)
        # Without the mass, v would relax on its own: a different spectrum.
        relaxing = linear_stability(rod, residual, (u, u), bcs=(0, None), jacobian=blocks, k=4)
        assert abs(relaxing.eigenvalues[0] - scalar.eigenvalues[0]) > 1e-2

    def test_along_a_branch(self, rod):
        D2 = rod.diff_operator(2)
        branch = continuation(rod, lambda u, v, lam: (D2(u) + lam * np.exp(v), v - u), (0, 0), 0, bcs=(0, None),
                              jacobian=lambda u, v, lam: [[D2, lam * np.exp(v)], [-1, 1]], ds=0.2, max_steps=25,
                              stability=3, mass=(1, 0))
        fold, = branch.bifurcations
        npt.assert_allclose(fold.param, BRATU_FOLD, atol=1e-9)
        assert isinstance(fold.u, tuple) and isinstance(fold.mode, tuple)
        assert isinstance(branch.fold_u, tuple) and branch.fold_u[0].shape == (1, 31)
        npt.assert_array_equal(branch.n_unstable, np.where(np.arange(26) <= fold.index, 0, 1))
        npt.assert_allclose(branch.eigenvalues[0], -(np.arange(1, 4) * np.pi) ** 2, atol=1e-8)
        # sign=-1 belongs to the reversed evolution: everything is unstable.
        reverse = continuation(rod, lambda u, lam: -D2(u) - lam * np.exp(u), 0, 0, bcs=0, ds=0.2, max_steps=5,
                               stability=3, sign=-1)
        npt.assert_allclose(reverse.eigenvalues[0], -(np.arange(1, 4) * np.pi) ** 2, atol=1e-6)
        assert np.all(reverse.stable)


class TestLargeProblems:
    """Shift-invert Arnoldi iterations instead of dense eigensolvers."""

    def test_same_eigenvalues(self, rod, monkeypatch):
        F, J = bratu(rod)
        x = rod.coords
        upper = solve_nonlinear(rod, lambda u: F(u, 1.0), 4 * np.sin(np.pi * x), bcs=0)
        dense = linear_stability(rod, F, upper, bcs=0, jacobian=J, param=1.0, k=4)
        monkeypatch.setattr(numgrids.stability, "_DENSE_EIG_LIMIT", 10)
        for limit in (3000, 10):  # dense and sparse Jacobian
            monkeypatch.setattr(numgrids.nonlinear, "_DENSE_LIMIT", limit)
            sparse = linear_stability(rod, F, upper, bcs=0, jacobian=J, param=1.0, k=4)
            npt.assert_allclose(sparse.eigenvalues, dense.eigenvalues, atol=1e-8)
            assert sparse.n_unstable == 1
            for j in range(4):
                same_up_to_sign(sparse.eigenfunctions[j], dense.eigenfunctions[j], atol=1e-7)

    def test_conditions_and_constraints(self, rod, monkeypatch):
        monkeypatch.setattr(numgrids.stability, "_DENSE_EIG_LIMIT", 10)
        D2 = rod.diff_operator(2)
        # A zero eigenvalue exactly on the default shift.
        result = linear_stability(rod, lambda u: D2(u), 0, bcs=Neumann(0), jacobian=lambda u: D2, k=4)
        npt.assert_allclose(result.eigenvalues, -(np.arange(4) * np.pi) ** 2, atol=1e-7)
        assert not result.stable and result.n_unstable == 0
        npt.assert_allclose(rod.diff(result.eigenfunctions[1], 1)[[0, -1]], 0, atol=1e-7)
        result = linear_stability(rod, lambda u, v: (D2(u) + 5 * v, v - u), (0, 0), bcs=(0, None), mass=(1, 0),
                                  k=3, sigma=-10.0)
        npt.assert_allclose(result.eigenvalues, 5 - (np.arange(1, 4) * np.pi) ** 2, atol=1e-7)

    def test_branch(self, rod, monkeypatch):
        monkeypatch.setattr(numgrids.stability, "_DENSE_EIG_LIMIT", 10)
        D2 = rod.diff_operator(2)
        branch = continuation(rod, lambda u, lam: D2(u) + lam * (u - u ** 3), 0, 0, bcs=0,
                              jacobian=lambda u, lam: D2 + lam * (1 - 3 * u ** 2), ds=2.0, param_range=(0, 50),
                              stability=3)
        assert [bif.kind for bif in branch.bifurcations] == ["branch_point"] * 2
        npt.assert_allclose([bif.param for bif in branch.bifurcations], np.pi ** 2 * np.array([1, 4]), atol=1e-7)


class TestErrors:

    def test_linear_stability(self, rod):
        F = lambda u: rod.diff(u, 2) + u  # noqa: E731
        with pytest.raises(ValueError, match="positive integer"):
            linear_stability(rod, F, 0, bcs=0, k=0)
        with pytest.raises(ValueError, match="sign"):
            linear_stability(rod, F, 0, bcs=0, sign=2)
        with pytest.raises(TypeError, match="real"):
            linear_stability(rod, F, 1j, bcs=0)
        with pytest.raises(TypeError, match="mass"):
            linear_stability(rod, F, 0, bcs=0, mass="heavy")
        with pytest.raises(ValueError, match="mass"):
            linear_stability(rod, lambda u, v: (F(u), F(v)), (0, 0), bcs=0, mass=(1, 1, 1))
        with pytest.raises(ValueError, match="shape"):
            linear_stability(rod, F, 0, bcs=0, mass=scipy.sparse.identity(5))

    def test_more_eigenvalues_than_unknowns(self):
        grid = Grid(x=chebyshev(6, 0, 1))
        result = linear_stability(grid, lambda u: grid.diff(u, 2), 0, bcs=0, k=10)
        assert result.eigenvalues.shape == (4,)
        assert result.eigenfunctions.shape == (4, 6)

    def test_continuation(self, rod):
        F, J = bratu(rod)
        with pytest.raises(ValueError, match="positive integer"):
            continuation(rod, F, 0, 0, bcs=0, stability=0)
        with pytest.raises(ValueError, match="sign"):
            continuation(rod, F, 0, 0, bcs=0, stability=2, sign=0)
        with pytest.raises(ValueError, match="tangent"):
            continuation(rod, F, 0, 0, bcs=0, tangent=np.zeros(31))
        with pytest.raises(ValueError, match="field"):
            continuation(rod, lambda u, v, lam: (F(u, lam), F(v, lam)), (0, 0), 0, bcs=0, tangent=np.ones(31))

    def test_verbose_reports_unstable_modes(self, rod, capsys):
        F, J = bratu(rod)
        continuation(rod, F, 0, 3, bcs=0, jacobian=J, ds=0.3, max_steps=6, stability=2, verbose=True)
        out = capsys.readouterr().out
        assert "0 unstable mode(s)" in out and "1 unstable mode(s)" in out
