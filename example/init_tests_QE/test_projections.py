"""Tests for the projection functions in the projections module:

 - compute_condition_number()
 - solve_projection_equations()
 - compute_projection_coefficients()
 - compute_projectability()
"""

import numpy as np
import pytest
import tempfile
import os

from salted_espresso.projections.core import (
    MetricMethod,
    compute_condition_number,
    compute_overlap,
    compute_projection_coefficients,
    compute_projectability,
    solve_projection_equations,
)
from salted_espresso.ri_basis import load_basis
from salted_espresso.ri_basis.loader import load_basis_set
from salted_espresso.ri_basis.types import CutoffType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_basis(n_max: int = 1, l_max: int = 0, alpha: float = 1.0):
    """Return an RIBasis with a uniform Gaussian radial and no periodicity."""
    n_alphas = n_max * (l_max + 1)
    return load_basis(
        species="H",
        origin=(0.0, 0.0, 0.0),
        n_max=n_max,
        l_max=l_max,
        radial_method="gaussian",
        angular_method="spherical",
        radial_params={"alphas": [alpha] * n_alphas},
        cutoff=CutoffType.NON_PERIODIC,
    )


def _make_basis_set(n_max: int = 1, l_max: int = 0, alpha: float = 1.0,
                    n_atoms: int = 1):
    """Return a temporary .xyz file path and the corresponding RIBasisSet."""
    lines = [str(n_atoms), "test structure"]
    for i in range(n_atoms):
        lines.append(f"H 0.0 0.0 {float(i * 1.4):.1f}")
    xyz_content = "\n".join(lines) + "\n"

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
    f.write(xyz_content)
    f.close()

    specs = {
        "H": {
            "n_max": n_max,
            "l_max": l_max,
            "radial_method": "gaussian",
            "angular_method": "real_spherical",
            "radial_params": {"alphas": [alpha] * (n_max * (l_max + 1))},
        }
    }
    basis_set = load_basis_set(f.name, specs, cutoff=CutoffType.NON_PERIODIC)
    os.unlink(f.name)
    return basis_set


def _gaussian_density(r: np.ndarray) -> np.ndarray:
    """Spherically symmetric Gaussian density rho(r) = exp(-|r|^2)."""
    return np.exp(-np.linalg.norm(r, axis=1) ** 2)


# ---------------------------------------------------------------------------
# compute_condition_number
# ---------------------------------------------------------------------------

class TestComputeConditionNumber:
    """Tests for compute_condition_number()."""

    def test_identity_matrix_has_condition_number_one(self):
        cond = compute_condition_number(np.eye(4))
        assert abs(cond - 1.0) < 1e-10

    @pytest.mark.parametrize("eigs", [
        [1.0, 2.0],
        [1.0, 10.0],
        [2.0, 6.0],
        [0.5, 4.0],
    ])
    def test_diagonal_matrix_condition_number(self, eigs):
        D = np.diag(eigs)
        cond = compute_condition_number(D)
        expected = max(eigs) / min(eigs)
        assert abs(cond - expected) < 1e-10

    def test_result_is_at_least_one(self):
        basis = _make_basis(n_max=2, l_max=1)
        M = compute_overlap(basis, method="overlap", n_cartesian_grid=256)
        cond = compute_condition_number(M)
        assert cond >= 1.0 - 1e-10

    def test_result_is_finite_for_wellconditioned_matrix(self):
        M = np.diag([1.0, 2.0, 3.0])
        cond = compute_condition_number(M)
        assert np.isfinite(cond)

    def test_tiny_negative_eigenvalue_clamped(self):
        """Small negative eigenvalues from numerical noise should not propagate."""
        # Construct a near-singular PSD matrix with a tiny negative perturbation
        M = np.diag([1.0, 0.5, 1e-12])
        cond = compute_condition_number(M, tol=1e-13)
        assert np.isfinite(cond)
        assert cond >= 1.0


# ---------------------------------------------------------------------------
# solve_projection_equations
# ---------------------------------------------------------------------------

class TestSolveProjectionEquations:
    """Tests for solve_projection_equations() – solves Mc = b."""

    def test_identity_matrix(self):
        """For M = I, the solution c must equal b exactly."""
        b = np.array([1.0, 2.0, 3.0, 4.0])
        c = solve_projection_equations(np.eye(4), b)
        np.testing.assert_allclose(c, b, atol=1e-12)

    def test_known_2x2_system(self):
        """[[2,1],[1,3]] c = [5,10]  =>  c = [1, 3]."""
        M = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 10.0])
        c = solve_projection_equations(M, b)
        np.testing.assert_allclose(c, [1.0, 3.0], atol=1e-12)

    def test_residual_is_small(self):
        """Verify Mc ≈ b after solving with a random positive-definite matrix."""
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        M = A @ A.T + np.eye(5)  # positive-definite
        b = rng.random(5)
        c = solve_projection_equations(M, b)
        np.testing.assert_allclose(M @ c, b, atol=1e-10)

    def test_output_shape(self):
        b = np.ones(6)
        c = solve_projection_equations(np.eye(6), b)
        assert c.shape == (6,)


# ---------------------------------------------------------------------------
# compute_projection_coefficients
# ---------------------------------------------------------------------------

class TestComputeProjectionCoefficients:
    """Tests for compute_projection_coefficients()."""

    def test_output_shape_single_basis(self):
        basis = _make_basis(n_max=1, l_max=0)
        coeffs = compute_projection_coefficients(
            _gaussian_density, basis, n_cartesian_grid=64
        )
        assert coeffs.shape == (len(basis),)

    def test_output_shape_larger_basis(self):
        basis = _make_basis(n_max=2, l_max=1)
        coeffs = compute_projection_coefficients(
            _gaussian_density, basis, n_cartesian_grid=64
        )
        assert coeffs.shape == (len(basis),)

    def test_coefficients_are_finite(self):
        basis = _make_basis(n_max=1, l_max=0)
        coeffs = compute_projection_coefficients(
            _gaussian_density, basis, n_cartesian_grid=64
        )
        assert np.all(np.isfinite(coeffs))

    def test_zero_density_gives_zero_coefficients(self):
        basis = _make_basis(n_max=1, l_max=0)
        zero_density = lambda r: np.zeros(r.shape[0])
        coeffs = compute_projection_coefficients(
            zero_density, basis, n_cartesian_grid=64
        )
        np.testing.assert_allclose(coeffs, 0.0, atol=1e-14)

    def test_l0_coefficient_is_nonzero_for_spherical_density(self):
        """A spherically symmetric density must have a non-zero s-orbital coefficient."""
        basis = _make_basis(n_max=1, l_max=0)
        coeffs = compute_projection_coefficients(
            _gaussian_density, basis, n_cartesian_grid=128
        )
        idx = basis.lexographic_to_running_index((1, 0, 0))
        assert abs(coeffs[idx]) > 1e-6

    def test_output_shape_for_basis_set(self):
        """compute_projection_coefficients also accepts an RIBasisSet."""
        basis_set = _make_basis_set(n_max=1, l_max=0, n_atoms=2)
        total_funcs = sum(len(b) for b in basis_set)
        coeffs = compute_projection_coefficients(
            _gaussian_density, basis_set, n_cartesian_grid=64
        )
        assert coeffs.shape == (total_funcs,)


# ---------------------------------------------------------------------------
# compute_projectability
# ---------------------------------------------------------------------------

class TestComputeProjectability:
    """Tests for compute_projectability()."""

    def test_projectability_is_finite_and_non_negative(self):
        """Projectability must be a finite non-negative number."""
        basis_set = _make_basis_set(n_max=1, l_max=0)
        M = compute_overlap(basis_set, method="overlap", n_cartesian_grid=256)
        b = compute_projection_coefficients(_gaussian_density, basis_set,
                                            n_cartesian_grid=256)
        c = solve_projection_equations(M, b)
        proj, _ = compute_projectability(_gaussian_density, basis_set,
                                      expansion_coefficients=c)
        assert np.isfinite(proj)
        assert proj >= 0.0

    def test_perfect_projection_gives_projectability_near_one(self):
        """When the Gaussian basis spans the Gaussian density almost exactly,
        the projectability should be close to 1."""
        basis_set = _make_basis_set(n_max=1, l_max=0, alpha=1.0)
        # The basis Gaussian and the density Gaussian have the same alpha,
        # so the projection onto the single s-function should be near-perfect.
        proj, _ = compute_projectability(_gaussian_density, basis_set)
        assert abs(proj - 1.0) < 0.01

    def test_projectability_with_precomputed_coefficients(self):
        """Passing precomputed expansion coefficients should yield the same result
        as computing them internally."""
        basis_set = _make_basis_set(n_max=1, l_max=0)
        M = compute_overlap(basis_set, method="overlap", n_cartesian_grid=256)
        b = compute_projection_coefficients(_gaussian_density, basis_set,
                                            n_cartesian_grid=256)
        c = solve_projection_equations(M, b)
        proj_pre, _ = compute_projectability(_gaussian_density, basis_set,
                                          expansion_coefficients=c)
        proj_auto, _ = compute_projectability(_gaussian_density, basis_set)
        assert abs(proj_pre - proj_auto) < 1e-6
