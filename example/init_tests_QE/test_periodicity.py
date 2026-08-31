"""Tests for the periodicity features added to ri_basis:

 - PrimitiveGaussianRadials.estimate_cutoff()
 - RIBasis._resolve_cutoff() (tested via the ``cutoff`` constructor parameter)
 - RIBasis._get_lattice_vectors() (tested via ``cell_vectors`` + ``cutoff`` parameters)
 - RIBasis.__call__ with periodic boundary conditions
"""

import numpy as np
import pytest

from salted_espresso.ri_basis import load_basis
from salted_espresso.ri_basis.core import RIBasis
from salted_espresso.ri_basis.gaussian import PrimitiveGaussianRadials
from salted_espresso.ri_basis.real_spher_harmonic import RealSphericalHarmonics
from salted_espresso.ri_basis.types import CutoffType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_radials(n_max: int = 1, l_max: int = 0,
                  alpha: float = 1.0) -> PrimitiveGaussianRadials:
    """Return a PrimitiveGaussianRadials with uniform alpha for all (n,l) pairs."""
    n_alphas = n_max * (l_max + 1)
    return PrimitiveGaussianRadials(
        "H", (0.0, 0.0, 0.0), n_max, l_max,
        alphas=[alpha] * n_alphas,
    )


def _make_basis(n_max: int = 1, l_max: int = 0, alpha: float = 1.0,
                origin=(0.0, 0.0, 0.0),
                cell_vectors=None,
                cutoff: CutoffType | float = CutoffType.NON_PERIODIC) -> RIBasis:
    """Return an RIBasis with optional periodic cell."""
    n_alphas = n_max * (l_max + 1)
    return load_basis(
        species="H",
        origin=origin,
        n_max=n_max,
        l_max=l_max,
        radial_method="gaussian",
        angular_method="spherical",
        radial_params={"alphas": [alpha] * n_alphas},
        cell_vectors=cell_vectors,
        cutoff=cutoff,
    )


# ---------------------------------------------------------------------------
# PrimitiveGaussianRadials.estimate_cutoff
# ---------------------------------------------------------------------------

class TestEstimateCutoff:
    """Tests for PrimitiveGaussianRadials.estimate_cutoff()."""

    def test_returns_positive_float(self):
        rad = _make_radials(alpha=1.0)
        cutoff = rad.estimate_cutoff()
        assert isinstance(cutoff, float)
        assert cutoff > 0.0

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_cutoff_is_positive_for_various_alphas(self, alpha):
        rad = _make_radials(alpha=alpha)
        assert rad.estimate_cutoff() > 0.0

    def test_smaller_alpha_gives_larger_cutoff(self):
        """Loose Gaussians (small alpha) extend further in space."""
        rad_loose = _make_radials(alpha=0.1)
        rad_tight = _make_radials(alpha=5.0)
        assert rad_loose.estimate_cutoff() > rad_tight.estimate_cutoff()

    def test_larger_threshold_gives_smaller_cutoff(self):
        """A less strict threshold permits a smaller cutoff radius."""
        rad = _make_radials(alpha=1.0)
        cutoff_strict = rad.estimate_cutoff(threshold=1e-10)
        cutoff_loose = rad.estimate_cutoff(threshold=1e-2)
        assert cutoff_strict > cutoff_loose

    def test_cutoff_formula_matches_expected(self):
        """The cutoff satisfies r = sqrt(-ln(threshold) / alpha) by construction."""
        alpha = 1.5
        threshold = 1e-6
        rad = _make_radials(alpha=alpha)
        expected = np.sqrt(-np.log(threshold) / alpha)
        assert abs(rad.estimate_cutoff(threshold=threshold) - expected) < 1e-12

    def test_multi_alpha_uses_minimum_alpha(self):
        """When multiple (n, l) pairs exist, the smallest alpha governs the cutoff."""
        rad = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 2, 0, alphas=[0.1, 5.0])
        expected = np.sqrt(-np.log(1e-6) / 0.1)
        assert abs(rad.estimate_cutoff() - expected) < 1e-12

    def test_estimate_increases_with_n_max(self):
        """More radials with the same small alpha produce the same cutoff (min alpha governs)."""
        rad_1 = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 1, 0, alphas=[0.5])
        rad_2 = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 2, 0, alphas=[0.5, 0.5])
        assert abs(rad_1.estimate_cutoff() - rad_2.estimate_cutoff()) < 1e-12


# ---------------------------------------------------------------------------
# RIBasis._resolve_cutoff (via the cutoff constructor parameter)
# ---------------------------------------------------------------------------

class TestResolveCutoff:
    """Tests for cutoff resolution logic in RIBasis.__init__."""

    def test_non_periodic_sets_cutoff_to_none(self):
        basis = _make_basis(cutoff=CutoffType.NON_PERIODIC)
        assert basis.cutoff is None

    def test_float_cutoff_stored_as_float(self):
        basis = _make_basis(cutoff=5.0)
        assert basis.cutoff == 5.0
        assert isinstance(basis.cutoff, float)

    def test_integer_cutoff_converted_to_float(self):
        basis = _make_basis(cutoff=3)
        assert basis.cutoff == 3.0
        assert isinstance(basis.cutoff, float)

    def test_estimate_cutoff_matches_radial_estimate(self):
        """CutoffType.ESTIMATE should equal radial_funcs.estimate_cutoff()."""
        alpha = 1.0
        expected = _make_radials(alpha=alpha).estimate_cutoff()
        basis = _make_basis(alpha=alpha, cutoff=CutoffType.ESTIMATE)
        assert abs(basis.cutoff - expected) < 1e-12

    def test_invalid_cutoff_type_raises_type_error(self):
        """Passing an unsupported type (e.g. None) as cutoff must raise TypeError."""
        with pytest.raises(TypeError, match="Invalid cutoff type"):
            RIBasis(
                "H", (0.0, 0.0, 0.0), 1, 0,
                radial_cls=PrimitiveGaussianRadials,
                angular_cls=RealSphericalHarmonics,
                radial_kwargs={"alphas": [1.0]},
                cutoff=None,
            )


# ---------------------------------------------------------------------------
# RIBasis._get_lattice_vectors (via cell_vectors + cutoff constructor params)
# ---------------------------------------------------------------------------

class TestGetLatticeVectors:
    """Tests for periodic lattice vector generation in RIBasis."""

    def test_no_cell_vectors_gives_only_origin(self):
        basis = _make_basis(cutoff=CutoffType.NON_PERIODIC)
        assert basis.lattice_vectors.shape == (1, 3)
        np.testing.assert_array_equal(basis.lattice_vectors[0], [0, 0, 0])

    def test_non_periodic_cutoff_with_cell_gives_only_origin(self):
        """NON_PERIODIC cutoff suppresses image generation even when cell is provided."""
        basis = _make_basis(cell_vectors=np.eye(3) * 5.0, cutoff=CutoffType.NON_PERIODIC)
        assert basis.lattice_vectors.shape == (1, 3)
        np.testing.assert_array_equal(basis.lattice_vectors[0], [0, 0, 0])

    def test_origin_is_always_included(self):
        """The zero translation vector must always be present."""
        basis = _make_basis(cell_vectors=np.eye(3) * 2.0, cutoff=5.0)
        has_origin = np.any(np.all(basis.lattice_vectors == 0.0, axis=1))
        assert has_origin

    def test_all_vectors_within_cutoff(self):
        cell = np.eye(3) * 2.0
        cutoff = 3.5
        basis = _make_basis(cell_vectors=cell, cutoff=cutoff)
        norms = np.linalg.norm(basis.lattice_vectors, axis=1)
        assert np.all(norms <= cutoff + 1e-12)

    def test_larger_cutoff_produces_more_vectors(self):
        cell = np.eye(3) * 2.0
        basis_small = _make_basis(cell_vectors=cell, cutoff=1.5)
        basis_large = _make_basis(cell_vectors=cell, cutoff=5.0)
        assert len(basis_large.lattice_vectors) > len(basis_small.lattice_vectors)

    def test_lattice_vectors_are_integer_combinations_of_cell(self):
        """Each lattice vector must be an integer linear combination of cell vectors."""
        cell = np.eye(3) * 3.0
        basis = _make_basis(cell_vectors=cell, cutoff=4.5)
        inv_cell = np.linalg.inv(cell)
        fractional = basis.lattice_vectors @ inv_cell
        residuals = fractional - np.round(fractional)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_non_orthogonal_cell(self):
        """Lattice vector generation works for non-orthogonal cells."""
        cell = np.array([[2.0, 0.0, 0.0],
                         [1.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]])
        basis = _make_basis(cell_vectors=cell, cutoff=3.0)
        norms = np.linalg.norm(basis.lattice_vectors, axis=1)
        assert np.all(norms <= 3.0 + 1e-12)
        has_origin = np.any(np.all(basis.lattice_vectors == 0.0, axis=1))
        assert has_origin

    def test_single_neighbour_shell_cubic_cell(self):
        """For a cubic cell with cutoff just above the cell parameter, only the 6
        face-centre images and the origin should be within range."""
        a = 3.0
        cell = np.eye(3) * a
        # Cutoff slightly above a includes the 6 nearest neighbors + origin
        basis = _make_basis(cell_vectors=cell, cutoff=a + 0.01)
        # Exactly the 6 face-centered images ([±a,0,0] etc.) + origin
        assert len(basis.lattice_vectors) == 7


# ---------------------------------------------------------------------------
# RIBasis.__call__ with periodic boundary conditions
# ---------------------------------------------------------------------------

class TestRIBasisPeriodicEvaluation:
    """Tests that __call__ correctly sums over periodic images."""

    def test_periodic_with_large_cell_matches_nonperiodic(self):
        """When cell >> cutoff, images contribute negligibly: periodic ≈ non-periodic."""
        alpha = 1.0
        r = np.array([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])
        basis_np = _make_basis(alpha=alpha, cutoff=CutoffType.NON_PERIODIC)
        # Large cell: nearest image at ±100 Å is negligible for alpha=1.0
        cutoff = _make_radials(alpha=alpha).estimate_cutoff()
        basis_p = _make_basis(alpha=alpha, cell_vectors=np.eye(3) * 100.0, cutoff=cutoff)
        np.testing.assert_allclose(basis_p(r), basis_np(r), atol=1e-8)

    def test_output_shape_matches_nonperiodic(self):
        """Periodic and non-periodic bases should return identical shapes."""
        rng = np.random.default_rng(0)
        r = rng.random((5, 3))
        basis_np = _make_basis(n_max=2, l_max=1, cutoff=CutoffType.NON_PERIODIC)
        basis_p = _make_basis(n_max=2, l_max=1, cell_vectors=np.eye(3) * 10.0, cutoff=3.72)
        assert basis_p(r).shape == basis_np(r).shape

    def test_periodic_images_increase_l0_value(self):
        """At a point midway between the origin and a periodic image of an l=0 basis,
        the periodic sum is larger than the non-periodic (single-image) value."""
        alpha = 5.0   # Tight Gaussian so images are distinguishable
        cell = np.eye(3) * 2.0
        cutoff = 3.0
        r = np.array([[1.0, 0.0, 0.0]])   # Midpoint to [2,0,0] image
        basis_np = _make_basis(alpha=alpha, cutoff=CutoffType.NON_PERIODIC)
        basis_p = _make_basis(alpha=alpha, cell_vectors=cell, cutoff=cutoff)
        val_np = basis_np(r)[0, 0]
        val_p = basis_p(r)[0, 0]
        # l=0 Gaussians are non-negative; adding images can only increase the sum
        assert val_p > val_np

    def test_output_is_finite_for_all_points(self):
        """All returned values must be finite (no NaN/Inf) for random inputs."""
        cell = np.eye(3) * 3.0
        basis_p = _make_basis(n_max=1, l_max=0, cell_vectors=cell, cutoff=4.0)
        rng = np.random.default_rng(42)
        r = rng.random((20, 3)) * 2.0
        vals = basis_p(r)
        assert np.all(np.isfinite(vals))

    def test_periodic_basis_with_offset_origin(self):
        """Periodic basis with a non-zero origin evaluates correctly at absolute coords."""
        origin = (1.0, -0.5, 2.0)
        basis_at_origin = _make_basis(n_max=1, l_max=0, origin=(0.0, 0.0, 0.0),
                                      cell_vectors=np.eye(3) * 100.0, cutoff=3.72)
        basis_shifted = _make_basis(n_max=1, l_max=0, origin=origin,
                                    cell_vectors=np.eye(3) * 100.0, cutoff=3.72)
        # Evaluating the origin-at-zero basis at r gives the same result as evaluating
        # the shifted basis at r + origin (both land at the same relative coordinate).
        r_rel = np.array([[0.5, 0.5, 0.3], [1.0, 0.0, 1.0]])
        r_abs = r_rel + np.array(origin)
        np.testing.assert_allclose(basis_at_origin(r_rel), basis_shifted(r_abs), atol=1e-14)
