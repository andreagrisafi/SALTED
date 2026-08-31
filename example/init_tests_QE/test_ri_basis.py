"""Tests for src/ri_basis/ – Gaussian radials, real spherical harmonics,
combined RI basis, and the loader factory."""

import numpy as np
import pytest
from scipy.integrate import quad


from salted.qe import load_basis
from salted.qe.core  import AngularFunctions, RadialFunctions, RIBasis
from salted.qe.gaussian import PrimitiveGaussian, PrimitiveGaussianRadials
from salted.qe.realspherharmonic import RealSphericalHarmonics
from salted.qe.combine_gaussian import CompoundGaussianRadials
from salted.qe.qe_types import Cutoff, CutoffType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_basis(n_max: int | list[int], l_max: int, origin=(0.0, 0.0, 0.0),
                alpha: float = 1.0) -> RIBasis:
    """Convenience factory that creates a Gaussian/spherical RI basis."""
    if isinstance(n_max, int):
        n_alphas = n_max * (l_max + 1)
    else:
        n_alphas = sum(n_max)
    return load_basis(
        species="H",
        origin=origin,
        n_max=n_max,
        l_max=l_max,
        cutoff = CutoffType.NON_PERIODIC,
        radial_method="gaussian",
        angular_method="spherical",
        radial_params={"alphas": [alpha] * n_alphas},
    )


def _unit_sphere_grid(n_theta: int = 80, n_phi: int = 160) -> np.ndarray:
    """Return a (N, 3) array of unit-sphere sample points."""
    theta = np.linspace(0.0, np.pi, n_theta, endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tg, pg = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(tg) * np.cos(pg)
    y = np.sin(tg) * np.sin(pg)
    z = np.cos(tg)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (re-used across test classes to avoid recomputation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basis_2_2():
    """RIBasis with n_max=2, l_max=2, alpha=1.0, origin at zero."""
    return _make_basis(n_max=2, l_max=2)


@pytest.fixture(scope="module")
def unit_sphere_pts():
    """Dense grid of unit-sphere points for angular tests."""
    return _unit_sphere_grid()


# ---------------------------------------------------------------------------
# PrimitiveGaussian
# ---------------------------------------------------------------------------

class TestPrimitiveGaussian:
    """Tests for the single-Gaussian radial function R(r) = A r^l exp(-α r²)."""

    @pytest.mark.parametrize("alpha,l", [
        (0.5, 0), (1.0, 0), (2.0, 0),
        (1.0, 1), (1.0, 2), (2.0, 2),
    ])
    def test_normalization(self, alpha, l):
        """∫₀^∞ |R(r)|² r² dr == 1 for all (alpha, l)."""
        g = PrimitiveGaussian(alpha=alpha, l=l)
        integrand = lambda r: r ** 2 * g(np.asarray([r]))[0] ** 2
        val, _ = quad(integrand, 0.0, np.inf)
        assert abs(val - 1.0) < 1e-8, (
            f"alpha={alpha}, l={l}: norm²={val:.10f}, expected 1.0"
        )

    def test_amplitude_is_positive(self):
        g = PrimitiveGaussian(alpha=1.0, l=0)
        assert g.amplitude > 0.0

    def test_l0_peak_at_origin(self):
        """For l=0, R(r) is maximised at r=0; value should equal amplitude."""
        g = PrimitiveGaussian(alpha=1.0, l=0)
        assert abs(g(np.array([0.0]))[0] - g.amplitude) < 1e-14

    def test_l_positive_zero_at_origin(self):
        """For l>0, r^l=0 at r=0."""
        for l in (1, 2, 3):
            g = PrimitiveGaussian(alpha=1.0, l=l)
            assert abs(g(np.array([0.0]))[0]) < 1e-14, f"l={l}: R(0) should be 0"

    def test_decay_at_large_r(self):
        """R(r) should be negligible at r >> 1/sqrt(alpha)."""
        g = PrimitiveGaussian(alpha=1.0, l=0)
        val = g(np.array([100.0]))[0]
        assert abs(val) < 1e-10

    def test_output_is_1d_array(self):
        g = PrimitiveGaussian(alpha=1.0, l=0)
        r = np.linspace(0.0, 3.0, 20)
        result = g(r)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_output_is_non_negative_for_l0(self):
        """R(r) ≥ 0 for l=0 (since A > 0 and r^0 * exp term > 0)."""
        g = PrimitiveGaussian(alpha=1.0, l=0)
        r = np.linspace(0.0, 5.0, 50)
        assert np.all(g(r) >= 0.0)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_larger_alpha_faster_decay(self, alpha):
        """Larger alpha decays faster: at large r the tighter Gaussian is smaller.

        Normalization changes the amplitude, so the comparison is only robust
        at large r (e.g. r=3) where the exp(-α r²) decay dominates.
        """
        g_small = PrimitiveGaussian(alpha=0.1, l=0)
        g_large = PrimitiveGaussian(alpha=alpha, l=0)
        r = np.array([3.0])  # far enough that exp decay dominates
        assert g_large(r)[0] < g_small(r)[0]


# ---------------------------------------------------------------------------
# PrimitiveGaussianRadials
# ---------------------------------------------------------------------------

class TestPrimitiveGaussianRadials:
    """Tests for the collection of Gaussian radial functions."""

    def test_creation_with_list_of_alphas(self):
        n_max, l_max = 2, 2
        n_alphas = n_max * (l_max + 1)
        rad = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), n_max, l_max,
                                       alphas=[1.0] * n_alphas)
        assert len(rad.radials) == n_alphas

    def test_creation_with_dict_of_alphas(self):
        n_max, l_max = 2, 1
        alphas_dict = {(n, l): 1.0
                       for n in range(1, n_max + 1)
                       for l in range(l_max + 1)}
        rad = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), n_max, l_max,
                                       alphas=alphas_dict)
        assert len(rad) == n_max * (l_max + 1)

    def test_raises_for_wrong_list_length(self):
        with pytest.raises(ValueError):
            PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 2, 1,
                                     alphas=[1.0])  # too short

    def test_raises_for_wrong_dict_keys(self):
        with pytest.raises(ValueError):
            PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 2, 1,
                                     alphas={(0, 0): 1.0})  # incomplete

    def test_len(self):
        for n_max, l_max in [(1, 0), (2, 1), (3, 2)]:
            rad = PrimitiveGaussianRadials(
                "H", (0.0, 0.0, 0.0), n_max, l_max,
                alphas=[1.0] * (n_max * (l_max + 1)),
            )
            assert len(rad) == n_max * (l_max + 1)

    def test_output_shape(self):
        n_max, l_max = 2, 2
        rad = PrimitiveGaussianRadials(
            "H", (0.0, 0.0, 0.0), n_max, l_max,
            alphas=[1.0] * (n_max * (l_max + 1)),
        )
        r = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.5, 0.5, 0.0]])
        vals = rad(r)
        assert vals.shape == (3, len(rad))

    def test_raises_for_wrong_input_shape(self):
        rad = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 1, 0,
                                       alphas=[1.0])
        with pytest.raises(ValueError):
            rad(np.zeros((4, 4)))

    def test_index_roundtrip(self):
        """running -> lexographic -> running must be the identity."""
        n_max, l_max = 3, 2
        rad = PrimitiveGaussianRadials(
            "H", (0.0, 0.0, 0.0), n_max, l_max,
            alphas=[1.0] * (n_max * (l_max + 1)),
        )
        for idx in range(len(rad)):
            nl = rad.running_to_lexographic_index(idx)
            assert rad.lexographic_to_running_index(nl) == idx

    def test_alpha_affects_radial_values(self):
        """Larger alpha → tighter Gaussian → smaller value at r=1 Å."""
        r = np.array([[1.0, 0.0, 0.0]])
        rad_small = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 1, 0,
                                              alphas=[0.1])
        rad_large = PrimitiveGaussianRadials("H", (0.0, 0.0, 0.0), 1, 0,
                                              alphas=[5.0])
        # r=1: exp(-0.1) > exp(-5), so rad_small > rad_large
        assert rad_small(r)[0, 0] > rad_large(r)[0, 0]

    def test_n_max_list_support(self):
        rad = PrimitiveGaussianRadials(
            "H",
            (0.0, 0.0, 0.0),
            [2, 1, 0],
            2,
            alphas=[0.5, 0.6, 1.0],
        )
        assert rad.n_max == [2, 1, 0]
        assert len(rad) == 3
        assert rad.running_to_lexographic_index(2) == (2, 0)

class TestCompoundGaussianRadials:
    def test_creation_with_list_of_alphas(self):
        n_max, l_max = 2, 2
        n_alphas = n_max * (l_max + 1)
        rad = CompoundGaussianRadials("H", (0.0, 0.0, 0.0), n_max, l_max,
                                      alphas=[(1.0, 0.5, 0.25)] * n_alphas, coeffs =[(0.5, 0.5, np.sqrt(2)/2)] * n_alphas)
        assert len(rad.radials) == n_alphas

    def test_creation_with_dict_of_alphas(self):
        n_max, l_max = 2, 1
        alphas_dict = {(n, l): [1.0, 0.5, 0.25]
                       for n in range(1, n_max + 1)
                       for l in range(l_max + 1)}
        coeffs_dict = {(n, l): [0.5, 0.3, 0.2]
                       for n in range(1, n_max + 1)
                       for l in range(l_max + 1)}
        rad = CompoundGaussianRadials("H", (0.0, 0.0, 0.0), n_max, l_max,
                                      alphas=alphas_dict, coeffs=coeffs_dict)
        assert len(rad) == n_max * (l_max + 1)
    def test_call(self):
        n_max, l_max = 2, 1
        alphas_dict = {(n, l): [1.0, 0.5, 0.25]
                       for n in range(1, n_max + 1)
                       for l in range(l_max + 1)}
        coeffs_dict = {(n, l): [0.5, 0.3, 0.2]
                       for n in range(1, n_max + 1)
                       for l in range(l_max + 1)}
        rad = CompoundGaussianRadials("H", (0.0, 0.0, 0.0), n_max, l_max,
                                      alphas=alphas_dict, coeffs=coeffs_dict)
        r = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.0, 0.0, 1.0]])
        vals = rad(r)
        assert vals.shape == (4, len(rad))

    def test_raises_for_wrong_list_length(self):
        with pytest.raises(ValueError):
            CompoundGaussianRadials("H", (0.0, 0.0, 0.0), 2, 1,
                                    alphas=[1.0],
                                    coeffs=[0.5])  # too short

    def test_raises_for_wrong_dict_keys(self):
        with pytest.raises(ValueError):
            CompoundGaussianRadials("H", (0.0, 0.0, 0.0), 2, 1,
                                    alphas={(0, 0): [1.0]},
                                    coeffs={(0, 0): [0.5]})  # incomplete


# ---------------------------------------------------------------------------
# RealSphericalHarmonics
# ---------------------------------------------------------------------------

class TestRealSphericalHarmonics:
    """Tests for the real spherical-harmonic angular component."""

    def test_len(self):
        for l_max in (0, 1, 2, 3):
            sph = RealSphericalHarmonics("H", (0.0, 0.0, 0.0), l_max=l_max)
            assert len(sph) == (l_max + 1) ** 2

    def test_output_shape(self, unit_sphere_pts):
        l_max = 2
        sph = RealSphericalHarmonics("H", (0.0, 0.0, 0.0), l_max=l_max)
        vals = sph(unit_sphere_pts)
        assert vals.shape == (len(unit_sphere_pts), (l_max + 1) ** 2)

    def test_raises_for_wrong_input_shape(self):
        sph = RealSphericalHarmonics("H", (0.0, 0.0, 0.0), l_max=1)
        with pytest.raises(ValueError):
            sph(np.zeros((5, 2)))

    def test_l0_is_constant(self, unit_sphere_pts):
        """Y_0^0 = 1/sqrt(4π) is direction-independent."""
        sph = RealSphericalHarmonics("H", (0.0, 0.0, 0.0), l_max=2)
        vals = sph(unit_sphere_pts)
        y00 = vals[:, 0]
        expected = 1.0 / np.sqrt(4.0 * np.pi)
        np.testing.assert_allclose(y00, expected, atol=1e-12)

    def test_index_roundtrip(self):
        """running -> lexographic -> running must be the identity."""
        l_max = 3

        class _Ang(AngularFunctions):
            def __call__(self, r):
                pass

        ang = _Ang("H", (0.0, 0.0, 0.0), l_max=l_max)
        for idx in range(len(ang)):
            lm = ang.running_to_lexographic_index(idx)
            assert ang.lexographic_to_running_index(lm) == idx


# ---------------------------------------------------------------------------
# RIBasis
# ---------------------------------------------------------------------------

class TestRIBasis:
    """Tests for the combined radial × angular RI basis."""

    @pytest.mark.parametrize("n_max,l_max", [(1, 0), (1, 1), (2, 1), (2, 2)])
    def test_len(self, n_max, l_max):
        basis = _make_basis(n_max, l_max)
        assert len(basis) == n_max * (l_max + 1) ** 2

    def test_output_shape(self, basis_2_2):
        r = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        vals = basis_2_2(r)
        assert vals.shape == (3, len(basis_2_2))

    def test_output_is_finite_at_off_origin_points(self, basis_2_2):
        r = np.array([[0.5, 0.3, 0.2],
                      [1.0, 1.0, 1.0]])
        vals = basis_2_2(r)
        assert np.all(np.isfinite(vals))

    def test_raises_for_wrong_input_shape(self, basis_2_2):
        with pytest.raises(ValueError):
            basis_2_2(np.zeros((3, 4)))

    def test_compute_equals_call_shifted_by_origin(self):
        """basis.compute(r_abs) == basis(r_abs - origin)."""
        origin = (1.0, -0.5, 2.0)
        basis = _make_basis(n_max=2, l_max=1, origin=origin)
        r_abs = np.array([[1.5, 0.0, 2.3],
                          [2.0, -0.5, 3.0]])
        r_rel = r_abs - np.array(origin)
        np.testing.assert_allclose(basis.compute(r_abs), basis(r_rel), atol=1e-14)

    def test_compute_raises_for_wrong_shape(self):
        basis = _make_basis(n_max=1, l_max=0)
        with pytest.raises(ValueError):
            basis.compute(np.zeros((3,)))

    @pytest.mark.parametrize("n_max,l_max", [
        (1, 0), (1, 1), (2, 1), (2, 2), (3, 2),
    ])
    def test_index_roundtrip(self, n_max, l_max):
        """running_to_lexographic and lexographic_to_running are inverses."""
        basis = _make_basis(n_max, l_max)
        for idx in range(len(basis)):
            nlm = basis.running_to_lexographic_index(idx)
            assert basis.lexographic_to_running_index(nlm) == idx, (
                f"n_max={n_max}, l_max={l_max}: roundtrip failed at idx={idx} "
                f"via {nlm}"
            )

    def test_known_index_values(self):
        """Spot-check specific (n,l,m) <-> index mappings for n_max=2, l_max=1."""
        basis = _make_basis(n_max=2, l_max=1)
        # Expected lexographic order (n, l, m): n=1 block then n=2 block
        # l=0: (1,0,0); l=1: (1,1,-1),(1,1,0),(1,1,1)
        expected = [
            (1, 0, 0), (1, 1, -1), (1, 1, 0), (1, 1, 1),
            (2, 0, 0), (2, 1, -1), (2, 1, 0), (2, 1, 1),
        ]
        for idx, nlm in enumerate(expected):
            assert basis.running_to_lexographic_index(idx) == nlm, (
                f"idx={idx}: expected {nlm}, "
                f"got {basis.running_to_lexographic_index(idx)}"
            )
            assert basis.lexographic_to_running_index(nlm) == idx, (
                f"nlm={nlm}: expected {idx}, "
                f"got {basis.lexographic_to_running_index(nlm)}"
            )

    def test_species_and_origin_stored(self):
        origin = (1.0, 2.0, 3.0)
        basis = _make_basis(n_max=1, l_max=0, origin=origin)
        assert basis.species == "H"
        np.testing.assert_array_equal(basis.origin, np.array(origin))

    def test_radial_and_angular_components_accessible(self, basis_2_2):
        assert hasattr(basis_2_2, "radial_funcs")
        assert hasattr(basis_2_2, "angular_funcs")
        assert isinstance(basis_2_2.radial_funcs, RadialFunctions)
        assert isinstance(basis_2_2.angular_funcs, AngularFunctions)

    def test_single_point_is_row_vector(self, basis_2_2):
        """Evaluating at a single point must return shape (1, n_basis)."""
        r = np.array([[1.0, 0.5, 0.3]])
        vals = basis_2_2(r)
        assert vals.shape == (1, len(basis_2_2))

    def test_different_n_max_changes_output_columns(self):
        r = np.array([[1.0, 0.0, 0.0]])
        b1 = _make_basis(n_max=1, l_max=1)
        b2 = _make_basis(n_max=2, l_max=1)
        assert b2(r).shape[1] == 2 * b1(r).shape[1]

    def test_basis_values_scale_with_alpha(self):
        """Larger alpha produces a tighter function, so smaller absolute value at r=1."""
        r = np.array([[1.0, 0.0, 0.0]])
        b_small = _make_basis(n_max=1, l_max=0, alpha=0.1)
        b_large = _make_basis(n_max=1, l_max=0, alpha=5.0)
        # The (n=0, l=0, m=0) column is radial × Y_0^0
        v_small = abs(b_small(r)[0, 0])
        v_large = abs(b_large(r)[0, 0])
        # exp(-0.1) > exp(-5) after normalisation
        assert v_small > v_large

    def test_n_max_list_support(self):
        basis = _make_basis(n_max=[2, 1, 0], l_max=2)
        assert basis.n_max == [2, 1, 0]
        assert len(basis) == 5
        expected = [
            (1, 0, 0),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 0, 0),
        ]
        for idx, nlm in enumerate(expected):
            assert basis.running_to_lexographic_index(idx) == nlm


# ---------------------------------------------------------------------------
# load_basis factory
# ---------------------------------------------------------------------------

class TestLoader:
    """Tests for the registry-based load_basis factory."""

    def test_returns_ribasis_instance(self):
        basis = _make_basis(n_max=1, l_max=0)
        assert isinstance(basis, RIBasis)

    def test_unknown_radial_method_raises(self):
        with pytest.raises(ValueError, match="radial"):
            load_basis("H", (0.0, 0.0, 0.0), 1, 0,
                       cutoff = CutoffType.NON_PERIODIC,
                       radial_method="nonexistent",
                       angular_method="spherical",
                       radial_params={"alphas": [1.0]})

    def test_unknown_angular_method_raises(self):
        with pytest.raises(ValueError, match="angular"):
            load_basis("H", (0.0, 0.0, 0.0), 1, 0,
                       cutoff = CutoffType.NON_PERIODIC,
                       radial_method="gaussian",
                       angular_method="nonexistent",
                       radial_params={"alphas": [1.0]})

    def test_spherical_alias(self):
        """'spherical' and 'real_spherical' should both work."""
        alphas = [1.0] * (2 * (2 + 1))  # n_max * (l_max + 1) = 2 * 3 = 6
        b1 = load_basis("H", (0.0, 0.0, 0.0), 2, 2,
                         cutoff = CutoffType.NON_PERIODIC,
                         angular_method="spherical",
                         radial_method="gaussian",
                         radial_params={"alphas": alphas})
        b2 = load_basis("H", (0.0, 0.0, 0.0), 2, 2,
                         cutoff = CutoffType.NON_PERIODIC,
                         angular_method="real_spherical",
                         radial_method="gaussian",
                         radial_params={"alphas": alphas})
        r = np.array([[1.0, 0.5, 0.3]])
        np.testing.assert_allclose(b1(r), b2(r), atol=1e-14)

    def test_different_species_label(self):
        """Species label should be stored correctly."""
        for species in ("H", "O", "C", "N"):
            alphas = [1.0]
            b = load_basis(species, (0.0, 0.0, 0.0), 1, 0,
                           cutoff = CutoffType.NON_PERIODIC,
                           radial_method="gaussian",
                           angular_method="spherical",
                           radial_params={"alphas": alphas})
            assert b.species == species

    def test_none_radial_params_uses_defaults(self):
        """radial_params=None should raise because alphas are required."""
        with pytest.raises((ValueError, TypeError)):
            load_basis("H", (0.0, 0.0, 0.0), 1, 0,\
                       cutoff = CutoffType.NON_PERIODIC,
                       radial_method="gaussian",
                       angular_method="spherical",
                       radial_params=None)

    @pytest.mark.parametrize("n_max,l_max", [(1, 0), (2, 1), (2, 2)])
    def test_basis_size_matches_n_max_l_max(self, n_max, l_max):
        b = _make_basis(n_max, l_max)
        assert len(b) == n_max * (l_max + 1) ** 2
