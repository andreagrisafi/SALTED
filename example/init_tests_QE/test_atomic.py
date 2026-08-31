from __future__ import annotations

from typing import Literal

import numpy as np
import torch  #Pytorch for algorthmic differentiation
import scipy.optimize


from salted.qe import load_basis
from salted.qe.core  import AngularFunctions, RadialFunctions, RIBasis
from salted.qe import Atomic_Rad
from salted.qe.qe_types import Cutoff, CutoffType
from salted.qe.realspherharmonic import RealSphericalHarmonics
from salted.qe.combine_gaussian import CompoundGaussianRadials
from salted.qe.interpolate_gaussian import interpolate_atomic_basis, fit_gaussians



def _make_basis(n_max: int | list[int], l_max: int, origin=(0.0, 0.0, 0.0)) -> RIBasis:
    """Convenience factory that creates a Gaussian/spherical RI basis."""
    if isinstance(n_max, int):
        n_alphas = n_max * (l_max + 1)
    else:
        n_alphas = sum(n_max)
    return load_basis(
        species="Na",
        origin=origin,
        n_max=n_max,
        l_max=l_max,
        cutoff = CutoffType.NON_PERIODIC,
        radial_method="atomic",
        angular_method="spherical",
        radial_params={"filename": "/home/goutie_a/Software/SALTED/example/init_tests_QE/data/atomic_densities/Na_TZP_rc10.0_qe.dat"}
    )



class TestAtomicRadials:
    
    def test_radial_evaluation(self):
        """Test that the Atomic_Rad class correctly evaluates the radial functions."""
        origin = (0.0, 0.0, 0.0)
        n_max = 2
        l_max = 2
        basis = _make_basis(n_max, l_max, origin)

        # Test evaluation at multiple points
        r_multi = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])  # (N=2, 3)
        expected_shape_multi = (r_multi.shape[0], len(basis.radial_funcs))  # (N, n_radials)
        radial_values_multi = basis.radial_funcs(r_multi)  # (N, n_radials)
        assert radial_values_multi.shape == expected_shape_multi, f"Expected shape {expected_shape_multi}, got {radial_values_multi.shape}"

    def test_gaussian_fit(self):
        """Fit a radial function with a Gaussian mixture and verify the approximation is sane."""
        origin = (0.0, 0.0, 0.0)
        n_max = 2
        l_max = 2
        basis = _make_basis(n_max, l_max, origin)

        basis_int = interpolate_atomic_basis(basis, species="Na", position=origin, num_max=10)

        r = np.linspace(0, 10, 512)
        radial_values = basis.radial_funcs.radials[3](r)
        fitted_values = basis_int.radials[3](r)

        assert np.all(np.isfinite(fitted_values)), "Fitted values contain NaN or Inf"
        assert np.isfinite(basis_int.radials[0].amplitude), "Normalization amplitude is invalid"
        assert np.max(fitted_values) > 0.0, "Fitted Gaussian mixture should stay positive in the physical range"
        print(radial_values[:20])
        print(fitted_values[:20])
        assert np.allclose(fitted_values[:20], radial_values[:20], rtol=0.01, atol=0.01), (
            "Fitted Gaussian approximation is not close enough to the target radial shape"
        )

    def test_gaussian_fit_is_deterministic(self):
        """Repeated fits of the same function should give the same Gaussian basis."""
        func = lambda r: np.exp(-0.5 * r**2)

        r = np.linspace(0, 10, 512)
        rho = func(r)

        alphas_1, coeffs_1 = fit_gaussians(r, rho, l=0, n_gauss=3)
        alphas_2, coeffs_2 = fit_gaussians(r, rho, l=0, n_gauss=3)

        assert np.allclose(alphas_1, alphas_2)
        assert np.allclose(coeffs_1, coeffs_2)
