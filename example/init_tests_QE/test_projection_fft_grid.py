import numpy as np
import pytest
from typing import cast

from salted_espresso.electronic_density.cube2rho import PlaneWaveDensity, compute_rho_g
from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.projections.core import compute_projection_vector, compute_projection_vector_FFT
from salted_espresso.ri_basis.core import RIBasisSet


class FakeBasisSet:
    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.column_stack(
            (
                np.ones_like(x),
                x + 2.0 * y - z,
                x * x + y * z,
            )
        )


def make_density() -> PlaneWaveDensity:
    rho_r = np.random.default_rng(4).random((4, 3, 2))
    spacing = np.diag([0.3, 0.2, 0.4])
    rhog = compute_rho_g(rho_r, spacing)
    return PlaneWaveDensity(
        rho_g=rhog.rho_g,
        G=rhog.G,
        cell_grid=rhog.cell_grid,
        grid_shape=rhog.grid_shape,
        fft_data=rho_r,
        origin=np.zeros(3),
    )


def test_compute_projection_vector_fft_matches_standard_quadrature() -> None:
    rho = make_density()
    basis = cast(RIBasisSet, cast(object, FakeBasisSet()))

    expected = compute_projection_vector(rho, basis)
    actual = compute_projection_vector_FFT(rho, basis)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_compute_projection_vector_fft_requires_integrate_against() -> None:
    basis = cast(RIBasisSet, cast(object, FakeBasisSet()))

    def plain_density(points: np.ndarray) -> np.ndarray:
        return np.ones(len(points), dtype=float)

    plain_density_callable = cast(DensityFunction, cast(object, plain_density))
    with pytest.raises(ValueError, match="integrate_against"):
        compute_projection_vector_FFT(plain_density_callable, basis)



