"""Tests for projections.overlap metric assembly."""

import numpy as np
import pytest

from salted_espresso.projections.overlap import MetricMethod, compute_overlap
from salted_espresso.ri_basis import load_basis


def _make_basis(n_max: int = 2, l_max: int = 1):
    n_alphas = n_max * (l_max + 1)
    return load_basis(
        species="H",
        origin=(0.0, 0.0, 0.0),
        n_max=n_max,
        l_max=l_max,
        radial_method="gaussian",
        angular_method="spherical",
        radial_params={"alphas": [1.0] * n_alphas},
    )


def test_overlap_metric_shape_and_symmetry() -> None:
    basis = _make_basis(n_max=2, l_max=1)
    matrix = compute_overlap(basis, method=MetricMethod.OVERLAP, n_cartesian_grid=256)

    assert matrix.shape == (len(basis), len(basis))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_overlap_alias_s_matches_overlap() -> None:
    basis = _make_basis(n_max=2, l_max=1)
    m1 = compute_overlap(basis, method="overlap", n_cartesian_grid=256)
    m2 = compute_overlap(basis, method="s", n_cartesian_grid=256)
    np.testing.assert_allclose(m1, m2, atol=1e-12)


def test_coulomb_alias_j_matches_coulomb() -> None:
    basis = _make_basis(n_max=2, l_max=1)
    m1 = compute_overlap(basis, method="coulomb", n_cartesian_grid=16)
    m2 = compute_overlap(basis, method="j", n_cartesian_grid=16)
    np.testing.assert_allclose(m1, m2, atol=1e-12)


def test_different_l_or_m_blocks_do_not_couple() -> None:
    basis = _make_basis(n_max=2, l_max=1)
    matrix = compute_overlap(basis, method="overlap", n_cartesian_grid=128)

    idx_l0_m0 = basis.lexographic_to_running_index((1, 0, 0))
    idx_l1_m0 = basis.lexographic_to_running_index((1, 1, 0))
    idx_l1_m1 = basis.lexographic_to_running_index((1, 1, 1))

    assert abs(matrix[idx_l0_m0, idx_l1_m0]) < 1e-14
    assert abs(matrix[idx_l1_m0, idx_l1_m1]) < 1e-14


def test_single_basis_overlap_is_unity_with_reasonable_quadrature() -> None:
    basis = _make_basis(n_max=1, l_max=0)
    matrix = compute_overlap(basis, method="overlap", n_cartesian_grid=512)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(1.0, abs=5e-3)


def test_invalid_method_raises() -> None:
    basis = _make_basis(n_max=1, l_max=0)
    with pytest.raises(ValueError, match="Unknown metric method"):
        compute_overlap(basis, method="invalid")
