"""Lightweight tests for streamed rho(r) evaluation behavior."""

from collections.abc import Iterator

import numpy as np
import pytest

from salted_espresso.electronic_density.cube2rho import PlaneWaveDensity


def make_small_density() -> PlaneWaveDensity:
    rng = np.random.default_rng(11)
    n_pairs = 16
    g_half = rng.normal(size=(n_pairs, 3))
    coeff_half = rng.normal(size=n_pairs) + 1j * rng.normal(size=n_pairs)

    # Conjugate symmetry rho(-G)=conj(rho(G)) guarantees real rho(r).
    rho_g = np.concatenate([coeff_half, np.conjugate(coeff_half)])
    G = np.vstack([g_half, -g_half])
    return PlaneWaveDensity(rho_g=rho_g, G=G, max_batch_memory_mb=0.0005)


def dense_reference(rho_obj: PlaneWaveDensity, points: np.ndarray) -> np.ndarray:
    n_terms = len(rho_obj.rho_g)
    values = np.exp(1j * (points @ rho_obj.G.T)) @ rho_obj.rho_g / n_terms
    return np.real_if_close(values, tol=1e-6)


def test_streamed_ndarray_matches_dense_reference() -> None:
    rho_obj = make_small_density()
    points = np.random.default_rng(7).normal(size=(9, 3))

    actual = rho_obj(points)
    expected = dense_reference(rho_obj, points)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_generator_input_returns_iterator_and_matches_reference() -> None:
    rho_obj = make_small_density()
    points = np.random.default_rng(8).normal(size=(6, 3))

    streamed = rho_obj((point for point in points))
    assert isinstance(streamed, Iterator)

    actual = np.fromiter(streamed, dtype=float, count=len(points))
    expected = dense_reference(rho_obj, points)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_batch_chunk_size_is_at_least_one_for_tiny_memory_budget() -> None:
    rho_obj = make_small_density()
    rho_obj.max_batch_memory_mb = 0.0
    assert rho_obj._batch_chunk_size() >= 1


def make_asymmetric_density(policy: str) -> PlaneWaveDensity:
    # No conjugate pairing in rho_g/G, so rho(r) is generally complex.
    rho_g = np.array([1.0 + 0.2j, -0.3 + 0.1j], dtype=np.complex128)
    G = np.array([[0.3, 0.0, 0.0], [0.0, 0.7, 0.0]], dtype=float)
    return PlaneWaveDensity(
        rho_g=rho_g,
        G=G,
        complex_result_policy=policy,
        imag_abs_tol=1e-12,
        imag_rel_tol=0.0,
    )


def test_complex_residual_raises_in_strict_mode() -> None:
    rho_obj = make_asymmetric_density(policy="raise")
    with pytest.raises(ValueError, match="complex beyond tolerance"):
        rho_obj(np.array([0.4, -0.2, 0.0]))


def test_complex_residual_warns_and_coerces_to_real() -> None:
    rho_obj = make_asymmetric_density(policy="warn")
    with pytest.warns(RuntimeWarning, match="imaginary residual"):
        value = rho_obj(np.array([0.4, -0.2, 0.0]))
    assert np.isrealobj(value)


def test_small_imaginary_noise_is_accepted() -> None:
    rho_obj = make_small_density()
    rho_obj.complex_result_policy = "raise"
    rho_obj.imag_abs_tol = 1e-6

    value = rho_obj._to_real_scalar(np.complex128(2.0 + 5e-7j))
    assert value == pytest.approx(2.0)
