"""Tests for src/electronic_density/cube2rho.py."""

from pathlib import Path

import numpy as np
import pytest

from salted_espresso.electronic_density import cube2rho

DATA_DIR = Path(__file__).parent / "data" / "cube_files"
CUBE_PATH = DATA_DIR / "nvp_rho.cube"


# ---------------------------------------------------------------------------
# Module-scoped fixtures so the expensive load + FFT runs only once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cube():
    return cube2rho.load_cubefile(CUBE_PATH)


@pytest.fixture(scope="module")
def rho_obj() -> cube2rho.PlaneWaveDensity:
    return cube2rho.load_rho_from_cube(CUBE_PATH)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def grid_point_position(i, j, k, origin, spacing):
    """Return the Cartesian coordinate of grid index (i, j, k)."""
    step_matrix = np.asarray(spacing, dtype=float)
    if step_matrix.shape == (3,):
        step_matrix = np.diag(step_matrix)
    return origin + i * step_matrix[0] + j * step_matrix[1] + k * step_matrix[2]


# ---------------------------------------------------------------------------
# load_cubefile
# ---------------------------------------------------------------------------


class TestLoadCubefile:
    def test_required_keys_present(self, cube):
        for key in ("atoms", "data", "origin", "spacing"):
            assert key in cube

    def test_data_is_3d_array(self, cube):
        assert isinstance(cube["data"], np.ndarray)
        assert cube["data"].ndim == 3

    def test_origin_shape(self, cube):
        assert np.asarray(cube["origin"]).shape == (3,)

    def test_spacing_shape(self, cube):
        spacing = np.asarray(cube["spacing"])
        assert spacing.shape in [(3,), (3, 3)]

    def test_data_is_finite(self, cube):
        assert np.all(np.isfinite(cube["data"]))


# ---------------------------------------------------------------------------
# compute_rho_g
# ---------------------------------------------------------------------------


class TestComputeRhoG:
    def test_output_shapes_match_grid(self, cube):
        nx, ny, nz = cube["data"].shape
        n_g = nx * ny * nz
        rhog = cube2rho.compute_rho_g(cube["data"], cube["spacing"])
        assert rhog.rho_g.shape == (n_g,)
        assert rhog.G.shape == (n_g, 3)

    def test_grid_shape_attribute(self, cube):
        rhog = cube2rho.compute_rho_g(cube["data"], cube["spacing"])
        assert rhog.grid_shape == cube["data"].shape

    def test_rho_g_dtype_is_complex(self, cube):
        rhog = cube2rho.compute_rho_g(cube["data"], cube["spacing"])
        assert np.issubdtype(rhog.rho_g.dtype, np.complexfloating)

    def test_raises_for_non_3d_input(self):
        with pytest.raises(ValueError):
            cube2rho.compute_rho_g(np.zeros((4, 4)), np.array([1.0, 1.0, 1.0]))

    def test_raises_for_invalid_spacing_shape(self, cube):
        with pytest.raises(ValueError):
            cube2rho.compute_rho_g(cube["data"], np.array([1.0, 1.0]))

    def test_orthorhombic_spacing_accepted(self):
        rho_r = np.random.default_rng(0).random((4, 4, 4))
        rhog = cube2rho.compute_rho_g(rho_r, np.array([0.5, 0.5, 0.5]))
        assert rhog.rho_g.shape == (64,)

    def test_matrix_spacing_accepted(self):
        rho_r = np.random.default_rng(0).random((4, 4, 4))
        rhog = cube2rho.compute_rho_g(rho_r, np.eye(3) * 0.5)
        assert rhog.rho_g.shape == (64,)


# ---------------------------------------------------------------------------
# FFT roundtrip
# ---------------------------------------------------------------------------


class TestFftRoundtrip:
    def test_ifftn_fftn_recovers_original(self, cube):
        """Pure NumPy FFT roundtrip: ifftn(fftn(rho)) == rho."""
        rho_grid = cube["data"]
        rho_back = np.fft.ifftn(np.fft.fftn(rho_grid))
        np.testing.assert_allclose(
            rho_back, rho_grid, atol=1e-10,
            err_msg="ifftn(fftn(rho)) should recover rho within numerical precision"
        )


# ---------------------------------------------------------------------------
# rho reconstruction
# ---------------------------------------------------------------------------


class TestRhoReconstruction:
    def test_fft_data_is_stored_on_loaded_density(self, cube, rho_obj):
        np.testing.assert_allclose(rho_obj.fft_data, cube["data"])

    def test_invalid_r_shape_raises(self, rho_obj):
        with pytest.raises(ValueError):
            rho_obj(np.zeros((2, 2)))

    def test_single_point_returns_scalar(self, cube, rho_obj):
        origin = np.asarray(cube["origin"], dtype=float)
        val = rho_obj(origin)
        assert np.ndim(val) == 0, (
            f"Single-point evaluation should return a 0-d array, got shape {np.asarray(val).shape}"
        )

    def test_batch_points_return_correct_shape(self, cube, rho_obj):
        origin = np.asarray(cube["origin"], dtype=float)
        spacing = np.asarray(cube["spacing"], dtype=float)
        step_matrix = spacing if spacing.ndim == 2 else np.diag(spacing)
        r = np.array([
            origin,
            origin + step_matrix[0],
            origin + step_matrix[1],
        ])
        vals = rho_obj(r)
        assert vals.shape == (3,)

    def test_selected_grid_points_match_reference(self, cube, rho_obj):
        """Reconstruction at exact grid points must match the original density."""
        rho_grid = cube["data"]
        origin = np.asarray(cube["origin"], dtype=float)
        spacing = np.asarray(cube["spacing"], dtype=float)
        nx, ny, nz = rho_grid.shape

        points = [
            (0, 0, 0),
            (nx // 2, ny // 2, nz // 2),
            (nx - 1, ny - 1, nz - 1),
            (nx // 3, ny // 4, nz // 5),
        ]
        for i, j, k in points:
            r = grid_point_position(i, j, k, origin, spacing)
            expected = rho_grid[i, j, k]
            actual = rho_obj(r)
            assert abs(actual - expected) < 1e-10, (
                f"Grid point ({i},{j},{k}): expected {expected}, got {actual}"
            )

    def test_random_grid_points_match_reference(self, cube, rho_obj):
        """Batch reconstruction at random grid points stays within numerical noise."""
        rho_grid = cube["data"]
        origin = np.asarray(cube["origin"], dtype=float)
        spacing = np.asarray(cube["spacing"], dtype=float)
        nx, ny, nz = rho_grid.shape

        rng = np.random.default_rng(42)
        n_samples = 10
        indices = [
            (int(rng.integers(0, nx)), int(rng.integers(0, ny)), int(rng.integers(0, nz)))
            for _ in range(n_samples)
        ]
        r_batch = np.array([
            grid_point_position(i, j, k, origin, spacing)
            for i, j, k in indices
        ])
        refs = np.array([rho_grid[i, j, k] for i, j, k in indices])

        vals = rho_obj(r_batch)

        np.testing.assert_allclose(
            vals, refs, atol=1e-10,
            err_msg="Batch reconstruction at random grid points should match reference density"
        )

    def test_result_is_real(self, cube, rho_obj):
        """rho(r) must return a real result at grid points."""
        origin = np.asarray(cube["origin"], dtype=float)
        val = rho_obj(origin)
        assert np.isrealobj(val), f"Expected real result, got dtype {np.asarray(val).dtype}"

    def test_imaginary_part_below_absolute_tolerance(self, cube, rho_obj):
        """
        Imaginary part of the raw reconstruction at grid points must be
        below the absolute tolerance of 1e-6 used in rho.__call__.

        Per NumPy documentation for np.real_if_close: "If the tolerance is
        <=1, then the absolute tolerance is used."  With tol=1e-6 (<=1) the
        check is simply ``|imag| < 1e-6``.
        """
        rho_grid = cube["data"]
        origin = np.asarray(cube["origin"], dtype=float)
        spacing = np.asarray(cube["spacing"], dtype=float)
        nx, ny, nz = rho_grid.shape

        N = len(rho_obj.rho_g)
        points = [
            (0, 0, 0),
            (nx // 2, ny // 2, nz // 2),
            (nx - 1, ny - 1, nz - 1),
        ]
        for i, j, k in points:
            r = grid_point_position(i, j, k, origin, spacing)
            raw = np.sum(rho_obj.rho_g * np.exp(1j * (rho_obj.G @ r))) / N
            imag_abs = abs(np.imag(raw))
            assert imag_abs < 1e-6, (
                f"Grid point ({i},{j},{k}): |Im(rho)| = {imag_abs:.2e} "
                f"exceeds absolute tolerance 1e-6"
            )

    def test_chunked_batch_matches_vectorized_reference(self, rho_obj):
        rng = np.random.default_rng(7)
        r_batch = rng.normal(size=(64, 3))

        previous = rho_obj.max_batch_memory_mb
        try:
            # Force chunked execution path to run many chunks.
            rho_obj.max_batch_memory_mb = 0.001
            actual = rho_obj(r_batch)
        finally:
            rho_obj.max_batch_memory_mb = previous

        N = len(rho_obj.rho_g)
        expected = np.exp(1j * (r_batch @ rho_obj.G.T)) @ rho_obj.rho_g / N
        expected = np.real_if_close(expected, tol=1e-6)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_chunk_size_is_at_least_one(self, rho_obj):
        previous = rho_obj.max_batch_memory_mb
        try:
            rho_obj.max_batch_memory_mb = 0.0
            assert rho_obj._batch_chunk_size() >= 1
        finally:
            rho_obj.max_batch_memory_mb = previous

    def test_integrate_against_matches_manual_quadrature(self, cube, rho_obj):
        rho_grid = cube["data"]
        origin = np.asarray(cube["origin"], dtype=float)
        step_matrix = np.asarray(rho_obj.cell_grid, dtype=float)
        nx, ny, nz = rho_grid.shape

        f1 = np.linspace(0, 1, nx, endpoint=False)
        f2 = np.linspace(0, 1, ny, endpoint=False)
        f3 = np.linspace(0, 1, nz, endpoint=False)
        F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing="ij")
        frac_pts = np.stack((F1.ravel(), F2.ravel(), F3.ravel()), axis=-1)
        points = origin + frac_pts @ step_matrix
        dV = abs(np.linalg.det(step_matrix)) / (nx * ny * nz)

        def vectorized_func(pts):
            return np.sum(pts**2, axis=-1) + 0.25

        rho_vals = np.asarray(rho_grid, dtype=float).ravel()
        func_vals = np.asarray(vectorized_func(points), dtype=float).ravel()
        expected = float(np.dot(rho_vals, func_vals) * dV)
        actual = rho_obj.integrate_against(vectorized_func)
        assert actual == pytest.approx(expected, abs=1e-10)

        def scalar_only_func(point):
            point = np.asarray(point, dtype=float)
            if point.shape != (3,):
                raise TypeError("scalar-only callable expects a single point")
            return float(np.dot(point, point) + 0.25)

        scalar_vals = np.array([scalar_only_func(point) for point in points], dtype=float)
        scalar_expected = float(np.dot(rho_vals, scalar_vals) * dV)
        scalar_actual = rho_obj.integrate_against(scalar_only_func)
        assert scalar_actual == pytest.approx(scalar_expected, abs=1e-10)

