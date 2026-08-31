import numpy as np
from typing import cast

from salted_espresso.projections.core import compute_overlap_matrix
from salted_espresso.ri_basis.core import RIBasisSet


class FakeBasisSet:
    def __init__(self) -> None:
        self.cell_vectors = np.eye(3)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        x = pts[:, 0]
        return np.column_stack(
            (
                np.ones_like(x),
                np.exp(2j * np.pi * x),
                np.exp(4j * np.pi * x),
            )
        )


def test_compute_overlap_matrix_matches_orthonormal_fourier_basis() -> None:
    basis = cast(RIBasisSet, cast(object, FakeBasisSet()))

    matrix = compute_overlap_matrix(basis)

    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)

